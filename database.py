import json
import os
import sqlite3
import time
from contextlib import contextmanager
from itertools import combinations

import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "musicden.db")

TRACK_COLUMNS = (
    "id",
    "file_path",
    "filename",
    "title",
    "artist",
    "album",
    "genre",
    "year",
    "duration_sec",
    "bpm",
    "camelot_key",
    "energy",
    "file_date",
    "source_folder",
    "analyzed_at",
)
TRACK_SELECT = ", ".join(TRACK_COLUMNS)
TRACK_SELECT_WITH_ALIAS = ", ".join(f"t.{column}" for column in TRACK_COLUMNS)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _serialize_embedding(embedding):
    if embedding is None:
        return None
    return json.dumps([float(value) for value in embedding])


def _deserialize_embedding_array(embedding_text):
    if not embedding_text:
        return None
    try:
        return np.asarray(json.loads(embedding_text), dtype=np.float32)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _deserialize_embedding_list(embedding_text):
    embedding = _deserialize_embedding_array(embedding_text)
    if embedding is None or embedding.size == 0:
        return None
    return embedding.tolist()


def _track_file_size(file_path: str) -> int:
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0


def _track_format(file_path: str) -> str:
    ext = os.path.splitext(file_path or "")[1].lower()
    return ext.lstrip(".").upper() or "UNKNOWN"


def _attach_runtime_track_fields(track: dict) -> dict:
    item = dict(track)
    file_path = item.get("file_path", "")
    item["file_size"] = _track_file_size(file_path)
    item["format"] = _track_format(file_path)
    return item


def _attach_runtime_track_fields_many(rows) -> list[dict]:
    return [_attach_runtime_track_fields(dict(row)) for row in rows]


def _track_quality_rank(track: dict) -> tuple[int, int]:
    ext = os.path.splitext(track.get("file_path", ""))[1].lower()
    quality = 2 if ext == ".flac" else 1 if ext == ".wav" else 0
    return quality, int(track.get("file_size") or 0)


def _decorate_duplicate_track(track: dict) -> dict:
    return _attach_runtime_track_fields(track)


def _mark_duplicate_group(tracks: list[dict], match_type: str) -> dict:
    decorated = [_decorate_duplicate_track(track) for track in tracks]
    keep_track = max(
        decorated,
        key=lambda track: (_track_quality_rank(track), track.get("analyzed_at") or 0, -(track.get("id") or 0)),
    )

    ordered_tracks = sorted(
        decorated,
        key=lambda track: (
            0 if track["id"] == keep_track["id"] else 1,
            -_track_quality_rank(track)[0],
            -_track_quality_rank(track)[1],
            track.get("title") or "",
            track.get("id") or 0,
        ),
    )
    for track in ordered_tracks:
        track["recommendation"] = "keep" if track["id"] == keep_track["id"] else "remove"
    return {"match_type": match_type, "tracks": ordered_tracks}


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                filename TEXT,
                title TEXT,
                artist TEXT,
                album TEXT,
                genre TEXT,
                year TEXT,
                duration_sec REAL,
                bpm REAL,
                camelot_key TEXT,
                energy REAL,
                file_date REAL,
                source_folder TEXT,
                analyzed_at REAL,
                clap_embedding TEXT,
                waveform_data TEXT
            );

            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at REAL NOT NULL,
                filters_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS playlist_tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                playlist_id INTEGER NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
                track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
                position INTEGER NOT NULL DEFAULT 0,
                UNIQUE(playlist_id, track_id)
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS track_embeddings (
                track_id INTEGER PRIMARY KEY,
                chroma_vector TEXT,
                FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS play_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
                played_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tracks_bpm ON tracks(bpm);
            CREATE INDEX IF NOT EXISTS idx_tracks_camelot ON tracks(camelot_key);
            CREATE INDEX IF NOT EXISTS idx_tracks_file_date ON tracks(file_date);
            CREATE INDEX IF NOT EXISTS idx_playlist_tracks_playlist ON playlist_tracks(playlist_id, position);
            CREATE INDEX IF NOT EXISTS idx_play_history_played_at ON play_history(played_at DESC);
        """)
        if not _column_exists(conn, "tracks", "clap_embedding"):
            conn.execute("ALTER TABLE tracks ADD COLUMN clap_embedding TEXT")
        if not _column_exists(conn, "tracks", "waveform_data"):
            conn.execute("ALTER TABLE tracks ADD COLUMN waveform_data TEXT")


def get_setting(key: str, default=None):
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default


def set_setting(key: str, value: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def upsert_track(data: dict) -> int:
    serialized = dict(data)
    if "clap_embedding" in serialized:
        serialized["clap_embedding"] = _serialize_embedding(serialized["clap_embedding"])

    cols = list(serialized.keys())
    placeholders = ", ".join("?" * len(cols))
    col_str = ", ".join(cols)
    update_str = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "file_path")
    sql = (
        f"INSERT INTO tracks ({col_str}) VALUES ({placeholders}) "
        f"ON CONFLICT(file_path) DO UPDATE SET {update_str}"
    )
    with get_conn() as conn:
        cur = conn.execute(sql, list(serialized.values()))
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute("SELECT id FROM tracks WHERE file_path=?", (serialized["file_path"],)).fetchone()
        return row["id"]


def get_track_by_path(file_path: str):
    with get_conn() as conn:
        return conn.execute(
            f"SELECT {TRACK_SELECT}, clap_embedding, waveform_data FROM tracks WHERE file_path=?",
            (file_path,),
        ).fetchone()


def build_track_query(
    bpm_min=None,
    bpm_max=None,
    camelot_key=None,
    date_from=None,
    date_to=None,
    source_folder=None,
    search=None,
    sort_by="file_date",
    sort_dir="desc",
    page=1,
    per_page=50,
):
    allowed_sort = {"bpm", "camelot_key", "file_date", "energy", "title", "artist", "duration_sec"}
    if sort_by not in allowed_sort:
        sort_by = "file_date"
    sort_dir = "ASC" if sort_dir.lower() == "asc" else "DESC"

    conditions = []
    params = []

    if bpm_min is not None:
        conditions.append("bpm >= ?")
        params.append(bpm_min)
    if bpm_max is not None:
        conditions.append("bpm <= ?")
        params.append(bpm_max)
    if camelot_key:
        keys = [k.strip() for k in camelot_key.split(",") if k.strip()]
        if keys:
            placeholders = ",".join("?" * len(keys))
            conditions.append(f"camelot_key IN ({placeholders})")
            params.extend(keys)
    if date_from is not None:
        conditions.append("file_date >= ?")
        params.append(date_from)
    if date_to is not None:
        conditions.append("file_date <= ?")
        params.append(date_to)
    if source_folder:
        conditions.append("source_folder = ?")
        params.append(source_folder)
    if search:
        conditions.append("(LOWER(title) LIKE ? OR LOWER(artist) LIKE ?)")
        like = f"%{search.lower()}%"
        params.extend([like, like])

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    offset = (page - 1) * per_page

    count_sql = f"SELECT COUNT(*) as cnt FROM tracks {where}"
    data_sql = (
        f"SELECT {TRACK_SELECT} FROM tracks {where} "
        f"ORDER BY {sort_by} {sort_dir} "
        f"LIMIT ? OFFSET ?"
    )
    return count_sql, data_sql, params, offset, per_page


def query_tracks(
    bpm_min=None,
    bpm_max=None,
    camelot_key=None,
    date_from=None,
    date_to=None,
    source_folder=None,
    search=None,
    sort_by="file_date",
    sort_dir="desc",
    page=1,
    per_page=50,
):
    count_sql, data_sql, params, offset, limit = build_track_query(
        bpm_min,
        bpm_max,
        camelot_key,
        date_from,
        date_to,
        source_folder,
        search,
        sort_by,
        sort_dir,
        page,
        per_page,
    )
    with get_conn() as conn:
        total = conn.execute(count_sql, params).fetchone()["cnt"]
        rows = conn.execute(data_sql, params + [limit, offset]).fetchall()
        return _attach_runtime_track_fields_many(rows), total


def get_tracks_with_embeddings():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT id, file_path, filename, title, artist, duration_sec, bpm, camelot_key, energy, file_date, source_folder, clap_embedding
            FROM tracks
            WHERE clap_embedding IS NOT NULL AND clap_embedding != ''
        """).fetchall()

    tracks = []
    for row in rows:
        track = _attach_runtime_track_fields(row)
        embedding = _deserialize_embedding_array(track.pop("clap_embedding"))
        if embedding is None or embedding.size == 0:
            continue
        track["embedding"] = embedding
        tracks.append(track)
    return tracks


def get_tracks_missing_embeddings() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT t.id, t.file_path
            FROM tracks t
            LEFT JOIN track_embeddings te ON te.track_id = t.id
            WHERE te.track_id IS NULL OR te.chroma_vector IS NULL OR te.chroma_vector = ''
            ORDER BY t.id
        """).fetchall()
        return [dict(row) for row in rows]


def save_embedding(track_id: int, vector: list):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO track_embeddings(track_id, chroma_vector)
            VALUES(?, ?)
            """,
            (track_id, _serialize_embedding(vector)),
        )


def get_embedding(track_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT chroma_vector FROM track_embeddings WHERE track_id = ?",
            (track_id,),
        ).fetchone()
        if not row:
            return None
        return _deserialize_embedding_list(row["chroma_vector"])


def get_all_embeddings() -> list[tuple[int, list]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT track_id, chroma_vector FROM track_embeddings ORDER BY track_id"
        ).fetchall()

    embeddings = []
    for row in rows:
        vector = _deserialize_embedding_list(row["chroma_vector"])
        if vector is None:
            continue
        embeddings.append((row["track_id"], vector))
    return embeddings


def count_embeddings() -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM track_embeddings WHERE chroma_vector IS NOT NULL AND chroma_vector != ''"
        ).fetchone()
        return row["c"]


def save_waveform(track_id: int, data_json: str):
    with get_conn() as conn:
        conn.execute("UPDATE tracks SET waveform_data=? WHERE id=?", (data_json, track_id))


def get_waveform(track_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT waveform_data FROM tracks WHERE id=?", (track_id,)).fetchone()
    if not row or not row["waveform_data"]:
        return None
    try:
        return json.loads(row["waveform_data"])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def get_stats():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM tracks").fetchone()["c"]
        tracks_with_embeddings = conn.execute(
            "SELECT COUNT(*) as c FROM track_embeddings WHERE chroma_vector IS NOT NULL AND chroma_vector != ''"
        ).fetchone()["c"]
        dur = conn.execute("SELECT SUM(duration_sec) as s FROM tracks").fetchone()["s"] or 0

        bpm_rows = conn.execute(
            "SELECT bpm FROM tracks WHERE bpm IS NOT NULL ORDER BY bpm"
        ).fetchall()
        histogram = {}
        for row in bpm_rows:
            bucket_start = int(row["bpm"] // 5) * 5
            label = f"{bucket_start}-{bucket_start+4}"
            histogram[label] = histogram.get(label, 0) + 1
        bpm_histogram = [{"range": k, "count": v} for k, v in sorted(histogram.items())]

        key_rows = conn.execute(
            "SELECT camelot_key as k, COUNT(*) as c FROM tracks WHERE camelot_key IS NOT NULL GROUP BY camelot_key ORDER BY c DESC"
        ).fetchall()
        key_distribution = [{"key": r["k"], "count": r["c"]} for r in key_rows]

        date_row = conn.execute(
            "SELECT MIN(file_date) as oldest, MAX(file_date) as newest FROM tracks WHERE file_date IS NOT NULL"
        ).fetchone()

        folder_rows = conn.execute(
            "SELECT source_folder as folder, COUNT(*) as c FROM tracks WHERE source_folder IS NOT NULL GROUP BY source_folder ORDER BY c DESC"
        ).fetchall()
        source_folders = [{"folder": r["folder"], "count": r["c"]} for r in folder_rows]

        return {
            "total_tracks": total,
            "tracks_with_embeddings": tracks_with_embeddings,
            "total_duration_sec": dur,
            "bpm_histogram": bpm_histogram,
            "key_distribution": key_distribution,
            "date_range": {"oldest": date_row["oldest"], "newest": date_row["newest"]},
            "source_folders": source_folders,
        }


def get_all_playlists():
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT p.*, COUNT(pt.id) as track_count
            FROM playlists p
            LEFT JOIN playlist_tracks pt ON pt.playlist_id = p.id
            GROUP BY p.id ORDER BY p.created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]


def create_playlist(name: str, description: str = "") -> dict:
    import time

    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO playlists(name,description,created_at,filters_json) VALUES(?,?,?,?)",
            (name, description, time.time(), "{}"),
        )
        pid = cur.lastrowid
        return dict(conn.execute("SELECT * FROM playlists WHERE id=?", (pid,)).fetchone())


def get_playlist(pid: int):
    with get_conn() as conn:
        playlist = conn.execute("SELECT * FROM playlists WHERE id=?", (pid,)).fetchone()
        if not playlist:
            return None
        tracks = conn.execute(f"""
            SELECT {TRACK_SELECT_WITH_ALIAS}, pt.position
            FROM tracks t
            JOIN playlist_tracks pt ON pt.track_id = t.id
            WHERE pt.playlist_id = ?
            ORDER BY pt.position ASC
        """, (pid,)).fetchall()
        result = dict(playlist)
        result["tracks"] = _attach_runtime_track_fields_many(tracks)
        return result


def add_tracks_to_playlist(pid: int, track_ids: list):
    with get_conn() as conn:
        max_pos = conn.execute(
            "SELECT COALESCE(MAX(position), -1) as m FROM playlist_tracks WHERE playlist_id=?",
            (pid,),
        ).fetchone()["m"]
        for index, track_id in enumerate(track_ids):
            conn.execute(
                "INSERT OR IGNORE INTO playlist_tracks(playlist_id,track_id,position) VALUES(?,?,?)",
                (pid, track_id, max_pos + 1 + index),
            )


def remove_track_from_playlist(pid: int, track_id: int):
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM playlist_tracks WHERE playlist_id=? AND track_id=?",
            (pid, track_id),
        )


def reorder_playlist(pid: int, track_ids: list):
    with get_conn() as conn:
        for position, track_id in enumerate(track_ids):
            conn.execute(
                "UPDATE playlist_tracks SET position=? WHERE playlist_id=? AND track_id=?",
                (position, pid, track_id),
            )


def delete_playlist(pid: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM playlists WHERE id=?", (pid,))


def get_tracks_by_ids(track_ids: list) -> list:
    if not track_ids:
        return []
    placeholders = ",".join("?" * len(track_ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT {TRACK_SELECT} FROM tracks WHERE id IN ({placeholders})",
            track_ids,
        ).fetchall()
        by_id = {row["id"]: _attach_runtime_track_fields(row) for row in rows}
        return [by_id[track_id] for track_id in track_ids if track_id in by_id]


def get_tracks_by_source_folders(source_folders: list[str]) -> list:
    if not source_folders:
        return []
    placeholders = ",".join("?" * len(source_folders))
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT {TRACK_SELECT}
            FROM tracks
            WHERE source_folder IN ({placeholders})
            ORDER BY source_folder, artist, title, id
            """,
            source_folders,
        ).fetchall()
        return _attach_runtime_track_fields_many(rows)


def get_track_by_id(track_id: int):
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT {TRACK_SELECT} FROM tracks WHERE id=?",
            (track_id,),
        ).fetchone()
    return _attach_runtime_track_fields(row) if row else None


def record_play(track_id: int):
    played_at = time.time()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO play_history(track_id, played_at) VALUES(?, ?)",
            (track_id, played_at),
        )
        conn.execute(
            """
            DELETE FROM play_history
            WHERE id NOT IN (
                SELECT id
                FROM play_history
                ORDER BY played_at DESC, id DESC
                LIMIT 500
            )
            """
        )


def get_play_history(limit: int = 50) -> list[dict]:
    limit = max(1, min(int(limit or 50), 500))
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT {TRACK_SELECT_WITH_ALIAS}, ph.played_at
            FROM play_history ph
            JOIN tracks t ON t.id = ph.track_id
            ORDER BY ph.played_at DESC, ph.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return _attach_runtime_track_fields_many(rows)


def delete_tracks(track_ids: list[int]) -> int:
    if not track_ids:
        return 0

    placeholders = ",".join("?" * len(track_ids))
    with get_conn() as conn:
        conn.execute(
            f"DELETE FROM playlist_tracks WHERE track_id IN ({placeholders})",
            track_ids,
        )
        deleted = conn.execute(
            f"DELETE FROM tracks WHERE id IN ({placeholders})",
            track_ids,
        ).rowcount
        return deleted


def find_duplicates():
    exact_groups = []
    exact_pairs: set[frozenset[int]] = set()

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT LOWER(title) as normalized_title,
                   LOWER(artist) as normalized_artist,
                   COUNT(*) as duplicate_count,
                   GROUP_CONCAT(id) as id_list
            FROM tracks
            GROUP BY LOWER(title), LOWER(artist)
            HAVING COUNT(*) > 1
        """).fetchall()

    for row in rows:
        track_ids = [int(value) for value in (row["id_list"] or "").split(",") if value]
        tracks = get_tracks_by_ids(track_ids)
        if len(tracks) < 2:
            continue
        exact_groups.append(_mark_duplicate_group(tracks, "exact"))
        for left_id, right_id in combinations(track_ids, 2):
            exact_pairs.add(frozenset((left_id, right_id)))

    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT {TRACK_SELECT}
            FROM tracks
            WHERE bpm IS NOT NULL
              AND camelot_key IS NOT NULL
              AND duration_sec IS NOT NULL
            ORDER BY camelot_key, bpm, duration_sec, id
            """
        ).fetchall()

    buckets: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        track = dict(row)
        bucket_key = (track["camelot_key"], round(track["bpm"]))
        buckets.setdefault(bucket_key, []).append(track)

    fuzzy_groups = []
    for bucket_tracks in buckets.values():
        if len(bucket_tracks) < 2:
            continue

        adjacency: dict[int, set[int]] = {}
        tracks_by_id = {track["id"]: track for track in bucket_tracks}

        for index, left_track in enumerate(bucket_tracks):
            for right_track in bucket_tracks[index + 1:]:
                pair = frozenset((left_track["id"], right_track["id"]))
                if pair in exact_pairs:
                    continue
                if abs((left_track.get("duration_sec") or 0) - (right_track.get("duration_sec") or 0)) >= 5:
                    continue

                adjacency.setdefault(left_track["id"], set()).add(right_track["id"])
                adjacency.setdefault(right_track["id"], set()).add(left_track["id"])

        seen_ids = set()
        for start_id in adjacency:
            if start_id in seen_ids:
                continue

            stack = [start_id]
            component_ids = set()
            while stack:
                current_id = stack.pop()
                if current_id in component_ids:
                    continue
                component_ids.add(current_id)
                stack.extend(adjacency.get(current_id, set()) - component_ids)

            seen_ids.update(component_ids)
            if len(component_ids) < 2:
                continue
            component_tracks = [tracks_by_id[track_id] for track_id in sorted(component_ids)]
            fuzzy_groups.append(_mark_duplicate_group(component_tracks, "fuzzy"))

    exact_groups.sort(key=lambda group: (-len(group["tracks"]), (group["tracks"][0].get("title") or "").lower()))
    fuzzy_groups.sort(key=lambda group: (-len(group["tracks"]), (group["tracks"][0].get("title") or "").lower()))
    return exact_groups + fuzzy_groups


def normalize_energy():
    """Rescale all track energies to [0, 1] using p5/p95 percentile normalization."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT energy FROM tracks WHERE energy IS NOT NULL AND energy > 0"
        ).fetchall()
        if not rows:
            return
        values = sorted(row["energy"] for row in rows)
        count = len(values)
        p5 = values[max(0, int(count * 0.05) - 1)]
        p95 = values[min(count - 1, int(count * 0.95))]
        if p95 == p5:
            conn.execute("UPDATE tracks SET energy = 0.5 WHERE energy IS NOT NULL")
            return
        conn.execute(
            "UPDATE tracks SET energy = MIN(1.0, MAX(0.0, "
            "CAST((energy - ?) AS REAL) / CAST((? - ?) AS REAL))) "
            "WHERE energy IS NOT NULL",
            (p5, p95, p5),
        )


def clear_database():
    with get_conn() as conn:
        conn.execute("DELETE FROM playlist_tracks")
        conn.execute("DELETE FROM playlists")
        conn.execute("DELETE FROM track_embeddings")
        conn.execute("DELETE FROM tracks")


def get_source_folders_detail():
    """Return list of {folder, track_count, last_scanned} for all source folders."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT source_folder as folder,
                   COUNT(*) as track_count,
                   MAX(analyzed_at) as last_scanned
            FROM tracks
            WHERE source_folder IS NOT NULL
            GROUP BY source_folder
            ORDER BY source_folder
        """).fetchall()
        return [dict(r) for r in rows]


def delete_source(folder: str) -> int:
    with get_conn() as conn:
        conn.execute(
            """
            DELETE FROM track_embeddings
            WHERE track_id IN (
                SELECT id FROM tracks WHERE source_folder = ?
            )
            """,
            (folder,),
        )
        cur = conn.execute("DELETE FROM tracks WHERE source_folder = ?", (folder,))
        return cur.rowcount


def reset_library(mode: str) -> int:
    with get_conn() as conn:
        if mode == "tracks_only":
            count = conn.execute("SELECT COUNT(*) as c FROM tracks").fetchone()["c"]
            conn.execute("DELETE FROM playlist_tracks")
            conn.execute("DELETE FROM track_embeddings")
            conn.execute("DELETE FROM tracks")
            return count

        count = conn.execute("SELECT COUNT(*) as c FROM tracks").fetchone()["c"]
        conn.executescript("""
            DELETE FROM playlist_tracks;
            DELETE FROM playlists;
            DELETE FROM track_embeddings;
            DELETE FROM tracks;
            DELETE FROM settings;
        """)
        return count
