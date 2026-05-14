import sqlite3
import json
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "musicden.db")


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
                analyzed_at REAL
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

            CREATE INDEX IF NOT EXISTS idx_tracks_bpm ON tracks(bpm);
            CREATE INDEX IF NOT EXISTS idx_tracks_camelot ON tracks(camelot_key);
            CREATE INDEX IF NOT EXISTS idx_tracks_file_date ON tracks(file_date);
            CREATE INDEX IF NOT EXISTS idx_playlist_tracks_playlist ON playlist_tracks(playlist_id, position);
        """)


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
    cols = list(data.keys())
    placeholders = ", ".join("?" * len(cols))
    col_str = ", ".join(cols)
    update_str = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "file_path")
    sql = (
        f"INSERT INTO tracks ({col_str}) VALUES ({placeholders}) "
        f"ON CONFLICT(file_path) DO UPDATE SET {update_str}"
    )
    with get_conn() as conn:
        cur = conn.execute(sql, list(data.values()))
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute("SELECT id FROM tracks WHERE file_path=?", (data["file_path"],)).fetchone()
        return row["id"]


def get_track_by_path(file_path: str):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM tracks WHERE file_path=?", (file_path,)).fetchone()


def build_track_query(
    bpm_min=None, bpm_max=None, camelot_key=None,
    date_from=None, date_to=None, source_folder=None,
    search=None, sort_by="file_date", sort_dir="desc",
    page=1, per_page=50,
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
        f"SELECT * FROM tracks {where} "
        f"ORDER BY {sort_by} {sort_dir} "
        f"LIMIT ? OFFSET ?"
    )
    return count_sql, data_sql, params, offset, per_page


def query_tracks(
    bpm_min=None, bpm_max=None, camelot_key=None,
    date_from=None, date_to=None, source_folder=None,
    search=None, sort_by="file_date", sort_dir="desc",
    page=1, per_page=50,
):
    count_sql, data_sql, params, offset, limit = build_track_query(
        bpm_min, bpm_max, camelot_key, date_from, date_to,
        source_folder, search, sort_by, sort_dir, page, per_page,
    )
    with get_conn() as conn:
        total = conn.execute(count_sql, params).fetchone()["cnt"]
        rows = conn.execute(data_sql, params + [limit, offset]).fetchall()
        return [dict(r) for r in rows], total


def get_stats():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM tracks").fetchone()["c"]
        dur = conn.execute("SELECT SUM(duration_sec) as s FROM tracks").fetchone()["s"] or 0

        # BPM histogram in 5-bpm buckets
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
            "total_duration_sec": dur,
            "bpm_histogram": bpm_histogram,
            "key_distribution": key_distribution,
            "date_range": {"oldest": date_row["oldest"], "newest": date_row["newest"]},
            "source_folders": source_folders,
        }


# ── Playlist helpers ─────────────────────────────────────────────────────────

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
        p = conn.execute("SELECT * FROM playlists WHERE id=?", (pid,)).fetchone()
        if not p:
            return None
        tracks = conn.execute("""
            SELECT t.*, pt.position FROM tracks t
            JOIN playlist_tracks pt ON pt.track_id = t.id
            WHERE pt.playlist_id = ?
            ORDER BY pt.position ASC
        """, (pid,)).fetchall()
        result = dict(p)
        result["tracks"] = [dict(r) for r in tracks]
        return result


def add_tracks_to_playlist(pid: int, track_ids: list):
    with get_conn() as conn:
        max_pos = conn.execute(
            "SELECT COALESCE(MAX(position), -1) as m FROM playlist_tracks WHERE playlist_id=?", (pid,)
        ).fetchone()["m"]
        for i, tid in enumerate(track_ids):
            conn.execute(
                "INSERT OR IGNORE INTO playlist_tracks(playlist_id,track_id,position) VALUES(?,?,?)",
                (pid, tid, max_pos + 1 + i),
            )


def remove_track_from_playlist(pid: int, track_id: int):
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM playlist_tracks WHERE playlist_id=? AND track_id=?", (pid, track_id)
        )


def reorder_playlist(pid: int, track_ids: list):
    with get_conn() as conn:
        for pos, tid in enumerate(track_ids):
            conn.execute(
                "UPDATE playlist_tracks SET position=? WHERE playlist_id=? AND track_id=?",
                (pos, pid, tid),
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
            f"SELECT * FROM tracks WHERE id IN ({placeholders})", track_ids
        ).fetchall()
        by_id = {r["id"]: dict(r) for r in rows}
        return [by_id[tid] for tid in track_ids if tid in by_id]


def get_track_by_id(track_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM tracks WHERE id=?", (track_id,)).fetchone()


def normalize_energy():
    """Rescale all track energies to [0, 1] using min-max over the whole library."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(energy) as mn, MAX(energy) as mx FROM tracks WHERE energy IS NOT NULL"
        ).fetchone()
        mn, mx = row["mn"], row["mx"]
        if mn is None or mx is None or mx == mn:
            return
        conn.execute(
            "UPDATE tracks SET energy = (energy - ?) / (? - ?) WHERE energy IS NOT NULL",
            (mn, mx, mn),
        )


def clear_database():
    with get_conn() as conn:
        conn.execute("DELETE FROM playlist_tracks")
        conn.execute("DELETE FROM playlists")
        conn.execute("DELETE FROM tracks")
