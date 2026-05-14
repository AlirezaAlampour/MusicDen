import os
import re
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database as db
import analyzer
import downloader as dl
import organizer

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_CONFIG = {
    "default_download_path": "D:/Music/DeezerDownloads",
    "default_quality": "FLAC",
    "library_folders": [],
    "port": 7337,
    "arl_token": "",
}

# ── Input validation constants ────────────────────────────────────────────────

DEEZER_URL_RE = re.compile(
    r'^https://www\.deezer\.com/(en/)?(playlist|album|track)/\d+$'
)
VALID_QUALITIES = {"FLAC", "MP3_320", "MP3_128"}


def _validate_path(path: str, must_exist: bool = False, label: str = "Path") -> str:
    """Resolve to absolute path; reject traversal sequences and optionally require existence."""
    if ".." in Path(path).parts:
        raise HTTPException(400, f"{label} traversal not allowed")
    resolved = os.path.abspath(path)
    if must_exist and not os.path.exists(resolved):
        raise HTTPException(400, f"{label} does not exist: {resolved}")
    return resolved

# ── Scan progress state ──────────────────────────────────────────────────────
_scan_state = {
    "active": False,
    "total": 0,
    "processed": 0,
    "current_file": "",
    "errors": 0,
    "new": 0,
    "skipped": 0,
}

_executor = ThreadPoolExecutor(max_workers=4)


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


app = FastAPI(title="MusicDen")

db.init_db()
if not os.path.exists(CONFIG_PATH):
    save_config(DEFAULT_CONFIG)


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


# ── Settings ──────────────────────────────────────────────────────────────────

class SettingsPayload(BaseModel):
    default_download_path: Optional[str] = None
    default_quality: Optional[str] = None
    library_folders: Optional[List[str]] = None
    port: Optional[int] = None
    arl_token: Optional[str] = None


@app.get("/api/settings")
def get_settings():
    return load_config()


@app.put("/api/settings")
def update_settings(payload: SettingsPayload):
    cfg = load_config()
    if payload.default_download_path is not None:
        cfg["default_download_path"] = payload.default_download_path
    if payload.default_quality is not None:
        cfg["default_quality"] = payload.default_quality
    if payload.library_folders is not None:
        cfg["library_folders"] = payload.library_folders
    if payload.port is not None:
        if not (1024 <= payload.port <= 65535):
            raise HTTPException(400, "Port must be between 1024 and 65535")
        cfg["port"] = payload.port
    if payload.arl_token is not None:
        cfg["arl_token"] = payload.arl_token
    save_config(cfg)
    return cfg


# ── Scan ──────────────────────────────────────────────────────────────────────

class ScanPayload(BaseModel):
    folders: List[str]
    rescan: bool = False


def _collect_files(folders: List[str]) -> List[str]:
    files = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for root, _, filenames in os.walk(folder):
            for fname in filenames:
                if fname.lower().endswith((".flac", ".wav")):
                    files.append(os.path.join(root, fname))
    return files


def _analyze_one(file_path: str, rescan: bool):
    global _scan_state
    _scan_state["current_file"] = os.path.basename(file_path)

    if not rescan:
        existing = db.get_track_by_path(file_path)
        if existing:
            mtime = os.path.getmtime(file_path)
            if existing["file_date"] and abs(existing["file_date"] - mtime) < 1:
                _scan_state["skipped"] += 1
                _scan_state["processed"] += 1
                return

    try:
        result = analyzer.analyze_file(file_path)
        db.upsert_track(result)
        _scan_state["new"] += 1
    except Exception:
        _scan_state["errors"] += 1
    finally:
        _scan_state["processed"] += 1


@app.post("/api/scan")
async def scan_library(payload: ScanPayload):
    global _scan_state
    if _scan_state["active"]:
        raise HTTPException(409, "Scan already in progress")

    validated_folders = []
    for folder in payload.folders:
        vf = _validate_path(folder, must_exist=True, label="Scan folder")
        if not os.path.isdir(vf):
            raise HTTPException(400, f"Not a directory: {vf}")
        validated_folders.append(vf)
    files = _collect_files(validated_folders)
    _scan_state = {
        "active": True,
        "total": len(files),
        "processed": 0,
        "current_file": "",
        "errors": 0,
        "new": 0,
        "skipped": 0,
    }

    loop = asyncio.get_event_loop()

    async def run_scan():
        global _scan_state
        futures = [
            loop.run_in_executor(_executor, _analyze_one, f, payload.rescan)
            for f in files
        ]
        await asyncio.gather(*futures)
        await loop.run_in_executor(None, db.normalize_energy)
        _scan_state["active"] = False

    asyncio.create_task(run_scan())

    return {
        "total": len(files),
        "new": 0,
        "skipped": 0,
        "errors": 0,
        "message": "Scan started",
    }


@app.get("/api/scan/progress")
async def scan_progress():
    async def generate():
        while _scan_state["active"] or _scan_state["processed"] < _scan_state["total"]:
            data = json.dumps({
                "processed": _scan_state["processed"],
                "total": _scan_state["total"],
                "current_file": _scan_state["current_file"],
                "new": _scan_state["new"],
                "skipped": _scan_state["skipped"],
                "errors": _scan_state["errors"],
                "active": _scan_state["active"],
            })
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.5)
        # send final state
        data = json.dumps({
            "processed": _scan_state["processed"],
            "total": _scan_state["total"],
            "current_file": "",
            "new": _scan_state["new"],
            "skipped": _scan_state["skipped"],
            "errors": _scan_state["errors"],
            "active": False,
            "done": True,
        })
        yield f"data: {data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Tracks ────────────────────────────────────────────────────────────────────

@app.put("/api/tracks/renormalize")
def renormalize_energy():
    db.normalize_energy()
    return {"ok": True}


@app.post("/api/tracks/reanalyze-keys")
async def reanalyze_keys():
    with db.get_conn() as conn:
        rows = conn.execute("SELECT id, file_path FROM tracks").fetchall()
        tracks = [dict(r) for r in rows]

    def _rekey_one(track):
        try:
            import librosa
            y, sr = librosa.load(track["file_path"], sr=22050, mono=True, duration=60)
            camelot = analyzer.detect_key(y, sr)
            with db.get_conn() as conn:
                conn.execute(
                    "UPDATE tracks SET camelot_key=? WHERE id=?",
                    (camelot, track["id"]),
                )
            return 1
        except Exception:
            return 0

    loop = asyncio.get_event_loop()
    futures = [loop.run_in_executor(_executor, _rekey_one, t) for t in tracks]
    results = await asyncio.gather(*futures)
    return {"updated": sum(results)}


@app.get("/api/stream/{track_id}")
def stream_track(track_id: int):
    track = db.get_track_by_id(track_id)
    if not track:
        raise HTTPException(404, "Track not found")
    path = track["file_path"]
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found on disk")
    ext = os.path.splitext(path)[1].lower()
    media_type = "audio/flac" if ext == ".flac" else "audio/wav"
    return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})


@app.get("/api/tracks")
def get_tracks(
    bpm_min: Optional[float] = None,
    bpm_max: Optional[float] = None,
    camelot_key: Optional[str] = None,
    date_from: Optional[float] = None,
    date_to: Optional[float] = None,
    source_folder: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "file_date",
    sort_dir: str = "desc",
    page: int = 1,
    per_page: int = 50,
):
    tracks, total = db.query_tracks(
        bpm_min=bpm_min, bpm_max=bpm_max, camelot_key=camelot_key,
        date_from=date_from, date_to=date_to, source_folder=source_folder,
        search=search, sort_by=sort_by, sort_dir=sort_dir,
        page=page, per_page=per_page,
    )
    return {"tracks": tracks, "total": total, "page": page, "per_page": per_page}


@app.get("/api/tracks/stats")
def get_stats():
    return db.get_stats()


# ── Playlists ─────────────────────────────────────────────────────────────────

class PlaylistCreate(BaseModel):
    name: str
    description: str = ""


class TrackIds(BaseModel):
    track_ids: List[int]


class ReorderPayload(BaseModel):
    track_ids: List[int]


class SortPayload(BaseModel):
    sort_by: str
    sort_dir: str = "asc"


class GeneratePlaylistPayload(BaseModel):
    name: str
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None
    camelot_keys: Optional[List[str]] = None
    date_from: Optional[float] = None
    date_to: Optional[float] = None
    limit: Optional[int] = 100
    sort_by: str = "bpm"
    sort_dir: str = "asc"


class ExportPayload(BaseModel):
    output_path: str


@app.get("/api/playlists")
def list_playlists():
    return db.get_all_playlists()


@app.post("/api/playlists")
def create_playlist(payload: PlaylistCreate):
    return db.create_playlist(payload.name, payload.description)


@app.get("/api/playlists/{pid}")
def get_playlist(pid: int):
    pl = db.get_playlist(pid)
    if not pl:
        raise HTTPException(404, "Playlist not found")
    return pl


@app.delete("/api/playlists/{pid}")
def delete_playlist(pid: int):
    db.delete_playlist(pid)
    return {"ok": True}


@app.post("/api/playlists/{pid}/tracks")
def add_tracks(pid: int, payload: TrackIds):
    db.add_tracks_to_playlist(pid, payload.track_ids)
    return db.get_playlist(pid)


@app.delete("/api/playlists/{pid}/tracks/{track_id}")
def remove_track(pid: int, track_id: int):
    db.remove_track_from_playlist(pid, track_id)
    return {"ok": True}


@app.put("/api/playlists/{pid}/tracks/reorder")
def reorder_tracks(pid: int, payload: ReorderPayload):
    db.reorder_playlist(pid, payload.track_ids)
    return {"ok": True}


@app.post("/api/playlists/{pid}/sort")
def sort_playlist(pid: int, payload: SortPayload):
    pl = db.get_playlist(pid)
    if not pl:
        raise HTTPException(404, "Playlist not found")

    tracks = pl["tracks"]
    if not tracks:
        return {"ok": True}

    if payload.sort_by == "camelot_harmonic":
        ordered = _harmonic_sort(tracks)
    else:
        allowed = {"bpm", "camelot_key", "file_date", "energy", "title", "artist"}
        if payload.sort_by not in allowed:
            raise HTTPException(400, f"Invalid sort_by: {payload.sort_by}")
        rev = payload.sort_dir.lower() == "desc"
        ordered = sorted(tracks, key=lambda t: (t.get(payload.sort_by) or ""), reverse=rev)

    db.reorder_playlist(pid, [t["id"] for t in ordered])
    return db.get_playlist(pid)


def _camelot_neighbors(key: str) -> set:
    if not key or len(key) < 2:
        return set()
    try:
        num = int(key[:-1])
        letter = key[-1]
    except ValueError:
        return set()
    neighbors = {key}
    for n in [(num - 1) % 12 or 12, (num + 1) % 12 or 12]:
        neighbors.add(f"{n}{letter}")
    other = "B" if letter == "A" else "A"
    neighbors.add(f"{num}{other}")
    return neighbors


def _harmonic_sort(tracks: list) -> list:
    if not tracks:
        return tracks
    remaining = list(tracks)
    ordered = [remaining.pop(0)]
    while remaining:
        last_key = ordered[-1].get("camelot_key") or ""
        neighbors = _camelot_neighbors(last_key)
        best_idx = None
        for i, t in enumerate(remaining):
            if t.get("camelot_key") in neighbors:
                best_idx = i
                break
        if best_idx is None:
            best_idx = 0
        ordered.append(remaining.pop(best_idx))
    return ordered


@app.post("/api/playlists/generate")
def generate_playlist(payload: GeneratePlaylistPayload):
    camelot_key = ",".join(payload.camelot_keys) if payload.camelot_keys else None
    tracks, _ = db.query_tracks(
        bpm_min=payload.bpm_min,
        bpm_max=payload.bpm_max,
        camelot_key=camelot_key,
        date_from=payload.date_from,
        date_to=payload.date_to,
        sort_by=payload.sort_by,
        sort_dir=payload.sort_dir,
        page=1,
        per_page=payload.limit or 100,
    )
    pl = db.create_playlist(payload.name)
    if tracks:
        db.add_tracks_to_playlist(pl["id"], [t["id"] for t in tracks])
    return db.get_playlist(pl["id"])


@app.post("/api/playlists/{pid}/export/m3u")
def export_m3u(pid: int, payload: ExportPayload):
    pl = db.get_playlist(pid)
    if not pl:
        raise HTTPException(404, "Playlist not found")

    out_dir = _validate_path(payload.output_path, label="Output path")
    os.makedirs(out_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in pl["name"])
    out_file = os.path.join(out_dir, f"{safe_name}.m3u8")

    lines = ["#EXTM3U"]
    for t in pl["tracks"]:
        dur = int(t.get("duration_sec") or 0)
        artist = t.get("artist") or ""
        title = t.get("title") or t.get("filename") or ""
        lines.append(f'#EXTINF:{dur},{artist} - {title}')
        lines.append(t["file_path"])

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"path": out_file}


@app.post("/api/playlists/{pid}/export/rekordbox")
def export_rekordbox(pid: int, payload: ExportPayload):
    import urllib.parse
    from datetime import datetime

    pl = db.get_playlist(pid)
    if not pl:
        raise HTTPException(404, "Playlist not found")

    out_dir = _validate_path(payload.output_path, label="Output path")
    os.makedirs(out_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in pl["name"])
    out_file = os.path.join(out_dir, f"{safe_name}_rekordbox.xml")

    def esc(s):
        return (str(s)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))

    def to_location(path: str) -> str:
        fwd = path.replace("\\", "/")
        encoded = urllib.parse.quote(fwd, safe="/:@")
        return f"file:///{encoded}"

    tracks_xml = []
    for i, t in enumerate(pl["tracks"], 1):
        ts = t.get("file_date") or time.time()
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        bpm = f"{float(t['bpm']):.2f}" if t.get("bpm") else "0.00"
        loc = to_location(t["file_path"])
        tracks_xml.append(
            f'    <TRACK TrackID="{t["id"]}" Name="{esc(t.get("title",""))}" '
            f'Artist="{esc(t.get("artist",""))}" Album="{esc(t.get("album",""))}" '
            f'Genre="{esc(t.get("genre",""))}" TotalTime="{int(t.get("duration_sec") or 0)}" '
            f'BPM="{bpm}" Tonality="{esc(t.get("camelot_key",""))}" '
            f'DateAdded="{date_str}" Location="{loc}" />'
        )

    playlist_nodes = "\n".join(
        f'      <TRACK Key="{t["id"]}" />' for t in pl["tracks"]
    )

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <COLLECTION Entries="{len(pl["tracks"])}">
{chr(10).join(tracks_xml)}
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT">
      <NODE Name="{esc(pl["name"])}" Type="1" KeyType="0" Entries="{len(pl["tracks"])}">
{playlist_nodes}
      </NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(xml)

    return {"path": out_file}


# ── Organize ──────────────────────────────────────────────────────────────────

class OrganizePayload(BaseModel):
    track_ids: Optional[List[int]] = None
    playlist_id: Optional[int] = None
    output_folder: str


@app.post("/api/organize")
def organize(payload: OrganizePayload):
    # Path traversal check — makedirs is done inside organizer after validation
    _validate_path(payload.output_folder, label="Output folder")

    if payload.playlist_id is not None:
        pl = db.get_playlist(payload.playlist_id)
        if not pl:
            raise HTTPException(404, "Playlist not found")
        tracks = pl["tracks"]
    elif payload.track_ids:
        tracks = db.get_tracks_by_ids(payload.track_ids)
    else:
        raise HTTPException(400, "Provide track_ids or playlist_id")

    return organizer.organize_tracks(tracks, payload.output_folder)


# ── Downloads ─────────────────────────────────────────────────────────────────

class DownloadPayload(BaseModel):
    url: str
    quality: str = "FLAC"
    output_path: str


@app.post("/api/download")
def start_download(payload: DownloadPayload):
    if not DEEZER_URL_RE.match(payload.url):
        raise HTTPException(400, {"error": "Invalid Deezer URL"})
    if payload.quality not in VALID_QUALITIES:
        raise HTTPException(400, {"error": f"Invalid quality. Must be one of: {', '.join(sorted(VALID_QUALITIES))}"})
    out = _validate_path(payload.output_path, label="Output path")
    if not os.path.isdir(out):
        raise HTTPException(400, {"error": "Output path does not exist or is not a directory"})
    cfg = load_config()
    arl = cfg.get("arl_token", "").strip()
    if not arl:
        raise HTTPException(400, {"error": "ARL token not set. Add it in Settings."})
    job_id = dl.start_download(payload.url, payload.quality, out)
    return {"job_id": job_id}


@app.get("/api/download/{job_id}/stream")
async def stream_download(job_id: str):
    return StreamingResponse(
        dl.stream_job_lines(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── DB Management ─────────────────────────────────────────────────────────────

@app.delete("/api/database")
def clear_database():
    db.clear_database()
    return {"ok": True}
