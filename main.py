import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import analyzer
import database as db
import downloader as dl
import organizer

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
APP_STATE_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "MusicDen")
AUTH_TOKEN_PATH = os.path.join(APP_STATE_DIR, "auth.token")
ARL_MASK = "***"
LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "default_download_path": "D:/Music/DeezerDownloads",
    "default_quality": "FLAC",
    "library_folders": [],
    "auto_scan_startup": True,
    "port": 7337,
    "arl_token": "",
}

DEEZER_URL_RE = re.compile(r"^https://www\.deezer\.com/(en/)?(playlist|album|track)/\d+$")
VALID_QUALITIES = {"FLAC", "MP3_320", "MP3_128"}

_scan_state = {
    "active": False,
    "total": 0,
    "processed": 0,
    "current_file": "",
    "errors": 0,
    "new": 0,
    "skipped": 0,
    "embedding_status": "",
}

_executor = ThreadPoolExecutor(max_workers=4)
_consolidate_jobs = {}
_consolidate_lock = threading.Lock()
_usb_export_jobs = {}
_usb_export_lock = threading.Lock()


def _load_or_create_auth_token() -> str:
    os.makedirs(APP_STATE_DIR, exist_ok=True)
    if os.path.exists(AUTH_TOKEN_PATH):
        with open(AUTH_TOKEN_PATH, encoding="utf-8") as file_obj:
            token = file_obj.read().strip()
        if token:
            return token

    token = secrets.token_urlsafe(32)
    with open(AUTH_TOKEN_PATH, "w", encoding="utf-8") as file_obj:
        file_obj.write(token)
    try:
        os.chmod(AUTH_TOKEN_PATH, 0o600)
    except OSError:
        pass
    return token


_AUTH_TOKEN = _load_or_create_auth_token()


def _get_allowed_root_prefixes() -> list[str]:
    if os.name == "nt":
        roots = {os.path.realpath(BASE_DIR)}
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            drive_root = f"{letter}:{os.sep}"
            if os.path.exists(drive_root):
                roots.add(os.path.realpath(drive_root))
        return sorted(roots, key=len, reverse=True)
    return [os.path.realpath(os.sep)]


def _starts_with_allowed_root(resolved_path: str, allowed_roots: list[str]) -> bool:
    normalized_path = os.path.normcase(resolved_path)
    for root in allowed_roots:
        normalized_root = os.path.normcase(os.path.realpath(root))
        root_with_sep = normalized_root if normalized_root.endswith(os.sep) else normalized_root + os.sep
        if normalized_path == normalized_root or normalized_path.startswith(root_with_sep):
            return True
    return False


def _validate_path(path: str, must_exist: bool = False, label: str = "Path") -> str:
    candidate = (path or "").strip()
    if not candidate:
        raise HTTPException(400, f"{label} is required")
    if candidate.startswith("\\\\"):
        raise HTTPException(400, f"{label} UNC paths are not allowed")

    resolved = os.path.realpath(candidate)
    if resolved.startswith("\\\\"):
        raise HTTPException(400, f"{label} UNC paths are not allowed")
    if not _starts_with_allowed_root(resolved, _get_allowed_root_prefixes()):
        raise HTTPException(400, f"{label} must stay within an allowed local root")
    if must_exist and not os.path.exists(resolved):
        raise HTTPException(400, f"{label} does not exist: {resolved}")
    return resolved


def _to_client_path(path: str) -> str:
    if not path:
        return ""
    normalized = os.path.abspath(path).replace("\\", "/")
    if re.fullmatch(r"[A-Za-z]:", normalized):
        return normalized + "/"
    return normalized


def _validate_folder_list(
    folders: list[str],
    *,
    label: str = "Folder",
    must_exist: bool = True,
    strict: bool = True,
) -> list[str]:
    validated = []
    seen = set()
    for folder in folders or []:
        candidate = (folder or "").strip()
        if not candidate:
            continue
        try:
            resolved = _validate_path(candidate, must_exist=must_exist, label=label)
        except HTTPException:
            if strict:
                raise
            LOGGER.warning("Skipping invalid %s: %s", label.lower(), candidate)
            continue
        if not os.path.isdir(resolved):
            if strict:
                raise HTTPException(400, f"{label} is not a directory: {resolved}")
            LOGGER.warning("Skipping non-directory %s: %s", label.lower(), resolved)
            continue
        if resolved not in seen:
            seen.add(resolved)
            validated.append(resolved)
    return validated


def _reset_scan_state(total_files: int):
    global _scan_state
    _scan_state = {
        "active": True,
        "total": total_files,
        "processed": 0,
        "current_file": "",
        "errors": 0,
        "new": 0,
        "skipped": 0,
        "embedding_status": "Indexing audio embeddings..." if total_files else "",
    }


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as file_obj:
            return {**DEFAULT_CONFIG, **json.load(file_obj)}
    return dict(DEFAULT_CONFIG)


def save_config(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2)


def _mask_config(config: dict) -> dict:
    masked = dict(config)
    masked["arl_token"] = ARL_MASK if config.get("arl_token") else ""
    return masked


app = FastAPI(title="MusicDen")

db.init_db()
if not os.path.exists(CONFIG_PATH):
    save_config(DEFAULT_CONFIG)


@app.middleware("http")
async def musicden_auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        token = request.headers.get("X-MusicDen-Token", "") or request.query_params.get("token", "")
        if token != _AUTH_TOKEN:
            return JSONResponse({"detail": "Missing or invalid X-MusicDen-Token"}, status_code=401)
    return await call_next(request)


@app.on_event("startup")
async def preload_clap_model():
    def _preload():
        try:
            analyzer.load_clap_model()
        except Exception:
            LOGGER.exception("Background CLAP preload failed")

    threading.Thread(target=_preload, name="musicden-clap-preload", daemon=True).start()


@app.on_event("startup")
async def auto_scan_library_on_startup():
    config = load_config()
    if not config.get("auto_scan_startup", True):
        return

    folders = _validate_folder_list(
        config.get("library_folders") or [],
        label="Startup scan folder",
        must_exist=True,
        strict=False,
    )
    if not folders or _scan_state["active"]:
        return

    try:
        await _start_scan_job(folders, rescan=False, persist_config=False)
        LOGGER.info("Started MusicDen startup auto-scan for %d folder(s)", len(folders))
    except Exception:
        LOGGER.exception("MusicDen startup auto-scan failed")


def _collect_files(folders: List[str]) -> List[str]:
    files = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                if filename.lower().endswith((".flac", ".wav")):
                    files.append(os.path.join(root, filename))
    return files


def _analyze_one(file_path: str, rescan: bool):
    global _scan_state

    _scan_state["current_file"] = os.path.basename(file_path)
    _scan_state["embedding_status"] = "Indexing audio embeddings..."

    if not rescan:
        existing = db.get_track_by_path(file_path)
        if existing:
            mtime = os.path.getmtime(file_path)
            has_embedding = bool(existing["clap_embedding"])
            if existing["file_date"] and abs(existing["file_date"] - mtime) < 1 and has_embedding:
                _scan_state["skipped"] += 1
                _scan_state["processed"] += 1
                return

    try:
        result = analyzer.analyze_file(file_path)
        db.upsert_track(result)
        _scan_state["new"] += 1
    except Exception:
        LOGGER.exception("Scan analysis failed for %s", file_path)
        _scan_state["errors"] += 1
    finally:
        _scan_state["processed"] += 1


async def _run_scan_files(files: list[str], rescan: bool):
    global _scan_state
    loop = asyncio.get_running_loop()
    try:
        futures = [loop.run_in_executor(_executor, _analyze_one, file_path, rescan) for file_path in files]
        if futures:
            await asyncio.gather(*futures)
        await loop.run_in_executor(None, db.normalize_energy)
    finally:
        _scan_state["active"] = False
        _scan_state["current_file"] = ""
        _scan_state["embedding_status"] = ""


async def _start_scan_job(validated_folders: list[str], *, rescan: bool, persist_config: bool) -> int:
    if _scan_state["active"]:
        raise HTTPException(409, "Scan already in progress")

    if persist_config:
        config = load_config()
        config["library_folders"] = [_to_client_path(folder) for folder in validated_folders]
        save_config(config)

    files = _collect_files(validated_folders)
    _reset_scan_state(len(files))
    asyncio.create_task(_run_scan_files(files, rescan))
    return len(files)


def _semantic_results_for_embedding(query_embedding: np.ndarray, tracks: list[dict], limit: int, exclude_track_id: Optional[int] = None) -> list[dict]:
    query_vector = np.asarray(query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0.0:
        return []

    results = []
    for track in tracks:
        if exclude_track_id is not None and track["id"] == exclude_track_id:
            continue

        embedding = np.asarray(track["embedding"], dtype=np.float32)
        embedding_norm = float(np.linalg.norm(embedding))
        if embedding_norm == 0.0:
            continue

        similarity = float(np.dot(query_vector, embedding) / (query_norm * embedding_norm))
        results.append(
            {
                "id": track["id"],
                "file_path": track["file_path"],
                "filename": track.get("filename"),
                "title": track.get("title"),
                "artist": track.get("artist"),
                "duration_sec": track.get("duration_sec"),
                "bpm": track.get("bpm"),
                "camelot_key": track.get("camelot_key"),
                "energy": track.get("energy"),
                "file_date": track.get("file_date"),
                "source_folder": track.get("source_folder"),
                "similarity": similarity,
            }
        )

    results.sort(key=lambda item: item["similarity"], reverse=True)
    return results[:limit]


def cosine_sim(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _escape_xml(value):
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _rekordbox_location(path: str) -> str:
    import urllib.parse

    forward = path.replace("\\", "/")
    encoded = urllib.parse.quote(forward, safe="/:@")
    return f"file:///{encoded}"


def _build_rekordbox_xml(playlist_name: str, tracks: list[dict]) -> str:
    from datetime import datetime

    tracks_xml = []
    for track in tracks:
        timestamp = track.get("file_date") or time.time()
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        bpm = f"{float(track['bpm']):.2f}" if track.get("bpm") else "0.00"
        location = _rekordbox_location(track["file_path"])
        tracks_xml.append(
            f'    <TRACK TrackID="{track["id"]}" Name="{_escape_xml(track.get("title", ""))}" '
            f'Artist="{_escape_xml(track.get("artist", ""))}" Album="{_escape_xml(track.get("album", ""))}" '
            f'Genre="{_escape_xml(track.get("genre", ""))}" TotalTime="{int(track.get("duration_sec") or 0)}" '
            f'BPM="{bpm}" Tonality="{_escape_xml(track.get("camelot_key", ""))}" '
            f'DateAdded="{date_str}" Location="{location}" />'
        )

    playlist_nodes = "\n".join(f'      <TRACK Key="{track["id"]}" />' for track in tracks)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <COLLECTION Entries="{len(tracks)}">
{chr(10).join(tracks_xml)}
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT">
      <NODE Name="{_escape_xml(playlist_name)}" Type="1" KeyType="0" Entries="{len(tracks)}">
{playlist_nodes}
      </NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>"""


def _resample_waveform(values: list[float], points: int) -> list[float]:
    if points <= 0:
        return []
    if not values:
        return [0.0] * points
    if len(values) == points:
        return [float(max(0.0, min(1.0, value))) for value in values]
    x_old = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=points, dtype=np.float32)
    resampled = np.interp(x_new, x_old, np.asarray(values, dtype=np.float32))
    return [float(max(0.0, min(1.0, value))) for value in resampled.tolist()]


def _compute_waveform_samples(file_path: str, points: int) -> list[float]:
    import librosa

    y, _sr = librosa.load(file_path, sr=22050, mono=True, duration=None)
    if y.size == 0:
        return [0.0] * points

    chunks = np.array_split(y, points)
    peaks = np.array(
        [float(np.max(np.abs(chunk))) if chunk.size else 0.0 for chunk in chunks],
        dtype=np.float32,
    )
    max_peak = float(np.max(peaks)) if peaks.size else 0.0
    if max_peak > 0:
        peaks = peaks / max_peak
    return [float(max(0.0, min(1.0, value))) for value in peaks.tolist()]


def _similarity_ranked_embeddings(track_id: int, limit: int) -> tuple[list[float], list[tuple[int, list[float], float]]]:
    target_embedding = db.get_embedding(track_id)
    if target_embedding is None:
        raise HTTPException(404, "Run Compute Embeddings in Settings first")

    ranked = []
    for other_track_id, vector in db.get_all_embeddings():
        if other_track_id == track_id:
            continue
        ranked.append((other_track_id, vector, cosine_sim(target_embedding, vector)))

    ranked.sort(key=lambda item: item[2], reverse=True)
    return target_embedding, ranked[:limit]


def _resolve_track_selection(playlist_id: Optional[int] = None, track_ids: Optional[list[int]] = None) -> list[dict]:
    if playlist_id is not None:
        playlist = db.get_playlist(playlist_id)
        if not playlist:
            raise HTTPException(404, "Playlist not found")
        return playlist["tracks"]

    if track_ids:
        tracks = db.get_tracks_by_ids(track_ids)
        if not tracks:
            raise HTTPException(400, "No matching tracks found")
        return tracks

    raise HTTPException(400, "Provide playlist_id or track_ids")


def _update_usb_export_job(job_id: str, update: dict):
    with _usb_export_lock:
        if job_id in _usb_export_jobs:
            _usb_export_jobs[job_id].update(update)


def _usb_export_destination(track: dict, output_base: str, structure: str) -> str:
    if structure == "organized":
        dest_dir = os.path.join(output_base, organizer.dest_subdir(track))
        filename = organizer.dest_filename(track)
    elif structure == "by_key":
        dest_dir = os.path.join(output_base, organizer.dest_subdir(track))
        filename = organizer.flat_filename(track)
    elif structure == "flat":
        dest_dir = output_base
        filename = organizer.flat_filename(track)
    else:
        raise ValueError(f"Unsupported USB export structure: {structure}")

    os.makedirs(dest_dir, exist_ok=True)
    return os.path.join(dest_dir, filename)


def _start_usb_export_job(
    job_id: str,
    tracks: list[dict],
    output_base: str,
    folder_name: str,
    structure: str,
    include_rekordbox_xml: bool,
):
    def _worker():
        copied = 0
        skipped = 0
        errors = []
        total = len(tracks)
        exported_tracks = []
        seen_identity = set()

        try:
            os.makedirs(output_base, exist_ok=True)
            for index, track in enumerate(tracks, start=1):
                src = track.get("file_path", "")
                current = os.path.basename(src) or (track.get("filename") or "")
                ext = os.path.splitext(src)[1].lower()
                title = (track.get("title") or track.get("filename") or "").strip().lower()
                artist = (track.get("artist") or "").strip().lower()
                identity = (title, artist)

                if ext not in {".flac", ".wav"}:
                    skipped += 1
                elif any(identity) and identity in seen_identity:
                    skipped += 1
                elif not os.path.isfile(src):
                    errors.append({"file": src, "error": "Source file not found"})
                else:
                    try:
                        if any(identity):
                            seen_identity.add(identity)
                        dest = _usb_export_destination(track, output_base, structure)
                        if os.path.exists(dest):
                            skipped += 1
                        else:
                            shutil.copy2(src, dest)
                            copied += 1
                        exported_tracks.append({**track, "file_path": dest})
                    except Exception as exc:
                        errors.append({"file": src, "error": str(exc)})

                _update_usb_export_job(
                    job_id,
                    {
                        "copied": copied,
                        "processed": index,
                        "total": total,
                        "current": current,
                        "done": False,
                        "skipped": skipped,
                        "errors": errors[-10:],
                    },
                )

            xml_written = False
            if include_rekordbox_xml:
                xml_path = os.path.join(output_base, "rekordbox.xml")
                with open(xml_path, "w", encoding="utf-8") as file_obj:
                    file_obj.write(_build_rekordbox_xml(folder_name, exported_tracks))
                xml_written = True

            _update_usb_export_job(
                job_id,
                {
                    "copied": copied,
                    "processed": total,
                    "total": total,
                    "current": "",
                    "done": True,
                    "skipped": skipped,
                    "errors": errors,
                    "output_base": output_base,
                    "rekordbox_xml_written": xml_written,
                },
            )
        except Exception as exc:
            LOGGER.exception("USB export failed for job %s", job_id)
            errors.append({"file": "", "error": str(exc)})
            _update_usb_export_job(
                job_id,
                {
                    "done": True,
                    "current": "",
                    "errors": errors,
                },
            )

    threading.Thread(target=_worker, name=f"musicden-usb-export-{job_id}", daemon=True).start()


def _compute_track_embedding(track: dict) -> int:
    try:
        import librosa

        y, sr = librosa.load(track["file_path"], sr=22050, mono=True)
        vector = analyzer.compute_embedding(y, sr)
        db.save_embedding(track["id"], vector)
        return 1
    except Exception:
        LOGGER.exception("Chroma embedding failed for %s", track["file_path"])
        return 0


def _update_consolidate_job(job_id: str, update: dict):
    with _consolidate_lock:
        if job_id in _consolidate_jobs:
            _consolidate_jobs[job_id].update(update)


def _start_consolidate_job(job_id: str, source_folders: list[str], output_folder: str, mode: str):
    def _worker():
        try:
            tracks = db.get_tracks_by_source_folders(source_folders)
            _update_consolidate_job(job_id, {"total": len(tracks), "current": ""})
            if not tracks:
                _update_consolidate_job(
                    job_id,
                    {"processed": 0, "total": 0, "done": True, "copied": 0, "skipped": 0},
                )
                return

            organizer.consolidate_tracks(
                tracks,
                output_folder,
                mode,
                progress_callback=lambda payload: _update_consolidate_job(job_id, payload),
            )
        except Exception as exc:
            LOGGER.exception("Consolidation failed for job %s", job_id)
            _update_consolidate_job(job_id, {"done": True, "error": str(exc)})

    threading.Thread(target=_worker, name=f"musicden-consolidate-{job_id}", daemon=True).start()


def _cover_art_response(data: bytes, mime_type: str):
    return Response(content=data, media_type=mime_type, headers={"Cache-Control": "max-age=3600"})


def _folder_has_audio_one_level(folder_path: str) -> bool:
    try:
        for entry in os.scandir(folder_path):
            try:
                if entry.is_file() and entry.name.lower().endswith((".flac", ".wav")):
                    return True
                if entry.is_dir(follow_symlinks=False):
                    try:
                        for child in os.scandir(entry.path):
                            if child.is_file() and child.name.lower().endswith((".flac", ".wav")):
                                return True
                    except OSError:
                        continue
            except OSError:
                continue
    except OSError:
        return False
    return False


class SettingsPayload(BaseModel):
    default_download_path: Optional[str] = None
    default_quality: Optional[str] = None
    library_folders: Optional[List[str]] = None
    auto_scan_startup: Optional[bool] = None
    port: Optional[int] = None
    arl_token: Optional[str] = None


class ScanPayload(BaseModel):
    folders: List[str]
    rescan: bool = False


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


class OrganizePayload(BaseModel):
    track_ids: Optional[List[int]] = None
    playlist_id: Optional[int] = None
    output_folder: str


class OrganizeByDatePayload(BaseModel):
    source_folder: str


class DownloadPayload(BaseModel):
    url: str
    quality: str = "FLAC"
    output_path: str


class LibraryResetPayload(BaseModel):
    mode: str = "tracks_only"


class SemanticSearchPayload(BaseModel):
    query: str
    limit: int = 20


class DuplicateRemovePayload(BaseModel):
    track_ids: List[int]


class ConsolidatePayload(BaseModel):
    source_folders: List[str]
    output_folder: str
    mode: str


class UsbExportPayload(BaseModel):
    playlist_id: Optional[int] = None
    track_ids: Optional[List[int]] = None
    drive: str
    folder_name: str = "MusicDen_Export"
    structure: str
    include_rekordbox_xml: bool = True


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/token")
def get_token():
    return JSONResponse({"token": _AUTH_TOKEN}, headers={"Cache-Control": "no-store"})


@app.get("/api/settings")
def get_settings():
    return _mask_config(load_config())


@app.get("/api/browse")
def browse_folders(path: Optional[str] = None):
    raw_path = (path or "").strip()
    if not raw_path:
        drives = [f"{drive}:/" for drive in string.ascii_uppercase if os.path.exists(f"{drive}:/")]
        return {
            "path": "",
            "parent": None,
            "items": [{"name": drive, "type": "drive", "path": drive} for drive in drives],
        }

    current_path = _validate_path(raw_path, must_exist=True, label="Browse path")
    if not os.path.isdir(current_path):
        raise HTTPException(400, "Cannot access this folder")

    try:
        items = []
        for entry in os.scandir(current_path):
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue

            resolved_entry = os.path.realpath(entry.path)
            if not _starts_with_allowed_root(resolved_entry, _get_allowed_root_prefixes()):
                continue

            items.append(
                {
                    "name": entry.name,
                    "type": "folder",
                    "path": _to_client_path(resolved_entry),
                    "has_audio": _folder_has_audio_one_level(resolved_entry),
                }
            )
    except OSError as exc:
        raise HTTPException(400, "Cannot access this folder") from exc

    items.sort(key=lambda item: item["name"].lower())
    stripped = current_path.rstrip("\\/")
    parent = os.path.dirname(stripped)
    if not parent or parent == stripped:
        parent_path = None
    else:
        parent_path = _to_client_path(parent)

    return {
        "path": _to_client_path(current_path),
        "parent": parent_path,
        "items": items,
    }


@app.put("/api/settings")
def update_settings(payload: SettingsPayload):
    config = load_config()
    if payload.default_download_path is not None:
        config["default_download_path"] = payload.default_download_path
    if payload.default_quality is not None:
        config["default_quality"] = payload.default_quality
    if payload.library_folders is not None:
        config["library_folders"] = [_to_client_path(folder) for folder in payload.library_folders if (folder or "").strip()]
    if payload.auto_scan_startup is not None:
        config["auto_scan_startup"] = payload.auto_scan_startup
    if payload.port is not None:
        if not (1024 <= payload.port <= 65535):
            raise HTTPException(400, "Port must be between 1024 and 65535")
        config["port"] = payload.port
    if payload.arl_token is not None and payload.arl_token != ARL_MASK:
        config["arl_token"] = payload.arl_token
    save_config(config)
    return _mask_config(config)


@app.post("/api/scan")
async def scan_library(payload: ScanPayload):
    validated_folders = _validate_folder_list(payload.folders, label="Scan folder", must_exist=True, strict=True)
    total_files = await _start_scan_job(validated_folders, rescan=payload.rescan, persist_config=True)

    return {
        "total": total_files,
        "new": 0,
        "skipped": 0,
        "errors": 0,
        "message": "Scan started",
    }


@app.get("/api/scan/progress")
async def scan_progress():
    async def generate():
        while _scan_state["active"] or _scan_state["processed"] < _scan_state["total"]:
            data = json.dumps(
                {
                    "processed": _scan_state["processed"],
                    "total": _scan_state["total"],
                    "current_file": _scan_state["current_file"],
                    "new": _scan_state["new"],
                    "skipped": _scan_state["skipped"],
                    "errors": _scan_state["errors"],
                    "active": _scan_state["active"],
                    "embedding_status": _scan_state["embedding_status"],
                    "clap": analyzer.get_clap_status(),
                }
            )
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.5)

        data = json.dumps(
            {
                "processed": _scan_state["processed"],
                "total": _scan_state["total"],
                "current_file": "",
                "new": _scan_state["new"],
                "skipped": _scan_state["skipped"],
                "errors": _scan_state["errors"],
                "active": False,
                "done": True,
                "embedding_status": "",
                "clap": analyzer.get_clap_status(),
            }
        )
        yield f"data: {data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.put("/api/tracks/renormalize")
def renormalize_energy():
    db.normalize_energy()
    return {"ok": True}


@app.post("/api/tracks/reanalyze-keys")
async def reanalyze_keys():
    with db.get_conn() as conn:
        rows = conn.execute("SELECT id, file_path FROM tracks").fetchall()
        tracks = [dict(row) for row in rows]

    def _rekey_one(track):
        try:
            import librosa

            y, sr = librosa.load(track["file_path"], sr=22050, mono=True, duration=60)
            camelot = analyzer.detect_key(y, sr)
            with db.get_conn() as conn:
                conn.execute("UPDATE tracks SET camelot_key=? WHERE id=?", (camelot, track["id"]))
            return 1
        except Exception:
            return 0

    loop = asyncio.get_running_loop()
    futures = [loop.run_in_executor(_executor, _rekey_one, track) for track in tracks]
    results = await asyncio.gather(*futures)
    return {"updated": sum(results)}


@app.post("/api/tracks/reanalyze-energy")
async def reanalyze_energy():
    with db.get_conn() as conn:
        rows = conn.execute("SELECT id, file_path FROM tracks").fetchall()
        tracks = [dict(row) for row in rows]

    def _reenergy_one(track):
        try:
            import librosa

            y, sr = librosa.load(track["file_path"], sr=22050, mono=True, duration=60)
            energy = analyzer.compute_energy(y, sr)
            with db.get_conn() as conn:
                conn.execute("UPDATE tracks SET energy=? WHERE id=?", (energy, track["id"]))
            return 1
        except Exception:
            return 0

    loop = asyncio.get_running_loop()
    futures = [loop.run_in_executor(_executor, _reenergy_one, track) for track in tracks]
    results = await asyncio.gather(*futures)
    db.normalize_energy()
    return {"updated": sum(results)}


@app.post("/api/tracks/search/semantic")
def semantic_track_search(payload: SemanticSearchPayload):
    query = payload.query.strip()
    if not query:
        raise HTTPException(400, "Query is required")

    limit = max(1, min(payload.limit, 200))
    tracks = db.get_tracks_with_embeddings()
    if not tracks:
        return {"tracks": [], "query": query, "limit": limit}

    try:
        query_embedding = analyzer.generate_clap_text_embedding(query)
    except Exception as exc:
        raise HTTPException(503, f"CLAP model unavailable: {exc}") from exc

    return {
        "tracks": _semantic_results_for_embedding(query_embedding, tracks, limit),
        "query": query,
        "limit": limit,
    }


@app.post("/api/tracks/compute-embeddings")
async def compute_track_embeddings():
    tracks = db.get_tracks_missing_embeddings()
    with db.get_conn() as conn:
        total_tracks = conn.execute("SELECT COUNT(*) as c FROM tracks").fetchone()["c"]

    already_had = max(0, total_tracks - len(tracks))
    if not tracks:
        return {"computed": 0, "already_had": already_had}

    loop = asyncio.get_running_loop()
    futures = [loop.run_in_executor(_executor, _compute_track_embedding, track) for track in tracks]
    results = await asyncio.gather(*futures)
    return {"computed": sum(results), "already_had": already_had}


@app.get("/api/tracks/embedding-map")
def embedding_map(track_id: int, limit: int = 50):
    target_embedding, ranked = _similarity_ranked_embeddings(track_id, max(1, min(limit, 200)))

    target_track = db.get_track_by_id(track_id)
    if not target_track:
        raise HTTPException(404, "Track not found")

    similar_track_ids = [other_track_id for other_track_id, _vector, _similarity in ranked]
    similar_tracks = db.get_tracks_by_ids(similar_track_ids)
    similar_by_id = {track["id"]: track for track in similar_tracks}
    vectors = [np.asarray(target_embedding, dtype=np.float32)] + [
        np.asarray(vector, dtype=np.float32) for _other_track_id, vector, _similarity in ranked
    ]

    if len(vectors) >= 2:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2).fit_transform(np.vstack(vectors))
    else:
        coords = np.zeros((1, 2), dtype=np.float32)

    target_payload = {
        "track_id": target_track["id"],
        "x": float(coords[0][0]),
        "y": float(coords[0][1]),
        "title": target_track.get("title"),
        "artist": target_track.get("artist"),
        "camelot_key": target_track.get("camelot_key"),
    }

    similar_payload = []
    for index, (other_track_id, _vector, similarity) in enumerate(ranked, start=1):
        track = similar_by_id.get(other_track_id)
        if not track:
            continue
        similar_payload.append(
            {
                "track_id": track["id"],
                "x": float(coords[index][0]),
                "y": float(coords[index][1]),
                "title": track.get("title"),
                "artist": track.get("artist"),
                "camelot_key": track.get("camelot_key"),
                "bpm": track.get("bpm"),
                "similarity": similarity,
            }
        )

    return {"target": target_payload, "similar": similar_payload}


@app.get("/api/tracks/{track_id}/similar")
def similar_tracks(track_id: int, limit: int = 10):
    limit = max(1, min(limit, 200))
    _target_embedding, ranked = _similarity_ranked_embeddings(track_id, limit)
    top_matches = [(other_track_id, similarity) for other_track_id, _vector, similarity in ranked]
    tracks = db.get_tracks_by_ids([track_id for track_id, _similarity in top_matches])
    tracks_by_id = {track["id"]: track for track in tracks}

    results = []
    for other_track_id, similarity in top_matches:
        track = tracks_by_id.get(other_track_id)
        if not track:
            continue
        result = dict(track)
        result["similarity"] = similarity
        results.append(result)

    return {
        "tracks": results,
        "source_track_id": track_id,
        "limit": limit,
    }


@app.get("/api/tracks/{track_id}/waveform")
def track_waveform(track_id: int, points: int = 300):
    points = max(10, min(points, 2000))
    track = db.get_track_by_id(track_id)
    if not track:
        raise HTTPException(404, "Track not found")

    cached = db.get_waveform(track_id)
    if cached:
        return {"waveform": _resample_waveform(cached, points)}

    file_path = track["file_path"]
    if not os.path.isfile(file_path):
        raise HTTPException(404, "File not found on disk")

    base_points = max(points, 400)
    waveform = _compute_waveform_samples(file_path, base_points)
    db.save_waveform(track_id, json.dumps(waveform))
    return {"waveform": _resample_waveform(waveform, points)}


@app.get("/api/tracks/{track_id}/cover")
def track_cover(track_id: int):
    track = db.get_track_by_id(track_id)
    if not track:
        raise HTTPException(404, "Track not found")

    file_path = track["file_path"]
    if not os.path.isfile(file_path):
        raise HTTPException(404, "File not found on disk")

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".flac":
            from mutagen.flac import FLAC

            flac = FLAC(file_path)
            if flac.pictures:
                pic = flac.pictures[0]
                return _cover_art_response(pic.data, pic.mime)
        elif ext == ".wav":
            from mutagen.id3 import ID3

            tags = ID3(file_path)
            apic_frames = tags.getall("APIC")
            if apic_frames:
                apic = apic_frames[0]
                return _cover_art_response(apic.data, apic.mime)
    except Exception:
        pass

    raise HTTPException(404, "No embedded cover art found")


@app.get("/api/stream/{track_id}")
def stream_track(track_id: int):
    track = db.get_track_by_id(track_id)
    if not track:
        raise HTTPException(404, "Track not found")
    path = track["file_path"]
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found on disk")
    extension = os.path.splitext(path)[1].lower()
    media_type = "audio/flac" if extension == ".flac" else "audio/wav"
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
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        camelot_key=camelot_key,
        date_from=date_from,
        date_to=date_to,
        source_folder=source_folder,
        search=search,
        sort_by=sort_by,
        sort_dir=sort_dir,
        page=page,
        per_page=per_page,
    )
    return {"tracks": tracks, "total": total, "page": page, "per_page": per_page}


@app.post("/api/tracks/resolve")
def resolve_tracks(payload: TrackIds):
    return {"tracks": db.get_tracks_by_ids(payload.track_ids)}


@app.get("/api/tracks/stats")
def get_stats():
    return db.get_stats()


@app.post("/api/tracks/{track_id}/played")
def track_played(track_id: int):
    track = db.get_track_by_id(track_id)
    if not track:
        raise HTTPException(404, "Track not found")
    db.record_play(track_id)
    return {"ok": True}


@app.get("/api/history")
def get_history(limit: int = 50):
    return {"tracks": db.get_play_history(limit)}


@app.get("/api/duplicates")
def get_duplicates():
    groups = db.find_duplicates()
    exact_count = sum(1 for group in groups if group["match_type"] == "exact")
    fuzzy_count = sum(1 for group in groups if group["match_type"] == "fuzzy")
    return {
        "groups": groups,
        "exact_count": exact_count,
        "fuzzy_count": fuzzy_count,
        "group_count": len(groups),
    }


@app.post("/api/duplicates/remove")
def remove_duplicates(payload: DuplicateRemovePayload):
    return {"removed": db.delete_tracks(payload.track_ids)}


@app.get("/api/playlists")
def list_playlists():
    return db.get_all_playlists()


@app.post("/api/playlists")
def create_playlist(payload: PlaylistCreate):
    return db.create_playlist(payload.name, payload.description)


@app.get("/api/playlists/{pid}")
def get_playlist(pid: int):
    playlist = db.get_playlist(pid)
    if not playlist:
        raise HTTPException(404, "Playlist not found")
    return playlist


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
    playlist = db.get_playlist(pid)
    if not playlist:
        raise HTTPException(404, "Playlist not found")

    tracks = playlist["tracks"]
    if not tracks:
        return {"ok": True}

    if payload.sort_by == "camelot_harmonic":
        ordered = _harmonic_sort(tracks)
    else:
        allowed = {"bpm", "camelot_key", "file_date", "energy", "title", "artist"}
        if payload.sort_by not in allowed:
            raise HTTPException(400, f"Invalid sort_by: {payload.sort_by}")
        reverse = payload.sort_dir.lower() == "desc"
        ordered = sorted(tracks, key=lambda track: (track.get(payload.sort_by) or ""), reverse=reverse)

    db.reorder_playlist(pid, [track["id"] for track in ordered])
    return db.get_playlist(pid)


def _camelot_neighbors(key: str) -> set:
    if not key or len(key) < 2:
        return set()
    try:
        number = int(key[:-1])
        letter = key[-1]
    except ValueError:
        return set()
    neighbors = {key}
    for neighbor_number in [(number - 1) % 12 or 12, (number + 1) % 12 or 12]:
        neighbors.add(f"{neighbor_number}{letter}")
    other_letter = "B" if letter == "A" else "A"
    neighbors.add(f"{number}{other_letter}")
    return neighbors


def _harmonic_sort(tracks: list) -> list:
    if not tracks:
        return tracks
    remaining = list(tracks)
    ordered = [remaining.pop(0)]
    while remaining:
        last_key = ordered[-1].get("camelot_key") or ""
        neighbors = _camelot_neighbors(last_key)
        best_index = None
        for index, track in enumerate(remaining):
            if track.get("camelot_key") in neighbors:
                best_index = index
                break
        if best_index is None:
            best_index = 0
        ordered.append(remaining.pop(best_index))
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
    playlist = db.create_playlist(payload.name)
    if tracks:
        db.add_tracks_to_playlist(playlist["id"], [track["id"] for track in tracks])
    return db.get_playlist(playlist["id"])


@app.post("/api/playlists/{pid}/export/m3u")
def export_m3u(pid: int, payload: ExportPayload):
    playlist = db.get_playlist(pid)
    if not playlist:
        raise HTTPException(404, "Playlist not found")

    output_dir = _validate_path(payload.output_path, label="Output path")
    os.makedirs(output_dir, exist_ok=True)
    safe_name = "".join(char if char.isalnum() or char in " _-" else "_" for char in playlist["name"])
    output_file = os.path.join(output_dir, f"{safe_name}.m3u8")

    lines = ["#EXTM3U"]
    for track in playlist["tracks"]:
        duration = int(track.get("duration_sec") or 0)
        artist = track.get("artist") or ""
        title = track.get("title") or track.get("filename") or ""
        lines.append(f"#EXTINF:{duration},{artist} - {title}")
        lines.append(track["file_path"])

    with open(output_file, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))

    return {"path": output_file}


@app.post("/api/playlists/{pid}/export/rekordbox")
def export_rekordbox(pid: int, payload: ExportPayload):
    playlist = db.get_playlist(pid)
    if not playlist:
        raise HTTPException(404, "Playlist not found")

    output_dir = _validate_path(payload.output_path, label="Output path")
    os.makedirs(output_dir, exist_ok=True)
    safe_name = "".join(char if char.isalnum() or char in " _-" else "_" for char in playlist["name"])
    output_file = os.path.join(output_dir, f"{safe_name}_rekordbox.xml")

    with open(output_file, "w", encoding="utf-8") as file_obj:
        file_obj.write(_build_rekordbox_xml(playlist["name"], playlist["tracks"]))

    return {"path": output_file}


@app.post("/api/organize")
def organize(payload: OrganizePayload):
    output_folder = _validate_path(payload.output_folder, label="Output folder")

    if payload.playlist_id is not None:
        playlist = db.get_playlist(payload.playlist_id)
        if not playlist:
            raise HTTPException(404, "Playlist not found")
        tracks = playlist["tracks"]
    elif payload.track_ids:
        tracks = db.get_tracks_by_ids(payload.track_ids)
    else:
        raise HTTPException(400, "Provide track_ids or playlist_id")

    return organizer.organize_tracks(tracks, output_folder)


@app.post("/api/organize-by-date")
def organize_by_date(payload: OrganizeByDatePayload):
    import shutil
    from datetime import datetime

    source_folder = _validate_path(payload.source_folder, must_exist=True, label="Source folder")
    if not os.path.isdir(source_folder):
        return JSONResponse({"error": "Source folder is not a directory"}, status_code=400)

    organized = 0
    folders_created: set[str] = set()

    for filename in os.listdir(source_folder):
        if not filename.lower().endswith((".flac", ".wav")):
            continue
        file_path = os.path.join(source_folder, filename)
        if not os.path.isfile(file_path):
            continue
        month_str = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m")
        destination_dir = os.path.join(source_folder, month_str)
        os.makedirs(destination_dir, exist_ok=True)
        destination_path = os.path.join(destination_dir, filename)
        if not os.path.exists(destination_path):
            shutil.copy2(file_path, destination_path)
        folders_created.add(month_str)
        organized += 1

    return {"organized": organized, "folders": sorted(folders_created)}


@app.post("/api/consolidate")
def consolidate(payload: ConsolidatePayload):
    if payload.mode not in {"flat", "by_source", "organized"}:
        raise HTTPException(400, "mode must be one of: flat, by_source, organized")
    if not payload.source_folders:
        raise HTTPException(400, "Select at least one source folder")

    validated_sources = []
    seen_sources = set()
    for folder in payload.source_folders:
        validated = _validate_path(folder, must_exist=True, label="Source folder")
        if not os.path.isdir(validated):
            raise HTTPException(400, f"Source folder is not a directory: {validated}")
        if validated not in seen_sources:
            seen_sources.add(validated)
            validated_sources.append(validated)

    output_folder = _validate_path(payload.output_folder, label="Output folder")
    job_id = secrets.token_hex(8)
    with _consolidate_lock:
        _consolidate_jobs[job_id] = {
            "processed": 0,
            "total": 0,
            "current": "",
            "done": False,
            "copied": 0,
            "skipped": 0,
            "error": "",
            "output_folder": output_folder,
            "mode": payload.mode,
        }

    _start_consolidate_job(job_id, validated_sources, output_folder, payload.mode)
    return {"job_id": job_id}


@app.get("/api/consolidate/{job_id}/status")
async def consolidate_status(job_id: str):
    with _consolidate_lock:
        if job_id not in _consolidate_jobs:
            raise HTTPException(404, "Consolidation job not found")

    async def generate():
        while True:
            with _consolidate_lock:
                state = dict(_consolidate_jobs.get(job_id, {}))
            yield f"data: {json.dumps(state)}\n\n"
            if state.get("done"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/drives")
def list_drives():
    drives = []
    for letter in string.ascii_uppercase:
        path = f"{letter}:/"
        if os.path.exists(path):
            total, used, free = shutil.disk_usage(path)
            drives.append(
                {
                    "letter": path,
                    "free_gb": round(free / 1e9, 1),
                    "total_gb": round(total / 1e9, 1),
                }
            )
    return {"drives": drives}


@app.post("/api/export/usb")
def export_usb(payload: UsbExportPayload):
    if payload.structure not in {"organized", "flat", "by_key"}:
        raise HTTPException(400, "structure must be one of: organized, flat, by_key")

    tracks = _resolve_track_selection(payload.playlist_id, payload.track_ids)
    if not tracks:
        raise HTTPException(400, "No tracks selected for export")

    drive = _validate_path(payload.drive, must_exist=True, label="Drive")
    if not os.path.isdir(drive):
        raise HTTPException(400, "Drive is not accessible")

    folder_name = organizer.sanitize(payload.folder_name or "MusicDen_Export") or "MusicDen_Export"
    output_base = os.path.join(drive, folder_name)

    ordered_tracks = []
    seen_ids = set()
    for track in tracks:
        track_id = track.get("id")
        if track_id in seen_ids:
            continue
        seen_ids.add(track_id)
        ordered_tracks.append(track)

    job_id = secrets.token_hex(8)
    with _usb_export_lock:
        _usb_export_jobs[job_id] = {
            "copied": 0,
            "processed": 0,
            "total": len(ordered_tracks),
            "current": "",
            "done": False,
            "skipped": 0,
            "errors": [],
            "output_base": output_base,
            "rekordbox_xml_written": False,
        }

    _start_usb_export_job(
        job_id,
        ordered_tracks,
        output_base,
        folder_name,
        payload.structure,
        payload.include_rekordbox_xml,
    )
    return {"job_id": job_id}


@app.get("/api/export/usb/{job_id}/status")
async def export_usb_status(job_id: str):
    with _usb_export_lock:
        if job_id not in _usb_export_jobs:
            raise HTTPException(404, "USB export job not found")

    async def generate():
        while True:
            with _usb_export_lock:
                state = dict(_usb_export_jobs.get(job_id, {}))
            yield f"data: {json.dumps(state)}\n\n"
            if state.get("done"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/download")
def start_download(payload: DownloadPayload):
    if not DEEZER_URL_RE.match(payload.url):
        return JSONResponse({"error": "Invalid Deezer URL"}, status_code=400)
    if payload.quality not in VALID_QUALITIES:
        return JSONResponse(
            {"error": f"Invalid quality. Must be one of: {', '.join(sorted(VALID_QUALITIES))}"},
            status_code=400,
        )

    output_path = _validate_path(payload.output_path, label="Output path")
    if not os.path.isdir(output_path):
        return JSONResponse({"error": "Output path does not exist or is not a directory"}, status_code=400)

    config = load_config()
    arl = config.get("arl_token", "").strip()
    if not arl:
        return JSONResponse({"error": "ARL token not set. Add it in Settings."}, status_code=400)

    job_id = dl.start_download(payload.url, payload.quality, output_path, arl)
    return {"job_id": job_id}


@app.get("/api/download/{job_id}/stream")
async def stream_download(job_id: str):
    return StreamingResponse(
        dl.stream_job_lines(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.delete("/api/database")
def clear_database():
    db.clear_database()
    return {"ok": True}


@app.get("/api/sources")
def list_sources():
    return db.get_source_folders_detail()


@app.delete("/api/sources/{encoded_path:path}")
def delete_source(encoded_path: str):
    import urllib.parse

    folder = urllib.parse.unquote(encoded_path)
    removed = db.delete_source(folder)
    return {"removed": removed}


@app.post("/api/library/reset")
def reset_library(payload: LibraryResetPayload):
    if payload.mode not in ("tracks_only", "everything"):
        return JSONResponse({"error": "mode must be 'tracks_only' or 'everything'"}, status_code=400)
    cleared = db.reset_library(payload.mode)
    return {"cleared": cleared}
