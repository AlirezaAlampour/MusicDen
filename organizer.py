import os
import shutil
import re
from pathlib import Path

ILLEGAL_CHARS = re.compile(r'[\\/:*?"<>|]')


def sanitize(name: str) -> str:
    return ILLEGAL_CHARS.sub("_", name).strip(". ")


def dest_subdir(track: dict) -> str:
    return track.get("camelot_key") or "Unknown"


def dest_filename(track: dict) -> str:
    artist = sanitize(track.get("artist") or "Unknown Artist")
    title = sanitize(track.get("title") or track.get("filename") or "Unknown")
    bpm = track.get("bpm")
    bpm_str = f" ({round(float(bpm), 1)}bpm)" if bpm is not None else ""
    ext = os.path.splitext(track.get("file_path", ""))[1].lower()
    return f"{artist} - {title}{bpm_str}{ext}"


def _safe_output_path(path: str) -> str:
    if ".." in Path(path).parts:
        raise ValueError("Path traversal not allowed in output path")
    resolved = os.path.abspath(path)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _track_quality_rank(track: dict) -> int:
    ext = os.path.splitext(track.get("file_path", ""))[1].lower()
    if ext == ".flac":
        return 2
    if ext == ".wav":
        return 1
    return 0


def _track_file_size(track: dict) -> int:
    try:
        return os.path.getsize(track.get("file_path", ""))
    except OSError:
        return 0


def _track_identity_key(track: dict) -> tuple[str, str]:
    title = (track.get("title") or track.get("filename") or "").strip().lower()
    artist = (track.get("artist") or "").strip().lower()
    return title, artist


def flat_filename(track: dict) -> str:
    artist = sanitize(track.get("artist") or "Unknown Artist")
    title = sanitize(track.get("title") or track.get("filename") or "Unknown")
    ext = os.path.splitext(track.get("file_path", ""))[1].lower()
    return f"{artist} - {title}{ext}"


def _unique_destination(dest_dir: str, filename: str) -> str:
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{stem}_{suffix}{ext}")
        suffix += 1
    return candidate


def _consolidated_destination(track: dict, output_folder: str, mode: str) -> str:
    if mode == "flat":
        dest_dir = output_folder
        filename = flat_filename(track)
    elif mode == "by_source":
        source_name = sanitize(os.path.basename(track.get("source_folder") or "") or "Unknown Source")
        dest_dir = os.path.join(output_folder, source_name)
        filename = flat_filename(track)
    elif mode == "organized":
        dest_dir = os.path.join(output_folder, dest_subdir(track))
        filename = dest_filename(track)
    else:
        raise ValueError(f"Unsupported consolidate mode: {mode}")

    os.makedirs(dest_dir, exist_ok=True)
    return _unique_destination(dest_dir, filename)


def organize_tracks(tracks: list, output_folder: str) -> dict:
    output_folder = _safe_output_path(output_folder)

    copied = 0
    skipped = 0
    errors = []

    for track in tracks:
        src = track.get("file_path", "")
        if not os.path.isfile(src):
            errors.append({"file": src, "error": "Source file not found"})
            continue

        dest_dir = os.path.join(output_folder, dest_subdir(track))
        fname = dest_filename(track)

        try:
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, fname)
            if os.path.exists(dest):
                skipped += 1
                continue
            shutil.copy2(src, dest)
            copied += 1
        except Exception as e:
            errors.append({"file": src, "error": str(e)})

    return {"copied": copied, "skipped": skipped, "errors": errors}


def consolidate_tracks(tracks: list, output_folder: str, mode: str, progress_callback=None) -> dict:
    output_folder = _safe_output_path(output_folder)
    ordered_tracks = sorted(
        tracks,
        key=lambda track: (
            _track_identity_key(track),
            -_track_quality_rank(track),
            -_track_file_size(track),
            track.get("id") or 0,
        ),
    )

    copied = 0
    skipped = 0
    seen_keys: set[tuple[str, str]] = set()
    total = len(ordered_tracks)

    for index, track in enumerate(ordered_tracks, start=1):
        src = track.get("file_path", "")
        current = os.path.basename(src) or (track.get("filename") or "")
        identity_key = _track_identity_key(track)
        ext = os.path.splitext(src)[1].lower()

        if identity_key in seen_keys or ext not in {".flac", ".wav"} or not os.path.isfile(src):
            skipped += 1
        else:
            try:
                dest = _consolidated_destination(track, output_folder, mode)
                if os.path.abspath(src) == os.path.abspath(dest):
                    skipped += 1
                else:
                    shutil.copy2(src, dest)
                    copied += 1
                    seen_keys.add(identity_key)
            except Exception:
                skipped += 1

        if progress_callback:
            progress_callback(
                {
                    "processed": index,
                    "total": total,
                    "current": current,
                    "done": False,
                    "copied": copied,
                    "skipped": skipped,
                }
            )

    result = {
        "processed": total,
        "total": total,
        "current": "",
        "done": True,
        "copied": copied,
        "skipped": skipped,
    }
    if progress_callback:
        progress_callback(result)
    return result
