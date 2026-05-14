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
