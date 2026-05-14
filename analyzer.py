import os
import time
import numpy as np

MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

CAMELOT_MAP = {
    ("C", "major"): "8B",  ("G", "major"): "9B",  ("D", "major"): "10B",
    ("A", "major"): "11B", ("E", "major"): "12B", ("B", "major"): "1B",
    ("F#", "major"): "2B", ("Db", "major"): "3B", ("Ab", "major"): "4B",
    ("Eb", "major"): "5B", ("Bb", "major"): "6B", ("F", "major"): "7B",
    ("A", "minor"): "8A",  ("E", "minor"): "9A",  ("B", "minor"): "10A",
    ("F#", "minor"): "11A", ("Db", "minor"): "12A", ("Ab", "minor"): "1A",
    ("Eb", "minor"): "2A", ("Bb", "minor"): "3A", ("F", "minor"): "4A",
    ("C", "minor"): "5A",  ("G", "minor"): "6A",  ("D", "minor"): "7A",
}


def _correlate(profile, chroma_mean):
    profile = np.array(profile)
    profile -= profile.mean()
    chroma = np.array(chroma_mean)
    chroma -= chroma.mean()
    denom = np.std(profile) * np.std(chroma)
    if denom == 0:
        return 0.0
    return float(np.dot(profile, chroma) / (len(profile) * denom))


def detect_key(chroma_mean):
    best_corr = -2.0
    best_root = 0
    best_mode = "major"

    for root in range(12):
        rotated_major = np.roll(MAJOR_PROFILE, root)
        rotated_minor = np.roll(MINOR_PROFILE, root)

        c_major = _correlate(rotated_major, chroma_mean)
        c_minor = _correlate(rotated_minor, chroma_mean)

        if c_major > best_corr:
            best_corr = c_major
            best_root = root
            best_mode = "major"
        if c_minor > best_corr:
            best_corr = c_minor
            best_root = root
            best_mode = "minor"

    note = NOTE_NAMES[best_root]
    return note, best_mode


def analyze_file(file_path: str) -> dict:
    import librosa
    import mutagen

    result = {
        "file_path": file_path,
        "filename": os.path.basename(file_path),
        "file_date": os.path.getmtime(file_path),
        "source_folder": os.path.dirname(file_path),
        "analyzed_at": time.time(),
    }

    # Read tags
    try:
        tags = mutagen.File(file_path, easy=True)
        if tags:
            result["title"] = str(tags.get("title", [os.path.splitext(os.path.basename(file_path))[0]])[0])
            result["artist"] = str(tags.get("artist", [""])[0])
            result["album"] = str(tags.get("album", [""])[0])
            result["genre"] = str(tags.get("genre", [""])[0])
            result["year"] = str(tags.get("date", [""])[0])
        else:
            result["title"] = os.path.splitext(os.path.basename(file_path))[0]
            result["artist"] = ""
            result["album"] = ""
            result["genre"] = ""
            result["year"] = ""
    except Exception:
        result["title"] = os.path.splitext(os.path.basename(file_path))[0]
        result["artist"] = ""
        result["album"] = ""
        result["genre"] = ""
        result["year"] = ""

    # Audio analysis (first 60s for speed)
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=60)

        # Duration (full file via mutagen for accuracy)
        try:
            full = mutagen.File(file_path)
            result["duration_sec"] = float(full.info.length) if full and hasattr(full, "info") else float(len(y) / sr)
        except Exception:
            result["duration_sec"] = float(len(y) / sr)

        # BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        result["bpm"] = round(float(tempo), 1)

        # Key via Krumhansl-Schmuckler
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        note, mode = detect_key(chroma_mean)
        result["camelot_key"] = CAMELOT_MAP.get((note, mode), "")

        # Energy — store raw RMS; global normalization runs after each scan
        rms = librosa.feature.rms(y=y)
        result["energy"] = float(np.mean(rms))

    except Exception as e:
        result["bpm"] = None
        result["camelot_key"] = None
        result["energy"] = None
        result["duration_sec"] = None

    return result
