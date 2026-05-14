import os
import time
import numpy as np

NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

CAMELOT_MAP = {
    ('C', 'major'): '8B',   ('G', 'major'): '9B',   ('D', 'major'): '10B',
    ('A', 'major'): '11B',  ('E', 'major'): '12B',  ('B', 'major'): '1B',
    ('F#', 'major'): '2B',  ('Db', 'major'): '3B',  ('Ab', 'major'): '4B',
    ('Eb', 'major'): '5B',  ('Bb', 'major'): '6B',  ('F', 'major'): '7B',
    ('A', 'minor'): '8A',   ('E', 'minor'): '9A',   ('B', 'minor'): '10A',
    ('F#', 'minor'): '11A', ('Db', 'minor'): '12A', ('Ab', 'minor'): '1A',
    ('Eb', 'minor'): '2A',  ('Bb', 'minor'): '3A',  ('F', 'minor'): '4A',
    ('C', 'minor'): '5A',   ('G', 'minor'): '6A',   ('D', 'minor'): '7A',
}

# Temperley profiles — better for dance/electronic music
_MAJOR_PROFILE = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
_MINOR_PROFILE = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]


def detect_key(y, sr) -> str:
    """Detect musical key using CENS chroma + Temperley profiles. Returns Camelot string."""
    import librosa
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    best_corr = -np.inf
    best_key = 'C'
    best_mode = 'major'

    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        major_corr = np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1]
        minor_corr = np.corrcoef(rotated, _MINOR_PROFILE)[0, 1]
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = NOTE_NAMES[i]
            best_mode = 'major'
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = NOTE_NAMES[i]
            best_mode = 'minor'

    return CAMELOT_MAP.get((best_key, best_mode), '8A')


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

        # Key via CENS chroma + Temperley profiles
        result["camelot_key"] = detect_key(y, sr)

        # Energy — raw RMS; global min-max normalization runs after each scan
        result["energy"] = float(np.mean(librosa.feature.rms(y=y)))

    except Exception:
        result["bpm"] = None
        result["camelot_key"] = None
        result["energy"] = None
        result["duration_sec"] = None

    return result
