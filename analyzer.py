import hashlib
import json
import logging
import os
import threading
import time

import numpy as np

NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
ENHARMONIC = {
    "C#": "Db",
    "D#": "Eb",
    "G#": "Ab",
    "A#": "Bb",
    "Fb": "E",
    "Cb": "B",
    "E#": "F",
    "B#": "C",
}

CAMELOT_MAP = {
    ("C", "major"): "8B",
    ("G", "major"): "9B",
    ("D", "major"): "10B",
    ("A", "major"): "11B",
    ("E", "major"): "12B",
    ("B", "major"): "1B",
    ("F#", "major"): "2B",
    ("Db", "major"): "3B",
    ("Ab", "major"): "4B",
    ("Eb", "major"): "5B",
    ("Bb", "major"): "6B",
    ("F", "major"): "7B",
    ("A", "minor"): "8A",
    ("E", "minor"): "9A",
    ("B", "minor"): "10A",
    ("F#", "minor"): "11A",
    ("Db", "minor"): "12A",
    ("Ab", "minor"): "1A",
    ("Eb", "minor"): "2A",
    ("Bb", "minor"): "3A",
    ("F", "minor"): "4A",
    ("C", "minor"): "5A",
    ("G", "minor"): "6A",
    ("D", "minor"): "7A",
}

_MAJOR_PROFILE = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
_MINOR_PROFILE = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]

LOGGER = logging.getLogger(__name__)
MODEL_HASHES_PATH = os.path.join(os.path.dirname(__file__), "model_hashes.json")
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_MODEL_FILES = [
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

_clap_model = None
_clap_processor = None
_clap_device = None
_clap_loading = False
_clap_load_error = None
_clap_model_lock = threading.Lock()
_clap_model_ready = threading.Event()


def detect_key_chroma(y, sr) -> str:
    """Fallback key detection using CENS chroma + Temperley profiles."""
    import librosa

    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    best_corr = -np.inf
    best_key = "C"
    best_mode = "major"

    for index in range(12):
        rotated = np.roll(chroma_mean, -index)
        major_corr = np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1]
        minor_corr = np.corrcoef(rotated, _MINOR_PROFILE)[0, 1]
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = NOTE_NAMES[index]
            best_mode = "major"
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = NOTE_NAMES[index]
            best_mode = "minor"

    return CAMELOT_MAP.get((best_key, best_mode), "8A")


def detect_key(y, sr) -> str:
    try:
        import essentia.standard as es

        audio = y.astype("float32")
        key_extractor = es.KeyExtractor(
            profileType="temperley",
            usePolyphony=True,
            useThreeChords=True,
        )
        key, scale, _strength = key_extractor(audio)
        key = ENHARMONIC.get(key, key)
        camelot = CAMELOT_MAP.get((key, scale))
        if camelot:
            return camelot
    except ImportError:
        pass
    except Exception:
        pass

    return detect_key_chroma(y, sr)


def compute_energy(y, sr) -> float:
    import librosa

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    contrast_score = float(np.mean(contrast))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_score = float(np.mean(onset_env))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff_score = float(np.mean(rolloff)) / (sr / 2)
    return (onset_score * 0.5) + (contrast_score * 0.3) + (rolloff_score * 0.2)


def compute_embedding(y, sr) -> list:
    import librosa

    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    mean = np.mean(chroma, axis=1)
    std = np.std(chroma, axis=1)
    vector = np.concatenate([mean, std]).tolist()
    return vector


def _load_clap_hash_metadata() -> dict:
    with open(MODEL_HASHES_PATH, encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _sha256_file(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _ensure_clap_snapshot() -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    metadata = _load_clap_hash_metadata()[CLAP_MODEL_ID]
    revision = metadata["revision"]

    try:
        snapshot_dir = snapshot_download(
            repo_id=CLAP_MODEL_ID,
            revision=revision,
            allow_patterns=CLAP_MODEL_FILES,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        snapshot_dir = snapshot_download(
            repo_id=CLAP_MODEL_ID,
            revision=revision,
            allow_patterns=CLAP_MODEL_FILES,
            local_files_only=False,
        )

    model_path = os.path.join(snapshot_dir, "pytorch_model.bin")
    actual_hash = _sha256_file(model_path)
    expected_hash = metadata["files"]["pytorch_model.bin"]
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"CLAP model hash mismatch for {CLAP_MODEL_ID}: expected {expected_hash}, got {actual_hash}"
        )

    return snapshot_dir


def load_clap_model():
    global _clap_model, _clap_processor, _clap_device, _clap_loading, _clap_load_error

    if _clap_model is not None:
        return _clap_model

    should_wait = False
    with _clap_model_lock:
        if _clap_model is not None:
            return _clap_model
        if _clap_loading:
            should_wait = True
        else:
            _clap_loading = True
            _clap_load_error = None
            _clap_model_ready.clear()

    if should_wait:
        _clap_model_ready.wait()
        if _clap_model is None:
            raise RuntimeError(_clap_load_error or "CLAP model failed to load")
        return _clap_model

    try:
        import torch
        from transformers import AutoModel, AutoProcessor

        snapshot_dir = _ensure_clap_snapshot()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            processor = AutoProcessor.from_pretrained(snapshot_dir, local_files_only=True)
            model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True)
            model = model.to(device)
            model.eval()

        with _clap_model_lock:
            _clap_model = model
            _clap_processor = processor
            _clap_device = device
            _clap_load_error = None
        LOGGER.info("CLAP model ready on %s", device)
        return _clap_model
    except Exception as exc:
        LOGGER.exception("Failed to load CLAP model")
        with _clap_model_lock:
            _clap_load_error = str(exc)
        raise
    finally:
        with _clap_model_lock:
            _clap_loading = False
        _clap_model_ready.set()


def get_clap_status() -> dict:
    return {
        "loaded": _clap_model is not None,
        "loading": _clap_loading,
        "error": _clap_load_error,
    }


def _get_clap_components():
    model = load_clap_model()
    return model, _clap_processor, _clap_device


def _get_track_duration_seconds(file_path: str) -> float:
    import librosa
    import mutagen

    try:
        audio_file = mutagen.File(file_path)
        if audio_file and hasattr(audio_file, "info") and getattr(audio_file.info, "length", None):
            return float(audio_file.info.length)
    except Exception:
        pass

    try:
        return float(librosa.get_duration(path=file_path))
    except Exception:
        return 0.0


def generate_clap_text_embedding(query: str) -> np.ndarray:
    import torch

    model, processor, device = _get_clap_components()
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        features = model.get_text_features(**inputs)

    return features[0].detach().cpu().numpy().astype(np.float32)


def generate_clap_embedding(file_path: str) -> list[float]:
    import librosa
    import torch

    model, processor, device = _get_clap_components()
    sample_rate = int(getattr(processor.feature_extractor, "sampling_rate", 48_000))
    clip_duration = float(getattr(processor.feature_extractor, "max_length_s", 10) or 10)
    total_duration = _get_track_duration_seconds(file_path)
    offset = max(0.0, (total_duration / 2.0) - (clip_duration / 2.0))
    audio, sr = librosa.load(
        file_path,
        sr=sample_rate,
        mono=True,
        offset=offset,
        duration=clip_duration,
    )
    if audio.size == 0:
        raise ValueError("Loaded empty audio segment for CLAP embedding")

    inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        features = model.get_audio_features(**inputs)

    embedding = features[0].detach().cpu().numpy().astype(np.float32)
    return embedding.tolist()


def analyze_file(file_path: str) -> dict:
    import librosa
    import mutagen

    result = {
        "file_path": file_path,
        "filename": os.path.basename(file_path),
        "file_date": os.path.getmtime(file_path),
        "source_folder": os.path.dirname(file_path),
        "analyzed_at": time.time(),
        "clap_embedding": None,
        "waveform_data": None,
    }

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

    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=60)

        try:
            full = mutagen.File(file_path)
            if full and hasattr(full, "info"):
                result["duration_sec"] = float(full.info.length)
            else:
                result["duration_sec"] = float(len(y) / sr)
        except Exception:
            result["duration_sec"] = float(len(y) / sr)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        result["bpm"] = round(float(tempo), 1)
        result["camelot_key"] = detect_key(y, sr)
        result["energy"] = compute_energy(y, sr)
    except Exception:
        result["bpm"] = None
        result["camelot_key"] = None
        result["energy"] = None
        result["duration_sec"] = None

    try:
        result["clap_embedding"] = generate_clap_embedding(file_path)
    except Exception:
        LOGGER.exception("CLAP embedding failed for %s", file_path)
        result["clap_embedding"] = None

    return result
