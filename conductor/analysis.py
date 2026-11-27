"""Audio analysis: what is this song doing, second by second?

For every track we compute:
  * tempo + beat grid      → so switches land on the beat / bar
  * a per-second feature curve (loudness, rhythmic density, brightness)
  * structural segments    → intro / verse / chorus-like blocks from timbre+harmony novelty
  * key & mode             → a coarse "dark vs bright" mood cue

Raw features are stored per track. Turning them into a 0..1 *energy* scale happens in
library.py, normalised across your whole library, so "energy 0.9" means "among the most
intense moments of YOUR music", not some absolute standard.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

ANALYSIS_VERSION = 2
SR = 22050
HOP = 512

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def cache_key(path: str) -> str:
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{ANALYSIS_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _estimate_key(chroma: np.ndarray) -> tuple[str, str, float]:
    profile = chroma.mean(axis=1)
    best = ("C", "major", -1.0)
    for i in range(12):
        for mode, tmpl in (("major", _MAJOR), ("minor", _MINOR)):
            r = float(np.corrcoef(profile, np.roll(tmpl, i))[0, 1])
            if r > best[2]:
                best = (_KEYS[i], mode, r)
    return best


def _per_second(values: np.ndarray, frames_per_sec: float, n_sec: int) -> list[float]:
    out = []
    for s in range(n_sec):
        a, b = int(s * frames_per_sec), int((s + 1) * frames_per_sec)
        chunk = values[a:b]
        out.append(float(chunk.mean()) if chunk.size else float(values[-1]))
    return out


def _pulse_curve(onset_env: np.ndarray, sr: int, n_sec: int) -> list[float]:
    """Beat clarity per second, 0..1: how strongly the rhythm repeats at a danceable period
    (60–200 bpm), from the onset autocorrelation (tempogram). A driving four-on-the-floor or
    breakbeat scores high; ambient pads, rubato and drones score low, even when loud."""
    import librosa
    fps = sr / HOP
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP, win_length=384)
    lo, hi = int(60 * fps / 200), int(60 * fps / 60) + 1
    clarity = tg[lo:hi].max(axis=0)
    # scale by how much rhythmic energy is there at all, so near-silence can't look "clear"
    strength = np.clip(onset_env / (np.percentile(onset_env, 95) + 1e-6), 0, 1)
    return [round(v, 4) for v in _per_second(clarity * np.sqrt(strength), fps, n_sec)]


def add_pulse(data: dict) -> dict:
    """Backfill the pulse curve into an older cached analysis (cheaper than a full re-analysis)."""
    import librosa
    y, sr = librosa.load(data["path"], sr=SR, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    data["curve"]["pulse"] = _pulse_curve(onset_env, sr, len(data["curve"]["rms_db"]))
    return data


def add_loops(data: dict, y: np.ndarray | None = None) -> dict:
    """Find seamless loop regions for every section (see loops.py)."""
    import librosa
    from .loops import find_loops
    if y is None:
        y, _ = librosa.load(data["path"], sr=SR, mono=True)
    data["loops"] = find_loops(y, data["tempo"], data["beats"], data["segments"])
    return data


def add_hits(data: dict, y: np.ndarray | None = None) -> dict:
    """Impacts and cut-outs usable as abrupt transition points (see hits.py)."""
    import librosa
    from .hits import find_hits
    if y is None:
        y, _ = librosa.load(data["path"], sr=SR, mono=True)
    data.update(find_hits(y, data["beats"]))
    return data


def analyze(path: str) -> dict:
    import librosa  # heavy import, only when analysing

    y, sr = librosa.load(path, sr=SR, mono=True)
    duration = len(y) / sr
    n_sec = max(1, int(duration))
    fps = sr / HOP

    # Rhythm
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)

    # Loudness / brightness curves
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-5), ref=1.0)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]

    # Harmony / timbre for structure and key
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    key, mode, key_conf = _estimate_key(chroma)

    # Structural segmentation on beat-synchronous features
    segments_t: list[float] = [0.0]
    if len(beat_frames) > 16:
        feat = np.vstack([
            librosa.util.normalize(librosa.util.sync(chroma, beat_frames), axis=1),
            librosa.util.normalize(librosa.util.sync(mfcc, beat_frames), axis=1),
        ])
        k = int(np.clip(round(duration / 25.0), 3, 16))
        k = min(k, feat.shape[1] - 1)
        bounds = librosa.segment.agglomerative(feat, k)
        bound_frames = librosa.util.fix_frames(beat_frames[np.clip(bounds, 0, len(beat_frames) - 1)], x_min=0)
        segments_t = sorted(set(float(t) for t in librosa.frames_to_time(bound_frames, sr=sr, hop_length=HOP)))
    # Clean boundaries: start at 0, drop anything within MIN_SEG of the previous boundary
    # or of the end, so every segment is long enough to be a musical section.
    MIN_SEG = 6.0
    clean = [0.0]
    for t in segments_t:
        if t - clean[-1] >= MIN_SEG and duration - t >= MIN_SEG:
            clean.append(t)
    segments = [
        {"start": round(a, 3), "end": round(b, 3)}
        for a, b in zip(clean, clean[1:] + [duration])
    ]

    return {
        "version": ANALYSIS_VERSION,
        "path": os.path.abspath(path),
        "title": os.path.splitext(os.path.basename(path))[0],
        "duration": round(duration, 3),
        "tempo": round(tempo, 2),
        "beats": [round(float(t), 3) for t in beat_times],
        "key": key,
        "mode": mode,
        "key_confidence": round(key_conf, 3),
        "curve": {
            "rms_db": [round(v, 2) for v in _per_second(rms_db, fps, n_sec)],
            "onset": [round(v, 4) for v in _per_second(onset_env, fps, n_sec)],
            "centroid": [round(v, 1) for v in _per_second(centroid, fps, n_sec)],
            "pulse": _pulse_curve(onset_env, sr, n_sec),
        },
        "segments": segments,
    }


def add_jumps(data: dict, y: np.ndarray | None = None) -> dict:
    """Bar-to-bar jump points inside the song (see jumps.py)."""
    import librosa
    from .jumps import find_jumps
    if y is None:
        y, _ = librosa.load(data["path"], sr=SR, mono=True)
    data["jumps"] = find_jumps(y, data["beats"], data.get("downbeats"))
    return data


class NotAnalysed(Exception):
    """cached_only: this song hasn't been through `conductor ingest` yet."""


def load_or_analyze(path: str, cache_dir: str, force: bool = False, backfill: bool = True,
                    cached_only: bool = False) -> dict:
    """Cached analysis for `path`. With backfill=False, an existing cache entry is returned
    as-is even if newer features (pulse/loops/hits) are missing — fast startup for the app."""
    os.makedirs(cache_dir, exist_ok=True)
    cpath = os.path.join(cache_dir, cache_key(path) + ".json")
    if cached_only and not os.path.exists(cpath):
        raise NotAnalysed(path)
    if not force and os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            data = json.load(f)
        if cached_only and not data.get("ml"):
            raise NotAnalysed(path)
        from . import ml_apply
        if backfill and data.get("ml") != ml_apply.wanted_tag(path):
            ml = ml_apply.load_ml(path)
            if ml is not None:            # GPU pass is available: redo everything that depends on beats
                import librosa
                data = ml_apply.apply(data, ml)
                y, _ = librosa.load(data["path"], sr=SR, mono=True)
                data = add_jumps(add_hits(add_loops(data, y), y), y)
                with open(cpath, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return data
        if not backfill or ("pulse" in data["curve"] and "loops" in data and "hits" in data
                            and "jumps" in data):
            return data
        if "pulse" not in data["curve"]:
            data = add_pulse(data)
        if "loops" not in data:
            data = add_loops(data)
        if "hits" not in data:
            data = add_hits(data)
        if "jumps" not in data:
            data = add_jumps(data)
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
    data = add_jumps(add_hits(add_loops(analyze(path))))
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data
