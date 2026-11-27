"""Join measurements: describe any transition "leave song A at t_out → enter song B at t_in" with
numbers, so a model can learn which joins sound natural.

Per song we cache curves at 10 Hz (cache/joinfeat/<key>.npz):
  rms   loudness              low   bass energy (<150 Hz)
  flux  onset strength        high  crash/cymbal energy (>6 kHz)
Features describe the music just before the exit and just after the entry, and how they relate.
"""
from __future__ import annotations

import os

import numpy as np

FPS = 10
SR = 22050
_CURVES: dict = {}
ROLES = ("explore", "build", "bridge", "skirmish", "horde", "peak")


def curves(track, cache_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")) -> dict:
    if track.path in _CURVES:
        return _CURVES[track.path]
    from .analysis import cache_key
    p = os.path.join(cache_dir, "joinfeat", cache_key(track.path) + ".npz")
    if os.path.exists(p):
        d = dict(np.load(p))
    else:
        import librosa
        y, sr = librosa.load(track.path, sr=SR, mono=True)
        hop = SR // FPS
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
        f = librosa.fft_frequencies(sr=sr, n_fft=2048)
        db = lambda x: 10 * np.log10(x + 1e-10)
        d = {
            "rms": db(S.mean(0)),
            "low": db(S[f < 150].mean(0)),
            "high": db(S[f > 6000].mean(0)),
            "flux": librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr).astype(np.float32),
        }
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez_compressed(p, **d)
    # normalise per song so features compare across songs
    for k in ("rms", "low", "high"):
        v = d[k]
        d[k] = (v - np.percentile(v, 50)) / (np.percentile(v, 95) - np.percentile(v, 5) + 1e-6)
    d["flux"] = d["flux"] / (np.percentile(d["flux"], 95) + 1e-6)
    _CURVES[track.path] = d
    return d


def _win(c, a: float, b: float) -> np.ndarray:
    i, j = max(0, int(a * FPS)), max(0, int(b * FPS))
    seg = c[i:j]
    return seg if len(seg) else np.zeros(1)


def _period(track, t):
    from .tempo import period_at
    return period_at(track, t)


def _bar_offset(track, t) -> float:
    """Distance from t to the nearest bar line, in beats (0 = on a bar)."""
    bars = track.downbeats or track.beats[::4]
    if not bars:
        return 0.0
    b = min(bars, key=lambda x: abs(x - t))
    return abs(t - b) / max(1e-3, _period(track, t))


def _role(track, t):
    from .states import role_of
    return role_of(track.segment_at(t + 0.05)) or "skip"


_PC = {n: k for k, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def _fifths(A, B) -> float:
    """Steps apart on the circle of fifths (minor keys via their relative major): 0 = same/relative
    key, 1 = neighbours (mix well), 6 = as far apart as keys get."""
    def pos(t):
        pc = _PC.get(t.key)
        if pc is None:
            return None
        return ((pc + (3 if t.mode == "minor" else 0)) * 7) % 12
    a, b = pos(A), pos(B)
    if a is None or b is None:
        return float("nan")
    d = abs(a - b)
    return float(min(d, 12 - d))


def features(A, t_out: float, B, t_in: float) -> dict:
    """Numbers describing the join A@t_out → B@t_in."""
    ca, cb = curves(A), curves(B)
    f = {}
    for k in ("rms", "low", "high", "flux"):
        before, after = _win(ca[k], t_out - 4, t_out), _win(cb[k], t_in, t_in + 4)
        f[f"{k}_out"], f[f"{k}_in"] = float(before.mean()), float(after.mean())
        f[f"{k}_jump"] = f[f"{k}_in"] - f[f"{k}_out"]
    # what happens right at the seam: the last half second out, the first half second in
    f["rms_edge_out"] = float(_win(ca["rms"], t_out - 0.5, t_out).mean())
    f["rms_edge_in"] = float(_win(cb["rms"], t_in, t_in + 0.5).mean())
    f["crash_in"] = float(_win(cb["high"], t_in - 0.1, t_in + 0.4).max() - _win(cb["high"], t_in - 2, t_in - 0.2).mean()) \
        if t_in > 2 else float(_win(cb["high"], t_in, t_in + 0.4).max())
    f["hit_in"] = float(_win(cb["flux"], t_in - 0.1, t_in + 0.3).max())
    f["hit_out"] = float(_win(ca["flux"], t_out - 0.3, t_out + 0.1).max())
    # energy direction going in (a ramp down before the exit, a build before the entry)
    x = _win(ca["rms"], t_out - 8, t_out)
    f["slope_out"] = float(np.polyfit(np.arange(len(x)), x, 1)[0] * FPS) if len(x) > 4 else 0.0
    y = _win(cb["rms"], t_in, t_in + 8)
    f["slope_in"] = float(np.polyfit(np.arange(len(y)), y, 1)[0] * FPS) if len(y) > 4 else 0.0
    # tempo
    pa, pb = _period(A, t_out), _period(B, t_in)
    r = np.log2(pb / pa)
    r_fold = r - np.round(r)                       # half/double time counts as related
    f["tempo_ratio"], f["tempo_ratio_folded"] = float(abs(r)), float(abs(r_fold))
    f["bpm_out"], f["bpm_in"] = 60 / pa, 60 / pb
    # grid
    f["bar_out"], f["bar_in"] = _bar_offset(A, t_out), _bar_offset(B, t_in)
    # sound similarity (sound fingerprints of the two parts)
    sa, sb = A.segment_at(t_out - 0.05), B.segment_at(t_in + 0.05)
    f["sound_sim"] = float(sa.timbre @ sb.timbre) if sa.timbre is not None and sb.timbre is not None else 0.0
    f["key_same"] = float(A.key == B.key and A.mode == B.mode)
    f["key_fifths"] = _fifths(A, B)
    # roles (your Song Map)
    ro, ri = _role(A, t_out - 0.05), _role(B, t_in)
    for r_ in ROLES:
        f[f"out_{r_}"], f[f"in_{r_}"] = float(ro == r_), float(ri == r_)
    f["role_step"] = float((ROLES.index(ri) if ri in ROLES else 0) - (ROLES.index(ro) if ro in ROLES else 0))
    return f


# features a model may use (no "same song" leak, nothing that only real section changes have)
MODEL_FEATURES = None


def vector(f: dict) -> np.ndarray:
    global MODEL_FEATURES
    if MODEL_FEATURES is None:
        MODEL_FEATURES = sorted(f.keys())
    return np.array([f[k] for k in MODEL_FEATURES], dtype=np.float32)
