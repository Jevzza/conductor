"""Groove fingerprint per section (user 2026-11-25: "the beat needs to flow into the next song").

From the drum stem, on the fitted beat grid: where in the bar the low hits (kick) and the high hits
(snare/hats) fall, as a 16-step pattern each, plus how busy the drums are and the tempo. Two sections
"groove alike" when their patterns line up and their tempos meet (direct or half/double time).

    python -m conductor.groove <title substring>     # print one song's sections
Cache: cache/groove/<key>.json
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "cache", "groove")
SR = 11025
VERSION = 1
_MEM: dict = {}


def _bands(path: str):
    """Onset envelopes (low, high) of the drum stem at 100 fps."""
    import librosa
    from scipy.signal import butter, sosfiltfilt
    y, _ = librosa.load(path, sr=SR, mono=True)
    lo = sosfiltfilt(butter(4, 150 / (SR / 2), output="sos"), y)
    hi = sosfiltfilt(butter(4, 1800 / (SR / 2), "high", output="sos"), y)
    hop = SR // 100
    env = []
    for x in (lo, hi):
        e = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same")[::hop] + 1e-12)
        e = np.log(e + 1e-4)
        d = np.maximum(0, np.diff(e, prepend=e[0]))         # rises = hits
        env.append(d)
    return env, float(np.sqrt(np.mean(y ** 2)) + 1e-9)


def sections(track) -> list[dict]:
    if track.path in _MEM:
        return _MEM[track.path]
    from .analysis import cache_key
    from . import grid
    os.makedirs(OUT, exist_ok=True)
    cp = os.path.join(OUT, cache_key(track.path) + ".json")
    try:
        d = json.load(open(cp, encoding="utf-8"))
        if d.get("v") == VERSION:
            _MEM[track.path] = d["sections"]
            return d["sections"]
    except (OSError, ValueError):
        pass
    stem = os.path.join(HERE, "cache", "stems", cache_key(track.path) + "_drums.flac")
    out = []
    if os.path.exists(stem):
        (lo, hi), rms = _bands(stem)
        n = len(lo)
        for s in track.segments:
            f = grid.fit_at(track, s.start + 0.05)
            a, b = s.start, s.start + s.length
            row = {"start": a, "end": b, "period": f["period"] if f else None}
            if f and b - a >= 4 * f["period"] * 2:
                p = f["period"]
                k0 = int(np.ceil((a - f["t0"]) / (4 * p)))
                k1 = int(np.floor((b - f["t0"]) / (4 * p)))
                if k1 - k0 >= 2:
                    pat = np.zeros((2, 16))
                    for k in range(k0, k1):
                        t = f["t0"] + 4 * p * k + np.arange(16) * p / 4
                        idx = np.clip(np.round(t * 100).astype(int), 0, n - 1)
                        for j, e in enumerate((lo, hi)):
                            # max over +-20 ms so a slightly late hit still counts on its step
                            w = np.stack([e[np.clip(idx + o, 0, n - 1)] for o in (-2, -1, 0, 1, 2)]).max(0)
                            pat[j] += w
                    pat /= (k1 - k0)
                    seg_lo = lo[int(a * 100):int(b * 100)]
                    seg_hi = hi[int(a * 100):int(b * 100)]
                    row.update(kick=[round(float(v), 4) for v in pat[0]], top=[round(float(v), 4) for v in pat[1]],
                               busy=round(float(seg_lo.mean() + seg_hi.mean()), 4),
                               # how much of the drum energy sits ON the 16th grid (vs between): a steady beat
                               on_grid=round(float(pat.sum() / (pat.sum() + 1e-6 + 16 * (seg_lo.mean() + seg_hi.mean()))), 4))
            out.append(row)
    try:
        json.dump({"v": VERSION, "sections": out}, open(cp, "w", encoding="utf-8"))
    except OSError:
        pass
    _MEM[track.path] = out
    return out


def at(track, t: float) -> dict | None:
    for s in sections(track):
        if s["start"] - 0.1 <= t < s["end"]:
            return s
    return None


def _norm(v):
    v = np.asarray(v, float)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


def _pattern_sim(x: dict, y: dict) -> float:
    """Correlation of the bar patterns. Half/double time: compare against y's pattern at the other speed."""
    best = -1.0
    for warp in ("same", "half", "double"):
        s = 0.0
        for key in ("kick", "top"):
            a, b = np.asarray(x[key]), np.asarray(y[key])
            if warp == "half":                 # y at half tempo: its bar = 2 of x's; y step i = x 16th 2i
                a, b = a[::2], (b[:8] + b[8:]) / 2
            elif warp == "double":
                a, b = (a[:8] + a[8:]) / 2, b[::2]
            s += float(_norm(a) @ _norm(b))
        best = max(best, s / 2)
    return best


def tempo_off(x: dict, y: dict) -> float | None:
    """How far apart the beats are after half/double time, in octaves (0.085 ~ 6 %)."""
    if not x.get("period") or not y.get("period"):
        return None
    r = abs(np.log2(x["period"] / y["period"]))
    return float(abs(r - round(r)))


def match(A, ta: float, B, tb: float) -> dict:
    """How well A's groove at ta flows into B's at tb."""
    x, y = at(A, ta), at(B, tb)
    out = {"pattern": None, "tempo_off": None, "busy_ratio": None}
    if not x or not y:
        return out
    out["tempo_off"] = tempo_off(x, y)
    if "kick" in x and "kick" in y:
        out["pattern"] = _pattern_sim(x, y)
        out["busy_ratio"] = float(min(x["busy"], y["busy"]) / max(x["busy"], y["busy"], 1e-6))
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    from conductor.ab import training_library
    q = " ".join(sys.argv[1:]).lower()
    for T in training_library().tracks:
        if q in T.title.lower():
            print(T.title)
            for s in sections(T):
                print(f"  {s['start']:6.1f}-{s['end']:6.1f}  bpm {60 / s['period'] if s.get('period') else 0:5.1f}  busy {s.get('busy', '-')}",
                      "kick", "".join("X" if v > 0.6 * max(s["kick"]) else "." for v in s["kick"]) if "kick" in s else "")
            break
