"""Where a song is singing (user 2026-11-26: "if we've identified any sort of melodies or anything as
vocals, we should never blend vocals on top of each other").

From the Demucs vocal stem loudness (cache/ml/<key>.ml.json stems_db, 10 per second): singing = the
vocal stem within 6 dB of the whole mix. Gaps between words (< 0.6 s) are bridged and blips (< 0.4 s)
dropped, so a phrase reads as one stretch. Choirs count - they clash just the same.
"""
from __future__ import annotations

import json

import numpy as np

HZ = 10
_MEM: dict = {}


def curve(track) -> np.ndarray | None:
    """0/1 singing at 10 per second, or None (no GPU analysis for this song)."""
    if track.path in _MEM:
        return _MEM[track.path]
    from .ml_pass import ml_path
    out = None
    try:
        s = json.load(open(ml_path(track.path), encoding="utf-8"))["stems_db"]
        v = np.asarray(s["vocals"], float)
        mix = 10 * np.log10(sum(10 ** (np.asarray(x, float) / 10) for x in s.values()) + 1e-12)
        on = (v > mix - 6) & (v > -50)
        out = _clean(on, close=int(0.6 * HZ), min_len=int(0.4 * HZ)).astype(np.float32)
    except (OSError, ValueError, KeyError):
        pass
    _MEM[track.path] = out
    if len(_MEM) > 64:
        _MEM.pop(next(iter(_MEM)))
    return out


def _runs(b: np.ndarray):
    d = np.diff(np.concatenate([[0], b.astype(int), [0]]))
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def _clean(on: np.ndarray, close: int, min_len: int) -> np.ndarray:
    on = on.copy()
    for a, b in _runs(~on):                       # bridge short gaps between words
        if a > 0 and b < len(on) and b - a < close:
            on[a:b] = True
    for a, b in _runs(on):                        # drop blips
        if b - a < min_len:
            on[a:b] = False
    return on


def at(track, times: np.ndarray) -> np.ndarray:
    """Singing (0/1) at these times in the song (seconds); zeros where unknown."""
    c = curve(track)
    t = np.asarray(times, float)
    if c is None or not len(c):
        return np.zeros(len(t), np.float32)
    i = np.clip(np.round(t * HZ).astype(int), 0, len(c) - 1)
    out = c[i]
    out[(t < 0) | (t > len(c) / HZ)] = 0
    return out


def frac(track, t0: float, t1: float) -> float:
    """Share of [t0, t1] that is sung."""
    if t1 <= t0:
        return 0.0
    return float(at(track, np.arange(t0, t1, 1 / HZ)).mean())


def sung(track, t0: float, length: float, share: float = 0.15) -> bool:
    return frac(track, t0, t0 + length) >= share


def whole_phrase(track, t0: float, length: float) -> bool:
    """user 2026-11-26: "taking a couple of words out and not playing the full bar is really jarring".
    A cut of [t0, t0+length] is fine if it's (nearly) instrumental, or if every sung phrase in it starts
    and ends INSIDE it - never chopped at either edge."""
    c = at(track, np.arange(t0, t0 + length, 1 / HZ))
    if not len(c) or c.mean() < 0.08:
        return True
    edge = max(1, int(0.25 * HZ))
    return not c[:edge].any() and not c[-edge:].any()
