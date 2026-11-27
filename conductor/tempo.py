"""Tempo map: a beat grid that follows tempo changes inside a song.

The original grid assumed one tempo per song (plus a beat tracker that sometimes jumps to double
or half time). Here the beat comes from the predominant local pulse (PLP), which follows tempo
drifts and changes through the song. It is kept within ±40 % of the song's main tempo so it can't
flip octave (136 ↔ 273); a half-time section keeps the same pulse, so cuts still land on its beat.
Bar lines: the neural tracker's downbeats, moved onto the nearest pulse beat; where it found none
(breakdowns), bars continue every 4 beats.

Cached per song in cache/tempo/<key>.json (computing takes ~1-2 s a song).
"""
from __future__ import annotations

import bisect
import json
import os

import numpy as np

HOP = 256
SR = 22050


def _compute(path: str, main_bpm: float, downbeats: list[float]) -> dict:
    import librosa
    y, sr = librosa.load(path, sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    bpm = float(main_bpm) if main_bpm and main_bpm > 0 else 120.0
    while bpm > 180:
        bpm /= 2
    while bpm < 70:
        bpm *= 2
    pulse = librosa.beat.plp(onset_envelope=env, sr=sr, hop_length=HOP,
                             tempo_min=0.7 * bpm, tempo_max=1.4 * bpm, win_length=512)
    pk = np.flatnonzero(librosa.util.localmax(pulse))
    pk = pk[pulse[pk] > 0.05 * pulse.max()]
    beats = librosa.frames_to_time(pk, sr=sr, hop_length=HOP).tolist()
    # fill holes in the pulse (silence) by continuing the local beat length
    out = []
    for a, b in zip(beats, beats[1:]):
        out.append(a)
        loc = np.median(np.diff(out[-9:])) if len(out) > 2 else (b - a)
        if b - a > 1.6 * loc:
            n = int(round((b - a) / loc))
            out += [a + (b - a) * k / n for k in range(1, n)]
    if beats:
        out.append(beats[-1])
    beats = out
    # bar lines: the neural downbeats, moved onto the pulse; continue every 4 beats where there are none
    bars, arr = [], np.asarray(beats)
    if len(arr):
        snapped = sorted({int(np.argmin(np.abs(arr - d))) for d in downbeats
                          if len(arr) and abs(arr[np.argmin(np.abs(arr - d))] - d) < 0.15})
        if snapped:
            idx = [snapped[0]]
            for s in snapped[1:] + [None]:
                target = s if s is not None else len(arr)
                while target - idx[-1] > 6:             # no bar line found for a while: every 4 beats
                    idx.append(idx[-1] + 4)
                if s is not None and s - idx[-1] >= 2:
                    idx.append(s)
            k = idx[0]
            while k - 4 >= 0:
                idx.insert(0, k - 4)
                k -= 4
            bars = [float(arr[i]) for i in idx if i < len(arr)]
        else:
            bars = [float(arr[i]) for i in range(0, len(arr), 4)]
    return {"beats": [round(b, 4) for b in beats], "bars": [round(b, 4) for b in bars], "bpm": bpm}


def _onset_env(path: str):
    import librosa
    y, sr = librosa.load(path, sr=SR, mono=True)
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)


def _fit(beats: list[float], env, a: float, b: float) -> float:
    """How well these beats sit on the drums in [a, b): on-beat vs midpoint-offbeat onset strength."""
    x = np.asarray([t for t in beats if a <= t < b])
    if len(x) < 6:
        return 0.0
    fr = lambda ts: np.clip(np.round(np.asarray(ts) * SR / HOP).astype(int), 0, len(env) - 1)
    sc = lambda ts: float(np.mean([env[max(0, i - 2):i + 3].max() for i in fr(ts)]))
    return sc(x[:-1]) / max(1e-6, sc(x[:-1] + np.diff(x) / 2))


def _hybrid(track, orig_beats, orig_bars, plp, env) -> dict:
    """Per section, keep whichever grid fits the drums better (tempo changes live in sections)."""
    beats, bars, picks = [], [], []
    for seg in track.segments:
        a, b = seg.start, seg.end
        fo, fp = _fit(orig_beats, env, a, b), _fit(plp["beats"], env, a, b)
        use_plp = fp > fo + 0.05
        picks.append("plp" if use_plp else "orig")
        src_b, src_d = (plp["beats"], plp["bars"]) if use_plp else (orig_beats, orig_bars)
        beats += [t for t in src_b if a <= t < b]
        bars += [t for t in src_d if a <= t < b]
    return {"beats": sorted(beats), "bars": sorted(bars), "picks": picks}


def apply_all(library, cache_dir: str, progress=lambda m: None) -> int:
    from .analysis import cache_key
    d = os.path.join(cache_dir, "tempo")
    os.makedirs(d, exist_ok=True)
    n = 0
    for t in library.tracks:
        p = os.path.join(d, cache_key(t.path) + ".json")
        data = None
        if os.path.exists(p):
            try:
                data = json.load(open(p, encoding="utf-8"))
            except (OSError, ValueError):
                data = None
        if data is None or "picks" not in data:
            try:
                progress(f"tempo map: {t.title}")
                plp = data if data and "beats" in data else _compute(t.path, t.tempo, list(getattr(t, "downbeats", []) or []))
                stem = next((x for x in (os.path.join(cache_dir, "stems", cache_key(t.path) + "_rhythm" + e)
                                         for e in (".flac", ".wav")) if os.path.exists(x)), None)
                env = _onset_env(stem or t.path)       # judge against the drums+bass stem when we have it
                data = _hybrid(t, list(t.beats), list(t.downbeats or t.beats[::4]), plp, env)
                data["plp"] = {"beats": plp["beats"], "bars": plp["bars"]}
                json.dump(data, open(p, "w", encoding="utf-8"))
            except Exception as e:                     # keep the old grid rather than fail
                progress(f"tempo map failed for {t.title}: {e}")
                continue
        if len(data.get("beats", [])) > 16 and len(data.get("bars", [])) > 4:
            t.beats, t.downbeats = data["beats"], data["bars"]
            t.tempo_map = True
            n += 1
    return n


def _warm_one(args) -> str:
    track, cache_dir = args
    from types import SimpleNamespace
    try:
        apply_all(SimpleNamespace(tracks=[track]), cache_dir)
        return "ok"
    except Exception as e:
        return f"FAIL {track.title}: {e!r}"


def warm(library, cache_dir: str, procs: int = 8, progress=lambda i, n: None) -> None:
    """Fill the tempo cache for a whole library in parallel (ingest does this, so the app's
    playlist switch never computes it - 40 songs took 98 s serially on the app's main loop)."""
    from multiprocessing import get_context
    Pool = get_context("spawn").Pool   # 2026-11-24: fork hung on Linux (threads + fork)
    todo = [t for t in library.tracks]
    with Pool(procs) as pool:
        for i, _ in enumerate(pool.imap_unordered(_warm_one, [(t, cache_dir) for t in todo]), 1):
            progress(i, len(todo))


def period_at(track, t: float) -> float:
    """Seconds per beat around time t (from the local beats)."""
    b = track.beats
    if not b or len(b) < 3:
        return track._period()
    i = min(max(1, bisect.bisect_right(b, t)), len(b) - 1)
    lo, hi = max(1, i - 4), min(len(b), i + 4)
    return float(np.median(np.diff(b[lo - 1:hi])))
