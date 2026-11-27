"""Stingers: short dramatic moments cut from a song, played over the music for game events
(player downed, player died). Only songs with a convincing moment get one.

A good stinger is an IMPACT that RINGS OUT: a big hit (drop landing, orchestral stab, slam)
whose sound then decays rather than continuing as a busy groove, so it works on its own.
Scored on: attack (loudness jump), weight (share of low end), darkness (low spectral
centroid) and tail (how long it rings before falling well away / how cleanly it decays).

    python -m conductor.stingers <folder>      → cache/stingers/<cache key>.json + clips
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import get_context
Pool = get_context("spawn").Pool   # 2026-11-24: fork hung on Linux (threads + fork)

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(CACHE, "stingers")
SR = 22050
HOP = 256
MAX_LEN = {"downed": 2.5, "died": 4.0}
MIN_SCORE = {"downed": 0.55, "died": 0.6}


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-6))


def candidates(y: np.ndarray, times: list[float]) -> list[dict]:
    import librosa
    fps = SR / HOP
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    db = _db(rms)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    low = S[freqs < 150].sum(0) / (S.sum(0) + 1e-9)
    cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    song_peak = np.percentile(db, 98)
    out = []
    for t in times:
        i = int(t * fps)
        if i < int(0.5 * fps) or i + int(4.5 * fps) >= len(db):
            continue
        pre = db[i - int(0.4 * fps): i - 1].mean()
        hit = db[i: i + int(0.15 * fps)].max()
        attack = hit - pre
        if attack < 5 or hit < song_peak - 9:
            continue
        # tail: how the loudness falls after the hit (impact rings out vs. groove carries on)
        seg = db[i: i + int(4.0 * fps)]
        rel = seg - hit
        fall_15 = -rel[int(1.5 * fps)]                       # dB fallen after 1.5 s
        busy = np.mean(np.abs(np.diff(seg[int(0.3 * fps):])))   # re-attacks = the groove carries on
        ring_len = next((k / fps for k in range(int(0.3 * fps), len(rel)) if rel[k] < -18), 4.0)
        weight = float(low[i: i + int(0.5 * fps)].mean())
        dark = float(np.clip(1 - (cent[i: i + int(1.0 * fps)].mean() - 800) / 3000, 0, 1))
        a = np.clip((attack - 5) / 12, 0, 1)
        decays = np.clip(fall_15 / 12, 0, 1) * np.clip(1 - (busy - 0.6) / 1.5, 0, 1)
        w = np.clip(weight / 0.35, 0, 1)
        down = 0.4 * a + 0.3 * decays + 0.2 * w + 0.1 * dark
        died = 0.3 * a + 0.3 * decays + 0.2 * dark + 0.2 * np.clip(ring_len / 3.5, 0, 1)
        out.append({"t": round(float(t), 3), "attack_db": round(float(attack), 1), "fall_db_1_5s": round(float(fall_15), 1),
                    "busy": round(float(busy), 2), "ring_s": round(float(ring_len), 2), "weight": round(weight, 3),
                    "dark": round(dark, 2), "downed": round(float(down), 3), "died": round(float(died), 3)})
    return out


def pick(cands: list[dict]) -> dict:
    """Best per event, at least 8 s apart from each other; nothing if it doesn't clear the bar."""
    chosen = {}
    for ev in ("died", "downed"):
        for c in sorted(cands, key=lambda c: -c[ev]):
            if c[ev] < MIN_SCORE[ev]:
                break
            if all(abs(c["t"] - o["t"]) >= 8 for o in chosen.values()):
                chosen[ev] = {**c, "score": c[ev], "len": min(MAX_LEN[ev], max(1.2, c["ring_s"] + 0.4))}
                break
    return chosen


def work(path: str) -> str:
    import librosa
    import soundfile as sf
    from .analysis import cache_key
    c = os.path.join(CACHE, cache_key(path) + ".json")
    if not os.path.exists(c):
        return "missing"
    d = json.load(open(c, encoding="utf-8"))
    y, _ = librosa.load(path, sr=SR, mono=True)
    times = sorted({h["t"] for h in d.get("hits", [])} | set(d.get("downbeats", [])))
    cands = candidates(y, times)
    chosen = pick(cands)
    key = cache_key(path)
    os.makedirs(OUT, exist_ok=True)
    if chosen:
        full, sr = librosa.load(path, sr=44100, mono=False)
        full = np.atleast_2d(full)
        for ev, s in chosen.items():
            a, n = int((s["t"] - 0.02) * sr), int(s["len"] * sr)
            clip = full[:, a:a + n].T.copy()
            fade = np.ones(len(clip)); f = min(len(clip), int(0.6 * sr))
            fade[-f:] = np.linspace(1, 0, f) ** 2
            fade[:int(0.005 * sr)] = np.linspace(0, 1, int(0.005 * sr))
            clip *= fade[:, None]
            s["file"] = os.path.join(OUT, f"{key}_{ev}.wav")
            sf.write(s["file"], clip, sr, subtype="PCM_16")
    json.dump({"path": path, "title": d["title"], "stingers": chosen, "n_candidates": len(cands)},
              open(os.path.join(OUT, key + ".json"), "w", encoding="utf-8"), indent=1)
    summary = ", ".join(f"{k} {v['score']:.2f} @{v['t']:.0f}s" for k, v in chosen.items()) or "none"
    return f"{d['title'][:40]}: {summary}"


def load(path: str) -> dict:
    from .analysis import cache_key
    p = os.path.join(OUT, cache_key(path) + ".json")
    return json.load(open(p, encoding="utf-8")).get("stingers", {}) if os.path.exists(p) else {}


if __name__ == "__main__":
    from .library import scan
    files = [p for f in (sys.argv[1:] or [__import__("conductor.playlists", fromlist=["train_music"]).train_music()]) for p in scan(f)]
    with Pool(8) as pool:
        for r in pool.imap_unordered(work, files):
            print(r, flush=True)
