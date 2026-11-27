"""Jump points inside a song: pairs of bars that sound alike, so playback can leave at the end
of one bar and carry on from the matching place elsewhere in the song (the "infinite jukebox"
idea). Each bar is described by harmony (chroma), timbre (MFCC), rhythm (onset pattern across
the bar) and loudness. A jump from the end of bar i to the start of bar j+1 is scored on how
alike bar i is to bar j (what you just heard) AND bar i+1 to bar j+1 (what you're about to hear).

Scores run 0..1. Strong matches (≳0.85) are near-seamless with a very short blend; weaker ones
still land on a bar line and get a longer crossfade, so they're usable as transitions too.
"""
from __future__ import annotations

import numpy as np

SR = 22050
HOP = 512
BEATS_PER_BAR = 4
MIN_GAP_BARS = 4          # don't "jump" to a neighbouring bar
PER_BAR = 6               # keep this many destinations per source bar
COARSE = 12               # bar-feature shortlist per source bar, re-scored at the seam
MIN_SCORE = 0.45
# Seam test (same as loops.py, whose thresholds were checked by ear): onset envelope and chroma
# ±2 s around the leave point vs the landing point, landing point fine-aligned ±50 ms.
SEAM_HOP = 128
SEAM_RHYTHM = 0.7
SEAM_HARMONY = 0.45
ALIGN_S = 0.05
W = {"harmony": 0.35, "timbre": 0.25, "rhythm": 0.25, "loud": 0.15}
RHYTHM_BINS = 16


def _cos(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    m = m / np.maximum(n, 1e-9)
    return m @ m.T


def bar_similarity(y: np.ndarray, beats: list[float], downbeats: list[float] | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(bar start times, n×n similarity matrix) for the song's bars."""
    import librosa
    bars = np.asarray(downbeats if downbeats else beats[::BEATS_PER_BAR], dtype=float)
    if len(bars) < MIN_GAP_BARS + 3:
        return bars, np.zeros((0, 0))
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)[1:]
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    nfr = min(chroma.shape[1], mfcc.shape[1], len(onset), len(rms))
    fr = np.clip((bars * SR / HOP).astype(int), 0, nfr - 1)
    H, T, R, L = [], [], [], []
    for a, b in zip(fr[:-1], fr[1:]):
        b = max(b, a + 2)
        H.append(chroma[:, a:b].mean(axis=1))
        T.append(mfcc[:, a:b].mean(axis=1))
        seg = onset[a:b]
        R.append(np.interp(np.linspace(0, len(seg) - 1, RHYTHM_BINS), np.arange(len(seg)), seg))
        L.append(20 * np.log10(np.mean(rms[a:b]) + 1e-6))
    H, T, R, L = map(np.asarray, (H, T, R, L))
    harm = np.clip((_cos(H) - 0.6) / 0.4, 0, 1)
    Tz = (T - T.mean(axis=0)) / (T.std(axis=0) + 1e-9)
    timbre = np.clip(_cos(Tz), 0, 1)
    Rz = R - R.mean(axis=1, keepdims=True)
    rhythm = np.clip(_cos(Rz), 0, 1)
    loud = np.exp(-np.abs(L[:, None] - L[None, :]) / 4.0)
    sim = W["harmony"] * harm + W["timbre"] * timbre + W["rhythm"] * rhythm + W["loud"] * loud
    return bars, sim


def _ncc(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def seam_score(r: float, h: float) -> float:
    """0..1. ≥0.85 means it passes the seamless-loop test; below scales with the weaker of the two."""
    if r >= SEAM_RHYTHM and h >= SEAM_HARMONY:
        x = ((r - SEAM_RHYTHM) / (1 - SEAM_RHYTHM) + (h - SEAM_HARMONY) / (1 - SEAM_HARMONY)) / 2
        return 0.85 + 0.15 * min(1.0, max(0.0, x))
    return max(0.0, 0.85 * min(r / SEAM_RHYTHM, h / SEAM_HARMONY, 1.0))


def find_jumps(y: np.ndarray, beats: list[float], downbeats: list[float] | None = None) -> list[list[float]]:
    """[[from_t, to_t, score], ...]: leave at from_t (a bar line), continue at to_t."""
    import librosa
    bars, sim = bar_similarity(y, beats, downbeats)
    n = len(sim)
    if n == 0:
        return []
    fps = SR / SEAM_HOP
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=SEAM_HOP)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=SEAM_HOP * 8)
    win = 2 * int(2.0 * fps)
    cw = max(1, int(2.0 * fps / 8))
    al = int(ALIGN_S * fps)
    N = len(onset)
    out = []
    for i in range(n - 1):
        cands = []
        for j in range(n - 1):
            if abs(i - j) < MIN_GAP_BARS:
                continue
            cands.append((0.5 * sim[i, j] + 0.5 * sim[i + 1, j + 1], j))
        a = int(bars[i + 1] * fps)
        if a - win // 2 < 0 or a + win // 2 >= N:
            continue
        ref = onset[a - win // 2: a + win // 2]
        ca = a // 8
        if ca - cw < 0 or ca + cw >= chroma.shape[1]:
            continue
        cref = chroma[:, ca - cw: ca + cw].ravel()
        scored = []
        for _, j in sorted(cands, reverse=True)[:COARSE]:
            b0 = int(bars[j + 1] * fps)
            r_best, b_best = -1.0, None
            for b in range(b0 - al, b0 + al + 1):
                if b - win // 2 < 0 or b + win // 2 >= N:
                    continue
                r = _ncc(ref, onset[b - win // 2: b + win // 2])
                if r > r_best:
                    r_best, b_best = r, b
            if b_best is None:
                continue
            cb = b_best // 8
            if cb - cw < 0 or cb + cw >= chroma.shape[1]:
                continue
            h = _ncc(cref, chroma[:, cb - cw: cb + cw].ravel())
            s = seam_score(r_best, h)
            if s >= MIN_SCORE:
                scored.append((s, b_best / fps))
        for s, to in sorted(scored, reverse=True)[:PER_BAR]:
            out.append([round(float(bars[i + 1]), 4), round(float(to), 4), round(float(s), 3)])
    return out
