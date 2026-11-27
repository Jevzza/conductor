"""Find seamless loop regions inside song sections.

For a section we want [a, b) such that jumping from b back to a is inaudible: the music
just after b must match the music just after a, in rhythm (onset envelope, ~6 ms resolution)
and in harmony (chroma). The loop length is a whole number of bars from the tempo estimate,
then fine-tuned by searching ±3 % for the best self-similarity, so it doesn't depend on every
detected beat being exact.
"""
from __future__ import annotations

import numpy as np

SR = 22050
HOP = 128                      # ~5.8 ms resolution for seam alignment
BAR_CHOICES = (16, 12, 8, 4, 2)  # whole phrases of the song's own bar length
MIN_LOOP_S = 6.0
MAX_LOOP_S = 40.0
MIN_RHYTHM = 0.7               # onset patterns must line up across the seam…
MIN_HARMONY = 0.45             # …and so must the chords, or the jump is audible
TEMPO_RATIOS = (1.0, 0.5, 2.0)  # half/double-time readings keep loops whole-bar


def _ncc(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def find_loops(y: np.ndarray, tempo: float, beats: list[float], segments: list[dict]) -> list[dict | None]:
    import librosa
    fps = SR / HOP
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    chroma = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP * 8)   # ~46 ms
    cfps = fps / 8
    n = len(onset)
    beats = np.asarray(beats)
    if tempo <= 0:
        return [None] * len(segments)
    win = 2 * int(2.0 * fps)                     # compare ~4 s around each seam (even length)
    cw = max(1, int(2.0 * cfps))                 # chroma: 2 s each side
    out = []
    for seg in segments:
        s, e = seg["start"], seg["end"]
        best = None                              # (key, info)
        for ratio in TEMPO_RATIOS:
            bar = 4 * 60.0 / (tempo * ratio)
            inside = beats[(beats >= s) & (beats < s + bar)]
            if len(inside) == 0:
                continue
            fr = (inside * fps).astype(int)
            a = int(fr[np.argmax(onset[np.clip(fr, 0, n - 1)])])   # likely downbeat
            ref = onset[max(0, a - win // 2): a + win // 2]
            if len(ref) < win:
                continue
            ca = int(a / 8)
            for bars in BAR_CHOICES:
                L0 = bars * bar
                if L0 < MIN_LOOP_S or L0 > MAX_LOOP_S or a / fps + L0 > e + 0.5 * bar:
                    continue
                lo, hi = int(L0 * 0.97 * fps), int(L0 * 1.03 * fps)
                r_best, b_best = -1.0, None
                for L in range(lo, hi + 1):
                    b = a + L
                    if b + win // 2 >= n:
                        break
                    r = _ncc(ref, onset[b - win // 2: b + win // 2])
                    if r > r_best:
                        r_best, b_best = r, b
                if b_best is None:
                    continue
                cb = int(b_best / 8)
                h = (_ncc(chroma[:, max(0, ca - cw): ca + cw].ravel(),
                          chroma[:, max(0, cb - cw): cb + cw].ravel())
                     if ca - cw >= 0 and cb + cw < chroma.shape[1] else 0.0)
                ok = r_best >= MIN_RHYTHM and h >= MIN_HARMONY
                info = {"start": round(a / fps, 4), "end": round(b_best / fps, 4), "bars": bars,
                        "bpm": round(tempo * ratio, 2), "rhythm": round(r_best, 3),
                        "harmony": round(h, 3), "ok": bool(ok)}
                # prefer: seamless first, then longer phrases, then closer match
                key = (ok, L0 if ok else 0, r_best + h)
                if best is None or key > best[0]:
                    best = (key, info)
        out.append(best[1] if best else None)
    return out
