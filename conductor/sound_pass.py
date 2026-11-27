"""Per-section sound fingerprint and per-downbeat drum strength (CPU; run after ml/allin1 merge).

  seg["timbre"]        — 32 numbers: MFCC 1–12 mean/std, spectral contrast mean (7), chroma
                         energy spread (1): "what it sounds like", compared across songs
  data["downbeat_strength"] — drum-stem onset peak at each downbeat (0..~1): where the big beats are

    python -m conductor.sound_pass <folder> [<folder> ...]
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
SR = 22050
HOP = 512


def fingerprint(y: np.ndarray, a: float, b: float) -> list[float]:
    import librosa
    seg = y[int(a * SR): int(b * SR)]
    if len(seg) < SR:
        seg = y[int(a * SR): int(a * SR) + SR]
    mf = librosa.feature.mfcc(y=seg, sr=SR, hop_length=HOP, n_mfcc=13)[1:]
    con = librosa.feature.spectral_contrast(y=seg, sr=SR, hop_length=HOP)
    chroma = librosa.feature.chroma_stft(y=seg, sr=SR, hop_length=HOP)
    spread = float(np.mean(np.max(chroma, axis=0) - np.mean(chroma, axis=0)))
    v = np.concatenate([mf.mean(1), mf.std(1), con.mean(1), [spread]])
    return [round(float(x), 4) for x in v]


def work(path: str) -> str:
    import librosa
    from .analysis import cache_key
    from .ml_apply import load_ml
    c = os.path.join(CACHE, cache_key(path) + ".json")
    if not os.path.exists(c):
        return "missing " + os.path.basename(path)
    with open(c, encoding="utf-8") as f:
        d = json.load(f)
    y, _ = librosa.load(path, sr=SR, mono=True)
    for sd in d["segments"]:
        sd["timbre"] = fingerprint(y, sd["start"], sd["end"])
    ml = load_ml(path)
    if ml is not None and d.get("downbeats"):
        on = np.asarray(ml["drum_onset"], dtype=float)
        st = []
        for t in d["downbeats"]:
            i = int(t * 50)
            w = on[max(0, i - 2): i + 3]
            st.append(round(float(w.max()) if len(w) else 0.0, 3))
        # Not every big beat is a drum: a synth stab or orchestral hit makes the whole mix jump.
        from .ml_apply import _mix_db
        mix = _mix_db(ml["stems_db"])
        for k, t in enumerate(d["downbeats"]):
            i = int(t * 10)
            pre = mix[max(0, i - 3): i].mean() if i > 0 else mix[0]
            post = mix[i: i + 2].max() if i < len(mix) else pre
            st[k] = round(max(st[k], (post - pre) / 10.0), 3)     # +10 dB jump ≈ a strong drum hit
        d["downbeat_strength"] = st
    d["sound"] = 1
    with open(c, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return "ok"


def main(folders: list[str]) -> None:
    from .library import scan
    files = [p for fo in folders for p in scan(fo)]
    with Pool(10) as pool:
        res = list(pool.imap_unordered(work, files))
    print(len(res), "songs;", [r for r in res if r != "ok"][:5])


if __name__ == "__main__":
    main(sys.argv[1:] or [__import__("conductor.playlists", fromlist=["train_music"]).train_music()])
