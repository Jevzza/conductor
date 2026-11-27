"""GPU analysis pass (run with the ML venv: .venv-ml\\Scripts\\python -m conductor.ml_pass <music folder>).

Per song, writes cache/ml/<cache key>.json with:
  beats, downbeats      — Beat This! (CPJKU, 2024) beat and downbeat times: real bar lines
  stems_db              — Demucs htdemucs stem loudness (drums/bass/other/vocals), dB every 0.1 s
  drum_onset            — onset envelope of the drum stem, 50 per second (for "driving beat")

The core reads these if present and falls back to librosa-only analysis otherwise, so the
app itself never needs torch.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(HERE, "cache", "ml")
ML_VERSION = 1


def ml_path(path: str) -> str:
    from .analysis import cache_key
    return os.path.join(ML_DIR, cache_key(path) + ".ml.json")


def main(folder: str, force: bool = False) -> None:
    import librosa
    import torch
    from beat_this.inference import Audio2Beats
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from .library import scan

    os.makedirs(ML_DIR, exist_ok=True)
    a2b = Audio2Beats(checkpoint_path="final0", device="cuda", dbn=False)
    model = get_model("htdemucs").cuda().eval()
    sr = model.samplerate
    files = scan(folder)
    for i, path in enumerate(files, 1):
        out = ml_path(path)
        if os.path.exists(out) and not force:
            continue
        t0 = time.time()
        y, _ = librosa.load(path, sr=sr, mono=False)
        y = np.atleast_2d(y)
        if y.shape[0] == 1:
            y = np.vstack([y, y])
        beats, downs = a2b(y.T.copy(), sr)
        wav = torch.tensor(y[:2], dtype=torch.float32)[None].cuda()
        ref = wav.mean(1)
        mu, sd = ref.mean(), ref.std()
        with torch.no_grad():
            stems = apply_model(model, (wav - mu) / sd, device="cuda", split=True, overlap=0.25)[0]
        stems = stems * sd + mu
        try:                                    # share the split with All-In-One + stems_pass (one Demucs run)
            import soundfile as sf
            from .allin1_pass import demix_dir
            dd = demix_dir(path)
            os.makedirs(dd, exist_ok=True)
            st = stems.cpu().numpy()
            for name, s in zip(model.sources, st):
                sf.write(os.path.join(dd, name + ".wav"), np.clip(s.T, -1, 1), sr, subtype="PCM_16")
        except Exception as e:
            print("stems not shared:", e, flush=True)
        stems = stems.mean(1).cpu().numpy()      # (4, n) mono stems
        hop = sr // 10
        n = stems.shape[1] // hop
        stems_db = {}
        for name, s in zip(model.sources, stems):
            frames = s[: n * hop].reshape(n, hop)
            stems_db[name] = np.round(20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-6), 1).tolist()
        drums = librosa.resample(stems[model.sources.index("drums")], orig_sr=sr, target_sr=22050)
        onset = librosa.onset.onset_strength(y=drums, sr=22050, hop_length=441)   # 50 fps
        onset = onset / (np.percentile(onset, 99) + 1e-9)
        data = {"version": ML_VERSION, "path": path,
                "beats": np.round(beats, 4).tolist(), "downbeats": np.round(downs, 4).tolist(),
                "stems_db": stems_db, "drum_onset": np.round(onset, 3).tolist()}
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"[{i}/{len(files)}] {time.time() - t0:4.1f}s  {os.path.basename(path)}", flush=True)
        del wav, stems
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else __import__("conductor.playlists", fromlist=["train_music"]).train_music(), force="--force" in sys.argv)
