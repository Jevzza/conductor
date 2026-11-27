"""Live critic server (ML venv): the app sends candidate changes, this renders and LISTENS to each one
(critic + style taste layer, ~68% agreement with your A/B picks vs ~63% for the trees) and answers.

    .venv-ml\\Scripts\\python -m conductor.critic_live      (started by the app: livecritic.py)

stdin  (one JSON per line): {"id": n, "music": "<playlist folder>", "routes": [route, ...]}
stdout (one JSON per line): {"ready": true} once, then {"id": n, "scores": [float | null, ...], "ms": ...}
Models stay loaded (cold start ~11 s, ~0.8 GB VRAM); each change ~0.7 s to render + ~0.3 s to hear.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np


def main() -> None:
    import librosa
    import torch
    from . import critic, routes
    from .distill_trees import TEACHER, _opt_vec
    from .library import Library
    tw = json.load(open(TEACHER, encoding="utf-8"))
    w = np.asarray(tw["w"], np.float64)
    model = critic.make_model().cuda()
    model.load_state_dict(torch.load(os.path.join(critic.OUT, "critic.pt")))
    model.eval()
    clap = critic.Clap()
    n = int(critic.HALF * critic.SR)
    libs: dict = {}
    print(json.dumps({"ready": True, "teacher": tw.get("built")}), flush=True)

    def hear(by, r) -> float:
        clip, seams = routes.render_audio(by, r)
        y = librosa.to_mono(np.ascontiguousarray(clip.T))
        y = librosa.resample(y, orig_sr=routes.SR, target_sr=critic.SR)
        waves = []
        for s in seams or [routes.PRE]:
            seg = y[max(0, int(s * critic.SR) - n): int(s * critic.SR) + n]
            waves.append(np.pad(seg, (0, max(0, 2 * n - len(seg))))[:2 * n].astype(np.float32))
        waves = np.stack(waves)
        mel = torch.tensor(critic.mel_batch(waves)).float().cuda()
        cf = torch.tensor(critic.clap_feats(clap, waves)).float().cuda()
        with torch.no_grad():
            F = model.features(mel, cf).cpu().numpy()
            P = torch.sigmoid(model(mel, cf)).cpu().numpy()
        rows = [np.concatenate([F[i], [P[i], np.log(P[i] + 1e-4)]]) for i in range(len(F))]
        vec = np.concatenate([np.mean(rows, 0), np.min(rows, 0)[-2:]])
        return float(np.r_[vec, _opt_vec(routes.as_route(r))] @ w)

    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        t0 = time.time()
        try:
            music = req["music"]
            if music not in libs:
                libs.clear()                       # one playlist at a time (memory)
                lib = Library.build(music, os.path.join(critic.HERE, "cache"), progress=lambda m: None,
                                    backfill=False, cached_only=True)
                libs[music] = {t.title: t for t in lib.tracks}
            by = libs[music]
            scores = []
            for r in req["routes"]:
                try:
                    scores.append(hear(by, r))
                except Exception as e:             # one unplayable route shouldn't sink the batch
                    print(json.dumps({"warn": f"{type(e).__name__}: {e}"}), flush=True)
                    scores.append(None)
            print(json.dumps({"id": req.get("id"), "scores": scores, "ms": int(1000 * (time.time() - t0))}), flush=True)
        except Exception as e:
            print(json.dumps({"id": req.get("id"), "error": f"{type(e).__name__}: {e}"}), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
