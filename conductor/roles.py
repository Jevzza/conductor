"""What is each part of each song FOR? (ML venv, GPU)

Zero-shot with CLAP (laion/larger_clap_music): every section is compared with text descriptions
of the gameplay roles. Writes cache/roles/<key>.json: per-section role probabilities.

    .venv-ml\\Scripts\\python -m conductor.roles <folder>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "cache", "roles")
ROLES = {
    "explore": ["calm dark ambient atmospheric music", "quiet eerie soundscape with no beat",
                "slow mellow background music for exploring", "soft minimal ambient synth pads"],
    "build": ["tense music slowly building up, rising tension, drums not yet in",
              "suspenseful build-up before a drop", "ominous rising synths, anticipation",
              "a slow brooding groove that is building"],
    "skirmish": ["steady driving electronic beat with moderate energy",
                 "mid-tempo dark techno groove, focused and restrained",
                 "rhythmic action music, not too intense", "pulsing bassline and steady drums"],
    "horde": ["intense fast aggressive action music with heavy drums",
              "relentless hard-hitting industrial techno", "frantic high energy combat music",
              "heavy distorted bass and pounding drums"],
    "peak": ["massive epic climax, the hardest most intense part of the song",
             "huge aggressive drop for a boss battle", "overwhelming wall of sound, maximum intensity",
             "brutal climactic heavy drop"],
}
ORDER = list(ROLES)


def main(folders):
    import librosa
    import torch
    from .analysis import cache_key
    from .critic import Clap
    from .library import scan
    os.makedirs(OUT, exist_ok=True)
    clap = Clap()
    with torch.no_grad():
        tin = clap.p(text=[t for r in ORDER for t in ROLES[r]], return_tensors="pt", padding=True)
        tin = {k: v.cuda() for k, v in tin.items()}
        te = clap.m.get_text_features(**tin)
        te = getattr(te, "pooler_output", te)
        te = torch.nn.functional.normalize(te, dim=-1).cpu().numpy()
    per = len(ROLES["explore"])
    for path in [p for f in folders for p in scan(f)]:
        key = cache_key(path)
        out = os.path.join(OUT, key + ".json")
        d = json.load(open(os.path.join(HERE, "cache", key + ".json"), encoding="utf-8"))
        y, sr = librosa.load(path, sr=22050, mono=True)
        rows = []
        for sd in d["segments"]:
            a, b = sd["start"], sd["end"]
            # up to 3 ten-second windows spread over the section
            starts = [a + f * max(0.0, (b - a) - 10) for f in ((0.5,) if b - a < 20 else (0.15, 0.5, 0.85))]
            wins = []
            for s0 in starts:
                w = y[int(s0 * sr): int((s0 + min(10, b - a)) * sr)]
                if len(w) < sr:
                    continue
                wins.append(np.pad(w, (0, max(0, 10 * sr - len(w))))[: 10 * sr].astype(np.float32))
            if not wins:
                rows.append({"start": a, "end": b, "probs": None, "emb": None})
                continue
            from .critic import SR as CSR
            e = clap.embed(wins).astype(np.float32).mean(0)
            e = e / (np.linalg.norm(e) + 1e-9)
            sims = te @ e                                     # (roles*per,)
            role_sim = sims.reshape(len(ORDER), per).mean(1)
            p = np.exp(role_sim * 40); p /= p.sum()            # CLAP logit scale ≈ temperature
            rows.append({"start": a, "end": b, "probs": {r: round(float(x), 3) for r, x in zip(ORDER, p)},
                         "emb": [round(float(x), 4) for x in e]})
        json.dump({"path": path, "title": d["title"], "sections": rows}, open(out, "w", encoding="utf-8"))
        top = [max(r["probs"], key=r["probs"].get) if r["probs"] else "?" for r in rows]
        print(f"{d['title'][:38]:<38} {' '.join(t[:4] for t in top)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or [__import__("conductor.playlists", fromlist=["train_music"]).train_music()])
