"""Song Map data: one playable clip + a compact sound fingerprint per section, plus Conductor's
current belief about its role. (ML venv for nothing — plain numpy/librosa; uses cache/roles.)

    python -m conductor.songmap <out_dir> <folder>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def belief(seg) -> str:
    """What the director currently treats this section as."""
    if seg.role == "combat":
        return "peak" if seg.energy >= 0.8 else "horde" if seg.energy >= 0.62 else "skirmish"
    if seg.role == "build" or seg.label == "build":
        return "build"
    return "explore"


def main(out_dir: str, folder: str):
    sys.path.insert(0, HERE)
    from conductor.analysis import cache_key
    from conductor.library import Library
    lib = Library.build(folder, os.path.join(HERE, "cache"), progress=lambda m: None, backfill=False, cached_only=True)
    os.makedirs(os.path.join(out_dir, "sc"), exist_ok=True)
    rows, embs = [], []
    for ti, t in enumerate(sorted(lib.tracks, key=lambda t: t.title.lower())):
        rj = os.path.join(HERE, "cache", "roles", cache_key(t.path) + ".json")
        rsec = {round(r["start"], 2): r for r in json.load(open(rj, encoding="utf-8"))["sections"]} if os.path.exists(rj) else {}
        for si, seg in enumerate(t.segments):
            r = rsec.get(round(seg.start, 2))
            if not r or not r.get("emb") or seg.length < 4:
                continue
            sid = f"s{ti:02d}_{si:02d}"
            mid = seg.start + max(0.0, seg.length / 2 - 4)
            dur = min(8.0, seg.length)
            mp3 = os.path.join(out_dir, "sc", sid + ".mp3")
            if not os.path.exists(mp3):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{mid:.2f}", "-t", f"{dur:.2f}", "-i", t.path,
                                "-ac", "1", "-b:a", "64k", "-af", "afade=t=in:d=0.2,afade=t=out:st=%.2f:d=0.4" % (dur - 0.4), mp3],
                               check=True)
            rows.append({"id": sid, "song": t.title, "si": si, "start": round(seg.start, 1), "end": round(seg.end, 1),
                         "len": round(seg.length, 1), "energy": round(seg.energy, 3), "label": seg.label,
                         "belief": belief(seg), "file": f"sc/{sid}.mp3"})
            embs.append(r["emb"])
    E = np.array(embs, dtype=np.float32)
    E -= E.mean(0)
    U, S, Vt = np.linalg.svd(E, full_matrices=False)          # PCA → 32-d fingerprint
    F = (E @ Vt[:32].T)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9
    for r, f in zip(rows, F):
        r["f"] = [round(float(x), 3) for x in f]
    json.dump(rows, open(os.path.join(out_dir, "sections.json"), "w", encoding="utf-8"))
    print(len(rows), "sections from", len(lib.tracks), "songs")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else __import__("conductor.playlists", fromlist=["train_music"]).train_music())
