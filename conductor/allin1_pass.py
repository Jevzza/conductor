"""Section structure with All-In-One (Kim & Nam 2023): phrase-level boundaries and labels
(intro/verse/chorus/bridge/inst/solo/outro). Run with the ML venv:
    .venv-ml\\Scripts\\python -m conductor.allin1_pass <music folder>
Writes cache/allin1/json/<song>.json (All-In-One's own format); ml_apply picks it up.
On Windows the DiNAT attention uses a pure-PyTorch NATTEN stand-in (.venv-ml natten/)."""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(HERE, "cache", "allin1")


def json_path(path: str) -> str:
    return os.path.join(BASE, "json", os.path.splitext(os.path.basename(path))[0] + ".json")


def demix_dir(path: str) -> str:
    """Where All-In-One looks for a song's 4 Demucs stems. ml_pass writes them here so the song is split
    once, not three times (ml_pass, All-In-One, stems_pass - 2026-11-22: sections took ~35 s/song)."""
    return os.path.join(BASE, "demix", "htdemucs", os.path.splitext(os.path.basename(path))[0])


def main(folder: str) -> None:
    import allin1
    from .library import scan
    files = [p for p in scan(folder) if not os.path.exists(json_path(p))]
    for i, p in enumerate(files, 1):
        t = time.time()
        try:
            allin1.analyze(p, out_dir=os.path.join(BASE, "json"), demix_dir=os.path.join(BASE, "demix"),
                           spec_dir=os.path.join(BASE, "spec"), device="cuda", multiprocess=False,
                           keep_byproducts=True)          # stems_pass reuses the stems, then deletes them
            print(f"[{i}/{len(files)}] {time.time() - t:5.1f}s {os.path.basename(p)}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(files)}] FAIL {os.path.basename(p)}: {e!r}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else __import__("conductor.playlists", fromlist=["train_music"]).train_music())
