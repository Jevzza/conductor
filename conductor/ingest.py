"""Get a playlist folder ready for Conductor (run in the background; heavy, uses the GPU):

    python -m conductor ingest --music "<music folder>" [--spotify <playlist url>]

  1. optional: download the Spotify playlist into the folder with spotDL
  2. librosa analysis of every song (parallel)
  3. GPU pass: Beat This! beats/downbeats + Demucs stems     (ML venv)
  4. All-In-One phrase sections                               (ML venv)
  5. merge everything and re-derive loops / hits / jump points (parallel)
  6. register the folder as a playlist in config.json and refresh playlists.txt

Steps skip songs that are already done, so it can be re-run any time (e.g. after adding songs).
A lock file stops two ingests running at once.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from multiprocessing import get_context
Pool = get_context("spawn").Pool   # 2026-11-24: fork hung on Linux (threads + fork)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(HERE, ".venv-ml", "Scripts", "python.exe")
LOCK = os.path.join(HERE, "cache", "ingest.lock")
CACHE = os.path.join(HERE, "cache")


def _analyze(path: str) -> str:
    from .analysis import load_or_analyze
    try:
        load_or_analyze(path, CACHE, backfill=True)
        return "ok"
    except Exception as e:
        return f"FAIL {os.path.basename(path)}: {e!r}"


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def _eta(sec: float) -> str:
    return f"~{sec / 60:.0f} min left" if sec >= 90 else f"~{max(5, sec):.0f} s left"


SUBSTEP = {"Demixing": "splitting stems", "Extracting spectrograms": "spectrograms", "Analyzing": "finding sections"}


def _gpu_pass(module: str, folder: str, label: str, n_files: int, report) -> None:
    """Run a GPU pass in the ML venv, streaming its per-song lines into the in-game bar
    (song k/n, current sub-step, ETA). report(fraction_of_this_pass, text)."""
    import re
    t0, done = time.time(), 0
    report(0.0, f"{label}: loading the model")
    p = subprocess.Popen([VENV_PY, "-W", "ignore", "-m", module, folder], cwd=HERE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    tail = []                          # the pass's other output: shown in the log if it fails
    for line in p.stdout:              # text mode turns tqdm's carriage-return updates into lines too
        line = line.rstrip()
        if not line:
            continue
        if not line.startswith("["):
            tail = (tail + [line])[-25:]
        m = re.match(r"\[(\d+)(?:/(\d+))?\]\s+(?:[\d.]+s|FAIL)\s*(.*)", line)
        if m:
            print(line, flush=True)
            done = int(m.group(1))
            n = int(m.group(2) or n_files) or 1
            report(done / n, f"{label} {done}/{n} - {_eta((time.time() - t0) / done * (n - done))}")
            continue
        m = re.search(r"(Demixing|Extracting spectrograms|Analyzing)", line)
        if m:
            n = n_files or 1
            eta = f" - {_eta((time.time() - t0) / done * (n - done))}" if done else ""
            report(done / n, f"{label} {min(done + 1, n)}/{n}: {SUBSTEP[m.group(1)]}{eta}")
    p.wait()
    if p.returncode != 0 or (n_files and done == 0):
        print(f"{label}: {module} exited {p.returncode} after {done} songs. Its last output:", flush=True)
        for line in tail:
            print("   ", line, flush=True)


def ingest(folder: str, spotify: str | None = None, name: str | None = None, procs: int = 8, progress=None) -> None:
    prog = progress or (lambda *a: None)      # (stage, k, n, text) - for the in-game bar (addlist.py)
    from .library import scan
    from .playlists import register, write_index
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
    except FileExistsError:
        _log(f"another ingest is running ({LOCK}); delete it if that's not true")
        return
    try:
        os.makedirs(folder, exist_ok=True)
        if spotify:
            _log(f"downloading {spotify} → {folder}")
            subprocess.run([sys.executable, "-m", "spotdl", "download", spotify, "--output",
                            os.path.join(folder, "{artists} - {title}.{output-ext}")], check=False)
        files = scan(folder)
        _log(f"{len(files)} songs in {folder}")
        register(name or os.path.basename(os.path.normpath(folder)), folder)
        _log("step 1/4: analysis")
        with Pool(procs) as pool:
            for i, r in enumerate(pool.imap_unordered(_analyze, files), 1):
                prog("analysis", i, len(files), f"analysing {i}/{len(files)}")
                if r != "ok":
                    _log(r)
        for i, (step, mod) in enumerate((("2/4: beats + stems (GPU)", "conductor.ml_pass"), ("3/4: sections (GPU)", "conductor.allin1_pass"))):
            _log("step " + step)
            lab = ("beats + stems (GPU)", "song sections (GPU)")[i]
            _gpu_pass(mod, folder, lab, len(files), lambda f, text, i=i: prog("gpu", i + f, 3, text))
        _log("step 4/4: merge + loops/hits/jump points + sound fingerprints")
        with Pool(procs) as pool:
            for i, r in enumerate(pool.imap_unordered(_analyze, files), 1):
                prog("merge", i, 2 * len(files), f"jump points {i}/{len(files)}")
                if r != "ok":
                    _log(r)
        from .sound_pass import main as sound_main
        sound_main([folder])
        _log("stems + call-outs (GPU)")
        _gpu_pass("conductor.stems_pass", folder, "call-outs (GPU)", len(files),
                  lambda f, text: prog("merge", len(files) * (1 + f), 2 * len(files), text))
        from .stingers import work as sting_work
        for i, f in enumerate(files, 1):
            prog("merge", len(files) + i, 2 * len(files), f"stingers {i}/{len(files)}")
            sting_work(f)
        _log("tempo maps")             # else the app computes them on its main loop at switch time (hung it)
        try:
            from .library import Library
            from . import tempo
            lib = Library.build(folder, CACHE, progress=lambda m: None, backfill=False, cached_only=True,
                                tempo_map=False)
            tempo.warm(lib, CACHE, procs, lambda i, n: prog("merge", 2 * n, 2 * n, f"tempo maps {i}/{n}"))
        except Exception as e:
            _log(f"tempo maps: {e!r}")
        write_index()
        _log("done")
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass
