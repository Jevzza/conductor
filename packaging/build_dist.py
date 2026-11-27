r"""Build the drop-in mod folder (build/conductor) and its zip from this repo.

    python packaging/build_dist.py [pack]

Needs build/conductor/runtime/ first (see packaging/README-build.md): the Windows embeddable Python 3.13
with pip in runtime/python, ffmpeg.exe + ffprobe.exe in runtime/bin, and the wheels pip can't fetch
(madmom, beat-this) in runtime/wheels. Regenerates app/conductor (made portable), app/natten (NATTEN
shim), app/seed (trained models, if you have them in cache/) and scripts/ (the mod Lua).
"""
import glob
import os
import re
import shutil

DEV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_MOD = os.path.join(DEV, "mod")
DIST = os.path.join(DEV, "build")
PACK = os.path.join(DEV, "packaging")
OUT = os.path.join(DIST, "conductor")
EXTRA = os.path.join(PACK, "src_extra")
DEV_ONLY = {"moments.py", "obs_record.py", "lab_watch.py", "review_server.py", "live_replay.py",
            "mission_render.py", "generations.py"}
SEED = ["taste_model.pkl", "join_model.pkl", os.path.join("live", "policy.json"),
        os.path.join("live", "model_styles.json")]

H = '__import__("conductor._home", fromlist=["HOME"]).HOME'


def rep(s, a, b, name, count=1):
    n = s.count(a)
    assert n == count, f"{name}: expected {count} of {a[:70]!r}, found {n}"
    return s.replace(a, b)


def build_app():
    pkg = os.path.join(OUT, "app", "conductor")
    if os.path.isdir(pkg):
        shutil.rmtree(pkg)
    os.makedirs(pkg)
    for f in glob.glob(os.path.join(DEV, "conductor", "*.py")):
        if os.path.basename(f) not in DEV_ONLY:
            shutil.copy2(f, pkg)
    for f in ("_home.py", "boot.py"):
        shutil.copy2(os.path.join(EXTRA, f), pkg)
    for f in glob.glob(os.path.join(pkg, "*.py")):
        name = os.path.basename(f)
        if name in ("_home.py", "boot.py"):
            continue
        s = open(f, encoding="utf-8").read()
        s = s.replace("os.path.dirname(os.path.dirname(os.path.abspath(__file__)))", H)
        s = s.replace('os.path.join(HERE, ".venv-ml", "Scripts", "python.exe")', "sys.executable")
        s = s.replace("os.path.join(" + H + ', ".venv-ml", "Scripts", "python.exe")', "sys.executable")
        if name == "addlist.py":
            # the game's mods path is ~100 chars; "all artists - title" overflowed Windows' 260-char limit
            # (os.stat failed on a 4-artist GTA VI track in testing). Main artist + capped filename.
            s = rep(s, '"{list-name}", "{artists} - {title}.{output-ext}")]',
                    '"{list-name}", "{artist} - {title}.{output-ext}"),\n                             "--max-filename-length", "80"]', name)
            # spotDL keeps its config + temp downloads in ~/.spotdl: keep them inside the mod folder
            s = rep(s, '''                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    info = {''', '''                            env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                 "USERPROFILE": os.path.join(HERE, "spotdl_home")})
    info = {''', name)
            s = rep(s, '''    os.makedirs(DATA, exist_ok=True)
''', '''    os.makedirs(DATA, exist_ok=True)
    os.makedirs(os.path.join(HERE, "spotdl_home"), exist_ok=True)
''', name)
            s = rep(s, '    lock = os.path.join(HERE, "cache", "ingest.lock")',
                    '    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)\n    lock = os.path.join(HERE, "cache", "ingest.lock")', name)
        if name == "playlists.py":          # data/playlists (HERE = data/ in the drop-in)
            assert '_load().get("playlists_dir") or os.path.join(HERE, "playlists")' in s, "playlists DATA default"
        if name == "app.py":
            assert 'r"D:' not in s, "app.py: music default should be empty"
            s = rep(s, '''    if not lib.tracks:
        print(f"No playable audio found in {music}")
        return
''', open(os.path.join(EXTRA, "app_idle_call.py.txt"), encoding="utf-8").read(), name)
            s = rep(s, "def _taper(slider: float) -> float:",
                    open(os.path.join(EXTRA, "app_idle_defs.py.txt"), encoding="utf-8").read() + "def _taper(slider: float) -> float:", name)
        left = [l for l in s.splitlines() if re.search(r'r"[A-Z]:\\(programs|Music)', l) and not l.lstrip().startswith(("#", "main(", "files =", "lib =", "FOLDERS"))]
        open(f, "w", encoding="utf-8").write(s)
        if left and name not in ("ab.py", "changelog.py", "critic.py", "distill.py", "joinmodel.py", "checks.py"):
            print("  note:", name, "still has", left[:2])
    nat = os.path.join(OUT, "app", "natten")
    if os.path.isdir(nat):
        shutil.rmtree(nat)
    shutil.copytree(os.path.join(PACK, "natten"), nat,
                    ignore=shutil.ignore_patterns("__pycache__"))
    seed = os.path.join(OUT, "app", "seed", "cache")
    for rel in SEED:
        if not os.path.exists(os.path.join(DEV, "cache", rel)):
            print("  no trained", rel, "- the build ships without it (the app starts untrained)")
            continue
        os.makedirs(os.path.dirname(os.path.join(seed, rel)), exist_ok=True)
        shutil.copy2(os.path.join(DEV, "cache", rel), os.path.join(seed, rel))
    prof = os.path.join(OUT, "app", "seed", "profiles")
    os.makedirs(prof, exist_ok=True)
    shutil.copy2(os.path.join(DEV, "profiles", "darktide.json"), prof)


def build_linux():
    """linux/launch.sh (Steam launch option) + runtime/requirements-linux.lock (the Windows lock minus
    torch/madmom, which boot installs per graphics card / from wheels-linux)."""
    os.makedirs(os.path.join(OUT, "linux"), exist_ok=True)
    sh = open(os.path.join(EXTRA, "linux", "launch.sh"), encoding="utf-8").read().replace("\r\n", "\n")
    open(os.path.join(OUT, "linux", "launch.sh"), "w", encoding="utf-8", newline="\n").write(sh)
    lock = open(os.path.join(OUT, "runtime", "requirements.lock"), encoding="utf-8").read().splitlines()
    keep = [l for l in lock if l.strip() and not l.startswith(("#", "--", "torch==", "torchaudio==", "madmom=="))]
    open(os.path.join(OUT, "runtime", "requirements-linux.lock"), "w", encoding="utf-8", newline="\n").write(
        "# Linux: boot.py installs torch (per graphics card) and madmom separately\n" + "\n".join(keep) + "\n")
    wl = os.path.join(EXTRA, "linux", "wheels")
    if os.path.isdir(wl):
        shutil.copytree(wl, os.path.join(OUT, "runtime", "wheels-linux"), dirs_exist_ok=True)


def build_mod():
    """mod/ is already path-free: copy it as it is."""
    shutil.copy2(os.path.join(LIVE_MOD, "conductor.mod"), OUT)
    dst = os.path.join(OUT, "scripts")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(LIVE_MOD, "scripts"), dst)
    rt = os.path.join(OUT, "runtime")
    os.makedirs(rt, exist_ok=True)
    for f in ("requirements.lock", "requirements.top.txt"):
        shutil.copy2(os.path.join(PACK, "runtime", f), rt)


STANDALONE_CMD = r"""@echo off
rem Conductor standalone - no game needed (tabletop / D&D): a page with Calm / Tension / Fight / Horde / Boss.
rem Leave this window open while you play; close it to stop. Add --lan to control it from a phone or tablet.
cd /d "%~dp0"
title Conductor standalone
start "" http://127.0.0.1:8770
"runtime\python\python.exe" -m conductor.boot standalone %*
if errorlevel 1 pause
"""


def build_standalone():
    open(os.path.join(OUT, "Conductor Standalone.cmd"), "w", encoding="utf-8", newline="\r\n").write(STANDALONE_CMD)


def pack(zip_path=os.path.join(DIST, "conductor.zip")):
    """The shippable zip: the mod folder WITHOUT installed packages (first launch installs them),
    without data/ contents (the user's), without caches. Unzips to a 'conductor' folder."""
    import time
    import zipfile
    site = os.path.join(OUT, "runtime", "python", "Lib", "site-packages")
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, dirs, files in os.walk(OUT):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            rel_root = os.path.relpath(root, OUT)
            if rel_root == "data" or rel_root.startswith("data" + os.sep):
                dirs[:] = []
                continue
            if root.startswith(site):
                top = os.path.relpath(root, site).split(os.sep)[0]
                if top != "." and not (top == "pip" or (top.startswith("pip-") and top.endswith(".dist-info"))):
                    dirs[:] = []
                    continue
                if top == ".":
                    dirs[:] = [d for d in dirs if d == "pip" or (d.startswith("pip-") and d.endswith(".dist-info"))]
                    files = []
            if root == os.path.join(OUT, "runtime", "python", "Scripts"):
                files = [f for f in files if f.lower().startswith("pip")]
            for f in files:
                p = os.path.join(root, f)
                arc = os.path.join("conductor", os.path.relpath(p, OUT)).replace(os.sep, "/")
                if f.endswith(".sh"):                     # keep it executable when unzipped on Linux
                    zi = zipfile.ZipInfo(arc, date_time=time.localtime(os.path.getmtime(p))[:6])
                    zi.create_system, zi.external_attr, zi.compress_type = 3, (0o100755 << 16), zipfile.ZIP_DEFLATED
                    z.writestr(zi, open(p, "rb").read())
                else:
                    z.write(p, arc)
                n += 1
        z.write(os.path.join(PACK, "README.txt"), "conductor/README.txt")
        z.write(os.path.join(DEV, "INSTALL.md"), "conductor/INSTALL.md")
        z.writestr("conductor/data/README.txt", "Your settings, playlists and logs live here. Safe to keep when updating Conductor.\n")
    print(f"packed {n} files -> {zip_path} ({os.path.getsize(zip_path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    import sys
    build_app()
    build_mod()
    build_linux()
    build_standalone()
    print("built", OUT)
    if "pack" in sys.argv[1:]:
        pack()
