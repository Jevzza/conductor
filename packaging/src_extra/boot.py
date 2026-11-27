"""Drop-in start-up: the mod launches this (runtime\\python\\python.exe -m conductor.boot, hidden window).

  1. points everything at the mod folder (data/, runtime/bin ffmpeg, HF models in data/hf)
  2. first launch (or a new requirements.lock): installs the Python packages into runtime/python,
     with progress in the game's on-screen bar (the same file "Add playlist" uses)
  3. runs the app (python -m conductor run --background)
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time

from ._home import HOME, MOD, PORTABLE, RUNTIME

LINUX = os.name != "nt"
# Linux (2026-11-24): runtime/linux/ holds uv + its own Python 3.13 + venv (made by linux/launch.sh);
# packages come from requirements-linux.lock, PyTorch from the index that matches the graphics card
LOCK_FILE = (os.path.join(RUNTIME, "requirements-linux.lock" if LINUX else "requirements.lock")) if RUNTIME else None
TORCH = "2.11.0"
STAMP = os.path.join(HOME, ".setup_ok")
PROGRESS = os.path.join(os.environ.get("TEMP", "/tmp"), "conductor_addlist.txt")   # the mod's HUD bar reads this
NOWIN = 0x08000000 if os.name == "nt" else 0
EXPECTED = 130          # packages pip collects on a clean install (torch + deps + ML passes), for the bar


def _env() -> None:
    os.makedirs(HOME, exist_ok=True)
    os.environ.setdefault("CONDUCTOR_HOME", HOME)
    os.environ.setdefault("HF_HOME", os.path.join(HOME, "hf"))
    os.environ.setdefault("TORCH_HOME", os.path.join(HOME, "torch"))
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if RUNTIME and LINUX:
        os.environ["PATH"] = os.pathsep.join([os.path.dirname(sys.executable), os.environ.get("PATH", "")])
    elif RUNTIME:
        extra = [os.path.join(RUNTIME, "bin"), os.path.join(RUNTIME, "python"), os.path.join(RUNTIME, "python", "Scripts")]
        os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])


def _report(frac: float, text: str, state: str = "running") -> None:
    try:
        tmp = PROGRESS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(f"{max(0.0, min(1.0, frac)):.3f}|{text.replace('|', '/')}|{state}|\n")
        os.replace(tmp, PROGRESS)
    except OSError:
        pass
    print(time.strftime("%H:%M:%S"), f"{frac:5.1%}", text, flush=True)


def torch_flavour() -> str:
    """cu128 (NVIDIA), rocm7.1 (AMD Radeon RX 7000/9000) or cpu. CONDUCTOR_TORCH overrides."""
    f = os.environ.get("CONDUCTOR_TORCH")
    if f:
        return f
    try:
        if subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0:
            return "cu128"
    except Exception:
        pass
    import glob
    for v in glob.glob("/sys/class/drm/card*/device/vendor"):
        try:
            if open(v).read().strip() == "0x1002" and os.path.exists("/dev/kfd"):
                return "rocm7.1"
        except OSError:
            pass
    return "cpu"


def _lock_hash() -> str:
    h = hashlib.md5(open(LOCK_FILE, "rb").read()).hexdigest() if LOCK_FILE and os.path.exists(LOCK_FILE) else ""
    return h + ("|" + torch_flavour() if LINUX and h else "")


def needs_setup() -> bool:
    if not PORTABLE:
        return False
    try:
        return open(STAMP, encoding="utf-8").read().strip() != _lock_hash()
    except OSError:
        return True


def setup_linux() -> bool:
    """uv pip install into runtime/linux/venv: PyTorch for this card, the locked packages, madmom."""
    import threading
    uv = os.path.join(RUNTIME, "linux", "uv", "uv")
    py = sys.executable
    env = dict(os.environ, UV_CACHE_DIR=os.path.join(RUNTIME, "linux", "uvcache"))
    fl = torch_flavour()
    size = {"cu128": "~3.5 GB", "rocm7.1": "~4 GB", "cpu": "~1 GB"}.get(fl, "")
    steps = [
        (f"PyTorch ({fl}, {size})", [uv, "pip", "install", "--python", py, "--index-url",
                                     f"https://download.pytorch.org/whl/{fl}", f"torch=={TORCH}", f"torchaudio=={TORCH}"]),
        ("music libraries", [uv, "pip", "install", "--python", py, "-r", LOCK_FILE, "cython"]),
        ("madmom (beat tracking)", [uv, "pip", "install", "--python", py, "--no-build-isolation",
                                    "--find-links", os.path.join(RUNTIME, "wheels-linux"), "madmom==0.17.dev0"]),
    ]
    _report(0.0, f"First start: getting Conductor ready (one-time download, {size})")
    for k, (name, cmd) in enumerate(steps):
        t0, done = time.time(), [False]

        def tick(k=k, name=name):
            while not done[0]:
                mins = (time.time() - t0) / 60
                _report((k + 1 - 0.5 ** (mins / 2)) / len(steps), f"First start: installing {name} ({mins:.0f} min)")
                time.sleep(5)
        threading.Thread(target=tick, daemon=True).start()
        p = subprocess.run(cmd, cwd=RUNTIME, env=env, stdout=sys.stdout, stderr=subprocess.STDOUT)
        if p.returncode != 0 and name.startswith("madmom"):            # no prebuilt wheel: build it (needs gcc + git)
            p = subprocess.run([uv, "pip", "install", "--python", py, "--no-build-isolation",
                                "madmom @ git+https://github.com/CPJKU/madmom"], cwd=RUNTIME, env=env,
                               stdout=sys.stdout, stderr=subprocess.STDOUT)
        done[0] = True
        if p.returncode != 0:
            _report(0.0, f"Setup failed ({name}) - see mods/conductor/data/boot.log", "error")
            return False
    with open(STAMP, "w", encoding="utf-8") as f:
        f.write(_lock_hash())
    _report(1.0, "Conductor is ready", "done")
    return True


def setup() -> bool:
    """pip install -r runtime/requirements.lock into runtime/python. ~3.5 GB download (CUDA torch)."""
    if LINUX:
        return setup_linux()
    _report(0.0, "First start: getting Conductor ready (one-time download, ~3.5 GB)")
    cmd = [sys.executable, "-m", "pip", "install", "--no-warn-script-location", "--progress-bar", "off",
           "--disable-pip-version-check", "--find-links", os.path.join(RUNTIME, "wheels"), "-r", LOCK_FILE]
    p = subprocess.Popen(cmd, cwd=RUNTIME, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", creationflags=NOWIN)
    seen, phase, last = 0, "collect", ""
    started = time.time()
    for line in p.stdout:
        line = line.rstrip()
        print(line, flush=True)
        m = re.match(r"\s*Collecting (\S+)", line)
        if m:
            seen += 1
            last = re.split(r"[=<>!~;\[]", m.group(1))[0]
            _report(0.75 * min(1.0, seen / EXPECTED), f"First start: finding {last} ({seen} packages)")
        elif line.lstrip().startswith("Downloading "):
            d = re.match(r"\s*Downloading (?:\S*/)?([A-Za-z0-9_.]+?)-\d\S*\s*\(([\d.]+ [kMG]B)\)", line)
            if d:
                _report(0.75 * min(1.0, seen / EXPECTED), f"First start: downloading {d.group(1)} ({d.group(2)})")
        elif line.startswith("Installing collected packages"):
            phase = "install"
            t_inst = time.time()

            def tick():          # pip is silent while it unpacks (torch alone ~ 5 GB on disk): keep the bar moving
                while p.poll() is None:
                    mins = (time.time() - t_inst) / 60
                    _report(0.8 + 0.18 * (1 - 0.5 ** (mins / 3)), f"First start: installing ({mins:.0f} min so far)")
                    time.sleep(5)
            import threading
            threading.Thread(target=tick, daemon=True).start()
    rc = p.wait()
    if rc != 0:
        _report(0.0, "Setup failed - see mods/conductor/data/boot.log", "error")
        return False
    with open(STAMP, "w", encoding="utf-8") as f:
        f.write(_lock_hash())
    _report(1.0, "Conductor is ready", "done")
    return True


def seed() -> int:
    """Shipped models + profile (app/seed) -> data/ on first run. Never overwrites: data/ is the user's."""
    import shutil
    src_root, n = os.path.join(MOD, "app", "seed"), 0
    for root, _, files in os.walk(src_root):
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(HOME, os.path.relpath(src, src_root))
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
    return n


class _Tee:
    """Standalone: the console window shows what's happening too (first-start install, the page address)."""
    def __init__(self, *fs):
        self.fs = fs

    def write(self, s):
        for f in self.fs:
            try:
                f.write(s)
                f.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        pass


def main() -> int:
    _env()
    standalone = sys.argv[1:2] == ["standalone"]
    log = open(os.path.join(HOME, "boot.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _Tee(log, sys.__stdout__) if standalone and sys.__stdout__ else log
    if standalone and needs_setup():
        print("First start: installing Conductor's Python packages (one time, about 3.5 GB). "
              "This takes a while - leave this window open.", flush=True)
    print(f"\n===== boot {time.strftime('%Y-%m-%d %H:%M:%S')}  home={HOME}  portable={PORTABLE}", flush=True)
    import socket
    guard = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        guard.bind(("127.0.0.1", 47390))     # one boot at a time: the mod relaunches every 60 s while setup runs
    except OSError:
        print("another boot is running (setup in progress); exiting", flush=True)
        return 0
    if needs_setup() and not setup():
        return 1
    guard.close()
    seed()
    from .__main__ import main as app_main
    if standalone:
        sys.argv = ["conductor"] + sys.argv[1:]
        return app_main(sys.argv[1:]) or 0
    sys.argv = ["conductor", "run", "--background"]
    return app_main(["run", "--background"]) or 0


if __name__ == "__main__":
    raise SystemExit(main())
