"""Run the REAL Conductor app, live and in real time, against a recorded mission: the recorded
bridge events are written into a private events file at the pace they happened, exactly as the
game mod writes them, and the unmodified app (`python -m conductor run`) plays through the speakers.
Nothing is pre-rendered. The review page starts this and plays the gameplay video in sync.

    python -m conductor.live_replay [video_seconds]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(HERE, "recordings", "darktide_2026-11-18_mission9.jsonl")
VIDEO_ZERO = 32.42                 # recording time at the gameplay video's 0:00
LIVE = os.path.join(HERE, "review", "live")
EVENTS = os.path.join(os.environ.get("TEMP", "/tmp"), "conductor_live_replay.jsonl")


def _status(**kw):
    os.makedirs(LIVE, exist_ok=True)
    tmp = os.path.join(LIVE, "status.json.tmp")
    json.dump(dict(kw, updated=time.time()), open(tmp, "w", encoding="utf-8"))
    os.replace(tmp, os.path.join(LIVE, "status.json"))


def main(video_from: float = 0.0, playlist: str = "Darktide II"):
    os.makedirs(LIVE, exist_ok=True)
    open(EVENTS, "w").close()
    # the app saves the playlist it switches to as the user's choice; put the real one back after
    from . import app as _app, playlists as _pls
    saved = _app.load_config().get("playlist")
    log = open(os.path.join(LIVE, "app.log"), "w", encoding="utf-8", buffering=1)
    _status(state="starting", video_from=video_from)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    app = subprocess.Popen([sys.executable, "-m", "conductor", "run", "--events", EVENTS, "--player", os.environ.get("CONDUCTOR_LIVE_PLAYER", "local")],
                           cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           encoding="utf-8", errors="replace")
    ready = threading.Event()

    def pump():
        for line in app.stdout:
            log.write(f"{time.time():.2f} {line}")
            if line.startswith("Conductor:") or "already running" in line:
                ready.set()
    threading.Thread(target=pump, daemon=True).start()
    if not ready.wait(60) or app.poll() is not None:
        _status(state="failed", reason="the app did not start (is the real Conductor app already running?)")
        app.kill()
        return
    lines = []
    for raw in open(REC, encoding="utf-8"):
        try:
            ev = json.loads(raw)
        except ValueError:
            continue
        t = ev.get("t")
        # the mission was recorded with the in-game music slider at 0 and the native direction, and
        # with alt-tab focus mutes: replay it as a Conductor session at full level instead
        if ev.get("k") == "focus":
            continue
        if ev.get("k") == "volume":
            ev.update(master=100, music=100, level=80, direction="conductor", enabled=True, playlist=playlist)
            raw = json.dumps(ev)
        lines.append((t, raw if raw.endswith("\n") else raw + "\n"))
    t_from = VIDEO_ZERO + video_from
    out = open(EVENTS, "a", encoding="utf-8", buffering=1)
    k = 0
    while k < len(lines) and (lines[k][0] is None or lines[k][0] <= t_from):   # everything up to here at once
        out.write(lines[k][1]); k += 1
    start = time.time()
    vol_p = os.path.join(LIVE, "volume.txt")         # the review page's music slider (0..1)
    vol_m = None

    def volume():
        nonlocal vol_m
        try:
            m = os.path.getmtime(vol_p)
        except OSError:
            return
        if m == vol_m:
            return
        vol_m = m
        try:
            v = max(0.0, min(1.0, float(open(vol_p).read().strip())))
        except ValueError:
            return
        # the same event the mod sends when you move its sliders
        out.write(json.dumps({"k": "volume", "t": t_from + time.time() - start, "master": 100, "music": 100,
                              "level": round(v * 100), "enabled": True}) + chr(10))
    volume()
    _status(state="running", video_from=video_from, start_wall=start, pid=app.pid, playlist=playlist)
    try:
        while k < len(lines) and app.poll() is None:
            volume()
            t, raw = lines[k]
            if t is not None:
                wait = start + (t - t_from) - time.time()
                if wait > 0:
                    time.sleep(min(wait, 0.2))
                    continue
            out.write(raw); k += 1
        while app.poll() is None:
            volume()
            time.sleep(0.2)
    finally:
        app.kill()
        if saved:
            try:
                _pls.set_current(saved)
            except Exception:
                pass
        _status(state="stopped", video_from=video_from)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0, sys.argv[2] if len(sys.argv) > 2 else "Darktide II")
