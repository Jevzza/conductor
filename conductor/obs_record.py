"""Record a play session with OBS and keep a sync log so video ↔ game telemetry ↔ Conductor line up.

    python -m conductor.obs_record            # start recording now; stop when Darktide closes
    (or create %TEMP%\\conductor_obs_stop.txt to stop early)

recordings/obs_<stamp>.sync.json holds:
  record_start_wall   wall-clock time OBS reported the recording started
  samples             [wall_time, game_t] each time a new bridge event arrived (≈ every event)
  video_path          the file OBS wrote
so any game time t maps to a video time:  video_s = wall(t) - record_start_wall.
The OBS websocket password is read from OBS's own config on this machine (never printed).
"""
from __future__ import annotations

import configparser
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = os.path.join(os.environ.get("TEMP", "/tmp"), "conductor_events.jsonl")
STOP = os.path.join(os.environ.get("TEMP", "/tmp"), "conductor_obs_stop.txt")


def obs_settings():
    ini = configparser.ConfigParser(strict=False, interpolation=None)
    ini.read(os.path.join(os.environ["APPDATA"], "obs-studio", "global.ini"), encoding="utf-8-sig")
    s = ini["OBSWebSocket"] if "OBSWebSocket" in ini else {}
    return int(s.get("ServerPort", 4455)), (s.get("ServerPassword") if s.get("AuthRequired", "true") == "true" else None)


def game_running() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Darktide.exe", "/NH"], capture_output=True, text=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return "darktide.exe" in out.stdout.lower()


def game_state() -> str | None:
    """Latest music_game_state the bridge reported (loadout, loading, mission, …)."""
    try:
        with open(EVENTS, encoding="utf-8", errors="replace") as f:
            f.seek(max(0, os.path.getsize(EVENTS) - 200000))
            last = None
            for line in f:
                if '"music_game_state"' in line:
                    try:
                        last = json.loads(line).get("value")
                    except ValueError:
                        pass
            return last
    except OSError:
        return None


def connect(wait_s=7200):
    """Wait for OBS's WebSocket server (Tools → WebSocket Server Settings → Enable) and for the game."""
    import logging
    import obsws_python as obs
    logging.getLogger("obsws_python").setLevel(logging.CRITICAL)
    logging.disable(logging.ERROR)
    t0, said = time.time(), set()
    while time.time() - t0 < wait_s:
        port, pw = obs_settings()
        try:
            cl = obs.ReqClient(host="127.0.0.1", port=port, password=pw, timeout=3)
        except Exception:
            if "obs" not in said:
                print("waiting for the OBS WebSocket server to be enabled…", flush=True); said.add("obs")
            time.sleep(5)
            continue
        if not game_running():
            if "game" not in said:
                print("OBS connected; waiting for Darktide to start…", flush=True); said.add("game")
            time.sleep(5)
            continue
        if game_state() not in ("loadout", "loadout_ready", "loading", "mission_briefing", "mission_intro"):
            if "loadout" not in said:
                print("game running; waiting for the loadout screen…", flush=True); said.add("loadout")
            time.sleep(1)
            continue
        return cl
    raise SystemExit("gave up waiting")


END_STATES = ("victory", "defeat", "game_score", "game_score_win", "game_score_lose")
END_TAIL = 10.0          # keep recording this long after the mission ends


def main():
    """One recording per mission: loadout → victory/defeat (+10 s). Re-arms until the game closes.
    --once: record only the first mission, then exit."""
    import sys
    once = "--once" in sys.argv
    while True:
        cl = connect()
        why = record_one(cl)
        if why == "game_closed" or once:
            print("done (--once)" if once else "game closed", flush=True)
            break


def ensure_audio(cl):
    """Last time the recording had no sound: make sure OBS's audio sources are on before recording."""
    try:
        for i in cl.get_input_list().inputs:
            kind, name = i.get("inputKind", ""), i["inputName"]
            if kind not in ("wasapi_output_capture", "wasapi_process_output_capture"):
                continue                         # desktop/game audio only: never touch a microphone
            try:
                if cl.get_input_mute(name).input_muted:
                    cl.set_input_mute(name, False)
                    print(f"unmuted OBS audio source: {name}", flush=True)
                db = cl.get_input_volume(name).input_volume_db
                if db < -30:
                    cl.set_input_volume(name, vol_db=0.0)
                    print(f"OBS audio source {name} was at {db:.0f} dB: set to 0 dB", flush=True)
            except Exception as e:
                print(f"couldn't check {name}: {e}", flush=True)
    except Exception as e:
        print("audio check failed:", e, flush=True)


def record_one(cl):
    ensure_audio(cl)
    if cl.get_record_status().output_active:
        print("OBS is already recording; using that recording")
    else:
        cl.start_record()
    t_start = time.time()
    for _ in range(50):                                   # wait until OBS confirms it's recording
        if cl.get_record_status().output_active:
            break
        time.sleep(0.1)
    rec_start = time.time() - cl.get_record_status().output_duration / 1000.0
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = os.path.join(HERE, "recordings", f"obs_{stamp}.sync.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sync = {"record_start_wall": rec_start, "events_file": EVENTS, "samples": [], "video_path": None}
    print(f"recording; sync → {out}", flush=True)
    if os.path.exists(STOP):
        os.remove(STOP)
    pos, last_save, ended_at, why = 0, 0.0, None, "mission_end"
    try:
        while True:
            # sample (wall, game t) as bridge events arrive
            try:
                size = os.path.getsize(EVENTS)
                if size < pos:
                    pos = 0                                   # bridge restarted its log
                with open(EVENTS, encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                now = time.time()
                for line in chunk.splitlines():
                    try:
                        t = json.loads(line).get("t")
                    except ValueError:
                        continue
                    if isinstance(t, (int, float)):
                        if not sync["samples"] or t - sync["samples"][-1][1] >= 0.5 or t < sync["samples"][-1][1]:
                            sync["samples"].append([round(now, 3), t])
            except OSError:
                pass
            if time.time() - last_save > 10:
                json.dump(sync, open(out, "w", encoding="utf-8"))
                last_save = time.time()
            st = game_state()
            if st in END_STATES and ended_at is None:
                ended_at = time.time()
                print(f"mission ended ({st}); stopping in {END_TAIL:.0f}s", flush=True)
            if ended_at is not None and time.time() - ended_at >= END_TAIL:
                break
            if os.path.exists(STOP):
                why = "stopped"
                break
            if not game_running():
                why = "game_closed"
                break
            time.sleep(0.2)
    finally:
        try:
            r = cl.stop_record()
            sync["video_path"] = getattr(r, "output_path", None)
        except Exception as e:
            print("stop failed:", e)
        sync["record_stop_wall"] = time.time()
        json.dump(sync, open(out, "w", encoding="utf-8"), indent=1)
        print(f"stopped ({why}); video {sync['video_path']}; {len(sync['samples'])} sync samples → {out}", flush=True)
        try:                                   # did the recording actually capture sound?
            import re as _re
            time.sleep(3)
            r = subprocess.run(["ffmpeg", "-hide_banner", "-ss", "60", "-t", "120", "-i", sync["video_path"],
                                "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            m = _re.search(r"max_volume: (-?[\d.]+) dB", r.stderr)
            print(f"recording audio check: max {m.group(1) if m else '?'} dB"
                  + ("  ← SILENT!" if m and float(m.group(1)) < -60 else "  (has sound)"), flush=True)
        except Exception as e:
            print("audio check skipped:", e, flush=True)
    if why == "mission_end":                         # wait until we're out of the end screens before re-arming
        while game_running() and game_state() in END_STATES + ("loading",):
            time.sleep(2)
    return why


if __name__ == "__main__":
    main()
