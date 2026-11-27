"""Sync-only logger (no OBS control needed): samples wall-clock ↔ game time from the bridge's event
log, so a manually recorded video (whose start time is in its file name / creation time) can be
aligned with the telemetry. Stops ~10 s after the mission ends or when the game closes.

    python -m conductor.sync_log <video_path_or_empty>
"""
import json, os, sys, time
from .obs_record import EVENTS, END_STATES, game_running, game_state, HERE

video = sys.argv[1] if len(sys.argv) > 1 else ""
out = os.path.join(HERE, "recordings", f"manual_{time.strftime('%Y-%m-%d_%H%M%S')}.sync.json")
sync = {"video_path": video, "video_ctime": os.path.getctime(video) if video and os.path.exists(video) else None,
        "samples": [], "events_file": EVENTS}
pos, ended, last_save = 0, None, 0
while game_running():
    try:
        if os.path.getsize(EVENTS) < pos:
            pos = 0
        with open(EVENTS, encoding="utf-8", errors="replace") as f:
            f.seek(pos); chunk = f.read(); pos = f.tell()
        now = time.time()
        for line in chunk.splitlines():
            try:
                t = json.loads(line).get("t")
            except ValueError:
                continue
            if isinstance(t, (int, float)) and (not sync["samples"] or t - sync["samples"][-1][1] >= 0.5
                                                or t < sync["samples"][-1][1]):
                sync["samples"].append([round(now, 3), t])
    except OSError:
        pass
    if game_state() in END_STATES and ended is None:
        ended = time.time()
    if ended and time.time() - ended > 10:
        break
    if time.time() - last_save > 10:
        json.dump(sync, open(out, "w", encoding="utf-8")); last_save = time.time()
    time.sleep(0.2)
sync["end_wall"] = time.time()
json.dump(sync, open(out, "w", encoding="utf-8"), indent=1)
print(f"sync log done: {len(sync['samples'])} samples → {out}", flush=True)
