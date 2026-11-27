"""After every training round: replay the recorded mission through the WHOLE live system (game
telemetry -> director -> learned planner -> live player) and publish it on the mission review page
as a new numbered version. Versions are never overwritten, so moment ratings stay linked to what
was actually heard (rated moments train the model: ratedjoins.py).

    python -m conductor.mission_render [label]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MISSION = os.path.join(HERE, "review", "m9")
REC = os.path.join(HERE, "recordings", "darktide_2026-11-18_mission9.jsonl")
T0, T1 = 32.42, 1479.42          # = the gameplay video's 0:00 .. end (see review/m9)


def render(label: str | None = None) -> dict:
    vers_p = os.path.join(MISSION, "versions.json")
    vers = json.load(open(vers_p, encoding="utf-8")) if os.path.exists(vers_p) else []
    n = 1 + max([v.get("n", 0) for v in vers] or [5])
    d = f"conductor_r{n}"
    tmp = tempfile.mkdtemp(prefix="mission_")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, os.path.join(HERE, "tools", "showcase_real.py"), tmp, REC, str(T0), str(T1),
                        "conductor"], cwd=HERE, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    shutil.copy(os.path.join(tmp, "showcase_conductor.mp3"), os.path.join(MISSION, f"music_{d}.mp3"))
    shutil.copy(os.path.join(tmp, "timeline_conductor.json"), os.path.join(MISSION, f"timeline_{d}.json"))
    shutil.rmtree(tmp, ignore_errors=True)
    chk = subprocess.run([sys.executable, "-m", "conductor.checks", os.path.join(MISSION, f"timeline_{d}.json")],
                         cwd=HERE, env=env, capture_output=True, text=True, encoding="utf-8").stdout
    summary = [l.strip() for l in chk.splitlines() if l.strip().startswith(("fits", "boss", "heard", "silent", "level"))]
    pol = {}
    try:
        pol = json.load(open(os.path.join(HERE, "cache", "live", "policy.json"), encoding="utf-8"))
    except Exception:
        pass
    v = {"n": n, "dir": d, "label": label or f"r{n} · {time.strftime('%d %b %H:%M')}", "when": time.time(),
         "policy": pol.get("version"), "checks": summary}
    vers.append(v)
    json.dump(vers, open(vers_p, "w", encoding="utf-8"), indent=1)
    return v


if __name__ == "__main__":
    print(json.dumps(render(sys.argv[1] if len(sys.argv) > 1 else None), indent=1))
