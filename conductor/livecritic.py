"""App side of the live critic: a background process (critic_live.py, ML venv) listens to candidate changes
ahead of time; the planner uses its verdicts when they're ready and the trees taste model when they aren't.

Off unless config "live_critic": true or CONDUCTOR_LIVE_CRITIC=1. Never blocks the music loop: submit()
writes one line and returns; a reader thread files the answers.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(HERE, ".venv-ml", "Scripts", "python.exe")
NOWIN = 0x08000000 if os.name == "nt" else 0


def key(route) -> tuple:
    """Same change = same destination parts and styles (t_out differs between when we listen and when we play)."""
    return tuple((s["b"], round(float(s["t_in"]), 1), s.get("style")) for s in route["steps"])


class LiveCritic:
    MAX_SCORES = 3000

    def __init__(self):
        self.scores: dict = {}
        self.pending: dict = {}          # request id -> [keys]
        self.ready = False
        self.failed = None
        self._id = 0
        self._lock = threading.Lock()
        self.stats = {"sent": 0, "heard": 0, "used": 0, "missed": 0, "top4_sent_before": 0, "top4_total": 0}
        self.sent_keys: set = set()
        env = dict(os.environ, HF_HOME=os.path.join(HERE, ".hf"), PYTHONIOENCODING="utf-8")
        try:
            self.proc = subprocess.Popen([VENV_PY, "-W", "ignore", "-m", "conductor.critic_live"], cwd=HERE,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, encoding="utf-8", errors="replace", env=env, creationflags=NOWIN)
        except OSError as e:
            self.proc, self.failed = None, str(e)
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                d = json.loads(line)
            except ValueError:
                continue                        # library build chatter
            if d.get("ready"):
                self.ready = True
                print(f"[conductor] live critic ready (taste layer {d.get('teacher')})", flush=True)
                continue
            if "id" not in d:
                continue
            with self._lock:
                keys = self.pending.pop(d["id"], [])
                for k, s in zip(keys, d.get("scores") or []):
                    if s is not None:
                        self.scores[k] = s
                        self.stats["heard"] += 1
                if len(self.scores) > self.MAX_SCORES:
                    for k in list(self.scores)[: len(self.scores) - self.MAX_SCORES]:
                        del self.scores[k]
        self.ready = False
        self.failed = self.failed or "critic process ended"

    def busy(self) -> bool:
        return bool(self.pending)

    def submit(self, music: str, routes_) -> int:
        """Queue changes to listen to (skips ones already heard or queued). Returns how many were sent."""
        if not self.ready or self.proc is None or self.proc.poll() is not None:
            return 0
        with self._lock:
            queued = {k for ks in self.pending.values() for k in ks}
            todo = [r for r in routes_ if key(r) not in self.scores and key(r) not in queued]
            if not todo:
                return 0
            self._id += 1
            self.pending[self._id] = [key(r) for r in todo]
            self.sent_keys.update(self.pending[self._id])
        try:
            self.proc.stdin.write(json.dumps({"id": self._id, "music": music, "routes": todo}) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            self.failed = "couldn't reach the critic process"
            return 0
        self.stats["sent"] += len(todo)
        return len(todo)

    def score(self, route):
        return self.scores.get(key(route))

    def close(self):
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
