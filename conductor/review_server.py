"""Local review server: serves review/ (page + assets) and the gameplay video with HTTP Range
support (so the browser can seek a multi-GB file), and saves pins posted by the page.

    python -m conductor.review_server <mission_dir_name> <video_path> [port]
    → http://127.0.0.1:8766/review.html?m=<mission_dir_name>
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "review")
MISSION = sys.argv[1] if len(sys.argv) > 1 else "m8"
VIDEO = sys.argv[2] if len(sys.argv) > 2 else ""
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8766
PINS = os.path.join(ROOT, MISSION, "pins.json")
AB_ANSWERS = os.path.join(ROOT, "ab", "answers.json")
MOM_ANSWERS = os.path.join(ROOT, "moments", "answers.json")
HOOK_PICKS = os.path.join(ROOT, "hooks", "picks.json")
FILES = {"/pins": PINS, "/ab_answers": AB_ANSWERS, "/moments_answers": MOM_ANSWERS, "/hook_picks": HOOK_PICKS}


LIVE = {"proc": None}
TWEAK = {"by": None, "n": 0}
import threading as _threading
_TWEAK_LOCK = _threading.Lock()


def _tweak_render(body: dict) -> dict:
    """Trainer "Tweak": re-render one option of a set with per-lane volume / end changes."""
    sys.path.insert(0, HERE)
    import copy
    from conductor import ab, routes
    qid, k, tw = str(body["quad"]), int(body["k"]), dict(body.get("tweaks") or {})
    quads = json.load(open(os.path.join(ROOT, "ab", "quads.json"), encoding="utf-8"))
    q = next(x for x in quads if x["id"] == qid)
    with _TWEAK_LOCK:
        if TWEAK["by"] is None:
            TWEAK["by"] = {t.title: t for t in ab.training_library().tracks}
        r = copy.deepcopy(q["options"][k])
        if q.get("stretch"):
            r["stretch"] = q["stretch"]
        TWEAK["n"] += 1
        clip = f"ab/clips/{qid}_p{k}_tw{TWEAK['n']}_{int(time.time())}.mp3"
        if "plan" in tw:                            # lanes drawn on the page
            routes.TWEAKS = {"plan": {ln: [[float(x), float(g)] for x, g in pts] for ln, pts in tw["plan"].items()}}
        else:
            routes.TWEAKS = {ln: {kk: float(v) for kk, v in d.items() if kk in ("db", "end")} for ln, d in tw.items()}
        try:
            seams = routes.render(TWEAK["by"], r, os.path.join(ROOT, clip))
            plan = routes.LAST_PLAN
        finally:
            routes.TWEAKS = {}
    return {"clip": clip, "seams": seams, "plan": plan}
LIVE_DIR = os.path.join(ROOT, "live")


def _live_stop():
    p = LIVE["proc"]
    if p is not None and p.poll() is None:
        import subprocess
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)   # replay + the app it runs
    LIVE["proc"] = None
    if LIVE.get("saved"):                       # the app saved the live run's playlist as yours: undo
        try:
            from conductor import playlists
            playlists.set_current(LIVE["saved"])
        except Exception:
            pass


def _live_start(video_from: float, playlist: str = "Darktide II"):
    import subprocess
    _live_stop()
    try:
        sys.path.insert(0, HERE)
        from conductor import app as _app
        LIVE["saved"] = _app.load_config().get("playlist")
    except Exception:
        pass
    try:
        os.remove(os.path.join(LIVE_DIR, "status.json"))
    except OSError:
        pass
    LIVE["proc"] = subprocess.Popen([sys.executable, "-m", "conductor.live_replay", str(video_from), playlist], cwd=HERE,
                                    env={k: v for k, v in dict(os.environ, PYTHONIOENCODING="utf-8").items() if k != "CONDUCTOR_LIVE_PLAYER"},
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def log_message(self, *a):
        pass

    def _send_file(self, path, ctype):
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                left -= len(chunk)

    def do_GET(self):
        if self.path.split("?")[0] == "/live_status":
            st = {}
            try:
                st = json.load(open(os.path.join(LIVE_DIR, "status.json"), encoding="utf-8"))
            except (OSError, ValueError):
                st = {"state": "starting" if LIVE["proc"] is not None and LIVE["proc"].poll() is None else "idle"}
            try:
                st["log"] = open(os.path.join(LIVE_DIR, "app.log"), encoding="utf-8").read().splitlines()[-400:]
            except OSError:
                st["log"] = []
            st["now"] = time.time()
            body = json.dumps(st).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        if self.path.split("?")[0] == "/video":
            return self._send_file(VIDEO, "video/mp4")
        if self.path.split("?")[0] in FILES:
            f = FILES[self.path.split("?")[0]]
            body = open(f, "rb").read() if os.path.exists(f) else b"[]"
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        p = os.path.join(ROOT, self.path.split("?")[0].lstrip("/"))
        if os.path.isfile(p) and self.headers.get("Range"):
            return self._send_file(p, mimetypes.guess_type(p)[0] or "application/octet-stream")
        return super().do_GET()

    def do_POST(self):
        route = self.path.split("?")[0]
        if route in ("/live_start", "/live_stop", "/live_volume"):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if "volume" in body:
                os.makedirs(LIVE_DIR, exist_ok=True)
                open(os.path.join(LIVE_DIR, "volume.txt"), "w").write(str(float(body["volume"])))
            if route == "/live_volume":
                self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
                return
            _live_start(float(body.get("from", 0)), str(body.get("playlist") or "Darktide II")) if route == "/live_start" else _live_stop()
            self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
            return
        if route == "/ab_tweak":
            n = int(self.headers.get("Content-Length", 0))
            try:
                out = _tweak_render(json.loads(self.rfile.read(n) or b"{}"))
                code = 200
            except Exception as e:
                import traceback
                traceback.print_exc()
                out, code = {"error": str(e)}, 500
            body = json.dumps(out, default=float).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            return
        if route not in FILES:
            self.send_response(404); self.end_headers(); return
        target = FILES[route]
        n = int(self.headers.get("Content-Length", 0))
        pins = json.loads(self.rfile.read(n) or b"[]")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"
        json.dump(pins, open(tmp, "w", encoding="utf-8"), indent=1)
        for attempt in range(60):                   # Windows: a reader (e.g. a trainer renderer) can hold it briefly
            try:
                os.replace(tmp, target)
                break
            except PermissionError:
                time.sleep(0.05 * (1 + attempt // 10))
        else:
            self.send_response(503); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b'{"ok":true}')


if __name__ == "__main__":
    print(f"review: http://127.0.0.1:{PORT}/review.html?m={MISSION}", flush=True)

    def _warm():                                    # the trainer's Tweak renders fast from the first one
        try:
            sys.path.insert(0, HERE)
            from conductor import ab
            with _TWEAK_LOCK:
                if TWEAK["by"] is None:
                    TWEAK["by"] = {t.title: t for t in ab.training_library().tracks}
        except Exception as e:
            print("tweak warm-up:", e, flush=True)
    _threading.Thread(target=_warm, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
