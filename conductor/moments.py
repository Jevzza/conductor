"""Moments: judge the music IN the gameplay. Each moment is ~22 s of mission video around a game
state change (or a song change the live system made on its own) with 3-4 music versions synced to it:

    live   what the full live system did (tools/showcase_real.py render of the recording)
    stay   no change: the song that was playing just carries on
    now    the planner's best route, played on the bar after the game changed
    early  the same route, 3 s before the game changed (anticipating it)

You pick best / worst. Picks label WHETHER and WHEN to change, not only how a join sounds.

    python -m conductor.moments make [n]      # new set -> review/moments/set.json + clips
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "review", "moments")
SR = 44100
PRE, W = 8.0, 22.0                   # seconds of video before the moment / clip length
EARLY = 3.0
# recorded missions to review: cache/moments/missions.json = {"m9": {"video": "<recording.mp4>"}, ...}
MISSIONS = {k: {"render": os.path.join(HERE, "cache", "moments", "render_" + k), **v}
            for k, v in (json.load(open(os.path.join(HERE, "cache", "moments", "missions.json"), encoding="utf-8"))
                         if os.path.exists(os.path.join(HERE, "cache", "moments", "missions.json")) else {}).items()}
STATE = {"explore": "calm", "tier1": "fight", "tier2": "horde", "tier3": "boss"}
RANK = {"calm": 0, "fight": 2, "horde": 3, "boss": 4}
NAME = {"calm": "Calm", "fight": "Fight", "horde": "Horde", "boss": "Boss"}


def _answers() -> list[dict]:
    p = os.path.join(OUT, "answers.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []


def _changes(pos) -> list[float]:
    """Times the live system started a new song or jumped inside one."""
    out = []
    for (t0, s0, p0), (t1, s1, p1) in zip(pos, pos[1:]):
        if s1 != s0 or abs((p1 - p0) - (t1 - t0)) > 1.5:
            out.append(t1)
    return out


def find(m: str) -> list[dict]:
    tl = json.load(open(os.path.join(MISSIONS[m]["render"], "timeline_conductor.json"), encoding="utf-8"))
    ph = [p for p in tl["phases"]]
    moms, last_state = [], None
    for k, p in enumerate(ph):
        st = STATE.get(p["phase"])
        prev = STATE.get(ph[k - 1]["phase"]) if k else None
        if st is None:
            last_state = None
            continue
        if prev is not None and st != prev and (not moms or p["t"] - moms[-1]["t"] > 12):
            moms.append({"t": p["t"], "from": prev, "to": st, "kind": "game"})
    lc = _changes(tl["pos"])
    for t in lc:                                  # the live system changing with no game change
        if all(abs(t - x["t"]) > 10 for x in moms):
            st = None
            for p in ph:
                if p["t"] <= t:
                    st = STATE.get(p["phase"])
            if st is not None:
                moms.append({"t": t, "from": st, "to": st, "kind": "music"})
    dur = tl.get("duration", 1e9)
    moms = [dict(x, m=m, id=f"{m}_{x['t']:.1f}") for x in moms if PRE < x["t"] < dur - W]
    return sorted(moms, key=lambda x: x["t"])


def _song_at(pos, t):
    """(title, song position) at video time t, from the per-second timeline."""
    best = None
    for tt, s, p in pos:
        if tt <= t:
            best = (s, p + (t - tt))
        else:
            break
    return best


def _rms_norm(y):
    r = float(np.sqrt(np.mean(y ** 2))) + 1e-9
    y = y * (0.12 / r)
    return y / max(1.0, float(np.abs(y).max()) / 0.97)


def _write(y, path):
    import soundfile as sf
    n = int(W * SR)
    y = y[:n]
    if len(y) < n:
        y = np.vstack([y, np.zeros((n - len(y), 2), np.float32)])
    wav = path[:-4] + ".wav"
    sf.write(wav, _rms_norm(y), SR)
    subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-b:a", "160k", "-y", path], check=True)
    os.remove(wav)


def _decode(path, t0):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{W:.3f}", "-i", path, "-f", "f32le",
                        "-ac", "2", "-ar", str(SR), "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2).copy()


def _plan(lib, by, A, t_out, to, frm, rng, prefer=None):
    """The live planner's choice (same pool, same filters, same taste model) for a change at t_out."""
    from . import ab, routes
    from .states import StateDirector, serves
    if prefer is not None:
        B, tin = prefer
    else:
        cands = [(T, s) for T in lib.tracks if T is not A for s in T.segments if serves(s, to) and s.end - s.start >= 6]
        B, s = rng.choice(cands)
        tin = StateDirector._land(B, s)
    j = {"a": A.title, "t_out": t_out, "b": B.title, "t_in": tin}
    pool = routes.natural_routes(lib, j, to, RANK[to] < RANK.get(frm, 0), rng) + routes.candidates(lib, j, to, rng)
    ok = []
    for r in pool:
        st = r["steps"]
        fin = st[-1]["b"]
        if any(x["b"] != fin for x in st) or any(x["style"] in ("riser", "filter", "drop") for x in st):
            continue
        if len(st) > 1 and (len(st) != 2 or st[1]["style"] != "natural"):
            continue
        if st[0]["t_in"] < 0.5 or not serves(by[fin].segment_at(st[-1]["t_in"] + 0.05), to):
            continue
        ok.append(r)
    if not ok:
        return None
    sc = ab.taste_many(by, ok)
    k = int(np.argmax(sc))
    return dict(ok[k], score=round(float(sc[k]), 3))


def _render_route(by, route, lead, path):
    """route rendered so its change lands `lead` seconds into the clip."""
    from . import routes
    st = route["steps"]
    d1 = (st[0]["until"] - st[0]["t_in"]) if len(st) == 2 and st[0].get("until") else 0.0
    keep = routes.PRE, routes.POST
    routes.PRE, routes.POST = lead, max(1.0, W - lead - d1 + 1.0)
    try:
        routes.render(by, route, path)
    finally:
        routes.PRE, routes.POST = keep
    y = _decode(path, 0.0)
    _write(y, path)


def make(n: int = 6, seed: int | None = None) -> dict:
    from .library import Library
    from . import routes
    lib = Library.build(__import__("conductor.playlists", fromlist=["train_music"]).train_music(), os.path.join(HERE, "cache"), progress=lambda m: None,
                        backfill=False, cached_only=True)
    by = {t.title: t for t in lib.tracks}
    rng = random.Random(seed if seed is not None else time.time())
    done = {a["id"] for a in _answers()}
    pool = [x for m in MISSIONS if os.path.exists(os.path.join(MISSIONS[m]["render"], "timeline_conductor.json"))
            for x in find(m) if x["id"] not in done]
    game = [x for x in pool if x["kind"] == "game"]
    music = [x for x in pool if x["kind"] == "music"]
    pick = rng.sample(game, min(len(game), n - n // 3)) + rng.sample(music, min(len(music), n // 3))
    rng.shuffle(pick)
    sid = f"s{int(time.time())}"
    d = os.path.join(OUT, sid)
    os.makedirs(d, exist_ok=True)
    items = []
    for i, x in enumerate(pick):
        M = MISSIONS[x["m"]]
        tl = json.load(open(os.path.join(M["render"], "timeline_conductor.json"), encoding="utf-8"))
        v0 = x["t"] - PRE
        sa = _song_at(tl["pos"], v0)
        if sa is None or sa[0] not in by:
            continue
        A, pA = by[sa[0]], sa[1]
        base = f"{sid}/{i}"
        opts = []
        try:
            # the clip: video + game sound
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{v0:.3f}", "-i", M["video"], "-t", f"{W:.3f}",
                            "-vf", "scale=-2:720", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                            "-c:a", "aac", "-b:a", "96k", "-y", os.path.join(OUT, base + "_video.mp4")], check=True)
            live = _decode(os.path.join(M["render"], "showcase_conductor.mp3"), v0)
            _write(live, os.path.join(OUT, base + "_live.mp3"))
            lc = [t - v0 for t in _changes(tl["pos"]) if v0 < t < v0 + W]
            after = _song_at(tl["pos"], x["t"] + 4)
            opts.append({"kind": "live", "file": base + "_live.mp3", "changes": [round(c, 2) for c in lc],
                         "desc": "live system: " + (f"changed at {', '.join(f'{c:.1f}s' for c in lc)}" if lc else "no change")})
            if lc:
                _write(routes._take(routes._audio(A), pA, W), os.path.join(OUT, base + "_stay.mp3"))
                opts.append({"kind": "stay", "file": base + "_stay.mp3", "changes": [], "desc": f"no change: {A.title} plays on"})
            prefer = None
            if after and after[0] in by and after[0] != A.title:
                prefer = (by[after[0]], after[1] - 4.0)
            t_now = A.next_bar(pA + PRE + 0.5)
            r = _plan(lib, by, A, t_now, x["to"], x["from"], rng, prefer)
            if r is not None:
                _render_route(by, r, t_now - pA, os.path.join(OUT, base + "_now.mp3"))
                opts.append({"kind": "now", "file": base + "_now.mp3", "changes": [round(t_now - pA, 2)],
                             "desc": f"planner, on the bar after the game changed: {routes.describe(r)}", "route": r})
                t_e = A.next_bar(pA + PRE - EARLY)
                if t_e < t_now - 1.0:
                    re = dict(r, t_out=t_e)
                    _render_route(by, re, t_e - pA, os.path.join(OUT, base + "_early.mp3"))
                    opts.append({"kind": "early", "file": base + "_early.mp3", "changes": [round(t_e - pA, 2)],
                                 "desc": f"same route {t_now - t_e:.1f}s early (anticipating)", "route": re})
        except Exception as e:
            print("skip", x["id"], e, file=sys.stderr)
            continue
        if len(opts) < 2:
            continue
        rng.shuffle(opts)
        items.append({"id": x["id"], "mission": x["m"], "kind": x["kind"], "from": x["from"], "to": x["to"],
                      "at": PRE, "video": base + "_video.mp4", "song": A.title, "song_pos": round(pA, 2),
                      "context": (f"{NAME[x['from']]} → {NAME[x['to']]}" if x["kind"] == "game" else f"{NAME[x['to']]} (no game change)"),
                      "options": opts})
        print(f"  {len(items)}. {x['id']} {items[-1]['context']}: {len(opts)} options", flush=True)
    s = {"id": sid, "made": time.time(), "items": items}
    json.dump(s, open(os.path.join(OUT, "set.json"), "w", encoding="utf-8"), indent=1)
    return {"set": sid, "moments": len(items), "pool_left": len(pool) - len(items)}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "make":
        print(json.dumps(make(int(sys.argv[2]) if len(sys.argv) > 2 else 6), indent=1))
    else:
        for m in MISSIONS:
            try:
                f = find(m)
                print(m, len(f), "moments:", sum(x["kind"] == "game" for x in f), "game,", sum(x["kind"] == "music" for x in f), "music-only")
            except OSError as e:
                print(m, "not rendered yet")
