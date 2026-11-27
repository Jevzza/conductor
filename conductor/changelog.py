"""In-game change ratings. The app logs every music change it makes (what -> what, where, how, the
game state); the "rate last change" keys in the mod attach 👍/👎 to the latest one. Offline,
`python -m conductor.changelog` re-renders each rated change as a clip and the nightly training
reads them as labelled joins (ab.labelled).

    cache/live/changes.jsonl         every change the app made (one line each)
    cache/live/change_ratings.jsonl  your in-game verdicts, each with its change
"""
from __future__ import annotations

import json
import os
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = os.path.join(HERE, "cache", "live")
CHANGES = os.path.join(LIVE, "changes.jsonl")
RATINGS = os.path.join(LIVE, "change_ratings.jsonl")
CLIPS = os.path.join(HERE, "review", "ingame")
_last: dict = {}
_recent: list = []          # recent changes, for timing verdicts
TIMING = os.path.join(LIVE, "timing_ratings.jsonl")
WINDOW = 15.0
STYLE = {"punch": "punch", "seam": "cut", "plain": "cut", "blend": "xfade1"}   # hand-made moves -> route styles


def _append(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def record(cue, prev_track, prev_pos, game_t, state, prev_state=None) -> None:
    if cue.action not in ("play",) or cue.track is None:
        return
    r = cue.route or {}
    t_out = cue.at if cue.at is not None else (prev_pos or 0.0) + (cue.delay or 0.0)
    steps = r.get("steps") or [{"b": cue.track.title, "t_in": float(cue.start), "bars": None,
                                "style": r.get("style") or STYLE.get(cue.style, "punch")}]
    row = {"wall": time.time(), "game_t": game_t, "state": state, "reason": cue.reason,
           "a": prev_track.title if prev_track else None, "a_pos": prev_pos, "t_out": float(t_out),
           "b": cue.track.title, "t_in": float(steps[0]["t_in"]), "steps": steps,
           "style": r.get("style") or cue.style, "planned": bool(r), "plan": r.get("plan"), "score": r.get("score"), "by": r.get("by"),
           "heard_wall": time.time() + max(0.0, cue.delay or 0.0), "prev_state": prev_state}
    try:                                  # 2026-11-23: up / down / same, so in-game 👍/👎 can teach direction-aware taste
        from .ab import direction
        row["dir"] = direction(prev_state, state)
    except Exception:
        row["dir"] = None
    _last.clear()
    _last.update(row)
    _recent.append(row)
    del _recent[:-10]
    _append(CHANGES, row)


# Darktide's chat font has no emoji (user 2026-11-23): coloured words instead, in the game's text markup
UP = "{#color(90,220,110)}UP{#reset()}"
DOWN = "{#color(235,90,80)}DOWN{#reset()}"


def rate(value: int, game_t) -> str:
    if not _last:
        return "No change to rate yet"
    ago = time.time() - _last["heard_wall"]
    _append(RATINGS, {"value": int(value), "wall": time.time(), "game_t": game_t, "ago": round(ago, 1), "change": dict(_last)})
    a = (_last["a"] or "silence")[:24]
    return f"{UP if value > 0 else DOWN} change ({ago:.0f}s ago): {a} → {_last['b'][:24]} [{_last['style']}]"


def rate_timing(value: int, game_t) -> str:
    """Timing verdict over the last WINDOW s: 2+ changes = flapping, 1 = that change's moment,
    0 = it should have changed by now (missed / late)."""
    now = time.time()
    win = [c for c in _recent if now - c["heard_wall"] <= WINDOW]
    kind = "flap" if len(win) >= 2 else "change" if win else "missed"
    if value > 0 and not win and _recent:
        win, kind = [_recent[-1]], "change"            # 👍 with nothing recent: the last change was right
    _append(TIMING, {"value": int(value), "wall": now, "game_t": game_t, "kind": kind, "changes": win})
    if value > 0:
        return f"{UP} timing: {'that change came at the right moment' if win else 'no change needed'}"
    return {"flap": f"{DOWN} timing: {len(win)} changes in {WINDOW:.0f}s - too many",
            "change": f"{DOWN} timing: that change ({now - win[-1]['heard_wall']:.0f}s ago) was the wrong moment" if win else "",
            "missed": f"{DOWN} timing: it should have changed by now"}[kind]


def timing_rated() -> list[dict]:
    if not os.path.exists(TIMING):
        return []
    return [json.loads(l) for l in open(TIMING, encoding="utf-8") if l.strip()]


def rated() -> list[dict]:
    if not os.path.exists(RATINGS):
        return []
    out = []
    for l in open(RATINGS, encoding="utf-8"):
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def rows(lib) -> list[dict]:
    """Labelled joins for training: the change as a route, good = your thumb."""
    by = {t.title for t in lib.tracks}
    out = []
    for r in rated():
        c = r["change"]
        if not c.get("a") or c["a"] not in by or c["b"] not in by or c["a"] == c["b"] and not c.get("planned"):
            continue
        out.append({"a": c["a"], "t_out": c["t_out"], "b": c["b"], "t_in": c["t_in"], "steps": c["steps"],
                    "good": int(r["value"] > 0), "w": 1.0, "src": "ingame", "dir": c.get("dir")})
    # rated by feel in game: a 👎 is often the loop/timing, not the join's sound -> low weight (2026-11-21),
    # and the same change rated again and again counts once in total
    seen = {}
    for o in out:
        seen.setdefault((o["a"], round(o["t_out"]), o["b"], round(o["t_in"]), o["steps"][0]["style"]), []).append(o)
    for grp in seen.values():
        for o in grp:
            o["w"] = 1.0 / len(grp)
    return out


def clips() -> int:
    """Re-render every rated change (10 s around it) into review/ingame/ to listen back."""
    from .library import Library
    from . import routes
    lib = Library.build(__import__("conductor.playlists", fromlist=["train_music"]).train_music(), os.path.join(HERE, "cache"), progress=lambda m: None,
                        backfill=False, cached_only=True)
    bylib = {t.title: t for t in lib.tracks}
    os.makedirs(CLIPS, exist_ok=True)
    n = 0
    for row in rows(lib):
        name = f"{int(row['t_out'] * 10)}_{row['a'][:20]}_{row['b'][:20]}_{'up' if row['good'] else 'down'}.mp3"
        name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        p = os.path.join(CLIPS, name)
        if not os.path.exists(p):
            try:
                routes.render(bylib, row, p)
                n += 1
            except Exception as e:
                print("skip", name, e)
    return n


def why() -> None:
    """After a session: which situations your 👎 cluster in (no need to know what the change was -
    the log knows the game state, the trigger, the song and what the music did)."""
    import collections
    ch = [json.loads(l) for l in open(CHANGES, encoding="utf-8")] if os.path.exists(CHANGES) else []
    idx = {c["wall"]: i for i, c in enumerate(ch)}

    def feats(c):
        i = idx.get(c["wall"])
        prev = ch[max(0, i - 8):i] if i is not None else []
        land = (c["b"], round(c["steps"][-1]["t_in"]))
        st = c["reason"].split(":")[1] if c["reason"].startswith("st:") else "scene/menu"
        return {"same song": c["a"] == c["b"],
                "build-up -> drop route": len(c["steps"]) == 2,
                "loops back to a part heard <3 min ago": any((p["b"], round(p["steps"][-1]["t_in"])) == land
                                                            and c["wall"] - p["wall"] < 180 for p in prev),
                "2+ changes in the minute before": sum(c["wall"] - p["wall"] < 60 for p in prev) >= 2,
                "game state": c.get("state"), "trigger": st, "join": c.get("style")}
    rows = [(feats(r["change"]), r["value"] < 0, "transition") for r in rated()]
    rows += [(feats(r["changes"][-1]), r["value"] < 0, "timing") for r in timing_rated() if r["changes"]]
    if not rows:
        print("no in-game ratings yet")
        return
    print(f"{len(rows)} verdicts, {round(100 * sum(b for _, b, _ in rows) / len(rows))}% 👎 overall\n")
    for k in rows[0][0]:
        g = collections.defaultdict(list)
        for f, b, _ in rows:
            g[f[k]].append(b)
        cells = [f"{v}: {round(100 * sum(x) / len(x))}% 👎 (n={len(x)})" for v, x in sorted(g.items(), key=lambda t: -len(t[1])) if len(x) >= 3]
        print(f"{k:40s}" + "   ".join(cells))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "why":
        why()
        raise SystemExit
    rs = rated()
    print(len(rs), "in-game change ratings:", sum(r["value"] > 0 for r in rs), "👍", sum(r["value"] < 0 for r in rs), "👎")
    print(clips(), "new clips in", CLIPS)
