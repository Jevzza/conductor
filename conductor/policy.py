"""The live transition policy: which joining style to use for each kind of change.

Rebuilt after every training round from (a) your A/B ratings and (b) the model's own scores on
thousands of rendered changes (critic + taste layer, `cache/live/model_styles.json`). The app
re-reads `cache/live/policy.json` whenever it changes, so a new round reaches the game without a
restart. Every build is kept in `cache/live/history/` so a bad round can be rolled back.

    python -m conductor.policy build       # merge ratings + model evidence -> policy.json
    python -m conductor.policy show
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = os.path.join(HERE, "cache", "live")
POLICY = os.path.join(LIVE, "policy.json")
MODEL_STYLES = os.path.join(LIVE, "model_styles.json")

# styles the live player can build (all of them via routes._join); filter/riser change the music
# BEFORE the switch, which a live player can't take back, so they stay offline-only
LIVE_STYLES = {
    "lead": ("xfade1", "xfade2", "dj", "djverb", "djblend", "mix", "echo", "game"),   # into a build-up, then its drop
    "direct": ("xfade1", "xfade2", "xfade4", "echo", "punch", "game"),               # straight into a section
}
FALLBACK = {("same", "lead"): "xfade2", ("other", "lead"): "xfade2",
            ("same", "direct"): "xfade1", ("other", "direct"): "xfade2"}


def _kind(route) -> tuple[str, str]:
    steps = route.get("steps") or [{"b": route.get("b"), "style": route.get("style")}]
    same = "same" if all(s["b"] == route["a"] for s in steps) else "other"
    lead = "lead" if len(steps) > 1 and steps[-1]["style"] == "natural" else "direct"
    return same, lead


def user_evidence() -> dict:
    """Per (same/other, lead/direct, style): your thumbs and best/worst picks."""
    ans = json.load(open(os.path.join(HERE, "review", "ab", "answers.json"), encoding="utf-8"))
    ev = defaultdict(lambda: {"up": 0, "down": 0, "best": 0, "worst": 0, "shown": 0})
    for a in ans:
        if a.get("kind") != "quad":
            continue
        q = a["quad"]
        for k, r in enumerate(q["options"]):
            if not r.get("steps"):
                continue
            same, lead = _kind(r)
            e = ev[f"{same}|{lead}|{r['steps'][0]['style']}"]
            e["shown"] += 1
            th = (a.get("hops") or {}).get(str(k), [])
            if th and th[0] == 1:
                e["up"] += 1
            elif th and th[0] == -1:
                e["down"] += 1
            e["best"] += a.get("best") == k
            e["worst"] += a.get("worst") == k
            if r.get("stretch"):                  # which tempo-matching method this clip used
                e2 = ev[f"stretch|{r['stretch']}"]
                e2["shown"] += 1
                e2["up"] += bool(th and th[0] == 1); e2["down"] += bool(th and th[0] == -1)
                e2["best"] += a.get("best") == k; e2["worst"] += a.get("worst") == k
    return dict(ev)


def build() -> dict:
    user = user_evidence()
    model = json.load(open(MODEL_STYLES, encoding="utf-8")) if os.path.exists(MODEL_STYLES) else {}
    choices = {}
    for same in ("same", "other"):
        for lead in ("lead", "direct"):
            rows = []
            for st in LIVE_STYLES[lead]:
                u = user.get(f"{same}|{lead}|{st}", {"up": 0, "down": 0, "best": 0, "worst": 0, "shown": 0})
                # your verdicts: thumbs rate and best-minus-worst, both shrunk toward neutral when few
                thumbs = (u["up"] + 2) / (u["up"] + u["down"] + 4)
                bw = (u["best"] - u["worst"]) / (u["shown"] + 6)
                m = model.get(f"{same}|{lead}|{st}")
                mscore = m["rank"] if m else 0.5            # model: mean within-moment rank (1 = its favourite)
                mw = min(1.0, (m["n"] if m else 0) / 40) * 0.5
                score = (1 - mw) * (thumbs + 0.5 * bw) + mw * mscore
                rows.append({"style": st, "score": round(score, 4), "thumbs": f"{u['up']}/{u['up'] + u['down']}",
                             "best": u["best"], "worst": u["worst"], "model_rank": round(mscore, 3),
                             "model_n": m["n"] if m else 0})
            rows.sort(key=lambda r: -r["score"])
            choices[f"{same}|{lead}"] = rows
    hist = os.path.join(HERE, "review", "ab", "history.json")
    metrics = {}
    if os.path.exists(hist):
        try:
            h = json.load(open(hist, encoding="utf-8"))
            last = h[-1] if isinstance(h, list) and h else {}
            metrics = {k: last.get(k) for k in ("taste_model", "critic_taste", "critic_pick_of_4", "when") if k in last}
        except Exception:
            pass
    st_rows = {}
    for meth in ("resample", "rubberband"):
        u = user.get(f"stretch|{meth}", {"up": 0, "down": 0, "best": 0, "worst": 0, "shown": 0})
        st_rows[meth] = {"score": round((u["up"] + 2) / (u["up"] + u["down"] + 4) + 0.5 * (u["best"] - u["worst"]) / (u["shown"] + 6), 4),
                         "thumbs": f"{u['up']}/{u['up'] + u['down']}"}
    # until your ratings say otherwise: resample (2026-11-20: stretched cross-song joins 2/11 with
    # pitch-safe stretching vs 11/21 with the old pitch-fader resample)
    stretch = max(st_rows, key=lambda k: st_rows[k]["score"] + (0.01 if k == "resample" else 0))
    pol = {"version": time.strftime("%Y%m%d-%H%M%S"), "built": time.time(), "choices": choices, "stretch": stretch, "stretch_evidence": st_rows,
           "model_evidence": bool(model), "metrics": metrics,
           "note": "top style per kind is used live; the runner-up gets a 20% share for variety"}
    os.makedirs(os.path.join(LIVE, "history"), exist_ok=True)
    tmp = POLICY + ".tmp"
    json.dump(pol, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, POLICY)
    shutil.copy(POLICY, os.path.join(LIVE, "history", f"policy_{pol['version']}.json"))
    return pol


# ── live side (the app): cheap, reloads when the file changes ───────────────────
_cache = {"mtime": None, "pol": None}


def current() -> dict | None:
    try:
        m = os.path.getmtime(POLICY)
    except OSError:
        return None
    if m != _cache["mtime"]:
        try:
            _cache["pol"] = json.load(open(POLICY, encoding="utf-8"))
            _cache["mtime"] = m
            print(f"[conductor] transition policy {_cache['pol'].get('version')} loaded", flush=True)
        except Exception:
            return _cache["pol"]
    return _cache["pol"]


def choose(same: bool, lead: bool, rng) -> str | None:
    """The style to use live for this kind of change, or None (no policy: keep the old behaviour)."""
    if os.environ.get("CONDUCTOR_ROUTES", "1") == "0":
        return None
    pol = current()
    key = f"{'same' if same else 'other'}|{'lead' if lead else 'direct'}"
    if not pol or key not in pol.get("choices", {}):
        return None
    rows = pol["choices"][key]
    if len(rows) > 1 and rng.random() < 0.2:
        return rows[1]["style"]
    return rows[0]["style"]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "build":
        p = build()
        print("policy", p["version"], "model evidence:", p["model_evidence"])
    pol = current() or {}
    for k, rows in pol.get("choices", {}).items():
        print(f"{k:14s}", "  ".join(f"{r['style']}({r['score']:.2f} {r['thumbs']})" for r in rows[:4]))
