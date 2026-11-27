"""Turn review ratings into labelled joins: for each rated moment, find the music changes heard in
it (from that round's saved timeline) and what was left / entered.

    joins_from_reviews(lib) -> [{"a","t_out","b","t_in","good","moment","mission","round"}]
"""
from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEW = os.path.join(HERE, "review")


def _find(lib, name):
    for t in lib.tracks:
        if t.title == name or t.title in name or name in t.title:
            return t


def _at(pos, t):
    """(song, position) heard at time t, interpolated from the 1-per-second position log."""
    prev = None
    for row in pos:
        if row[0] > t:
            break
        prev = row
    if prev is None:
        return None
    return prev[1], prev[2] + (t - prev[0])


def joins_in(tl, lib, a, b):
    out = []
    for e in tl.get("music", []):
        if e["text"][:1] not in ("→", "S") or not (a <= e["t"] <= b):
            continue
        t = e["t"]
        before, after = _at(tl["pos"], t - 0.6), None
        for row in tl["pos"]:
            if row[0] >= t + 0.6:
                after = (row[1], row[2] - (row[0] - t))
                break
        if not before or not after:
            continue
        A, B = _find(lib, before[0]), _find(lib, after[0])
        if A and B:
            out.append({"a": A.title, "t_out": before[1] + 0.6, "b": B.title, "t_in": max(0.0, after[1]), "heard": t})
    return out


def joins_from_reviews(lib) -> list[dict]:
    rows = []
    for mdir in sorted(glob.glob(os.path.join(REVIEW, "m*"))):
        pins_p = os.path.join(mdir, "pins.json")
        if not os.path.exists(pins_p):
            continue
        pins = [p for p in json.load(open(pins_p, encoding="utf-8")) if p.get("moment") and "rating" in p]
        for d in sorted({p["dir"] for p in pins}):
            tlp = os.path.join(mdir, f"timeline_{d}.json")
            if not os.path.exists(tlp):
                continue
            tl = json.load(open(tlp, encoding="utf-8"))
            for p in (x for x in pins if x["dir"] == d):
                js = joins_in(tl, lib, p["t"] - 2, p["t"] + 14)
                if p.get("heard_at") is not None and p["rating"] < 0:
                    # you paused right after the bad one: blame the last join before that point
                    before = [j for j in js if j["heard"] <= p["heard_at"] + 0.5]
                    js = before[-1:] or js
                for j in js:
                    rows.append({**j, "good": int(p["rating"] > 0), "moment": p["t"],
                                 "mission": os.path.basename(mdir), "round": d})
    return rows
