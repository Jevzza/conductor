"""User-taught section roles (from the Song Map) → what the director plays where.

cache/roles/songmap_sections.json   fingerprints per section (written by songmap.py)
cache/roles/labels.json             {"sections": {id: role}, "songs": {title: role}} exported from the page

The predictor is the same as the page's: weighted nearest neighbours over sound fingerprints,
plus the song's tag and Conductor's previous belief as priors. apply(library) rewrites each
section's role/energy and each song's tier so every existing rule follows the user's taste.
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(HERE, "cache", "roles")
ROLES = ["explore", "build", "bridge", "skirmish", "horde", "peak", "skip"]
CENTER = {"explore": 0.25, "build": 0.45, "bridge": 0.42, "skirmish": 0.6, "horde": 0.76, "peak": 0.9}
TIER = {"skirmish": 1, "horde": 2, "peak": 3}


def predict(sections: list[dict], labels: dict, tags: dict) -> dict:
    lab = [s for s in sections if s["id"] in labels]
    out = {}
    for s in sections:
        if s["id"] in labels:
            out[s["id"]] = (labels[s["id"]], 1.0, True)
            continue
        v = {r: 0.0 for r in ROLES}
        v[s["belief"]] += 0.6
        if s["song"] in tags:
            v[tags[s["song"]]] += 2.0
        nn = sorted(((sum(a * b for a, b in zip(s["f"], o["f"])) + (0.08 if o["song"] == s["song"] else 0), o) for o in lab),
                    key=lambda x: -x[0])[:7]
        for c, o in nn:
            if (c > 0.25) if o["song"] == s["song"] else (c > 0.62):     # other songs only when very alike
                v[labels[o["id"]]] += math.exp((c - 1) / 0.18) * 3
        tot = sum(v.values()) or 1
        r, w = max(v.items(), key=lambda kv: kv[1])
        out[s["id"]] = (r, w / tot, False)
    return out


def apply(library) -> int:
    """Returns how many sections were (re)classified from the user's taste; 0 if no labels yet."""
    sp, lp = os.path.join(DIR, "songmap_sections.json"), os.path.join(DIR, "labels.json")
    if not (os.path.exists(sp) and os.path.exists(lp)):
        return 0
    sections = json.load(open(sp, encoding="utf-8"))
    lab = json.load(open(lp, encoding="utf-8"))
    labels, tags = lab.get("sections", {}), lab.get("songs", {})
    if not labels and not tags:
        return 0
    pred = predict(sections, labels, tags)
    by_key = {(s["song"], round(s["start"], 1)): pred[s["id"]] for s in sections}
    n = 0
    for t in library.tracks:
        tiers = []
        for seg in t.segments:
            p = by_key.get((t.title, round(seg.start, 1)))
            if not p:
                continue
            role, conf, mine = p
            seg.user_role = role
            if role == "skip":
                seg.role = "explore"
                seg.skip = True
            elif role == "bridge":
                seg.role = "explore"
                seg.bridge = True                    # a connector: passed through, never a destination
            elif role in TIER:
                seg.role = "combat"
                tiers.append(TIER[role])
            else:
                seg.role = "build" if role == "build" else "explore"
                if role == "build":
                    seg.label = "build"
            if role != "build" and seg.label == "build":
                # the analysis called it a build-up but you didn't: never treat it as one
                seg.label = {"explore": "calm", "bridge": "calm", "skip": "calm",
                             "skirmish": "groove"}.get(role, "peak")
            if role in CENTER:
                w = 0.85 if mine else 0.6           # your labels move the energy decisively
                seg.energy = w * CENTER[role] + (1 - w) * seg.energy
            n += 1
        if t.title in tags and tags[t.title] in TIER:
            t.tier = TIER[tags[t.title]]
        elif tiers:
            t.tier = max(1, round(sum(tiers) / len(tiers) + 0.25))   # what the song can mostly carry
        elif any(s.user_role in ("explore", "build") for s in t.segments if getattr(s, "user_role", None)):
            t.tier = 1
    return n
