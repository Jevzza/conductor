"""Merge the GPU pass (ml_pass.py) into a song's analysis: real beats/downbeats, section
boundaries on bar lines, drum-stem beat strength, and per-section entry/exit fade lengths.
Pure numpy — runs in the app's normal Python."""
from __future__ import annotations

import json
import os

import numpy as np

ML_TAG = "beat_this+demucs:1"
ALLIN1_TAG = "+allin1:1"
STEM_HZ = 10           # stems_db rate
ONSET_HZ = 50          # drum_onset rate
MIN_SEG = 6.0


def load_ml(path: str) -> dict | None:
    from .ml_pass import ml_path
    p = ml_path(path)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_allin1(path: str) -> dict | None:
    from .allin1_pass import json_path
    p = json_path(path)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def wanted_tag(path: str) -> str:
    return ML_TAG + (ALLIN1_TAG if load_allin1(path) is not None else "")


def _win(arr: np.ndarray, hz: float, a: float, b: float) -> np.ndarray:
    i, j = max(0, int(a * hz)), max(0, int(b * hz))
    return arr[i:max(i + 1, j)] if i < len(arr) else arr[-1:]


def _mix_db(stems: dict) -> np.ndarray:
    lin = sum(10 ** (np.asarray(v) / 10) for v in stems.values())
    return 10 * np.log10(lin + 1e-12)


def entry_exit(seg_start: float, seg_end: float, stems: dict, drum_onset: np.ndarray) -> tuple[float, float]:
    """How a section wants to be entered and left, in beats.
    entry: 0.25 = hard (starts on an impact), 1 = normal, 4 = soft (no drums), 8 = ambient
    exit:  1 = dense (cut the tail short), 4 = normal, 8 = sparse/ambient (let it ring)"""
    mix = _mix_db(stems)
    drums = np.asarray(stems["drums"])
    s, e = seg_start, seg_end
    pre = _win(mix, STEM_HZ, s - 0.5, s).mean()
    post = _win(mix, STEM_HZ, s, s + 0.5).mean()
    drums_on = _win(drums, STEM_HZ, s, s + 1.0).mean() > _win(mix, STEM_HZ, s, s + 1.0).mean() - 12
    seg_drums = (_win(drums, STEM_HZ, s, e) > _win(mix, STEM_HZ, s, e) - 12).mean()
    peak = _win(drum_onset, ONSET_HZ, s - 0.05, s + 0.1).max()
    if post - pre >= 4 or (drums_on and peak >= 0.6):
        entry = 0.25
    elif not drums_on and seg_drums < 0.2:
        entry = 8.0
    elif not drums_on:
        entry = 4.0
    else:
        entry = 1.0
    tail = (_win(drums, STEM_HZ, e - 4, e) > _win(mix, STEM_HZ, e - 4, e) - 12).mean()
    exit_ = 1.0 if tail >= 0.6 else (4.0 if tail >= 0.2 else 8.0)
    return entry, exit_


def apply(data: dict, ml: dict) -> dict:
    """Rewrite beats, tempo and section boundaries from the ML pass; add drum curves and
    per-section entry/exit. Loops/hits/jumps must be recomputed afterwards (they depend on beats)."""
    beats = [float(b) for b in ml["beats"]]
    downs = [float(b) for b in ml["downbeats"]]
    if len(beats) < 8 or len(downs) < 4:
        return data
    data["beats"] = beats
    data["downbeats"] = downs
    data["tempo"] = round(60.0 / float(np.median(np.diff(beats))), 2)
    dur = data["duration"]
    # section boundaries onto the nearest bar line
    bar = float(np.median(np.diff(downs)))
    dn = np.asarray(downs)
    a1 = load_allin1(data["path"])
    if a1 is not None:      # phrase-level sections from All-In-One replace the clustering
        src = [{"start": float(x["start"]), "label": x["label"]} for x in a1["segments"]
               if x["label"] not in ("start", "end") or float(x["start"]) > 0]
    else:
        src = data["segments"]
    labels = {}
    cuts = [0.0]
    for sd in src:
        t = sd["start"]
        if t <= 0.5:
            continue
        k = int(np.argmin(np.abs(dn - t)))
        if abs(dn[k] - t) <= 0.75 * bar:
            t = float(dn[k])
        if t - cuts[-1] >= MIN_SEG and dur - t >= MIN_SEG:
            cuts.append(t)
            labels[t] = sd.get("label")
    if a1 is not None and a1["segments"]:
        labels[0.0] = a1["segments"][0]["label"]
    data["segments"] = [{"start": round(a, 4), "end": round(b, 4), "struct": labels.get(a)}
                        for a, b in zip(cuts, cuts[1:] + [dur])]
    # drum curves per second (for the "driving beat" score)
    stems = ml["stems_db"]
    onset = np.asarray(ml["drum_onset"], dtype=float)
    n = len(data["curve"]["rms_db"])
    drums = np.asarray(stems["drums"])
    mix = _mix_db(stems)
    pulse, presence = [], []
    for i in range(n):
        o = _win(onset, ONSET_HZ, i, i + 1)
        pulse.append(round(float(o.mean()), 4))
        presence.append(round(float((_win(drums, STEM_HZ, i, i + 1) > _win(mix, STEM_HZ, i, i + 1) - 12).mean()), 3))
    data["curve"]["drum_pulse"] = pulse
    data["curve"]["drum_presence"] = presence
    for sd in data["segments"]:
        sd["entry_beats"], sd["exit_beats"] = entry_exit(sd["start"], sd["end"], stems, onset)
    data["ml"] = ML_TAG + (ALLIN1_TAG if a1 is not None else "")
    return data
