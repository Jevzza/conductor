"""Render Darktide's OWN music transitions from its Wwise score: real segments (decoded stems),
real markers and bar grid, and the composer's 375 transition rules (sync point, fades, pre-entry,
post-exit, bridge segments). Every clip is a transition the game itself would play.
Each also gets a "wrong" twin: the same two segments spliced at a random moment with no rules.

    python tools/wwise_render.py [n_examples]
Output: cache/game_joins/{good,bad}_<k>.wav (4 s before the switch, 8 s after) + manifest.json
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

MR = os.environ.get("DARKTIDE_MODDING_RESOURCES", "")   # <Darktide>\\ModdingResources (the decoded game score)
DUMP = os.path.join(MR, "music_bnk", "music.bnk.txt")
RULES = os.path.join(MR, "music_bnk", "state_rules.json")
WAV = os.path.join(MR, "music_wav")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "game_joins")
SR = 44100
PRE, POST = 4.0, 8.0


def parse():
    L = open(DUMP, encoding="utf-8", errors="replace").read().splitlines()
    segs, tracks, ranseq, cur = {}, {}, {}, None
    for l in L:
        m = re.search(r"^(\s*)obj  (CAkMusic\w+)\[", l)
        if m and len(m.group(1)) <= 20:
            cur = {"type": m.group(2), "children": [], "clips": [], "markers": [], "random": False}
            continue
        if cur is None:
            continue
        g = lambda k: float(re.search(k + r" = (-?[\d.e+-]+)", l).group(1))
        if "sid  ulID = " in l and "id" not in cur:
            cur["id"] = int(re.search(r"ulID = (\d+)", l).group(1))
            {"CAkMusicSegment": segs, "CAkMusicTrack": tracks, "CAkMusicRanSeqCntr": ranseq}.get(cur["type"], {})[cur["id"]] = cur
        elif "ulChildID = " in l:
            cur["children"].append(int(re.search(r"ulChildID = (\d+)", l).group(1)))
        elif "fTempo = " in l and "tempo" not in cur:
            cur["tempo"] = g("fTempo")
        elif "uTimeSigNumBeatsBar = " in l and "bpb" not in cur:
            cur["bpb"] = int(g("uTimeSigNumBeatsBar"))
        elif "fGridPeriod = " in l and "grid" not in cur:
            cur["grid"] = g("fGridPeriod")
        elif "fDuration = " in l:
            cur["dur"] = g("fDuration")
        elif "fPosition = " in l:
            cur["markers"].append(g("fPosition"))
        elif "eTrackType = " in l:
            cur["random"] = "[Random]" in l
        elif "trackID = " in l:
            cur["clips"].append({"sub": int(g("trackID"))})
        elif cur["clips"] and "sourceID = " in l and "src" not in cur["clips"][-1]:
            cur["clips"][-1]["src"] = int(re.search(r"sourceID = (\d+)", l).group(1))
        elif cur["clips"] and "fPlayAt = " in l:
            cur["clips"][-1]["at"] = g("fPlayAt")
        elif cur["clips"] and "fBeginTrimOffset = " in l:
            cur["clips"][-1]["b"] = g("fBeginTrimOffset")
        elif cur["clips"] and "fEndTrimOffset = " in l:
            cur["clips"][-1]["e"] = g("fEndTrimOffset")
        elif cur["clips"] and "fSrcDuration = " in l:
            cur["clips"][-1]["d"] = g("fSrcDuration")
    return segs, tracks, ranseq


_wav = {}


def _source(sid):
    if sid not in _wav:
        p = os.path.join(WAV, f"{sid}.wav")
        if not os.path.exists(p):
            _wav[sid] = None
        else:
            y, sr = sf.read(p, dtype="float32", always_2d=True)
            if y.shape[1] == 1:
                y = np.repeat(y, 2, 1)
            y = y[:, :2]
            if sr != SR:
                import librosa
                y = librosa.resample(y.T, orig_sr=sr, target_sr=SR).T.astype(np.float32)
            _wav[sid] = y
    return _wav[sid]


def render_segment(seg, tracks, rng):
    """(audio, index of segment time 0) - every track's clips placed on the segment timeline."""
    pieces = []
    for tid in seg["children"]:
        t = tracks.get(tid)
        if not t:
            continue
        clips = [c for c in t["clips"] if "src" in c and "at" in c]
        if t["random"] and clips:
            sub = rng.choice(sorted({c["sub"] for c in clips}))
            clips = [c for c in clips if c["sub"] == sub]
        for c in clips:
            y = _source(c["src"])
            if y is None:
                continue
            b, e, d = c.get("b", 0.0), c.get("e", 0.0), c.get("d", len(y) / SR * 1000)
            a0, a1 = int(b / 1000 * SR), int((d + e) / 1000 * SR)
            pieces.append((int((c["at"] + b) / 1000 * SR), y[a0:max(a0, a1)]))
    if not pieces:
        return None, 0
    lo = min(0, min(p for p, _ in pieces))
    hi = max(int(seg.get("dur", 0) / 1000 * SR), max(p + len(a) for p, a in pieces))
    out = np.zeros((hi - lo, 2), np.float32)
    for p, a in pieces:
        out[p - lo:p - lo + len(a)] += a
    return out, -lo


def leaf_segments(node, segs, ranseq, depth=0):
    if node in segs:
        return [node]
    if node in ranseq and depth < 6:
        out = []
        for c in ranseq[node]["children"]:
            out += leaf_segments(c, segs, ranseq, depth + 1)
        return out
    return []


def sync_time(seg, kind, t, rng):
    """Segment time (ms) where the switch happens, waiting from t."""
    entry = min(seg["markers"]) if seg["markers"] else 0.0
    exit_ = max(seg["markers"]) if seg["markers"] else seg.get("dur", 0.0)
    beat = 60000.0 / (seg.get("tempo") or 120.0)
    bar = beat * (seg.get("bpb") or 4)
    step = {"NextBar": bar, "NextBeat": beat, "NextGrid": seg.get("grid") or bar}.get(kind)
    if kind == "ExitMarker":
        return exit_
    if kind in ("NextMarker", "NextUserMarker"):
        nxt = [m for m in sorted(seg["markers"]) if m > t]
        return nxt[0] if nxt else exit_
    if step:
        k = int(np.ceil((t - entry) / step))
        return min(exit_, entry + k * step)
    return t                                    # Immediate


def main(n=600, seed=11):
    rng = random.Random(seed)
    segs, tracks, ranseq = parse()
    rules = [r for r in json.load(open(RULES)) if r["src_rule"] and r["dst_rule"]]
    # switch container children (any state) for "ANY"
    by_switch = defaultdict(set)
    for r in rules:
        for side in ("src_id", "dst_id"):
            if r[side] not in (-1, None):
                by_switch[r["switch"]].add(r[side])
    os.makedirs(OUT, exist_ok=True)
    man, k, tries = [], 0, 0
    while k < n and tries < n * 20:
        tries += 1
        r = rng.choice(rules)
        pool = sorted(by_switch[r["switch"]])
        sn = r["src_id"] if r["src_id"] not in (-1, None) else (rng.choice(pool) if pool else None)
        dn = r["dst_id"] if r["dst_id"] not in (-1, None) else (rng.choice(pool) if pool else None)
        if sn is None or dn is None or sn == dn:
            continue
        ss, ds = leaf_segments(sn, segs, ranseq), leaf_segments(dn, segs, ranseq)
        if not ss or not ds:
            continue
        S, Dg = segs[rng.choice(ss)], segs[rng.choice(ds)]
        A, a0 = render_segment(S, tracks, rng)
        B, b0 = render_segment(Dg, tracks, rng)
        if A is None or B is None:
            continue
        sr_, dr_ = r["src_rule"], r["dst_rule"]
        entryA = min(S["markers"]) if S["markers"] else 0.0
        exitA = max(S["markers"]) if S["markers"] else S.get("dur", 0.0)
        if exitA - entryA < 3000:
            continue
        t = rng.uniform(entryA + 2000, exitA)
        sw = sync_time(S, sr_.get("eSyncType"), t, rng)                      # ms, in A's segment time
        swi = a0 + int(sw / 1000 * SR)
        if swi < int(PRE * SR):
            continue
        # destination: from its entry marker (or a random marker), full level unless it fades in
        entryB = min(Dg["markers"]) if Dg["markers"] else 0.0
        if dr_.get("eEntryType") in ("RandomMarker", "RandomUserMarker") and len(Dg["markers"]) > 2:
            entryB = rng.choice(sorted(Dg["markers"])[:-1])
        ebi = b0 + int(entryB / 1000 * SR)
        n_pre, n_post = int(PRE * SR), int(POST * SR)
        fo, off = sr_.get("transitionTime", 0) / 1000, sr_.get("iFadeOffset", 0) / 1000
        fi = dr_.get("transitionTime", 0) / 1000
        beat = 60.0 / (S.get("tempo") or 120.0)
        bar = beat * (S.get("bpb") or 4)

        def make(shift=0.0, fade="rule", b_shift=0.0):
            """The game's transition, optionally broken: switch `shift` s off the grid, the wrong
            fade, or the new part entered `b_shift` s off its marker."""
            sw_ = swi + int(shift * SR)
            if sw_ < n_pre or sw_ + n_post > len(A) + n_post:
                return None
            clip = np.zeros((n_pre + n_post, 2), np.float32)
            a = A[sw_ - n_pre:sw_ + n_post]
            tt = (np.arange(len(a)) - n_pre) / SR
            f_o = {"rule": fo, "cut": 0.0, "long": max(3.0, 3 * fo)}[fade]
            if f_o > 0:
                g = np.clip(1 - (tt - (-off if fade == "rule" else 0.0)) / f_o, 0, 1)
            else:
                g = (tt < 0).astype(np.float32)
            clip[:len(a)] += a * g[:, None]
            eb = ebi + int(b_shift * SR)
            pre_b = min(eb if (dr_.get("bPlayPreEntry") and fade == "rule") else 0, eb, n_pre)
            bseg = B[eb - pre_b:eb + n_post]
            f_i = {"rule": fi, "cut": 0.0, "long": max(3.0, 3 * fi)}[fade]
            gb = np.ones(len(bseg), np.float32)
            if f_i > 0:
                tb = (np.arange(len(bseg)) - pre_b) / SR
                gb = np.clip((tb + pre_b / SR) / (f_i + pre_b / SR), 0, 1)
            clip[n_pre - pre_b:n_pre - pre_b + len(bseg)] += bseg * gb[:, None]
            return clip

        clip = make()
        if clip is None or np.abs(clip).max() < 1e-3:
            continue
        rms = lambda x: 20 * np.log10(np.sqrt(np.mean(x.mean(1) ** 2)) + 1e-9)
        if rms(clip[n_pre - int(2.5 * SR):n_pre]) < -48 or rms(clip[n_pre + int(0.2 * SR):n_pre + int(3 * SR)]) < -48:
            continue                            # (near) silence on one side: nothing to learn
        norm = lambda c: c / max(1.0, float(np.abs(c).max()) / 0.95)
        # the "wrong" twin: same two segments, cut at a random moment off the grid, no rules
        cut = a0 + int(rng.uniform(entryA + 1000, exitA - 500) / 1000 * SR) + rng.randint(-4000, 4000)
        bin_ = b0 + int(rng.uniform(entryB, max(entryB + 1, (Dg.get('dur') or entryB) - POST * 1000)) / 1000 * SR)
        bad = np.zeros_like(clip)
        aa = A[max(0, cut - n_pre):cut]
        bad[n_pre - len(aa):n_pre] = aa
        bb = B[bin_:bin_ + n_post]
        bad[n_pre:n_pre + len(bb)] = bb
        # 2026-11-20 (user): a ladder from the game's own rule down to a random splice, so the critic
        # learns degrees of right, not just right/wrong
        ladder = [make(),                                                   # 0 the game's rule
                  make(shift=rng.choice((-1, 1)) * beat),                    # 1 a beat early/late
                  make(fade=rng.choice(("cut", "long"))),                    # 2 the wrong fade
                  make(shift=rng.choice((-1, 1)) * bar * 0.5),               # 3 half a bar off the grid
                  make(b_shift=rng.uniform(0.3, 0.7) * bar),                 # 4 new part entered mid-phrase
                  bad]                                                      # 5 random splice
        if any(c is None for c in ladder):
            continue
        sf.write(os.path.join(OUT, f"good_{k:04d}.wav"), norm(clip), SR)
        sf.write(os.path.join(OUT, f"bad_{k:04d}.wav"), norm(bad), SR)
        for sev, c in enumerate(ladder):
            sf.write(os.path.join(OUT, f"rank_{k:04d}_{sev}.wav"), norm(c), SR)
        man.append({"k": k, "switch": r["switch"], "src": str(r["src"])[:120], "dst": str(r["dst"])[:120],
                    "sync": sr_.get("eSyncType"), "fade_out": fo, "fade_in": fi, "entry": dr_.get("eEntryType"),
                    "bridge": bool(r.get("link")), "seam": PRE})
        k += 1
        if k % 50 == 0:
            print(k, "rendered", flush=True)
            json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=0)
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=0)
    print("done", k, "game transitions (+", k, "wrong twins)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 600)
