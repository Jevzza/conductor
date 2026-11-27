"""Darktide's interactive music score (wwiser dump of the Wwise v141 music bank) -> labelled
training datasets for Conductor. Reads only game-derived files under ModdingResources; writes only
cache/game_data/.

    python tools/wwise_datasets.py [--no-audio] [--joins N]

Outputs (cache/game_data/):
  segments.json        every segment: timing, meter (inherited if not overridden), markers, sources,
                       every game-state path that reaches it + derived intensity labels
  segments/<id>.wav    each segment rendered (seed 7, 44.1 kHz stereo; silent ones skipped)
  playlists.json       every RanSeq container: playlist tree, own transition rules, sequence_pairs
  sequence_joins/      composer A->B playlist transitions rendered (good_k / bad_k twin) + manifest.json
  stingers.json        stinger definitions (the main bank has none; see 'note')
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wwise_render as WR  # noqa: E402  (reuse the source loader / SR)

MR = WR.MR
DUMP = WR.DUMP
NAMES = os.path.join(MR, "music_bnk", "names.json")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "game_data")
# extra state names found by FNV-1 guessing (not in names.json)
EXTRA_NAMES = {s: None for s in (
    "credits defeat expeditions extraction game_score game_score_lose gauntlet_event horde_mode hub_event "
    "kill_event_2 kill_event_3 last_man_standing mission_loading ogryn operation_stage_1 operation_stage_2 "
    "operation_stage_3 operation_stage_4 prologue safe_room scanning_event twins_event zone_twins zone_6 zone_7 "
    "zone_8 shooting_range music_character_create character_creation adamant broker psyker veteran zealot").split()}


def fnv1(s):
    h = 2166136261
    for b in s.lower().encode():
        h = (h * 16777619) & 0xffffffff
        h ^= b
    return h


COMBAT_VALUES = ["none", "normal", "horde_low", "horde_high", "boss"]
SR = WR.SR
PRE, POST = 4.0, 8.0

# ----------------------------------------------------------------------------- dump -> tree
_LINE = re.compile(r"^(\w+)\s+(\w+)(?:\[(\d+)\])?(?: = (.*))?$")


def parse_tree(path=DUMP):
    """wwiser txt dump -> nested dicts {"t","n","v","lab","k":[children]} using column indentation."""
    root = {"t": "root", "n": "root", "k": []}
    stack = [(-1, root)]
    for raw in open(path, encoding="utf-8", errors="replace"):
        raw = raw.rstrip("\n")
        if re.match(r"^[0-9a-f]{8}", raw):
            raw = " " * 8 + raw[8:]
        s = raw.lstrip(" ")
        if not s:
            continue
        col = len(raw) - len(s)
        m = _LINE.match(s)
        if not m:
            continue
        t, n, idx, v = m.groups()
        lab = None
        if v is not None:
            mm = re.match(r"^(.*?) \[(.*)\]$", v)
            if mm:
                v, lab = mm.groups()
        node = {"t": t, "n": n, "v": v, "lab": lab, "k": []}
        while stack[-1][0] >= col:
            stack.pop()
        stack[-1][1]["k"].append(node)
        stack.append((col, node))
    return root


def val(node, name, default=None, conv=None):
    """first descendant field `name` (breadth-first, so own fields win over nested ones)."""
    q = list(node["k"])
    while q:
        c = q.pop(0)
        if c["n"] == name and c["v"] is not None:
            v = c["v"]
            if conv is None:
                return v
            try:
                return conv(v)
            except ValueError:
                return conv(float(v)) if conv is int else default
        q.extend(c["k"])
    return default


def lab(node, name):
    q = list(node["k"])
    while q:
        c = q.pop(0)
        if c["n"] == name:
            return c["lab"]
        q.extend(c["k"])
    return None


def kids(node, name):
    return [c for c in node["k"] if c["n"] == name]


def walk(node, name):
    out = []
    for c in node["k"]:
        if c["n"] == name:
            out.append(c)
        out.extend(walk(c, name))
    return out


def ival(v):
    return int(v, 0) if isinstance(v, str) and v.startswith("0x") else int(float(v))


# ----------------------------------------------------------------------------- objects
def meter(o):
    m = walk(o, "AkMeterInfo")
    if not m:
        return None
    m = m[0]
    return {"tempo": val(m, "fTempo", None, float), "bpb": val(m, "uTimeSigNumBeatsBar", None, ival),
            "beat_value": val(m, "uTimeSigBeatValue", None, ival), "grid": val(m, "fGridPeriod", None, float),
            "grid_offset": val(m, "fGridOffset", 0.0, float), "override": ival(val(o, "bMeterInfoFlag", "0"))}


def children_ids(o):
    ch = walk(o, "Children")
    return [ival(c["v"]) for c in walk(ch[0], "ulChildID")] if ch else []


def parse_rules(o):
    rules = []
    for r in walk(o, "AkMusicTransitionRule"):
        src = walk(r, "AkMusicTransSrcRule")[0]
        dst = walk(r, "AkMusicTransDstRule")[0]

        def side(x, keys):
            d = {}
            for k in keys:
                for c in x["k"]:
                    if c["n"] == k:
                        d[k] = c["lab"] if c["lab"] and not c["lab"].isdigit() and k.startswith("e") else ival(c["v"])
            return d
        rule = {"src_ids": [ival(c["v"]) for c in walk(r, "srcID")],
                "dst_ids": [ival(c["v"]) for c in walk(r, "dstID")],
                "src": side(src, ["transitionTime", "eFadeCurve", "iFadeOffset", "eSyncType", "uCueFilterHash", "bPlayPostExit"]),
                "dst": side(dst, ["transitionTime", "eFadeCurve", "iFadeOffset", "uCueFilterHash", "uJumpToID",
                                  "eJumpToType", "eEntryType", "bPlayPreEntry", "bDestMatchSourceCueName"]),
                "bridge": None}
        if ival(val(r, "AllocTransObjectFlag", "0")):
            to = walk(r, "AkMusicTransitionObject")[0]
            fo, fi = walk(to, "fadeOutParams"), walk(to, "fadeInParams")
            rule["bridge"] = {"segmentID": ival(val(to, "segmentID", "0")),
                              "fade_out": {"transitionTime": val(fo[0], "transitionTime", 0, ival), "iFadeOffset": val(fo[0], "iFadeOffset", 0, ival)} if fo else None,
                              "fade_in": {"transitionTime": val(fi[0], "transitionTime", 0, ival), "iFadeOffset": val(fi[0], "iFadeOffset", 0, ival)} if fi else None,
                              "bPlayPreEntry": val(to, "bPlayPreEntry", 0, ival), "bPlayPostExit": val(to, "bPlayPostExit", 0, ival)}
        rules.append(rule)
    return rules


def parse_playlist_item(it):
    d = {"segment": ival(val(it, "SegmentID", "0")), "item_id": ival(val(it, "playlistItemID", "0")),
         "rs_type": lab(it, "eRSType") or "None", "loop": val(it, "Loop", 1, ival),
         "loop_min": val(it, "LoopMin", 0, ival), "loop_max": val(it, "LoopMax", 0, ival),
         "weight": val(it, "Weight", 50000, ival) / 1000.0,
         "avoid_repeat": val(it, "wAvoidRepeatCount", 0, ival),
         "using_weight": val(it, "bIsUsingWeight", 0, ival), "shuffle": val(it, "bIsShuffle", 0, ival),
         "children": []}
    for lst in kids(it, "pPlayList"):
        for c in lst["k"]:
            if c["n"] == "AkMusicRanSeqPlaylistItem":
                d["children"].append(parse_playlist_item(c))
    if not d["children"]:
        d.pop("children")
        d.pop("rs_type") if d["rs_type"] == "None" else None
    else:
        d.pop("segment") if d["segment"] == 0 else None
    return d


def load(path=DUMP):
    root = parse_tree(path)
    objs = {}
    q = [root]
    while q:
        c = q.pop()
        if c["n"].startswith("CAk") and c["t"] == "obj":
            oid = val(c, "ulID", None, ival)
            if oid is not None:
                objs[oid] = (c["n"], c)
            continue
        q.extend(c["k"])
    return root, objs


def build(objs):
    segs, tracks, ranseq, switch, other = {}, {}, {}, {}, {}
    for oid, (typ, o) in objs.items():
        base = {"id": oid, "parent": val(o, "DirectParentID", None, ival)}
        if typ == "CAkMusicSegment":
            mk = [{"id": val(m, "id", 0, ival), "pos": val(m, "fPosition", 0.0, float), "name": val(m, "pMarkerName", "") or ""}
                  for m in walk(o, "AkMusicMarkerWwise")]
            segs[oid] = {**base, "children": children_ids(o), "meter": meter(o), "dur": val(o, "fDuration", 0.0, float),
                         "markers_full": sorted(mk, key=lambda m: m["pos"]), "markers": sorted(m["pos"] for m in mk),
                         "stingers": walk(o, "CAkStinger") + walk(o, "AkStinger")}
        elif typ == "CAkMusicTrack":
            clips = []
            for c in walk(o, "AkTrackSrcInfo"):
                clips.append({"sub": val(c, "trackID", 0, ival), "src": val(c, "sourceID", 0, ival),
                              "at": val(c, "fPlayAt", 0.0, float), "b": val(c, "fBeginTrimOffset", 0.0, float),
                              "e": val(c, "fEndTrimOffset", 0.0, float), "d": val(c, "fSrcDuration", 0.0, float)})
            autos = []
            for a in walk(o, "AkClipAutomation"):
                pts = [(val(p, "From", 0.0, float), val(p, "To", 0.0, float)) for p in walk(a, "AkRTPCGraphPoint")]
                autos.append({"clip": val(a, "uClipIndex", 0, ival), "type": lab(a, "eAutoType"), "points": pts})
            tracks[oid] = {**base, "clips": clips, "auto": autos, "random": lab(o, "eTrackType") == "Random",
                           "track_type": lab(o, "eTrackType"), "n_sub": val(o, "numSubTrack", 1, ival),
                           "sources": [ival(s["v"]) for s in walk(o, "sourceID")]}
        elif typ == "CAkMusicRanSeqCntr":
            items = []
            for lst in walk(o, "pPlayList")[:1]:
                items = [parse_playlist_item(c) for c in lst["k"] if c["n"] == "AkMusicRanSeqPlaylistItem"]
            ranseq[oid] = {**base, "children": children_ids(o), "meter": meter(o), "rules": parse_rules(o),
                           "playlist": items, "stingers": walk(o, "CAkStinger") + walk(o, "AkStinger")}
        elif typ == "CAkMusicSwitchCntr":
            args = [ival(c["v"]) for c in walk(walk(o, "Arguments")[0], "ulGroup")] if walk(o, "Arguments") else []
            gtypes = [c["lab"] for c in walk(walk(o, "Arguments")[0], "eGroupType")] if walk(o, "Arguments") else []
            leaves = []

            def rec(node, path):
                for c in node["k"]:
                    if c["n"] == "Node":
                        key = val_own(c, "key")
                        an = val_own(c, "audioNodeId")
                        p = path + [key]
                        sub = [x for x in c["k"] if x["n"] == "Node" or x["n"] == "pNodes" or x["t"] == "lst"]
                        if an is not None and not any(x["n"] == "Node" for x in walk(c, "Node")):
                            leaves.append((p, an))
                        else:
                            rec(c, p)
                    elif c["t"] == "lst":
                        rec(c, path)
            dt = walk(o, "AkDecisionTree")
            if dt:
                rec(dt[0], [])
            # first level is the tree root (key 0 / no arg)
            leaves = [(p[1:], an) for p, an in leaves]
            if not args:   # 0-argument switch: tree is a bare root (audioNodeId = 1); it plays its child
                leaves = [([], c) for c in children_ids(o)]
            switch[oid] = {**base, "children": children_ids(o), "meter": meter(o), "rules": parse_rules(o),
                           "args": args, "group_types": gtypes, "leaves": leaves, "mode": lab(o, "uMode"),
                           "continue": val(o, "bIsContinuePlayback", 0, ival),
                           "stingers": walk(o, "CAkStinger") + walk(o, "AkStinger")}
        else:
            other[oid] = {**base, "type": typ}
    return segs, tracks, ranseq, switch, other


def val_own(node, name):
    for c in node["k"]:
        if c["n"] == name and c["v"] is not None:
            return ival(c["v"])
    return None


# ----------------------------------------------------------------------------- meter inheritance
def resolve_meter(oid, allm):
    seen = 0
    cur = oid
    while cur in allm and seen < 10:
        o = allm[cur]
        m = o.get("meter")
        if m and m["override"]:
            return m, cur
        cur = o.get("parent")
        seen += 1
    m = allm[oid].get("meter") or {}
    return m, None


# ----------------------------------------------------------------------------- state labels
def label_states(segs, ranseq, switch, names):
    nm = lambda h: names.get(h, str(h))
    vname = lambda h: "*" if h == 0 else names.get(h, str(h))
    referenced = set()
    for s in switch.values():
        referenced |= {an for _, an in s["leaves"]} | set(s["children"])
    for r in ranseq.values():
        referenced |= set(r["children"])
    roots = [sid for sid in switch if sid not in referenced]
    seg_paths = defaultdict(list)       # seg -> list of (path dict, chain)
    unreached_children = []

    def rec(node, path, chain, depth=0):
        if depth > 12:
            return
        if node in segs:
            seg_paths[node].append((dict(path), chain + [node]))
        elif node in ranseq:
            for c in ranseq[node]["children"]:
                rec(c, path, chain + [node], depth + 1)
        elif node in switch:
            sw = switch[node]
            hit = set()
            argn = [nm(g) for g in sw["args"]]
            ci = argn.index("music_combat") if "music_combat" in argn else None
            for keys, an in sw["leaves"]:
                p = dict(path)
                for g, k in zip(sw["args"], keys):
                    p[nm(g)] = vname(k)
                if ci is not None:
                    if keys[ci] == 0:
                        # BestMatch: '*' wins for every combat value no sibling leaf names explicitly
                        others = keys[:ci] + keys[ci + 1:]
                        named = {vname(k2[ci]) for k2, _ in sw["leaves"] if k2[:ci] + k2[ci + 1:] == others and k2[ci] != 0}
                        p["_combat_covers"] = [v for v in COMBAT_VALUES if v not in named]
                    else:
                        p["_combat_covers"] = [vname(keys[ci])]
                hit.add(an)
                rec(an, p, chain + [node], depth + 1)
            for c in sw["children"]:
                if c not in hit:
                    unreached_children.append({"switch": node, "child": c})
    for r in roots:
        rec(r, {}, [])
    return seg_paths, roots, unreached_children


COMBAT = {"none": "calm", "normal": "fight", "horde_low": "horde", "horde_high": "big_horde", "boss": "boss"}


def intensity_of(p):
    """one label per state path. Mission paths: the combat values the leaf actually wins under
    BestMatch ('*' = whatever siblings don't name). '*' winning both none+normal -> calm_fight."""
    gs = p.get("music_game_state")
    if gs in (None, "*", "mission"):
        cov = p.get("_combat_covers")
        if cov == []:
            return "shadowed"          # every combat value is named by a sibling leaf: never plays
        if cov is None:
            return "unknown" if gs is None else "mission_any"
        labs = [COMBAT[c] for c in cov]
        if set(cov) == set(COMBAT_VALUES):
            # same music whatever the combat state: objective/event music, or a generic mission bed
            return "event" if p.get("music_objective") not in (None, "*", "none") else "mission_any"
        if set(cov) >= {"none", "normal"}:
            return "calm_fight" if len(cov) == 2 else "mission_any"
        return labs[0] if len(labs) == 1 else "mission_any"
    return gs


# ----------------------------------------------------------------------------- playlists -> pairs
def item_graph(it):
    """(firsts, lasts, pairs) of segment ids for a playlist item. pairs: (a, b, kind, item_id)"""
    if "children" not in it:
        s = it["segment"]
        pairs = [(s, s, "loop", it["item_id"])] if it["loop"] != 1 else []
        return {s}, {s}, pairs
    subs = [item_graph(c) for c in it["children"]]
    pairs = [p for _, _, ps in subs for p in ps]
    n = len(subs)
    t = it["rs_type"]
    looping = it["loop"] != 1
    if t == "ContinuousSequence":
        F, Lst = subs[0][0], subs[-1][1]
        for i in range(n - 1):
            pairs += [(a, b, "sequence", it["item_id"]) for a in subs[i][1] for b in subs[i + 1][0]]
        if looping:
            pairs += [(a, b, "loop", it["item_id"]) for a in subs[-1][1] for b in subs[0][0]]
        return F, Lst, pairs
    F = set().union(*[s[0] for s in subs]) if subs else set()
    Lst = set().union(*[s[1] for s in subs]) if subs else set()
    if t == "ContinuousRandom" or (looping and t.startswith("Step")):
        for i in range(n):
            for j in range(n):
                if i == j and n > 1 and (it["avoid_repeat"] or it["shuffle"] or t == "StepSequence"):
                    continue
                if t == "StepSequence" and j != (i + 1) % n:
                    continue
                pairs += [(a, b, "random" if "Random" in t else "step", it["item_id"]) for a in subs[i][1] for b in subs[j][0]]
    return F, Lst, pairs


def pick_rule(rules, a, b, cid, ranseq, switch):
    """most specific transition rule for a->b in container cid (walk up parents if it has none)."""
    cur = cid
    for _ in range(8):
        c = ranseq.get(cur) or switch.get(cur)
        if c is None:
            break
        best, score = None, -1
        for r in c["rules"]:
            sm = a in r["src_ids"] or cur in r["src_ids"]
            dm = b in r["dst_ids"] or cur in r["dst_ids"]
            sa, da = -1 in r["src_ids"], -1 in r["dst_ids"]
            if (sm or sa) and (dm or da):
                sc = 2 * sm + dm
                if sc > score:
                    best, score = r, sc
        if best:
            return best, cur
        cur = c.get("parent")
    return None, None


# ----------------------------------------------------------------------------- sources
EXTRA_WAV = os.path.join(OUT, "small_events", "wav")
_extra = {}


def load_source(sid):
    y = WR._source(sid)
    if y is not None:
        return y
    if sid not in _extra:
        p = os.path.join(EXTRA_WAV, f"{sid}.wav")
        _extra[sid] = None
        if os.path.exists(p):
            y, sr = sf.read(p, dtype="float32", always_2d=True)
            y = np.repeat(y, 2, 1) if y.shape[1] == 1 else y[:, :2]
            if sr != SR:
                import librosa
                y = librosa.resample(y.T, orig_sr=sr, target_sr=SR).T.astype(np.float32)
            _extra[sid] = y
    return _extra[sid]


# ----------------------------------------------------------------------------- small music events
EVENTS = os.path.join(MR, "ww_events")
BUNDLE = os.path.join(os.path.dirname(MR), "bundle")
AUDIO_OUT = os.path.join(MR, "audio_out")
VGM = os.path.join(MR, "vgmstream", "vgmstream-cli.exe")
WWISER = os.path.join(MR, "wwiser_dl", "wwiser.pyz")
MAIN_EVENT = "df73bd0ecb2fa1e6"


def murmur64(data, seed=0):
    m, r, M = 0xc6a4a7935bd1e995, 47, 0xFFFFFFFFFFFFFFFF
    h = (seed ^ (len(data) * m)) & M
    i = 0
    while i + 8 <= len(data):
        k = (int.from_bytes(data[i:i + 8], "little") * m) & M
        k ^= k >> r
        k = (k * m) & M
        h = ((h ^ k) * m) & M
        i += 8
    if len(data) > i:
        h = ((h ^ int.from_bytes(data[i:], "little")) * m) & M
    h ^= h >> r
    h = (h * m) & M
    return h ^ (h >> r)


def music_event_files():
    """wwise_event files whose HIRC holds music objects (segment/track/switch/ranseq)."""
    import struct
    out = []
    for f in sorted(os.listdir(EVENTS)):
        b = open(os.path.join(EVENTS, f), "rb").read()
        i = b.find(b"BKHD")
        j = b.find(b"HIRC", max(i, 0))
        if i < 0 or j < 0:
            continue
        n, p, mus = struct.unpack_from("<I", b, j + 8)[0], j + 12, 0
        for _ in range(n):
            if p + 5 > len(b):
                break
            mus += b[p] in (0x0A, 0x0B, 0x0C, 0x0D)
            p += 5 + struct.unpack_from("<I", b, p + 1)[0]
        if mus:
            out.append(f[:-len(".wwise_event")])
    return out


def decode_event_bank(name, outdir):
    """strip wrapper before BKHD + trailing bytes, write xorpad.bin, run wwiser -> dump path."""
    import struct
    import subprocess
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    txt = os.path.join(d, name + ".bnk.txt")
    if os.path.exists(txt):
        return txt
    b = open(os.path.join(EVENTS, name + ".wwise_event"), "rb").read()
    b = b[b.find(b"BKHD"):]
    p = 0
    while p + 8 <= len(b):
        sz = struct.unpack_from("<I", b, p + 4)[0]
        if not b[p:p + 4].isalpha() or p + 8 + sz > len(b):
            break
        p += 8 + sz
    b = b[:p]
    ver, _, lang, alt = struct.unpack_from("<IIII", b, 8)
    open(os.path.join(d, name + ".bnk"), "wb").write(b)
    open(os.path.join(d, "xorpad.bin"), "wb").write(bytes(8) + struct.pack("<IIII", ver ^ 141, 0, lang, alt))
    subprocess.run([sys.executable, "-I", WWISER, "-d", "txt", name + ".bnk"], cwd=d, capture_output=True, timeout=300)
    return txt


def decode_source(sid):
    """game stream wwise/<sid> -> small_events/wav/<sid>.wav (vgmstream on a temp copy)."""
    import shutil
    import subprocess
    import tempfile
    out = os.path.join(EXTRA_WAV, f"{sid}.wav")
    if os.path.exists(out) or os.path.exists(os.path.join(WR.WAV, f"{sid}.wav")):
        return True
    ptr = os.path.join(AUDIO_OUT, "%016x.wwise_stream" % murmur64(f"wwise/{sid}".encode()))
    if not os.path.exists(ptr):
        return False
    m = re.search(rb"data/([0-9a-f]{2}/[0-9a-f]+[.]stream)", open(ptr, "rb").read())
    if not m:
        return False
    src = os.path.join(BUNDLE, "data", m.group(1).decode().replace("/", os.sep))
    if not os.path.exists(src):
        return False
    os.makedirs(EXTRA_WAV, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ti, to = os.path.join(tmp, "in.lwav"), os.path.join(tmp, "out.wav")
        shutil.copy2(src, ti)
        subprocess.run([VGM, "-o", to, ti], capture_output=True, timeout=60)
        if os.path.exists(to):
            shutil.move(to, out)
            return True
    return False


def decode_bank_media(bnk_path, want):
    """in-bank (Data/bnk) media of the given source ids -> EXTRA_WAV via vgmstream. Returns decoded ids."""
    import struct
    import subprocess
    import tempfile
    b = open(bnk_path, "rb").read()
    p, chunks = 0, {}
    while p + 8 <= len(b):
        tag, sz = b[p:p + 4], struct.unpack_from("<I", b, p + 4)[0]
        chunks[tag] = (p + 8, sz)
        p += 8 + sz
    if b"DIDX" not in chunks or b"DATA" not in chunks:
        return []
    d0, dsz = chunks[b"DIDX"]
    data0 = chunks[b"DATA"][0]
    got = []
    os.makedirs(EXTRA_WAV, exist_ok=True)
    for i in range(dsz // 12):
        sid, off, sz = struct.unpack_from("<III", b, d0 + 12 * i)
        if sid not in want:
            continue
        out = os.path.join(EXTRA_WAV, f"{sid}.wav")
        if not os.path.exists(out):
            with tempfile.TemporaryDirectory() as tmp:
                ti = os.path.join(tmp, f"{sid}.wem")
                open(ti, "wb").write(b[data0 + off:data0 + off + sz])
                subprocess.run([VGM, "-o", out, ti], capture_output=True, timeout=60)
        if os.path.exists(out):
            got.append(sid)
    return got


def small_events(names, do_audio):
    """catalogue (and render) the music objects of every small music wwise_event."""
    sd = os.path.join(OUT, "small_events")
    stingers, events = [], []
    for ev in music_event_files():
        if ev == MAIN_EVENT:
            continue
        txt = decode_event_bank(ev, sd)
        if not os.path.exists(txt):
            events.append({"file": ev, "error": "wwiser failed"})
            continue
        _, objs = load(txt)
        segs, tracks, ranseq, switch, other = build(objs)
        evs = [k for k, o in other.items() if o["type"] == "CAkEvent"]
        acts = [{"id": k, "target": val(objs[k][1], "idExt", None, ival), "type": lab(objs[k][1], "ulActionType")}
                for k, o in other.items() if o["type"].startswith("CAkAction")]
        srcs = sorted({c["src"] for t in tracks.values() for c in t["clips"]})
        ok = {s: decode_source(s) for s in srcs} if do_audio else {}
        rec = {"file": ev, "event_ids": evs, "actions": acts, "sources": srcs,
               "sources_decoded": sorted(s for s, v in ok.items() if v), "segments": [], "ranseq": sorted(ranseq),
               "switches": [{"id": k, "args": [names.get(a, str(a)) for a in v["args"]],
                             "leaves": [{"path": [names.get(x, "*" if x == 0 else str(x)) for x in p], "node": n} for p, n in v["leaves"]]}
                            for k, v in switch.items()]}
        for sid, sg in segs.items():
            info = {"id": sid, "duration_ms": round(sg["dur"], 3), "markers_ms": [round(x, 3) for x in sg["markers"]],
                    "meter": sg["meter"], "has_audio": False}
            if do_audio:
                y, _ = render_segment(sg, tracks, random.Random(7))
                if y is not None and dbfs(y) > -60:
                    os.makedirs(os.path.join(sd, "segments"), exist_ok=True)
                    write_wav(os.path.join(sd, "segments", f"{sid}.wav"), y)
                    info.update(has_audio=True, rms_dbfs=round(dbfs(y), 1), render_len_s=round(len(y) / SR, 3))
            rec["segments"].append(info)
        for oid, o in list(segs.items()) + list(ranseq.items()) + list(switch.items()):
            for st in o["stingers"]:
                d = {"file": ev, "owner": oid, "trigger_id": val_own(st, "TriggerID"), "segment_id": val_own(st, "SegmentID"),
                     "sync_play_at": lab(st, "SyncPlayAt"), "cue_filter_hash": val_own(st, "uCueFilterHash"),
                     "dont_repeat_ms": val_own(st, "DontRepeatTime"), "segment_lookahead": val_own(st, "numSegmentLookAhead"),
                     "has_audio": False}
                sg = segs.get(d["segment_id"])
                if sg is not None:
                    d["duration_ms"] = round(sg["dur"], 3)
                    if do_audio:
                        y, _ = render_segment(sg, tracks, random.Random(7))
                        if y is not None and dbfs(y) > -60:
                            os.makedirs(os.path.join(OUT, "stingers"), exist_ok=True)
                            write_wav(os.path.join(OUT, "stingers", f"{d['segment_id']}.wav"), y)
                            d["has_audio"] = True
                stingers.append(d)
        events.append(rec)
    return stingers, events


# ----------------------------------------------------------------------------- rendering
def _automation_gain(t, clip_len, autos):
    """per-sample gain for a clip from its FadeIn / FadeOut / Volume automation (linear interp)."""
    g = np.ones(clip_len, np.float32)
    ts = np.arange(clip_len) / SR
    for a in autos:
        pts = a["points"]
        if not pts:
            continue
        xs = np.array([p[0] for p in pts], np.float32)
        ys = np.array([p[1] for p in pts], np.float32)
        if a["type"] == "FadeIn":
            g *= np.interp(ts, xs, ys, left=ys[0], right=ys[-1])
        elif a["type"] == "FadeOut":
            # x measured from the fade-out start; the fade ends at the clip end
            span = xs[-1] - xs[0]
            x0 = clip_len / SR - span
            g *= np.interp(ts - x0, xs - xs[0], ys, left=ys[0], right=ys[-1])
    return g


def render_segment(seg, tracks, rng):
    """like wwise_render.render_segment, plus clip fade automation. -> (audio, index of time 0)"""
    pieces = []
    for tid in seg["children"]:
        t = tracks.get(tid)
        if not t:
            continue
        clips = [(i, c) for i, c in enumerate(t["clips"])]
        if t["random"] and clips:
            sub = rng.choice(sorted({c["sub"] for _, c in clips}))
            clips = [(i, c) for i, c in clips if c["sub"] == sub]
        for i, c in clips:
            y = load_source(c["src"])
            if y is None:
                continue
            b, e, d = c["b"], c["e"], c["d"] or len(y) / SR * 1000
            a0, a1 = int(b / 1000 * SR), int((d + e) / 1000 * SR)
            a = y[a0:max(a0, min(a1, len(y)))]
            au = [x for x in t["auto"] if x["clip"] == i and x["type"] in ("FadeIn", "FadeOut")]
            if au and len(a):
                a = a * _automation_gain(0, len(a), au)[:, None]
            pieces.append((int((c["at"] + b) / 1000 * SR), a))
    if not pieces:
        return None, 0
    lo = min(0, min(p for p, _ in pieces))
    hi = max(int(seg.get("dur", 0) / 1000 * SR), max(p + len(a) for p, a in pieces))
    out = np.zeros((hi - lo, 2), np.float32)
    for p, a in pieces:
        out[p - lo:p - lo + len(a)] += a
    return out, -lo


def dbfs(x):
    return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


def write_wav(path, y):
    y = y / max(1.0, float(np.abs(y).max()) / 0.95)
    sf.write(path, y, SR, subtype="PCM_16")


def sync_time(seg, m, kind, t):
    entry, exit_ = seg["markers"][0] if seg["markers"] else 0.0, seg["markers"][-1] if seg["markers"] else seg["dur"]
    beat = 60000.0 / (m.get("tempo") or 120.0)
    bar = beat * (m.get("bpb") or 4)
    step = {"NextBar": bar, "NextBeat": beat, "NextGrid": m.get("grid") or bar}.get(kind)
    if kind == "ExitMarker":
        return exit_
    if kind in ("NextMarker", "NextUserMarker"):
        nxt = [x for x in seg["markers"] if x > t]
        return nxt[0] if nxt else exit_
    if step:
        return min(exit_, entry + int(np.ceil((t - entry) / step)) * step)
    return t


def render_join(A, a0, Sa, ma, B, b0, Sb, rule, rng):
    """composer transition A->B per rule (switch at A's sync point); returns clip, info or None."""
    srcr, dstr = rule["src"] if rule else {}, rule["dst"] if rule else {}
    kind = srcr.get("eSyncType", "ExitMarker")
    entryA, exitA = Sa["markers"][0], Sa["markers"][-1]
    if kind == "ExitMarker":
        sw = exitA
    else:
        sw = sync_time(Sa, ma, kind, rng.uniform(entryA + 2000, max(entryA + 2001, exitA)))
    swi = a0 + int(sw / 1000 * SR)
    n_pre, n_post = int(PRE * SR), int(POST * SR)
    if swi < n_pre:
        return None, None
    clip = np.zeros((n_pre + n_post, 2), np.float32)
    a = A[swi - n_pre:swi + n_post]
    tt = (np.arange(len(a)) - n_pre) / SR
    fo, off = srcr.get("transitionTime", 0) / 1000, srcr.get("iFadeOffset", 0) / 1000
    if fo > 0:
        g = np.clip(1 - (tt + off) / fo, 0, 1)
    elif srcr.get("bPlayPostExit"):
        g = np.ones(len(a), np.float32)      # post-exit tail rings out (A's audio past its exit marker)
    else:
        g = (tt < 0).astype(np.float32)
    if kind != "ExitMarker" and not fo and srcr.get("bPlayPostExit"):
        # mid-segment switch: only audio past the exit marker is a tail; cut A's body at the switch
        g = (tt < 0).astype(np.float32)
    clip[:len(a)] += a * g[:, None]
    entryB = Sb["markers"][0] if Sb["markers"] else 0.0
    if dstr.get("eEntryType") in ("RandomMarker", "RandomUserMarker") and len(Sb["markers"]) > 2:
        entryB = rng.choice(Sb["markers"][:-1])
    ebi = b0 + int(entryB / 1000 * SR)
    pre_b = min(ebi, n_pre) if dstr.get("bPlayPreEntry", 1) else 0
    bseg = B[ebi - pre_b:ebi + n_post]
    fi = dstr.get("transitionTime", 0) / 1000
    gb = np.ones(len(bseg), np.float32)
    if fi > 0:
        tb = (np.arange(len(bseg)) - pre_b) / SR
        gb = np.clip((tb + pre_b / SR) / (fi + pre_b / SR), 0, 1)
    clip[n_pre - pre_b:n_pre - pre_b + len(bseg)] += bseg * gb[:, None]
    return clip, {"switch_ms": sw, "entry_ms": entryB, "pre_entry_s": pre_b / SR, "sync": kind,
                  "fade_out_s": fo, "fade_in_s": fi, "post_exit": bool(srcr.get("bPlayPostExit"))}


# ----------------------------------------------------------------------------- main
def main(argv):
    do_audio = "--no-audio" not in argv
    n_joins = int(argv[argv.index("--joins") + 1]) if "--joins" in argv else 400
    names = {int(k): v for k, v in json.load(open(NAMES)).items()}
    names.update({fnv1(s): s for s in EXTRA_NAMES})
    nm = lambda h: names.get(h, str(h))
    os.makedirs(OUT, exist_ok=True)
    root, objs = load()
    segs, tracks, ranseq, switch, other = build(objs)
    allm = {**segs, **ranseq, **switch}
    print(f"{len(segs)} segments, {len(tracks)} tracks, {len(ranseq)} ranseq, {len(switch)} switch, other {Counter(o['type'] for o in other.values())}")

    # ---- labels
    seg_paths, roots, unreached = label_states(segs, ranseq, switch, names)
    playlist_of = defaultdict(set)
    for rid, r in ranseq.items():
        for c in r["children"]:
            playlist_of[c].add(rid)

    seg_out = []
    for sid, s in sorted(segs.items()):
        m, msrc = resolve_meter(sid, allm)
        paths = [p for p, _ in seg_paths.get(sid, [])]
        uniq = []
        for p in paths:
            if p not in uniq:
                uniq.append(p)
        votes = Counter(intensity_of(p) for p in uniq)
        votes.pop("shadowed", None)
        ints = sorted(votes) or ["unknown"]
        specific = {k: v for k, v in votes.items() if k not in ("mission_any", "event")} or votes
        top = max(specific.values()) if specific else 0
        best = [k for k, v in specific.items() if v == top]
        primary = best[0] if len(best) == 1 else ("shared" if best else "unknown")   # tie: used across states
        events = sorted({p[k] for p in uniq for k in ("music_objective", "event_category", "music_objective_progression") if k in p and p[k] != "*"})
        exp = sorted({p["expeditions_intensity"] for p in uniq if p.get("expeditions_intensity") not in (None, "*")})
        zones = sorted({p["music_zone"] for p in uniq if p.get("music_zone") not in (None, "*")})
        srcs = sorted({c["src"] for t in s["children"] if t in tracks for c in tracks[t]["clips"]})
        seg_out.append({
            "id": sid, "duration_ms": round(s["dur"], 3),
            "tempo": m.get("tempo"), "beats_per_bar": m.get("bpb"), "beat_value": m.get("beat_value"),
            "grid_ms": m.get("grid"), "grid_offset_ms": m.get("grid_offset"), "meter_from": msrc,
            "markers_ms": [round(x, 3) for x in s["markers"]],
            "entry_ms": round(s["markers"][0], 3) if s["markers"] else 0.0,
            "exit_ms": round(s["markers"][-1], 3) if s["markers"] else round(s["dur"], 3),
            "cues": [{"ms": round(x["pos"], 3), "name": x["name"], "id": x["id"]} for x in s["markers_full"][1:-1]],
            "tracks": [{"id": t, "type": tracks[t]["track_type"], "n_sub": tracks[t]["n_sub"],
                        "sources": sorted({c["src"] for c in tracks[t]["clips"]})} for t in s["children"] if t in tracks],
            "source_ids": srcs, "parent": s["parent"], "playlists": sorted(playlist_of.get(sid, [])),
            "state_paths": uniq, "intensities": ints,
            "intensity": ints[0] if len(ints) == 1 else "multi:" + "|".join(ints),
            "intensity_votes": dict(votes), "intensity_primary": primary,
            "objective_labels": events, "expeditions_intensity": exp, "zones": zones,
            "n_paths": len(paths), "reachable": bool(uniq), "has_audio": False, "rms_dbfs": None,
        })
    by_id = {s["id"]: s for s in seg_out}

    # ---- playlists + sequence pairs
    pl_out, all_pairs = [], []
    for rid, r in sorted(ranseq.items()):
        pairs = []
        for it in r["playlist"]:
            _, _, ps = item_graph(it)
            pairs += ps
        seen = set()
        sp = []
        for a, b, kind, item in pairs:
            if (a, b, kind) in seen or a not in segs or b not in segs:
                continue
            seen.add((a, b, kind))
            rule, rfrom = pick_rule(r["rules"], a, b, rid, ranseq, switch) if r["rules"] else pick_rule([], a, b, r["parent"], ranseq, switch)
            sp.append({"a": a, "b": b, "kind": kind, "item": item, "rule_from": rfrom, "rule": rule})
        all_pairs += [dict(p, playlist=rid) for p in sp]
        m, _ = resolve_meter(rid, allm)
        pl_out.append({"id": rid, "parent": r["parent"], "children": r["children"], "meter": m,
                       "play_mode": sorted({x for x in json.dumps(r["playlist"]).split('"') if x in ("ContinuousSequence", "ContinuousRandom", "StepSequence", "StepRandom")}),
                       "playlist": r["playlist"], "rules": r["rules"], "sequence_pairs": sp,
                       "intensities": sorted({i for c in r["children"] if c in by_id for i in by_id[c]["intensities"]})})
    json.dump(pl_out, open(os.path.join(OUT, "playlists.json"), "w"), indent=1)

    # ---- switch containers (for reference)
    sw_out = [{"id": k, "parent": v["parent"], "args": [nm(a) for a in v["args"]], "mode": v["mode"],
               "continue": v["continue"], "root": k in roots,
               "leaves": [{"path": {nm(g): ("*" if x == 0 else nm(x)) for g, x in zip(v["args"], p)}, "node": an,
                           "node_type": "segment" if an in segs else "ranseq" if an in ranseq else "switch" if an in switch else "missing"}
                          for p, an in v["leaves"]],
               "rules": v["rules"]} for k, v in sorted(switch.items())]
    json.dump(sw_out, open(os.path.join(OUT, "switches.json"), "w"), indent=1)

    # ---- stingers
    st = [{"owner": k, "trigger_id": val_own(x, "TriggerID"), "segment_id": val_own(x, "SegmentID")}
          for k, o in list(segs.items()) + list(ranseq.items()) + list(switch.items()) for x in o["stingers"]]
    ev_st, ev_rec = small_events(names, do_audio) if "--no-events" not in argv else ([], [])
    if ev_rec:
        json.dump(ev_rec, open(os.path.join(OUT, "small_events.json"), "w"), indent=1)
    json.dump({"stingers": ev_st, "main_bank_stingers": st,
               "note": "The main score (df73bd0ecb2fa1e6) has NumStingers = 0 on every node; the stingers listed come "
                       "from the small music events (see small_events.json for their segments and events)."},
              open(os.path.join(OUT, "stingers.json"), "w"), indent=1)
    print(len(ev_st), "stingers;", len(ev_rec), "small music events;",
          sum(sg["has_audio"] for e in ev_rec for sg in e.get("segments", [])), "of",
          sum(len(e.get("segments", [])) for e in ev_rec), "small-event segments rendered")

    # ---- audio
    if do_audio:
        srcs = {c["src"] for t in tracks.values() for c in t["clips"]}
        miss = {s for s in srcs if not os.path.exists(os.path.join(WR.WAV, f"{s}.wav"))}
        got = set(decode_bank_media(os.path.join(MR, "music_bnk", "music.bnk"), miss))
        srcdur = {c["src"]: c["d"] for t in tracks.values() for c in t["clips"]}
        for s in list(got):      # prefetch-streamed sources only keep a head in the bank: drop those
            info = sf.info(os.path.join(EXTRA_WAV, f"{s}.wav"))
            if info.frames / info.samplerate * 1000 < 0.9 * srcdur.get(s, 0):
                os.remove(os.path.join(EXTRA_WAV, f"{s}.wav"))
                got.discard(s)
        got |= {s for s in miss - got if decode_source(s)}
        print(f"{len(srcs)} sources, {len(miss)} not in music_wav, recovered {sorted(got)}, still missing {sorted(miss - got)}")
        sd = os.path.join(OUT, "segments")
        os.makedirs(sd, exist_ok=True)
        for f in os.listdir(sd):
            if f.endswith(".wav"):
                os.remove(os.path.join(sd, f))
        for s in seg_out:
            S = segs[s["id"]]
            y, z = render_segment(S, tracks, random.Random(7))
            if y is None or dbfs(y) < -60:
                s["rms_dbfs"] = None if y is None else round(dbfs(y), 1)
                continue
            s["has_audio"] = True
            s["rms_dbfs"] = round(dbfs(y), 1)
            s["render_t0_s"] = round(z / SR, 4)      # file index of segment time 0 (pre-entry before it)
            s["render_len_s"] = round(len(y) / SR, 3)
            write_wav(os.path.join(sd, f"{s['id']}.wav"), y)
        print(sum(s["has_audio"] for s in seg_out), "segments rendered")
    json.dump(seg_out, open(os.path.join(OUT, "segments.json"), "w"), indent=1)

    # ---- sequence joins
    if do_audio:
        jd = os.path.join(OUT, "sequence_joins")
        os.makedirs(jd, exist_ok=True)
        rng = random.Random(11)
        cand = [p for p in all_pairs if by_id[p["a"]]["has_audio"] and by_id[p["b"]]["has_audio"]]
        # prefer distinct A->B transitions (not self-loops) first, then loops
        rng.shuffle(cand)
        for f in os.listdir(jd):
            if f.endswith(".wav"):
                os.remove(os.path.join(jd, f))
        cand.sort(key=lambda p: p["a"] == p["b"])
        man, k = [], 0
        has_rand = lambda sid: any(tracks[t]["random"] for t in segs[sid]["children"] if t in tracks)
        # round 0 = every pair once; later rounds re-render pairs whose random tracks give new audio
        work = [(0, p) for p in cand] + [(v, p) for v in (1, 2) for p in cand if has_rand(p["a"]) or has_rand(p["b"])]
        for variant, p in work:
            if k >= n_joins:
                break
            Sa, Sb = segs[p["a"]], segs[p["b"]]
            ma, _ = resolve_meter(p["a"], allm)
            if not Sa["markers"] or Sa["markers"][-1] - Sa["markers"][0] < 1000:
                continue
            rr = random.Random(hash((p["a"], p["b"], variant)) & 0xffffffff)
            A, a0 = render_segment(Sa, tracks, rr)
            B, b0 = render_segment(Sb, tracks, rr)
            clip, info = render_join(A, a0, Sa, ma, B, b0, Sb, p["rule"], rr)
            if clip is None:
                continue
            n_pre = int(PRE * SR)
            if dbfs(clip[n_pre - int(2.5 * SR):n_pre]) < -48 or dbfs(clip[n_pre + int(0.2 * SR):n_pre + int(3 * SR)]) < -48:
                continue
            write_wav(os.path.join(jd, f"good_{k:04d}.wav"), clip)
            # wrong twin: same segments, random off-grid cut, B from a random point, no rules
            entryA, exitA = Sa["markers"][0], Sa["markers"][-1]
            cut = a0 + int(rr.uniform(entryA + 1000, max(entryA + 1001, exitA - 500)) / 1000 * SR) + rr.randint(-4000, 4000)
            entryB = Sb["markers"][0] if Sb["markers"] else 0.0
            bin_ = b0 + int(rr.uniform(entryB, max(entryB + 1, Sb["dur"] - POST * 1000)) / 1000 * SR)
            bad = np.zeros_like(clip)
            aa = A[max(0, cut - n_pre):cut]
            bad[n_pre - len(aa):n_pre] = aa
            bb = B[bin_:bin_ + int(POST * SR)]
            bad[n_pre:n_pre + len(bb)] = bb
            write_wav(os.path.join(jd, f"bad_{k:04d}.wav"), bad)
            r = p["rule"] or {}
            man.append({"k": k, "playlist": p["playlist"], "a": p["a"], "b": p["b"], "kind": p["kind"],
                        "a_intensity": by_id[p["a"]]["intensity"], "b_intensity": by_id[p["b"]]["intensity"],
                        "rule_from": p["rule_from"], "rule_src": r.get("src"), "rule_dst": r.get("dst"),
                        "bridge": r.get("bridge"), "variant": variant, "seam_s": PRE, **info})
            k += 1
        json.dump(man, open(os.path.join(jd, "manifest.json"), "w"), indent=1)
        print(k, "sequence joins rendered (+ wrong twins) from", len(cand), "candidate pairs")

    # ---- summary
    print("roots", roots, "unreached switch children", len(unreached))
    print("intensity:", Counter(s["intensity"] for s in seg_out).most_common())
    print("pairs:", len(all_pairs), Counter(p["kind"] for p in all_pairs))
    json.dump({"roots": roots, "unreached_switch_children": unreached,
               "unresolved_hashes": sorted({str(a) for v in switch.values() for a in v["args"] if a not in names} |
                                           {str(x) for v in switch.values() for p, _ in v["leaves"] for x in p if x and x not in names})},
              open(os.path.join(OUT, "unresolved.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
