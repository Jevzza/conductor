"""Transition routes: a change is not just "fade A into B". It can travel through other places
first (a build-up in the target song, the current song's own build, a breakdown) and each hop can
use its own joining style, including layering one section over another.

    route = {"a": title, "t_out": s, "steps": [{"b": title, "t_in": s, "bars": n|None, "style": str}, ...],
             "b": final title, "t_in": final start}     # b/t_in mirror the last step (old code reads them)

    candidates(lib, j, state, rng) -> [route]     other ways to make the change at moment j
    render(lib_by, route, out_mp3) -> [seam seconds]
    features(lib_by, route) -> dict               numbers for the taste model
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

SR = 44100
PRE, POST = 4.0, 6.0   # 2026-11-19: 4 s lead-in (was 6) - more picks per minute

# how one piece of music hands over to the next at a seam (the seam is always the new part's bar line)
STYLES = {
    "cut":     "hard cut on the bar line",
    "punch":   "cut on the beat, old part rings one beat underneath",
    "xfade1":  "equal-power crossfade over one bar",
    "xfade2":  "equal-power crossfade over two bars",
    "xfade4":  "equal-power crossfade over four bars (slow blend between songs)",
    "filter":  "old part sweeps down through a low-pass over the bar before, new part drops in",
    "echo":    "old part's last beat echoes out under the new part",
    "drop":    "old part stops a beat early - one beat of silence - new part slams in",
    "riser":   "the target song's build-up is layered in under the old part for two bars, then cut",
    "natural": "the song's own flow (the build leads into this part in the original song)",
    "dj":      "DJ swap: old song stays at full volume, the new song's build rises underneath with its bass cut, "
               "then the old song stops dead on the new song's drop",
    "djverb":  "DJ swap, but on the drop the old song's last moment washes out in a ~2 s reverb tail "
               "under the new song instead of stopping dead",
    "djblend": "DJ swap, but the old song plays on past the new song's drop for a bar or two (bass cut, "
               "fading out) so the two blend before the old one is gone",
    "game":    "Darktide's own rules (its Wwise score, 2026-11-20): change on the bar, the new part enters at "
               "full level; going up the old part rings out over 1 s, going down it fades over 5 s, between "
               "combat tiers the two crossfade over 2.5 s",
    "mix":     "DJ mix: old song tempo-matched to the new one, its mids/highs fade out while the new song's "
               "build rises in without bass; the basses swap on the new song's drop",
}
# 2026-11-24 signposted joins (user: in game every change is a bang - "build-up and bang" with no
# lead-in, jarring over the game's noise). Built AFTER the switch point like "dj" (the live player
# only builds audio from the switch on): the old song carries on, beat-matched, while the new song
# is announced, then the new song lands N bars later. Kept out of STYLES so the trees' features
# don't change shape. step["t_in"] = where the announcing starts (in the new song's time); it lands
# LEAD_STYLES[style] bars later.
LEAD_STYLES = {
    "tease2": 2, "tease4": 4,   # the new song's drums (no kick/bass) creep in under the old song
    "strip2": 2, "strip4": 4,   # the old song thins out (bass/kick go, top dips), the new song's riser bridges
    "fill1": 1,                 # one bar: the old song loses its low end, a drum fill announces the drop
    "strip4s": 4,               # strip4, but the old song snaps back to full on the last beat (user, set 12 C)
    "hook4": 4,                 # tease4 + the new song's hook (its most repeated melodic phrase) floats in
    # 2026-11-24 layered blends (user: "it's not a fade of A to B... still a bang"): the old song's last
    # bars play over the new song's FIRST bars (outro over intro), swapped layer by layer - drums, then
    # bass, then melody - so B is fully in by the end and simply carries on. step t_in = B's section start.
    "layer4": 4, "layer8": 8,
    "layer4d": 4,               # + "D": B's hook floats in over A first, A's hook trails over B at the end
    # 2026-11-24 drop swap (user's idea; = the DJ "bass swap" / Vande Veire & De Bie's DnB double drop):
    # A keeps its drums + bass, A's melody fades while B's BUILD-UP melody comes in over it, and the
    # drums + bass swap in one hit on B's drop. step t_in = where B's build starts; lands on the drop.
    "swap4": 4, "swap8": 8,
    "swap4d": 4, "swap8d": 8,   # ...and A's bass leaves halfway: A's drums under B's melody before the drop
}                               # over the last 2 bars - "an iconic bit sprinkled in before the transition"
# not a lead-in: lands like a 1-bar blend, then the OLD song's hook trails over the new song's first bars
AFTER_STYLES = ("hookout",)
LEAD_NAMES = {"strip4s": "Strip + snap-back (4 bars)", "hook4": "Hook in (4 bars)", "hookout": "Hook trails out",
              "tease2": "Tease (2 bars)", "tease4": "Tease (4 bars)", "strip2": "Strip + riser (2 bars)",
              "strip4": "Strip + riser (4 bars)", "fill1": "Drum fill (1 bar)"}
LIVE_STYLES = ("cut", "punch", "xfade1", "xfade2", "filter", "echo", "riser", "game")   # "drop" removed: its beat of silence reads as the music dying
UP_ORDER = {"explore": 0, "bridge": 0, "build": 1, "skirmish": 2, "horde": 3, "peak": 4}


def as_route(j: dict) -> dict:
    """Old-style join {a,t_out,b,t_in} -> a one-step route with the style the director used."""
    if "steps" in j:
        return j
    return {**j, "steps": [{"b": j["b"], "t_in": j["t_in"], "bars": None, "style": j.get("style", "auto")}]}


def _role(track, t):
    from .states import role_of
    return role_of(track.segment_at(t + 0.05)) or ""


def _auto(A, t_out, B, t_in) -> str:
    return "punch" if UP_ORDER.get(_role(B, t_in), 2) >= UP_ORDER.get(_role(A, t_out - 0.1), 2) else "xfade1"


# ── candidates ──────────────────────────────────────────────────────────────
def _builds_into(track, seg):
    """(build start for its last 2 bars) when `seg` is preceded by a build/bridge in its song."""
    from .states import role_of
    segs = track.segments
    k = segs.index(seg)
    if k == 0 or role_of(segs[k - 1]) not in ("build", "bridge"):
        return None
    bars = [b for b in (track.downbeats or track.beats[::4]) if segs[k - 1].start - 0.1 <= b < seg.start - 0.5]
    return bars[-2] if len(bars) >= 2 else (bars[-1] if bars else None)


def _own_builds(track, after: float):
    from .states import role_of
    return [s for s in track.segments if role_of(s) in ("build", "bridge") and s.end - s.start >= 4]


def candidates(lib, j: dict, state: str, rng) -> list[dict]:
    """Other ways the music could make this change: same destination with each joining style,
    via the destination's own build-up (then its natural drop), via the current song's build-up,
    and other fitting parts / songs."""
    from .states import POOL, StateDirector, role_of, serves
    by = {t.title: t for t in lib.tracks}
    A, B = by[j["a"]], by[j["b"]]
    out = []
    dests = [(B, j["t_in"])]
    for T in [B] + rng.sample([t for t in lib.tracks if t is not B], 3):
        segs = [s for s in T.segments if serves(s, state) and s.end - s.start >= 6]
        for s in rng.sample(segs, min(2, len(segs))):
            t = StateDirector._land(T, s)
            if T is not B or abs(t - j["t_in"]) > 2:
                dests.append((T, t))
    for T, t in dests:
        base = {"a": j["a"], "t_out": j["t_out"], "b": T.title, "t_in": t}
        for st in LIVE_STYLES:
            if st in ("cut", "filter") and rng.random() < 0.8:
                continue                                     # lost 0/6 and 1/8: sample them rarely
            if st == "riser" and _builds_into(T, T.segment_at(t + 0.05)) is None:
                continue
            out.append({**base, "steps": [{"b": T.title, "t_in": t, "bars": None, "style": st}]})
        seg = T.segment_at(t + 0.05)
        bi = _builds_into(T, seg)
        if bi is not None:                                   # hop into its build-up, ride the song's own drop
            for st in ("xfade2", "xfade1", "echo"):
                out.append({**base, "steps": [{"b": T.title, "t_in": bi, "bars": None, "style": st, "until": t},
                                              {"b": T.title, "t_in": t, "bars": None, "style": "natural"}]})
        if UP_ORDER.get(_role(T, t), 2) - UP_ORDER.get(_role(A, j["t_out"] - 0.1), 2) >= 2:
            for s in _own_builds(A, j["t_out"])[:3]:         # big jump: climb through this song's build first
                bars = [b for b in (A.downbeats or A.beats[::4]) if s.start - 0.1 <= b < s.end - 1]
                if len(bars) >= 3:
                    st1 = rng.choice(("xfade2", "xfade1", "echo"))
                    st2 = rng.choice(("cut", "punch", "echo", "riser" if _builds_into(T, seg) is not None else "cut"))
                    out.append({**base, "steps": [{"b": A.title, "t_in": bars[-3], "bars": 2, "style": st1},
                                                  {"b": T.title, "t_in": t, "bars": None, "style": st2}]})
    return out


ROLE_RANK = {"explore": 0, "bridge": 0, "build": 1, "skirmish": 2, "horde": 3, "peak": 4}
STATE_RANK = {"calm": 0, "fight": 2, "horde": 3, "boss": 4}


def natural_routes(lib, j: dict, state: str, down: bool, rng) -> list[dict]:
    """The song's own move: enter a passage that the song itself carries into a part for `state`
    (up: a build-up into its drop; down: an intense part into the calmer part after it), entered
    1, 2 or 4 bars before the boundary with an easing style, then the song plays on."""
    from .states import POOL, role_of, serves
    by = {t.title: t for t in lib.tracks}
    A = by[j["a"]]
    out = []
    # songs that suit each other blend best: near tempo (or half/double) and near key
    from .joins import _fifths
    pa = 60.0 / (A.tempo or 120)

    def fit(T):
        r = np.log2((60.0 / (T.tempo or 120)) / pa)
        k = _fifths(A, T)
        return abs(r - np.round(r)) * 10 + (0.5 if np.isnan(k) else k / 6) + rng.random() * 0.4
    others = sorted([t for t in lib.tracks if t is not A], key=fit)
    for T in [A] + others[:4] + rng.sample(others[4:], min(2, len(others[4:]))):
        segs = T.segments
        for k, s in enumerate(segs[:-1]):
            nxt, r = segs[k + 1], role_of(s)
            if not down and r == "bridge":
                continue                                     # going up: through build-ups only
            src = (ROLE_RANK.get(r, 0) > STATE_RANK[state] and r not in ("build", "bridge")) if down else r in ("build", "bridge")
            if not src or not serves(nxt, state) or nxt.end - nxt.start < 6:
                continue
            bars = [b for b in (T.downbeats or T.beats[::4]) if s.start - 0.1 <= b < nxt.start - 0.5]
            for nb in (1, 2, 4):
                if len(bars) < nb:
                    continue
                lo = min(T.energy_at(t) for t in np.arange(bars[-nb], nxt.start + 3, 0.5))
                if lo < 0.25 * T.mean_energy:
                    continue                                 # the song goes (near) silent in there
                # into another song: echo lost 1/9, 1-bar 0/4, 2-bar 6/12 -> try 2- and 4-bar blends
                # user 2026-11-19: long fades sound cheap; keep A up and swap on B's drop like a DJ
                st = rng.choice(("xfade1", "xfade2", "echo", "game") if T is A else ("mix", "mix", "xfade2", "xfade2", "dj", "djverb", "djblend", "game"))   # 2026-11-19: xfade2 64%, dj/djverb 45%, djblend 32%
                out.append({"a": j["a"], "t_out": j["t_out"], "b": T.title, "t_in": nxt.start,
                            "lead_bars": nb,
                            "steps": [{"b": T.title, "t_in": bars[-nb], "bars": None, "style": st, "until": nxt.start},
                                      {"b": T.title, "t_in": nxt.start, "bars": None, "style": "natural"}]})
    return out


def describe(route) -> str:
    r = as_route(route)
    return " → ".join(f"{s['style']}:{s['b'][:18]}@{s['t_in']:.0f}" for s in r["steps"])


def hops(lib_by, route) -> list[dict]:
    """Each change in a route as its own one-step route (so each can be judged on its own)."""
    r = as_route(route)
    out, src, t_out = [], r["a"], r["t_out"]
    for k, s in enumerate(r["steps"]):
        out.append({"a": src, "t_out": t_out, "b": s["b"], "t_in": s["t_in"],
                    "steps": [{"b": s["b"], "t_in": s["t_in"], "bars": None, "style": s["style"]}]})
        if k + 1 < len(r["steps"]):
            T = lib_by[s["b"]]
            t_out = s.get("until") or (s["t_in"] + _bars_len(T, s["t_in"], s["bars"] or 2))
            if r["steps"][k + 1]["style"] == "natural":
                t_out = r["steps"][k + 1]["t_in"]
        src = s["b"]
    return out


# ── features ────────────────────────────────────────────────────────────────
def features(lib_by, route) -> dict:
    from . import joins, joinmodel
    r = as_route(route)
    A = lib_by[r["a"]]
    f = {}
    hops = []
    cur, t_out = A, r["t_out"]
    for k, s in enumerate(r["steps"]):
        T = lib_by[s["b"]]
        style = s["style"] if s["style"] != "auto" else _auto(cur, t_out, T, s["t_in"])
        hops.append((cur, t_out, T, s.get("land") or s["t_in"], style))   # signposted: judge where it lands
        if k + 1 < len(r["steps"]):
            nxt = r["steps"][k + 1]
            t_out = s.get("until") or (s["t_in"] + _bars_len(T, s["t_in"], s["bars"] or 2))
            if nxt["style"] == "natural":
                t_out = nxt["t_in"]
        cur = T
    first, last = hops[0], hops[-1]
    jf = joins.features(first[0], first[1], first[2], first[3])
    f.update(jf)
    f["natural"] = joinmodel.score(first[0], first[1], first[2], first[3]) or 0.0
    f["same_song"] = float(r["a"] == first[2].title)
    f["hops"] = float(len(hops))
    for st in STYLES:
        f["style1_" + st] = float(first[4] == st)
        f["style2_" + st] = float(len(hops) > 1 and last[4] == st)
    f["jump"] = float(UP_ORDER.get(_role(last[2], last[3]), 2) - UP_ORDER.get(_role(A, r["t_out"] - 0.1), 2))
    f["lead_bars"] = float(as_route(route).get("lead_bars", 0))
    # 2026-11-25: the joins by FAMILY, not 40 sparse names - and the early lead-ins (tease/strip/fill/hook/
    # swap/layer, mostly judged while buggy and mostly lost) apart from the layered recipes (which win),
    # so "it has a lead-in" no longer lumps them together (the trees scored 27 % on how-questions)
    rec = _recipe_of(first[4])
    for fam in RECIPES:
        f["recipe_" + fam] = float(rec is not None and rec[0] == fam)
    f["recipe_bars"] = float(rec[1]) if rec else 0.0
    f["recipe_hooks"] = float(bool(rec and rec[2]))
    old_lead = first[4] in LEAD_STYLES and rec is None or first[4] in AFTER_STYLES
    f["early_lead"] = float(old_lead)
    f["early_lead_bars"] = float(LEAD_STYLES.get(first[4], 0)) if old_lead else 0.0
    f["quick_join"] = float(first[4] in ("cut", "punch", "xfade1", "xfade2", "game", "echo", "filter"))
    f["lead_same_song"] = float(len(hops) > 1 and hops[1][2].title == r["a"])
    f["via_build"] = float(len(hops) > 1 and _role(hops[1][0], hops[1][1] - 0.2) in ("build", "bridge"))
    return f


# ── rendering ───────────────────────────────────────────────────────────────
_AUDIO: dict = {}
_GAIN: dict = {}
_STEM: dict = {}


def _stem(track, kind: str = "rhythm") -> np.ndarray | None:
    """The song's drums+bass stem (cache/stems/<key>_rhythm.flac) or drums alone ("drums"), at the
    mix's level; None if missing."""
    if (track.path, kind) not in _STEM:
        from .analysis import cache_key
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "stems",
                         cache_key(track.path) + f"_{kind}.flac")
        y = None
        if os.path.exists(p):
            import librosa
            y = np.atleast_2d(librosa.load(p, sr=SR, mono=False)[0])
            y = np.vstack([y, y]) if y.shape[0] == 1 else y[:2]
            _audio(track)
            y = np.ascontiguousarray(y.T, dtype=np.float32) * _GAIN.get(track.path, 1.0)
        _STEM[(track.path, kind)] = y
        if len(_STEM) > 12:
            _STEM.pop(next(iter(_STEM)))
    return _STEM[(track.path, kind)]


def _callout(track, kind: str, max_s: float) -> np.ndarray | None:
    """The song's best-scoring call-out of this kind (riser/fill/impact, cut from its own stems and
    ending on a downbeat) no longer than max_s, at the mix's level."""
    import json
    from .analysis import cache_key
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "stems",
                     cache_key(track.path) + "_callouts.json")
    try:
        cs = [c for c in json.load(open(p, encoding="utf-8"))["callouts"] if c["type"] == kind and c["len"] <= max_s + 0.05]
    except (OSError, ValueError, KeyError):
        return None
    for c in sorted(cs, key=lambda c: -c.get("score", 0)):
        try:
            import soundfile as sf
            y, sr = sf.read(c["file"], dtype="float32", always_2d=True)
        except Exception:
            continue
        if sr != SR:
            import librosa
            y = librosa.resample(y.T, orig_sr=sr, target_sr=SR).T
        y = np.vstack([y.T, y.T]).T[:, :2] if y.shape[1] == 1 else y[:, :2]
        _audio(track)
        return np.ascontiguousarray(y, dtype=np.float32) * _GAIN.get(track.path, 1.0)
    return None


def _hold_level(y: np.ndarray, ref: np.ndarray, max_db: float = 6.0, under_db: float = 1.5,
                ref_after: np.ndarray | None = None) -> np.ndarray:
    g = _hold_gain(y, ref, max_db, under_db, ref_after=ref_after)
    if g is None:
        return y
    return (0.98 * np.tanh(y * g / 0.98)).astype(np.float32)            # soft limit


def _hold_gain(y: np.ndarray, ref: np.ndarray, max_db: float = 6.0, under_db: float = 1.5,
               ref_after: np.ndarray | None = None, over_db: float = 1.5, cut_db: float = 9.0):
    """Make-up gain: keep a lead-in as loud as the music just before it (user 2026-11-24: lead-ins
    dipped 6-10 dB and vanish under a horde). Only ever lifts, follows a smoothed 200 ms loudness,
    and a soft limiter keeps the peaks under full scale."""
    if len(y) < SR // 2 or len(ref) < SR // 4:
        return None
    # loudness as heard: bass carries most of the raw energy but little of the loudness, so a strip that
    # removed the bass got lifted far too high (user 2026-11-24: "now too loud in the blend")
    _, ref_hi = _split(ref, 120.0)
    _, y_hi = _split(y, 120.0)
    level = float(np.sqrt(np.mean(ref_hi ** 2)) + 1e-9)
    hop = int(0.05 * SR)
    win = 4 * hop
    mono = y_hi.mean(1)
    env = np.array([np.sqrt(np.mean(mono[i:i + win] ** 2)) + 1e-9 for i in range(0, len(y), hop)])
    # user 2026-11-26: hooks/sounds on top of both songs made blends clearly LOUDER than either song. The
    # level now aims along a line from the music before (ref) to the new song after the blend (ref_after):
    # lifted when more than under_db below it, cut when more than over_db above it (up to cut_db).
    if ref_after is not None and len(ref_after) >= SR // 4:
        _, ra_hi = _split(ref_after, 120.0)
        after = float(np.sqrt(np.mean(ra_hi ** 2)) + 1e-9)
    else:
        after = level
    line = level + (after - level) * np.linspace(0, 1, len(env))
    lo, hi = line * 10 ** (-under_db / 20), line * 10 ** (over_db / 20)
    g = np.where(env < lo, lo / env, np.where(env > hi, hi / env, 1.0))
    g = np.clip(g, 10 ** (-cut_db / 20), 10 ** (max_db / 20))
    k = 8                                                              # ~400 ms smoothing, no pumping
    g = np.convolve(np.pad(g, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    return np.interp(np.arange(len(y)), np.arange(len(g)) * hop + win / 2, g).astype(np.float32)[:, None]


LAYER_STYLES = ("layer4", "layer8", "layer4d", "swap4", "swap8", "swap4d", "swap8d") + tuple(
    f"{r}{L}{h}" for r in ("melfirst", "rhyfirst", "drumbridge", "basscarry") for L in (4, 8) for h in ("", "h"))
SWAP_STYLES = ("swap4", "swap8", "swap4d", "swap8d")

# 2026-11-24 blend recipes (user: every swap was the same shape - melody first, drums-only gap, rhythm on
# the drop - so it would win by default). Each recipe = which layers of A and B play in each quarter of
# the blend; the landing (after the last quarter) is always B in full. d = drums, b = bass, m = melody.
RECIPES = {
    "melfirst": [("dbm", ""), ("db", ""), ("db", "m"), ("db", "m")],      # melody first (the old swap)
    "rhyfirst": [("dbm", ""), ("m", "d"), ("m", "db"), ("m", "db")],      # user's flip: B's rhythm under A's melody
    "drumbridge": [("dbm", ""), ("d", ""), ("d", "m"), ("d", "m")],       # A's drums carry B's melody
    "basscarry": [("dbm", ""), ("bm", "d"), ("b", "dm"), ("b", "dm")],    # A's bass under B's drums + melody
    # user 2026-11-24 (set 6 A, "awesome"): A plays on in full while B's hook floats over it, then B lands
    "hookover": [("dbm", ""), ("dbm", ""), ("dbm", ""), ("dbm", "")],
    # user 2026-11-25: when the beats can't meet, don't fake it - A drops into a breakdown (melody only),
    # B's hooks float over it, B's melody takes over, then B lands on a CALM part (no beat to clash)
    "breakdown": [("dbm", ""), ("m", ""), ("m", "m"), ("", "m")],
}
# ── 2026-11-25 generated blends (user: rule recipes have no idea what's playing; let the AI find the shape).
# A plan is encoded in its style name so every process can rebuild it:
#   gen{L}_{Ad Ab Am}{Bd Bb Bm}{6 fades}h{hook bar}{hook bars}t{A hook trail}
# A lanes switch OFF at bar a (1..L), B lanes ON at bar b (0..L; L = at the landing); fades 0 = 1 beat,
# 1 = 1 bar, 2 = 2 bars; hook bar 0 = no B hook, else it enters at bar (k-1) for n bars.
import re as _re
_GEN = _re.compile(r"gen(\d)_([0-8]{3})([0-8]{3})([0-2]{6})h(\d)(\d)(\d?)t(\d)$")   # 3rd hook digit: 1 = loud tease


def gen_parse(style: str):
    m = _GEN.match(style or "")
    if not m:
        return None
    L = int(m.group(1))
    return {"L": L, "a": [int(c) for c in m.group(2)], "b": [int(c) for c in m.group(3)],
            "fade": [int(c) for c in m.group(4)], "hook": (int(m.group(5)), int(m.group(6))),
            "loud": int(m.group(7) or 0), "trail": int(m.group(8))}


def gen_name(p: dict) -> str:
    return (f"gen{p['L']}_{''.join(map(str, p['a']))}{''.join(map(str, p['b']))}{''.join(map(str, p['fade']))}"
            f"h{p['hook'][0]}{p['hook'][1]}{p.get('loud', 0) or ''}t{p['trail']}")


def register_gen(style: str) -> bool:
    """Make a generated style known to the renderer (LEAD_STYLES / LAYER_STYLES)."""
    global LAYER_STYLES
    p = gen_parse(style)
    if p is None:
        return False
    LEAD_STYLES[style] = p["L"]
    if style not in LAYER_STYLES:
        LAYER_STYLES = LAYER_STYLES + (style,)
    return True


def gen_plan(style: str, drop: bool = True) -> list[dict]:
    p = gen_parse(style)
    L = p["L"]
    beat, bar = 1 / (4 * L), 1 / L
    lanes = []
    for side, sw in (("A", p["a"]), ("B", p["b"])):
        for i, lane in enumerate(("drums", "bass", "melody")):
            x = sw[i] / L
            f = p["fade"][i + (0 if side == "A" else 3)]
            w = (beat, bar, 2 * bar)[f]
            if side == "A":
                if x >= 1:                                  # A plays to the landing, lets go ON it
                    w = min(w, beat / 2 if drop and lane != "melody" else w)
                    pts = [(0.0, 1.0), (1 - w, 1.0), (1.0, 0.0)]
                else:                                       # fades out starting on its bar line
                    pts = [(0.0, 1.0), (x, 1.0), (min(1.0, x + w), 0.0), (1.0, 0.0)]
            else:
                if x >= 1:                                  # B comes in on the landing: finish ON the line
                    w = beat / 2 if drop and lane != "melody" else min(w, bar)
                    pts = [(0.0, 0.0), (1 - w, 0.0), (1.0, 1.0)]
                else:
                    pts = [(0.0, 0.0), (x, 0.0), (min(1.0, x + w), 1.0), (1.0, 1.0)]
            lanes.append({"lane": f"{side} {lane}", "pts": _clean(pts)})
    k, n = p["hook"]
    if k:
        x0 = (k - 1) / L; x1 = min(1.0, x0 + n / L)
        v = 1.0 if p.get("loud") else 0.6
        lanes.insert(3, {"lane": "B hook", "pts": [(0, 0), (x0, 0), (x0 + 0.01, v), (max(x0 + 0.02, x1 - 0.02), v), (x1, 0)]})
    if p["trail"]:
        lanes.append({"lane": "A hook", "pts": [(0, 0), (0.72, 0), (0.78, 0.45), (1, 0)]})
    return lanes


BEAT_RECIPES = ("melfirst", "rhyfirst", "drumbridge", "basscarry", "hookover")   # need both beats locked
CHILL_ROLES = ("explore", "build", "bridge")


def tempo_match(A, ta: float, B, tb: float, tol: float = 0.03) -> bool:
    """User 2026-11-25 + answers: tempos within ~2 % (direct or half/double time) were good 64 % of the time,
    further apart 23 %. Only then can two beats run together."""
    from . import grid
    try:
        r = abs(np.log2(grid.period(A, ta) / grid.period(B, tb)))
    except Exception:
        return False
    return abs(r - round(r)) < tol


def chill_at(B, t: float) -> bool:
    return _role(B, t) in CHILL_ROLES
for _r in RECIPES:
    for _L in (4, 8):
        for _h in ("", "h"):
            LEAD_STYLES[f"{_r}{_L}{_h}"] = _L
RECIPE_STYLES = tuple(f"{r}{L}{h}" for r in RECIPES for L in (4, 8) for h in ("", "h"))
RECIPE_NAMES = {"melfirst": "Melody first", "rhyfirst": "Rhythm first", "drumbridge": "Drum bridge",
                "basscarry": "Bass carry", "hookover": "Hook over"}


def _recipe_of(style: str):
    for r in RECIPES:
        if style.startswith(r):
            rest = style[len(r):]
            return r, int(rest.rstrip("h")), rest.endswith("h")
    return None


def recipe_plan(style: str, drop: bool = True) -> list[dict]:
    """Lanes for a recipe: each layer switches at quarter lines with a 1-beat fade (melody: 1 bar);
    at the landing B comes in full - a half-beat swap on a real drop, a 1-bar crossfade anywhere else.
    'h' adds sprinkles: B's hook over A in the first half, A's hook trailing in the last quarter."""
    name, L, sprinkle = _recipe_of(style)
    phases = RECIPES[name] + [("", "dbm")]
    beat, bar = 1 / (4 * L), 1 / L
    lanes = []
    for side in ("A", "B"):
        for code, lane in (("d", "drums"), ("b", "bass"), ("m", "melody")):
            on = [code in (ph[0] if side == "A" else ph[1]) for ph in phases]
            pts = [(0.0, float(on[0]))]
            for i in range(1, len(on)):
                if on[i] == on[i - 1]:
                    continue
                t = i / 4
                w = (beat / 2 if drop else bar) if i == 4 and code != "m" else (bar if code == "m" or name == "breakdown" else beat)
                if i == 4:                                      # landing: the change finishes ON the line
                    pts += [(t - w, float(on[i - 1])), (t, float(on[i]))]
                else:                                           # mid-blend: starts on the line
                    pts += [(t, float(on[i - 1])), (min(1.0, t + w), float(on[i]))]
            pts.append((1.0, float(on[-1]) if on[-1] == on[-2] or True else float(on[-2])))
            lanes.append({"lane": f"{side} {lane}", "pts": _clean(pts)})
    if name == "hookover":                                      # the hook IS the recipe: over A's second half
        lanes.insert(3, {"lane": "B hook", "pts": [(0, 0), (0.5, 0), (0.52, 0.7), (0.98, 0.7), (1, 0)]})
    elif sprinkle:
        lanes.insert(3, {"lane": "B hook", "pts": [(0, 0), (0.05, 0.5), (0.45, 0.5), (0.5, 0)]})
        lanes.append({"lane": "A hook", "pts": [(0, 0), (0.72, 0), (0.78, 0.45), (1, 0)]})
    return lanes


LAST_PLAN: list = []          # the lanes the last layered blend actually played (for the trainer's picture)


def _bar_db(y: np.ndarray, L: int, m: int) -> np.ndarray:
    k = max(1, m // (2 * L))                                       # half-bar resolution
    return np.array([20 * np.log10(np.sqrt(np.mean(y[i:i + k] ** 2)) + 1e-9) for i in range(0, m - k + 1, k)])


def _adapt_plan(plan: list, layers: dict, ref_db: float, L: int, m: int, keep_beat: bool = True):
    """2026-11-24 (user): a plan that says "A's drums play" when the song has a silent break there just
    plays nothing. Look at the audio: where the old song falls quiet before the landing, B announces
    itself (its hook goes in the hole - added by the caller) and B's melody starts blending in from
    there instead of waiting for its slot. B's drums + bass still wait for the drop."""
    P = {p["lane"]: _env(p["pts"], m)[:, 0] for p in plan}
    # user 2026-11-24 (set 6): never a stretch of melody with no beat. Where no drums are really playing
    # before the landing, keep A's drums going if A has drums there, else bring B's drums in early.
    half, beat = 1 / (2 * L), 1 / (4 * L)
    ad, bd = _bar_db(layers["A drums"], L, m), _bar_db(layers["B drums"], L, m)
    live = lambda d: np.clip((d - (ref_db - 30)) / 24, 0, 1)
    idx = lambda i: min(m - 1, int((i + 0.5) * half * m))
    pres = [P["A drums"][idx(i)] * live(ad[i]) + P["B drums"][idx(i)] * live(bd[i]) for i in range(len(ad))]
    holes = [i for i, v in enumerate(pres) if v < 0.3 and (i + 1) * half <= 1 - half + 1e-9]
    if holes and keep_beat:                  # a breakdown bridge drops the beat on purpose (2026-11-25)
        i0, j0 = holes[0], holes[-1]
        lane = {p["lane"]: p for p in plan}
        bar_dn = lambda x: np.floor(x * L + 1e-6) / L               # user 2026-11-24 (set 8): changes only ON a
        bar_up = lambda x: np.ceil(x * L - 1e-6) / L                # bar line - off-bar drops feel wrong
        if np.median([live(ad[i]) for i in holes]) > 0.3:          # A has drums there: let them play on
            end = min(1.0, bar_up((j0 + 1) * half))
            lane["A drums"]["pts"] = ([(0, 1), (end, 1), (min(1.0, end + beat), 0)] if end < 1 - beat
                                      else [(0, 1), (1 - beat / 2, 1), (1, 0)])
        elif np.median([live(bd[i]) for i in holes]) > 0.3:        # A doesn't: B's drums come in early
            x0 = bar_dn(i0 * half)
            lane["B drums"]["pts"] = [(0, 0), (x0, 0), (x0 + beat, 0.8), (1 - 0.002, 0.8), (1, 1)]
        P = {p["lane"]: _env(p["pts"], m)[:, 0] for p in plan}
    a_out = sum(layers[k] * P[k][:, None] for k in ("A drums", "A bass", "A melody"))
    db = _bar_db(a_out, L, m)
    gaps = [i for i, v in enumerate(db) if v < ref_db - 12 and (i + 1) * half < 1 - 1 / L]
    if not gaps:
        return plan, None
    g = np.floor(gaps[0] * half * L + 1e-6) / L                        # the hook/melody enter on a bar line
    for p in plan:
        if p["lane"] == "B melody":
            on = next((x for x, v in p["pts"] if v >= 0.5), 1.0)
            if on > g + 1e-6:                                      # bring it forward to the gap, 1-bar fade
                p["pts"] = [(0, 0), (g, 0), (min(1.0, g + 1 / L), 0.8), (1, 1)]
    return plan, g


def _show_actual(plan: list, layers: dict, ref_db: float, L: int, m: int) -> list:
    """Each lane's height = planned gain x how much that layer is really playing (per half bar), so a
    silent break shows as empty instead of a full bar."""
    out = []
    for p in plan:
        q = dict(p)
        if p["lane"] in layers:
            db = _bar_db(layers[p["lane"]], L, m)
            q["act"] = [round(float(np.clip((v - (ref_db - 30)) / 24, 0, 1)), 2) for v in db]
        out.append(q)
    return out


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    lo, hi = max(20.0, float(lo)), min(SR / 2 - 100, float(hi))
    return sosfiltfilt(butter(4, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos"), x, axis=0).astype(np.float32)


LAST_LOCK = None
LAST_SHIFT = 0            # samples the last join's body is longer than the new song's own span (tempo glide)
TWEAKS: dict = {}         # trainer "Tweak": {lane: {"db": gain dB, "end": +/- beats}} for the blend being rendered


def _tw_gain(lane: str) -> float:
    return 10 ** (float((TWEAKS.get(lane) or {}).get("db", 0.0)) / 20)


def _tw_end(lane: str, L: int) -> float:
    """The lane's end moved by this many beats, as a fraction of a blend L bars long."""
    return float((TWEAKS.get(lane) or {}).get("end", 0.0)) / (4 * max(1, L))


def _tw_pts(lane: str):
    """The envelope you drew for this lane on the trainer page, or None."""
    d = (TWEAKS.get("plan") or {}).get(lane)
    if not d or len(d) < 2:
        return None
    pts = sorted((min(1.0, max(0.0, float(x))), min(2.0, max(0.0, float(g)))) for x, g in d)
    if pts[0][0] > 0:
        pts = [(0.0, pts[0][1])] + pts
    if pts[-1][0] < 1:
        pts = pts + [(1.0, pts[-1][1])]
    return pts


def _tw_plan(plan: list, L: int) -> list:
    """Move each tweaked lane's last fade-out (its end) by whole beats; clamp inside the blend."""
    out = []
    for p in plan:
        drawn = _tw_pts(p["lane"])
        if drawn is not None:
            out.append({**p, "pts": drawn})
            continue
        d = _tw_end(p["lane"], L)
        pts = [tuple(x) for x in p["pts"]]
        if d and len(pts) >= 2:
            on = [j for j, (_, g) in enumerate(pts) if g > 0.05]
            if on and on[-1] + 1 < len(pts):
                j = on[-1]
                lo = pts[j - 1][0] + 1e-3 if j > 0 else 0.0
                x0 = min(max(pts[j][0] + d, lo), 0.995)
                x1 = min(max(pts[j + 1][0] + d, x0 + 1e-3), 1.0)
                pts[j], pts[j + 1] = (x0, pts[j][1]), (x1, pts[j + 1][1])
                pts = [q for q in pts[:j + 2]] + [q for q in pts[j + 2:] if q[0] > x1]
            elif on and on[-1] == len(pts) - 1 and d < 0:        # plays to the end: make it stop earlier
                x0 = max(0.0, 1.0 + d)
                pts = [q for q in pts if q[0] < x0] + [(x0, pts[-1][1]), (min(1.0, x0 + 0.01), 0.0), (1.0, 0.0)]
        out.append({**p, "pts": pts})
    return out


def _drum_lag(a: np.ndarray, b: np.ndarray, beat: float) -> float:
    """Seconds A's drums (already on B's beat) hit late vs B's (+ = late), from the onset envelopes; 0
    when unsure. Searches +-30 % of a beat."""
    import librosa
    hop = 256
    ea = librosa.onset.onset_strength(y=a.mean(1), sr=SR, hop_length=hop)
    eb = librosa.onset.onset_strength(y=b.mean(1), sr=SR, hop_length=hop)
    n = min(len(ea), len(eb))
    if n < 50:
        return 0.0
    ea, eb = ea[:n], eb[:n]
    ea = (ea - ea.mean()) / (ea.std() + 1e-9); eb = (eb - eb.mean()) / (eb.std() + 1e-9)
    r = max(1, int(0.3 * beat * SR / hop))
    lags = np.arange(-r, r + 1)
    c = np.array([np.mean(ea[max(0, l):n + min(0, l)] * eb[max(0, -l):n - max(0, l)]) for l in lags])
    best = int(np.argmax(c))
    if c[best] < 0.3 or c[best] < 1.1 * c[r]:
        return 0.0
    return float(lags[best] * hop / SR)


class _VoxGuard:
    """Singing in both songs across a blend (vocals.py), on a 20 Hz grid, turned into beat-faded gains."""
    HZ = 20

    def __init__(self, A, a_times, B, t_new: float, m: int, beat: float):
        from . import vocals
        self.m = m
        self.idx = np.arange(0, m, max(1, SR // self.HZ))
        self.va = vocals.at(A, np.asarray(a_times, float)[self.idx])
        self.vb = vocals.at(B, t_new + self.idx / SR)
        self.r = max(2, int(round(beat * self.HZ)))
        self.da = np.ones(len(self.idx), np.float32)              # A's melody after the duck

    def _g(self, P):
        return np.ones(len(self.idx)) if P is None else P.ravel()[self.idx]

    def _smooth(self, bad):
        """Bad grid points -> gain: 0 over them (and half a beat either side), beat-long fades."""
        if not bad.any():
            return None
        h = self.r // 2
        d = np.convolve(bad.astype(float), np.ones(2 * h + 1), "same") > 0
        return 1 - np.convolve(d.astype(float), np.ones(self.r) / self.r, "same")

    def _up(self, g, n: int, off: int = 0):
        return np.interp(np.arange(off, off + n), self.idx, g).astype(np.float32)[:, None]

    def duck_a(self, pa, pb):
        bad = (self.va * self._g(pa) > 0.25) & (self.vb * self._g(pb) > 0.25)
        if bad.any():                                  # duck A's WHOLE phrase, never out mid-word
            d = np.diff(np.concatenate([[0], (self.va > 0).astype(int), [0]]))
            for s0, s1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if bad[s0:s1].any():
                    bad[s0:s1] = True
        g = self._smooth(bad)
        if g is None:
            return None
        self.da = g
        return self._up(g, self.m)

    def gate(self, T, t_src: float, off: int, n: int, pa, pb, speed: float = 1.0):
        """Gain for a hook/sound cut from T at t_src, laid in at sample `off` for n samples: 0 where it
        sings over audible singing in the blend."""
        from . import vocals
        sel = (self.idx >= off) & (self.idx < off + n)
        hv = np.zeros(len(self.idx), np.float32)
        hv[sel] = vocals.at(T, t_src + (self.idx[sel] - off) / SR * speed)
        if not hv.any():
            return 1.0
        from . import vocals as _v
        if not _v.whole_phrase(T, t_src, n / SR * speed):     # a couple of words chopped out: never
            return 0.0
        bed = self.va * self._g(pa) * self.da + self.vb * self._g(pb)
        return 0.0 if ((hv > 0) & (bed > 0.25)).any() else 1.0   # all or nothing: no words cut mid-hook

    def gate_a_over_b(self, pb):
        g = self._smooth((self.va > 0) & (self.vb * self._g(pb) > 0.25))
        return 1.0 if g is None else self._up(g, self.m)

    def pts(self, c):
        k = np.linspace(0, self.m - 1, 65).astype(int)
        return [(round(float(i / max(1, self.m - 1)), 4), round(float(c[i, 0]), 3)) for i in k]


def _sprinkle_sounds(mix: np.ndarray, A, B, a_ml: np.ndarray, t_new: float, m: int, plan: list, L: int = 4,
                     loud: bool = False, vg=None, pa=None, pb=None, a_acc=None) -> np.ndarray:
    """User-isolated signature sounds (hook picker "Sounds", 2026-11-24 - e.g. Miasma's high spooky tone).
    B's sound, filtered to the user's band, fades in under the first half as the announcement; A's own
    sound (its live melodic layer through A's band) carries on as the rest of A leaves, then fades."""
    from . import hooks
    x = np.linspace(0, 1, m, dtype=np.float32)[:, None]
    sb = hooks.sounds(B)
    if sb:
        # beat-snapped (hooks.snap_sound); enters on the blend's first bar line, and a sound shorter than the
        # first 60 % repeats on the bar grid (every bar, or every n bars for longer ones) - never off the beat
        c = min(sb, key=lambda h: abs(h["t"] - t_new))
        bar = m / max(1, L)
        k = min(int(0.6 * m), int(c["len"] * SR))
        src = _bandpass(hooks.melodic(B, c["t"], k), c["lo"], c["hi"])
        f = min(k // 4, int(0.006 * SR))
        if f > 0:
            src[:f] *= np.linspace(0, 1, f, dtype=np.float32)[:, None]
            src[-f:] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
        src = _hook_level(src, mix[:k], rel_db=0.0 if loud else -8.0)
        every = int(bar * max(1, int(np.ceil(k / bar - 1e-6))))
        pos, n = 0, 0
        stop = min(1.0, max(0.1, 0.6 + _tw_end("B sound", L)))           # trainer tweak: end earlier / later
        src = src * _tw_gain("B sound")
        drawn = _tw_pts("B sound")
        if drawn is not None:
            envd = _env(drawn, m)
            while pos + k <= m and n < 32:
                seg = envd[pos:pos + k]
                if float(seg.max()) > 0.01:
                    g = seg if vg is None else seg * vg.gate(B, c["t"], pos, k, pa, pb)
                    mix[pos:pos + k] += src * g
                pos += every; n += 1
            pos = m + 1                                                        # skip the planned loop below
        while pos + k <= int(stop * m) + 1 and n < 16:
            g = 1.0 if loud else 0.6 if n == 0 else 0.75                      # first one a touch softer
            if vg is not None:                                                 # a sung sound never over singing
                g = g * vg.gate(B, c["t"], pos, k, pa, pb)
            mix[pos:pos + k] += src * g
            pos += every; n += 1
        v = 1.0 if loud else 0.6
        plan.append({"lane": "B sound", "pts": drawn if drawn is not None else [(0, 0), (0.02, v), (stop - 0.02, v), (stop, 0)]})
    sa = hooks.sounds(A)
    if sa:
        c = sa[0]
        src = _bandpass(a_ml, c["lo"], c["hi"])
        e = min(1.0, max(0.55, 1.0 + _tw_end("A sound", L)))
        env = np.clip((x - 0.5) * 4, 0, 1) * np.clip((e - x) * 4, 0, 1) * _tw_gain("A sound")    # as A leaves
        if _tw_pts("A sound") is not None:
            env = _env(_tw_pts("A sound"), m) / 0.6
        if vg is not None:                                 # A's band carries A's voice: never over B singing
            env = env * vg.gate_a_over_b(pb)
        add = _hook_level(src, mix, rel_db=-8.0) * env
        mix += add
        if a_acc is not None:
            a_acc += add
        plan.append({"lane": "A sound", "pts": _tw_pts("A sound") or [(0, 0), (0.5, 0), (0.75, 0.6), (1, 0)]})
    return mix


def _hook_level(h: np.ndarray, under: np.ndarray, rel_db: float = -6.0) -> np.ndarray:
    """A hook at a fixed level vanished under a full old song (user 2026-11-24: "sometimes you can't even
    hear the hook"). Scale it to sit rel_db under the music it plays over (both measured on the part you
    hear, bass ignored), capped so a near-silent hook isn't blown up into noise."""
    _, hu = _split(under, 120.0)
    t = float(np.sqrt(np.mean(hu ** 2)) + 1e-9) * 10 ** (rel_db / 20)
    c = float(np.sqrt(np.mean(h ** 2)) + 1e-9)
    return h * min(8.0, max(0.25, t / c))


def _clean(pts):
    out = []
    for x, g in sorted(pts, key=lambda p: p[0]):
        x = max(0.0, min(1.0, x))
        if out and abs(out[-1][0] - x) < 1e-9:
            out[-1] = (x, g)
        else:
            out.append((x, g))
    return out


def grid_ok(track, t0: float, t1: float, max_cv: float = 0.04) -> bool:
    """Steady, drum-confirmed fitted grid over [t0, t1] (grid.ok)."""
    from . import grid
    return grid.ok(track, t0, t1)


def _grid_ok_cv(track, t0: float, t1: float, max_cv: float = 0.04) -> bool:
    """The beat grid is steady enough here to line two songs up on it (user 2026-11-24: set 2 D was off
    beat - both songs' detected beats wobbled 0.07-0.8 s apart in that stretch). Coefficient of variation
    of the beat gaps, half/double-time slips folded."""
    b = np.array([x for x in track.beats if t0 - 0.5 <= x <= t1 + 0.5])
    if len(b) < 6:
        return False
    d = np.diff(b)
    med = float(np.median(d))
    d = np.where(d < 0.75 * med, d * 2, np.where(d > 1.5 * med, d / 2, d))
    return float(np.std(d) / (np.mean(d) + 1e-9)) < max_cv


def foreign_wind_down(A, B, t_land: float, down: bool) -> bool:
    """User 2026-11-24: going DOWN (fight -> calm), never land on ANOTHER song's breakdown, build-up or
    ending - "it's like another song's about to start, but it doesn't". The current song's own
    breakdown is fine; another song only via a real calm part of it."""
    if not down or A is B:
        return False
    seg = B.segment_at(t_land + 0.05)
    if seg is None:
        return True
    segs = B.segments
    last = seg is segs[-1] or (len(segs) > 1 and seg is segs[-2] and segs[-1].length < 20)
    return _role(B, t_land + 0.05) in ("bridge", "build") or last or seg.start > B.duration - 30


def lands_on_drop(B, t_land: float) -> bool:
    """The new music lands on a drop (a peak/horde part): a hard swap is fine there, nowhere else."""
    return _role(B, t_land + 0.1) in ("peak", "horde")


def layer_plan(style: str, drop: bool = True) -> list[dict]:
    """Gain envelopes of a layered blend, as fractions of its length: [{"lane", "pts": [(x, gain), ...]}].
    The trainer draws these as lanes under the waveform; _join renders exactly these."""
    if gen_parse(style):
        return gen_plan(style, drop)
    if style not in LAYER_STYLES:
        return []
    if _recipe_of(style):
        return recipe_plan(style, drop)
    if style in SWAP_STYLES:
        # user 2026-11-24: phases, not one long overlap - A full; A's melody out over a bar; the groove on
        # its own; B's melody in; B's drop. Every hand-off fades (>= 1 beat); only a real drop swaps hard.
        L = LEAD_STYLES[style]
        b, bar = 1 / (4 * L), 1 / L
        hand = b / 2 if drop else bar                       # drums/bass hand-over at the landing
        a_out = [(0, 1), (1 - hand, 1), (1, 0)]
        b_in = [(0, 0), (1 - (0.002 if drop else bar), 0), (1, 1)]
        m0 = 0.25 if L >= 8 else 0.0                        # A's melody leaves after a quarter (8 bars) / at once (4)
        mel_out = [(0, 1), (m0, 1), (m0 + bar, 0), (1, 0)] if L >= 8 else [(0, 1), (bar, 0), (1, 0)]
        b_mel = [(0, 0), (0.5, 0), (0.5 + bar, 0.8), (1, 1)]
        plan = [{"lane": "A drums", "pts": a_out},
                {"lane": "A bass", "pts": a_out if not style.endswith("d") else [(0, 1), (0.5, 1), (0.5 + b, 0), (1, 0)]},
                {"lane": "A melody", "pts": mel_out},
                {"lane": "B melody", "pts": b_mel},
                {"lane": "B drums", "pts": b_in},
                {"lane": "B bass", "pts": b_in}]
        return plan
    q = 1 / LEAD_STYLES[style]                                    # one bar
    plan = [
        {"lane": "A drums", "pts": [(0, 1), (0.5 - q * 0.15, 1), (0.5, 0)]},
        {"lane": "A bass", "pts": [(0, 1), (0.75 - 0.02, 1), (0.75, 0)]},
        {"lane": "A melody", "pts": [(0, 1), (0.5, 1), (1, 0)]},
        {"lane": "B drums", "pts": [(0, 0), (0.25, 0.7), (0.5, 1), (1, 1)]},
        {"lane": "B bass", "pts": [(0, 0), (0.75 - 0.02, 0), (0.75, 1), (1, 1)]},
        {"lane": "B melody", "pts": [(0, 0), (0.5, 0), (1, 1)]},
    ]
    if style == "layer4d":
        plan.insert(3, {"lane": "B hook", "pts": [(0, 0), (0.1, 0.55), (0.45, 0.55), (0.5, 0)]})
        plan.append({"lane": "A hook", "pts": [(0, 0), (0.7, 0), (0.78, 0.5), (1, 0)]})
    return plan


def show_plan(style: str, drop: bool = True) -> list[dict]:
    """Lanes for the trainer's picture of ANY join (approximate for the older styles)."""
    if style in LAYER_STYLES:
        return layer_plan(style, drop)
    if style.startswith("tease") or style == "hook4":
        p = [{"lane": "A (all)", "pts": [(0, 1), (1, 0.8)]}, {"lane": "B drums", "pts": [(0, 0), (1, 0.75)]}]
        if style == "hook4":
            p.append({"lane": "B hook", "pts": [(0, 0), (0.5, 0), (0.52, 0.35), (1, 0.75)]})
        return p + [{"lane": "B (all)", "pts": [(0, 0), (0.99, 0), (1, 1)]}]
    if style.startswith("strip"):
        top = [(0, 1), (1, 0.6)] if style != "strip4s" else [(0, 1), (0.75, 0.7), (0.76, 1), (1, 1)]
        bass = [(0, 1), (1, 0)] if style != "strip4s" else [(0, 1), (0.75, 0.06), (0.76, 1), (1, 1)]
        return [{"lane": "A top", "pts": top}, {"lane": "A bass+kick", "pts": bass},
                {"lane": "B riser", "pts": [(0, 0), (0.5, 0), (0.52, 0.3), (1, 0.9)]},
                {"lane": "B (all)", "pts": [(0, 0), (0.99, 0), (1, 1)]}]
    return []


def _env(pts, m: int) -> np.ndarray:
    xs, ys = zip(*pts)
    return np.interp(np.linspace(0, 1, m), xs, ys).astype(np.float32)[:, None]


def _layers(y: np.ndarray, stem: np.ndarray | None, drums: np.ndarray | None = None) -> tuple:
    """(drums, bass, melody) of a stretch of a song. With the separate drum stem: bass = rhythm - drums
    (2026-11-24: splitting the rhythm stem at 150 Hz also took the kick's body - heard as a filter sweep)."""
    if stem is None:
        lo, hi = _split(y, 150.0)
        return hi * 0.0, lo, hi
    if drums is not None:
        return drums, stem - drums, y - stem
    bass, dr = _split(stem, 150.0)
    return dr, bass, y - stem


def lead_start(B, t_land: float, style: str) -> float | None:
    """Where in the new song the announcing starts so it lands on t_land (whole bars of the new song)."""
    register_gen(style)
    if style in LAYER_STYLES and style not in SWAP_STYLES and not _recipe_of(style) and not gen_parse(style):
        return t_land                                             # the blend IS the new section's first bars
    L = LEAD_STYLES[style]
    db = [b for b in (B.downbeats or B.beats[::4]) if b < t_land - 0.1]
    return db[-L] if len(db) >= L and db[-L] >= 0 else None


def _audio(track) -> np.ndarray:
    if track.path not in _AUDIO:
        import librosa
        from .players import _level_gain
        y, _ = librosa.load(track.path, sr=SR, mono=False)
        y = np.atleast_2d(y)
        y = np.vstack([y, y]) if y.shape[0] == 1 else y[:2]
        y = np.ascontiguousarray(y.T, dtype=np.float32)
        g = _level_gain(track.path, y, SR)
        y *= g
        _GAIN[track.path] = g
        _AUDIO[track.path] = y
        if len(_AUDIO) > 12:
            _AUDIO.pop(next(iter(_AUDIO)))
    return _AUDIO[track.path]


def _take(y, t0: float, dur: float = 0.0, n: int | None = None) -> np.ndarray:
    a = int(t0 * SR)
    n = int(round(dur * SR)) if n is None else n
    out = np.zeros((n, 2), np.float32)
    lo = max(0, a)
    seg = y[lo:a + n] if a + n > 0 else y[:0]
    out[lo - a:lo - a + len(seg)] = seg
    return out


def _bars_len(track, t: float, bars: int) -> float:
    from .tempo import period_at
    db = [b for b in (track.downbeats or track.beats[::4]) if b > t + 0.1]
    if len(db) >= bars:
        return db[bars - 1] - t
    return bars * 4 * period_at(track, t)


def _lowpass_sweep(x: np.ndarray, f0=16000.0, f1=700.0) -> np.ndarray:
    from scipy.signal import butter, sosfilt, sosfilt_zi
    out = np.empty_like(x)
    blk = 512
    nb = max(1, int(np.ceil(len(x) / blk)))
    zi = None
    for k in range(nb):
        fc = f0 * (f1 / f0) ** (k / max(1, nb - 1))
        sos = butter(2, fc / (SR / 2), output="sos")
        if zi is None:
            zi = np.stack([sosfilt_zi(sos) * x[0, c] for c in range(2)], axis=-1)
        sl = slice(k * blk, (k + 1) * blk)
        y, zi = sosfilt(sos, x[sl], axis=0, zi=zi)
        out[sl] = y
    return out


def _reverb_wash(old: np.ndarray, t_stop: float, ref: np.ndarray, rt60: float = 2.0) -> np.ndarray:
    """What a DJ's reverb throw leaves behind: the old song's next quarter-second sent into a
    big reverb, wet only, bass cut so the new drop's low end stays clean. ~rt60 s long."""
    from scipy.signal import butter, fftconvolve, sosfilt
    send = _take(old, t_stop, 0.25)
    send *= np.cos(np.linspace(0, 1, len(send), dtype=np.float32)[:, None] * np.pi / 2)
    n = int(rt60 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.exp(-6.9 * t / rt60) * np.minimum(1.0, t / 0.01)      # -60 dB at rt60, soft onset
    rs = np.random.RandomState(7)
    ir = np.stack([rs.randn(n) * env, rs.randn(n) * env], 1).astype(np.float32)   # decorrelated L/R
    wet = np.stack([fftconvolve(send[:, c], ir[:, c])[:n + len(send)] for c in range(2)], 1)
    wet = sosfilt(butter(2, 300 / (SR / 2), btype="high", output="sos"), wet, axis=0)
    wet = sosfilt(butter(2, 7000 / (SR / 2), btype="low", output="sos"), wet, axis=0).astype(np.float32)
    ref_rms = float(np.sqrt(np.mean(ref[-int(0.5 * SR):] ** 2))) + 1e-6
    head = wet[:int(0.3 * SR)]
    wet *= 0.4 * ref_rms / (float(np.sqrt(np.mean(head ** 2))) + 1e-6)   # starts ~8 dB under the music
    return wet


def _speed(A, t_old: float, B, t_new: float) -> float:
    """Playback speed for the old song so its beat lines up with the new one's (half/double time
    counts). Like a DJ's pitch fader: only small nudges (<6%); bigger gaps are left alone."""
    if A is B:
        return 1.0
    from .tempo import period_at
    r = np.log2(period_at(A, t_old) / period_at(B, t_new))
    r -= np.round(r)
    return float(2 ** r) if abs(r) < 0.085 else 1.0


RAMP_FROM = 2 * 4 * 0.45      # blends at least ~2 bars long ease the tempo in instead of jumping


def _speed_curve(n: int, speed: float, ramp: bool) -> np.ndarray:
    """Per-output-sample playback speed. Without a ramp it's constant. With one it starts at 1.0
    (no jump at the seam) and ends at `speed`, shaped so the total distance travelled equals
    speed*n -- the old song lands on exactly the same beat at the drop as a constant nudge would."""
    if not ramp or abs(speed - 1.0) < 1e-4:
        return np.full(n, speed, np.float64)
    u = np.linspace(0, 1, n)
    g = u + (np.pi / 4) * np.sin(np.pi * u)     # g(0)=0, g(1)=1, mean(g)=1
    return 1.0 + (speed - 1.0) * g


def _wsola(src: np.ndarray, pos: np.ndarray, n: int) -> np.ndarray:
    """Time-stretch without changing pitch (WSOLA). `pos[i]` = where in `src` output sample i
    should come from; any rate curve works, so tempo can glide."""
    W, H, TOL = 2048, 512, 384
    win = np.hanning(W).astype(np.float32)[:, None]
    out = np.zeros((n + W, 2), np.float32)
    norm = np.zeros(n + W, np.float32)
    mono = src.mean(1)
    prev = None                                   # where the last grain's natural continuation sits
    for o in range(0, n, H):
        want = int(pos[min(o, n - 1)])
        best = want
        if prev is not None:
            lo, hi = max(0, want - TOL), min(len(src) - W, want + TOL)
            if hi > lo:
                ref = mono[prev:prev + W]
                if len(ref) == W:
                    seg = mono[lo:hi + W]
                    c = np.correlate(seg, ref, "valid")
                    e = np.sqrt(np.convolve(seg ** 2, np.ones(W), "valid")) + 1e-6
                    best = lo + int(np.argmax(c / e))
        best = max(0, min(best, len(src) - W))
        out[o:o + W] += src[best:best + W] * win
        norm[o:o + W] += win[:, 0]
        prev = best + H
    return (out[:n] / np.maximum(norm[:n], 1e-3)[:, None]).astype(np.float32)


STRETCH = os.environ.get("CONDUCTOR_STRETCH", "rubberband")   # 2026-11-20: home-made WSOLA rated 1/7


def _rubberband(y, t0: float, n: int, curve: np.ndarray) -> np.ndarray | None:
    """Rubber Band (studio-grade stretch, via ffmpeg): `n` output samples of `y` from t0, tempo
    following `curve` (per output sample), pitch unchanged. A second of pre-roll lets the engine
    settle before the part we keep."""
    import tempfile
    import soundfile as sf
    pre = 1.0
    pos = np.concatenate([[0.0], np.cumsum(curve)[:-1]])
    src = _take(y, t0 - pre, n=int(pre * SR + pos[-1]) + SR)
    d = tempfile.mkdtemp(prefix="rb_")
    fi, fo, fc = (os.path.join(d, x) for x in ("in.wav", "out.wav", "cmd.txt"))
    try:
        sf.write(fi, src, SR, subtype="FLOAT")
        step = int(0.1 * SR)                              # tempo updates every 100 ms of output
        cmds = [f"{(pre * SR + pos[o]) / SR:.4f} rubberband tempo {curve[o]:.5f};" for o in range(0, n, step)]
        open(fc, "w").write("\n".join(cmds))
        af = ("asendcmd=f=cmd.txt,"
              f"rubberband=tempo={curve[0]:.5f}:transients=crisp:detector=compound:phase=laminar:window=standard")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "in.wav", "-af", af, "out.wav"], capture_output=True,
                           cwd=d, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))   # relative: no path escaping   # no console flashing over the game
        if r.returncode != 0:
            return None
        out, _ = sf.read(fo, dtype="float32", always_2d=True)
        k = int(round(pre * SR / curve[0] + 0.009 * SR))      # + Rubber Band's ~9 ms latency
        out = out[k:k + n]
        if len(out) < n:
            out = np.vstack([out, np.zeros((n - len(out), out.shape[1]), np.float32)])
        return out[:, :2] if out.shape[1] >= 2 else np.repeat(out, 2, 1)
    finally:
        for f in (fi, fo, fc):
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(d)


_RB_LAT = {}


def _rb_pitch_run(y: np.ndarray, factor: np.ndarray) -> np.ndarray | None:
    """Rubber Band pitch-only pass (tempo 1: timing untouched): pitch x factor[i] around output sample i,
    updated every 50 ms. Returns len(y) samples, its latency removed (measured once with a click)."""
    import tempfile
    import soundfile as sf
    pad = int(0.5 * SR)
    src = np.concatenate([np.zeros((pad, 2), np.float32), y.astype(np.float32), np.zeros((pad, 2), np.float32)])
    d = tempfile.mkdtemp(prefix="rbp_")
    fi, fo, fc = (os.path.join(d, x) for x in ("in.wav", "out.wav", "cmd.txt"))
    try:
        sf.write(fi, src, SR, subtype="FLOAT")
        step = int(0.05 * SR)
        cmds = [f"{(pad + o) / SR:.4f} rubberband pitch {float(factor[o]):.6f};" for o in range(0, len(y), step)]
        open(fc, "w").write("\n".join(cmds))
        af = ("asendcmd=f=cmd.txt,"
              f"rubberband=tempo=1:pitch={float(factor[0]):.6f}:transients=crisp:detector=compound:"
              f"phase=laminar:window=standard:pitchq=quality")
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "in.wav", "-af", af, "out.wav"], capture_output=True,
                           cwd=d, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))   # relative: no path escaping
        if r.returncode != 0:
            print("pitch fix failed:", r.stderr.decode(errors="replace")[-300:], flush=True)
            return None
        out, _ = sf.read(fo, dtype="float32", always_2d=True)
    finally:
        for f in (fi, fo, fc):
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(d)
    out = out[:, :2] if out.shape[1] >= 2 else np.repeat(out, 2, 1)
    return out


def _rb_latency() -> int:
    if "lat" not in _RB_LAT:
        click = np.zeros((SR, 2), np.float32)
        click[int(0.3 * SR):int(0.3 * SR) + 32] = 0.8
        o = _rb_pitch_run(click, np.full(SR, 1.02))
        _RB_LAT["lat"] = (int(np.argmax(np.abs(o[:, 0]))) - int(0.5 * SR) - int(0.3 * SR)) if o is not None else 0
    return _RB_LAT["lat"]


def _pitch_fix(y: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """y with its pitch multiplied by factor (per sample), timing unchanged. On failure: y as it was."""
    if len(y) < SR // 4 or np.abs(factor - 1).max() < 0.002:
        return y
    lat = _rb_latency()
    o = _rb_pitch_run(y, factor)
    if o is None:
        return y
    k = int(0.5 * SR) + lat
    o = o[k:k + len(y)]
    if len(o) < len(y):
        o = np.vstack([o, np.zeros((len(y) - len(o), 2), np.float32)])
    return o.astype(np.float32)


def _glide(a: np.ndarray, b: np.ndarray, m: int, rho: np.ndarray, glide: bool) -> np.ndarray:
    """The blend (built on B's timeline: a = A's part, b = B's part, A's playback rate there = rho) played on an
    output timeline whose tempo glides from A's own to B's own (smoothstep, whole blend), with each song's
    pitch put back. Returns the output-timeline audio (length differs from m when the tempos differ)."""
    r0 = 1.0 / float(np.mean(rho[:max(1, min(len(rho), SR // 2))])) if glide else 1.0
    if abs(r0 - 1.0) < 0.002:
        M, pos, r = m, np.arange(m, dtype=np.float64), np.ones(m)
    else:
        M = int(round(m * 2.0 / (r0 + 1.0)))
        u = np.linspace(0, 1, M)
        r = r0 + (1.0 - r0) * (u * u * (3 - 2 * u))                   # B-timeline seconds per output second
        pos = np.concatenate([[0.0], np.cumsum(r)[:-1]])
        pos *= (m - 1) / pos[-1]                                     # lands exactly where B carries on
    xs = np.arange(m)

    def at(y):
        return np.stack([np.interp(pos, xs, y[:, c]) for c in range(2)], 1).astype(np.float32)

    a_o, b_o = (at(a), at(b)) if M != m else (a, b)
    pa = np.interp(pos, xs, rho) * r                                 # A's pitch as played (resampled twice)
    _rb_latency()                                                    # calibrate once, before the threads
    if M != m:                                                       # both songs' fixes at once (separate ffmpegs)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as ex:
            fa, fb = ex.submit(_pitch_fix, a_o, 1.0 / pa), ex.submit(_pitch_fix, b_o, 1.0 / r)
            a_o, b_o = fa.result(), fb.result()
    else:
        a_o = _pitch_fix(a_o, 1.0 / pa)
    return a_o + b_o


def _take_speed(y, t0: float, n: int, speed: float, ramp: bool = False, ramp_n: int | None = None) -> np.ndarray:
    """`n` samples of `y` from t0 played at `speed` x tempo, pitch unchanged. With `ramp` the tempo
    glides from normal to `speed` over the first `ramp_n` samples (default: all), then holds."""
    if abs(speed - 1.0) < 1e-4:
        return _take(y, t0, n=n)
    rn = n if ramp_n is None else min(n, ramp_n)
    curve = np.concatenate([_speed_curve(rn, speed, ramp), np.full(n - rn, speed)])
    if STRETCH == "resample":                  # DJ pitch fader: tempo and pitch move together
        pos = np.concatenate([[0.0], np.cumsum(curve)[:-1]])
        src = _take(y, t0, n=int(pos[-1]) + 4)
        xs = np.arange(len(src))
        return np.stack([np.interp(pos, xs, src[:, c]) for c in range(2)], 1).astype(np.float32)
    if STRETCH in ("rubberband", "glide"):
        out = _rubberband(y, t0, n, curve)
        if out is not None:
            return out
    pos = np.concatenate([[0.0], np.cumsum(curve)[:-1]])
    src = _take(y, t0, n=int(pos[-1]) + 4096)
    return _wsola(src, pos, n)


def _split(x: np.ndarray, hz: float = 200.0):
    """(lows, rest) with perfect reconstruction (zero-phase low-pass, rest = x - lows)."""
    from scipy.signal import butter, sosfiltfilt
    lo = sosfiltfilt(butter(4, hz / (SR / 2), output="sos"), x, axis=0).astype(np.float32)
    return lo, x - lo


def _overhang(old: np.ndarray, track, t_stop: float, bars: int, speed: float = 1.0) -> np.ndarray:
    """The old song carrying on past the new drop for `bars` of its own bars: bass cut so the new
    drop owns the low end, fading out equal-power."""
    from scipy.signal import butter, sosfilt
    from .tempo import period_at
    y = _take_speed(old, t_stop, int(bars * 4 * period_at(track, t_stop) / speed * SR), speed)
    y = sosfilt(butter(2, 250 / (SR / 2), btype="high", output="sos"), y, axis=0).astype(np.float32)
    y *= np.cos(np.linspace(0, 1, len(y), dtype=np.float32)[:, None] * np.pi / 2)
    return y


def _join(out: list, seams: list, old, t_old: float, new, t_new: float, dur: float, style: str,
          A, B, riser_from: float | None) -> None:
    """Append `dur` s of `new` (from t_new) to the output, joining from `old` (playing up to t_old)."""
    global LAST_SHIFT
    LAST_SHIFT = 0
    from .tempo import period_at
    beat = period_at(A, t_old)
    bar = 4 * beat
    body = _take(new, t_new, dur)
    buf = out[-1]                                   # the last piece; its end is the seam
    n = len(buf)
    x = lambda m: np.linspace(0, 1, m, dtype=np.float32)[:, None]
    sp = _speed(A, t_old, B, t_new)                     # old song's beat nudged onto the new one's
    if style in ("cut", "natural"):
        m = int(0.012 * SR)
        tail = _take(old, t_old, n=m)
        body[:m] = body[:m] * np.sin(x(m) * np.pi / 2) + tail * np.cos(x(m) * np.pi / 2)
    elif style == "punch":
        m = int(beat * SR)
        body[:m] += _take(old, t_old, n=m) * np.cos(x(m) * np.pi / 2)
    elif style in ("xfade1", "xfade2", "xfade4"):
        m = min(len(body), int(bar * int(style[-1]) * SR))
        body[:m] = body[:m] * np.sin(x(m) * np.pi / 2) + _take_speed(old, t_old, m, sp, ramp=m / SR > RAMP_FROM) * np.cos(x(m) * np.pi / 2)
    elif style == "mix":
        m = len(body)
        a_lo, a_hi = _split(_take_speed(old, t_old, m, sp, ramp=m / SR > RAMP_FROM))
        _, b_hi = _split(body)
        f = int(0.02 * SR)
        a_lo[m - f:] *= np.cos(x(f) * np.pi / 2)                  # basses swap on the drop
        body[:] = a_hi * np.cos(x(m) * np.pi / 2) + a_lo + b_hi * np.sin(x(m) * np.pi / 2)
    elif style == "filter":
        m = min(n, int(bar * SR))
        buf[n - m:] = _lowpass_sweep(buf[n - m:]) * (1 - 0.1 * x(m))
        k = int(0.03 * SR)
        body[:k] *= x(k)
    elif style == "echo":
        d = int(beat / 2 * SR)
        src = buf[n - d:].copy()
        g = 0.55
        for r in range(1, 7):
            a = (r - 1) * d
            if a + d > len(body):
                break
            body[a:a + d] += src * (g ** r)
    elif style == "drop":
        m = min(n, int(beat * SR))
        f = int(0.015 * SR)
        buf[n - m:n - m + f] *= np.cos(x(f) * np.pi / 2)
        buf[n - m + f:] = 0
    elif style in ("dj", "djverb", "djblend"):
        # old keeps playing at full level; new (its build) rises underneath with the bass cut so the
        # low ends don't fight; the piece ends on the new song's drop, where the old one stops dead
        # (djblend: the old song's top end carries on past the drop for a bar or two).
        # 2026-11-20 fix (user: "filtering bugs" on djblend): bass is split with a zero-phase filter
        # (lows + rest == the original), so nothing jumps in phase where the filtering starts/stops,
        # and the old song is stretched in ONE pass through the overhang (no seam between passes)
        from .tempo import period_at
        m = len(body)
        oh = 0
        if style == "djblend":
            bars_oh = 1 if (t_new * 10) % 2 < 1 else 2
            oh = int(bars_oh * 4 * period_at(A, t_old + m / SR * sp) / sp * SR)
        whole = _take_speed(old, t_old, m + oh, sp, ramp=m / SR > RAMP_FROM, ramp_n=m)
        keep = whole[:m] * (1 - 0.12 * x(m) ** 3)                # barely dips, only at the very end
        f = int(0.02 * SR)
        if style != "djblend":
            keep[m - f:] *= np.cos(x(f) * np.pi / 2)              # 20 ms release right on the hit
        _, b_hi = _split(body, 220.0)
        body[:] = keep + b_hi * (0.85 * x(m) ** 1.5)
        if oh:
            lo, hi = _split(whole, 250.0)                        # continuous with the body above
            g = int(0.03 * SR)
            tail = hi[m:].copy()
            tail[:g] += lo[m:m + g] * np.cos(x(g) * np.pi / 2)    # its bass hands over in 30 ms
            tail *= np.cos(np.linspace(0, 1, len(tail), dtype=np.float32)[:, None] * np.pi / 2)
            seams.append(sum(len(p) for p in out) / SR)
            out.append(body)
            return tail
    elif style == "game":
        # Darktide's composer rules: new part at full level on the switch; the old one rings out
        ru = UP_ORDER.get(_role(B, t_new), 2) - UP_ORDER.get(_role(A, t_old - 0.1), 2)
        fo, fi = (1.0, 0.0) if ru > 0 else (5.0, 0.0) if ru < 0 else (2.5, 2.5)
        m = min(len(body), int(fo * SR))
        tail = _take(old, t_old, n=m) * (1 - x(m))                   # linear, like the game's curves
        if fi > 0:
            k = min(len(body), int(fi * SR))
            body[:k] *= x(k)
        body[:m] += tail
    elif style in LAYER_STYLES:
        m = min(len(body), int(_bars_len(B, t_new, LEAD_STYLES[style]) * SR))
        # A's outro on B's beat. Mix and stem MUST move identically or "melody = mix - stem" is phasey
        # garbage (user 2026-11-24: "sounds like water in my ears"): resample both (DJ pitch fader)
        global STRETCH
        from . import grid
        glide = STRETCH == "glide"                                  # tempo glide + pitch put back (trainer A/B)
        src = grid.warp_positions(A, t_old, B, t_new, m, SR)       # beat-for-beat: bar lines meet every bar
        keep_st, STRETCH = STRETCH, "resample"
        try:
          if src is not None:
            # 2026-11-26: the fitted grids can sit 30-60 ms apart - line the real drum hits up
            db0 = _stem(B, "drums")
            if _stem(A, "drums") is not None and db0 is not None:
                src = src + _drum_lag(grid.take_at(_stem(A, "drums"), src, SR), _take(db0, t_new, n=m), period_at(B, t_new))
            oa = grid.take_at(old, src, SR)
            sa, da = _stem(A), _stem(A, "drums")
            sa = grid.take_at(sa, src, SR) if sa is not None else None
            da = grid.take_at(da, src, SR) if da is not None and sa is not None else None
          else:
            oa = _take_speed(old, t_old, m, sp, ramp=m / SR > RAMP_FROM)
            sa, da = _stem(A), _stem(A, "drums")
            sa = _take_speed(sa, t_old, m, sp, ramp=m / SR > RAMP_FROM) if sa is not None else None
            da = _take_speed(da, t_old, m, sp, ramp=m / SR > RAMP_FROM) if da is not None and sa is not None else None
        finally:
            STRETCH = keep_st
        global LAST_LOCK
        LAST_LOCK = src is not None                                  # beat-for-beat, or the beats run free
        sb = _stem(B)
        nb = body[:m].copy()
        db_ = _stem(B, "drums")
        a_dr, a_bs, a_ml = _layers(oa, sa, da)
        b_dr, b_bs, b_ml = _layers(nb, _take(sb, t_new, n=m) if sb is not None else None,
                                   _take(db_, t_new, n=m) if sb is not None and db_ is not None else None)
        L = LEAD_STYLES[style]
        layers = {"A drums": a_dr, "A bass": a_bs, "A melody": a_ml, "B drums": b_dr, "B bass": b_bs, "B melody": b_ml}
        ref_db = 20 * np.log10(np.sqrt(np.mean(buf[max(0, n - int(1.5 * SR)):] ** 2)) + 1e-9)
        _gp = gen_parse(style)
        loud = bool(_gp and _gp.get("loud"))                         # a loud tease: hook/sound at full level
        plan, gap_at = _adapt_plan(layer_plan(style, lands_on_drop(B, t_new + m / SR)), layers, ref_db, L, m,
                                   keep_beat=src is not None and not style.startswith("breakdown"))   # 2026-11-26: unlocked = never refill A's beat over B
        if TWEAKS:
            plan = _tw_plan(plan, L)
        P = {p["lane"]: _env(p["pts"], m) for p in plan}
        # user 2026-11-26: never two vocals on top of each other. Where both songs sing at once, A's
        # (outgoing) melody layer ducks out on beat-long fades; hooks/sounds that sing are gated the same way.
        a_times = src if src is not None else t_old + np.arange(m) / SR * sp
        vg = _VoxGuard(A, a_times, B, t_new, m, period_at(B, t_new))
        duck = vg.duck_a(P["A melody"], P["B melody"])
        if duck is not None:
            a_ml *= duck
            plan.append({"lane": "A vocal ducked", "pts": vg.pts(1 - duck)})
        if TWEAKS:
            layers = {k: v * _tw_gain(k) for k, v in layers.items()}
        mix = sum(layers[k] * P[k] for k in layers)
        a_acc = sum(layers[k] * P[k] for k in layers if k.startswith("A ")).copy() if glide else None
        from . import hooks
        hb, ha = hooks.near(B, t_new + m / SR), hooks.near(A, t_old)
        if "B hook" in P and hb is not None:                       # the hook plays from ITS start, where its lane starts
            x0 = next((x for x, v in next(p for p in plan if p["lane"] == "B hook")["pts"] if v > 0), 0)
            g0 = int(round(x0 * L) / L * m)                            # on the bar line, never mid-beat
            _, h = _split(hooks.melodic(B, hb["t"], m - g0), 250.0)
            h = _hook_level(h, mix[g0:], rel_db=0.0 if loud else -6.0)    # user 2026-11-25: try it LOUD
            on = np.flatnonzero(P["B hook"][g0:, 0] > 0.05)
            span = int(on[-1]) + 1 if len(on) else m - g0
            h *= vg.gate(B, hb["t"], g0, span, P["A melody"], P["B melody"])
            hg = P["B hook"][g0:] / 0.7 if _tw_pts("B hook") else (1.0 if loud else np.minimum(1.0, P["B hook"][g0:] / 0.7))
            mix[g0:] += h * hg * _tw_gain("B hook")
        if gap_at is not None and hb is not None:                  # B announces itself in A's gap
            g0 = int(gap_at * m)
            k = min(m - g0, int(_bars_len(B, t_new, 2) * SR))
            _, h = _split(hooks.melodic(B, hb["t"], k), 250.0)
            h = _hook_level(h, mix[g0:g0 + k] + 1e-4, rel_db=-3.0)     # in a gap: the hook carries the moment
            h *= vg.gate(B, hb["t"], g0, k, P["A melody"], P["B melody"])
            fade = np.minimum(1, np.minimum(np.arange(k) / (0.05 * SR), (k - np.arange(k)) / (0.3 * SR)))
            mix[g0:g0 + k] += h * fade.astype(np.float32)[:, None] * _tw_gain("B hook")
            plan.append({"lane": "B hook", "pts": [(0, 0), (gap_at, 0), (gap_at + 0.01, 0.7), (gap_at + k / m - 0.02, 0.7), (gap_at + k / m, 0)]})
        global LAST_PLAN
        LAST_PLAN = _show_actual(plan, layers, ref_db, L, m)
        if "A hook" in P:
            if ha is not None:
                x0 = next((x for x, v in next(p for p in plan if p["lane"] == "A hook")["pts"] if v > 0), 0)
                g0 = int(round(x0 * L) / L * m)
                src_h = hooks.melodic(A, ha["t"], int((m - g0) * sp) + 4096)
                if glide:                                   # resampled like the rest of A: one pitch fix for all of A
                    keep_h, STRETCH = STRETCH, "resample"
                    try:
                        _, h = _split(_take_speed(src_h, 0.0, m - g0, sp), 250.0)
                    finally:
                        STRETCH = keep_h
                else:
                    _, h = _split(_take_speed(src_h, 0.0, m - g0, sp), 250.0)
                h = _hook_level(h, mix[g0:])
                on = np.flatnonzero(P["A hook"][g0:, 0] > 0.05)
                span = int(on[-1]) + 1 if len(on) else m - g0
                h *= vg.gate(A, ha["t"], g0, span, P["A melody"], P["B melody"], speed=sp)
                add = h * (P["A hook"][g0:] / 0.5 if _tw_pts("A hook") else np.minimum(1.0, P["A hook"][g0:] / 0.5)) * _tw_gain("A hook")
                mix[g0:] += add
                if a_acc is not None:
                    a_acc[g0:] += add
        nlanes = len(plan)
        mix = _sprinkle_sounds(mix, A, B, a_ml, t_new, m, plan, L, loud=loud,    # user-isolated signature sounds
                               vg=vg, pa=P["A melody"], pb=P["B melody"], a_acc=a_acc)
        LAST_PLAN = LAST_PLAN + plan[nlanes:]                          # show them on the trainer too
        if glide:
            g = _hold_gain(mix, buf[max(0, n - int(1.5 * SR)):], ref_after=body[m:m + int(1.5 * SR)])
            g = 1.0 if g is None else g
            if src is not None:
                rho = np.gradient(src) * SR                             # A seconds per B second
                k = int(0.25 * SR)
                c = np.concatenate([[0.0], np.cumsum(np.pad(rho, (k, k), mode="edge"))])
                rho = (c[2 * k + 1:] - c[:-2 * k - 1]) / (2 * k + 1)          # running mean (fast)
            else:
                rho = _speed_curve(m, sp, ramp=m / SR > RAMP_FROM)
            g_out = _glide(a_acc * g, (mix - a_acc) * g, m, rho, glide=src is not None)
            g_out = (0.98 * np.tanh(g_out / 0.98)).astype(np.float32)
            LAST_SHIFT = len(g_out) - m
            body = np.concatenate([g_out, body[m:]])
            M = len(g_out)
        else:
            mix = _hold_level(mix, buf[max(0, n - int(1.5 * SR)):], ref_after=body[m:m + int(1.5 * SR)])
            body[:m] = mix
            M = m
        seams.append(sum(len(p) for p in out) / SR)
        seams.append(seams[-1] + M / SR)                              # B is fully in
        out.append(body)
        return None
    elif style in LEAD_STYLES:
        m = min(len(body), int(_bars_len(B, t_new, LEAD_STYLES[style]) * SR))
        xr = x(m)
        o = _take_speed(old, t_old, m, sp, ramp=m / SR > RAMP_FROM)   # the old song carries on, on the new beat
        if style.startswith("tease") or style == "hook4":
            st = _stem(B)
            dr = _take(st if st is not None else new, t_new, n=m)
            _, hats = _split(dr, 180.0)                               # snare/hats, no kick or bass
            lead = o * (1 - 0.2 * xr ** 2) + hats * (0.75 * xr ** 1.3)
            if style == "hook4":
                from . import hooks
                hk = hooks.near(B, t_new)
                k = min(m, int(_bars_len(B, t_new, 2) * SR))
                from . import vocals                                  # never a sung hook over A singing
                if hk is not None and ((vocals.sung(B, hk["t"], k / SR) and vocals.sung(A, t_old + (m - k) / SR * sp, k / SR * sp))
                                       or not vocals.whole_phrase(B, hk["t"], k / SR)):
                    hk = None
                if hk is not None:
                    k = min(m, int(_bars_len(B, t_new, 2) * SR))      # the last 2 bars before landing
                    _, mel = _split(hooks.melodic(B, hk["t"], k), 250.0)
                    g = np.linspace(0.35, 0.75, k, dtype=np.float32)[:, None]
                    lead[m - k:] = lead[m - k:] * 0.85 + mel * g
        elif style.startswith("strip"):
            lo, hi = _split(o, 220.0)
            lead = hi * (1 - 0.4 * xr) + lo * (1 - xr) ** 2           # bass/kick drain away, top dips
            if style == "strip4s":                                    # ...and snaps back to full on the last beat
                kb = min(m, int(period_at(B, t_new + m / SR - 0.1) * SR))
                f2 = int(0.01 * SR)
                lead[m - kb:m - kb + f2] = (lead[m - kb:m - kb + f2] * (1 - x(f2)) + o[m - kb:m - kb + f2] * x(f2))
                lead[m - kb + f2:] = o[m - kb + f2:]
            rz = _callout(B, "riser", m / SR)
            if rz is not None:
                k = min(len(rz), m)
                lead[m - k:] += rz[len(rz) - k:] * (0.9 * np.linspace(0.3, 1, k, dtype=np.float32)[:, None])
            else:                                                     # no riser: the new song's own lead-up, filtered
                _, nh = _split(_take(new, t_new, n=m), 300.0)
                lead += nh * (0.6 * xr ** 2)
        else:                                                         # fill1
            lo, hi = _split(o, 220.0)
            lead = hi * 0.85 + lo * (1 - xr)
            fl = _callout(B, "fill", m / SR)
            if fl is None:
                st = _stem(B)
                fl = _take(st if st is not None else new, t_new, n=m)
            k = min(len(fl), m)
            lead[m - k:] += fl[len(fl) - k:] * 0.9
        lead = _hold_level(lead, buf[max(0, n - int(1.5 * SR)):],     # user 2026-11-24: lead-ins dipped 6-10 dB
                           ref_after=body[m:m + int(1.5 * SR)])           # ...and never louder than where it lands
        f = int(0.02 * SR)
        lead[m - f:] *= np.cos(x(f) * np.pi / 2)                     # the old song lets go right on the landing
        body[:m] = lead
        seams.append(sum(len(p) for p in out) / SR)
        seams.append(seams[-1] + m / SR)                              # the landing
        out.append(body)
        return None
    elif style == "hookout":
        m = min(len(body), int(bar * SR))                             # 1-bar blend into the new part
        body[:m] = body[:m] * np.sin(x(m) * np.pi / 2) + _take_speed(old, t_old, m, sp) * np.cos(x(m) * np.pi / 2)
        from . import hooks
        hk = hooks.near(A, t_old)
        k = min(len(body), int(_bars_len(B, t_new, 2) * SR))
        from . import vocals                                          # never a sung hook over B singing
        if hk is not None and ((vocals.sung(A, hk["t"], k / SR * sp) and vocals.sung(B, t_new, k / SR))
                               or not vocals.whole_phrase(A, hk["t"], k / SR * sp)):
            hk = None
        if hk is not None:                                            # the old song's hook trails over 2 bars
            k = min(len(body), int(_bars_len(B, t_new, 2) * SR))
            src = hooks.melodic(A, hk["t"], int(k * sp) + 4096)
            mel = _take_speed(src, 0.0, k, sp)
            _, mel = _split(mel, 250.0)
            body[:k] += mel * (0.6 * (1 - x(k)) ** 1.5)
    elif style == "riser" and riser_from is not None:
        m = min(n, int(2 * bar * SR))
        lay = _take(new, riser_from, n=m)
        ramp = x(m) ** 1.5
        buf[n - m:] = buf[n - m:] * (1 - 0.45 * ramp) + lay * (0.9 * ramp)
    seams.append(sum(len(p) for p in out) / SR)
    out.append(body)
    return None


def render(lib_by, route, out_mp3: str) -> list[float]:
    """(route["stretch"] picks the tempo-matching method for this clip: an A/B variable.)"""
    global STRETCH
    keep, STRETCH = STRETCH, route.get("stretch") or STRETCH
    try:
        return _render(lib_by, route, out_mp3)
    finally:
        STRETCH = keep


def render_audio(lib_by, route) -> tuple[np.ndarray, list[float]]:
    """The rendered change as (stereo float array at SR, seam times) - no files (live critic scoring)."""
    global STRETCH
    keep, STRETCH = STRETCH, route.get("stretch") or STRETCH
    try:
        return _render(lib_by, route, None)
    finally:
        STRETCH = keep


def _render(lib_by, route, out_mp3: str | None):
    r = as_route(route)
    A = lib_by[r["a"]]
    cur, t_old = A, r["t_out"]
    out, seams = [_take(_audio(A), t_old - PRE, PRE)], []
    steps = r["steps"]
    wash = None
    for k, s in enumerate(steps):
        T = lib_by[s["b"]]
        style = s["style"] if s["style"] != "auto" else _auto(cur, t_old, T, s["t_in"])
        register_gen(style)
        if k + 1 < len(steps):
            dur = (s["until"] - s["t_in"]) if s.get("until") else _bars_len(T, s["t_in"], s["bars"] or 2)
        else:
            dur = POST
        if style in LEAD_STYLES:
            dur += _bars_len(T, s["t_in"], LEAD_STYLES[style])      # the announcing, then POST after landing
        riser = _builds_into(T, T.segment_at(s["t_in"] + 0.05)) if style == "riser" else None
        if style == "natural":                      # continuous audio: extend the previous piece
            nxt = _take(_audio(T), s["t_in"], dur)
            if wash is not None:                    # the old song's reverb tail over the drop
                w = wash[:len(nxt)]
                nxt[:len(w)] += w
                wash = None
            out[-1] = np.vstack([out[-1], nxt])
            seams.append(sum(len(p) for p in out) / SR - dur)
        else:
            tail = _join(out, seams, _audio(cur), t_old, _audio(T), s["t_in"], dur, style, cur, T, riser)
            sp = _speed(cur, t_old, T, s["t_in"])
            if style == "djverb":
                wash = _reverb_wash(_audio(cur), t_old + dur * sp, out[-1])
            elif style == "djblend":
                wash = tail                         # the old song's top end, carrying on past the drop
        cur, t_old = T, s["t_in"] + dur
    clip = np.vstack(out)
    clip /= max(1.0, float(np.abs(clip).max()) / 0.95)
    if out_mp3 is None:
        return clip, [round(x, 2) for x in seams]
    wav = out_mp3[:-4] + ".wav"
    import soundfile as sf
    sf.write(wav, clip, SR)
    subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-b:a", "160k", "-y", out_mp3], check=True)
    os.remove(wav)
    return [round(x, 2) for x in seams]
