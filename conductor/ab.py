"""A/B join training: each night, 15 pairs of "two ways the music could change at this moment";
you pick the better one, the taste model learns, and tomorrow's pairs are the ones it's least
sure about.

    python -m conductor.ab nightly     # learn from answers, retrain, make tomorrow's pairs
    python -m conductor.ab pairs [n]   # just make pairs

Files (served by review_server):  review/ab/pairs.json · review/ab/clips/*.mp3 · review/ab/answers.json
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import random
import subprocess
import sys
import time

import numpy as np

from . import joins

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = os.path.join(HERE, "review", "ab")
TASTE = os.path.join(HERE, "cache", "taste_model.pkl")
SR = 44100
PRE, POST = 4.0, 6.0
CRITIC_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv-ml", "Scripts", "python.exe")
CRITIC_PT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "critic", "critic.pt")   # 2026-11-19: 4 s lead-in (was 6) - more picks per minute
STATE_NAME = {"explore": "Calm", "tier1": "Fight", "tier2": "Horde", "tier3": "Boss"}
UP_ORDER = {"explore": 0, "bridge": 0, "build": 1, "skirmish": 2, "horde": 3, "peak": 4}


# ── taste model (your judgements) ───────────────────────────────────────────
_RANK = {"calm": 0, "build": 1, "fight": 2, "horde": 3, "boss": 4}
USE_DIR = os.environ.get("CONDUCTOR_TASTE_DIR", "0") == "1"    # tested 2026-11-23: no gain for the trees (62-65% either way)


def direction(prev, st) -> str | None:
    """'up' / 'down' / 'same' for a state change (names any case), None if unknown."""
    a, b = _RANK.get(str(prev).lower()), _RANK.get(str(st).lower())
    if a is None or b is None:
        return None
    return "up" if b > a else "down" if b < a else "same"


def _ctx_dir(ctx) -> str | None:
    try:
        a, b = [s.strip() for s in str(ctx).split("→")]
        return direction(a, b)
    except ValueError:
        return None


def _feat(lib_by, j) -> dict:
    from . import routes
    f = routes.features(lib_by, j)
    if USE_DIR:          # 2026-11-23: the game's direction (calm-downs were coin flips without it)
        d = j.get("dir")
        f["dir_up"], f["dir_down"], f["dir_same"] = float(d == "up"), float(d == "down"), float(d == "same")
    return f


def labelled(lib) -> list[dict]:
    """Every join you've judged: rated review moments + A/B answers."""
    from .ratedjoins import joins_from_reviews
    from . import routes
    # mission moments judge the WHOLE live system (what it chose, when, how): they count most
    rows = [dict(r, w=2.5) for r in joins_from_reviews(lib)]
    try:                                                       # 👍/👎 pressed in game on a live change
        from .changelog import rows as _ingame
        rows += _ingame(lib)
    except Exception as e:
        print("in-game ratings skipped:", e)
    by = {t.title: t for t in lib.tracks}
    for a in _answers():
        p = a["pair"]
        d = _ctx_dir(p.get("context"))
        p = dict(p, a=dict(p["a"], dir=d), b=dict(p["b"], dir=d))
        if a["choice"] == "marks_only":
            for h, m in zip(routes.hops(by, p["a"]), a["hops"]["a"] or []):
                if m:
                    rows.append(dict(h, dir=d, good=int(m > 0), w=1.8))
            continue
        for side, ms in (a.get("hops") or {}).items():       # a single change you marked on its own
            for h, m in zip(routes.hops(by, p[side]), ms or []):
                if m:
                    rows.append(dict(h, dir=d, good=int(m > 0), w=1.8))
        if a["choice"] in ("a", "b"):
            win, lose = (p["a"], p["b"]) if a["choice"] == "a" else (p["b"], p["a"])
            kind = "how" if str(p.get("quad", "")).startswith("h") else "dest"
            rows += [dict(win, good=1, w=2.0, _kind=kind), dict(lose, good=0, w=2.0, _kind=kind)]    # clean labels count double
        elif a["choice"] == "both_bad":
            rows += [dict(p["a"], good=0, w=1.5), dict(p["b"], good=0, w=1.5)]
        elif a["choice"] == "both_good":
            rows += [dict(p["a"], good=1, w=1.5), dict(p["b"], good=1, w=1.5)]
    return rows


def train_taste(lib, save: bool = True) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    by = {t.title: t for t in lib.tracks}
    rows = [r for r in labelled(lib) if r["a"] in by and r["b"] in by]
    F = [_feat(by, r) for r in rows]
    keys = sorted(F[0])
    X = np.array([[f[k] for k in keys] for f in F]); y = np.array([r["good"] for r in rows])
    w = np.array([r.get("w", 1.0) for r in rows])
    # user 2026-11-20: build-up-into-drop routes were in every set and always rated well, so the model
    # learned "always ride a drop". Count them half so the other decisions carry weight.
    from . import routes as _routes
    w = w * np.array([0.5 if len(_routes.as_route(r)["steps"]) > 1 else 1.0 for r in rows])
    # one group per moment, so a moment never trains its own test. 2026-11-25: a STABLE hash - Python's hash()
    # changes every run, which moved the folds and the scores by 3-4 points between identical runs
    import zlib
    g = np.array([zlib.crc32(f"{r['a']}|{round(r['t_out'])}".encode("utf-8")) for r in rows])
    mk = lambda: HistGradientBoostingClassifier(max_depth=2, max_iter=150, learning_rate=0.05, l2_regularization=2.0)
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        p[te] = mk().fit(X[tr], y[tr], sample_weight=w[tr]).predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, p))
    # the honest test: on A/B pairs it never saw, does it pick the one you picked?
    ab_rows = [k for k, r in enumerate(rows) if r.get("w") == 2.0]
    hits = [p[ab_rows[k]] > p[ab_rows[k + 1]] for k in range(0, len(ab_rows) - 1, 2)]
    split = {}
    for kind in ("dest", "how"):                 # 2026-11-25: where-to-go and how-to-join questions apart
        hk = [p[ab_rows[k]] > p[ab_rows[k + 1]] for k in range(0, len(ab_rows) - 1, 2) if rows[ab_rows[k]].get("_kind") == kind]
        if hk:
            split[f"ab_{kind}_matched"] = f"{sum(hk)}/{len(hk)} ({100 * sum(hk) / len(hk):.0f}%)"
    model = mk().fit(X, y, sample_weight=w)
    if save:
        if os.path.exists(TASTE):                # keep the previous model (2026-11-25: one got overwritten unbacked)
            import shutil
            shutil.copy2(TASTE, TASTE.replace(".pkl", ".prev.pkl"))
        pickle.dump({"model": model, "keys": keys, "auc": auc, "n": len(y)}, open(TASTE, "wb"))
    return {"labelled_joins": len(y), "auc_unseen": round(auc, 3),
            "ab_picks_matched": f"{sum(hits)}/{len(hits)}" + (f" ({100 * sum(hits) / len(hits):.0f}%)" if hits else ""), **split}


_TM = None


_TM_MTIME = None


def _taste_model():
    """The trained taste model, reloaded whenever a training round rewrites it (live app too)."""
    global _TM, _TM_MTIME
    m = os.path.getmtime(TASTE)
    if _TM is None or m != _TM_MTIME:
        _TM, _TM_MTIME = pickle.load(open(TASTE, "rb")), m
    return _TM


def taste(lib_by, j) -> float:
    tm = _taste_model()
    f = _feat(lib_by, j)
    return float(tm["model"].predict_proba(np.array([[f[k] for k in tm["keys"]]]))[0, 1])


def taste_many(lib_by, routes_, dir: str | None = None) -> list[float]:
    """Score many candidate changes in one go (live planning). dir = the game's up/down/same."""
    tm = _taste_model()
    X = []
    for r in routes_:
        f = _feat(lib_by, dict(r, dir=dir) if dir else r)
        f.setdefault("dir_up", 0.0); f.setdefault("dir_down", 0.0); f.setdefault("dir_same", 0.0)
        X.append([f[k] for k in tm["keys"]])
    return tm["model"].predict_proba(np.array(X))[:, 1].tolist() if X else []


# ── clips ───────────────────────────────────────────────────────────────────
_AUDIO: dict = {}


def _audio(track) -> np.ndarray:
    if track.path not in _AUDIO:
        import librosa
        from .players import _level_gain
        y, _ = librosa.load(track.path, sr=SR, mono=False)
        y = np.atleast_2d(y)
        y = np.vstack([y, y]) if y.shape[0] == 1 else y[:2]
        y = np.ascontiguousarray(y.T, dtype=np.float32)
        y *= _level_gain(track.path, y, SR)
        _AUDIO[track.path] = y
        if len(_AUDIO) > 12:
            _AUDIO.pop(next(iter(_AUDIO)))
    return _AUDIO[track.path]


def render_clip(lib_by, j, out_mp3: str) -> None:
    """6 s before the change + 6 s after, joined the way the game does it: going up = a cut on the
    beat with a one-beat ring-out; going down = a one-bar equal-power crossfade after the beat."""
    from .states import role_of
    from .tempo import period_at
    A, B = lib_by[j["a"]], lib_by[j["b"]]
    a, b = _audio(A), _audio(B)
    o, i = int(j["t_out"] * SR), int(j["t_in"] * SR)
    pre = a[max(0, o - int(PRE * SR)):o]
    pre = np.vstack([np.zeros((int(PRE * SR) - len(pre), 2), np.float32), pre])
    up = UP_ORDER.get(role_of(B.segment_at(j["t_in"] + 0.05)) or "", 2) >= UP_ORDER.get(role_of(A.segment_at(j["t_out"] - 0.05)) or "", 2)
    fade = period_at(A, j["t_out"]) * (1 if up else 4)
    nf = int(fade * SR)
    post = b[i:i + int(POST * SR)]
    post = np.vstack([post, np.zeros((int(POST * SR) - len(post), 2), np.float32)])
    tail = a[o:o + nf]
    tail = np.vstack([tail, np.zeros((nf - len(tail), 2), np.float32)])
    x = np.linspace(0, 1, nf, dtype=np.float32)[:, None]
    mix = post.copy()
    if up:
        mix[:nf] += tail * np.cos(x * np.pi / 2)              # old part rings one beat
    else:
        mix[:nf] = post[:nf] * np.sin(x * np.pi / 2) + tail * np.cos(x * np.pi / 2)
    clip = np.vstack([pre, mix])
    clip /= max(1.0, float(np.abs(clip).max()) / 0.95)
    wav = out_mp3[:-4] + ".wav"
    import soundfile as sf
    sf.write(wav, clip, SR)
    subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-b:a", "160k", "-y", out_mp3], check=True)
    os.remove(wav)


# ── pairs ───────────────────────────────────────────────────────────────────
# 2026-11-25: "how" sets rendered before the beat-locked batch (21:10, 24 Nov) were judged on blends with
# known bugs (filter-sweep kick, off-beat drift, volume dips/surges, off-bar drops). Training skips them.
STALE_HOW_BEFORE = 1795575010


def stale(qid: str) -> bool:
    import re as _re
    m = _re.match(r"h(\d+)_", qid or "")
    return bool(m) and int(m.group(1)) < STALE_HOW_BEFORE


def fine_set(a) -> set:
    """Options you'd still be happy with in game (2026-11-24 "✓ Fine" mark; the best always is).
    Older answers have no marks: only the best."""
    return set(a.get("fine") or []) | ({a["best"]} if a.get("best") is not None else set())


def quad_pairs(a) -> list[tuple[int, int]]:
    """(winner, loser) head-to-heads a best/worst/fine answer settles. Your favourite beats every other
    option (a style choice, even over fine ones). The middle ones beat the worst - unless the worst was
    marked fine and they weren't. No worst (all others fine) = only the favourite's wins."""
    b, w = a.get("best"), a.get("worst")
    n = len(a["quad"]["options"])
    if b is None or b == w:
        return []
    fine = fine_set(a)
    others = [k for k in range(n) if k != b]
    mid = [k for k in others if k != w]
    return [(b, k) for k in others] + ([(k, w) for k in mid if not (w in fine and k not in fine)] if w is not None else [])


def _rank(a):
    """clip -> 2 best, 1 middle, 0 worst (None if the answer settles nothing)."""
    b, w = a.get("best"), a.get("worst")
    if b is None or b == w:
        return None
    return {c: (2 if k == b else 0 if k == w else 1) for k, c in enumerate(a["quad"]["clips"])}


def _answers() -> list[dict]:
    """All answers as A/B pairs. A four-way answer (best + worst of 4 options) becomes the 5
    head-to-heads it settles: best beats each of the other 3, the middle 2 each beat the worst."""
    p = os.path.join(AB, "answers.json")
    raw = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []
    out = []
    nz = noisy()
    for a in raw:
        if a.get("kind") != "quad":
            out.append(a)
            continue
        if a["quad"].get("repeat_of"):          # blind repeats only measure your own consistency (consistency())
            continue
        if stale(a["quad"]["id"]):
            continue
        q, opts = a["quad"], a["quad"]["options"]
        mk = lambda i, j, ch, hops=None: {"pair": {"id": f"{q['id']}_{i}{j}", "context": q["context"], "a": opts[i], "b": opts[j],
                                                   "model_pick": ("a" if q.get("model_pick") == i else "b" if q.get("model_pick") == j else None),
                                                   "quad": q["id"]},
                                          "choice": ch, "hops": hops or {}, "at": a.get("at")}
        best, worst = a.get("best"), a.get("worst")
        if a.get("all") in ("bad", "good"):
            ch = "both_bad" if a["all"] == "bad" else "both_good"
            out += [mk(0, 1, ch), mk(2, 3, ch)]
        elif best is not None and best != worst:
            fine = fine_set(a)
            for x, y in training_pairs(a, nz):
                p = mk(x, y, "a")
                if a.get("fine") is not None:
                    p["fine"] = {"a": x in fine, "b": y in fine}
                out.append(p)
        for k, ms in (a.get("hops") or {}).items():      # per-change marks, once per option
            if any(ms or []):
                k = int(k)
                out.append(mk(k, k, "marks_only", {"a": ms}))
    return out


def _fits(lib, j, phase) -> bool:
    """The music it actually landed on suits the game state (old timelines / phase lag can label a
    peak as 'calm' - those pairs just waste a pick)."""
    from .states import POOL, role_of, serves
    state = {"explore": "calm", "tier1": "fight", "tier2": "horde", "tier3": "boss"}[phase]
    T = next((t for t in lib.tracks if t.title == j["b"]), None)
    s = next((s for s in (T.segments if T else []) if s.start - 0.5 <= j["t_in"] < s.start + s.length), None)
    return s is not None and serves(s, state)


def _moments(lib):
    """Real joins from every rendered mission, with the game situation around them."""
    from .ratedjoins import joins_in
    out = []
    for tlp in glob.glob(os.path.join(HERE, "review", "m*", "timeline_conductor_v3.json")):
        tl = json.load(open(tlp, encoding="utf-8"))
        ph = tl.get("phases", [])
        for j in joins_in(tl, lib, 120, tl.get("duration", 1e9)):
            before = [p["phase"] for p in ph if p["t"] <= j["heard"] - 1]
            after = [p["phase"] for p in ph if p["t"] <= j["heard"] + 1]
            if before and after and after[-1] in STATE_NAME and _fits(lib, j, after[-1]) and (
                    before[-1] not in STATE_NAME or _fits(lib, {"b": j["a"], "t_in": j["t_out"] - 0.5}, before[-1])):
                j["context"] = f"{STATE_NAME.get(before[-1], before[-1])} → {STATE_NAME[after[-1]]}"
                j["state_role"] = after[-1]
                out.append(j)
    return out


def _synthetic(lib, rng, k: int = 40) -> list[dict]:
    """Big jumps the recordings rarely contain: from a calm moment straight to a horde or boss."""
    from .states import POOL, StateDirector, role_of, serves
    by = {t.title: t for t in lib.tracks}
    out = []
    # user 2026-11-19: the recordings only ever played 4-5 songs, so every pair sounded the same.
    # Start from ANY song in the library, in a part that suits the starting state.
    phases = ("explore", "tier1", "tier2", "tier3")
    st_of = {"explore": "calm", "tier1": "fight", "tier2": "horde", "tier3": "boss"}
    for _ in range(k * 6):
        T = rng.choice(lib.tracks)
        frm = rng.choice(phases)
        segs0 = [s for s in T.segments if serves(s, st_of[frm]) and s.end - s.start >= 10]
        if not segs0:
            continue
        s0 = rng.choice(segs0)
        pos = rng.uniform(s0.start + 6, s0.end - 2)
        b4 = [frm]
        if True:
            tgt = rng.choice([k for k in ("explore", "tier1", "tier2", "tier3", "tier3") if k != b4[-1]])
            st = {"explore": "calm", "tier1": "fight", "tier2": "horde", "tier3": "boss"}[tgt]
            D = rng.choice([T] + rng.sample(lib.tracks, 2))
            segs = [s for s in D.segments if serves(s, st) and s.end - s.start >= 6]
            if not segs:
                continue
            out.append({"a": T.title, "t_out": T.next_bar(pos), "b": D.title, "t_in": StateDirector._land(D, rng.choice(segs)),
                        "context": f"{STATE_NAME[b4[-1]]} → {STATE_NAME[tgt]}", "state_role": tgt, "synthetic": True})
    rng.shuffle(out)
    return out[:k]


def make_pairs(lib, n: int = 15, seed: int | None = None) -> list[dict]:
    """Each pair: two routes for the same moment. Half compare the director's way with another way,
    half compare two other ways. Priority: big jumps, joining styles you've judged least, then
    where the taste model is least sure."""
    from . import routes
    from collections import Counter
    rng = random.Random(seed if seed is not None else int(time.time()))
    by = {t.title: t for t in lib.tracks}
    answers = _answers()
    seen = {(routes.describe(a["pair"]["a"]), routes.describe(a["pair"]["b"])) for a in answers}
    judged = Counter()
    for a in answers:
        for side in ("a", "b"):
            for st in routes.as_route(a["pair"][side])["steps"]:
                judged[st["style"]] += 1
    moments = _moments(lib)
    rng.shuffle(moments)
    moments = moments[:40] + _synthetic(lib, rng, 160)
    state_of = {"explore": "calm", "tier1": "fight", "tier2": "horde", "tier3": "boss"}
    cands = []
    for j in moments:
        if j["t_out"] < PRE + 0.5:
            continue                                   # lead-in would start before the song does
        st_now = state_of[j["state_role"]]
        prev = {v: k for k, v in STATE_NAME.items()}.get(j["context"].split(" → ")[0])
        down = prev is not None and routes.STATE_RANK[state_of[prev]] > routes.STATE_RANK[st_now]
        try:
            nat = routes.natural_routes(lib, j, st_now, down, rng)
            alts = routes.candidates(lib, j, st_now, rng)
        except Exception:
            continue
        # user 2026-11-19: within a state (fight->fight, boss->boss) changing song is jarring however
        # well it's joined - those moments only compare moves inside the song already playing
        if prev is not None and state_of[prev] == st_now:
            nat = [r for r in nat if all(s["b"] == j["a"] for s in r["steps"])]
            alts = [r for r in alts if all(s["b"] == j["a"] for s in routes.as_route(r)["steps"])]
            if j["b"] != j["a"]:
                j = dict(j, b=None)
        if not nat:
            continue
        base = routes.as_route({k: j[k] for k in ("a", "t_out", "b", "t_in")}) if j["b"] else None
        # user 2026-11-19: let the model make the call - half the pairs put ITS favourite for this
        # moment against another option, so each pick also scores the model's own decision
        try:
            champ = max(nat + alts[:8], key=lambda r: taste(by, r))
        except Exception:
            champ = None
        for _ in range(4):
            # the question now: WHICH of the song's own moves, entered how far before the boundary.
            # 1 in 4 pairs keeps a straight splice in as a control.
            other = [r for r in nat if r["b"] != j["a"]]      # changing songs: our hardest problem
            # user 2026-11-19: get better at moving WITHIN one song too (calm to boss in one song)
            x = rng.choice(other) if other and rng.random() < 0.45 else rng.choice(nat)
            y = (rng.choice(other) if other and rng.random() < 0.4 else rng.choice(nat)) if rng.random() < 0.75 else (base if base and (rng.random() < 0.5 or not alts) else rng.choice(alts or nat))
            if champ is not None and rng.random() < 0.5:
                x = champ
            if x is y or routes.describe(x) == routes.describe(y) or (routes.describe(x), routes.describe(y)) in seen:
                continue
            try:
                sx, sy = taste(by, x), taste(by, y)
            except Exception:
                continue
            fresh = sum(1.0 / (1 + judged[s["style"]]) for r in (x, y) for s in r["steps"])
            jump = abs(routes.features(by, x)["jump"])
            cands.append((abs(sx - sy) - 0.6 * fresh - 0.08 * jump, j, x, y, abs(sx - sy), max(sx, sy),
                          "a" if x is champ else None))
    # don't spend your time on pairs where both sides look bad (late pairs were 48% "both bad")
    if cands:
        floor = float(np.percentile([c[5] for c in cands], 40))
        cands = [c for c in cands if c[5] >= floor]
    cands.sort(key=lambda c: c[0])
    pairs, used = [], set()
    starts, anywhere = Counter(), Counter()             # variety: no song dominates a batch
    for _, j, x, y, unsure, _best, mp in cands:
        key = (j["a"], round(j["t_out"]))
        if key in used:
            continue
        songs = {j["a"]} | {s["b"] for r in (x, y) for s in routes.as_route(r)["steps"]}
        if starts[j["a"]] >= 2 or any(anywhere[s] >= 4 for s in songs):
            continue
        used.add(key)
        starts[j["a"]] += 1
        anywhere.update(songs)
        if rng.random() < 0.5:
            x, y = y, x
            mp = {"a": "b", "b": "a"}.get(mp)
        pid = f"{int(time.time())}_{len(pairs)}"
        pairs.append({"id": pid, "context": j["context"], "a": x, "b": y, "seam": PRE, "model_pick": mp,
                      "clip_a": f"ab/clips/{pid}_a.mp3", "clip_b": f"ab/clips/{pid}_b.mp3",
                      "unsure": round(unsure, 3), "synthetic": bool(j.get("synthetic"))})
        if len(pairs) >= n:
            break
    rng.shuffle(pairs)                                     # position in the batch says nothing about quality
    for k, p in enumerate(pairs):
        p["id"] = f"{p['id'].split('_')[0]}_{k}"
        p["clip_a"], p["clip_b"] = f"ab/clips/{p['id']}_a.mp3", f"ab/clips/{p['id']}_b.mp3"
    os.makedirs(os.path.join(AB, "clips"), exist_ok=True)
    for p in pairs:
        p["seams_a"] = routes.render(by, p["a"], os.path.join(HERE, "review", p["clip_a"]))
        p["seams_b"] = routes.render(by, p["b"], os.path.join(HERE, "review", p["clip_b"]))
    json.dump(pairs, open(os.path.join(AB, "pairs.json"), "w", encoding="utf-8"), indent=1)
    return pairs


def _landing(r):
    from . import routes
    st = routes.as_route(r)["steps"]
    return st[-1]["b"], float(st[-1]["t_in"]), st[0]["style"]


def twins(r1, r2, secs: float = 4.0) -> bool:
    """Same destination, same style, landing < `secs` apart: only the timing differs, which you can't hear
    well enough to choose between (user 2026-11-23). Never show both in one set."""
    b1, t1, s1 = _landing(r1)
    b2, t2, s2 = _landing(r2)
    return b1 == b2 and s1 == s2 and abs(t1 - t2) < secs


def quiet_at(track, t: float, secs: float = 4.0, frac: float = 0.3) -> bool:
    """The `secs` before t are near-silent compared with the song's typical loudness."""
    e = np.asarray(track.energy, float)
    a, b = int(max(0.0, t - secs)), int(t)
    if b <= a or not len(e):
        return True
    return float(np.mean(e[a:b])) < frac * float(np.median(e))


NOISY = os.path.join(HERE, "cache", "live", "noisy_sets.json")


def mark_noisy(lib=None) -> dict:
    """Answered sets that were bad questions: started from near-silence (whole set), or held timing-only twins
    (those head-to-heads). Training skips them (critic.comparisons, _answers, distill_trees)."""
    lib = lib or training_library()
    by = {t.title: t for t in lib.tracks}
    raw = json.load(open(os.path.join(AB, "answers.json"), encoding="utf-8"))
    out = {}
    for a in raw:
        if a.get("kind") != "quad":
            continue
        q, opts = a["quad"], a["quad"]["options"]
        src = by.get(opts[0]["a"])
        tw = [[i, j] for i in range(len(opts)) for j in range(i + 1, len(opts)) if twins(opts[i], opts[j])]
        quiet = bool(src is not None and quiet_at(src, opts[0]["t_out"]))
        if quiet or tw:
            out[q["id"]] = {"quiet": quiet, "twins": tw}
    # 2026-11-23 (user): a head-to-head answered BOTH ways in a blind repeat can't be told apart -> drop it;
    # one answered the same way twice is confirmed -> counts double. Indices are the ORIGINAL set's options.
    for orig_id, pairs in contradictions(raw).items():
        e = out.setdefault(orig_id, {"quiet": False, "twins": []})
        e["contra"] = pairs["contra"]
        e["confirmed"] = pairs["confirmed"]
        e["resolved"] = pairs["resolved"]
    os.makedirs(os.path.dirname(NOISY), exist_ok=True)
    json.dump(out, open(NOISY, "w", encoding="utf-8"), indent=1)
    return out


def contradictions(raw=None) -> dict:
    """original quad id -> votes over every answer to that set (first + blind repeats + tie-breakers):
      "confirmed": [[w, l], ...]       unanimous over 2+ answers -> counts double
      "contra":    [[i, j, w_first, w_majority_or_None], ...]  dropped from the first answer's pairs:
                   a split vote (a tie), or a majority that went AGAINST the first answer
      "resolved":  [[w, l], ...]       that majority, added back the right way round
    Indices are the original set's options; repeats are matched by clip."""
    raw = raw if raw is not None else json.load(open(os.path.join(AB, "answers.json"), encoding="utf-8"))
    first = {a["quad"]["id"]: a for a in raw if a.get("kind") == "quad" and not a["quad"].get("repeat_of")}
    groups = {}
    for a in raw:
        q = a.get("quad") or {}
        if a.get("kind") == "quad" and q.get("repeat_of") in first and _rank(a):
            groups.setdefault(q["repeat_of"], []).append(a)
    out = {}
    for oid, reps in groups.items():
        f = first[oid]
        if not _rank(f):
            continue
        clips = f["quad"]["clips"]
        ranks = [_rank(a) for a in [f] + reps]
        e = out.setdefault(oid, {"contra": [], "confirmed": [], "resolved": []})
        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                v = [(r[clips[i]] > r[clips[j]]) for r in ranks if r[clips[i]] != r[clips[j]]]
                if len(v) < 2:
                    continue
                fv = ranks[0][clips[i]] - ranks[0][clips[j]]
                wi, n_i = (i if fv > 0 else j) if fv else None, sum(v)
                maj = i if n_i * 2 > len(v) else j if n_i * 2 < len(v) else None
                if wi is None:                                   # first answer didn't settle it
                    if maj is not None:
                        e["resolved"].append([maj, j if maj == i else i])
                elif maj is not None and n_i in (0, len(v)):
                    e["confirmed"].append([maj, j if maj == i else i])
                elif maj is None or maj != wi:
                    e["contra"].append([i, j, wi, maj])
                    if maj is not None:
                        e["resolved"].append([maj, j if maj == i else i])
    return out


def training_pairs(a, nz) -> list[tuple[int, int]]:
    """(winner, loser) head-to-heads one first answer contributes, after the noisy marks: dropped bad
    questions and ties, doubled confirmations, and majority verdicts from tie-breaker repeats."""
    qid = a["quad"]["id"]
    out = []
    for x, y in quad_pairs(a):
        if not skip_pair(nz, qid, x, y):
            out += [(x, y)] * (2 if confirmed(nz, qid, x, y) else 1)
    e = nz.get(qid) or {}
    if not e.get("quiet"):
        out += [tuple(p) for p in e.get("resolved", []) if sorted(p) not in e.get("twins", [])]
    return out


def noisy() -> dict:
    try:
        return json.load(open(NOISY, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def skip_pair(nz: dict, qid, i: int, j: int) -> bool:
    e = nz.get(qid)
    if not e:
        return False
    lo, hi = min(i, j), max(i, j)
    return bool(e["quiet"] or [lo, hi] in e["twins"]
                or any(c[0] == lo and c[1] == hi for c in e.get("contra", [])))


def confirmed(nz: dict, qid, w: int, l: int) -> bool:
    """You picked w over l in the original AND the blind repeat: count it twice."""
    e = nz.get(qid)
    return bool(e and [w, l] in e.get("confirmed", []))


def make_quads(lib, n: int = 12, seed: int | None = None) -> list[dict]:
    """Four ways the music could change at one moment. The user picks the best and the worst, which
    settles 5 of the 6 head-to-heads. One option is always the model's own favourite."""
    from . import routes
    from collections import Counter
    rng = random.Random(seed if seed is not None else int(time.time()))
    by = {t.title: t for t in lib.tracks}
    judged = Counter()
    for a in _answers():
        for side in ("a", "b"):
            for st in routes.as_route(a["pair"][side])["steps"]:
                judged[st["style"]] += 1
    moments = _moments(lib)
    rng.shuffle(moments)
    moments = moments[:30] + _synthetic(lib, rng, 160)
    state_of = {"explore": "calm", "tier1": "fight", "tier2": "horde", "tier3": "boss"}
    inv = {v: k for k, v in STATE_NAME.items()}
    same_song = lambda r: all(s["b"] == r["a"] for s in routes.as_route(r)["steps"])
    fresh = lambda r: sum(1.0 / (1 + judged[s["style"]]) for s in routes.as_route(r)["steps"])
    sty = lambda r: routes.as_route(r)["steps"][0]["style"]

    def choose(champ, pool):
        """The favourite + 3 other ideas: 2 moves inside the song, 1 into another, styles all
        different where possible (round 1 was 46/60 "DJ mix" because it was new)."""
        rest, pick = [r for r in pool if r is not champ], []
        score = lambda r: fresh(r) + rng.random() - 1.5 * Counter(sty(x) for x in [champ] + pick)[sty(r)]
        # user 2026-11-20: build-up-into-drop options always win and drown everything else out:
        # at most ONE per set, so the sets teach the other decisions too
        is_lead = lambda r: len(routes.as_route(r)["steps"]) > 1
        leads = lambda: sum(is_lead(x) for x in [champ] + pick)
        for want in (same_song, same_song, lambda r: not same_song(r)):
            c = [r for r in rest if want(r) and r not in pick and not (is_lead(r) and leads() >= 1)]
            if c and len(pick) < 3:
                pick.append(max(c, key=score))
        while len(pick) < 3:
            c = [r for r in rest if r not in pick and not (is_lead(r) and leads() >= 1)] or [r for r in rest if r not in pick]
            pick.append(max(c, key=score))
        opts = [champ] + pick
        rng.shuffle(opts)
        return opts

    cands = []
    for j in moments:
        if j["t_out"] < PRE + 0.5:
            continue
        st_now = state_of[j["state_role"]]
        prev = inv.get(j["context"].split(" → ")[0])
        down = prev is not None and routes.STATE_RANK[state_of[prev]] > routes.STATE_RANK[st_now]
        try:
            nat = routes.natural_routes(lib, j, st_now, down, rng)
            alts = routes.candidates(lib, j, st_now, rng)
        except Exception:
            continue
        if prev is not None and state_of[prev] == st_now:          # within a state: stay in the song
            nat, alts = [r for r in nat if same_song(r)], [r for r in alts if same_song(r)]
        _fw = lambda r: routes.foreign_wind_down(by[j["a"]], by[routes.as_route(r)["steps"][-1]["b"]],
                                                 routes.as_route(r)["steps"][-1]["t_in"], down)
        nat, alts = [r for r in nat if not _fw(r)], [r for r in alts if not _fw(r)]   # user 2026-11-24
        A = by.get(j["a"])
        if A is None or quiet_at(A, j["t_out"]):
            continue                        # user 2026-11-23: a silent lead-in leaves nothing to judge but the drop
        pool, seen_d = [], set()
        for r in nat + alts[:6]:
            d = routes.describe(r)
            if d not in seen_d and not any(twins(r, x) for x in pool):
                seen_d.add(d); pool.append(r)
        if len(pool) < 4 or not nat:
            continue
        try:
            scores = {id(r): taste(by, r) for r in pool}
        except Exception:
            continue
        champ = max(pool, key=lambda r: scores[id(r)])
        opts = choose(champ, pool)
        jump = abs(routes.features(by, champ)["jump"])
        cands.append((-sum(fresh(r) for r in opts) - 0.08 * jump - 0.3 * rng.random(), j, opts, opts.index(champ), pool))
    cands.sort(key=lambda c: c[0])
    quads, used, starts, anywhere = [], set(), Counter(), Counter()
    for _, j, opts, mp, pool in cands:
        key = (j["a"], round(j["t_out"]))
        songs = {j["a"]} | {s["b"] for r in opts for s in routes.as_route(r)["steps"]}
        if key in used or starts[j["a"]] >= 1 or any(anywhere[s] >= 5 for s in songs):
            continue
        used.add(key); starts[j["a"]] += 1; anywhere.update(songs)
        quads.append({"id": f"q{int(time.time())}_{len(quads)}", "context": j["context"], "options": opts,
                      "model_pick": mp, "picker": "trees", "synthetic": bool(j.get("synthetic")), "_pool": pool})
        if len(quads) >= n:
            break
    os.makedirs(os.path.join(AB, "clips"), exist_ok=True)
    rev = os.path.join(HERE, "review")
    for q in quads:
        pool = q.pop("_pool")
        # 2026-11-20: the critic (hears the audio) picks the favourite: it agreed with your picks 59%
        # without ever seeing one, vs the trees' 56% after training on them
        if os.path.exists(CRITIC_PY) and os.path.exists(CRITIC_PT):
            try:
                cand = list(dict.fromkeys(map(id, q["options"] + pool)))
                byid = {id(r): r for r in q["options"] + pool}
                cand = [byid[c] for c in cand][:9]
                for r in cand:                    # tempo matching method is an A/B variable too
                    if not same_song(r):
                        r["stretch"] = rng.choice(("rubberband", "resample"))
                rendered = []
                for m, r in enumerate(cand):
                    c = f"ab/clips/{q['id']}_p{m}.mp3"
                    rendered.append((c, routes.render(by, r, os.path.join(rev, c))))
                jin, jout = os.path.join(AB, "_score_in.json"), os.path.join(AB, "_score_out.json")
                json.dump([[os.path.join(rev, c), s] for c, s in rendered], open(jin, "w", encoding="utf-8"))
                env = dict(os.environ, HF_HOME=os.path.join(HERE, ".hf"), PYTHONIOENCODING="utf-8")
                subprocess.run([CRITIC_PY, "-m", "conductor.critic", "score", jin, jout], cwd=HERE, env=env, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                sc = json.load(open(jout, encoding="utf-8"))
                best = max(range(len(cand)), key=lambda m: sc[m])
                opts = choose(cand[best], cand)
                where = {id(r): m for m, r in enumerate(cand)}
                q["options"], q["model_pick"], q["picker"] = opts, opts.index(cand[best]), "critic"
                q["critic"] = [round(sc[where[id(r)]], 3) for r in opts]
                q["clips"], q["seams"] = [], []
                for k, r in enumerate(opts):
                    src, seams = rendered[where[id(r)]]
                    dst = f"ab/clips/{q['id']}_{k}.mp3"
                    os.replace(os.path.join(rev, src), os.path.join(rev, dst))
                    q["clips"].append(dst); q["seams"].append(seams)
                for c, _ in rendered:
                    if os.path.exists(os.path.join(rev, c)) and os.path.basename(c).startswith(q["id"] + "_p"):
                        os.remove(os.path.join(rev, c))
                continue
            except Exception as e:
                print("critic pick failed, trees pick:", e)
        q["clips"], q["seams"] = [], []
        for k, r in enumerate(q["options"]):
            c = f"ab/clips/{q['id']}_{k}.mp3"
            q["seams"].append(routes.render(by, r, os.path.join(rev, c)))
            q["clips"].append(c)
    quads = add_repeats(quads, REPEATS_PER_SET, rng)
    json.dump(quads, open(os.path.join(AB, "quads.json"), "w", encoding="utf-8"), indent=1)
    return quads


# fill1 dropped (won 2/13); 2026-11-24 layered blends added (user: the lead-ins still ended in a bang)
HOW_STYLES = ("swap8", "swap8d", "swap4", "swap4d", "layer4", "layer8")
HOW_ALWAYS = ("swap8", "swap8d")               # the user's drop swap (both versions) + 1 more vs the quick blend


def _gen_sample(rng, tm: bool) -> str:
    """One random blend plan (routes.gen_name). tm = the two beats can run together; if not, B's drums and
    bass only come in once A's are gone."""
    from . import routes
    L = rng.choice((4, 8))
    a = [rng.randint(1, L) for _ in range(3)]
    b = [rng.randint(0, L) for _ in range(3)]
    # user 2026-11-25: the new song always FADED in - B's early layers snap in on the bar more often
    fade = rng.choices((0, 1, 2), (0.3, 0.5, 0.2), k=3) + rng.choices((0, 1, 2), (0.5, 0.35, 0.15), k=3)
    hook = (rng.randint(1, L - 1), rng.choice((1, 2))) if rng.random() < 0.7 else (0, 0)
    if not tm:
        # user 2026-11-26 "still offbeat": beats that can't lock never sound together. A's drums+bass leave
        # (fade included) -> breakdown: A's melody with B's melody/hook over it -> A's melody leaves -> B's beat.
        fb = lambda i: (1, 1, 2)[fade[i]]
        gone = min(L, max(a[0] + fb(0), a[1] + fb(1)))
        a[2] = rng.randint(min(L, max(a[0], a[1])), L)
        b[2] = rng.randint(gone, L)
        b[0] = b[1] = min(L, max(gone, a[2] + fb(2)))
        hook = (rng.randint(gone + 1, L - 1), 1) if hook[0] and gone + 1 <= L - 1 else (0, 0)
    loud = int(rng.random() < 0.5)               # ...and teases at full level, solo and on the beat
    return routes.gen_name({"L": L, "a": a, "b": b, "fade": fade, "hook": hook, "loud": loud,
                            "trail": int(rng.random() < 0.4)})


def _gen_set(by, j, A, B, t_land, dbA, champ, mode, n_gen, rng, k_set, rev):
    """Render n_gen generated blends + one rule-based yardstick, let the critic (and your taste layer)
    listen at every bar line, and keep: its top 2, one wildcard from the rest, the yardstick."""
    from . import routes
    ctrl = f"drumbridge{rng.choice((4, 8))}h" if mode == "beat" else None
    styles = list(dict.fromkeys(_gen_sample(rng, mode == "beat") for _ in range(n_gen)))
    cands = []
    for st in styles + ([ctrl] if ctrl else []):
        routes.register_gen(st)
        Lb = routes.LEAD_STYLES[st]
        t_in = routes.lead_start(B, t_land, st)
        if t_in is None or len(dbA) < Lb:
            continue
        cands.append({"a": j["a"], "t_out": round(dbA[-Lb], 3), "b": B.title, "t_in": t_in,
                      "steps": [{"b": B.title, "t_in": t_in, "bars": None, "style": st, "land": t_land}]})
    if ctrl is None:
        cands.append(champ)                       # bridge sets: the trees' own quick join is the yardstick
    qid = f"g{int(time.time())}_{k_set}{os.environ.get('CONDUCTOR_WORKER', '')}"   # parallel renderers never clash
    rendered, plans = [], []
    for m, r in enumerate(cands):
        c = f"ab/clips/{qid}_p{m}.mp3"
        routes.LAST_PLAN = []
        try:
            seams = routes.render(by, r, os.path.join(rev, c))
        except Exception as e:
            print("   render failed:", e); seams = None
        st = routes.as_route(r)["steps"][0]["style"]
        if mode == "beat" and st in routes.LAYER_STYLES and not routes.LAST_LOCK:
            seams = None                          # 2026-11-26: a "beat" blend that didn't lock plays offbeat - drop it
        plans.append(routes.LAST_PLAN if st in routes.LAYER_STYLES and routes.LAST_PLAN
                     else routes.show_plan(st, routes.lands_on_drop(B, t_land)))
        rendered.append((c, seams))
    ok = [m for m, (_, sm) in enumerate(rendered) if sm]
    if len(ok) < 4:
        return None
    # the critic hears every bar line of the blend, not just its two ends
    jobs = []
    for m in ok:
        c, sm = rendered[m]
        Lb = routes.LEAD_STYLES.get(routes.as_route(cands[m])["steps"][0]["style"], 1)
        bars = list(np.linspace(sm[0], sm[-1], Lb + 1)) if len(sm) >= 2 else sm
        jobs.append([os.path.join(rev, c), [round(float(x), 3) for x in bars]])
    jin, jout = os.path.join(AB, "_score_in.json"), os.path.join(AB, "_score_out.json")
    json.dump(jobs, open(jin, "w", encoding="utf-8"))
    env = dict(os.environ, HF_HOME=os.path.join(HERE, ".hf"), PYTHONIOENCODING="utf-8")
    subprocess.run([CRITIC_PY, "-m", "conductor.critic", "score", jin, jout], cwd=HERE, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sc = dict(zip(ok, json.load(open(jout, encoding="utf-8"))))
    gen_ok = sorted((m for m in ok if routes.gen_parse(routes.as_route(cands[m])["steps"][0]["style"])), key=lambda m: -sc[m])
    yard = [m for m in ok if m not in gen_ok]
    if len(gen_ok) < 3 or not yard:
        return None
    pick = gen_ok[:2] + [rng.choice(gen_ok[2:])] + [yard[0]]
    role = ["critic top", "critic 2nd", "wildcard", "yardstick"]
    order = list(range(4)); rng.shuffle(order)
    q = {"id": qid, "kind": "how", "mode": mode, "gen": True, "context": j["context"], "options": [], "clips": [],
         "seams": [], "plans": [], "critic": [], "role": [], "landing": [B.title, round(t_land, 2)],
         "model_pick": None, "picker": "critic", "synthetic": bool(j.get("synthetic")),
         "pool": [[routes.as_route(cands[m])["steps"][0]["style"], round(sc[m], 4)] for m in ok]}
    for k, o in enumerate(order):
        m = pick[o]
        dst = f"ab/clips/{qid}_{k}.mp3"
        os.replace(os.path.join(rev, rendered[m][0]), os.path.join(rev, dst))
        q["options"].append(cands[m]); q["clips"].append(dst); q["seams"].append(rendered[m][1])
        q["plans"].append(plans[m]); q["critic"].append(round(sc[m], 4)); q["role"].append(role[o])
        if o == 0:
            q["model_pick"] = k
    for c, _ in rendered:
        f = os.path.join(rev, c)
        if os.path.exists(f) and os.path.basename(c).startswith(qid + "_p"):
            os.remove(f)
    return q


def make_how(lib, n: int = 12, seed: int | None = None, cross: float = 0.75, only_b: set | None = None,
             sprinkle: bool = False, gen: int = 0, on_set=None) -> list[dict]:
    """2026-11-24 "how" sets (user: the four-way sets mix songs AND methods, so a pick can't say which
    it liked). One moment, one landing (the trees' favourite place to go), joined four ways: that
    favourite's own join (the abrupt kind the game uses now) + 3 signposted joins (routes.LEAD_STYLES)
    that announce the new music before it lands. `cross` = share of sets that change song."""
    from . import routes
    from collections import Counter
    rng = random.Random(seed if seed is not None else int(time.time()))
    by = {t.title: t for t in lib.tracks}
    judged = Counter()
    for a in json.load(open(os.path.join(AB, "answers.json"), encoding="utf-8")):
        for r in (a.get("quad") or {}).get("options", []):
            for st in routes.as_route(r)["steps"]:
                judged[st["style"]] += 1
    moments = _moments(lib)
    rng.shuffle(moments)
    moments = moments[:40] + _synthetic(lib, rng, 160)
    rng.shuffle(moments)
    state_of = {"explore": "calm", "tier1": "fight", "tier2": "horde", "tier3": "boss"}
    inv = {v: k for k, v in STATE_NAME.items()}
    quads, used, anywhere = [], set(), Counter()
    rev = os.path.join(HERE, "review")
    os.makedirs(os.path.join(AB, "clips"), exist_ok=True)
    for j in moments:
        if len(quads) >= n:
            break
        if (j["a"], round(j["t_out"])) in used or anywhere[j["a"]] >= 2:
            continue
        A = by.get(j["a"])
        if A is None:
            continue
        # user 2026-11-24: leave a song at the END of a part (every set that left mid-part was all bad)
        seg = A.segment_at(j["t_out"] - 0.05)
        if seg is None:
            continue
        exit_ = seg.start + seg.length
        dbA = [d for d in (A.downbeats or A.beats[::4]) if d < exit_ - 0.1]
        if len(dbA) < 5 or dbA[-4] < PRE + 0.5 or exit_ > A.duration - 1 or quiet_at(A, exit_):
            continue
        j = dict(j, t_out=round(exit_, 3))
        st_now = state_of[j["state_role"]]
        prev = inv.get(j["context"].split(" → ")[0])
        down = prev is not None and routes.STATE_RANK[state_of[prev]] > routes.STATE_RANK[st_now]
        try:
            pool = routes.natural_routes(lib, j, st_now, down, rng) + routes.candidates(lib, j, st_now, rng)
        except Exception:
            continue
        want_cross = len(quads) < round(n * cross)
        pool = [r for r in pool if len(routes.as_route(r)["steps"]) == 1
                and (routes.as_route(r)["steps"][0]["b"] != j["a"]) == want_cross
                and not routes.foreign_wind_down(A, by[routes.as_route(r)["steps"][0]["b"]],
                                                 routes.as_route(r)["steps"][0]["t_in"], down)]
        if not pool:
            continue
        # user 2026-11-25: two kinds of set. "beat": the tempos meet (within ~2 %) and both grids are steady,
        # so the beats can run together - anywhere, even mid-chorus. "bridge": they don't, so A breaks down
        # to its melody, B's hooks float over, and B lands on a CALM part of itself.
        def _beat_ok(r):
            from . import grid
            s0 = routes.as_route(r)["steps"][0]; Bx = by[s0["b"]]
            b8 = 32 * grid.period(Bx, s0["t_in"])            # the beats must really lock over the last 8 bars
            return (routes.tempo_match(A, j["t_out"] - 1, Bx, s0["t_in"] + 1)
                    and grid.warp_positions(A, j["t_out"] - b8, Bx, s0["t_in"] - b8, int(b8 * 200), 200) is not None
                    and routes.grid_ok(A, dbA[-8] if len(dbA) >= 8 else dbA[0], j["t_out"])
                    and routes.grid_ok(Bx, s0["t_in"] - 8 * 4 * 60 / max(60.0, Bx.tempo or 120), s0["t_in"] + 4))
        okb = (lambda r: routes.as_route(r)["steps"][0]["b"] in only_b) if only_b is not None else (lambda r: True)
        beat_pool = [r for r in pool if okb(r) and _beat_ok(r)]
        chill_pool = [r for r in pool if okb(r) and routes.chill_at(by[routes.as_route(r)["steps"][0]["b"]],
                                                                       routes.as_route(r)["steps"][0]["t_in"] + 0.1)]
        n_beat = sum(1 for qq in quads if qq.get("mode") == "beat")
        mode = "beat" if beat_pool and (n_beat < n / 2 or not chill_pool) else "bridge" if chill_pool else None
        if mode is None or (mode == "beat" and n_beat >= (n + 1) // 2) or (mode == "bridge" and len(quads) - n_beat >= (n + 1) // 2):
            continue
        try:
            champ = max(beat_pool if mode == "beat" else chill_pool, key=lambda r: taste(by, r))
        except Exception:
            continue
        st0 = routes.as_route(champ)["steps"][0]
        B, t_land = by[st0["b"]], st0["t_in"]
        if anywhere[B.title] >= 2:
            continue
        if gen:                                  # 2026-11-25: AI-shaped blends, the critic picks what you hear
            q = _gen_set(by, j, A, B, t_land, dbA, champ, mode, gen, rng, len(quads), rev)
            if q is None:
                continue
            used.add((j["a"], round(j["t_out"]))); anywhere.update([j["a"], B.title])
            quads.append(q)
            if on_set:
                on_set(q)                        # the trainer sees each set as soon as it's rendered
            print(f"  {len(quads)}/{n} [{mode}] {j['a'][:28]} -> {B.title[:28]}  critic {q['critic']}  "
                  f"[{', '.join(routes.as_route(r)['steps'][0]['style'] for r in q['options'])}]", flush=True)
            continue
        # the old part's end and the new part's start meet; a lead-in starts its bars before that
        # 2026-11-24: three DIFFERENT blend recipes per set (user: one shape winning by default teaches
        # "always do this"), random length (4/8 bars) and with/without hook sprinkles
        if mode == "beat":
            leads = [f"{r}{rng.choice((4, 8))}{'h' if sprinkle else rng.choice(('', 'h'))}" for r in rng.sample(list(routes.BEAT_RECIPES), 3)]
        else:
            leads = ["breakdown4h", "breakdown8h", rng.choice(("hookover8h", "melfirst8h"))]
        opts = [champ]
        for st in leads:
            if len(opts) >= 4:
                break
            if st in routes.AFTER_STYLES:
                opts.append({"a": j["a"], "t_out": j["t_out"], "b": B.title, "t_in": t_land,
                             "steps": [{"b": B.title, "t_in": t_land, "bars": None, "style": st}]})
                continue
            L = routes.LEAD_STYLES[st]
            t_in = routes.lead_start(B, t_land, st)
            if t_in is not None and len(dbA) >= L:
                opts.append({"a": j["a"], "t_out": round(dbA[-L], 3), "b": B.title, "t_in": t_in,
                             "steps": [{"b": B.title, "t_in": t_in, "bars": None, "style": st, "land": t_land}]})
        if len(opts) < 4:
            continue
        for r in opts[1:]:
            judged[routes.as_route(r)["steps"][0]["style"]] += 1
        rng.shuffle(opts)
        q = {"id": f"h{int(time.time())}_{len(quads)}", "kind": "how", "mode": mode, "context": j["context"], "options": opts,
             "model_pick": opts.index(champ), "picker": "trees", "synthetic": bool(j.get("synthetic")),
             "landing": [B.title, round(t_land, 2)], "clips": [], "seams": []}
        try:
            q["plans"] = []
            for k, r in enumerate(opts):
                c = f"ab/clips/{q['id']}_{k}.mp3"
                routes.LAST_PLAN = []
                q["seams"].append(routes.render(by, r, os.path.join(rev, c)))
                q["clips"].append(c)
                st = routes.as_route(r)["steps"][0]["style"]       # what the blend REALLY played (adapted + activity)
                q["plans"].append(routes.LAST_PLAN if st in routes.LAYER_STYLES and routes.LAST_PLAN
                                  else routes.show_plan(st, routes.lands_on_drop(B, t_land)))
        except Exception as e:
            print("render failed:", e)
            continue
        used.add((j["a"], round(j["t_out"]))); anywhere.update([j["a"], B.title])
        quads.append(q)
        print(f"  {len(quads)}/{n}  {j['a'][:30]} -> {B.title[:30]}  [{', '.join(routes.as_route(r)['steps'][0]['style'] for r in opts)}]", flush=True)
    return quads


REPEATS_PER_SET = 3     # 2026-11-23: blind re-asks of answered sets, so we know how consistent the target itself is


def add_repeats(quads: list, k: int, rng=None) -> list:
    """Mix k earlier four-way sets back in, options reshuffled and under a new id, marked repeat_of.
    Training ignores them; consistency() compares the two answers."""
    import copy
    rng = rng or random.Random()
    p = os.path.join(AB, "answers.json")
    raw = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []
    rev = os.path.join(HERE, "review")
    done_rep = {a["quad"].get("repeat_of") for a in raw if a.get("kind") == "quad"}
    queued = {q.get("repeat_of") for q in quads}
    pool = [a["quad"] for a in raw if a.get("kind") == "quad" and not a["quad"].get("repeat_of")
            and a.get("best") is not None and a.get("worst") is not None and a["best"] != a["worst"]
            and a["quad"]["id"] not in done_rep and a["quad"]["id"] not in queued
            and all(os.path.exists(os.path.join(rev, c)) for c in a["quad"].get("clips", []))]
    out = list(quads)
    for src in rng.sample(pool, min(k, len(pool))):
        q = copy.deepcopy(src)
        perm = list(range(len(q["options"])))
        while perm == list(range(len(perm))):
            rng.shuffle(perm)
        for f in ("options", "clips", "seams", "critic"):
            if isinstance(q.get(f), list) and len(q[f]) == len(perm):
                q[f] = [q[f][m] for m in perm]
        if q.get("model_pick") is not None:
            q["model_pick"] = perm.index(q["model_pick"])
        q["repeat_of"], q["id"] = src["id"], f"r{int(time.time())}_{len(out)}"
        out.insert(rng.randrange(len(out) + 1), q)
    return out


def consistency() -> dict:
    """How often your repeat answer agrees with your first answer, over the head-to-heads both settle."""
    raw = json.load(open(os.path.join(AB, "answers.json"), encoding="utf-8"))
    first = {a["quad"]["id"]: a for a in raw if a.get("kind") == "quad" and not a["quad"].get("repeat_of")}

    order = _rank
    agree = total = sets = ties = 0
    agree_d = total_d = 0                      # leaving out timing-only twins (user 2026-11-23: can't be told apart)
    for a in raw:
        if a.get("kind") != "quad" or not a["quad"].get("repeat_of") or a["quad"]["repeat_of"] not in first:
            continue
        if a["quad"].get("conflict"):            # a labelled tie-breaker, not a blind check of your consistency
            continue
        f = first[a["quad"]["repeat_of"]]
        o1, o2 = order(f), order(a)
        if not o1 or not o2:
            continue
        opt = dict(zip(f["quad"]["clips"], f["quad"]["options"]))
        fine1 = {f["quad"]["clips"][k] for k in fine_set(f)} if f.get("fine") is not None else set()
        fine2 = {a["quad"]["clips"][k] for k in fine_set(a)} if a.get("fine") is not None else set()
        sets += 1
        cl = list(o1)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                d1, d2 = o1[cl[i]] - o1[cl[j]], o2[cl[i]] - o2[cl[j]]
                if d1 and d2:                      # both answers settle this head-to-head
                    same = (d1 > 0) == (d2 > 0)
                    if not same and {cl[i], cl[j]} <= fine1 & fine2:
                        same = True; ties += 1             # both fine both times: a tie, not a mistake
                    total += 1; agree += same
                    try:
                        tw = twins(opt[cl[i]], opt[cl[j]])
                    except Exception:
                        tw = False
                    if not tw:
                        total_d += 1; agree_d += same
    return {"repeat_sets": sets, "head_to_heads": total, "fine_fine_flips": ties,
            "self_agreement": round(agree / total, 3) if total else None,
            "distinct_head_to_heads": total_d,
            "self_agreement_distinct": round(agree_d / total_d, 3) if total_d else None}


def ingest_examples(lib) -> int:
    """Proven joins (A/B winners/losers) also teach the song-taught model."""
    from .joinmodel import EXTRA_PATH
    ex = []
    for a in _answers():
        p = a["pair"]
        if a["choice"] in ("a", "b"):
            win, lose = (p["a"], p["b"]) if a["choice"] == "a" else (p["b"], p["a"])
            ex += [dict(win, good=1, weight=3), dict(lose, good=0, weight=3)]
    json.dump(ex, open(EXTRA_PATH, "w", encoding="utf-8"), indent=1)
    return len(ex)


def training_library():
    """Every registered playlist's songs, once each (2026-11-23: answers and in-game ratings on songs outside
    Darktide II used to be dropped from training)."""
    from .library import Library
    from .playlists import all_playlists
    folders = [__import__("conductor.playlists", fromlist=["train_music"]).train_music()] + [f for f in all_playlists().values() if os.path.isdir(f)]
    tracks, seen = [], set()
    for f in dict.fromkeys(folders):
        for t in Library.build(f, os.path.join(HERE, "cache"), progress=lambda m: None,
                               backfill=False, cached_only=True).tracks:
            if t.title not in seen:
                seen.add(t.title); tracks.append(t)
    return Library(tracks)


def nightly(n: int = 15) -> dict:
    from . import joinmodel
    lib = training_library()
    try:
        mark_noisy(lib)                 # bad questions (silent lead-ins, timing-only twins) stay out of training
    except Exception as e:
        print("noisy-set marking failed:", e)
    # A/B answers train ONLY the taste model: feeding them to the song model too leaks them into its
    # "natural" score and makes the taste model look perfect on picks it has already seen
    from .joinmodel import EXTRA_PATH
    json.dump([], open(EXTRA_PATH, "w", encoding="utf-8"))
    report = {"answers": len(_answers())}
    report["song_model"] = joinmodel.train(lib)
    report["taste_model"] = train_taste(lib)
    # 2026-11-20: your picks also train a taste layer on the critic, which then chooses each set's favourite
    if os.path.exists(CRITIC_PY) and os.path.exists(CRITIC_PT):
        try:
            tout = os.path.join(AB, "_taste_out.json")
            env = dict(os.environ, HF_HOME=os.path.join(HERE, ".hf"), PYTHONIOENCODING="utf-8")
            subprocess.run([CRITIC_PY, "-m", "conductor.critic", "taste", tout], cwd=HERE, env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            report["critic_taste"] = json.load(open(tout, encoding="utf-8"))
            subprocess.run([CRITIC_PY, "-m", "conductor.critic", "styles"], cwd=HERE, env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            report["critic_taste"] = f"failed: {e}"
    # 2026-11-20: every training round updates what the game uses (the app re-reads policy.json)
    try:
        from . import policy
        pol = policy.build()
        report["live_policy"] = {k: v[0]["style"] for k, v in pol["choices"].items()} | {"version": pol["version"]}
    except Exception as e:
        report["live_policy"] = f"failed: {e}"
    # the model's own calls, judged by the user: did its favourite win?
    mp = [a for a in _answers() if a["pair"].get("model_pick") and a["choice"] in ("a", "b", "both_bad", "both_good")]
    if mp:
        won = sum(a["choice"] == a["pair"]["model_pick"] for a in mp)
        lost = sum(a["choice"] in ("a", "b") and a["choice"] != a["pair"]["model_pick"] for a in mp)
        report["model_pick"] = f"won {won}, lost {lost}, both-bad {sum(a['choice'] == 'both_bad' for a in mp)}, both-fine {sum(a['choice'] == 'both_good' for a in mp)}"
    raw = json.load(open(os.path.join(AB, "answers.json"), encoding="utf-8"))
    qa = [a for a in raw if a.get("kind") == "quad" and a["quad"].get("model_pick") is not None
          and not a["quad"].get("repeat_of")]
    try:
        report["self_consistency"] = consistency()
    except Exception as e:
        report["self_consistency"] = f"failed: {e}"
    # 2026-11-23: the live critic's taste layer (critic + style) learns from the new answers too
    # (critic_live.py reads it at app start: the next game launch uses it)
    try:
        env = dict(os.environ, HF_HOME=os.path.join(HERE, ".hf"), PYTHONIOENCODING="utf-8")
        subprocess.run([CRITIC_PY, "-W", "ignore", "-m", "conductor.distill_trees", "feats"], cwd=HERE, env=env,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from .distill_trees import export_teacher
        t = export_teacher()
        report["live_critic_taste"] = {"comparisons": t["comparisons"], "built": t["built"]}
    except Exception as e:
        report["live_critic_taste"] = f"failed: {e}"
    for picker in ("trees", "critic"):
        sub = [a for a in qa if a["quad"].get("picker", "trees") == picker and a.get("best") is not None]
        if sub:
            report[f"{picker}_pick_of_4"] = (f"best {sum(a['best'] == a['quad']['model_pick'] for a in sub)}, "
                                             f"worst {sum(a['worst'] == a['quad']['model_pick'] for a in sub)}, of {len(sub)} "
                                             f"(chance: 1 in 4 each)")
    # four-way sets now (2026-11-19); unrated ones still waiting and nothing new learned: keep them
    qp = os.path.join(AB, "quads.json")
    done = {a["quad"]["id"] for a in raw if a.get("kind") == "quad"}
    left = [x for x in (json.load(open(qp, encoding="utf-8")) if os.path.exists(qp) else []) if x["id"] not in done]
    nq = max(8, n // 2)
    report["quads"] = len(left) if len(left) >= nq // 2 else len(make_quads(lib, nq))
    report["when"] = time.strftime("%Y-%m-%d %H:%M")
    hist = os.path.join(AB, "history.json")
    h = json.load(open(hist, encoding="utf-8")) if os.path.exists(hist) else []
    json.dump(h + [report], open(hist, "w", encoding="utf-8"), indent=1)
    return report


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "nightly"
    if cmd == "nightly":
        print(json.dumps(nightly(int(sys.argv[2]) if len(sys.argv) > 2 else 15), indent=1))
    elif cmd == "quads":                    # quads [n] [playlist name or folder]: sets from that music
        from .library import Library
        src = sys.argv[3] if len(sys.argv) > 3 else __import__("conductor.playlists", fromlist=["train_music"]).train_music()
        if not os.path.isdir(src):
            from .playlists import all_playlists
            src = all_playlists()[src]
        lib = Library.build(src, os.path.join(HERE, "cache"), progress=lambda m: None,
                            backfill=False, cached_only=True)
        print(len(make_quads(lib, int(sys.argv[2]) if len(sys.argv) > 2 else 12)), "quads from", src)
    elif cmd == "how":                      # how [n]: same landing, four ways in (signposted joins); queued first
        lib = training_library()
        hs = make_how(lib, int(sys.argv[2]) if len(sys.argv) > 2 else 12)
        qp = os.path.join(AB, "quads.json")
        raw = json.load(open(os.path.join(AB, "answers.json"), encoding="utf-8"))
        done = {a["quad"]["id"] for a in raw if a.get("kind") == "quad"}
        keep = [q for q in (json.load(open(qp, encoding="utf-8")) if os.path.exists(qp) else [])
                if q["id"] not in done and q.get("tiebreak")]
        json.dump(keep + hs, open(qp, "w", encoding="utf-8"), indent=1)
        print(len(hs), "how-sets queued after", len(keep), "tie-breakers")
    elif cmd == "repeats":                  # mix N blind repeats into the sets waiting now
        qp = os.path.join(AB, "quads.json")
        qs = json.load(open(qp, encoding="utf-8")) if os.path.exists(qp) else []
        qs = add_repeats(qs, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
        json.dump(qs, open(qp, "w", encoding="utf-8"), indent=1)
        print(sum(1 for q in qs if q.get("repeat_of")), "repeats among", len(qs), "sets")
    elif cmd == "consistency":
        print(json.dumps(consistency(), indent=1))
    elif cmd == "pairs":
        lib = training_library()
        train_taste(lib)
        print(len(make_pairs(lib, int(sys.argv[2]) if len(sys.argv) > 2 else 15)), "pairs")
