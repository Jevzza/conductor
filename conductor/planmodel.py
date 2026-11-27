"""Plan model: learns from the blend RECIPE (which layer plays when, how loud) which way of joining two
songs you'd pick - the part the audio critic can't see (it hears short clips around each seam).

Training data: your four-way answers (best beats the rest, the rest beat the worst, contradictions
dropped and settled as ab.training_pairs does), plus your Tweaks (your edit beats the option you edited
and everything you ranked below it). Each option is described by its lanes (stored as rendered, or
rebuilt from the recipe name) plus the context (beat-locked or not, tempo gap, glide).

    python -m conductor.planmodel train      # cross-validated by set, compared with the critic; saves the model
    python -m conductor.planmodel report     # what it has learned (strongest features)
"""
from __future__ import annotations

import json
import os
import re
import zlib

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = os.path.join(HERE, "review", "ab")
MODEL = os.path.join(HERE, "cache", "plan_model.json")
LANES = ["A drums", "A bass", "A melody", "B drums", "B bass", "B melody", "A hook", "B hook", "A sound", "B sound"]
N = 8                                         # gain samples per lane across the blend
GLIDE_START = 1795749700                      # first glide-era AI set (2026-11-26 21:15)


def _gain(pts, x):
    xs, ys = zip(*[(float(a), float(b)) for a, b in pts])
    return float(np.interp(x, xs, ys))


def _style(opt) -> str:
    st = (opt.get("steps") or [{}])[0].get("style", "")
    return st or ""


def plan_of(opt, stored=None):
    """[{lane, pts}] for an option: the lanes as rendered if stored, else rebuilt from the recipe; None if
    the option isn't a layered blend (cut, crossfade, DJ mix...)."""
    if stored:
        return [p for p in stored if p.get("lane") in LANES]
    from . import routes
    st = _style(opt)
    routes.register_gen(st)
    if st not in routes.LAYER_STYLES:
        return None
    try:
        return routes.layer_plan(st, True)
    except Exception:
        return None


def _apply_edit(plan, edit):
    if not plan or not edit:
        return plan
    drawn = edit.get("plan") or {}
    out = [dict(p, pts=drawn[p["lane"]]) if p["lane"] in drawn else p for p in plan]
    have = {p["lane"] for p in out}
    out += [{"lane": ln, "pts": pts} for ln, pts in drawn.items() if ln not in have and ln in LANES]
    return out


FAMILIES = ["gen4", "gen8", "melfirst", "rhyfirst", "drumbridge", "basscarry", "hookover", "breakdown",
            "xfade", "punch", "riser", "dj", "game", "mix", "seam", "natural", "other"]


def _family(st: str) -> str:
    for f in FAMILIES[:-1]:
        if st.startswith(f):
            return f
    return "other"


def features(opt, plan, ctx: dict) -> np.ndarray:
    """One option -> a fixed-length vector."""
    xs = (np.arange(N) + 0.5) / N
    st = _style(opt)
    fam = [float(_family(st) == f) for f in FAMILIES]
    layered = plan is not None
    G = {ln: np.zeros(N) for ln in LANES}
    if layered:
        for p in plan:
            if p["lane"] in G and len(p["pts"]) >= 2:
                G[p["lane"]] = np.array([_gain(p["pts"], x) for x in xs])
    lanes = np.concatenate([G[ln] for ln in LANES])
    both = lambda a, b: float(np.mean((G[a] > 0.3) & (G[b] > 0.3)))
    locked = float(ctx.get("locked", 0.0))
    drum_ov, bass_ov, mel_ov = both("A drums", "B drums"), both("A bass", "B bass"), both("A melody", "B melody")
    first_on = lambda ln: next((x for x, g in zip(xs, G[ln]) if g > 0.3), 1.0)
    last_on = lambda ln: max([x for x, g in zip(xs, G[ln]) if g > 0.3] or [0.0])
    total = sum(G[ln] for ln in LANES[:6])
    summ = [
        layered, drum_ov, bass_ov, mel_ov,
        drum_ov * (1 - locked), bass_ov * (1 - locked),             # overlaps that clash when the beats aren't locked
        first_on("B drums"), last_on("A drums"), first_on("B bass"), last_on("A bass"),
        first_on("B melody"), last_on("A melody"),
        float(G["A hook"].max() > 0.05), float(G["B hook"].max() > 0.05),
        float(G["A sound"].max() > 0.05), float(G["B sound"].max() > 0.05),
        float(total.min()), float(total.max()), float(np.abs(np.diff(total)).max()),   # holes / jumps in the fullness
        float(np.mean(total < 0.6)),                                                    # time spent thin
    ]
    bars = 8.0 if st.startswith("gen8") or re.search(r"8h?$", st) else 4.0 if layered else 1.0
    m = re.match(r"gen\d_\d+h(\d+)t(\d)", st)
    ctxv = [locked, float(ctx.get("glide", 0.0)), float(ctx.get("tempo_gap", 0.0)), float(ctx.get("same_song", 0.0)),
            bars / 8.0, float(bool(m and m.group(2) == "1")), float(ctx.get("bridge", 0.0))]
    return np.array(fam + list(lanes) + summ + ctxv, dtype=np.float64)


def _ctx(q, opt) -> dict:
    st = _style(opt)
    steps = opt.get("steps") or [{}]
    return {"locked": float(q.get("mode") == "beat"), "bridge": float(q.get("mode") == "bridge"),
            "glide": float(q.get("stretch") == "glide"),
            "same_song": float(all(s.get("b") == opt.get("a") for s in steps)),
            "tempo_gap": float(q.get("tempo_gap", 0.0))}


def _weight(q) -> float:
    m = re.match(r"g(\d+)_", q.get("id", ""))
    if m and int(m.group(1)) < GLIDE_START and q.get("mode") == "beat":
        return 0.5                                   # judged with the old pitch-shifting tempo match
    return 1.0


def dataset():
    """Pairs (features winner, features loser, group, weight) from your answers and tweaks."""
    from . import ab
    raw = json.load(open(os.path.join(AB, "answers.json"), encoding="utf-8"))
    nz = ab.noisy()
    pairs = []
    for a in raw:
        if a.get("kind") != "quad":
            continue
        q = a["quad"]
        if q.get("repeat_of") or ab.stale(q["id"]):
            continue
        b, w = a.get("best"), a.get("worst")
        if b is None or b == w:
            continue
        opts, stored = q["options"], (q.get("plans") or [None] * len(q["options"]))
        F = []
        for k, o in enumerate(opts):
            F.append(features(o, plan_of(o, stored[k] if k < len(stored) else None), _ctx(q, o)))
        wt = _weight(q)
        for x, y in ab.training_pairs(a, nz):
            pairs.append((F[x], F[y], q["id"], wt))
        rank = ab._rank(a) or {}
        for k, e in (a.get("edits") or {}).items():          # your tweak beats what you edited and all below it
            k = int(k)
            o = opts[k]
            pe = _apply_edit(plan_of(o, stored[k] if k < len(stored) else None), e)
            fe = features(o, pe, _ctx(q, o))
            mine = rank.get(q["clips"][k], 1)
            pairs.append((fe, F[k], q["id"], 2.0 * wt))
            for j, c in enumerate(q["clips"]):
                if j != k and rank.get(c, 1) <= mine:
                    pairs.append((fe, F[j], q["id"], 1.5 * wt))
    return pairs


def _fold(g, k=5):
    return zlib.crc32(g.encode()) % k


def _fit(X, y, w):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=0.3, max_iter=4000, fit_intercept=False)
    m.fit(X, y, sample_weight=w)
    return m


def _xy(pairs):
    X, y, w, g = [], [], [], []
    for fa, fb, grp, wt in pairs:                   # both orders: the model can't lean on position
        X += [fa - fb, fb - fa]; y += [1, 0]; w += [wt, wt]; g += [grp, grp]
    return np.array(X), np.array(y), np.array(w), g


def train(save: bool = True) -> dict:
    pairs = dataset()
    X, y, w, g = _xy(pairs)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = X / sd                                     # differences: no centring needed (they're antisymmetric)
    folds = np.array([_fold(x) for x in g])
    correct = total = 0.0
    for f in range(5):
        tr, te = folds != f, folds == f
        if te.sum() == 0:
            continue
        m = _fit(Xs[tr], y[tr], w[tr])
        p = m.predict(Xs[te])
        correct += float(((p == y[te]) * w[te]).sum()); total += float(w[te].sum())
    acc = correct / total if total else None
    m = _fit(Xs, y, w)
    coef = (m.coef_[0] / sd).tolist()
    out = {"pairs": len(pairs), "sets": len(set(g)), "unseen_pair_accuracy": round(acc, 3) if acc else None,
           "critic_same_sets": critic_baseline(), "pick_of_4_unseen": pick_of_4(Xs, folds, sd)}
    if save:
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        json.dump({"coef": coef, "n": len(pairs), "acc": acc, "dim": len(coef)}, open(MODEL, "w"))
    return out


def pick_of_4(Xs, folds, sd) -> str:
    """On unseen sets: how often the model's top option was your best (chance 1 in 4)."""
    from . import ab
    raw = json.load(open(os.path.join(AB, "answers.json"), encoding="utf-8"))
    pairs = dataset()
    X, y, w, g = _xy(pairs)
    hit = n = 0
    for f in range(5):
        tr = np.array([_fold(x) for x in g]) != f
        m = _fit((X / sd)[tr], y[tr], w[tr])
        coef = m.coef_[0] / sd
        for a in raw:
            q = a.get("quad") or {}
            if a.get("kind") != "quad" or q.get("repeat_of") or ab.stale(q.get("id", "")) or _fold(q.get("id", "")) != f:
                continue
            if a.get("best") is None or a.get("best") == a.get("worst"):
                continue
            stored = q.get("plans") or [None] * len(q["options"])
            sc = [features(o, plan_of(o, stored[k] if k < len(stored) else None), _ctx(q, o)) @ coef for k, o in enumerate(q["options"])]
            hit += int(np.argmax(sc) == a["best"]); n += 1
    return f"best {hit} of {n} ({hit / max(1, n):.0%}; chance 25%)"


def critic_baseline() -> str:
    """The critic's pick on the same answered sets (stored when each set was made)."""
    from . import ab
    raw = json.load(open(os.path.join(AB, "answers.json"), encoding="utf-8"))
    hit = n = 0
    for a in raw:
        q = a.get("quad") or {}
        if a.get("kind") != "quad" or q.get("repeat_of") or ab.stale(q.get("id", "")):
            continue
        if a.get("best") is None or a.get("best") == a.get("worst") or q.get("model_pick") is None:
            continue
        hit += int(q["model_pick"] == a["best"]); n += 1
    return f"best {hit} of {n} ({hit / max(1, n):.0%})"


def score(opt, plan, ctx) -> float | None:
    """Higher = you'd more likely pick it. None until trained."""
    if not os.path.exists(MODEL):
        return None
    coef = np.array(json.load(open(MODEL))["coef"])
    f = features(opt, plan, ctx)
    return float(f @ coef) if len(f) == len(coef) else None


def report(top: int = 20) -> list:
    coef = json.load(open(MODEL))["coef"]
    names = ([f"family:{f}" for f in FAMILIES] + [f"{ln}@{(i + .5) / N:.2f}" for ln in LANES for i in range(N)] +
             ["layered", "drums overlap", "bass overlap", "melody overlap", "drums overlap unlocked", "bass overlap unlocked",
              "B drums in at", "A drums out at", "B bass in at", "A bass out at", "B melody in at", "A melody out at",
              "has A hook", "has B hook", "has A sound", "has B sound", "fullness min", "fullness max", "fullness jump",
              "time thin", "locked", "glide", "tempo gap", "same song", "bars/8", "loud tease", "bridge mode"])
    return sorted(zip(names, coef), key=lambda t: -abs(t[1]))[:top]


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        print(json.dumps(train(), indent=1))
    elif cmd == "report":
        for n_, c in report():
            print(f"{c:+.3f}  {n_}")
