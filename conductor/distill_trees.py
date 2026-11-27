"""Bring the critic's ear into the live planner (2026-11-23).

Offline, the critic's audio features + the join's style agree with your A/B picks ~68% (plateau_ctx.py),
but the live planner can't run the critic: it scores candidates with the trees taste model (ab.taste_many,
62-65%), which knows the style but not the audio. So: the critic+style "teacher" scores ~2,400 rendered
practice joins (cache/distill2, routes kept), and its confident calls become extra training rows for the
trees. The live planner stays exactly as cheap as before.

    python -m conductor.distill render2 400                         # main venv: practice joins + routes
    .venv-ml\\Scripts\\python -m conductor.distill_trees feats        # critic vectors for practice + A/B clips
    python -m conductor.distill_trees eval                          # honest folds: trees vs trees + teacher rows
    python -m conductor.distill_trees apply                         # write the rows ab.labelled() picks up

Honest test: same moment-grouped folds as ab.train_taste; in each fold the teacher is fitted only on
A/B picks from the training moments, so no held-out answer leaks in through the pseudo-labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D2 = os.path.join(HERE, "cache", "distill2")
VECS = os.path.join(D2, "vecs.npz")
ROWS = os.path.join(HERE, "cache", "live", "teacher_rows.json")
REV = os.path.join(HERE, "review")
STYLES = ["xfade1", "xfade2", "xfade4", "mix", "dj", "djblend", "djverb", "echo", "punch", "riser", "game", "natural",
          "melfirst", "rhyfirst", "drumbridge", "basscarry", "hookover", "early_lead", None]   # breakdown: unknown bucket until retrained (vector size is baked into the live taste layer)   # 2026-11-25: + recipe families

def style_key(st):
    """A join's style for the critic's taste layer: layered recipes by FAMILY (rhyfirst8h -> rhyfirst), the
    early lead-ins (tease/strip/fill/hook/swap/layer, 24 Nov) as one bucket, everything else as itself."""
    if not st:
        return None
    m = re.match(r"(melfirst|rhyfirst|drumbridge|basscarry|hookover|breakdown)\d", st)
    if m:
        return m.group(1)
    if re.match(r"(tease|strip|fill|hook4|hookout|swap|layer)", st):
        return "early_lead"
    return st



def _opt_vec(route) -> np.ndarray:
    st = route["steps"][0]
    v = np.zeros(len(STYLES) + 2)
    k = style_key(st.get("style"))
    v[STYLES.index(k) if k in STYLES else len(STYLES) - 1] = 1
    v[-2] = float(st["b"] == route["a"])
    v[-1] = float(bool(route.get("lead_bars")))
    return v


def _manifest():
    return json.load(open(os.path.join(D2, "manifest.json"), encoding="utf-8"))


def _ab_clips():
    """A/B clip -> its route, and the head-to-heads (winner clip, loser clip, moment key)."""
    R = json.load(open(os.path.join(REV, "ab", "answers.json"), encoding="utf-8"))
    route, comps = {}, []
    from .ab import noisy, training_pairs
    nz = noisy()
    for a in R:
        if a.get("kind") == "quad":
            q = a["quad"]
            from .ab import stale
            if q.get("repeat_of") or stale(q["id"]) or a.get("best") is None or a["best"] == a.get("worst"):
                continue
            if not all(os.path.exists(os.path.join(REV, c)) for c in q["clips"]):
                continue
            for c, o in zip(q["clips"], q["options"]):
                route[c] = o
            key = (q["options"][0]["a"], round(q["options"][0]["t_out"]))
            comps += [(q["clips"][x], q["clips"][y], key) for x, y in training_pairs(a, nz)]
        elif a.get("pair") and a.get("choice") in ("a", "b"):
            p = a["pair"]
            ca, cb = p.get("clip_a"), p.get("clip_b")
            if not (ca and cb and os.path.exists(os.path.join(REV, ca)) and os.path.exists(os.path.join(REV, cb))):
                continue
            route[ca], route[cb] = p["a"], p["b"]
            w, l = (ca, cb) if a["choice"] == "a" else (cb, ca)
            comps.append((w, l, (p["a"]["a"], round(p["a"]["t_out"]))))
    for o in route.values():
        o.setdefault("steps", [{"b": o.get("b"), "t_in": o.get("t_in"), "style": o.get("style")}])
    return route, comps


# ── ML venv: what the critic hears ─────────────────────────────────────────────
def feats():
    from . import critic
    man = _manifest()
    route, _ = _ab_clips()
    jobs, names = [], []
    for m in man:
        for c in m["clips"]:
            names.append("D:" + c["clip"]); jobs.append([os.path.join(D2, c["clip"]), c["seams"]])
    ab_seams = {}
    for a in json.load(open(os.path.join(REV, "ab", "answers.json"), encoding="utf-8")):
        if a.get("kind") == "quad":
            ab_seams.update(zip(a["quad"]["clips"], a["quad"]["seams"]))
        elif a.get("pair"):
            p = a["pair"]
            ab_seams[p.get("clip_a")] = p.get("seams_a") or [p.get("seam", 6.0)]
            ab_seams[p.get("clip_b")] = p.get("seams_b") or [p.get("seam", 6.0)]
    for c in route:
        names.append("R:" + c); jobs.append([os.path.join(REV, c), ab_seams.get(c) or [6.0]])
    # incremental: keep vectors already computed with this same critic (nightly only adds the new answers)
    mtime = os.path.getmtime(os.path.join(critic.OUT, "critic.pt"))
    have = {}
    if os.path.exists(VECS):
        z = np.load(VECS, allow_pickle=True)
        if float(z["critic_mtime"]) == mtime:
            have = dict(zip(list(z["names"]), z["vecs"]))
    todo = [(n, j) for n, j in zip(names, jobs) if n not in have]
    print(f"{len(names)} clips, {len(todo)} new", flush=True)
    for k in range(0, len(todo), 200):
        part = todo[k:k + 200]
        v, _ = critic.clip_vectors([j for _, j in part], cache=False)
        have.update({n: x for (n, _), x in zip(part, v)})
        print(f"  {min(k + 200, len(todo))}/{len(todo)}", flush=True)
    keep = [n for n in names if n in have]
    np.savez(VECS, names=np.array(keep, dtype=object), vecs=np.asarray([have[n] for n in keep], np.float32),
             critic_mtime=mtime)


# ── main venv: teacher -> rows for the trees ───────────────────────────────────
def _load_vecs():
    z = np.load(VECS, allow_pickle=True)
    return dict(zip(list(z["names"]), z["vecs"]))


def _teacher(V, route, comps, C=0.01):
    """Pairwise logistic on [critic vector, style vector] (winner - loser). Returns a clip scorer."""
    from sklearn.linear_model import LogisticRegression
    X = np.array([np.r_[V["R:" + w] - V["R:" + l], _opt_vec(route[w]) - _opt_vec(route[l])] for w, l, _ in comps])
    X = np.concatenate([X, -X]); y = np.r_[np.ones(len(comps)), np.zeros(len(comps))]
    sd = X.std(0) + 1e-6
    lr = LogisticRegression(C=C, max_iter=5000, fit_intercept=False).fit(X / sd, y)
    w = lr.coef_[0] / sd
    return lambda vec, r: float(np.r_[vec, _opt_vec(r)] @ w)


def _pseudo_rows(score, V, conf=0.34, weight=1.0):
    """Per practice moment: the teacher's favourite = good, its least favourite = bad, kept only where its
    preference is among the clearest `conf` share."""
    out, gaps = [], []
    for m in _manifest():
        sc = [(score(V["D:" + c["clip"]], c["route"]), c["route"]) for c in m["clips"] if "D:" + c["clip"] in V]
        if len(sc) < 3:
            continue
        sc.sort(key=lambda x: x[0])
        gaps.append((sc[-1][0] - sc[0][0], sc[-1][1], sc[0][1]))
    if not gaps:
        return out
    cut = np.quantile([g for g, _, _ in gaps], 1 - conf)
    full = lambda r: {"b": r["steps"][-1]["b"], "t_in": r["steps"][-1]["t_in"], **r}
    for g, best, worst in gaps:
        if g >= cut:
            out += [dict(full(best), good=1, w=weight, src="teacher"), dict(full(worst), good=0, w=weight, src="teacher")]
    return out


def _gkey(r):
    return int(hashlib.md5(f"{r['a']}|{round(r['t_out'])}".encode()).hexdigest()[:8], 16)


def evaluate(confs=(0.2, 0.34, 0.5), weights=(0.5, 1.0, 2.0)):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from . import ab, routes
    from .library import Library
    lib = Library.build(__import__("conductor.playlists", fromlist=["train_music"]).train_music(), os.path.join(HERE, "cache"), progress=lambda m: None,
                        backfill=False, cached_only=True)
    by = {t.title: t for t in lib.tracks}
    V = _load_vecs()
    route, comps = _ab_clips()
    comps = [c for c in comps if "R:" + c[0] in V and "R:" + c[1] in V]
    rows = [r for r in ab.labelled(lib) if r["a"] in by and r["b"] in by]
    F = [ab._feat(by, r) for r in rows]
    keys = sorted(F[0])
    X = np.array([[f[k] for k in keys] for f in F]); y = np.array([r["good"] for r in rows])
    wt = np.array([r.get("w", 1.0) for r in rows]) * np.array([0.5 if len(routes.as_route(r)["steps"]) > 1 else 1.0 for r in rows])
    g = np.array([_gkey(r) for r in rows])
    ab_rows = [k for k, r in enumerate(rows) if r.get("w") == 2.0]
    mk = lambda: HistGradientBoostingClassifier(max_depth=2, max_iter=150, learning_rate=0.05, l2_regularization=2.0)
    folds = list(GroupKFold(5).split(X, y, g))
    # teacher rows per fold (the teacher never sees that fold's moments), featurised once
    fold_pseudo = []
    for tr, te in folds:
        test_keys = {g[i] for i in te}
        ctr = [c for c in comps if int(hashlib.md5(f"{c[2][0]}|{c[2][1]}".encode()).hexdigest()[:8], 16) not in test_keys]
        sc = _teacher(V, route, ctr)
        fold_pseudo.append(sc)

    def run(conf, weight):
        p = np.zeros(len(y))
        for (tr, te), sc in zip(folds, fold_pseudo):
            Xa, ya, wa = X[tr], y[tr], wt[tr]
            if conf:
                pr = [r for r in _pseudo_rows(sc, V, conf, weight) if r["a"] in by and all(s["b"] in by for s in r["steps"])]
                if pr:
                    Fp = [ab._feat(by, r) for r in pr]
                    Xa = np.vstack([Xa, [[f.get(k, 0.0) for k in keys] for f in Fp]])
                    ya = np.r_[ya, [r["good"] for r in pr]]
                    wa = np.r_[wa, [r["w"] for r in pr]]
            p[te] = mk().fit(Xa, ya, sample_weight=wa).predict_proba(X[te])[:, 1]
        hits = [p[ab_rows[k]] > p[ab_rows[k + 1]] for k in range(0, len(ab_rows) - 1, 2)]
        return float(roc_auc_score(y, p)), sum(hits), len(hits)
    out = {}
    auc, h, n = run(0, 0)
    out["trees only"] = {"auc": round(auc, 3), "ab": f"{h}/{n} ({h / n:.1%})"}
    print(f"trees only                 AUC {auc:.3f}   A/B picks {h}/{n} ({h / n:.1%})", flush=True)
    for conf in confs:
        for weight in weights:
            auc, h, n = run(conf, weight)
            out[f"conf {conf} w {weight}"] = {"auc": round(auc, 3), "ab": f"{h}/{n} ({h / n:.1%})"}
            print(f"+ teacher conf {conf:<4} w {weight:<3}  AUC {auc:.3f}   A/B picks {h}/{n} ({h / n:.1%})", flush=True)
    json.dump(out, open(os.path.join(D2, "eval.json"), "w"), indent=1)
    return out


TEACHER = os.path.join(HERE, "cache", "live", "critic_taste.json")


def export_teacher(C=0.01) -> dict:
    """The critic + style taste layer for the live critic (critic_live.py): score = [clip vector, style vector] . w."""
    from sklearn.linear_model import LogisticRegression
    V = _load_vecs()
    route, comps = _ab_clips()
    comps = [c for c in comps if "R:" + c[0] in V and "R:" + c[1] in V]
    X = np.array([np.r_[V["R:" + w] - V["R:" + l], _opt_vec(route[w]) - _opt_vec(route[l])] for w, l, _ in comps])
    X = np.concatenate([X, -X]); y = np.r_[np.ones(len(comps)), np.zeros(len(comps))]
    sd = X.std(0) + 1e-6
    w = LogisticRegression(C=C, max_iter=5000, fit_intercept=False).fit(X / sd, y).coef_[0] / sd
    z = np.load(VECS, allow_pickle=True)
    out = {"w": w.tolist(), "styles": [str(s) for s in STYLES], "comparisons": len(comps),
           "critic_mtime": float(z["critic_mtime"]), "built": __import__("time").strftime("%Y-%m-%d %H:%M")}
    os.makedirs(os.path.dirname(TEACHER), exist_ok=True)
    json.dump(out, open(TEACHER, "w"))
    return out


def apply(conf=0.34, weight=1.0):
    """Teacher fitted on every A/B pick -> rows for the live trees (ab.labelled reads ROWS)."""
    V = _load_vecs()
    route, comps = _ab_clips()
    comps = [c for c in comps if "R:" + c[0] in V and "R:" + c[1] in V]
    rows = _pseudo_rows(_teacher(V, route, comps), V, conf, weight)
    os.makedirs(os.path.dirname(ROWS), exist_ok=True)
    json.dump({"conf": conf, "weight": weight, "comparisons": len(comps), "rows": rows}, open(ROWS, "w", encoding="utf-8"))
    return len(rows)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "feats":
        feats()
    elif cmd == "eval":
        evaluate()
    elif cmd == "teacher":
        t = export_teacher()
        print(t["comparisons"], "comparisons ->", TEACHER)
    elif cmd == "apply":
        a = [float(x) for x in sys.argv[2:]]
        print(apply(*a), "teacher rows ->", ROWS)
