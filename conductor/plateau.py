"""Why is taste agreement stuck at ~64%?  (dev tool, ML venv)

    .venv-ml\\Scripts\\python.exe -m conductor.plateau            -> review/ab/plateau.json

1. ensemble: score live + every saved fresh critic on the SAME folds as generations.agreement,
   then average their taste-layer probabilities (all, top-k chosen on SELECT only).
2. misses: break the live/ensemble misses down by what the question was really about
   (song vs join, join style, context, how confident the model was).
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import os

import numpy as np

from . import critic
from .generations import _locked

HERE = critic.HERE
FEAT = os.path.join(critic.OUT, "plateau_feats")
OUT = os.path.join(HERE, "review", "ab", "plateau.json")


def _feats(path: str, names, rev, clips) -> dict:
    os.makedirs(FEAT, exist_ok=True)
    key = hashlib.md5((path + str(os.path.getmtime(path)) + str(len(names))).encode()).hexdigest()[:12]
    f = os.path.join(FEAT, key + ".npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        if list(z["names"]) == names:
            return dict(zip(names, z["vecs"]))
    critic.MODEL_PATH = path
    vecs, _ = critic.clip_vectors([[os.path.join(rev, c), clips[c]] for c in names], cache=False)
    critic.MODEL_PATH = None
    np.savez(f, names=np.array(names, dtype=object), vecs=np.asarray(vecs))
    return dict(zip(names, vecs))


def _probs(feat, comps, S, L, g):
    """Out-of-fold P(winner) on SELECT (5 group folds, as generations.agreement) + P on LOCKED."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    D = np.array([feat[w] - feat[l] for w, l, _ in comps])
    X = np.concatenate([D[S], -D[S]]); y = np.r_[np.ones(S.sum()), np.zeros(S.sum())]; gg = np.r_[g[S], g[S]]
    sd = X.std(0) + 1e-6
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, gg):
        pred[te] = LogisticRegression(C=0.01, max_iter=3000).fit(X[tr] / sd, y[tr]).predict_proba(X[te] / sd)[:, 1]
    n = S.sum()
    p_sel = 0.5 * (pred[:n] + (1 - pred[n:]))          # symmetric: both orderings of each pick
    lr = LogisticRegression(C=0.01, max_iter=3000, fit_intercept=False).fit(X / sd, y)
    p_lock = lr.predict_proba(D[L] / sd)[:, 1]
    return p_sel, p_lock


def _meta():
    """clip path -> what that option was (song, join style, context)."""
    R = json.load(open(os.path.join(HERE, "review", "ab", "answers.json"), encoding="utf-8"))
    m = {}
    for a in R:
        if a.get("kind") == "quad":
            q = a["quad"]
            for c, o in zip(q["clips"], q["options"]):
                m[c] = {"ctx": q.get("context"), "from": o.get("a"), "to": o.get("b"),
                        "style": (o.get("steps") or [{}])[0].get("style"), "lead": o.get("lead_bars"), "q": "quad"}
        elif a.get("pair"):
            p = a["pair"]
            for k in ("a", "b"):
                o = p.get(k) or {}
                m[p.get("clip_" + k)] = {"ctx": p.get("context"), "from": o.get("a"), "to": o.get("b"),
                                          "style": (o.get("steps") or [{}])[0].get("style"), "lead": o.get("lead_bars"),
                                          "q": "pair"}
    return m


def _breakdown(comps, right, conf, meta):
    rows = collections.defaultdict(lambda: [0, 0])
    def add(k, ok):
        rows[k][0] += 1; rows[k][1] += int(ok)
    for (w, l, _), ok, c in zip(comps, right, conf):
        mw, ml = meta.get(w, {}), meta.get(l, {})
        same_song = mw.get("to") == ml.get("to")
        stay = mw.get("to") == mw.get("from") and ml.get("to") == ml.get("from")
        add("question: " + ("same destination, join only" if same_song else "different destination songs"), ok)
        if stay:
            add("question: both stay in the same song", ok)
        add("format: " + str(mw.get("q")), ok)
        add("context: " + str(mw.get("ctx")), ok)
        add("winner style: " + str(mw.get("style")), ok)
        add("styles: " + ("same style" if mw.get("style") == ml.get("style") else "different styles"), ok)
        add("model confidence: " + ("sure (>65/35)" if c > 0.15 else "leaning (55-65)" if c > 0.05 else "coin flip (<55)"), ok)
    return {k: {"n": n, "right": round(r / n, 3)} for k, (n, r) in sorted(rows.items()) if n >= 15}


def main():
    clips, comps, rev = critic.comparisons()
    names = sorted(clips)
    lock = np.array([_locked(c[2]) for c in comps])
    S, L = ~lock, lock
    g = np.array([int(hashlib.md5(str(c[2]).encode()).hexdigest()[:8], 16) for c in comps])
    paths = [("live", os.path.join(critic.OUT, "critic.pt"))]
    for run in ("20261123_0011", "20261123_0831"):
        for p in sorted(glob.glob(os.path.join(critic.OUT, "gens", run, "g*_fresh.pt"))):
            paths.append((run[-4:] + "/" + os.path.basename(p)[:3], p))
    P = {}
    for name, p in paths:
        P[name] = _probs(_feats(p, names, rev, clips), comps, S, L, g)
        ps, pl = P[name]
        print(f"{name:12s} select {(ps > .5).mean():.1%}  locked {(pl > .5).mean():.1%}", flush=True)
    nS, nL = S.sum(), L.sum()
    w = lambda s, l: (s * nS + l * nL) / (nS + nL)
    res = {"n_select": int(nS), "n_locked": int(nL), "critics": {}, "ensembles": {}}
    for k, (ps, pl) in P.items():
        res["critics"][k] = {"select": round(float((ps > .5).mean()), 4), "locked": round(float((pl > .5).mean()), 4)}
        res["critics"][k]["weighted"] = round(w(res["critics"][k]["select"], res["critics"][k]["locked"]), 4)
    fresh = [k for k in P if k != "live"]
    by_sel = sorted(fresh, key=lambda k: -res["critics"][k]["select"])
    sets = {"all fresh (20)": fresh, "all fresh + live (21)": fresh + ["live"],
            "top 5 on select": by_sel[:5], "top 10 on select": by_sel[:10], "top 5 + live": by_sel[:5] + ["live"]}
    for nm, ks in sets.items():
        ps = np.mean([P[k][0] for k in ks], 0); pl = np.mean([P[k][1] for k in ks], 0)
        s, l = float((ps > .5).mean()), float((pl > .5).mean())
        res["ensembles"][nm] = {"select": round(s, 4), "locked": round(l, 4), "weighted": round(w(s, l), 4), "members": ks}
        print(f"ENSEMBLE {nm:22s} select {s:.1%}  locked {l:.1%}  weighted {w(s, l):.1%}", flush=True)
    # misses: on every pick (SELECT out-of-fold + LOCKED), live and the all-fresh+live ensemble
    meta = _meta()
    order = np.r_[np.flatnonzero(S), np.flatnonzero(L)]
    cs = [comps[i] for i in order]
    for nm, ks in (("live", ["live"]), ("ensemble", fresh + ["live"])):
        p = np.r_[np.mean([P[k][0] for k in ks], 0), np.mean([P[k][1] for k in ks], 0)]
        res["misses_" + nm] = _breakdown(cs, p > .5, np.abs(p - .5), meta)
    # do critics disagree? (if they all miss the same picks, averaging can't help)
    allp = np.array([np.r_[P[k][0], P[k][1]] > .5 for k in fresh + ["live"]])
    vote = allp.mean(0)
    res["agreement_between_critics"] = {
        "all critics right": round(float((vote == 1).mean()), 3), "all critics wrong": round(float((vote == 0).mean()), 3),
        "split": round(float(((vote > 0) & (vote < 1)).mean()), 3)}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res["agreement_between_critics"]), flush=True)
    for nm in ("misses_live", "misses_ensemble"):
        print("\n" + nm)
        for k, v in res[nm].items():
            print(f"  {v['right']:.0%}  n={v['n']:4d}  {k}")


if __name__ == "__main__":
    main()
