"""Generations: train many critics on the SAME data, keep the one that agrees with you most.

    .venv-ml\\Scripts\\python.exe -m conductor.generations [hours=2] [explore=10] [margin=0.03]

Your A/B picks are split once, by a stable hash, into SELECT (80%) and LOCKED (20%).
  * score  = the taste layer's agreement with you on SELECT picks it wasn't fitted on (5-fold)
  * LOCKED = never used to choose; only to confirm the final winner isn't a fluke

Gen 0 is the live critic (the champion). Then:
  explore: fresh critics (new seed each) until one beats the champion by `margin`
  exploit: once one has, the remaining time fine-tunes the champion (new seed, few epochs, low LR);
           a child replaces it only if it also beats it by `margin`
At the end the champion goes live ONLY if it is a new critic AND it isn't worse on LOCKED.
Progress: review/ab/generations.json
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from . import critic

HERE = critic.HERE
RUN = os.path.join(critic.OUT, "gens", time.strftime("%Y%m%d_%H%M"))
REPORT = os.path.join(HERE, "review", "ab", "generations.json")


def _locked(group) -> bool:
    return int(hashlib.md5(("lock|" + str(group)).encode()).hexdigest()[:8], 16) % 5 == 0


def agreement(model_path: str) -> dict:
    """How often this critic (+ a taste layer fitted on SELECT picks) agrees with you."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    critic.MODEL_PATH = model_path
    clips, comps, rev = critic.comparisons()
    names = list(clips)
    vecs, _ = critic.clip_vectors([[os.path.join(rev, c), clips[c]] for c in names], cache=False)
    feat = dict(zip(names, vecs))
    lock = np.array([_locked(c[2]) for c in comps])
    D = np.array([feat[w] - feat[l] for w, l, _ in comps])
    g = np.array([int(hashlib.md5(str(c[2]).encode()).hexdigest()[:8], 16) for c in comps])
    alone = np.array([feat[w][-2] > feat[l][-2] for w, l, _ in comps])
    S, L = ~lock, lock
    X = np.concatenate([D[S], -D[S]]); y = np.r_[np.ones(S.sum()), np.zeros(S.sum())]; gg = np.r_[g[S], g[S]]
    sd = X.std(0) + 1e-6
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, gg):
        pred[te] = LogisticRegression(C=0.01, max_iter=3000).fit(X[tr] / sd, y[tr]).predict_proba(X[te] / sd)[:, 1]
    sel = float((pred[:S.sum()] > 0.5).mean())
    lr = LogisticRegression(C=0.01, max_iter=3000, fit_intercept=False).fit(X / sd, y)
    locked = float((lr.predict_proba(D[L] / sd)[:, 1] > 0.5).mean()) if L.any() else float("nan")
    critic.MODEL_PATH = None
    return {"select": round(sel, 4), "locked": round(locked, 4), "critic_alone_select": round(float(alone[S].mean()), 4),
            "n_select": int(S.sum()), "n_locked": int(L.sum())}


def _save(rep):
    tmp = REPORT + ".tmp"
    json.dump(rep, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, REPORT)


def _weighted(a: dict) -> float:
    """select and locked, weighted by how many picks each covers (fresh critics trade one for the other)."""
    n = a["n_select"] + a["n_locked"]
    return (a["select"] * a["n_select"] + a["locked"] * a["n_locked"]) / n if n else a["select"]


def run(hours: float = 2.0, explore: int = 10, margin: float = 0.03, max_gens: int = 0,
        score: str = "select") -> dict:
    """max_gens: stop after this many challengers (0 = only the time limit).
    score: 'select' (old rule) or 'weighted' (select + locked by pick count) for adoption."""
    key = (lambda a: a["select"]) if score == "select" else _weighted
    os.makedirs(RUN, exist_ok=True)
    t_end = time.time() + hours * 3600
    live = os.path.join(critic.OUT, "critic.pt")
    g0 = os.path.join(RUN, "g00_live.pt")
    shutil.copy(live, g0)
    rep = {"started": time.strftime("%Y-%m-%d %H:%M"), "hours": hours, "margin": margin, "score": score,
           "max_gens": max_gens, "explore": explore, "gens": [], "state": "running"}
    a = agreement(g0)
    rep["gens"].append({"gen": 0, "kind": "live", "path": g0, **a, "weighted": round(_weighted(a), 4)})
    champ = rep["gens"][0]
    rep["champion"] = 0
    _save(rep)
    print(f"gen  0  live      select {a['select']:.1%}  locked {a['locked']:.1%}", flush=True)
    gen, adopted, per = 0, False, None
    while True:
        gen += 1
        exploit = (adopted and score == "select") or gen > explore     # weighted runs stay fresh
        if max_gens and gen > max_gens:
            break
        if per is not None and time.time() + per * 1.1 > t_end:
            break
        t0 = time.time()
        path = os.path.join(RUN, f"g{gen:02d}_{'child' if exploit else 'fresh'}.pt")
        if exploit:
            tr = critic.train(epochs=4, seed=100 + gen, out=path, init=champ["path"], lr=5e-4)
        else:
            tr = critic.train(epochs=14, seed=gen, out=path)
        a = agreement(path)
        row = {"gen": gen, "kind": "child of %d" % champ["gen"] if exploit else "fresh", "path": path,
               "minutes": round((time.time() - t0) / 60, 1), **tr, **a, "weighted": round(_weighted(a), 4)}
        better = key(a) >= key(champ) + margin
        row["adopted"] = bool(better)
        rep["gens"].append(row)
        if better:
            champ, adopted = row, True
            rep["champion"] = gen
        _save(rep)
        print(f"gen {gen:2d}  {row['kind']:10s} select {a['select']:.1%}  locked {a['locked']:.1%}  "
              f"weighted {row['weighted']:.1%}  "
              f"({row['minutes']} min){'  <- new champion' if better else ''}", flush=True)
        took = time.time() - t0
        per = took if per is None else 0.5 * per + 0.5 * took
        if time.time() > t_end:
            break
    live_row = rep["gens"][0]
    if score == "select":
        deploy = champ["gen"] != 0 and champ["locked"] >= live_row["locked"]
        rep["verdict"] = ("no generation beat the live critic by the margin" if champ["gen"] == 0 else
                          f"gen {champ['gen']} won on SELECT (+{(champ['select'] - live_row['select']) * 100:.1f} pts) and "
                          + ("held up on LOCKED -> live" if deploy else "did NOT hold up on LOCKED -> kept the live critic"))
    else:                     # weighted already includes locked
        deploy = champ["gen"] != 0
        rep["verdict"] = ("no generation beat the live critic by the margin" if champ["gen"] == 0 else
                          f"gen {champ['gen']} won on WEIGHTED (+{(key(champ) - key(live_row)) * 100:.1f} pts) -> live")
    rep["deployed"] = bool(deploy)
    if deploy:
        shutil.copy(champ["path"], live)
        os.makedirs(os.path.join(critic.OUT, "versions"), exist_ok=True)
        shutil.copy(champ["path"], os.path.join(critic.OUT, "versions", time.strftime("critic_gen%Y%m%d_%H%M.pt")))
        rep["taste"] = critic.train_taste_head()
        critic.model_styles()
        subprocess.run(["python", "-m", "conductor.policy", "build"], cwd=HERE)
    rep["state"] = "done"
    rep["finished"] = time.strftime("%Y-%m-%d %H:%M")
    _save(rep)
    print(rep["verdict"], flush=True)
    return rep


if __name__ == "__main__":
    score = "weighted" if "--weighted" in sys.argv else "select"
    args = [float(x) for x in sys.argv[1:] if not x.startswith("--")]
    run(args[0] if args else 2.0, int(args[1]) if len(args) > 1 else 10, args[2] if len(args) > 2 else 0.03,
        int(args[3]) if len(args) > 3 else 0, score)
