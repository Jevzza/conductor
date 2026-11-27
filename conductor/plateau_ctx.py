"""Experiment: does a context-aware taste layer beat the audio-only one?  (dev, ML venv; needs plateau.py's cached features)

Audio-only (today): logistic regression on feat[winner] - feat[loser].
Context-aware: the same, plus that difference multiplied by the transition direction (up / down / same state),
plus join-style and same-song indicators (winner minus loser) and their direction interactions.
Same folds and SELECT/LOCKED split as generations.agreement.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re

import numpy as np

from . import critic
from .generations import _locked
from .plateau import _feats, _meta

RANK = {"Calm": 0, "Fight": 1, "Horde": 2, "Boss": 3}
STYLES = ["xfade1", "xfade2", "xfade4", "mix", "dj", "djblend", "djverb", "echo", "punch", "riser", "game", "natural",
          "melfirst", "rhyfirst", "drumbridge", "basscarry", "hookover", "early_lead", None]   # 2026-11-25: + recipe families

def style_key(st):
    """A join's style for the critic's taste layer: layered recipes by FAMILY (rhyfirst8h -> rhyfirst), the
    early lead-ins (tease/strip/fill/hook/swap/layer, 24 Nov) as one bucket, everything else as itself."""
    if not st:
        return None
    m = re.match(r"(melfirst|rhyfirst|drumbridge|basscarry|hookover)\d", st)
    if m:
        return m.group(1)
    if re.match(r"(tease|strip|fill|hook4|hookout|swap|layer)", st):
        return "early_lead"
    return st



def _direction(ctx):
    try:
        a, b = [s.strip() for s in ctx.split("→")]
        return "up" if RANK[b] > RANK[a] else "down" if RANK[b] < RANK[a] else "same"
    except Exception:
        return "same"


def _opt_vec(m):
    v = np.zeros(len(STYLES) + 2)
    k = style_key(m.get("style"))
    v[STYLES.index(k) if k in STYLES else len(STYLES) - 1] = 1
    v[-2] = float(m.get("to") == m.get("from"))           # stays in the same song
    v[-1] = float(bool(m.get("lead")))                     # build-up lead-in
    return v


def design(feat, comps, meta, ctx_on):
    """ctx_on: False (audio only) | True/'full' | 'style' (audio + style, no direction) |
    'styledir' (style x direction only, no audio) | 'audio+styledir' (no audio x direction)."""
    mode = "full" if ctx_on is True else ctx_on
    rows = []
    for w, l, _ in comps:
        d = feat[w] - feat[l]
        if not mode:
            rows.append(d); continue
        mw, ml = meta.get(w, {}), meta.get(l, {})
        dirn = _direction(mw.get("ctx") or "")
        oh = np.array([dirn == "up", dirn == "down", dirn == "same"], float)
        s = _opt_vec(mw) - _opt_vec(ml)
        parts = {"full": [d, np.kron(oh, d), s, np.kron(oh, s)], "style": [d, s],
                 "styledir": [s, np.kron(oh, s)], "audio+styledir": [d, s, np.kron(oh, s)]}[mode]
        rows.append(np.concatenate(parts))
    return np.array(rows)


def score(D, S, L, g, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = np.concatenate([D[S], -D[S]]); y = np.r_[np.ones(S.sum()), np.zeros(S.sum())]; gg = np.r_[g[S], g[S]]
    sd = X.std(0) + 1e-6
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, gg):
        pred[te] = LogisticRegression(C=C, max_iter=5000).fit(X[tr] / sd, y[tr]).predict_proba(X[te] / sd)[:, 1]
    n = S.sum()
    p_sel = 0.5 * (pred[:n] + (1 - pred[n:]))
    lr = LogisticRegression(C=C, max_iter=5000, fit_intercept=False).fit(X / sd, y)
    return p_sel, lr.predict_proba(D[L] / sd)[:, 1]


def main():
    clips, comps, rev = critic.comparisons()
    names = sorted(clips)
    S = ~np.array([_locked(c[2]) for c in comps]); L = ~S
    g = np.array([int(hashlib.md5(str(c[2]).encode()).hexdigest()[:8], 16) for c in comps])
    meta = _meta()
    nS, nL = S.sum(), L.sum()
    w = lambda s, l: (s * nS + l * nL) / (nS + nL)
    crit = {"live": os.path.join(critic.OUT, "critic.pt")}
    for run in ("20261123_0011", "20261123_0831"):
        for p in sorted(glob.glob(os.path.join(critic.OUT, "gens", run, "g*_fresh.pt"))):
            crit[run[-4:] + "/" + os.path.basename(p)[:3]] = p
    F = {k: _feats(p, names, rev, clips) for k, p in crit.items()}
    import sys
    if "--ablate" in sys.argv:
        for C in (0.003, 0.01, 0.03):
            for mode in (False, "style", "styledir", "audio+styledir", "full"):
                ps, pl = score(design(F["live"], comps, meta, mode), S, L, g, C)
                s, l = (ps > .5).mean(), (pl > .5).mean()
                print(f"C={C:<6} {str(mode or 'audio only'):15s} select {s:.1%} locked {l:.1%} weighted {w(s, l):.1%}", flush=True)
        return
    for C in (0.003, 0.01, 0.03):
        for ctx_on in (False, True):
            P = {k: score(design(F[k], comps, meta, ctx_on), S, L, g, C) for k in F}
            lv = P["live"]
            s, l = (lv[0] > .5).mean(), (lv[1] > .5).mean()
            allp = [np.mean([P[k][0] for k in P], 0), np.mean([P[k][1] for k in P], 0)]
            es, el = (allp[0] > .5).mean(), (allp[1] > .5).mean()
            print(f"C={C:<6} {'context' if ctx_on else 'audio  '}  live: select {s:.1%} locked {l:.1%} weighted {w(s, l):.1%}"
                  f"   |  all 21 averaged: select {es:.1%} locked {el:.1%} weighted {w(es, el):.1%}", flush=True)


if __name__ == "__main__":
    main()
