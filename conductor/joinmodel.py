"""Learned join score: how natural does "leave A at t_out → enter B at t_in" sound?

Trained without anyone rating anything (self-supervised):
  natural   = every real section change inside every song (what the composer wrote)
  unnatural = a section end spliced into the start of some other section (same song or another)
Later, proven-good joins (A/B winners, well-rated moments) are added as extra natural examples,
and proven-bad ones as unnatural — so it keeps learning from use.

    python -m conductor.joinmodel train      # build features, cross-validate by song, save
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys

import numpy as np

from . import joins

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(HERE, "cache", "join_model.pkl")
EXTRA_PATH = os.path.join(HERE, "cache", "join_examples.json")   # learned-from-use examples
_MODEL = None


def _usable(track, seg) -> bool:
    from .states import role_of
    return seg.length >= 6 and role_of(seg) is not None


def build_dataset(lib, rng: random.Random, neg_per_pos: int = 3):
    X, y, groups = [], [], []
    tracks = [t for t in lib.tracks if len(t.segments) >= 3]
    for gi, A in enumerate(tracks):
        segs = A.segments
        for i in range(len(segs) - 1):
            s, nxt = segs[i], segs[i + 1]
            if not (_usable(A, s) and _usable(A, nxt)):
                continue
            t = nxt.start
            X.append(joins.vector(joins.features(A, t, A, t))); y.append(1); groups.append(gi)
            for k in range(neg_per_pos):
                if k % 2 == 0:                              # same song, a different section
                    cands = [x for j, x in enumerate(segs) if j not in (i, i + 1) and _usable(A, x)]
                    if not cands:
                        continue
                    B, t_in = A, rng.choice(cands).start
                else:                                       # another song
                    B = rng.choice([x for x in tracks if x is not A])
                    cands = [x for x in B.segments if _usable(B, x)]
                    if not cands:
                        continue
                    t_in = rng.choice(cands).start
                X.append(joins.vector(joins.features(A, t, B, t_in))); y.append(0); groups.append(gi)
    # learned from use: proven good / bad joins
    if os.path.exists(EXTRA_PATH):
        by_title = {t.title: t for t in lib.tracks}
        for ex in json.load(open(EXTRA_PATH, encoding="utf-8")):
            A, B = by_title.get(ex["a"]), by_title.get(ex["b"])
            if A and B:
                for _ in range(int(ex.get("weight", 3))):  # your judgement counts more than one example
                    X.append(joins.vector(joins.features(A, ex["t_out"], B, ex["t_in"])))
                    y.append(int(ex["good"])); groups.append(-1)
    return np.array(X), np.array(y), np.array(groups)


def train(lib, seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    rng = random.Random(seed)
    X, y, g = build_dataset(lib, rng)
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=250,
                                           l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=250,
                                           l2_regularization=1.0, random_state=seed).fit(X, y)
    pickle.dump({"model": model, "features": joins.MODEL_FEATURES}, open(MODEL_PATH, "wb"))
    global _MODEL
    _MODEL = None
    return {"examples": int(len(y)), "natural": int(y.sum()), "auc_by_song": [round(a, 3) for a in aucs],
            "auc_mean": round(float(np.mean(aucs)), 3)}


def score(A, t_out: float, B, t_in: float) -> float | None:
    """0..1: how natural this join is (None if no model yet)."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            return None
        _MODEL = pickle.load(open(MODEL_PATH, "rb"))
        joins.MODEL_FEATURES = _MODEL["features"]
    f = joins.features(A, t_out, B, t_in)
    return float(_MODEL["model"].predict_proba(joins.vector(f)[None, :])[0, 1])


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "train":
    from .library import Library
    lib = Library.build(__import__("conductor.playlists", fromlist=["train_music"]).train_music(), os.path.join(HERE, "cache"), progress=lambda m: None,
                        backfill=False, cached_only=True)
    print(json.dumps(train(lib), indent=1))
