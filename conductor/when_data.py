"""When should the music change? Learn it from the GAME: every recorded mission logs Darktide's own
music decisions (music_combat: none/normal/horde_low/horde_high/boss) next to the AI director's
signals (tension, attack intensity, kills, nearby enemies, aggro, toughness, horde/boss near...).

    python -m conductor.when_data        # build cache/game_data/when.npz + a baseline report

Rows: one per 0.5 s of mission time. Features: the latest value of each signal, its change over
the last 2 s and 5 s, the current game music state, seconds since that state began.
Targets: the state 3 s from now; does it change within the next 3 s.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "cache", "game_data")
PARAMS = ["tension", "player_attack_intensity", "kills_5s", "minion_counter_global", "aggro_near", "toughness",
          "player_locked_in_melee", "player_health", "horde_near", "boss_near", "enemies_20m"]
STATES = {"none": 0, "None": 0, "normal": 1, "horde_low": 2, "horde_high": 3, "boss": 4}
STEP, AHEAD = 0.5, 3.0


def _events(path):
    for l in open(path, encoding="utf-8", errors="replace"):
        try:
            e = json.loads(l)
        except ValueError:
            continue
        if e.get("t") is not None:
            yield e


def rows_for(path):
    ev = list(_events(path))
    if not any(e.get("k") == "state" and e.get("group") == "music_combat" for e in ev):
        return None
    ev.sort(key=lambda e: e["t"])
    t_end = ev[-1]["t"]
    cur = {p: 0.0 for p in PARAMS}
    hist = []                                   # (t, values)
    state, since, in_mission = 0, 0.0, False
    states_t = []                               # (t, state) for targets
    k = 0
    X, S, T = [], [], []
    t = ev[0]["t"]
    while t <= t_end:
        while k < len(ev) and ev[k]["t"] <= t:
            e = ev[k]; k += 1
            if e.get("k") == "param" and e.get("name") in cur:
                try:
                    cur[e["name"]] = float(e.get("value", 0) or 0)
                except (TypeError, ValueError):
                    pass
            elif e.get("k") == "state" and e.get("group") == "music_combat":
                s = STATES.get(str(e.get("value")), 0)
                if s != state:
                    state, since = s, e["t"]
            elif e.get("k") == "state" and e.get("group") == "music_game_state":
                in_mission = e.get("value") == "mission"
        vals = np.array([cur[p] for p in PARAMS], np.float32)
        hist.append((t, vals))
        if in_mission:
            def ago(dt):
                for tt, v in reversed(hist):
                    if tt <= t - dt:
                        return v
                return hist[0][1]
            X.append(np.concatenate([vals, vals - ago(2.0), vals - ago(5.0), np.eye(5)[state], [min(120.0, t - since)]]))
            S.append(state); T.append(t)
        states_t.append((t, state))
        t += STEP
    if not X:
        return None
    st = dict(states_t)
    times = sorted(st)
    def state_at(x):
        i = np.searchsorted(times, x, side="right") - 1
        return st[times[max(0, i)]]
    future = np.array([state_at(tt + AHEAD) for tt in T])
    change = np.array([any(state_at(tt + d) != s for d in np.arange(STEP, AHEAD + 1e-6, STEP)) for tt, s in zip(T, S)])
    return np.array(X, np.float32), np.array(S), future, change.astype(np.int8)


def build():
    files = sorted(glob.glob(os.path.join(HERE, "recordings", "*.jsonl")))
    Xs, S, F, C, G = [], [], [], [], []
    used = []
    # some recordings are cuts/copies of the same session: keep the longest of any that share
    # most of their game music-state changes (bridge game time + value)
    sigs = {}
    for f in files:
        sigs[f] = {(round(e["t"], 1), str(e.get("value"))) for e in _events(f)
                   if e.get("k") == "state" and e.get("group") == "music_combat"}
    keep = []
    for f in sorted(files, key=lambda f: -len(sigs[f])):
        if sigs[f] and all(len(sigs[f] & sigs[k]) < 0.5 * len(sigs[f]) for k in keep):
            keep.append(f)
    for g, f in enumerate(files):
        if f not in keep:
            continue
        r = rows_for(f)
        if r is None:
            continue
        X, s, fut, ch = r
        Xs.append(X); S.append(s); F.append(fut); C.append(ch); G.append(np.full(len(s), g))
        used.append((os.path.basename(f), len(s), int(ch.sum())))
    X, S, F, C, G = map(np.concatenate, (Xs, S, F, C, G))
    os.makedirs(OUT, exist_ok=True)
    names = PARAMS + [p + "_d2" for p in PARAMS] + [p + "_d5" for p in PARAMS] + ["st_none", "st_fight", "st_horde", "st_bighorde", "st_boss", "since"]
    np.savez_compressed(os.path.join(OUT, "when.npz"), X=X, state=S, future=F, change=C, group=G, names=np.array(names))
    return used, X, S, F, C, G, names


def baseline(X, S, F, C, G):
    """How well do the director's signals predict the game's own music decisions (unseen missions)?"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    k = min(5, len(np.unique(G)))
    pc, pf = np.zeros(len(C)), np.zeros(len(F))
    for tr, te in GroupKFold(k).split(X, C, G):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05).fit(X[tr], C[tr])
        pc[te] = m.predict_proba(X[te])[:, 1]
        m2 = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05).fit(X[tr], F[tr])
        pf[te] = m2.predict(X[te])
    stay = (F == S).mean()
    return {"rows": int(len(C)), "missions": int(len(np.unique(G))), "change_rate": float(C.mean()),
            "change_within_3s_auc_unseen_missions": round(float(roc_auc_score(C, pc)), 3),
            "state_in_3s_accuracy_unseen": round(float((pf == F).mean()), 3),
            "accuracy_if_never_changes": round(float(stay), 3)}


if __name__ == "__main__":
    used, X, S, F, C, G, names = build()
    for u in used:
        print(f"  {u[0]:44s} {u[1]:6d} rows, {u[2]:5d} change rows")
    print(json.dumps(baseline(X, S, F, C, G), indent=1))
