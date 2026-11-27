"""Teach the big model (the critic) your taste via a small one (Matt's suggestion, 2026-11-20).

The small taste layer learns from your A/B picks. It then labels thousands of changes you never
heard ("you would probably prefer this one"), and the critic itself is fine-tuned on its most
confident calls plus your real picks. Tested by moment-grouped folds: in every fold the teacher
never saw the held-out picks either, so the score is honest.

    python -m conductor.distill render [n_moments]        # main venv: render unrated candidates
    .venv-ml\\Scripts\\python -m conductor.distill feats      # mel + CLAP for every join window
    .venv-ml\\Scripts\\python -m conductor.distill train      # folds report; saves critic_student.pt if it wins
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "cache", "distill")
REV = os.path.join(HERE, "review")


# ── 1. render changes nobody has rated ──────────────────────────────────────────
def render(n_moments: int = 400, per: int = 6, seed: int = 7, out_dir: str | None = None):
    """out_dir: render somewhere else (distill_trees.py uses cache/distill2, which also keeps each route)."""
    global D
    if out_dir:
        D = out_dir
    from . import ab, routes
    from .library import Library
    lib = Library.build(__import__("conductor.playlists", fromlist=["train_music"]).train_music(), os.path.join(HERE, "cache"), progress=lambda m: None,
                        backfill=False, cached_only=True)
    by = {t.title: t for t in lib.tracks}
    rng = random.Random(seed)
    os.makedirs(os.path.join(D, "clips"), exist_ok=True)
    man_p = os.path.join(D, "manifest.json")
    man = json.load(open(man_p, encoding="utf-8")) if os.path.exists(man_p) else []
    state_of = {"explore": "calm", "tier1": "fight", "tier2": "horde", "tier3": "boss"}
    inv = {v: k for k, v in ab.STATE_NAME.items()}
    same_song = lambda r: all(s["b"] == r["a"] for s in routes.as_route(r)["steps"])
    moments = ab._synthetic(lib, rng, n_moments * 3)
    t0 = time.time()
    for j in moments:
        if len(man) >= n_moments:
            break
        if j["t_out"] < ab.PRE + 0.5:
            continue
        st_now = state_of[j["state_role"]]
        prev = inv.get(j["context"].split(" → ")[0])
        down = prev is not None and routes.STATE_RANK[state_of[prev]] > routes.STATE_RANK[st_now]
        try:
            nat = routes.natural_routes(lib, j, st_now, down, rng)
            alts = routes.candidates(lib, j, st_now, rng)
        except Exception:
            continue
        if prev is not None and state_of[prev] == st_now:
            nat, alts = [r for r in nat if same_song(r)], [r for r in alts if same_song(r)]
        pool, seen = [], set()
        for r in nat + alts[:6]:
            d = routes.describe(r)
            if d not in seen:
                seen.add(d); pool.append(r)
        if len(pool) < 3:
            continue
        rng.shuffle(pool)
        mid = f"d{len(man):04d}"
        clips = []
        for k, r in enumerate(pool[:per]):
            rel = f"clips/{mid}_{k}.mp3"
            try:
                seams = routes.render(by, r, os.path.join(D, rel))
            except Exception:
                continue
            clips.append({"clip": rel, "seams": seams or [ab.PRE], "style": routes.as_route(r)["steps"][0]["style"],
                          "same": same_song(r), "route": routes.as_route(r)})
        if len(clips) >= 3:
            man.append({"id": mid, "context": j["context"], "clips": clips})
            json.dump(man, open(man_p, "w", encoding="utf-8"))
        if len(man) % 20 == 0:
            print(f"{len(man)}/{n_moments} moments  {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(man)} moments, {sum(len(m['clips']) for m in man)} clips", flush=True)


# ── 2. what the critic hears, for every join window ─────────────────────────────
def _rated():
    """Your real head-to-heads: (winner clip, loser clip, moment id), clips relative to review/."""
    R = json.load(open(os.path.join(REV, "ab", "answers.json"), encoding="utf-8"))
    clips, comps = {}, []

    def reg(c, seams):
        if c and os.path.exists(os.path.join(REV, c)):
            clips[c] = seams or [6.0]
            return True
        return False
    for a in R:
        if a.get("kind") == "quad":
            q = a["quad"]
            if a.get("best") is None or a.get("worst") is None or \
                    not all([reg(c, s) for c, s in zip(q["clips"], q["seams"])]):
                continue
            b, w = a["best"], a["worst"]
            mid = [k for k in range(4) if k not in (b, w)]
            comps += [(q["clips"][b], q["clips"][k], q["id"]) for k in mid + [w]]
            comps += [(q["clips"][k], q["clips"][w], q["id"]) for k in mid]
            # your per-change thumbs: a 👍 option beats a 👎 one in the same set (if best/worst didn't say so)
            first = {int(k): v[0] for k, v in (a.get("hops") or {}).items() if v and v[0] in (1, -1)}
            have = {(b, k) for k in range(4) if k != b} | {(k, w) for k in mid}
            comps += [(q["clips"][i], q["clips"][j], q["id"]) for i, vi in first.items() for j, vj in first.items()
                      if vi == 1 and vj == -1 and (i, j) not in have]
        elif a.get("pair") and a.get("choice") in ("a", "b"):
            p = a["pair"]
            if reg(p.get("clip_a"), p.get("seams_a") or [p.get("seam", 6.0)]) and \
                    reg(p.get("clip_b"), p.get("seams_b") or [p.get("seam", 6.0)]):
                w, l = ("a", "b") if a["choice"] == "a" else ("b", "a")
                comps.append((p["clip_" + w], p["clip_" + l], p["a"]["a"] + str(round(p["a"]["t_out"]))))
    return clips, comps


def feats():
    from . import critic
    clips, comps = _rated()
    man = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
    names, paths, seams = [], [], []
    for c, ss in clips.items():
        names.append("R:" + c); paths.append(os.path.join(REV, c)); seams.append(ss)
    for m in man:
        for c in m["clips"]:
            names.append("D:" + c["clip"]); paths.append(os.path.join(D, c["clip"])); seams.append(c["seams"])
    flat = [(p, s) for p, ss in zip(paths, seams) for s in ss]
    owner = np.array([i for i, ss in enumerate(seams) for _ in ss])
    print(f"{len(names)} clips, {len(flat)} join windows", flush=True)
    clap = critic.Clap()
    mels, caps = [], []
    for k in range(0, len(flat), 256):
        w = critic._clip_waves(flat[k:k + 256])
        mels.append(np.concatenate([critic.mel_batch(w[i:i + 64]) for i in range(0, len(w), 64)]))
        caps.append(np.concatenate([critic.clap_feats(clap, w[i:i + 32]) for i in range(0, len(w), 32)]))
        print(f"  {min(k + 256, len(flat))}/{len(flat)}", flush=True)
    np.savez(os.path.join(D, "feats.npz"), mel=np.concatenate(mels), clap=np.concatenate(caps).astype(np.float16),
             owner=owner, names=np.array(names))
    json.dump({"comps": comps}, open(os.path.join(D, "rated.json"), "w", encoding="utf-8"))


# ── 3. teacher -> pseudo-labels -> student, honest folds ────────────────────────
def _vectors(F, P, owner, n):
    """Taste-layer clip vectors (same recipe as critic.clip_vectors) from per-window outputs."""
    out = np.zeros((n, F.shape[1] + 4), np.float32)
    for i in range(n):
        rows = np.concatenate([F[owner == i], P[owner == i, None], np.log(P[owner == i, None] + 1e-4)], 1)
        out[i] = np.concatenate([rows.mean(0), rows.min(0)[-2:]])
    return out


def _teacher(V, idx, comps_tr):
    from sklearn.linear_model import LogisticRegression
    X = np.array([V[idx[w]] - V[idx[l]] for w, l, _ in comps_tr])
    X = np.concatenate([X, -X]); y = np.r_[np.ones(len(comps_tr)), np.zeros(len(comps_tr))]
    sd = X.std(0) + 1e-6
    lr = LogisticRegression(C=0.01, max_iter=3000, fit_intercept=False).fit(X / sd, y)
    return lr.coef_[0] / sd


def train(epochs: int = 6, conf: float = 0.3, real_w: float = 3.0, folds: int = 5, final: bool = True,
          rounds: int = 3, noise: bool = True):
    """Noisy Student (Xie et al. 2019): the teacher labels clean audio; the student learns on noised
    audio (masked bands, shifted timing, dropout) with its whole ear free to change, so it can't
    just copy the teacher. Then the student becomes the teacher and it repeats."""
    import copy
    import torch
    from sklearn.model_selection import GroupKFold
    from . import critic
    z = np.load(os.path.join(D, "feats.npz"))
    mel = torch.tensor(z["mel"]).float()
    cap = torch.tensor(z["clap"]).float()
    owner, names = z["owner"], list(z["names"])
    n = len(names)
    idx = {nm: i for i, nm in enumerate(names)}
    comps = [("R:" + w, "R:" + l, g) for w, l, g in _rated()[1]]
    comps = [c for c in comps if c[0] in idx and c[1] in idx]
    man = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
    dmom = [[idx["D:" + c["clip"]] for c in m["clips"] if "D:" + c["clip"] in idx] for m in man]
    # what each clip is, for the live page
    info = {}
    for a in json.load(open(os.path.join(REV, "ab", "answers.json"), encoding="utf-8")):
        if a.get("kind") == "quad":
            q = a["quad"]
            for c, r in zip(q["clips"], q["options"]):
                st = r["steps"][0]
                info["R:" + c] = {"style": st["style"], "same": st["b"] == r["a"], "context": q["context"],
                                  "from": r["a"], "to": st["b"]}
    for m in man:
        for c in m["clips"]:
            info["D:" + c["clip"]] = {"style": c["style"], "same": c["same"], "context": m["context"]}
    LIVE_P = os.path.join(REV, "ab", "distill_live.json")
    live = {"started": time.time(), "stage": "loading", "folds": folds, "epochs": epochs, "fold": 0, "rounds": rounds, "round": 0,
            "comparisons": len(comps), "moments_unrated": len(dmom), "loss": [], "results": [],
            "decisions": [], "pseudo": [], "train": {}}

    def dump():
        live["updated"] = time.time()
        tmp = LIVE_P + ".tmp"
        json.dump(live, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, LIVE_P)
    dump()

    base = critic.make_model().cuda()
    base.load_state_dict(torch.load(os.path.join(critic.OUT, "critic.pt")))
    base.eval()

    def run(model, fn):
        outs = []
        with torch.no_grad():
            for k in range(0, len(mel), 256):
                outs.append(fn(model, mel[k:k + 256].cuda(), cap[k:k + 256].cuda()).float().cpu().numpy())
        return np.concatenate(outs)
    F0 = run(base, lambda m, a, b: m.features(a, b))
    P0 = 1 / (1 + np.exp(-run(base, lambda m, a, b: m(a, b))))
    V0 = _vectors(F0, P0, owner, n)
    win_of = [np.where(owner == i)[0] for i in range(n)]

    def acc(score, cs):
        return float(np.mean([score[idx[w]] > score[idx[l]] for w, l, _ in cs])) if cs else float("nan")

    def fit_student(comps_tr, tscore, fold=None, rnd=1):
        # keep the teacher's strongest `conf` share of opinions (by score gap), labelled 0.9 / 0.1;
        # near-ties are left out (pairwise noise on similar clips just teaches "they're equal")
        cand = [(ids[a], ids[b], tscore[ids[a]] - tscore[ids[b]])
                for ids in dmom for a in range(len(ids)) for b in range(a + 1, len(ids))]
        cut = np.quantile([abs(d) for _, _, d in cand], 1 - conf) if cand else 0
        pseudo = [(i, j, 0.9 if d > 0 else 0.1, 1.0) for i, j, d in cand if abs(d) >= cut]
        real = [(idx[w], idx[l], 1.0, real_w) for w, l, _ in comps_tr]
        pairs = pseudo + real
        model = copy.deepcopy(base)
        head = torch.nn.Linear(F0.shape[1], 1).cuda()
        torch.nn.init.normal_(head.weight, std=0.01); torch.nn.init.zeros_(head.bias)
        groups = [{"params": [p for p in model.cnn.parameters()], "lr": 5e-5 if noise else 0.0},
                  {"params": [p for n_, p in model.named_parameters() if not n_.startswith("cnn.")], "lr": 3e-4},
                  {"params": list(head.parameters()), "lr": 1e-3}]
        opt = torch.optim.AdamW(groups, weight_decay=1e-3)
        rs = np.random.RandomState(rnd * 100 + (fold or 0))

        def noisy(m, c):
            if not noise:
                return m, c
            m = torch.roll(m, int(rs.randint(-8, 9)), dims=2)
            for _ in range(2):                                   # SpecAugment: mask bands and moments
                f0, fw = rs.randint(0, 96 - 12), rs.randint(4, 13)
                m[:, f0:f0 + fw, :] = 0
                t0, tw = rs.randint(0, m.shape[2] - 24), rs.randint(6, 25)
                m[:, :, t0:t0 + tw] = 0
            m = m + 0.1 * torch.randn_like(m)
            c = torch.nn.functional.dropout(c, 0.15, training=True)
            return m, c

        def clip_scores(ids):
            wins = np.concatenate([win_of[i] for i in ids])
            seg = np.concatenate([[k] * len(win_of[i]) for k, i in enumerate(ids)])
            f = model.features(*noisy(mel[wins].cuda(), cap[wins].cuda()))
            s = head(f).squeeze(1)
            segt = torch.tensor(seg, device=s.device)
            tot = torch.zeros(len(ids), device=s.device).index_add_(0, segt, s)
            cnt = torch.zeros(len(ids), device=s.device).index_add_(0, segt, torch.ones_like(s))
            return tot / cnt
        model.train()
        for mod in model.cnn.modules():                         # BatchNorm keeps its music statistics
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()
        nb = (len(pairs) + 63) // 64
        live["train"] = {"fold": fold, "round": rnd, "pseudo": len(pseudo), "real": len(real), "batches": nb, "epoch": 0, "batch": 0}
        run_loss, run_acc = [], []
        for ep in range(epochs):
            order = rs.permutation(len(pairs))
            for bi, k in enumerate(range(0, len(order), 64)):
                b = [pairs[t] for t in order[k:k + 64]]
                ids = list({x for i, j, _, _ in b for x in (i, j)})
                pos = {c: m for m, c in enumerate(ids)}
                s = clip_scores(ids)
                d = torch.stack([s[pos[i]] - s[pos[j]] for i, j, _, _ in b])
                y = torch.tensor([p for _, _, p, _ in b], device=d.device)
                wt = torch.tensor([w for _, _, _, w in b], device=d.device)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(d, y, reduction="none") * wt).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                run_loss.append(float(loss))
                run_acc.append(float(((d > 0) == (y > 0.5)).float().mean()))
                if bi % 8 == 0 or bi == nb - 1:
                    live["train"].update(epoch=ep + 1, batch=bi + 1)
                    live["loss"].append({"fold": fold, "round": rnd, "step": (rnd - 1) * epochs + ep + bi / nb, "loss": float(np.mean(run_loss[-8:])),
                                         "fit": float(np.mean(run_acc[-8:]))})
                    dump()
        model.eval()
        with torch.no_grad():
            Fs = run(model, lambda m, a, b_: m.features(a, b_))
            ws = head(torch.tensor(Fs).cuda()).squeeze(1).cpu().numpy()
        sc = np.array([ws[win_of[i]].mean() for i in range(n)])
        return sc, model, head, len(pseudo)

    g = np.array([hash(c[2]) % (1 << 30) for c in comps])
    res = {"teacher": [], "student": [], "critic_alone": [], "pseudo_pairs": []}
    crit = np.array([P0[win_of[i]].min() for i in range(n)])
    sig = lambda x: float(1 / (1 + np.exp(-x)))
    pub = os.path.join(REV, "ab", "distill_clips")
    os.makedirs(pub, exist_ok=True)

    def url(nm):
        if nm.startswith("R:"):
            return nm[2:]
        import shutil
        dst = os.path.join(pub, os.path.basename(nm[2:]))
        if not os.path.exists(dst):
            shutil.copy(os.path.join(D, nm[2:]), dst)
        return "ab/distill_clips/" + os.path.basename(nm[2:])
    live["stage"] = "folds"
    for f, (tr, te) in enumerate(GroupKFold(folds).split(np.zeros(len(comps)), groups=g)):
        ctr, cte = [comps[i] for i in tr], [comps[i] for i in te]
        live["fold"] = f + 1
        w_t = _teacher(V0, idx, ctr)
        if f == 0:                                  # show what the teacher thinks of unheard changes
            tsc = V0 @ w_t
            spread = sorted(range(len(dmom)), key=lambda m: -np.ptp([tsc[i] for i in dmom[m]]) if dmom[m] else 0)
            for m in spread[:8]:
                ids = dmom[m]
                mu = np.mean([tsc[i] for i in ids])
                live["pseudo"].append({"context": man[m]["context"], "clips": [
                    {"url": url(names[i]), "style": info[names[i]]["style"], "same": info[names[i]]["same"],
                     "teacher": float(tsc[i] - mu), "critic": float(crit[i])} for i in ids]})
            dump()
        tsc = V0 @ w_t
        teach = tsc
        for rnd in range(1, rounds + 1):
            live["round"] = rnd
            s_student, _, _, npse = fit_student(ctr, teach, fold=f + 1, rnd=rnd)
            r_acc = acc(s_student, cte)
            live["results"].append({"fold": f + 1, "round": rnd, "teacher": acc(tsc, cte), "student": r_acc,
                                    "critic": acc(crit, cte), "pseudo": npse})
            print(f"fold {f + 1} round {rnd}: student {r_acc:.0%}", flush=True)
            dump()
            teach = s_student                                  # the student becomes the teacher
        for w, l, grp in cte:
            i, j = idx[w], idx[l]
            live["decisions"].append({"fold": f + 1, "group": str(grp), "context": info.get(w, {}).get("context", ""),
                                      "you": url(w), "other": url(l),
                                      "you_style": info.get(w, {}).get("style", "?"), "other_style": info.get(l, {}).get("style", "?"),
                                      "you_same": info.get(w, {}).get("same"), "other_same": info.get(l, {}).get("same"),
                                      "critic": [float(crit[i]), float(crit[j])],
                                      "teacher": sig(tsc[i] - tsc[j]), "student": sig(s_student[i] - s_student[j])})
        if f == 0:
            for ex in live["pseudo"]:
                for c in ex["clips"]:
                    nm = "D:clips/" + os.path.basename(c["url"])
                    c["student"] = float(s_student[idx[nm]])
                mu = np.mean([c["student"] for c in ex["clips"]])
                for c in ex["clips"]:
                    c["student"] -= mu
        res["teacher"].append(acc(V0 @ w_t, cte))
        res["student"].append(acc(s_student, cte))
        res["critic_alone"].append(acc(crit, cte))
        res["pseudo_pairs"].append(npse)
        dump()
        print(f"fold {f + 1}: teacher {res['teacher'][-1]:.0%}  student {res['student'][-1]:.0%}  "
              f"critic {res['critic_alone'][-1]:.0%}  ({npse} pseudo pairs)", flush=True)
    summary = {k: (f"{np.mean(v):.1%}" if k != "pseudo_pairs" else int(np.mean(v))) for k, v in res.items()}
    summary["comparisons"] = len(comps)
    summary["moments_unrated"] = len(dmom)
    print(json.dumps(summary), flush=True)
    json.dump(summary, open(os.path.join(D, "result.json"), "w"))
    live["summary"] = summary
    live["stage"] = "final" if final and np.mean(res["student"]) > np.mean(res["teacher"]) else "done"
    dump()
    if final and np.mean(res["student"]) > np.mean(res["teacher"]):
        teach = V0 @ _teacher(V0, idx, comps)
        for rnd in range(1, rounds + 1):
            teach, model, head, _ = fit_student(comps, teach, rnd=rnd)
        torch.save({"model": model.state_dict(), "head": head.state_dict()}, os.path.join(critic.OUT, "critic_student.pt"))
        print("student beat the teacher: saved critic_student.pt", flush=True)
        live["stage"] = "done"; live["saved"] = True
        dump()
    return summary


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "render":
        render(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
    elif cmd == "render2":                     # practice joins that keep their routes (distill_trees.py)
        render(int(sys.argv[2]) if len(sys.argv) > 2 else 400, seed=11, out_dir=os.path.join(HERE, "cache", "distill2"))
    elif cmd == "feats":
        feats()
    elif cmd == "train":
        train()
