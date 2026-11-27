"""Transition critic — learns what a natural musical join sounds like (ML venv, GPU).

Step 1, self-supervised (the music teaches the model): every song already contains perfect
transitions — its own bar lines and section changes, played as written ("natural", label 1).
We splice thousands of artificial joins the way Conductor can (phrase-aligned section moves,
any-bar jumps, off-beat cuts, other songs, tempo clashes, loudness dips: label 0) and train a
model to tell them apart. Its P(natural) is a general "does this join sound like music" score.

Step 2, calibration: the user's ratings (Transition Bench rounds, showcase pins) fit a small
head on top, so scores line up with what the user actually hears.

    .venv-ml\\Scripts\\python -m conductor.critic build     # make the dataset (cache/critic)
    .venv-ml\\Scripts\\python -m conductor.critic train     # train + report held-out accuracy
    .venv-ml\\Scripts\\python -m conductor.critic calibrate <clips.json>...  # vs. user ratings
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "cache", "critic")
SR = 22050
HALF = 3.0                       # seconds each side of the join
N_MELS = 96
HOP = 441                        # 50 frames / s
FRAMES = int(2 * HALF * SR / HOP)
CLAP_NAME = "laion/larger_clap_music"
MODEL_PATH = None                # generations.py points scoring at a candidate critic (None = live critic.pt)
FOLDERS = __import__("conductor.playlists", fromlist=["train_folders"]).train_folders()   # training music first


# ── splicing like the player does ─────────────────────────────────────────────
def splice(a: np.ndarray, ta: float, b: np.ndarray, tb: float, fade: float = 0.25, curve: str = "power",
           dip: float = 0.0) -> np.ndarray:
    """2*HALF seconds: a up to ta, then b from tb, crossfaded over `fade` ending at the join."""
    n = int(HALF * SR)
    f = max(1, int(fade * SR))
    pre = a[int(ta * SR) - n: int(ta * SR)]
    post = b[int(tb * SR): int(tb * SR) + n]
    if len(pre) < n or len(post) < n:
        return None
    out = np.concatenate([pre, post]).astype(np.float32)
    # crossfade: b's lead-in under a's last `f` samples
    lead = b[int(tb * SR) - f: int(tb * SR)]
    if len(lead) == f:
        x = np.linspace(0, 1, f, dtype=np.float32)
        g_in, g_out = (np.sin(x * np.pi / 2), np.cos(x * np.pi / 2)) if curve == "power" else (x, 1 - x)
        if dip:
            g_in, g_out = g_in * (1 - dip), g_out * (1 - dip)
        out[n - f:n] = pre[-f:] * g_out + lead * g_in
    return out


def natural(a: np.ndarray, t: float) -> np.ndarray | None:
    n = int(HALF * SR)
    w = a[int(t * SR) - n: int(t * SR) + n]
    return w.astype(np.float32) if len(w) == 2 * n else None


def load_song(path: str):
    import librosa
    from .analysis import cache_key
    c = os.path.join(HERE, "cache", cache_key(path) + ".json")
    if not os.path.exists(c):
        return None
    d = json.load(open(c, encoding="utf-8"))
    y, _ = librosa.load(path, sr=SR, mono=True)
    return {"y": y, "d": d, "title": d["title"], "tempo": d.get("tempo", 120.0),
            "bars": d.get("downbeats") or d.get("beats", [])[::4], "beats": d.get("beats", []),
            "sections": [s["start"] for s in d["segments"]]}


def examples_for(song, others, rng, per_song=160):
    """(audio, label, kind) — kind tells which splice type made it."""
    y, bars, beats, secs = song["y"], song["bars"], song["beats"], song["sections"]
    dur = len(y) / SR
    ok = [b for b in bars if HALF + 0.5 < b < dur - HALF - 0.5]
    if len(ok) < 12:
        return []
    out = []
    period = 60.0 / song["tempo"] if song["tempo"] else 0.5
    for _ in range(per_song):
        r = rng.random()
        ta = rng.choice(ok)
        if r < 0.40:            # natural: a real bar line or section change, as written
            t = rng.choice([s for s in secs if HALF + 0.5 < s < dur - HALF - 0.5] or ok) if rng.random() < 0.4 else ta
            w = natural(y, t); kind, lab = "natural", 1
        elif r < 0.55:          # phrase-ish section move inside the song (what Conductor does)
            tb = rng.choice([s for s in secs if abs(s - ta) > 8 and HALF + 0.5 < s < dur - HALF - 0.5] or ok)
            w = splice(y, ta, y, tb, fade=period, curve="power"); kind, lab = "section_move", 0
        elif r < 0.68:          # any bar to any bar inside the song
            tb = rng.choice([b for b in ok if abs(b - ta) > 8] or ok)
            w = splice(y, ta, y, tb, fade=0.15, curve="linear"); kind, lab = "bar_jump", 0
        elif r < 0.78:          # off the beat
            tb = rng.choice(ok) + rng.uniform(0.12, 0.45) * period
            w = splice(y, ta + rng.uniform(-0.3, 0.3) * period, y, tb, fade=0.1); kind, lab = "off_beat", 0
        elif r < 0.92:          # another song, on bar lines
            o = rng.choice(others)
            ob = [b for b in o["bars"] if HALF + 0.5 < b < len(o["y"]) / SR - HALF - 0.5]
            if not ob:
                continue
            tb = rng.choice(ob)
            same_tempo = abs(np.log2(o["tempo"] / song["tempo"])) < 0.05 if o["tempo"] and song["tempo"] else False
            w = splice(y, ta, o["y"], tb, fade=rng.choice([0.3, 1.0, 2.0])); kind, lab = ("cross_song" if same_tempo else "cross_tempo"), 0
        else:                   # a loudness dip in an otherwise natural place
            tb = ta + rng.choice([-1, 1]) * 4 * period * rng.choice([4, 8])
            w = splice(y, ta, y, tb, fade=1.5, dip=0.5); kind, lab = "dip", 0
        if w is not None and np.abs(w).max() > 1e-3:
            out.append((w, lab, kind))
    return out


def mel_batch(waves: np.ndarray, device="cuda"):
    import torch, torchaudio
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=2048, hop_length=HOP, n_mels=N_MELS).to(device)
    x = torch.tensor(waves, device=device)
    m = torch.log(mel(x) + 1e-5)
    m = (m - m.mean(dim=(1, 2), keepdim=True)) / (m.std(dim=(1, 2), keepdim=True) + 1e-5)
    return m[:, :, :FRAMES].half().cpu().numpy()


class Clap:
    def __init__(self):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.m = ClapModel.from_pretrained(CLAP_NAME).cuda().eval()
        self.p = ClapProcessor.from_pretrained(CLAP_NAME)

    def embed(self, waves: list[np.ndarray]) -> np.ndarray:
        import torchaudio
        torch = self.torch
        x = torch.tensor(np.stack(waves)).cuda()
        x48 = torchaudio.functional.resample(x, SR, 48000).cpu().numpy()
        inp = self.p(audio=list(x48), sampling_rate=48000, return_tensors="pt")
        inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            e = self.m.get_audio_features(**inp)
        e = getattr(e, "pooler_output", e)
        e = torch.nn.functional.normalize(e, dim=-1)
        return e.cpu().numpy().astype(np.float16)


def clap_feats(clap: Clap, waves: np.ndarray) -> np.ndarray:
    """before / across / after embeddings (2 s each) and their similarities."""
    n = int(HALF * SR)
    w2 = int(2.0 * SR)
    before = [w[n - w2:n] for w in waves]
    after = [w[n:n + w2] for w in waves]
    across = [w[n - w2 // 2:n + w2 // 2] for w in waves]
    eb, ea, ex = clap.embed(before), clap.embed(after), clap.embed(across)
    sims = np.stack([(eb * ea).sum(1), (eb * ex).sum(1), (ea * ex).sum(1)], 1).astype(np.float16)
    return np.concatenate([eb, ea, ex, sims], 1)


def build(per_song=160, seed=0):
    from .library import scan
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(seed)
    paths = [p for f in FOLDERS for p in scan(f)]
    songs = []
    t0 = time.time()
    for p in paths:
        s = load_song(p)
        if s:
            songs.append(s)
    print(f"{len(songs)} songs loaded in {time.time() - t0:.0f}s", flush=True)
    clap = Clap()
    titles = [s["title"] for s in songs]
    for i, s in enumerate(songs):
        f = os.path.join(OUT, f"ex_{i:03d}.npz")
        if os.path.exists(f):
            continue
        others = [o for o in songs if o is not s]
        ex = examples_for(s, others, rng, per_song)
        if not ex:
            continue
        waves = np.stack([w for w, _, _ in ex])
        mels = np.concatenate([mel_batch(waves[k:k + 64]) for k in range(0, len(waves), 64)])
        cf = np.concatenate([clap_feats(clap, waves[k:k + 32]) for k in range(0, len(waves), 32)])
        np.savez_compressed(f, mel=mels, clap=cf, y=np.array([l for _, l, _ in ex], np.int8),
                            kind=np.array([k for _, _, k in ex]), song=i)
        print(f"[{i + 1}/{len(songs)}] {s['title'][:40]}: {len(ex)} examples", flush=True)
    json.dump(titles, open(os.path.join(OUT, "songs.json"), "w", encoding="utf-8"))


def build_game():
    """Darktide's own transitions (tools/wwise_render.py: the game's Wwise rules on its real score)
    as natural examples, each with a rule-less splice of the same two segments as its counterpart.
    Saved as ex_game_<g>.npz in 30 groups so the held-out test sees game examples too."""
    import librosa
    gdir = os.path.join(HERE, "cache", "game_joins")
    man = json.load(open(os.path.join(gdir, "manifest.json"), encoding="utf-8"))
    clap = Clap()
    n = int(HALF * SR)
    groups = {}
    for m in man:
        for lab, name in ((1, "good"), (0, "bad")):
            p = os.path.join(gdir, f"{name}_{m['k']:04d}.wav")
            if not os.path.exists(p):
                continue
            y, _ = librosa.load(p, sr=SR, mono=True)
            c = int(m.get("seam", 4.0) * SR)
            w = y[max(0, c - n):c + n]
            w = np.pad(w, (0, max(0, 2 * n - len(w))))[:2 * n].astype(np.float32)
            groups.setdefault(m["k"] % 30, []).append((w, lab, "game" if lab else "game_wrong"))
    for g, ex in groups.items():
        waves = np.stack([w for w, _, _ in ex])
        mels = np.concatenate([mel_batch(waves[k:k + 64]) for k in range(0, len(waves), 64)])
        cf = np.concatenate([clap_feats(clap, waves[k:k + 32]) for k in range(0, len(waves), 32)])
        np.savez_compressed(os.path.join(OUT, f"ex_game_{g:02d}.npz"), mel=mels, clap=cf,
                            y=np.array([l for _, l, _ in ex], np.int8), kind=np.array([k for _, _, k in ex]),
                            song=9000 + g)
        print(f"game group {g}: {len(ex)} examples", flush=True)


RANK_PATH = os.path.join(OUT, "rank.npz")


def build_rank(per_song: int = 16, seed: int = 5):
    """Graded ladders (user 2026-11-20): not just right/wrong but degrees of right.
    Game ladders (tools/wwise_render.py rank_<k>_<sev>.wav): 0 the game's own rule, 1 a beat off,
    2 the wrong fade, 3 half a bar off, 4 new part entered mid-phrase, 5 random splice.
    Song ladders (your playlists): 0 the song's own section change, 1 skip a bar (on the grid),
    2 skip a beat, 3 half a bar off, 4 off the beat, 5 another song."""
    import librosa
    rng = random.Random(seed)
    n = int(HALF * SR)
    W, grp, sev, src = [], [], [], []
    gdir = os.path.join(HERE, "cache", "game_joins")
    man = json.load(open(os.path.join(gdir, "manifest.json"), encoding="utf-8")) if os.path.exists(os.path.join(gdir, "manifest.json")) else []
    for m in man:
        ok = []
        for v in range(6):
            p = os.path.join(gdir, f"rank_{m['k']:04d}_{v}.wav")
            if not os.path.exists(p):
                break
            y, _ = librosa.load(p, sr=SR, mono=True)
            c = int(m.get("seam", 4.0) * SR)
            w = y[max(0, c - n):c + n]
            ok.append(np.pad(w, (0, max(0, 2 * n - len(w))))[:2 * n].astype(np.float32))
        if len(ok) == 6:
            for v, w in enumerate(ok):
                W.append(w); grp.append(100000 + m["k"]); sev.append(v); src.append(0)
    print(f"game ladders: {len(W) // 6}", flush=True)
    from .library import scan
    paths = [p for f in FOLDERS[:2] for p in scan(f)]
    songs = [s_ for s_ in (load_song(p) for p in paths) if s_]
    for si, song in enumerate(songs):
        y, secs, bars = song["y"], song["sections"], song["bars"]
        dur = len(y) / SR
        period = 60.0 / song["tempo"] if song["tempo"] else 0.5
        cand = [t for t in secs if HALF + 4 * 4 * period < t < dur - HALF - 4 * 4 * period]
        rng.shuffle(cand)
        for t in cand[:per_song]:
            others = [o for o in songs if o is not song]
            o = rng.choice(others)
            ob = [b for b in o["bars"] if HALF + 0.5 < b < len(o["y"]) / SR - HALF - 0.5]
            if not ob:
                continue
            bar = 4 * period
            ladder = [natural(y, t),
                      splice(y, t, y, t + bar, fade=0.05),
                      splice(y, t, y, t + period, fade=0.05),
                      splice(y, t, y, t + bar / 2, fade=0.05),
                      splice(y, t + rng.uniform(0.2, 0.45) * period, y, t + bar * 2 + rng.uniform(0.15, 0.4) * period, fade=0.05),
                      splice(y, t, o["y"], rng.choice(ob), fade=0.3)]
            if any(w is None for w in ladder):
                continue
            lid = 200000 + len(grp)                  # one id per ladder
            for v, w in enumerate(ladder):
                W.append(w); grp.append(lid); sev.append(v); src.append(1)
        print(f"[{si + 1}/{len(songs)}] {song['title'][:40]}", flush=True)
    waves = np.stack(W)
    clap = Clap()
    mels = np.concatenate([mel_batch(waves[k:k + 64]) for k in range(0, len(waves), 64)])
    cf = np.concatenate([clap_feats(clap, waves[k:k + 32]) for k in range(0, len(waves), 32)])
    np.savez_compressed(RANK_PATH, mel=mels, clap=cf, grp=np.array(grp), sev=np.array(sev, np.int8), src=np.array(src, np.int8))
    print(f"saved {len(W)} ladder clips ({len(set(grp))} ladders)", flush=True)


# ── model ─────────────────────────────────────────────────────────────────────
def make_model():
    import torch
    import torch.nn as nn

    class Critic(nn.Module):
        """Spectrogram CNN (rhythm/timbre continuity) + CLAP feature MLP (musical meaning)."""
        def __init__(self, clap_dim=512 * 3 + 3):
            super().__init__()
            def blk(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                                     nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(), nn.MaxPool2d(2))
            self.cnn = nn.Sequential(blk(1, 24), blk(24, 48), blk(48, 96), blk(96, 128))
            self.cnn_head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 4)), nn.Flatten(), nn.Linear(128 * 4, 128), nn.GELU())
            self.clap = nn.Sequential(nn.Linear(clap_dim, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 64), nn.GELU())
            self.out = nn.Sequential(nn.Dropout(0.2), nn.Linear(128 + 64, 1))

        def features(self, mel, clap):
            a = self.cnn_head(self.cnn(mel.unsqueeze(1)))
            b = self.clap(clap)
            return torch.cat([a, b], 1)

        def forward(self, mel, clap):
            return self.out(self.features(mel, clap)).squeeze(1)
    return Critic()


def _load_all():
    files = sorted(glob.glob(os.path.join(OUT, "ex_*.npz")))
    mel, clap, y, kind, song = [], [], [], [], []
    for f in files:
        z = np.load(f)
        mel.append(z["mel"]); clap.append(z["clap"]); y.append(z["y"]); kind.append(z["kind"])
        song.append(np.full(len(z["y"]), int(z["song"])))
    return (np.concatenate(mel), np.concatenate(clap), np.concatenate(y), np.concatenate(kind),
            np.concatenate(song))


def train(epochs=14, seed=0, out=None, init=None, lr=2e-3):
    """out: save ONLY there (a candidate; the live critic.pt is untouched). init: start from these weights."""
    import torch
    torch.manual_seed(seed)
    mel, clap, y, kind, song = _load_all()
    songs = np.unique(song)
    rs = np.random.RandomState(seed)
    test_songs = set(rs.choice(songs, size=max(1, len(songs) // 6), replace=False))
    te = np.array([s in test_songs for s in song])
    print(f"{len(y)} examples ({y.mean() * 100:.0f}% natural); test = {te.sum()} from {len(test_songs)} unseen songs")
    from . import lab
    kinds_count = {str(k): int((kind == k).sum()) for k in np.unique(kind)}
    lab.reset("train", examples=int(len(y)), natural=float(y.mean()), test_songs=len(test_songs),
              kinds=kinds_count, epochs=epochs, curve=[])
    model = make_model().cuda()
    if init:
        model.load_state_dict(torch.load(init))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (int((~te).sum()) // 128 + 1))
    lossf = torch.nn.BCEWithLogitsLoss()
    Xm, Xc, Y = torch.tensor(mel).float(), torch.tensor(clap).float(), torch.tensor(y).float()
    tr_idx, te_idx = np.where(~te)[0], np.where(te)[0]
    rank = None
    if os.path.exists(RANK_PATH):                      # graded ladders: learn degrees of right
        z = np.load(RANK_PATH)
        RM, RC, RG, RS = torch.tensor(z["mel"]).float(), torch.tensor(z["clap"]).float(), z["grp"], z["sev"]
        ladders = {}
        for i, g_ in enumerate(RG):
            ladders.setdefault(int(g_), []).append(i)
        lad = [v for v in ladders.values() if len(v) == 6]
        rs.shuffle(lad)
        cut_ = max(1, len(lad) // 6)
        rank = (RM, RC, RS, lad[cut_:], lad[:cut_])
        print(f"ranking: {len(lad)} ladders ({len(lad) - cut_} train / {cut_} held out)", flush=True)

    def rank_loss(model, ladders_):
        idx = [i for l_ in ladders_ for i in l_]
        m_ = rank[0][idx].cuda(); c_ = rank[1][idx].cuda()
        s_ = model(m_, c_).view(len(ladders_), 6)
        sv = torch.tensor(np.array([rank[2][l_] for l_ in ladders_]), device=s_.device).float()
        d = s_.unsqueeze(2) - s_.unsqueeze(1)                         # score_i - score_j
        better = (sv.unsqueeze(2) < sv.unsqueeze(1)).float()          # i is better than j
        return (torch.nn.functional.softplus(-d) * better).sum() / better.sum()

    def rank_acc(model, ladders_):
        if not ladders_:
            return float("nan")
        model.eval(); ok = tot = 0
        with torch.no_grad():
            for k in range(0, len(ladders_), 64):
                L_ = ladders_[k:k + 64]
                idx = [i for l_ in L_ for i in l_]
                s_ = model(rank[0][idx].cuda(), rank[1][idx].cuda()).view(len(L_), 6).cpu().numpy()
                sv = np.array([rank[2][l_] for l_ in L_])
                for a_ in range(6):
                    for b_ in range(6):
                        m_ = sv[:, a_] < sv[:, b_]
                        ok += int(((s_[:, a_] > s_[:, b_]) & m_).sum()); tot += int(m_.sum())
        model.train()
        return ok / max(1, tot)
    for ep in range(epochs):
        model.train()
        rs.shuffle(tr_idx)
        tot = 0.0
        for k in range(0, len(tr_idx), 128):
            b = tr_idx[k:k + 128]
            m = Xm[b].cuda()
            if rs.rand() < 0.5:                        # small time shift augmentation
                m = torch.roll(m, int(rs.randint(-6, 7)), dims=2)
            loss = lossf(model(m, Xc[b].cuda()), Y[b].cuda())
            if rank is not None and rank[3]:
                pick = [rank[3][i] for i in rs.randint(0, len(rank[3]), 16)]
                loss = loss + 0.5 * rank_loss(model, pick)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(b)
        acc, by = evaluate(model, Xm, Xc, Y, te_idx, kind)
        if rank is not None:
            print(f"   ladder order right on unseen ladders: {rank_acc(model, rank[4]) * 100:.1f}%", flush=True)
        print(f"epoch {ep + 1:2d}  loss {tot / len(tr_idx):.3f}  held-out acc {acc * 100:.1f}%", flush=True)
        lab.append("train", "curve", {"epoch": ep + 1, "loss": round(tot / len(tr_idx), 4), "acc": round(acc, 4),
                                      "by_kind": {str(k): round(v, 3) for k, v in by.items()}})
    print("held-out accuracy by kind:", {k: f"{v * 100:.0f}%" for k, v in by.items()})
    if out:
        torch.save(model.state_dict(), out)
        lab.update("train", done=True)
        return {"held_out": round(float(acc), 4), "ladders": round(float(rank_acc(model, rank[4])), 4) if rank is not None else None}
    torch.save(model.state_dict(), os.path.join(OUT, "critic.pt"))
    os.makedirs(os.path.join(OUT, "versions"), exist_ok=True)       # never lose a critic again
    torch.save(model.state_dict(), os.path.join(OUT, "versions", time.strftime("critic_%Y%m%d_%H%M.pt")))
    lab.update("train", done=True)


def evaluate(model, Xm, Xc, Y, idx, kind):
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(idx), 256):
            b = idx[k:k + 256]
            preds.append(torch.sigmoid(model(Xm[b].cuda(), Xc[b].cuda())).cpu().numpy())
    p = np.concatenate(preds)
    y = Y[idx].numpy()
    acc = float(((p > 0.5) == (y > 0.5)).mean())
    by = {}
    for k in np.unique(kind[idx]):
        sel = kind[idx] == k
        by[k] = float(((p[sel] > 0.5) == (y[sel] > 0.5)).mean())
    return acc, by


# ── scoring real rendered transitions (user-rated clips) ─────────────────────
def _clip_waves(paths_at):
    import librosa
    n = int(HALF * SR)
    waves, cache = [], {}
    for p, at in paths_at:
        if p not in cache:
            cache[p] = librosa.load(p, sr=SR, mono=True)[0]
        y = cache[p]
        w = y[max(0, int(at * SR) - n): int(at * SR) + n]
        waves.append(np.pad(w, (0, max(0, 2 * n - len(w))))[:2 * n].astype(np.float32))
    return np.stack(waves)


IN_CACHE = os.path.join(OUT, "clipinputs.pkl")


def _inputs(paths_at):
    """Mel + CLAP inputs for each (file, seam). They depend only on the audio, never on the critic,
    so they're cached: retraining the critic doesn't re-decode ~1000 clips (CPU-bound, ~20 min)."""
    import pickle
    memo = {}
    if os.path.exists(IN_CACHE):
        try:
            memo = pickle.load(open(IN_CACHE, "rb"))
        except Exception:
            memo = {}
    key = lambda p, at: (os.path.abspath(p), round(float(at), 3), os.path.getmtime(p))
    todo = [(p, at) for p, at in paths_at if key(p, at) not in memo]
    if todo:
        clap = Clap()
        waves = _clip_waves(todo)
        mel = np.concatenate([mel_batch(waves[k:k + 64]) for k in range(0, len(waves), 64)])
        cf = np.concatenate([clap_feats(clap, waves[k:k + 32]) for k in range(0, len(waves), 32)])
        for (p, at), m, c in zip(todo, mel, cf):
            memo[key(p, at)] = (m, c)
        if len(todo) > 8:                      # don't bloat the cache with one-off scoring jobs
            pickle.dump(memo, open(IN_CACHE, "wb"))
    got = [memo[key(p, at)] for p, at in paths_at]
    return np.stack([g[0] for g in got]), np.stack([g[1] for g in got])


def score_clips(paths_at: list[tuple[str, float]], features: bool = False):
    """P(natural) for audio files with the join at `at` seconds (and, with `features`, the
    network's 192 learned features for each join)."""
    import torch
    model = make_model().cuda()
    model.load_state_dict(torch.load(MODEL_PATH or os.path.join(OUT, "critic.pt")))
    model.eval()
    M, C = _inputs(paths_at)
    mel, cf = torch.tensor(M).float().cuda(), torch.tensor(C).float().cuda()
    with torch.no_grad():
        P = torch.sigmoid(model(mel, cf)).cpu().numpy()
        if not features:
            return P
        return P, model.features(mel, cf).cpu().numpy()


VEC_CACHE = os.path.join(OUT, "clipvec.pkl")


def clip_vectors(jobs, cache: bool = False) -> tuple[list, list]:
    """jobs = [[path, [seam, ...]], ...] -> per clip (taste vector, min P(natural) over seams).
    With `cache`, rated clips (which never change) are remembered until critic.pt is retrained."""
    import pickle
    stamp = os.path.getmtime(os.path.join(OUT, "critic.pt"))
    memo = {}
    if cache and os.path.exists(VEC_CACHE):
        try:
            got = pickle.load(open(VEC_CACHE, "rb"))
            memo = got["v"] if got.get("stamp") == stamp else {}
        except Exception:
            memo = {}
    key = lambda p, ss: (os.path.abspath(p), tuple(round(x, 3) for x in (ss or [6.0])), os.path.getmtime(p))
    todo = [(p, ss) for p, ss in jobs if key(p, ss) not in memo]
    if todo:
        flat = [(p, s) for p, ss in todo for s in (ss or [6.0])]
        P, F = score_clips(flat, features=True)
        k = 0
        for p, ss in todo:
            m = len(ss or [6.0])
            rows = [np.concatenate([F[i], [P[i], np.log(P[i] + 1e-4)]]) for i in range(k, k + m)]
            memo[key(p, ss)] = (np.concatenate([np.mean(rows, 0), np.min(rows, 0)[-2:]]), float(P[k:k + m].min()))
            k += m
    if cache:
        pickle.dump({"stamp": stamp, "v": memo}, open(VEC_CACHE, "wb"))
    out = [memo[key(p, ss)] for p, ss in jobs]
    return [v for v, _ in out], [m for _, m in out]


TASTE = os.path.join(OUT, "taste.json")


def model_styles() -> dict:
    """The model's view of each joining style: the current taste layer scores every practice clip
    (cache/distill, 400 moments x 6 changes); within each moment, rank the styles (1 = favourite).
    Written to cache/live/model_styles.json for policy.py."""
    import torch
    from collections import defaultdict
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dd = os.path.join(root, "cache", "distill")
    if not (os.path.exists(TASTE) and os.path.exists(os.path.join(dd, "feats.npz"))):
        return {}
    z = np.load(os.path.join(dd, "feats.npz"))
    names, owner = list(z["names"]), z["owner"]
    model = make_model().cuda()
    model.load_state_dict(torch.load(os.path.join(OUT, "critic.pt")))
    model.eval()
    F, P = [], []
    with torch.no_grad():
        for k in range(0, len(owner), 256):
            m = torch.tensor(z["mel"][k:k + 256]).float().cuda()
            c = torch.tensor(z["clap"][k:k + 256]).float().cuda()
            F.append(model.features(m, c).cpu().numpy()); P.append(torch.sigmoid(model(m, c)).cpu().numpy())
    F, P = np.concatenate(F), np.concatenate(P)
    w = np.array(json.load(open(TASTE))["w"])
    score = {}
    for i, nm in enumerate(names):
        if not nm.startswith("D:"):
            continue
        rows = np.concatenate([F[owner == i], P[owner == i, None], np.log(P[owner == i, None] + 1e-4)], 1)
        score[nm[2:]] = float(np.concatenate([rows.mean(0), rows.min(0)[-2:]]) @ w)
    man = json.load(open(os.path.join(dd, "manifest.json"), encoding="utf-8"))
    acc = defaultdict(list)
    for mo in man:
        cl = [c for c in mo["clips"] if c["clip"] in score]
        if len(cl) < 3:
            continue
        order = sorted(cl, key=lambda c: score[c["clip"]])
        for r, c in enumerate(order):
            kind = f"{'same' if c['same'] else 'other'}|{'lead' if len(c['seams']) > 1 else 'direct'}|{c['style']}"
            acc[kind].append(r / (len(order) - 1))
    out = {k: {"rank": float(np.mean(v)), "n": len(v)} for k, v in acc.items()}
    os.makedirs(os.path.join(root, "cache", "live"), exist_ok=True)
    json.dump(out, open(os.path.join(root, "cache", "live", "model_styles.json"), "w"), indent=1)
    return out


def train_taste_head() -> dict:
    """Your picks teach a linear taste layer on the critic's features (pairwise: winner - loser).
    2026-11-20: 64% agreement on unseen moments vs 58% for the critic alone (528 comparisons)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    clips, comps, rev = comparisons()
    names = list(clips)
    vecs, _ = clip_vectors([[os.path.join(rev, c), clips[c]] for c in names], cache=True)
    feat = dict(zip(names, vecs))
    X = np.array([feat[w] - feat[l] for w, l, _ in comps])
    import hashlib                       # stable folds: every critic version is scored on the same split
    g = np.array([int(hashlib.md5(str(c[2]).encode()).hexdigest()[:8], 16) for c in comps])
    X = np.concatenate([X, -X]); y = np.r_[np.ones(len(comps)), np.zeros(len(comps))]; g = np.r_[g, g]
    sd = X.std(0) + 1e-6
    Xs = X / sd
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xs, y, g):
        pred[te] = LogisticRegression(C=0.01, max_iter=3000).fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]
    acc = float((pred[:len(comps)] > 0.5).mean())
    base = float(np.mean([feat[w][-2] > feat[l][-2] for w, l, _ in comps]))
    lr = LogisticRegression(C=0.01, max_iter=3000, fit_intercept=False).fit(Xs, y)
    w = (lr.coef_[0] / sd).tolist()
    json.dump({"w": w, "unseen": acc, "critic_alone": base, "comparisons": len(comps)}, open(TASTE, "w"))
    return {"comparisons": len(comps), "taste_head_unseen": f"{acc:.0%}", "critic_alone": f"{base:.0%}"}


def comparisons():
    """Your A/B answers as (winner clip, loser clip, group) + every clip's seams."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rev = os.path.join(root, "review")
    R = json.load(open(os.path.join(rev, "ab", "answers.json"), encoding="utf-8"))
    clips, comps = {}, []
    def reg(c, seams):
        if c and os.path.exists(os.path.join(rev, c)):
            clips[c] = seams or [6.0]
            return True
        return False
    for a in R:
        if a.get("kind") == "quad":
            q = a["quad"]
            if q.get("repeat_of"):              # blind repeat: same clips as an earlier answer, consistency only
                continue
            from .ab import stale
            if stale(q["id"]):                       # judged on a buggy blend version (2026-11-25)
                continue
            if a.get("best") is None or a["best"] == a.get("worst") or not all([reg(c, s) for c, s in zip(q["clips"], q["seams"])]):
                continue
            # 2026-11-23: silent lead-ins / timing-only twins / pairs you answered both ways are dropped;
            # pairs a blind repeat confirmed count twice
            from .ab import noisy, training_pairs
            nz = noisy()
            comps += [(q["clips"][x], q["clips"][y], q["id"]) for x, y in training_pairs(a, nz)]
        elif a.get("pair") and a.get("choice") in ("a", "b"):
            p = a["pair"]
            if reg(p.get("clip_a"), p.get("seams_a") or [p.get("seam", 6.0)]) and                reg(p.get("clip_b"), p.get("seams_b") or [p.get("seam", 6.0)]):
                w, l = ("a", "b") if a["choice"] == "a" else ("b", "a")
                comps.append((p["clip_" + w], p["clip_" + l], p["a"]["a"] + str(round(p["a"]["t_out"]))))
    return clips, comps, rev


def calibrate(manifest: str):
    """How well does the critic agree with the user? Then fit a user-taste head on its features
    (cross-validated by rating round, so every score is for ratings the head never saw)."""
    import librosa, torch
    from sklearn.linear_model import LogisticRegression
    items = json.load(open(manifest, encoding="utf-8"))
    model = make_model().cuda()
    model.load_state_dict(torch.load(os.path.join(OUT, "critic.pt")))
    model.eval()
    clap = Clap()
    n = int(HALF * SR)
    waves = []
    for it in items:
        y, _ = librosa.load(it["file"], sr=SR, mono=True)
        w = y[int(it["at"] * SR) - n: int(it["at"] * SR) + n]
        waves.append(np.pad(w, (0, max(0, 2 * n - len(w)))).astype(np.float32))
    waves = np.stack(waves)
    mel = torch.tensor(np.concatenate([mel_batch(waves[k:k + 64]) for k in range(0, len(waves), 64)])).float().cuda()
    cf = torch.tensor(np.concatenate([clap_feats(clap, waves[k:k + 32]) for k in range(0, len(waves), 32)])).float().cuda()
    with torch.no_grad():
        p_nat = torch.sigmoid(model(mel, cf)).cpu().numpy()
        feats = model.features(mel, cf).cpu().numpy()
    user = np.array([{"bad": 0, "ok": 1, "good": 2}[it["verdict"]] for it in items])

    def auc(scores, lab):
        g, b = scores[lab == 2], scores[lab == 0]
        return float(np.mean([[1.0 if x > y else 0.5 if x == y else 0.0 for y in b] for x in g]))
    from scipy.stats import spearmanr
    print(f"self-supervised critic alone: good-vs-bad AUC {auc(p_nat, user):.2f}, Spearman {spearmanr(p_nat, user).correlation:.2f}")
    from . import lab
    lab.reset("critic", auc_raw=auc(p_nat, user), rows=[
        {"id": it["id"], "round": it["round"], "kind": it["kind"], "verdict": it["verdict"], "note": it.get("note", ""),
         "file": "clips/" + os.path.basename(it["file"]), "p_natural": round(float(p), 3)} for it, p in zip(items, p_nat)])
    print("   mean P(natural): good %.2f / ok %.2f / bad %.2f" % tuple(p_nat[user == k].mean() for k in (2, 1, 0)))
    # user-taste head, leave-one-round-out
    X = np.concatenate([feats, p_nat[:, None]], 1)
    rounds = np.array([it["round"] for it in items])
    pred = np.zeros(len(items))
    for r in np.unique(rounds):
        tr, te = rounds != r, rounds == r
        clf = LogisticRegression(C=0.05, max_iter=2000).fit(X[tr], (user[tr] == 2).astype(int))
        pred[te] = clf.predict_proba(X[te])[:, 1]
    print(f"calibrated on your ratings (each round predicted by a head trained on the other rounds): "
          f"good-vs-bad AUC {auc(pred, user):.2f}")
    d = lab._read("critic")
    for row, pr in zip(d.get("rows", []), pred):
        row["taste"] = round(float(pr), 3)
    d["auc_taste"] = auc(pred, user)
    d["done"] = True
    lab._write("critic", d)
    clf = LogisticRegression(C=0.05, max_iter=2000).fit(X, (user == 2).astype(int))
    import pickle
    pickle.dump(clf, open(os.path.join(OUT, "taste_head.pkl"), "wb"))
    rows = sorted(zip(pred, user, [it["kind"] for it in items], [it["note"] for it in items]), key=lambda r: r[0])
    print("lowest-scored (model thinks worst):")
    for p_, u, k, note in rows[:6]:
        print(f"   {p_:.2f}  you: {['bad','ok','good'][u]:>4}  {k:<18} {note[:60]}")
    print("highest-scored:")
    for p_, u, k, note in rows[-6:]:
        print(f"   {p_:.2f}  you: {['bad','ok','good'][u]:>4}  {k:<18} {note[:60]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "train":
        train()
    elif cmd == "build_rank":
        build_rank()
    elif cmd == "build_game":
        build_game()
    elif cmd == "calibrate":
        calibrate(sys.argv[2])
    elif cmd == "score":
        # score rendered clips: in = [[path, [seam, ...]], ...]; out = one score per clip: your taste
        # layer if trained (higher = you'd like it more), else min P(natural) over seams
        jobs = json.load(open(sys.argv[2], encoding="utf-8"))
        vecs, mins = clip_vectors(jobs) if jobs else ([], [])
        if os.path.exists(TASTE) and vecs:
            w = np.array(json.load(open(TASTE))["w"])
            out = [float(v @ w) for v in vecs]
        else:
            out = mins
        json.dump(out, open(sys.argv[3], "w", encoding="utf-8"))
    elif cmd == "styles":
        model_styles()
    elif cmd == "taste":
        r = train_taste_head()
        print(json.dumps(r))
        if len(sys.argv) > 2:
            json.dump(r, open(sys.argv[2], "w", encoding="utf-8"))
