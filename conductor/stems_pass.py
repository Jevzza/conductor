"""Stems + call-outs (ML venv, GPU):  .venv-ml\\Scripts\\python -m conductor.stems_pass <folder>

For each song:
  cache/stems/<key>_rhythm.flac   drums + bass at 44.1 kHz stereo, sample-aligned with the decoded
                                  mix — the player turns this down for lulls (vertical layering)
  cache/stems/<key>_callouts.json + clips: short reusable moments, each ending/starting on a bar line
      impact  — a big drum/bass hit that rings (stingers, punctuating a change)
      shout   — a short vocal burst with space around it (call-outs)
      riser   — 2–8 s of rising energy in the melodic stem ending on a downbeat (bridges into changes)
      fill    — a burst of drum onsets in the last bar of a phrase (bridges)
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "cache", "stems")
SR = 44100


def _rms_db(x: np.ndarray, hop: int) -> np.ndarray:
    n = len(x) // hop
    fr = x[: n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-6)


def _write(path, audio):
    import soundfile as sf
    fade = min(len(audio), int(0.25 * SR))
    audio = audio.copy()
    audio[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
    audio[: int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))[:, None]
    sf.write(path, audio, SR, subtype="PCM_16")


def find_callouts(stems: dict, downbeats: list[float], beats: list[float], key: str) -> list[dict]:
    hop = SR // 50                          # 20 ms frames
    fps = SR / hop
    db = {k: _rms_db(v.mean(1), hop) for k, v in stems.items()}
    dn = np.asarray(downbeats) if downbeats else np.asarray(beats[::4])
    out = []
    # impacts: drum+bass jumps on (or within 60 ms of) a downbeat, ringing afterwards
    rb = 10 * np.log10(10 ** (db["drums"] / 10) + 10 ** (db["bass"] / 10))
    cands = []
    for t in dn:
        i = int(t * fps)
        if i < 10 or i + 100 >= len(rb):
            continue
        jump = rb[i:i + 4].max() - rb[i - 10:i - 1].mean()
        ring = rb[i + 20:i + 60].mean() - rb[i:i + 4].max()       # how much is left 0.4–1.2 s later
        if jump >= 8:
            cands.append((jump + 0.3 * ring, t))
    for score, t in sorted(cands, reverse=True)[:4]:
        out.append({"type": "impact", "t": round(float(t), 3), "len": 1.6, "score": round(float(score), 1),
                    "source": "drums+bass"})
    # shouts: vocal bursts 0.3–3 s long, 6+ dB above the 1 s before and after
    v = db["vocals"]
    on = v > (np.percentile(v, 60) + 6)
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            L = (j - i) / fps
            if 0.3 <= L <= 3.0 and i > 50 and j + 50 < len(v):
                ctx = max(v[i - 50:i - 5].mean(), v[j + 5:j + 50].mean())
                if v[i:j].mean() - ctx >= 6:
                    t = i / fps
                    out.append({"type": "shout", "t": round(t, 3), "len": round(L + 0.3, 2),
                                "score": round(float(v[i:j].mean() - ctx), 1), "source": "vocals"})
            i = j
        i += 1
    shouts = sorted([c for c in out if c["type"] == "shout"], key=lambda c: -c["score"])[:6]
    out = [c for c in out if c["type"] != "shout"] + shouts
    # risers: the melodic stem climbs 9+ dB over 2–8 s into a downbeat
    o = db["other"]
    rises = []
    for t in dn:
        e = int(t * fps)
        for L in (2, 4, 8):
            s0 = e - int(L * fps)
            if s0 < 0 or e >= len(o):
                continue
            a, b = o[s0:s0 + 10].mean(), o[e - 10:e].mean()
            mono = np.mean(np.diff(np.convolve(o[s0:e], np.ones(10) / 10, mode="valid")) > 0)
            if b - a >= 9 and mono >= 0.55:
                rises.append((b - a, t, L))
    for gain, t, L in sorted(rises, reverse=True)[:3]:
        out.append({"type": "riser", "t": round(float(t - L), 3), "len": float(L), "end": round(float(t), 3),
                    "score": round(float(gain), 1), "source": "other"})
    # fills: many drum onsets in the last bar before a downbeat
    import librosa
    d_on = librosa.onset.onset_strength(y=stems["drums"].mean(1), sr=SR, hop_length=hop)
    peaks = librosa.util.peak_pick(d_on, pre_max=2, post_max=2, pre_avg=5, post_avg=5, delta=0.3, wait=3)
    pk = peaks / fps
    fills = []
    for k in range(1, len(dn)):
        a, b = dn[k - 1], dn[k]
        n_last = np.sum((pk >= a + (b - a) * 0.5) & (pk < b))
        n_first = np.sum((pk >= a) & (pk < a + (b - a) * 0.5))
        if n_last >= 6 and n_last >= 2 * max(1, n_first):
            fills.append((n_last, a + (b - a) * 0.5, b))
    for n, s0, e in sorted(fills, reverse=True)[:4]:
        out.append({"type": "fill", "t": round(float(s0), 3), "len": round(float(e - s0), 3), "end": round(float(e), 3),
                    "score": float(n), "source": "drums"})
    # cut the clips
    for c in out:
        src = {"drums+bass": stems["drums"] + stems["bass"], "vocals": stems["vocals"],
               "other": stems["other"], "drums": stems["drums"]}[c["source"]]
        a, n = int(c["t"] * SR), int(c["len"] * SR)
        clip = src[a:a + n]
        if len(clip) < SR // 4:
            continue
        c["file"] = os.path.join(OUT, f"{key}_{c['type']}_{int(c['t'] * 1000)}.wav")
        _write(c["file"], clip)
    return [c for c in out if c.get("file")]


def main(folder: str) -> None:
    import librosa
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from .analysis import cache_key
    from .library import scan
    os.makedirs(OUT, exist_ok=True)
    from .allin1_pass import demix_dir
    model = None
    for i, path in enumerate(scan(folder), 1):
        key = cache_key(path)
        meta = os.path.join(OUT, key + "_callouts.json")
        if os.path.exists(meta):
            continue
        t0 = time.time()
        d = json.load(open(os.path.join(HERE, "cache", key + ".json"), encoding="utf-8"))
        dd = demix_dir(path)
        names = ("drums", "bass", "other", "vocals")
        if all(os.path.exists(os.path.join(dd, n + ".wav")) for n in names):
            stems = {n: sf.read(os.path.join(dd, n + ".wav"), dtype="float32")[0] for n in names}   # split once, in ml_pass
        else:
            if model is None:
                model = get_model("htdemucs").cuda().eval()
            y, _ = librosa.load(path, sr=SR, mono=False)
            y = np.atleast_2d(y)
            if y.shape[0] == 1:
                y = np.vstack([y, y])
            wav = torch.tensor(y[:2], dtype=torch.float32)[None].cuda()
            ref = wav.mean(1)
            mu, sd = ref.mean(), ref.std()
            with torch.no_grad():
                st = apply_model(model, (wav - mu) / sd, device="cuda", split=True, overlap=0.25)[0]
            st = (st * sd + mu).cpu().numpy()
            stems = {name: st[k].T for k, name in enumerate(model.sources)}      # (n, 2) each
        rhythm = stems["drums"] + stems["bass"]
        sf.write(os.path.join(OUT, key + "_rhythm.flac"), rhythm, SR, subtype="PCM_16")
        sf.write(os.path.join(OUT, key + "_drums.flac"), stems["drums"], SR, subtype="PCM_16")   # bass = rhythm - drums
        calls = find_callouts(stems, d.get("downbeats", []), d.get("beats", []), key)
        json.dump({"path": path, "title": d["title"], "callouts": calls}, open(meta, "w", encoding="utf-8"), indent=1)
        counts = {}
        for c in calls:
            counts[c["type"]] = counts.get(c["type"], 0) + 1
        print(f"[{i}] {time.time() - t0:4.1f}s {d['title'][:40]}  {counts}", flush=True)
        if os.path.isdir(dd) and os.path.basename(os.path.dirname(dd)) == "htdemucs":
            import shutil
            shutil.rmtree(dd, ignore_errors=True)        # the shared split has served all three passes
        wav = st = None
        torch.cuda.empty_cache()


def backfill_drums(paths: list[str]) -> None:
    """2026-11-24: drums on their own (layered blends faked "bass" as < 150 Hz, which also stripped the
    kick's body - heard as a filter sweep). Re-splits songs that only have the combined rhythm stem."""
    import librosa
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from .analysis import cache_key
    model = get_model("htdemucs").cuda().eval()
    todo = [p for p in paths if not os.path.exists(os.path.join(OUT, cache_key(p) + "_drums.flac"))]
    t00 = time.time()
    for i, path in enumerate(todo, 1):
        t0 = time.time()
        y, _ = librosa.load(path, sr=SR, mono=False)
        y = np.atleast_2d(y)
        if y.shape[0] == 1:
            y = np.vstack([y, y])
        wav = torch.tensor(y[:2], dtype=torch.float32)[None].cuda()
        ref = wav.mean(1)
        mu, sd = ref.mean(), ref.std()
        with torch.no_grad():
            st = apply_model(model, (wav - mu) / sd, device="cuda", split=True, overlap=0.25)[0]
        st = (st * sd + mu).cpu().numpy()
        drums = st[model.sources.index("drums")].T
        sf.write(os.path.join(OUT, cache_key(path) + "_drums.flac"), drums, SR, subtype="PCM_16")
        print(f"[{i}/{len(todo)}] {time.time() - t0:4.1f}s {os.path.basename(path)[:50]}  ({time.time() - t00:.0f}s)", flush=True)
        wav = st = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "drums":          # drums <paths file>: backfill separate drum stems
        backfill_drums([l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else __import__("conductor.playlists", fromlist=["train_music"]).train_music())
