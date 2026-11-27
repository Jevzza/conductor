"""A song's hook: its most repeated melodic phrase (2026-11-24, user: "something iconic to that song
... sprinkled in before or after the transition to help blend").

    python -m conductor.hooks            # find hooks for every training song (cached)

The melodic part is the mix minus its drums+bass stem. Every 2-bar window gets a bar-synchronous
chroma fingerprint; the hook is the loud window that recurs most often elsewhere in the song
(the riff/melody the song keeps coming back to). Cached in cache/hooks/<key>.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "cache", "hooks")
BARS = 2
STEPS = 8          # chroma samples per bar


def _key(track) -> str:
    from .analysis import cache_key
    return cache_key(track.path)


PICKS = os.path.join(HERE, "review", "hooks", "picks.json")
_PICKS: dict = {}


def picked(track):
    """The user's hook pick (review/hooks/picks.json, by title): a hook dict, "none", or None (not picked)."""
    try:
        st = os.stat(PICKS).st_mtime
        if _PICKS.get("_m") != st:
            _PICKS.clear(); _PICKS.update(json.load(open(PICKS, encoding="utf-8"))); _PICKS["_m"] = st
    except (OSError, ValueError):
        return None
    return _PICKS.get(track.title)


def find(track) -> list[dict]:
    """Up to 3 hooks [{"t": start s, "len": s, "bars": 2, "repeats": n, "score": x}], best first.
    The user's pick (hook picker page) wins; "none" = this song has no usable hook."""
    pk = picked(track)
    if pk == "none":
        return []
    if isinstance(pk, dict):
        return [pk]
    if isinstance(pk, list) and pk:                 # several picked (2026-11-24); sounds are separate
        hs = [h for h in pk if h.get("kind") != "sound"]
        if hs:
            return hs
    p = os.path.join(CACHE, _key(track) + ".json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    import librosa
    from . import routes
    y = routes._audio(track)
    st = routes._stem(track)
    mel = (y - st[:len(y)] if st is not None and len(st) >= len(y) * 0.98 else y).mean(1)
    sr = 22050
    m = librosa.resample(mel, orig_sr=routes.SR, target_sr=sr)
    hop = 512
    ch = librosa.feature.chroma_cqt(y=m, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=m, hop_length=hop)[0]
    db = [b for b in (track.downbeats or track.beats[::4])]
    fr = lambda t: min(ch.shape[1] - 1, max(0, int(t * sr / hop)))
    wins = []
    for i in range(len(db) - BARS):
        cols = []
        for b in range(BARS):
            a, e = db[i + b], db[i + b + 1]
            for k in range(STEPS):
                lo, hi = fr(a + (e - a) * k / STEPS), fr(a + (e - a) * (k + 1) / STEPS)
                cols.append(ch[:, lo:max(hi, lo + 1)].mean(1))
        v = np.concatenate(cols)
        v = v - v.mean()
        n = np.linalg.norm(v)
        wins.append((db[i], db[i + BARS] - db[i], v / n if n else v, float(rms[fr(db[i]):fr(db[i + BARS]) + 1].mean())))
    if len(wins) < 8:
        out = []
    else:
        V = np.stack([w[2] for w in wins])
        S = V @ V.T
        loud = np.array([w[3] for w in wins])
        thr = np.percentile(loud, 50)
        idx = np.arange(len(wins))
        rep = np.array([int(((S[i] > 0.8) & (np.abs(idx - i) > BARS)).sum()) for i in range(len(wins))])
        L = np.array([w[1] for w in wins])
        sane = np.abs(L / np.median(L) - 1) < 0.25                # bar grid glitches give 0.3 s "2 bars"
        drone = rep > 0.4 * len(wins)                             # a pad that matches everything isn't a hook
        score = rep * (loud >= thr) * sane * (loud / (loud.max() or 1)) * np.where(drone, 0.3, 1.0)
        out, taken = [], []
        for i in np.argsort(-score):
            if score[i] <= 0 or len(out) >= 3:
                break
            if any(S[i, j] > 0.8 for j in taken):          # same phrase as one already chosen
                continue
            taken.append(i)
            out.append({"t": round(wins[i][0], 3), "len": round(wins[i][1], 3), "bars": BARS,
                        "repeats": int(rep[i]), "score": round(float(score[i]), 3)})
    os.makedirs(CACHE, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    return out


def candidates(track, k: int = 5) -> list[dict]:
    """Hook candidates for the picker (2026-11-24, user: "certain parts of songs where hooks stand out
    more... we choose something else"). Scores 2-bar melodic phrases on: how much the melody stands
    out over the backing (melodic / full-mix loudness), loudness, repetition (capped - a loop that
    plays all song isn't the hook), and the section it's in (peaks/drops/builds over intros). Spread
    over the song; the song's vocal call-outs join as candidates."""
    import librosa
    from . import routes
    from .states import role_of
    y = routes._audio(track)
    st = routes._stem(track)
    mixm = y.mean(1)
    mel = (y - st[:len(y)]).mean(1) if st is not None and len(st) >= len(y) * 0.98 else mixm
    sr = 22050
    m = librosa.resample(mel, orig_sr=routes.SR, target_sr=sr)
    full = librosa.resample(mixm, orig_sr=routes.SR, target_sr=sr)
    hop = 512
    ch = librosa.feature.chroma_cqt(y=m, sr=sr, hop_length=hop)
    rm = librosa.feature.rms(y=m, hop_length=hop)[0]
    rf = librosa.feature.rms(y=full, hop_length=hop)[0]
    db = list(track.downbeats or track.beats[::4])
    fr = lambda t: min(ch.shape[1] - 1, max(0, int(t * sr / hop)))
    W = []
    for i in range(0, len(db) - BARS):
        a, e = db[i], db[i + BARS]
        cols = [ch[:, fr(a + (e - a) * q / (BARS * STEPS)):max(fr(a + (e - a) * q / (BARS * STEPS)) + 1,
                    fr(a + (e - a) * (q + 1) / (BARS * STEPS)))].mean(1) for q in range(BARS * STEPS)]
        v = np.concatenate(cols); v -= v.mean(); v /= (np.linalg.norm(v) or 1)
        lo, hi = fr(a), fr(e) + 1
        seg = track.segment_at(a + 0.1)
        role = role_of(seg) if seg is not None else None
        W.append((a, e - a, v, float(rm[lo:hi].mean()), float(rm[lo:hi].mean() / (rf[lo:hi].mean() + 1e-9)), role))
    if len(W) < 8:
        return []
    V = np.stack([w[2] for w in W]); S = V @ V.T
    L = np.array([w[1] for w in W]); sane = np.abs(L / np.median(L) - 1) < 0.25
    idx = np.arange(len(W))
    rep = np.array([((S[i] > 0.8) & (np.abs(idx - i) > BARS)).sum() for i in range(len(W))])
    loud = np.array([w[3] for w in W]); prom = np.array([w[4] for w in W])
    rw = np.array([{"peak": 1.3, "horde": 1.2, "build": 1.15, "skirmish": 1.0}.get(w[5], 0.8) for w in W])
    score = (np.minimum(rep, 6) + 1) ** 0.5 * (loud / (loud.max() or 1)) * np.clip(prom, 0, 1.5) * rw * sane
    out, taken = [], []
    for i in np.argsort(-score):
        if score[i] <= 0 or len(out) >= k:
            break
        if any(S[i, j] > 0.8 or abs(W[i][0] - W[j][0]) < 4 * W[i][1] for j in taken):
            continue
        taken.append(i)
        out.append({"t": round(W[i][0], 3), "len": round(W[i][1], 3), "bars": BARS, "kind": "melody",
                    "score": round(float(score[i]), 3),
                    "f": {"rep": float(min(rep[i], 12)), "loud": float(loud[i] / (loud.max() or 1)), "prom": float(prom[i]),
                          "role": W[i][5] or "none", "pos": float(W[i][0] / max(1.0, track.duration))}})
    try:                                                     # the song's vocal call-outs
        from .analysis import cache_key
        cj = os.path.join(HERE, "cache", "stems", cache_key(track.path) + "_callouts.json")
        shouts = [c for c in json.load(open(cj, encoding="utf-8"))["callouts"] if c["type"] == "shout"][:1]
        for c in shouts:
            t0 = max([d for d in db if d <= c["t"] + 0.05] or [c["t"]])
            out.append({"t": round(t0, 3), "len": round(float(np.median(L)), 3), "bars": BARS, "kind": "vocal",
                        "f": {"rep": 0.0, "loud": float(c.get("score", 0)) / 30.0, "prom": 1.0,
                              "role": "vocal", "pos": float(t0 / max(1.0, track.duration))}})
    except (OSError, ValueError, KeyError):
        pass
    return sorted(out, key=lambda h: h["t"])


def sound_candidates(track, k: int = 3) -> list[dict]:
    """Moments to isolate a signature sound from (2026-11-24, user: "a really high-pitched spooky sound in
    the background of Miasma... isolate it, filtering out everything else"). Automatic tone-finding kept
    catching the melody's own notes, so this just finds where the song's melodic layer (mix minus drums+
    bass) has its high end most EXPOSED - high energy vs the rest of the melodic layer - in k different
    sections, and the picker's live filter + spectrum lets the user dial the sound in by ear."""
    import librosa
    from . import routes
    y = routes._audio(track)
    st = routes._stem(track)
    mel = (y - st[:len(y)]).mean(1) if st is not None and len(st) >= len(y) * 0.98 else y.mean(1)
    sr = 22050
    m = librosa.resample(mel, orig_sr=routes.SR, target_sr=sr)
    S = np.abs(librosa.stft(m, n_fft=2048, hop_length=1024)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hi = S[(f >= 2000) & (f <= 11000)].sum(0) + 1e-12
    rest = S[(f >= 100) & (f < 2000)].sum(0) + 1e-12
    expo = 10 * np.log10(hi) - 10 * np.log10(rest) + 0.25 * (10 * np.log10(hi))   # exposed AND present
    win = int(8 * sr / 1024)
    sm = np.convolve(expo, np.ones(win) / win, mode="same")
    db = list(track.downbeats or track.beats[::4])
    out, taken = [], []
    for fr in np.argsort(-sm):
        if len(out) >= k:
            break
        t = max(0.0, fr * 1024 / sr - 4.0)
        seg = track.segment_at(t + 4.0)
        if any(abs(t - u) < 20 for u in taken) or (seg is not None and any(o.get("_seg") is seg for o in out)):
            continue
        taken.append(t)
        t = max([d for d in db if d <= t] or [t])
        out.append({"t": round(float(t), 3), "len": 8.0, "lo": 2000, "hi": 12000, "kind": "sound", "_seg": seg})
    for o in out:
        o.pop("_seg", None)
    return sorted(out, key=lambda o: o["t"])


def sounds(track) -> list[dict]:
    """The user's picked signature sounds for this song (picker page), [] if none - snapped to the
    beat (user 2026-11-24: "I'm just roughly selecting them... you'll have to make sure they fit the beat")."""
    pk = picked(track)
    if isinstance(pk, list):
        return [snap_sound(track, h) for h in pk if h.get("kind") == "sound"]
    return []


def snap_sound(track, h: dict) -> dict:
    """Start on the beat nearest the user's box (the bar line if one is within half a beat); length to
    whole beats (whole bars from 2 bars up), at least one beat. Uses the song's audio-fitted grid."""
    from . import grid
    f = grid.fit_at(track, h["t"])
    if f is None and grid.fits(track):                      # no beats detected here (a breakdown): carry the
        f = min(grid.fits(track), key=lambda q: min(abs(h["t"] - q["start"]), abs(h["t"] - q["end"])))   # nearest line on
    if f is not None:                                       # the fitted line right here: exact beat positions
        p = f["period"]
        t = f["t0"] + round((h["t"] - f["t0"]) / p) * p
    else:
        p = grid.period(track, h["t"])
        t = min(track.beats, key=lambda b: abs(b - h["t"])) if track.beats else h["t"]
    bars = [d for d in (track.downbeats or []) if abs(d - h["t"]) < 0.5 * p]
    if bars:
        t = min(bars, key=lambda d: abs(d - h["t"]))
    beats = max(1, round(h["len"] / p))
    if beats >= 8:
        beats = 4 * round(beats / 4)
    return dict(h, t=round(float(t), 3), len=round(float(beats * p), 3), beats=int(beats), period=round(float(p), 4))


def near(track, t: float):
    """The song's hook to use at time t: of the user's picks (or the automatic ones), the one closest
    to where the blend happens - a drop's hook near the drop, an intro's near the intro."""
    hs = find(track)
    if not hs:
        return None
    return min(hs, key=lambda h: abs(h["t"] - t)) if len(hs) > 1 else hs[0]


def melodic(track, t: float, n: int) -> np.ndarray:
    """n samples of the song's melodic part (mix minus drums+bass) from t, at the mix's level."""
    from . import routes
    y = routes._take(routes._audio(track), t, n=n)
    st = routes._stem(track)
    return y - routes._take(st, t, n=n) if st is not None else y


if __name__ == "__main__":
    import time
    from .ab import training_library
    lib = training_library()
    t0 = time.time()
    for k, T in enumerate(lib.tracks):
        h = find(T)
        print(f"{k + 1}/{len(lib.tracks)} {T.title[:40]:40} {[(x['t'], x['repeats']) for x in h]}  {time.time() - t0:.0f}s",
              flush=True)
