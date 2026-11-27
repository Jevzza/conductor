"""Fitted beat grids per section, for long blends (2026-11-24).

User: "we should be sampling throughout the song as segments change... if we're doing these longer
blends, it's critical that the BPM is matched, on the bar."

The tempo map's beats jitter (a pulse tracker), and a blend used ONE speed taken at its first beat, so
a 1 % error drifted a third of a beat over 8 bars. Here:

  * every section gets a straight-line fit of its beats (time = t0 + k * period, outliers dropped):
    an exact tempo and phase per section, plus how far the detected beats sit off that line
  * the fitted grid is checked against the audio: drum onsets should land ON the beats, not between
    (tempo._fit; on-beat / off-beat onset strength)
  * blends map the old song beat-for-beat onto the new one (warp_positions), so the bar lines meet on
    every bar even when one of the songs drifts
"""
from __future__ import annotations

import bisect
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_FITS: dict = {}
_ENV: dict = {}
MAX_RESID_MS = 60.0       # detector beats vs the line: loose - the line is fitted to the audio now
MIN_SALIENCE = 1.25       # drum onsets on the beat vs half-way between


def _fit_section(beats: list[float], a: float, b: float) -> dict | None:
    x = np.array([t for t in beats if a <= t < b])
    if len(x) < 8:
        return None
    d = np.diff(x)
    p0 = float(np.median(d))
    k = np.round((x - x[0]) / p0)
    keep = np.ones(len(x), bool)
    for _ in range(3):                                   # robust: refit without the stray beats
        if keep.sum() < 6:
            return None
        slope, icpt = np.polyfit(k[keep], x[keep], 1)
        r = x - (icpt + slope * k)
        keep = np.abs(r) < max(0.025, 2.5 * np.median(np.abs(r[keep])))
    return {"start": a, "end": b, "t0": float(icpt), "period": float(slope),
            "resid_ms": float(np.median(np.abs(r[keep])) * 1000), "n": int(keep.sum())}


def _env(track):
    from . import tempo
    if track.path not in _ENV:
        _ENV[track.path] = tempo._onset_env(track.path)
        if len(_ENV) > 16:
            _ENV.pop(next(iter(_ENV)))
    return _ENV[track.path]


def _refine(f: dict, env, a: float, b: float) -> dict:
    """Fit the grid to the AUDIO: search tempo (+-3 %, 0.1 % steps) and phase (whole beat, 5 ms steps)
    for the line that puts the most drum-onset energy on the beats and the least half-way between.
    Fixes the beat detector's half-beat slips and small tempo errors instead of just averaging them."""
    from .tempo import HOP, SR as ESR
    fps = ESR / HOP
    p0 = f["period"]
    best = (-1e9, p0, f["t0"])
    # 2026-11-24: also try the song's main tempo, double and half - the detector sometimes locks onto an
    # odd pulse in a section (Miasma: 130 BPM sections next to "208 BPM" ones); the drums decide which
    centres = {p0}
    main = f.get("main_period")
    if main:
        centres |= {main, main / 2, main * 2}
    for c in centres:
        for p in c * (1 + np.arange(-0.03, 0.0301, 0.001)):
            nb = int((b - a) / p)
            if nb < 6 or not 0.22 <= p <= 1.3:              # 46-270 BPM
                continue
            ph = np.arange(0, p, 0.005)[:, None]
            t = a + ph + np.arange(nb)[None, :] * p          # every phase at once
            on = env[np.clip(np.round(t * fps).astype(int), 0, len(env) - 1)].mean(1)
            off = env[np.clip(np.round((t + p / 2) * fps).astype(int), 0, len(env) - 1)].mean(1)
            i = int(np.argmax(on - off))
            if on[i] - off[i] > best[0]:
                best = (float(on[i] - off[i]), float(p), float(a + ph[i, 0]))
    _, p, t0 = best
    # keep the original bar phase: shift the fitted origin by whole beats to the old line's nearest beat
    t0 = t0 + round((f["t0"] - t0) / p) * p
    return dict(f, period=float(p), t0=float(t0), refined=True)


def _fold(track, fits_: list[dict]) -> list[dict]:
    """Keep every section's beat at the song's main pulse (2026-11-25: Data Interference went 0.556 s ->
    0.278 s at a section line, so a blend crossing it warped the other song to double speed). Halving a
    period just adds the in-between beats; doubling keeps the on-beat half with the most drum energy."""
    from .tempo import HOP, SR as ESR
    env = None
    for f in fits_:
        main = f.get("main_period")
        if not main or f.get("folded"):
            continue
        p, t0 = f["period"], f["t0"]
        while p > main * 1.41:
            p /= 2
        while p < main / 1.41:
            if env is None:
                env = _env(track)
            fps = ESR / HOP
            nb = int((f["end"] - f["start"]) / (2 * p))
            def on(o):
                t = o + np.arange(max(nb, 1)) * 2 * p
                return env[np.clip(np.round(t * fps).astype(int), 0, len(env) - 1)].mean()
            if on(t0 + p) > on(t0):
                t0 += p
            p *= 2
        f.update(period=float(p), t0=float(t0), folded=True)
    return fits_


def _cache_path(track) -> str:
    from .analysis import cache_key
    d = os.path.join(HERE, "cache", "grid")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, cache_key(track.path) + ".json")


def fits(track) -> list[dict]:
    if track.path not in _FITS:
        import json
        try:
            _FITS[track.path] = _fold(track, json.load(open(_cache_path(track), encoding="utf-8")))
            return _FITS[track.path]
        except (OSError, ValueError):
            pass
        out, env = [], None
        for s in track.segments:
            f = _fit_section(track.beats, s.start, s.start + s.length)
            if f is not None:
                try:
                    env = env if env is not None else _env(track)
                    if getattr(track, "tempo", None):
                        f["main_period"] = 60.0 / float(track.tempo)
                    f = _refine(f, env, s.start, s.start + s.length)
                except Exception:
                    pass
                out.append(f)
        _FITS[track.path] = out = _fold(track, out)
        try:
            json.dump(out, open(_cache_path(track), "w", encoding="utf-8"))
        except OSError:
            pass
    return _FITS[track.path]


def fit_at(track, t: float) -> dict | None:
    for f in fits(track):
        if f["start"] - 0.05 <= t < f["end"]:
            return f
    return None


def beats_from(track, t: float, n: int) -> list[float] | None:
    """n+1 fitted beats starting at the fitted beat nearest t, following each section's own line
    (crossing into the next section picks up that section's grid)."""
    out, f = [], fit_at(track, t)
    if f is None:
        return None
    k = round((t - f["t0"]) / f["period"])
    cur = f["t0"] + k * f["period"]
    while len(out) <= n:
        out.append(cur)
        nxt = cur + f["period"]
        if nxt >= f["end"]:
            g = fit_at(track, nxt)
            if g is None:
                return None if len(out) <= n else out
            if g is not f:
                kk = round((nxt - g["t0"]) / g["period"])
                nxt = g["t0"] + kk * g["period"]
                if nxt <= cur + 0.5 * f["period"]:
                    nxt += g["period"]
                f = g
        cur = nxt
    return out


def _salience(track, a: float, b: float) -> float:
    from . import tempo
    bs = beats_from(track, a, max(8, int((b - a) / max(0.2, period(track, a)))))
    if not bs:
        return 0.0
    return tempo._fit(bs, _env(track), a - 0.01, b + 0.01)


def period(track, t: float) -> float:
    f = fit_at(track, t)
    if f is not None:
        return f["period"]
    from .tempo import period_at
    return period_at(track, t)


def ok(track, t0: float, t1: float) -> bool:
    """Every section in [t0, t1] has a tight fitted grid, and the drums really sit on it."""
    secs = [f for f in fits(track) if f["end"] > t0 and f["start"] < t1]
    covered = sum(min(f["end"], t1) - max(f["start"], t0) for f in secs)
    if not secs or covered < 0.95 * (t1 - t0):
        return False
    if any(f["resid_ms"] > MAX_RESID_MS and not f.get("refined") for f in secs):   # audio-fitted: the drums decide
        return False
    return _salience(track, t0, t1) >= MIN_SALIENCE


def warp_positions(A, t_old: float, B, t_new: float, n: int, sr: int) -> np.ndarray | None:
    """For each of n output samples on B's timeline (from t_new), the time in A that should play there,
    beat-for-beat: A's k-th beat after t_old lands on B's k-th beat after t_new. None if the tempos are
    too far apart for a pitch-fader nudge (> ~6 %, after half/double time) or a grid is missing."""
    pb = period(B, t_new)
    nb = int(n / sr / pb) + 3
    bb = beats_from(B, t_new, nb)
    pa = period(A, t_old)
    ratio = pb / pa                                      # A beats per B beat (2026-11-25: was inverted -> 1/4 or 4x speed)
    step = 2 ** round(np.log2(ratio))                    # half/double time
    if abs(np.log2(ratio / step)) > 0.085 or bb is None:
        return None
    ab = beats_from(A, t_old, int(nb * step) + 2)
    if ab is None:
        return None
    ab = np.array(ab)[:: int(step)] if step >= 1 else np.interp(np.arange(0, len(ab) - 1, step), np.arange(len(ab)), ab)
    k = min(len(ab), len(bb))
    if k < 2:
        return None
    ab, bb = np.asarray(ab[:k]), np.asarray(bb[:k])
    tau = t_new + np.arange(n) / sr
    # the fitted first beats can sit a few ms off t_old / t_new: keep the blend starting exactly there
    ab = ab - (ab[0] - t_old)
    bb = bb - (bb[0] - t_new)
    # 2026-11-26: a section crossing can snap the fitted grid by part of a beat (one beat 0.7x / 1.07x long),
    # which made the guard below refuse 80 % of "beat" blends -> they played unlocked (offbeat). Spread any
    # such jump over ~8 beats instead: the bar lines still meet, and the rate never jumps.
    if k >= 4:
        rate = float(np.median(np.diff(ab) / np.diff(bb)))
        lin = ab[0] + rate * (bb - bb[0])
        e = ab - lin
        w = min(9, k if k % 2 else k - 1)
        pad = np.concatenate([np.full(w // 2, e[0]), e, np.full(w // 2, e[-1])])
        es = np.convolve(pad, np.ones(w) / w, "valid")
        es = es - es[0]                                  # still starts exactly at t_old
        ab = lin + es
    src = np.interp(tau, bb, ab)
    past = tau > bb[-1]                                  # beyond the last mapped beat: carry on at the last rate
    if past.any():
        rate = (ab[-1] - ab[-2]) / (bb[-1] - bb[-2])
        src[past] = ab[-1] + (tau[past] - bb[-1]) * rate
    # never a speed jump mid-blend: A plays within ~6 % of normal speed, steady to 4 %, or no beat-lock at all
    r = np.diff(ab) / np.diff(bb)
    if r.min() < 0.94 or r.max() > 1.06 or r.max() / r.min() > 1.04:
        return None
    return src


def take_at(y: np.ndarray, src_times: np.ndarray, sr: int) -> np.ndarray:
    """Samples of y (n, 2) at these (fractional) times - the same positions for every layer, so mix,
    stems and drums stay sample-locked (no phasey subtraction)."""
    pos = src_times * sr
    lo = max(0, int(pos.min()) - 2)
    hi = min(len(y), int(pos.max()) + 3)
    seg = y[lo:hi]
    xs = np.arange(lo, lo + len(seg))
    return np.stack([np.interp(pos, xs, seg[:, c], left=0, right=0) for c in range(2)], 1).astype(np.float32)
