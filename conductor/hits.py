"""Find the moments in a song where an abrupt transition will sound intentional.

  hits  — impacts: a drop landing, a big downbeat crash. The music gets suddenly and
          substantially louder and stays loud. Cutting INTO a song exactly on a hit
          sounds like the game score slamming in.
  stops — cut-outs: the music suddenly falls away (a break, a stop before a drop).
          Cutting OUT of a song on a stop sounds like the song chose to end there.

Both are measured on a short-term loudness curve (~12 ms resolution) and snapped to the
nearest detected beat when one is close, so cuts land in time.
"""
from __future__ import annotations

import numpy as np

SR = 22050
HOP = 256
PRE, POST, SUSTAIN = 0.4, 0.4, 1.5      # seconds compared before/after, and how long a hit must hold
HIT_MIN_JUMP_DB = 6.0
BASS_MIN_JUMP_DB = 8.0                   # EDM drops: the sub-bass slams in even when overall level doesn't jump
BASS_HZ = 150
STOP_MIN_DROP_DB = 10.0
MIN_SPACING = 2.0
SNAP = 0.02                              # snap only when a beat is essentially on it: keep the real transient
MAX_EACH = 40


def _snap(t: float, beats: np.ndarray) -> float:
    if len(beats):
        i = int(np.argmin(np.abs(beats - t)))
        if abs(beats[i] - t) <= SNAP:
            return float(beats[i])
    return float(t)


def _space(cands: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep the strongest candidates at least MIN_SPACING apart."""
    kept: list[tuple[float, float]] = []
    for t, s in sorted(cands, key=lambda c: -c[1]):
        if all(abs(t - k) >= MIN_SPACING for k, _ in kept):
            kept.append((t, s))
        if len(kept) >= MAX_EACH:
            break
    return sorted(kept)


def find_hits(y: np.ndarray, beats: list[float], segments: list[dict] | None = None,
              labels: list[str] | None = None) -> dict:
    import librosa
    fps = SR / HOP
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    db = librosa.amplitude_to_db(np.maximum(rms, 1e-5), ref=1.0)
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    bass = librosa.amplitude_to_db(np.maximum(spec[freqs <= BASS_HZ].mean(axis=0), 1e-6), ref=1.0)
    m = min(len(db), len(bass), len(onset))
    db, bass, onset = db[:m], bass[:m], onset[:m]
    peaks = librosa.onset.onset_detect(onset_envelope=onset, sr=SR, hop_length=HOP, backtrack=False, units="frames")
    beats = np.asarray(beats)
    n = len(db)
    pre, post, sus = int(PRE * fps), int(POST * fps), int(SUSTAIN * fps)
    loud_ref = np.percentile(db, 60)
    on_ref = np.percentile(onset, 99) + 1e-6

    hits, stops = [], []
    for p in peaks:
        if p - pre < 0 or p + sus >= n:
            continue
        before = db[p - pre:p].mean()
        after = db[p:p + post].mean()
        held = db[p:p + sus].mean()
        jump = after - before
        bjump = bass[p:p + post].mean() - bass[p - pre:p].mean()
        bass_held = bass[p:p + sus].mean() >= np.percentile(bass, 60)
        if (jump >= HIT_MIN_JUMP_DB and held >= loud_ref) or (bjump >= BASS_MIN_JUMP_DB and bass_held):
            strength = max(jump, 0.75 * bjump) * (0.5 + 0.5 * min(1.0, onset[p] / on_ref))
            hits.append((_snap(p / fps, beats), round(float(strength), 2)))
    # Sections the structure analysis labelled as drops: their strongest onset near the start is a hit.
    if segments and labels:
        for seg, lab in zip(segments, labels):
            if lab != "drop":
                continue
            a, b = int((seg["start"] - 0.3) * fps), int((seg["start"] + 0.6) * fps)
            a, b = max(0, a), min(n, b)
            if b > a:
                p = a + int(np.argmax(onset[a:b]))
                hits.append((_snap(p / fps, beats), 10.0))
    # stops: scan the loudness curve directly (a cut-out has no onset of its own)
    step = int(0.05 * fps)
    for p in range(pre, n - post, step):
        before = db[p - pre:p].mean()
        after = db[p:p + int(0.3 * fps)].mean()
        drop = before - after
        if drop >= STOP_MIN_DROP_DB and before >= loud_ref:
            stops.append((_snap(p / fps, beats), round(float(drop), 2)))
    return {
        "hits": [{"t": round(t, 3), "strength": s} for t, s in _space(hits)],
        "stops": [{"t": round(t, 3), "drop": s} for t, s in _space(stops)],
    }
