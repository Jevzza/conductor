"""The music library: analysed tracks with energy curves normalised across the whole library."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .analysis import load_or_analyze

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".opus"}

# How much each raw feature contributes to perceived energy.
WEIGHTS = {"rms_db": 0.45, "onset": 0.25, "centroid": 0.30}
# How much each feature contributes to "has a driving beat".
BEAT_WEIGHTS = {"pulse": 0.7, "onset": 0.3}
# With separated stems: judge the beat on the drums alone (not pads/vocals that happen to pulse).
DRUM_BEAT_WEIGHTS = {"drum_pulse": 0.45, "drum_presence": 0.35, "pulse": 0.2}

# Section roles (gameplay pools). Combat needs BOTH intensity and a driving beat.
COMBAT_MIN_ENERGY = 0.45
COMBAT_MIN_BEAT = 0.45
# Very intense sections (heavy half-time drops) qualify with a weaker steady pulse.
COMBAT_INTENSE_ENERGY = 0.7
COMBAT_INTENSE_MIN_BEAT = 0.25


@dataclass
class Segment:
    start: float
    end: float
    energy: float = 0.0      # mean 0..1
    slope: float = 0.0       # energy change across the segment
    label: str = "groove"    # calm | groove | build | peak | release | drop
    beat: float = 0.0        # 0..1 how driving/steady the beat is (library-relative)
    role: str = "explore"    # explore | build | combat
    loop: tuple[float, float] | None = None   # seamless loop region inside this section, if any
    entry_beats: float = 1.0   # how to come in: 0.25 hard (impact) … 8 ambient (long fade before the beat)
    exit_beats: float = 4.0    # how to leave: 1 dense (short tail) … 8 sparse (let it ring)
    timbre: np.ndarray | None = None   # unit "what it sounds like" vector (library z-scored)
    user_role: str | None = None       # from the Song Map: explore/build/skirmish/horde/peak/skip
    skip: bool = False                 # the user said: never use this section
    bridge: bool = False               # breakdown / bridge: used to get from one mood to another

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class Track:
    path: str
    title: str
    duration: float
    tempo: float
    beats: list[float]
    mode: str
    key: str
    energy: np.ndarray                       # per-second 0..1
    beat: np.ndarray = None                  # per-second 0..1
    segments: list[Segment] = field(default_factory=list)
    spotify_uri: str | None = None
    hits: list[tuple[float, float]] = field(default_factory=list)    # (time, strength): impacts to cut INTO
    stops: list[tuple[float, float]] = field(default_factory=list)   # (time, drop dB): cut-outs to cut OUT on
    jumps: list[tuple[float, float, float]] = field(default_factory=list)  # (from bar, to bar, 0..1 match)
    downbeats: list[float] = field(default_factory=list)   # real bar lines (Beat This!), if analysed
    downbeat_strength: list[float] = field(default_factory=list)  # drum hit strength at each downbeat
    phrases: list[float] = field(default_factory=list)    # 4-bar phrase boundaries (from section starts)
    phrases8: list[float] = field(default_factory=list)   # 8-bar phrase starts (incl. every section start)
    intensity: float = 0.0     # 0..1 how hard this song can go (library-relative rank)
    tier: int = 1              # 1 skirmish, 2 horde, 3 peak — the fights this song can carry

    def next_beat(self, t: float) -> float:
        for b in self.beats:
            if b > t:
                return b
        return t

    def _period(self) -> float:
        return 60.0 / self.tempo if self.tempo and self.tempo > 0 else 0.5

    def beat_after(self, t: float) -> float:
        """First beat at/after t. Where the detected grid has a gap (breakdowns), continue the
        last known beat at the song's tempo so the answer is always on the pulse."""
        p = self._period()
        prev = None
        for b in self.beats:
            if b >= t:
                if prev is not None and b - prev > 1.5 * p:
                    k = max(0, int(np.ceil((t - prev) / p - 1e-9)))
                    syn = prev + k * p
                    if syn < b - 0.5 * p:
                        return syn
                return b
            prev = b
        if prev is None:
            return t
        k = max(0, int(np.ceil((t - prev) / p - 1e-9)))
        return prev + k * p

    def snap_start(self, t: float, beats_per_bar: int = 4) -> float:
        """Nearest bar line to t if within half a bar, else the nearest beat."""
        if not self.beats:
            return t
        p = self._period()
        bars = self.downbeats or self.beats[::beats_per_bar]
        bar = min(bars, key=lambda b: abs(b - t))
        if abs(bar - t) <= 0.5 * beats_per_bar * p:
            return bar
        return min(self.beats, key=lambda b: abs(b - t))

    def next_phrase(self, t: float, max_wait: float = 9.0, grid: list[float] | None = None) -> float | None:
        for x in (grid if grid is not None else self.phrases):
            if x > t + 0.05:
                return x if x - t <= max_wait else None
        return None

    def phrase_start_in(self, seg: "Segment") -> float:
        """The phrase start to enter `seg` on: its own start (always a phrase start)."""
        return seg.start

    def on_phrase(self, t: float, tol: float = 0.12, grid: list[float] | None = None) -> bool:
        g = grid if grid is not None else self.phrases
        import bisect
        i = bisect.bisect_left(g, t - tol)
        return i < len(g) and abs(g[i] - t) <= tol

    def best_hit(self, a: float, b: float) -> float | None:
        """Strongest impact in [a, b], or None."""
        cands = [(s, t) for t, s in self.hits if a <= t <= b]
        return max(cands)[1] if cands else None

    @property
    def mean_energy(self) -> float:
        return float(self.energy.mean())

    def energy_at(self, t: float) -> float:
        i = int(min(max(t, 0), len(self.energy) - 1))
        return float(self.energy[i])

    def segment_at(self, t: float) -> Segment:
        for s in self.segments:
            if s.start <= t < s.end:
                return s
        return self.segments[-1]

    def next_bar(self, t: float, beats_per_bar: int = 4) -> float:
        """Time of the next bar line after t (falls back to t if no beat grid)."""
        if self.downbeats:
            for b in self.downbeats:
                if b > t:
                    return b
            return t
        if len(self.beats) < beats_per_bar:
            return t
        for i in range(0, len(self.beats), beats_per_bar):
            if self.beats[i] > t:
                return self.beats[i]
        return t


def scan(folder: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(folder):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                out.append(os.path.join(root, f))
    return out


def _role(seg: Segment) -> str:
    if seg.label == "build":
        return "build"
    if seg.energy >= COMBAT_MIN_ENERGY and seg.beat >= COMBAT_MIN_BEAT:
        return "combat"
    if seg.energy >= COMBAT_INTENSE_ENERGY and seg.beat >= COMBAT_INTENSE_MIN_BEAT:
        return "combat"
    return "explore"


def _ranked(raws: list[dict], weights: dict) -> list[np.ndarray]:
    """Weighted library-wide z-score of features, smoothed ~4 s, as a 0..1 percentile per second."""
    stats = {}
    for feat in weights:
        allv = np.concatenate([np.asarray(r["curve"][feat], dtype=float) for r in raws])
        stats[feat] = (allv.mean(), allv.std() or 1.0)
    scores = []
    for r in raws:
        s = sum(w * (np.asarray(r["curve"][f], dtype=float) - stats[f][0]) / stats[f][1]
                for f, w in weights.items())
        if len(s) >= 4:
            s = np.convolve(np.pad(s, (2, 1), mode="edge"), np.ones(4) / 4, mode="valid")
        scores.append(s)
    ranked = np.sort(np.concatenate(scores))
    return [np.searchsorted(ranked, s) / max(1, len(ranked) - 1) for s in scores]


def _label(seg: Segment, prev: Segment | None) -> str:
    if prev is not None and seg.energy - prev.energy > 0.25 and seg.energy > 0.6:
        return "drop"
    if seg.slope > 0.2:
        return "build"
    if seg.slope < -0.2:
        return "release"
    if seg.energy >= 0.7:
        return "peak"
    if seg.energy <= 0.35:
        return "calm"
    return "groove"


BUILD_RUN_RISE = 0.2     # total energy gain across a run of rising sections that counts as a build-up
BUILD_MIN_SLOPE = 0.05   # each section in the run must itself be rising


def _mark_multi_segment_builds(segs: list[Segment]) -> None:
    """A build-up often spans several detected sections, each rising a little. Find runs of
    consecutive rising sections whose total rise is large, and label all but the arrival
    section as 'build' (the arrival keeps its own label, e.g. drop/peak/groove)."""
    i = 0
    while i < len(segs) - 1:
        j = i
        while (j + 1 < len(segs) and segs[j].slope >= BUILD_MIN_SLOPE
               and segs[j + 1].energy > segs[j].energy):
            j += 1
        if j > i and segs[j].energy - segs[i].energy >= BUILD_RUN_RISE:
            for k in range(i, j):
                if segs[k].label in ("calm", "groove"):
                    segs[k].label = "build"
        i = max(j, i + 1)


def _phrase_grid(t: "Track") -> None:
    """Music is built in 4/8/16-bar phrases that start at section boundaries. Count bars from each
    section's first downbeat; every 4th bar line is a phrase boundary, every 8th a phrase start."""
    bars = t.downbeats or (t.beats[::4] if t.beats else [])
    if not bars:
        return
    b = np.asarray(bars)
    p4, p8 = set(), set()
    for seg in t.segments:
        k0 = int(np.argmin(np.abs(b - seg.start)))
        p4.add(round(float(seg.start), 3)); p8.add(round(float(seg.start), 3))
        k = k0
        while k < len(b) and b[k] < seg.end - 0.1:
            if (k - k0) % 4 == 0:
                p4.add(round(float(b[k]), 3))
            if (k - k0) % 8 == 0:
                p8.add(round(float(b[k]), 3))
            k += 1
    t.phrases = sorted(p4)
    t.phrases8 = sorted(p8)


def _rate_intensity(tracks: list["Track"]) -> None:
    """How intense each song can get: the loud end of its fight sections (energy and driving
    beat), ranked across the library, then split into tiers: bottom 35 % skirmish, middle 40 %
    horde, top 25 % peak."""
    raw = []
    for t in tracks:
        secs = [s for s in t.segments if s.role == "combat"] or t.segments
        e = np.concatenate([t.energy[int(s.start):max(int(s.start) + 1, int(s.end))] for s in secs]) \
            if len(t.energy) else np.zeros(1)
        b = np.concatenate([t.beat[int(s.start):max(int(s.start) + 1, int(s.end))] for s in secs]) \
            if t.beat is not None and len(t.beat) else np.zeros(1)
        fight_share = sum(s.length for s in t.segments if s.role == "combat") / max(1.0, t.duration)
        raw.append(0.6 * float(np.percentile(e, 85)) + 0.25 * float(np.mean(b)) + 0.15 * fight_share)
    order = np.argsort(np.argsort(raw))
    n = max(1, len(tracks) - 1)
    for t, r in zip(tracks, order):
        t.intensity = float(r) / n
        t.tier = 1 if t.intensity < 0.35 else (2 if t.intensity < 0.75 else 3)


class Library:
    def __init__(self, tracks: list[Track]):
        self.tracks = tracks

    @classmethod
    def build(cls, folder: str, cache_dir: str, progress=print, backfill: bool = True,
              cached_only: bool = False, tempo_map: bool = True) -> "Library":
        """cached_only: use only songs fully processed by `conductor ingest` (instant; the app
        never analyses while you play)."""
        from .analysis import NotAnalysed
        raws = []
        paths = scan(folder)
        for i, p in enumerate(paths, 1):
            try:
                raws.append(load_or_analyze(p, cache_dir, backfill=backfill, cached_only=cached_only))
            except NotAnalysed:
                continue
                progress(f"[{i}/{len(paths)}] {os.path.basename(p)}")
            except Exception as e:  # unreadable file shouldn't kill the library
                progress(f"[{i}/{len(paths)}] SKIP {os.path.basename(p)}: {e}")
        lib = cls.from_analyses(raws)
        if not tempo_map:
            return lib
        try:                                   # beat grid that follows tempo changes inside songs
            from . import tempo
            if tempo.apply_all(lib, cache_dir, progress):
                for t in lib.tracks:
                    _phrase_grid(t)
        except Exception as e:
            progress(f"tempo map not applied: {e}")
        return lib

    @classmethod
    def from_analyses(cls, raws: list[dict]) -> "Library":
        if not raws:
            return cls([])
        energies = _ranked(raws, WEIGHTS)
        has_pulse = all("pulse" in r["curve"] for r in raws)
        has_drums = all("drum_pulse" in r["curve"] for r in raws)
        beats = _ranked(raws, DRUM_BEAT_WEIGHTS if has_drums else
                        BEAT_WEIGHTS if has_pulse else {"onset": 1.0})

        tv = [sd["timbre"] for r in raws for sd in r["segments"] if sd.get("timbre")]
        t_mu = np.mean(tv, axis=0) if tv else None
        t_sd = (np.std(tv, axis=0) + 1e-9) if tv else None
        tracks = []
        for r, energy, beat in zip(raws, energies, beats):
            segs, prev = [], None
            loops = r.get("loops") or [None] * len(r["segments"])
            for sd, lp in zip(r["segments"], loops):
                a, b = int(sd["start"]), max(int(sd["start"]) + 1, int(sd["end"]))
                e = energy[a:b]
                bt = beat[a:b]
                seg = Segment(sd["start"], sd["end"],
                              energy=float(e.mean()) if e.size else 0.0,
                              beat=float(bt.mean()) if bt.size else 0.0,
                              slope=float(e[-min(4, e.size):].mean() - e[:min(4, e.size)].mean()) if e.size else 0.0)
                if lp and lp.get("ok"):
                    seg.loop = (lp["start"], lp["end"])
                seg.entry_beats = sd.get("entry_beats", 1.0)
                if sd.get("timbre") and t_mu is not None:
                    z = (np.asarray(sd["timbre"]) - t_mu) / t_sd
                    seg.timbre = z / (np.linalg.norm(z) + 1e-9)
                seg.exit_beats = sd.get("exit_beats", 4.0)
                seg.label = _label(seg, prev)
                segs.append(seg)
                prev = seg
            _mark_multi_segment_builds(segs)
            for seg in segs:
                seg.role = _role(seg)
            hits = [(h["t"], h["strength"]) for h in r.get("hits", [])]
            beats_arr = r["beats"]
            for seg in segs:                       # drop sections land on a hit even if quiet in dB terms
                if seg.label == "drop" and not any(abs(t - seg.start) < 1.0 for t, _ in hits):
                    near = min(beats_arr, key=lambda b: abs(b - seg.start)) if beats_arr else seg.start
                    hits.append((near if abs(near - seg.start) < 0.6 else seg.start, 10.0))
            tracks.append(Track(
                jumps=[tuple(j) for j in r.get("jumps", [])],
                hits=sorted(hits), stops=[(s["t"], s["drop"]) for s in r.get("stops", [])],
                path=r["path"], title=r["title"], duration=r["duration"], tempo=r["tempo"],
                beats=r["beats"], mode=r["mode"], key=r["key"], energy=energy, beat=beat, segments=segs,
                downbeats=r.get("downbeats", []), downbeat_strength=r.get("downbeat_strength", []),
            ))
        _rate_intensity(tracks)
        for t in tracks:
            _phrase_grid(t)
        lib = cls(tracks)
        try:
            from .rolemap import apply as apply_roles
            n = apply_roles(lib)
            if n:
                print(f"Song Map: {n} sections follow your labels")
        except Exception as e:
            print("Song Map not applied:", e)
        return lib

    def summary(self) -> str:
        lines = []
        for t in sorted(self.tracks, key=lambda t: t.mean_energy):
            labels = " ".join(f"{s.role}/{s.label}@{int(s.start)}s" for s in t.segments)
            lines.append(f"{t.mean_energy:.2f}  {t.tempo:5.1f}bpm  {t.key}{'m' if t.mode == 'minor' else ''}  {t.title}\n        {labels}")
        return "\n".join(lines)
