"""State-machine director, modelled on how Darktide's own soundtrack works (decoded from its banks):
one piece per mission, a fixed set of states, each state served by its own cuts of that piece,
and one fixed kind of transition per pair of states. Fights start fast (next beat, short fade);
they end on a phrase through an outro. Few decisions, so nothing happens without a reason.

States (from the game intent):  calm · build · fight · horde · boss
Each state draws from the sections you labelled in the Song Map (POOL). The song only changes
when it has nothing for the new state, or in a calm stretch after it has played a long time.

Outside missions (hub, menus, loadout, Valkyrie, victory…) the original Director runs unchanged.
"""
from __future__ import annotations

import os
import random

import numpy as np

from .director import Cue, Director, Tuning
from .library import Library, Segment, Track
from .ratings import BAN, LIKE

POOL = {   # Song Map roles that serve each state, best first
    "calm": ("explore", "bridge"),
    "build": ("build",),
    "fight": ("skirmish", "horde"),
    "horde": ("horde", "peak"),
    "boss": ("peak", "horde"),
}
RANK = {"calm": 0, "build": 1, "fight": 2, "horde": 3, "boss": 4}
ROLE_RANK = {"explore": 0, "bridge": 0, "build": 1, "skirmish": 2, "horde": 3, "peak": 4}
LOUD_BRIDGE = float(os.environ.get("CONDUCTOR_LOUD_BRIDGE", 0.45))      # 2026-11-20: bridges this loud were marked "wrong for calm" (Mona Lisa, VLST...)


def serves(seg, st: str) -> bool:
    """Is this section the right kind of music for a state? Loud bridges don't count as calm."""
    r = role_of(seg)
    if r not in POOL[st]:
        return False
    return not (st == "calm" and r == "bridge" and seg.energy >= LOUD_BRIDGE)


MIN_LEN = 6.0           # never land on a sliver of a section
SONG_TIRED = 420.0
MIN_SONG = float(os.environ.get("CONDUCTOR_MIN_SONG", 150.0))   # user 2026-11-22: 44 song changes a session = 2x too many
UP_HOLD = float(os.environ.get("CONDUCTOR_UP_HOLD", 15.0))       # s in a state before a step up (not boss/fight start)
DOWN_WAIT = float(os.environ.get("CONDUCTOR_DOWN_WAIT", 8.0))    # s the lower state must hold before stepping down
DOWN_HOLD = float(os.environ.get("CONDUCTOR_DOWN_HOLD", 20.0))   # s in a state before a step down between fight intensities
CALM_WAIT = float(os.environ.get("CONDUCTOR_CALM_WAIT", 3.0))    # s of calm before leaving fight music
_SPAN_CACHE: dict = {}
_GRID: dict = {}


def _period_at(track, t):
    from .tempo import period_at
    return period_at(track, t)


def _grid_score(track) -> float:
    """How well the song's detected beats sit on its drum hits vs the offbeats (cache/beat_audit.json)."""
    if not _GRID:
        import json, os
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "beat_audit.json")
        try:
            _GRID.update({r["title"]: r["beat_vs_offbeat"] for r in json.load(open(p, encoding="utf-8"))})
        except (OSError, ValueError):
            _GRID["_"] = 0
    return _GRID.get(track.title, 2.0)      # after this long, a calm stretch may bring in a fresh song


def state_of(intent) -> str | None:
    if intent is None or not intent.dynamic:
        return None
    if intent.mode != "combat":
        return "calm"
    return {2: "horde", 3: "boss"}.get(intent.tier, "fight")


def role_of(seg: Segment) -> str | None:
    """The Song Map role of a section (or a best guess for unlabelled ones)."""
    if seg.skip:
        return None
    if seg.bridge:
        return "bridge"
    r = getattr(seg, "user_role", None)
    if r:
        return None if r == "skip" else r
    if seg.role == "build":
        return "build"
    if seg.role == "combat":
        return "peak" if seg.energy >= 0.85 else "horde" if seg.energy >= 0.7 else "skirmish"
    return "explore"


class StateDirector:
    def __init__(self, library: Library, tuning: Tuning | None = None, seed: int | None = None,
                 ratings=None):
        self.legacy = Director(library, tuning, seed=seed, ratings=ratings)
        self.ratings = ratings
        self.rng = random.Random(seed)
        self.state: str | None = None
        self.state_since = 0.0
        self._cand: tuple[str, float] | None = None
        self._busy_until = 0.0
        self._repeats = 0
        self._skip = False
        self.used: dict[tuple[str, float], float] = {}

    # the app talks to these
    @property
    def lib(self):
        return self.legacy.lib

    @lib.setter
    def lib(self, v):
        self.legacy.lib = v

    @property
    def tune(self):
        return self.legacy.tune

    @property
    def track(self) -> Track | None:
        return self.legacy.track

    def skip(self) -> None:
        self._skip = True
        self.legacy.skip()

    # ── main entry ─────────────────────────────────────────────────────────
    def tick(self, intent, now: float, position: float | None) -> Cue | None:
        cue = self._tick(intent, now, position)
        # 2026-11-23 in game: the critic heard only 12% of decisions because the lookahead ran only at the end
        # of _tick, which returns early through every hold after a change. Now: after any quiet tick.
        if cue is None and getattr(self, "livecritic", None) is not None and position is not None \
                and self.state is not None and self.legacy.track is not None:
            try:
                self._lookahead(now, position)
            except Exception as e:                       # listening ahead must never break the music
                print("live critic lookahead failed:", e, flush=True)
        return cue

    def _tick(self, intent, now: float, position: float | None) -> Cue | None:
        st = getattr(self, "forced_state", None) or state_of(intent)      # forced: standalone mode's buttons
        L = self.legacy
        if st is None or L.track is None or position is None:
            self.state = None
            return L.tick(intent, now, position)
        L._now, L._position, L._intent = now, position, intent
        if now < self._busy_until:
            return None
        if self._skip or (intent.scene != L.scene and intent.family != L.family
                          and now - getattr(self, "_mission_at", -1e9) > 120):
            # into the mission (or thumbs-down): choose the mission's song
            self._skip_was, self._skip = self._skip, False
            L.scene, L.family, L._skip_requested = intent.scene, intent.family, False
            self._mission_at = now
            keep = self._serves(L.track, st) and not self._skip_was      # user 2026-11-22: lobby song carries into the mission
            song = L.track if keep else self._choose_song(st, exclude=L.track)
            return self._enter(st, intent, now, position, force_song=song, why="st:mission")
        L.scene, L.family = intent.scene, intent.family
        if now < getattr(self, "_hold_until", 0.0) and st != "boss":
            if not (st == "horde" and now >= getattr(self, "_hold_strong_until", 0.0)):
                return None                # the part we just went to gets to play (a breakdown plays out)
        if self.state is None:
            return self._enter(st, intent, now, position, why="st:mission")
        if st != self.state:
            if self._cand is None or self._cand[0] != st:
                self._cand = (st, now)
            up = RANK[st] > RANK[self.state]
            # 2026-11-21 in game: the game flips fight<->horde every few seconds and the music followed
            # every flip (changes 4-7 s apart, "one bar of this, one bar of that"). Hysteresis:
            if st == "boss" or (up and self.state == "calm"):
                wait, hold = 0.0, 0.0           # a boss / a fight starting: move at once
            elif up:
                wait, hold = 0.0, UP_HOLD       # fight -> horde: once the current part has had its time
            elif st == "calm":
                wait, hold = CALM_WAIT, 0.0     # leaving the fight: responsive (user: engage/disengage can be quick)
            else:
                wait, hold = DOWN_WAIT, DOWN_HOLD   # between fight intensities: keep the thread going
            if now - self._cand[1] >= wait and now - self.state_since >= hold:
                return self._enter(st, intent, now, position)
        else:
            self._cand = None
        return self._continue(now, position)

    # ── pieces ─────────────────────────────────────────────────────────────
    @staticmethod
    def _span(track: Track, seg: Segment) -> tuple[float, float]:
        """Where the section actually has sound: songs fade out (and some parts drop to silence),
        so the usable end is the last second that is still audible."""
        key = (id(track), seg.start)
        cache = _SPAN_CACHE
        if key not in cache:
            e = track.energy
            a, b = int(seg.start), min(len(e), int(np.ceil(seg.end)))
            thr = max(0.12, 0.5 * float(np.median(e[a:b]))) if b > a else 0.12
            loud = [i for i in range(a, b) if e[i] >= thr]      # fading tails don't count as the part
            cache[key] = (max(seg.start, float(loud[0])) if loud else seg.start,
                          min(seg.end, float(loud[-1]) + 0.25) if loud else seg.start)   # leave before it fades
        return cache[key]

    def _ok(self, seg: Segment, st: str, track: Track | None = None) -> bool:
        r = role_of(seg)
        if r not in POOL[st]:      # (director keeps loud bridges for calm until A/B data backs serves())
            return False
        if track is not None and r != POOL[st][0] and any(
                role_of(x) == POOL[st][0] and self._span(track, x)[1] - self._span(track, x)[0] >= MIN_LEN
                for x in track.segments):
            return False                   # e.g. a skirmish never plays horde music if the song has skirmish parts
        if track is not None:
            a, b = self._span(track, seg)
            return b - a >= MIN_LEN
        return seg.length >= MIN_LEN

    STATE_ROLE_RANK = {"calm": 0, "fight": 2, "horde": 3, "boss": 4}
    FRESH = 180.0          # 2026-11-21 in-game: re-landing on a part heard <3 min ago = 56% 👎 (fresh: 29%)

    def _near(self, seg: Segment, st: str, track: Track) -> bool:
        """Close enough to keep playing: within one intensity step of the state, a real part (not
        near-silence). In-game 2026-11-21: jumping back to the 'exact' part every time it ended made
        the music loop its build-ups and drops constantly."""
        r = role_of(seg)
        if r is None or r not in ROLE_RANK:
            return False
        if abs(ROLE_RANK[r] - self.STATE_ROLE_RANK[st]) > 1 or (st != "calm" and r in ("explore", "bridge")):
            return False
        a, b = self._span(track, seg)
        return b - a >= MIN_LEN

    def _fresh(self, track: Track, seg: Segment, now: float) -> bool:
        u = self.used.get((track.path, seg.start))
        return u is None or now - u >= self.FRESH

    def _material(self, track: Track, st: str) -> float:
        """Seconds of music this song has for a state."""
        return sum(self._span(track, s)[1] - self._span(track, s)[0] for s in track.segments
                   if role_of(s) == POOL[st][0] and self._ok(s, st, track))     # the state's own kind of part

    def _serves(self, track: Track | None, st: str) -> bool:
        return track is not None and any(self._ok(s, st, track) for s in track.segments)

    def _choose_song(self, st: str, exclude: Track | None = None, near: Track | None = None) -> Track | None:
        """A song for this mission: must serve `st`; best if it serves every state (a whole piece)."""
        L = self.legacy
        best = None
        for t in L.lib.tracks:
            if t is exclude or not self._serves(t, st):
                continue
            if self.ratings is not None and self.ratings.get(t.path) == BAN:
                continue
            score = sum(min(60.0, self._material(t, s)) / 60.0 for s in RANK)   # enough music for every state
            if st == "boss":
                score += 2 * max([s.energy for s in t.segments if self._ok(s, "boss", t)] or [0])
            if self.ratings is not None and self.ratings.get(t.path) == LIKE:
                score += 0.5
            if t.path in L.recent:
                score -= 2.0 * (1 + list(L.recent)[::-1].index(t.path)) ** -0.5
            if near is not None and not L.tempo_ok(near, t):
                score -= 0.7
            score += self.rng.random() * 0.3
            if best is None or score > best[0]:
                best = (score, t)
        return best[1] if best else None

    def _pick(self, track: Track, st: str, now: float, prev: str | None, after: Segment | None = None) -> Segment | None:
        cands = [s for s in track.segments if self._ok(s, st, track)]
        if not cands:
            return None
        fresh = [s for s in cands if self._fresh(track, s, now) and s is not after]
        cands = fresh or cands               # 2026-11-21: parts not heard in the last 3 min first

        def cost(s):
            c = POOL[st].index(role_of(s)) * 1.0
            used = self.used.get((track.path, s.start))
            if used is not None and now - used < 240:
                c += 2.0 * (1 - (now - used) / 240)     # rotate through the song's fitting parts
            if after is not None and s is after:
                c += 1.5                                 # don't loop the part that just played
            if st == "calm" and prev in ("fight", "horde", "boss") and role_of(s) == "bridge":
                c -= 1.5                      # leave a fight through a breakdown (the outro)
            if after is not None and s.start >= after.end - 0.5:
                c -= 0.3 / (1 + (s.start - after.end) / 30)   # prefer carrying on forward
            return c + self.rng.random() * 0.1
        return min(cands, key=cost)

    def _enter(self, st, intent, now, pos, force_song=None, why=None) -> Cue | None:
        L = self.legacy
        prev, self.state, self.state_since, self._cand = self.state, st, now, None
        cur = L.track
        cur_seg = cur.segment_at(pos)
        up = prev is None or RANK[st] > RANK[prev]
        song = force_song
        if song is None:
            tired = st == "calm" and now - L.song_since >= SONG_TIRED
            song = cur if (self._serves(cur, st) and not tired) else (
                self._choose_song(st, exclude=cur if tired else None, near=cur) or cur)
        if song is cur and force_song is None:
            # already in a part that serves this state: nothing to do (no needless moves)
            r = role_of(cur_seg)
            if self._ok(cur_seg, st, cur) and cur_seg.end - pos > 2.0:
                print(f"⚑ game: {prev} → {st}  ·  no change: the current part already fits {st}", flush=True)
                return None
        seg = self._pick(song, st, now, prev, after=cur_seg if song is cur else None)
        if (seg is not None and song is cur and force_song is None and not self._fresh(song, seg, now)
                and now - L.song_since >= MIN_SONG):
            alt = self._choose_song(st, exclude=cur, near=cur)          # this song has nothing fresh left
            alt_seg = self._pick(alt, st, now, prev) if alt is not None else None
            if alt_seg is not None:
                song, seg = alt, alt_seg
        if seg is None:
            return None
        if why is None:
            why = ("st:boss" if st == "boss" else "st:build" if st == "build" else
                   "st:start" if up and prev in ("calm", "build") else "st:up" if up else
                   "st:calm" if st == "calm" else "st:down")
            if song is not cur:
                why += ":song"
        p_old = _period_at(cur, pos)
        # fixed transition per kind. Every move: cut on the beat INTO the exact section (no pre-roll
        # of the music before it); the old part rings out underneath for `fade` seconds.
        if prev is None or up:                             # cut on the beat; the old part rings one beat
            at, fade, style = cur.beat_after(pos + 0.05), p_old, "punch"
        else:                                              # easing off: on the next bar, a 1-bar crossfade
            at, style = cur.next_bar(pos + 0.05), "xfade"  # (new part from its exact start; old fades under it)
            fade = min(2.5, 4 * p_old)
        start, fade_in = self._land(song, seg), 0.02
        self._lead = None
        if prev is not None:
            # the song's own move first: up = ride a build-up into its drop; down = ride an intense
            # part into the calmer part the song itself follows it with. None in this song: another.
            down = not up
            lead = self._lead_in(cur if down and force_song is None else song, st, now, down=down)
            if lead is not None and down:
                # going down, only ride it if we're already in that passage; jumping to another
                # intense part to ride it down sounds like the fight starting again
                if force_song is None and lead[0] is cur_seg:
                    song = cur
                else:
                    lead = None
            if lead is None and force_song is None and not down:   # down: only the song already playing
                others = [t for t in L.lib.tracks if t is not song and self._serves(t, st)]
                self.rng.shuffle(others)
                for t in others[:8]:
                    lead = self._lead_in(t, st, now, down=down)
                    if lead is not None:
                        song = t
                        break
            if lead is not None and song is cur and lead[0] is cur_seg and lead[2].start - pos <= 9 * p_old:
                self._lead = (song, lead[0], lead[2])
                print(f"⚑ game: {prev} → {st}  ·  no change: the song is already building into its drop", flush=True)
                return None                                # already in that passage: let the song do it
            if lead is not None:
                self._lead = (song, lead[0], lead[2])
                start = lead[1]
                # a run-in, once started, finishes: no new move until its drop (a boss still can)
                lead_hold = now + (lead[2].start - start) + 0.5
                # ease into the build (cuts lost 0/6); into another song a 2-bar blend (1-bar lost 0/4)
                style, fade = "xfade", (min(2.5, 4 * p_old) if song is cur else min(4.5, 8 * p_old))
        # user 2026-11-19: no song change between fight states (fight/horde/boss) unless the song is worn
        combat = prev in ("fight", "horde", "boss") and st in ("fight", "horde", "boss")
        keep = (combat and self._serves(cur, st) and now - L.song_since < SONG_TIRED) or (
            force_song is None and song is cur and st != "boss" and now - L.song_since < MIN_SONG)
        if keep and song is not cur:
            # 2026-11-22 replay: planner told to stay found nothing, then the hand-made move switched song anyway
            s2 = self._pick(cur, st, now, prev, after=cur_seg)
            if s2 is not None:
                song, start, self._lead = cur, self._land(cur, s2), None
                style, fade = ("punch", p_old) if up else ("xfade", min(2.5, 4 * p_old))
        planned = self._plan(st, prev, now, pos, song, start, same_only=keep,
                             trigger=f"game: {prev} → {st}" + ("  (fight states: keep the song)" if keep else ""))
        if planned is not None:
            song, start, at, route, lead = planned
            self._lead = lead
            if lead is not None:
                lead_hold = now + (lead[2].start - start) + max(0.0, at - pos) + 0.5
        hit = None
        if self._lead is None and up and prev is not None and now - getattr(self, "_last_hit", -1e9) >= 20:
            imp = [c for c in L.callouts(cur) if c.get("type") == "impact"]
            if imp:
                hit = (max(imp, key=lambda c: c.get("score", 0))["file"], 0.0)
                self._last_hit = now
        cue = self._commit(song, start, fade, at, pos, now, why, style, fade_in, bridge=hit)
        if planned is not None:
            cue.route = planned[3]
        if self._lead is not None:
            self._hold_until = max(self._hold_until, lead_hold)
            self._hold_strong_until = max(self._hold_strong_until, lead_hold)
        return cue

    def _lead_in(self, track: Track, st: str, now: float, down: bool = False):
        """A passage where the SONG ITSELF makes this move, entered on its last 2 bars so the change
        begins now and the song carries it: going up, a build-up into a drop; going down, an
        intense part flowing into a calmer one (your A/B marks: the song's own move 18/18)."""
        segs, best = track.segments, None
        for i, s in enumerate(segs[:-1]):
            nxt = segs[i + 1]
            r = role_of(s)
            src = (ROLE_RANK.get(r, 0) > RANK[st] and r not in ("build", "bridge")) if down else r == "build"     # a breakdown first = quieter as the fight starts
            if not src or not self._ok(nxt, st, track):
                continue
            p = track._period()
            want = nxt.start - (8 * p if 8 * p <= 4.5 else 4 * p)   # 2 bars before the drop (1 if slow)
            bars = [b for b in (track.downbeats or track.beats[::4]) if s.start - 0.1 <= b <= nxt.start - 2 * p]
            if not bars:
                continue
            start = min(bars, key=lambda b: abs(b - want))
            if min(track.energy_at(t) for t in np.arange(start, nxt.start + 3, 0.5)) < 0.25 * track.mean_energy:
                continue                                   # goes (near) silent in there: not a carry
            used = self.used.get((track.path, nxt.start))
            cost = (0 if role_of(s) == "build" or down else 0.5) + (POOL[st].index(role_of(nxt)))                 + (1.0 if used is not None and now - used < 240 else 0) + abs(start - want) / 4
            if best is None or cost < best[0]:
                best = (cost, s, start, nxt)
        return None if best is None else (best[1], best[2], best[3])

    @staticmethod
    def _land(track: Track, seg: Segment) -> float:
        bars = track.downbeats or track.beats[::4]
        a = StateDirector._span(track, seg)[0]     # skip silence at the top of the section
        for b in bars:
            if b >= max(seg.start, a - 0.5) - 0.12:
                return b if b < seg.start + 0.6 * seg.length else seg.start
        return seg.start

    def _continue(self, now, pos) -> Cue | None:
        """Inside a state: let the song play; just before a section that doesn't serve the state
        (or the song's end), jump on the bar line to another part that does."""
        L = self.legacy
        cur = L.track
        seg = cur.segment_at(pos)
        ld = getattr(self, "_lead", None)
        if ld is not None:
            if ld[0] is cur and seg is ld[1]:
                return None                                # the build-up is flowing into its drop
            self._lead = None
        idx = cur.segments.index(seg)
        nxt = cur.segments[idx + 1] if idx + 1 < len(cur.segments) else None
        lead = cur._period() + 0.35            # decide at the last beat, so a state change can get in first
        a_end = self._span(cur, seg)[1]           # where the sound actually stops (fade-outs, drop-outs)
        gap = seg.end - a_end > 0.75
        if a_end - pos > lead:
            if self._ok(seg, self.state, cur) or self._near(seg, self.state, cur) or a_end - pos < 1.0:
                return None
            # in a part that doesn't serve (e.g. landed before a state change settled): move on the bar
            dest = self._pick(cur, self.state, now, self.state, after=seg)
            if dest is None or dest is seg:
                return None
            return self._commit_planned(cur, self._land(cur, dest), cur._period(), cur.next_bar(pos + 0.05), pos, now,
                                        "st:next", "punch", 0.02, same_only=True)
        if nxt is not None and not gap and (self._ok(nxt, self.state, cur) or self._near(nxt, self.state, cur)):
            return None                                    # the song carries on into a fitting (or close) part
        dest = self._pick(cur, self.state, now, self.state, after=seg)
        song = cur
        self._repeats = self._repeats + 1 if dest is seg else 0
        worn = dest is seg and self._repeats >= 2 and self._material(cur, self.state) < 45
        # user 2026-11-19: a song change inside a fight is jarring however well it's joined - repeat
        # the song's own fitting parts instead, unless it has been on far too long
        # user 2026-11-21 (in game): but looping the same parts is worse - only FRESH parts of this song
        stay = self.state != "calm" and now - L.song_since < SONG_TIRED
        stale = dest is not None and (dest is seg or not self._fresh(cur, dest, now))
        if stale and now - L.song_since < MIN_SONG and dest is not None and (dest is not seg or seg.length >= 12):
            stale = False                                  # young song: repeat one of its parts, don't switch
        if dest is None or stale or (not stay and (worn or (dest is seg and seg.length < 12))):
            song = self._choose_song(self.state, exclude=cur, near=cur)
            if song is None:
                return None
            dest = self._pick(song, self.state, now, self.state)
            if dest is None:
                return None
        at = cur.beat_after(min(a_end, seg.end) - 0.03)
        return self._commit_planned(song, self._land(song, dest), cur._period(), at, pos, now,
                                    "st:next" if song is cur else "st:next:song", "punch", 0.02, same_only=song is cur)

    @staticmethod
    def _say_route(r, cur_title) -> str:
        from .feed import JOIN_NAMES
        st = r["steps"]
        fin = st[-1]["b"]
        where = "same song" if fin == cur_title else fin[:34]
        how = JOIN_NAMES.get(st[0]["style"], st[0]["style"])
        if len(st) == 2:
            return f"{where} · into its build-up @{st[0]['t_in']:.0f}s, ride the drop @{st[1]['t_in']:.0f}s · {how}"
        return f"{where} @{st[0]['t_in']:.0f}s · {how}"

    # ── learned planning (2026-11-20): the live game makes changes exactly the way the A/B sets do ──
    def _plan(self, st, prev, now, pos, song, start, same_only, trigger=""):
        """All the ways the A/B sets would make this change (routes.natural_routes + candidates:
        the song's own build-ups and drops, tempo/key-matched songs, every joining style), scored
        by the latest trained taste model; the best one is played. None = keep the hand-made move.
        Returns (track, start, at, route, lead)."""
        import os as _os
        if _os.environ.get("CONDUCTOR_PLAN", "1") == "0":
            return None
        try:
            from . import ab, routes
            L = self.legacy
            cur = L.track
            # decide at a bar far enough ahead to build the join (another song may need decoding)
            at = cur.next_bar(pos + (0.8 if same_only else 1.6))
            pool, ok = self._candidates(st, prev, now, song, start, same_only, at)
            by = self._by
            head = (f"⚑ DECIDE  {trigger or (str(prev) + ' → ' + st)}  ·  now: {cur.title[:30]} @{pos:.0f}s"
                    f"  ·  {len(pool)} ideas, {len(ok)} playable" + ("  ·  must stay in this song" if same_only else ""))
            if not ok:
                print(head + "  ·  nothing playable → hand-made move", flush=True)
                return None
            sc = ab.taste_many(by, ok)
            k = int(np.argmax(sc))
            top = sorted(range(len(ok)), key=lambda i: -sc[i])[:3]
            # 2026-11-23: the live critic LISTENED to some of these ahead of time (critic + style ~68% vs
            # trees ~63% on your A/B picks). If it has heard at least 2 of the trees' top 4, it decides.
            by_who, heard = "trees", {}
            lc = getattr(self, "livecritic", None)
            if lc is not None:
                top4 = sorted(range(len(ok)), key=lambda i: -sc[i])[:4]
                from .livecritic import key as _lkey
                sent = sum(_lkey(ok[i]) in lc.sent_keys for i in top4)
                lc.stats["top4_total"] += len(top4)
                lc.stats["top4_sent_before"] += sent
                heard = {i: lc.score(ok[i]) for i in top4}
                heard = {i: v for i, v in heard.items() if v is not None}
                if len(heard) >= 2:
                    k = max(heard, key=heard.get)
                    by_who = f"critic ({len(heard)}/{len(top4)} heard)"
                    lc.stats["used"] += 1
                else:
                    lc.stats["missed"] += 1
                    # why not: never asked about these (lookahead guessed other options) vs asked but not answered yet
                    by_who = f"trees (critic heard {len(heard)}/{len(top4)}, asked about {sent}/{len(top4)}" \
                             f"{', still busy' if lc.busy() else ''})"
            print(head + f"  ·  decided by {by_who}", flush=True)
            for rank, i in enumerate(top, 1):
                ear = f"  ear {heard[i]:+.2f}" if i in heard else ""
                print(f"    {'▶' if i == k else ' '} {rank}. {sc[i]:.2f}{ear}  {self._say_route(ok[i], cur.title)}", flush=True)
            if k not in top:
                print(f"    ▶ critic's pick {sc[k]:.2f}  ear {heard[k]:+.2f}  {self._say_route(ok[k], cur.title)}", flush=True)
            print(f"    plays on the bar at {at:.1f}s (in {max(0.0, at - pos):.1f}s)", flush=True)
            r = ok[k]
            # 2026-11-26 (user: "the multi-layer blends aren't working"): the top pick alone almost never had
            # two lockable beats. Try the top 3 for a beat-locked blend; failing that, the pick as a breakdown
            # (the beats never overlap, so it doesn't need them to lock).
            for i in [k] + [i for i in top if i != k]:
                up = self._layered(ok[i], cur, pos, now)
                if up is not None:                   # 2026-11-24: the A/B winners, live
                    return up
            up = self._layered(r, cur, pos, now, unlocked=True)
            if up is not None:
                return up
            s0, T = r["steps"][0], by[r["steps"][-1]["b"]]
            lead = None
            if len(r["steps"]) == 2:
                dur = s0["until"] - s0["t_in"]
                lead = (T, T.segment_at(s0["t_in"] + 0.05), T.segment_at(r["steps"][1]["t_in"] + 0.05))
            elif s0["style"].startswith("xfade"):
                dur = 4 * T._period() * int(s0["style"][-1]) + 0.05
            elif s0["style"] in ("mix", "dj", "djverb", "djblend"):
                dur = 6.0                       # the A/B clips' length after a one-step change
            else:
                dur = 4 * T._period()
            if not 0.3 <= dur <= 20:
                return None
            route = {"style": s0["style"], "dur": float(dur), "lead": lead is not None, "fallback": "punch",
                     "score": round(float(sc[k]), 3), "of": len(ok), "plan": routes.describe(r), "steps": r["steps"],
                     "by": by_who}
            self._route_log = getattr(self, "_route_log", []) + [(s0["style"], T is cur, lead is not None)]
            return T, float(s0["t_in"]), float(at), route, lead
        except Exception as e:
            print("planner failed, hand-made move:", e)
            return None

    # A/B results 2026-11-24 (28 sets): bass carry 75 %, rhythm first 70 %, drum bridge 62 % of head-to-heads
    # won vs the quick 2-bar blend's 26 %. Weighted pick so none becomes the only one heard.
    LAYERED = (("basscarry", 0.37), ("rhyfirst", 0.35), ("drumbridge", 0.28))
    LAYER_BARS = 4                     # ~8 s: the change starts now and lands 4 bars later, on the bar

    # no beat lock: A's drums+bass leave over bar 1-2, the melodies cross (B's from bar 2, A's out by bar 4),
    # B's beat lands on the landing. One hook bar at bar 3 (vocal guard applies). Beats never sound together.
    UNLOCKED = "gen4_113442111001h31t0"

    def _layered(self, r, cur, pos, now, unlocked=False):
        """Upgrade the planner's chosen change to a layered blend when it can be done properly: one step,
        both beat grids steady and drum-confirmed over the blend, tempos within a pitch-fader nudge, and
        time to build it. Starts on the next bar ~4 s out; lands on the planner's chosen section start.
        Returns the _plan tuple, or None to keep the normal join."""
        import os as _os
        if _os.environ.get("CONDUCTOR_LAYERED", "1") == "0" or len(r["steps"]) != 1:
            return None
        from . import routes, grid
        by = self._by
        s0 = r["steps"][0]
        B = by.get(s0["b"])
        if B is None:
            return None
        if not hasattr(self, "_lrng"):
            import random as _r
            self._lrng = _r.Random(23)            # its own dice: never shifts the director's choices
        names, w = zip(*self.LAYERED)
        style = self.UNLOCKED if unlocked else f"{self._lrng.choices(names, w)[0]}{self.LAYER_BARS}"
        t_land = float(s0["t_in"])
        # 2026-11-27: the plan model (learned from your picks and tweaks: which layer plays when) chooses the
        # recipe among the hand-made ones and a dozen freshly generated ones; the dice above if it isn't trained
        try:
            from . import ab, planmodel
            pool = ([self.UNLOCKED] if unlocked else [f"{n}{self.LAYER_BARS}" for n in names])
            pool += [ab._gen_sample(self._lrng, not unlocked) for _ in range(12)]
            drop = routes.lands_on_drop(B, t_land)
            ctx = {"locked": float(not unlocked), "bridge": float(unlocked), "glide": float(routes.STRETCH == "glide")}
            scored = []
            for st in dict.fromkeys(pool):
                routes.register_gen(st)
                v = planmodel.score({"a": cur.title, "steps": [{"b": B.title, "style": st}]}, routes.layer_plan(st, drop), ctx)
                if v is not None:
                    scored.append((v, st))
            if scored:
                v, style = max(scored)
                print(f"    plan model: {style} ({v:+.2f}, best of {len(scored)} recipes)", flush=True)
        except Exception as e:
            print("    plan model skipped:", e, flush=True)
        t_in = routes.lead_start(B, t_land, style)
        at = cur.next_bar(pos + 4.0)               # time to load stems and build the blend
        if t_in is None or t_in < 0 or at - pos > 7.0:
            return None
        blend = routes._bars_len(B, t_in, routes.LEAD_STYLES.get(style, self.LAYER_BARS))
        why = None
        if unlocked:
            pass
        elif not grid.ok(cur, at, at + blend * 1.1):
            why = "old song's beat grid not steady here"
        elif not grid.ok(B, t_in, t_land + 2.0):
            why = "new song's beat grid not steady here"
        elif not routes.tempo_match(cur, at + 1, B, t_in + 1):      # 2026-11-25: >2 % apart was good 23 % of the time
            why = "tempos don't meet (within 2 %)"
        elif grid.warp_positions(cur, at, B, t_in, int(blend * routes.SR), routes.SR) is None:
            why = "tempos too far apart"
        if why:
            print(f"    layered blend skipped: {why} - normal join", flush=True)
            return None
        if unlocked and B is cur and abs(t_in - at) < blend:
            return None                              # same song, overlapping itself: not a blend
        dur = blend + 8 * B._period()
        route = {"style": style, "dur": float(dur), "lead": False, "fallback": "punch", "layered": True,
                 "land": t_land, "plan": routes.describe(r), "by": "layered (A/B winners)",
                 "steps": [{"b": B.title, "t_in": t_in, "bars": None, "style": style, "land": t_land}]}
        # hold everything else until it has landed and settled
        self._layer_hold = now + (at - pos) + blend + 3.0
        print(f"    ▶ LAYERED {style}: starts on the bar at {at:.1f}s, lands {B.title[:30]} @{t_land:.0f}s "
              f"{blend:.1f}s later", flush=True)
        return B, float(t_in), float(at), route, None

    def _candidates(self, st, prev, now, song, start, same_only, at, rng=None):
        """Every playable way to make this change at bar `at` (the planner's pool, filtered).
        rng: the lookahead passes its own, so listening ahead never shifts the real decisions."""
        from . import routes
        L = self.legacy
        cur = L.track
        if getattr(self, "_by_lib", None) is not L.lib:        # playlist switched: re-index
            self._by, self._by_lib = {t.title: t for t in L.lib.tracks}, L.lib
        if not hasattr(self, "_plan_rng"):
            import random as _r
            self._plan_rng = _r.Random(23)
        by, rng = self._by, rng or self._plan_rng
        if True:
            j = {"a": cur.title, "t_out": at, "b": song.title, "t_in": start}
            down = prev is not None and RANK.get(prev, 0) > RANK[st]
            pool = routes.natural_routes(L.lib, j, st, down, rng) + routes.candidates(L.lib, j, st, rng)
            ok = []
            for r in pool:
                steps = r["steps"]
                fin = steps[-1]["b"]
                if any(x["b"] != fin for x in steps) or any(x["style"] in ("riser", "filter", "drop") for x in steps):
                    continue                    # the live player builds joins into one destination song
                if (st == "boss" or prev == st) and any(x["style"] in ("dj", "djverb", "djblend", "xfade2") for x in steps):
                    continue                    # boss / part-ending: dj + xfade2 joins rated 9/9 bad in game (2026-11-22)
                if len(steps) > 1 and (len(steps) != 2 or steps[1]["style"] != "natural"):
                    continue
                if len(steps) == 2 and (prev == st or (prev is not None and RANK.get(prev, 0) >= RANK[st])):
                    continue                    # build-up -> drop only when the intensity rises (in-game 2026-11-21)
                land_seg = by[fin].segment_at(steps[-1]["t_in"] + 0.05)
                if fin == cur.title and not self._fresh(cur, land_seg, now):
                    continue                    # never loop back into a part heard <3 min ago
                if same_only and fin != cur.title:
                    continue
                if self.ratings is not None and self.ratings.get(by[fin].path) == BAN:
                    continue
                if steps[0]["t_in"] < 0.5:
                    continue
                land = by[fin].segment_at(steps[-1]["t_in"] + 0.05)
                if routes.foreign_wind_down(cur, by[fin], steps[-1]["t_in"], down or st == "calm" and prev != "calm"):
                    continue                    # user 2026-11-24: not another song's breakdown/ending on the way down
                if not self._ok(land, st, by[fin]):
                    continue                    # somewhere the director would leave again at once
                ok.append(r)
        return pool, ok

    # ── live critic lookahead (2026-11-23): listen to the likely next changes BEFORE they're needed ──
    LOOK_EVERY = 3.0
    NEXT_STATES = {"calm": ("fight",), "build": ("fight", "calm"), "fight": ("horde", "calm"), "horde": ("fight",),
                   "boss": ()}

    def _lookahead(self, now, pos):
        """Throttled, cheap on the music loop: builds candidate pools for (1) the end of the current part and
        (2) the likeliest state change from here, and sends the trees' top 4 of each to the critic process.
        Skipped in horde / boss (keep the GPU for the game) and while the critic is still busy."""
        lc = getattr(self, "livecritic", None)
        if lc is None or not lc.ready or lc.busy() or self.state is None:
            return
        # horde / boss: only the part-ending listen (one short burst per part), no state-change guesses
        self._look_light = self.state in ("horde", "boss")
        if now - getattr(self, "_look_at", -1e9) < self.LOOK_EVERY:
            return
        self._look_at = now
        if not hasattr(self, "_look_rng"):
            import random as _r
            self._look_rng = _r.Random(31)
        keep_rng, self.rng = self.rng, self._look_rng          # _pick's dice: the director's own stay untouched
        try:
            self._lookahead_jobs(now, pos, lc)
        finally:
            self.rng = keep_rng

    def _lookahead_jobs(self, now, pos, lc):
        from . import ab
        L = self.legacy
        cur = L.track
        seg = cur.segment_at(pos)
        a_end = self._span(cur, seg)[1]
        jobs = []
        # (1) part ending: what _continue will ask for at the end of this part (same song first)
        if 2.5 < a_end - pos < 30 and getattr(self, "_look_seg", None) != (id(cur), seg.start):
            self._look_seg = (id(cur), seg.start)
            dest = self._pick(cur, self.state, now, self.state, after=seg)
            if dest is not None:
                jobs.append((self.state, self.state, cur, self._land(cur, dest), True,
                             cur.beat_after(min(a_end, seg.end) - 0.03)))
        # (2) the likeliest state change, decided "now" (refreshed every 20 s as the song moves on)
        elif now - getattr(self, "_look_state_at", -1e9) > (40 if getattr(self, "_look_light", False) else 20) \
                and self.state != "boss":
            self._look_state_at = now
            for nxt in self.NEXT_STATES.get(self.state, ()):
                dest = self._pick(cur, nxt, now, self.state, after=seg)
                start = self._land(cur, dest) if dest is not None else pos
                jobs.append((nxt, self.state, cur, start, False, cur.next_bar(pos + 1.6)))
        music = getattr(self, "music", None)
        for st, prev, song, start, same_only, at in jobs:
            pool, ok = self._candidates(st, prev, now, song, start, same_only, at, rng=self._look_rng)
            if len(ok) < 2:
                continue
            sc = ab.taste_many(self._by, ok)
            top4 = [ok[i] for i in sorted(range(len(ok)), key=lambda i: -sc[i])[:4]]
            if music:
                lc.submit(music, top4)

    def _commit_planned(self, song, start, fade, at, pos, now, reason, style, fade_in, same_only):
        planned = self._plan(self.state, self.state, now, pos, song, start, same_only,
                             trigger=f"still {self.state}: this part is ending / doesn't fit" + ("" if same_only else ", song worn out"))
        if planned is None:
            return self._commit(song, start, fade, at, pos, now, reason, style, fade_in)
        T, start, at, route, lead = planned
        self._lead = lead
        cue = self._commit(T, start, fade, at, pos, now, reason if T is song else "st:next:song", style, fade_in)
        cue.route = route
        if lead is not None:
            hold = now + (lead[2].start - start) + max(0.0, at - pos) + 0.5
            self._hold_until = max(self._hold_until, hold)
        return cue

    def _commit(self, track, start, fade, at, pos, now, reason, style, fade_in, bridge=None) -> Cue:
        L = self.legacy
        prev_track = L.track
        delay = max(0.0, at - pos)
        if style == "blend":
            at = max(at, pos + fade)                     # the crossfade finishes on the beat
            delay = at - pos
        if track is not L.track:
            L.recent.append(track.path)
            L.song_since = now
        L.track, L.last_change, L.loop, L.loop_seg, L.gap_since = track, now, None, None, None
        L._pending_switch = None
        seg = track.segment_at(start + 0.05)
        self.used[(track.path, seg.start)] = now
        self._busy_until = now + delay + 0.2
        # how long this part gets before anything else may change it (heard from `at`)
        a, b = self._span(track, seg)
        two_bars = 8 * track._period()
        ld = getattr(self, "_lead", None)
        if ld is not None and ld[0] is track and ld[1] is seg:
            dwell = (ld[2].start - start) + min(3.0, two_bars)        # through the build-up and into the drop
        elif role_of(seg) in ("bridge", "build") and self.state == "calm":
            dwell = max(two_bars, min(20.0, b - start))               # a breakdown out of a fight plays out
        elif role_of(seg) in ("bridge", "build"):
            dwell = max(two_bars, min(4.5, b - start))
        else:
            dwell = min(3.0, two_bars)
        self._hold_until = now + delay + dwell
        self._hold_strong_until = now + delay + min(dwell, two_bars)
        lh = getattr(self, "_layer_hold", 0.0)
        if lh > now:                                   # a layered blend: nothing else until it has landed
            self._hold_until = max(self._hold_until, lh)
            self._hold_strong_until = max(self._hold_strong_until, lh)
            self._layer_hold = 0.0
        route = self._route(prev_track, track, start, seg, style)
        return Cue(track, start, max(0.05, fade), delay, reason + (":callout" if bridge else ""), at=at,
                   style=style, fade_in=fade_in, bridge=bridge, route=route)

    def _route(self, old, track, start, seg, style):
        """The learned joining style for this change (policy.json, rebuilt every training round).
        Into a build-up: blend until its drop, then the song's own drop. Straight into a section:
        a blend of the style's length. None = the hand-made style above."""
        from . import policy
        if old is None or style in ("seam", "sequence"):
            return None
        ld = getattr(self, "_lead", None)
        lead = ld is not None and ld[0] is track and ld[1] is seg
        if not hasattr(self, "_prng"):
            import random as _r
            self._prng = _r.Random(17)          # its own dice: never shifts the director's choices
        st = policy.choose(old is track, lead, self._prng)
        self._route_log = getattr(self, "_route_log", []) + [(st, old is track, lead)]
        if st is None:
            return None
        p = track._period()
        if lead:
            dur = ld[2].start - start                   # up to the drop
        elif st.startswith("xfade"):
            dur = 4 * p * int(st[-1]) + 0.05
        else:
            dur = 4 * p
        if dur < 0.5 or dur > 20:
            return None
        return {"style": st, "dur": float(dur), "lead": lead, "fallback": style}
