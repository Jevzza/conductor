"""The Director: decides what plays, given the game's Intent and the analysed Library.

This is game-agnostic. It mirrors how adaptive game scores behave:
  * a scene change (menu → mission, mission → victory) gets a new piece of music
  * inside a dynamic scene, the music follows the energy target, rising quickly
    and falling slowly, without thrashing
  * changes always land on the beat: bar lines normally, the next beat when urgent
  * it prefers moving within the current song (jump to its chorus / its breakdown)
    before swapping to another song, so the soundtrack feels continuous
  * rising tension pulls toward "build" sections; urgent spikes toward "drop"/"peak"
"""
from __future__ import annotations

import random

import numpy as np
from collections import deque
from dataclasses import dataclass

from .library import Library, Segment, Track
from .profile import Intent
from .ratings import BAN, LIKE, Ratings


SCENE_KEEP = True     # user 2026-11-22: screens change PARTS of the song, never the song (lobby -> Valkyrie -> mission)
SCENE_HOLD = 20.0     # seconds: a menu/transit scene change this soon after the last one keeps the song


@dataclass
class Cue:
    track: Track
    start: float          # seconds into the track (for action="loop": the loop start)
    fade: float           # crossfade duration
    delay: float          # wait this long (to hit a bar line) before starting
    reason: str
    action: str = "play"  # play | loop (hold the current section with a seamless loop, no restart)
    loop: tuple[float, float] | None = None   # loop region for action="loop"
    at: float | None = None   # switch exactly when the current song reaches this position (a beat)
    curve: str = "power"      # crossfade shape: "linear" for near-identical audio (jump points)
    style: str = "punch"      # seam | blend | punch | plain (see LocalPlayer)
    fade_in: float | None = None   # seconds the destination fades in BEFORE the beat (None: player default)
    ring: bool = False        # the old part has a hit right on the switch: let it ring out instead of ducking it
    layer: float | None = None   # action="layer": drums+bass level (1 = full song, lower = lull)
    bridge: tuple | None = None  # (call-out file, seconds): played so that it ENDS on the switch beat
    sync: tuple[float, float] | None = None   # catch-up: (old song time, new song time) that coincide; start now, in phase
    route: dict | None = None    # learned join (policy.py): {style, dur, fallback} built by routes._join

    def describe(self) -> str:
        if self.action == "loop":
            a, b = self.loop
            return f"{self.reason:<14} ⟲ {self.track.title} looping {a:.1f}-{b:.1f}s ({b - a:.1f}s)"
        seg = self.track.segment_at(self.start)
        join = f" join={self.route['style']}" if self.route else ""
        return (f"{self.reason:<14} → {self.track.title} @ {self.start:.0f}s "
                f"[{seg.role}/{seg.label} e={seg.energy:.2f}] fade={self.fade:.1f}s delay={self.delay:.2f}s{join}")


@dataclass
class Tuning:
    mismatch: float = 0.25       # energy gap that counts as "wrong music"
    rise_dwell: float = 2.0      # seconds a rising gap must persist before reacting
    fall_dwell: float = 1.5      # once the fight is really over (see profile combat_sustain), calm down quickly
    min_hold: float = 30.0       # minimum seconds between non-urgent changes
    energy_hold: float = 45.0    # …and between changes made only for energy (right pool already)
    min_runway: float = 75.0     # prefer entering a new song where at least this much music remains
    breather: float = 20.0       # in combat, a non-combat section this short that leads back into combat is fine
    scene_fade: float = 2.0
    energy_fade: float = 2.0
    urgent_fade: float = 0.8
    end_margin: float = 6.0      # start the next song this long before the current one ends
    history: int = 15            # avoid repeating the last N tracks
    repeat_penalty: float = 0.6
    stay_bias: float = 0.35      # how much worse another song must NOT be before we leave this one
    urgent_cooldown: float = 30.0  # urgent flags flicker; allow an urgent cut at most this often
    energy_rise_tau: float = 3.0   # smoothing of the energy target (seconds): heat up fast…
    energy_fall_tau: float = 2.5   # …and cool down fairly quickly
    cut_fade: float = 0.04       # a hard cut (onto a hit / out on a stop): just enough to avoid clicks
    exit_fade: float = 0.35      # fade used when cutting out of a fight on a stop/hit
    quick_fade: float = 1.0      # fallback fade leaving a fight when no stop/hit is coming
    exit_lookahead: float = 4.0  # leaving a fight: wait at most this long for a stop/hit to cut on
    entry_window: float = 20.0   # entering a fight: look for a hit this far into the destination section
    high_intensity: float = 0.75 # at/above this energy (or urgent) the track never changes
    winddown_time: float = 12.0  # after winding down within a song, wait this long before fading to a new track
    switch_fade: float = 4.0     # track changes (only at low intensity) are long, gentle crossfades
    fightless_song_cost: float = 0.35  # while exploring in a mission, avoid songs that have no combat sections
    jump_wait: float = 8.0       # in-song moves may wait this long for a good jump point (they rate best)
    jump_min: float = 0.7        # weakest jump point used (0.67 rated bad, 0.74 good)
    leave_wait: float = 6.0      # wait up to this long for a strong moment in the current song to leave on
    slam_wait: float = 4.0       # …when a fight starts (it must still feel prompt)
    tempo_tolerance: float = 0.03  # songs within 3 % tempo (or half/double) can crossfade
    tempo_cost: float = 1.0      # choosing a song with a clashing tempo (hard cuts rated poorly in round 4)
    slam_max_rise: float = 0.35  # a slam may not raise intensity more than this ("too much of a beat increase")
    sound_cost: float = 1.0      # changing song: how much a different-sounding destination costs
    song_change_min_sound: float = 0.25  # mid-mission song changes only to parts that sound alike
    song_blend_bars: float = 2.0 # those changes crossfade over this many bars (longer = smoother)
    tier_cost: float = 0.6       # per tier of mismatch between the song's intensity and the game's
    escalate_gap: float = 40.0   # at most one "the fight got bigger: bigger song" change this often
    deescalate_after: float = 25.0  # fight this much smaller than the song for this long: calmer song
    jump_min_sound: float = 0.3  # jump points only between sections that sound alike ("drop at the wrong point")
    sound_cost_inside: float = 0.4  # moving within a song to a very different-sounding part
    slam_leave_wait: float = 2.0 # fight starts: leave on the old song's next big beat, before it relaxes
    strong_beat: float = 0.5     # drum strength that counts as a big beat
    catchup: float = 0.6         # fight starts just after the old song's big beat: join as if we'd switched on it
    tail_max: float = 1.5        # punch transitions: longest the old part's bass-less tail lingers
    switch_tail: float = 2.0     # …when it's a different song
    like_bonus: float = 0.3      # how strongly a thumbs-up song is favoured among similar fits
    skip_fade: float = 1.5
    scene_settle: float = 0.0    # a new scene must hold this long before the music follows (menus flicker)
    max_loop_time: float = 45.0  # never hold one loop longer than this…
    min_loop_time: float = 20.0  # …but always allow at least this long
    loop_passes: float = 2.0     # …aiming for about this many passes through the loop
    section_memory: float = 300.0  # a section played within this many seconds is less attractive
    section_repeat_penalty: float = 0.9   # strong: don't keep cycling the same few sections
    fatigue_time: float = 210.0  # after this long in one song, move to another at the next strong moment
    bridge_every: float = 60.0   # steady for this long: a short detour to another fitting part of the song
    phrase_wait: float = 9.0     # in-song moves wait up to this long for a phrase boundary
    phrase_dwell_bars: int = 8   # after a move, stay at least this many bars (one phrase)
    layer_lull: float = 0.5      # lull: drums+bass down to this (vertical layering, no jump)
    calm_for_song_change: float = 6.0   # song changes only after this long without a real fight
    bridge_gap: float = 30.0     # at most one call-out bridge this often
    lull_gap: float = 0.15       # the fight dropped this far below what the current part is playing…
    lull_debounce: float = 1.5   # …for this long: move to a calmer part of this song
    lull_max_wait: float = 2.0   # and leave within this long (jump point or bar line), ~5 s in total
    min_song_time: float = 150.0 # strongly prefer staying in a song at least this long…
    max_song_time: float = 300.0 # …and freshen up with a new one after this long
    song_stay_bonus: float = 0.6


class Director:
    def __init__(self, library: Library, tuning: Tuning | None = None, seed: int | None = None,
                 ratings: Ratings | None = None):
        self.lib = library
        self.ratings = ratings
        self._skip_requested = False
        self.family: str | None = None
        self._pending_scene: tuple[str, float] | None = None
        self.tune = tuning or Tuning()
        self.rng = random.Random(seed)
        self.track: Track | None = None
        self.scene: str | None = None
        self.last_change = -1e9
        self.gap_since: float | None = None
        self.recent: deque[str] = deque(maxlen=self.tune.history)
        self.was_urgent = False
        self.mode = "explore"
        self._energy_s: float | None = None
        self._last_tick: float | None = None
        self._last_urgent_cut = -1e9
        self._now = 0.0
        self.song_since = 0.0
        self.loop: tuple[float, float] | None = None
        self.loop_seg: Segment | None = None
        self.loop_since = 0.0
        self.section_used: dict[tuple[str, float], float] = {}   # (track, seg.start) -> last used
        self._pending_switch: float | None = None   # time when a wound-down song may hand over
        self._recent_tier, self._tier_seen, self._last_escalate = 1, -1e9, -1e9
        self._over_since: float | None = None
        self._tier_hist: deque = deque()
        self._lull_since: float | None = None
        self._wait_cap = 8.0
        self._dwell_until = -1e9
        self._layer = 1.0
        self._layer_since = -1e9
        self._calm_since: float | None = None
        self._prev_tier = 0
        self._outmatched_since: float | None = None
        self._last_bridge = -1e9
        self._callouts: dict = {}

    # ── scoring ────────────────────────────────────────────────────────────
    def _segment_cost(self, track: Track, seg: Segment, intent: Intent, direction: int) -> float:
        lo, hi = intent.energy_range
        cost = abs(seg.energy - intent.energy) * 2.0
        if not (lo - 0.05 <= seg.energy <= hi + 0.05):
            cost += 1.0
        if seg.label in intent.prefer:
            cost -= 0.25 if intent.dynamic else 0.7     # e.g. the Valkyrie wants a real build-up
        if intent.tension >= 0.5 and seg.label == "build":
            cost -= 0.3 * intent.tension
        if intent.urgent and direction > 0 and seg.label in ("drop", "peak"):
            cost -= 0.3
        if direction < 0 and seg.label in ("release", "calm"):
            cost -= 0.15
        if seg.length < 12:
            cost += 0.2              # too short to settle into
        if track is not self.track:
            runway = track.duration - seg.start
            if runway < self.tune.min_runway:   # don't start a new song at its outro
                cost += 0.5 * (1 - runway / self.tune.min_runway) + 0.2
        if track.path in self.recent and track is not self.track:
            cost += self.tune.repeat_penalty
        if self.track is not None and track is not self.track and not self.tempo_ok(self.track, track):
            cost += self.tune.tempo_cost
        if intent.dynamic and track is not self.track:
            want = intent.tier if intent.mode == "combat" else max(1, self._recent_tier)
            cost += self.tune.tier_cost * abs(track.tier - want)
        cur = self._cur_seg()
        if cur is not None and cur.timbre is not None and seg.timbre is not None:
            w = self.tune.sound_cost if track is not self.track else self.tune.sound_cost_inside
            cost += w * (1.0 - float(cur.timbre @ seg.timbre)) / 2
        if cur is not None and cur.label == "build" and seg.label in ("release", "calm"):
            cost += 0.5                  # a build-up must pay off, not fall into a calm-down
        if self.ratings is not None and self.ratings.get(track.path) == LIKE:
            cost -= self.tune.like_bonus
        if seg.loop is not None:
            cost -= 0.1              # loopable sections can be held as long as the moment lasts
        used = self.section_used.get((track.path, seg.start))
        if used is not None and self._now - used < self.tune.section_memory:
            cost += self.tune.section_repeat_penalty * (1 - (self._now - used) / self.tune.section_memory)
        cost += self._role_cost(seg, intent)
        cost += self._lobby_cost(track, intent)
        if track is self.track and intent.dynamic:
            cost += self._grammar_cost(seg, intent, direction)
        if intent.dynamic and intent.mode != "combat" and track is not self.track \
                and not any(x.role == "combat" for x in track.segments):
            cost += self.tune.fightless_song_cost
        return cost + self.rng.random() * 0.05

    def _keep_calm(self, intent: Intent, now: float, position: float) -> Cue | None:
        """Hub/menus/loading: the song must stay within the scene's energy. If it's heading into
        a loud part, move to a calm part of it, or hand over to a calmer song."""
        lo, hi = intent.energy_range
        if intent.urgent or now - self.last_change < 15.0:
            return None
        cur = self.track.segment_at(position)
        if max(cur.energy, self._energy_ahead(position)) <= hi + 0.08:
            return None
        inside = self._best(intent, -1, only=self.track, skip_seg=cur, must_beat=0.15)
        if inside and inside[1].energy <= hi and inside[1].role != "combat":
            return self._commit(self.track, inside[1].start, self.tune.energy_fade, self._bar_delay(position),
                                "calm:jump", now)
        other = self._best(intent, -1, exclude=self.track)
        if other:
            return self._commit(other[0], other[1].start, self.tune.switch_fade, self._bar_delay(position),
                                "handover:calm", now)
        return None

    def _lobby_cost(self, track: Track, intent: Intent) -> float:
        """A hub/menu song should be calm all the way through, not just where it starts."""
        if intent.dynamic or intent.urgent or track is self.track:
            return 0.0
        lo, hi = intent.energy_range
        loud = float(np.percentile(track.energy, 80)) if len(track.energy) else 0.0
        fight = sum(x.length for x in track.segments if x.role == "combat") / max(1.0, track.duration)
        return 1.5 * max(0.0, loud - hi) + 0.6 * fight

    @staticmethod
    def _role_cost(seg: Segment, intent: Intent) -> float:
        """Combat wants driving, intense sections; exploring wants everything else."""
        if not intent.dynamic:
            return 0.4 if seg.role == "combat" else 0.0       # hub/menus: keep it relaxed
        if intent.mode == "combat":
            if seg.role == "build":
                return -0.2 if getattr(intent, "building", False) else 1.6   # build BEFORE the fight, never in it
            return {"combat": 0.0}.get(seg.role, 1.2)
        if seg.role == "combat":
            return 0.9
        if seg.role == "build" and intent.tension < 0.5:
            return 0.15
        return 0.0

    @staticmethod
    def _role_fits(role: str, mode: str) -> bool:
        return role == "combat" if mode == "combat" else role != "combat"

    def _still_fits(self, track: Track, position: float, mode: str) -> bool:
        """Is the song, as it continues from here, still acceptable for this mode? More tolerant
        than _role_fits: in combat a build-up, or a short breather that leads back into a
        combat section, is part of the song's arc and not a reason to leave."""
        seg = track.segment_at(position + 4)
        if self._role_fits(seg.role, mode):
            return True
        if mode != "combat":
            return False
        if seg.role == "build":
            return bool(getattr(self._intent, "building", False))   # only while the fight is still coming
        idx = track.segments.index(seg)
        nxt = track.segments[idx + 1] if idx + 1 < len(track.segments) else None
        if getattr(self._intent, "tier", 0) >= 2:
            return False                 # in a horde, no breathers/breakdowns: move on before it
        return seg.length <= self.tune.breather and nxt is not None and nxt.role in ("combat", "build")

    def _best(self, intent: Intent, direction: int, *, only: Track | None = None,
              exclude: Track | None = None, after: float = -1.0,
              must_beat: float | None = None, need_mode: str | None = None,
              skip_seg: Segment | None = None, min_tier: int = 0,
              similar: bool = False, allow_bridge: bool = False) -> tuple[Track, Segment, float] | None:
        """Lowest-cost segment. `must_beat`: only accept segments whose energy is closer to the
        target than this (so a move always actually helps)."""
        best = None
        for track in ([only] if only else self.lib.tracks):
            if track is exclude:
                continue
            if self.ratings is not None and self.ratings.get(track.path) == BAN:
                continue
            if track.tier < min_tier:
                continue
            for seg in track.segments:
                if seg.skip:
                    continue                       # you said never to use this part
                if intent.dynamic and seg.length < 6 and not seg.bridge:
                    continue                       # a 1-s sliver lands you straight in the next part
                if seg.bridge and not allow_bridge and intent.mode == "combat":
                    continue                       # in a fight, bridges are passed through, not settled in
                if only is not None and seg.start <= after:
                    continue
                if similar and track is not self.track and not self._sounds_alike(track, seg):
                    continue
                if must_beat is not None and abs(seg.energy - intent.energy) > must_beat:
                    continue
                if need_mode is not None and not self._role_fits(seg.role, need_mode):
                    continue
                if seg is skip_seg:
                    continue
                c = self._segment_cost(track, seg, intent, direction)
                if best is None or c < best[2]:
                    best = (track, seg, c)
        return best

    # ── main entry ─────────────────────────────────────────────────────────
    def tick(self, intent: Intent | None, now: float, position: float | None) -> Cue | None:
        """Call regularly. `position` is seconds into the current track (None if nothing playing)."""
        if intent is None or not self.lib.tracks:
            return None
        t = self.tune
        self._now = now
        self._position = position
        newly_urgent = (intent.urgent and not self.was_urgent
                        and now - self._last_urgent_cut >= t.urgent_cooldown)
        self.was_urgent = intent.urgent
        self._raw_energy = intent.energy        # "high intensity" reacts to the unsmoothed level
        intent = self._smooth(intent, now)
        self._intent = intent
        if now < getattr(self, "_busy_until", 0.0) and not self._skip_requested:
            return None

        # 0. Player asked to get rid of this song (thumbs down).
        if self._skip_requested:
            self._skip_requested = False
            hit = self._best(intent, 0, exclude=self.track)
            if hit:
                self.scene = intent.scene
                self.family = intent.family
                start = hit[1].start if (intent.dynamic or intent.urgent or intent.prefer) else 0.0
                return self._commit(hit[0], start, t.skip_fade, 0.0, "skip", now)

        # 1. Scene change → new piece.
        if self.track is not None and position is not None and intent.scene != self.scene:
            if intent.family == self.family:
                # e.g. hub → mission board → loading: same family, keep the song going.
                self.scene = intent.scene
                self._pending_scene = None
            elif not intent.urgent:
                if self._pending_scene is None or self._pending_scene[0] != intent.scene:
                    self._pending_scene = (intent.scene, now)
                if now - self._pending_scene[1] < t.scene_settle:
                    return None
        if (intent.scene != self.scene and self.track is not None and position is not None and not intent.dynamic
                and (SCENE_KEEP or now - getattr(self, "_scene_song_at", -1e9) < SCENE_HOLD)):
            # 2026-11-21 in game: loadout -> Valkyrie, victory -> loading -> cinematic -> hub each picked a
            # new song seconds apart. Menus / transit screens in quick succession share one song.
            self.scene = intent.scene
            self.family = intent.family
            self._pending_scene = None
            return None
        if intent.scene != self.scene or self.track is None or position is None:
            self.scene = intent.scene
            self.family = intent.family
            self._pending_scene = None
            if not intent.dynamic:
                self._scene_song_at = now
            hit = self._best(intent, 0, exclude=self.track)
            if not hit:
                return None
            track, seg, _ = hit
            start = seg.start if (intent.dynamic or intent.urgent or intent.prefer) else 0.0
            return self._commit(track, start, t.urgent_fade if intent.urgent else t.scene_fade, 0.0,
                                "scene:" + intent.scene, now)

        # 2. Song ending → continue at the current energy. In a mission, high intensity or any
        #    fight: go back into this song's own fitting music instead of starting another.
        if position >= self.track.duration - t.end_margin and intent.dynamic and (
                intent.mode == "combat" or self._is_high(intent)) and now - self.song_since < 2 * t.fatigue_time:
            back = self._best(intent, 0, only=self.track, need_mode=intent.mode)
            if back:
                return self._commit(self.track, back[1].start, t.energy_fade, 0.0, "song-end:jump", now)
        if position >= self.track.duration - t.end_margin:
            hit = self._best(intent, 0, exclude=self.track)
            if hit:
                return self._commit(hit[0], hit[1].start if intent.dynamic else 0.0, t.end_margin - 1, 0.0,
                                    "song-end", now)

        if not intent.dynamic:
            return self._keep_calm(intent, now, position)

        # 3. Inside a dynamic scene: stay in the song as long as it can serve the moment.
        self.mode = intent.mode
        if intent.mode == "combat":
            if intent.tier >= self._recent_tier or now - self._tier_seen > 120:
                self._recent_tier, self._tier_seen = intent.tier, now
        # The fight got bigger than this song can carry (a horde, a boss): go to a song that can,
        # on the old song's drop. The one exception to "no song changes at high intensity".
        if (intent.mode == "combat" and intent.tier >= 2 and self.track.tier < intent.tier
                and now - self._last_escalate >= self.tune.escalate_gap and now - self.last_change >= 6
                and self._pending_switch is None):
            cur_seg = self.track.segment_at(position)
            climb = [x for x in self.track.segments if x is not cur_seg and x.role == "combat"
                     and x.length >= 8 and x.energy >= max(cur_seg.energy + 0.05, intent.energy - 0.08)]
            song_top = max([x.energy for x in self.track.segments if x.role == "combat"] or [0.0])
            if song_top < intent.energy - 0.1:
                climb = []                       # even this song's biggest part can't carry this fight
            if climb and cur_seg.energy < intent.energy - 0.08:
                self._last_escalate = now        # this song can still go bigger: climb within it
                pick = max(climb, key=lambda x: -abs(x.energy - intent.energy))
                return self._commit(self.track, pick.start, t.energy_fade, self._bar_delay(position),
                                    f"climb{intent.tier}:jump", now)
            if not climb:
                up = self._best(intent, 1, exclude=self.track, need_mode="combat", min_tier=intent.tier,
                                similar=True)
                self._outmatched_since = self._outmatched_since or now
                if not up and now - self._outmatched_since >= 4:
                    up = self._best(intent, 1, exclude=self.track, need_mode="combat", min_tier=intent.tier)
                if up:
                    self._outmatched_since = None
                    self._last_escalate = now
                    return self._commit(up[0], up[1].start, t.energy_fade, self._bar_delay(position),
                                        f"combat:escalate{intent.tier}", now)
        # The big fight is over but this song is still built for it: after a while, hand over to
        # a song that fits the smaller fight (only when it isn't a high-intensity moment).
        # Judge "the fight eased off" on the average level over the last 30 s: the game's states
        # flicker between horde and normal every few seconds.
        self._tier_hist.append((now, intent.tier if intent.mode == "combat" else 0))
        while self._tier_hist and now - self._tier_hist[0][0] > 30.0:
            self._tier_hist.popleft()
        avg_tier = sum(x for _, x in self._tier_hist) / max(1, len(self._tier_hist))
        over = intent.mode == "combat" and self.track.tier - avg_tier >= 1.5 and not intent.urgent
        # "a lull when nothing is happening": no fight at all — or, if one song has run very long,
        # at least a quiet skirmish
        long_song = now - self.song_since >= 2 * self.tune.fatigue_time
        dip = intent.mode == "combat" and intent.tier <= max(1, avg_tier - 0.75)    # clearly below the recent fight
        calm_now = intent.mode != "combat" or (long_song and (intent.tier <= 1 or dip))
        if calm_now:
            self._calm_since = self._calm_since or now
        else:
            self._calm_since = None
        in_lull = self._calm_since is not None and now - self._calm_since >= t.calm_for_song_change
        rising = intent.mode == "combat" and intent.tier > self._prev_tier
        self._prev_tier = intent.tier if intent.mode == "combat" else 0
        new_wave = long_song and rising and intent.tier >= 2      # a big moment to bring in a fresh song
        if (intent.mode == "combat" and now - self.song_since >= 2.5 * t.fatigue_time
                and self.track.on_phrase(position + 0.5, tol=0.6, grid=self.track.phrases8)):
            new_wave = True                       # a non-stop fight: change on a phrase start, with an impact
        # Song fatigue: one song has carried long enough — move on, but only in a real lull, letting
        # the old song fade and bringing the new one in with its intro (never mid-fight).
        if (now - self.song_since >= self.tune.fatigue_time and now - self.last_change >= 20
                and self._pending_switch is None and not over and (in_lull or new_wave)):
            fresh = self._best(intent, 0, exclude=self.track, need_mode=intent.mode,
                               min_tier=intent.tier if intent.mode == "combat" else 0, similar=True)
            if not fresh and now - self.song_since >= 1.5 * self.tune.fatigue_time:
                # nothing sounds like this song: still move on eventually, with a long blend
                fresh = self._best(intent, 0, exclude=self.track, need_mode=intent.mode,
                                   min_tier=intent.tier if intent.mode == "combat" else 0)
            if fresh:
                kind = (f"combat:escalate{intent.tier}:fresh" if new_wave and not in_lull else
                        "combat:fatigue" if intent.mode == "combat" else "handover:fatigue")
                return self._commit(fresh[0], fresh[1].start, t.switch_fade, self._bar_delay(position), kind, now)
        if not over:
            self._over_since = None
        elif self._over_since is None:
            self._over_since = now
        elif (now - self._over_since >= self.tune.deescalate_after and now - self.last_change >= 20
              and self._pending_switch is None and in_lull):
            down = self._best(intent, -1, exclude=self.track, need_mode="combat", similar=True)
            if down and down[0].tier < self.track.tier:
                self._over_since = None
                return self._commit(down[0], down[1].start, t.switch_fade, self._bar_delay(position),
                                    "handover:deescalate", now)
        if self._pending_switch is not None:
            if self._is_high(intent):
                self._pending_switch = None            # things heated up: stay in this song
            elif now >= self._pending_switch:
                self._pending_switch = None
                nxt = self._best(intent, 0, exclude=self.track, need_mode=intent.mode)
                if nxt:
                    delay = self._bar_delay(position)
                    return self._commit(nxt[0], nxt[1].start, t.switch_fade, delay, "handover:switch", now)
        cur = self.loop_seg if (self.loop and self.loop_seg is not None) else self.track.segment_at(position)
        time_in_song = now - self.song_since

        ps = getattr(self, "_pending_song", None)
        if ps is not None:
            if now - ps[3] > 45 or self.track is ps[0]:
                self._pending_song = None
            elif cur.bridge and now >= self._dwell_until - self._phrase_secs(self.tune.phrase_dwell_bars) + self._phrase_secs(2):
                self._pending_song = None
                return self._commit(ps[0], ps[1], self.tune.switch_fade, self._bar_delay(position),
                                    ps[2].split(":")[0] + ":in-breakdown", now)
        dwelling = now < self._dwell_until and not newly_urgent
        # Lull, step 1 (instant, no jump): drums+bass down within the current section; back up as
        # soon as the fight returns. Only a lull that outlasts a phrase moves to a calmer section.
        raw = getattr(self, "_raw_energy", intent.energy)
        if self._layer < 1.0 and (raw >= cur.energy - 0.08 or intent.urgent):
            self._layer = 1.0
            return Cue(self.track, position, 0.0, 0.0, "layer:up", action="layer", layer=1.0)
        if (self._layer >= 1.0 and self._layer_ok(self.track) and cur.energy - raw >= t.lull_gap and not intent.urgent
                and self._lull_since is not None and now - self._lull_since >= t.lull_debounce):
            self._layer = t.layer_lull
            self._layer_since = now
            return Cue(self.track, position, 0.0, 0.0, "layer:lull", action="layer", layer=t.layer_lull)
        if dwelling and not (not self._role_ok(cur.role, intent) and intent.mode == "combat"):
            if cur.energy - raw >= t.lull_gap:
                self._lull_since = self._lull_since or now
            else:
                self._lull_since = None
            return None

        # Bridge: nothing has changed for a while — take a detour to another part of this song that
        # fits the moment (a different groove, a build, a breakdown) so it doesn't feel like one loop.
        if (intent.dynamic and not intent.urgent and now - self.last_change >= t.bridge_every
                and self._pending_switch is None and abs(cur.energy - intent.energy) < t.mismatch):
            def bridge_cost(x):
                sim = float(cur.timbre @ x.timbre) if cur.timbre is not None and x.timbre is not None else 0.5
                used = self.section_used.get((self.track.path, x.start))
                recent = 1.0 if used is not None and now - used < 120 else 0.0
                # a real contrast, but still the same song's world and the same energy band
                return abs(x.energy - intent.energy) * 2 + abs(sim - 0.45) + recent
            others = [x for x in self.track.segments
                      if x is not cur and x.length >= 8 and self._role_ok(x.role, intent)
                      and abs(x.energy - intent.energy) <= 0.15]
            if others:
                pick = min(others, key=bridge_cost)
                if bridge_cost(pick) < 1.0:
                    return self._commit(self.track, pick.start, t.energy_fade, self._bar_delay(position),
                                        "bridge:jump", now)

        # Lull: the fighting eased (horde cleared, moving between clusters) — answer it within ~5 s
        # inside this song, ignoring the usual holds. Uses the unsmoothed game level.
        raw = getattr(self, "_raw_energy", intent.energy)
        if cur.energy - raw >= t.lull_gap and not intent.urgent:
            if self._lull_since is None:
                self._lull_since = now
            elif (now - self._lull_since >= max(t.lull_debounce, self._phrase_secs(8))
                  and now - self.last_change >= 5.0):
                calmer = [x for x in self.track.segments
                          if x is not cur and self._role_ok(x.role, intent) and x.length >= 8
                          and x.energy <= cur.energy - 0.1 and not x.skip
                          and not (x.bridge and intent.mode == "combat")]
                if calmer:
                    def lull_cost(x):
                        sim = float(cur.timbre @ x.timbre) if cur.timbre is not None and x.timbre is not None else 0.0
                        used = self.section_used.get((self.track.path, x.start))
                        rep = 0.3 if used is not None and now - used < 60 else 0.0
                        return abs(x.energy - raw) + 0.3 * (1 - sim) / 2 + rep
                    pick = min(calmer, key=lull_cost)
                    self._lull_since = None
                    return self._commit(self.track, pick.start, t.energy_fade, 0.05, "lull:jump", now)
        else:
            self._lull_since = None

        if self.loop:
            # 3a. Holding a loop: keep it while the section fits and isn't worn out.
            if self._fits(cur, intent) and now - self.loop_since < self._loop_budget() and not newly_urgent:
                self.gap_since = None
                return None
        else:
            # 3b. Playing through: fine while what's coming still fits.
            ending = position >= self.track.duration - t.end_margin - 4
            ahead_ok = (not ending and self._still_fits(self.track, position, intent.mode)
                        and abs(self._energy_ahead(position) - intent.energy) < t.mismatch)
            can_hold = (cur.loop is not None and self._fits(cur, intent)
                        and position < cur.loop[1] - 1.0)
            if ahead_ok and not newly_urgent:
                nxt = self._next_segment(cur)
                if can_hold and (nxt is None or not self._fits(nxt, intent)):
                    return self._loop(cur, now, "hold")       # next section won't fit: hold this one
                self.gap_since = None
                return None
            if can_hold and not newly_urgent:
                return self._loop(cur, now, "hold")           # about to leave a fitting section

        # 3c. Something has to change. Wait out flicker unless urgent.
        wrong_role = not self._role_ok(cur.role, intent)
        if wrong_role:
            direction = 1 if intent.mode == "combat" else -1
        else:
            direction = 1 if intent.energy > cur.energy else -1
        if self.gap_since is None:
            self.gap_since = now
        dwell = 0.0 if newly_urgent else (t.rise_dwell if direction > 0 else t.fall_dwell)
        tired = self.loop is not None and self._fits(cur, intent)
        if tired or (self.loop is None and position >= self.track.duration - t.end_margin - 4):
            dwell = 0.0                                   # loop worn out / song ending: act now
        held = now - self.last_change
        bypass_hold = newly_urgent or tired or wrong_role      # changing pool always reacts promptly
        if not newly_urgent and (now - self.gap_since < dwell or (held < t.min_hold and not bypass_hold)):
            return None

        # The moment a boss/horde arrives cuts hard on the very next beat; everything else waits
        # for a bar line.
        fade = t.urgent_fade if newly_urgent else t.energy_fade
        delay = 0.0 if newly_urgent else self._bar_delay(position)
        if newly_urgent:
            self._last_urgent_cut = now

        # 3d. Same song first: another fitting section (loopable preferred), else another song.
        must_beat = None if (wrong_role or tired) else max(0.05, abs(intent.energy - cur.energy) - 0.1)
        inside = self._best(intent, direction, only=self.track, must_beat=must_beat,
                            need_mode=intent.mode, skip_seg=cur)
        elsewhere = self._best(intent, direction, exclude=self.track, must_beat=must_beat,
                               need_mode=intent.mode)
        bias = t.stay_bias
        if time_in_song < t.min_song_time:
            bias += t.song_stay_bonus
        elif time_in_song > t.max_song_time:
            bias -= t.song_stay_bonus
        if tired:
            kind = "tired"
        elif wrong_role:
            kind = "combat" if intent.mode == "combat" else "explore"
        else:
            kind = "energy"
        high = self._is_high(intent)
        song_can_serve = any(self._role_ok(x.role, intent) for x in self.track.segments)
        if inside and (high or not elsewhere or inside[2] <= elsewhere[2] + bias):
            dest, how = (self.track, inside[1]), "jump"
        elif elsewhere and not song_can_serve:
            # The only forced change: this song has nothing that fits (e.g. a calm-only song when
            # a fight starts).
            dest, how = (elsewhere[0], elsewhere[1]), "switch"
        elif elsewhere and not high:
            # Low intensity: allowed to move on, but wind down inside this song first, then hand over.
            if self._pending_switch is None:
                down = self._winddown_target(intent, cur)
                if down is not None:
                    self._pending_switch = now + t.winddown_time
                    delay = self._bar_delay(position)
                    return self._commit(self.track, down.start, t.energy_fade, delay, kind + ":winddown", now)
            return None if self._pending_switch is not None else \
                self._commit(elsewhere[0], elsewhere[1].start, t.switch_fade,
                             self._bar_delay(position), kind + ":switch", now)
        else:
            # High intensity and nothing better in this song: keep going with what we have.
            relaxed = self._best(intent, direction, only=self.track, need_mode=intent.mode, skip_seg=cur)
            if relaxed and not tired:
                dest, how = (self.track, relaxed[1]), "jump"
            else:
                if self.loop:
                    self.loop_since = now              # extend the current loop rather than switch
                self.gap_since = None
                return None
        track, seg = dest
        start = seg.start
        # A big shift inside the song (fight ↔ calm, or a big energy jump) passes through the song's
        # own breakdown / bridge first, unless it's urgent or a fight is starting.
        big = (cur.role == "combat") != (seg.role == "combat") or abs(seg.energy - cur.energy) >= 0.25
        rise = (track is self.track and seg.energy - cur.energy >= 0.2 and not newly_urgent and cur.label != "build"
                and (intent.mode != "combat" or getattr(intent, "building", False)))
        if rise:
            builds = [x for i, x in enumerate(self.track.segments[:-1])
                      if (x.label == "build" or getattr(x, "user_role", None) == "build") and x is not cur
                      and x.length >= 6 and not x.skip and self.track.segments[i + 1].energy >= seg.energy - 0.15]
            if builds:
                via = min(builds, key=lambda x: abs(self.track.segments[self.track.segments.index(x) + 1].energy - seg.energy))
                return self._commit(self.track, via.start, fade, self._bar_delay(position), f"{kind}:via-build", now)
        if (track is self.track and big and not newly_urgent and not cur.bridge and kind != "tired"
                and intent.mode != "combat"):     # breakdowns are for lulls, never mid-fight
            bridges = [x for x in self.track.segments if x.bridge and x is not cur and x.length >= 6]
            if bridges:
                via = min(bridges, key=lambda x: abs(x.energy - (cur.energy + seg.energy) / 2))
                return self._commit(self.track, via.start, fade, self._bar_delay(position), f"{kind}:via-bridge", now)
        if (wrong_role and intent.mode == "combat") or (newly_urgent and cur.role != "combat"):
            # SLAM IN: start the destination exactly on an impact, hard cut, on the beat.
            hit = self._slam_hit(track, seg, intent)
            if hit is not None:
                start, fade = hit, t.cut_fade
                delay = 0.0 if newly_urgent else self._bar_delay(position)
                kind += ":slam"
        elif wrong_role and intent.mode != "combat":
            # DROP OUT: leave the fight on a stop (or hit) in the current song, quickly.
            exit_at = self._next_exit(position)
            if exit_at is not None:
                delay, fade = exit_at, t.exit_fade
                kind += ":cutout"
            else:
                delay = max(0.0, self.track.next_beat(position) - position)
                fade = t.quick_fade
        return self._commit(track, start, fade, delay, f"{kind}:{how}", now)

    ALLOWED_NEXT = {
        "calm": {"calm", "build", "groove", "release"},
        "build": {"drop", "peak", "groove", "build"},
        "groove": {"groove", "build", "peak", "drop", "release", "calm"},
        "peak": {"peak", "groove", "release", "drop"},
        "drop": {"drop", "peak", "groove", "release"},
        "release": {"calm", "groove", "build", "release"},
    }

    def _grammar_cost(self, seg: Segment, intent: Intent, direction: int) -> float:
        """Moves inside a song should make musical sense: carrying on into the next section is
        best; some section orders don't (a drop straight into a build); big rises go through the
        song's own build-up."""
        cur = self._cur_seg()
        if cur is None or seg is cur:
            return 0.0
        cost = 0.0
        nxt = self._next_segment(cur)
        if seg is nxt:
            cost -= 0.35                                   # natural continuation
        if seg.label not in self.ALLOWED_NEXT.get(cur.label, {seg.label}):
            cost += 0.5
        if direction > 0 and seg.label == "build":
            after = self._next_segment(seg)
            if after is not None and self._role_ok(after.role, intent) and after.energy >= intent.energy - 0.1:
                cost -= 0.35                               # route the rise through a build-up
        if direction > 0 and seg.label in ("drop", "peak") and cur.label != "build" \
                and seg.energy - cur.energy >= 0.2:
            cost += 0.2                                    # a big rise without a build sounds like a skip
        return cost

    def _cur_seg(self) -> Segment | None:
        if self.track is None:
            return None
        if self.loop and self.loop_seg is not None:
            return self.loop_seg
        pos = getattr(self, "_position", None)
        return self.track.segment_at(pos) if pos is not None else None

    def _slam_leave(self, position: float, max_wait: float | None = None) -> float | None:
        """Fight starts: the old song's first big beat (strong drum downbeat or hit) within the
        window — switch ON it, before the old song relaxes. Else the next downbeat."""
        tr = self.track
        if not tr.downbeats and not tr.hits:
            return None
        max_wait = self.tune.slam_leave_wait if max_wait is None else max_wait
        loop = self.loop
        cands = []
        moments = [(d, i) for i, d in enumerate(tr.downbeats)]
        moments += [(h, -1) for h, s_ in tr.hits if s_ >= 6]     # hits count even off the bar grid
        for d, i in sorted(moments):
            w = d - position
            if loop:
                if not (loop[0] - 0.05 <= d < loop[1] - 0.05):
                    continue
                if w < 0.05:
                    w = (loop[1] - position) + (d - loop[0])
            if 0.05 <= w <= max_wait:
                cands.append((w, i, d))
        for i, d in [(i, d) for d, i in moments]:
            back = position - d
            if 0.0 <= back <= self.tune.catchup and not loop:
                strong = i < 0 or (i < len(tr.downbeat_strength) and tr.downbeat_strength[i] >= self.tune.strong_beat) \
                    or any(abs(h - d) <= 0.08 and s >= 6 for h, s in tr.hits)
                if strong:
                    return -back
        first = None
        for w, i, d in sorted(cands):
            if first is None:
                first = w
            strong = i < 0 or (i < len(tr.downbeat_strength) and tr.downbeat_strength[i] >= self.tune.strong_beat) \
                or any(abs(h - d) <= 0.08 and s >= 6 for h, s in tr.hits)
            if strong:
                return w
        return first

    def _layer_ok(self, track: Track) -> bool:
        """Only songs whose drums separate cleanly get the drums-down treatment (user: removing
        drums that can't be cleanly removed sounds bad). cache/stems/clean.json, lower = cleaner."""
        if not hasattr(self, "_clean"):
            import json, os
            p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "stems", "clean.json")
            try:
                self._clean = json.load(open(p, encoding="utf-8"))
            except (OSError, ValueError):
                self._clean = {}
        c = self._clean.get(track.path)
        return c is not None and c.get("beat_survives", 1.0) <= 0.93

    def _phrase_secs(self, bars: int) -> float:
        return bars * 4 * (self.track._period() if self.track is not None else 0.5)

    def _intro_start(self, track: Track, seg: Segment, intent: Intent) -> float:
        """A new song after a lull comes in from its intro when the intro suits the moment."""
        first = track.segments[0]
        if first is not seg and first.length >= 8 and first.energy >= intent.energy - 0.35 \
                and first.energy <= intent.energy + 0.25:
            return first.start
        return seg.start

    def callouts(self, track: Track) -> list[dict]:
        if track.path not in self._callouts:
            import json, os
            from .analysis import cache_key
            p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "stems",
                             cache_key(track.path) + "_callouts.json")
            try:
                self._callouts[track.path] = json.load(open(p, encoding="utf-8")).get("callouts", [])
            except (OSError, ValueError):
                self._callouts[track.path] = []
        return self._callouts[track.path]

    def _sounds_alike(self, track: Track, seg: Segment) -> bool:
        """A song change mid-mission only to something that sounds like where we are and shares
        the tempo ("completely different songs… are shit")."""
        if self.track is None:
            return True
        if not self.tempo_ok(self.track, track):
            return False
        cur = self._cur_seg()
        if cur is None or cur.timbre is None or seg.timbre is None:
            return True
        return float(cur.timbre @ seg.timbre) >= self.tune.song_change_min_sound

    def tempo_ok(self, a: Track, b: Track) -> bool:
        import math
        if not a.tempo or not b.tempo:
            return True
        r = b.tempo / a.tempo
        return min(abs(math.log2(r * k)) for k in (1.0, 2.0, 0.5)) <= math.log2(1 + self.tune.tempo_tolerance)

    def _strength(self, track: Track, x: float) -> float:
        """How strong a moment to change on: section boundary > big hit > 4-bar line > 2-bar > bar."""
        seg = track.segment_at(x + 0.05)
        if abs(seg.start - x) <= 0.15 or abs(seg.end - x) <= 0.15:
            return 3.0
        if any(abs(h - x) <= 0.08 and s >= 6 for h, s in track.hits):
            return 2.0
        if track.downbeats:
            n = sum(1 for d in track.downbeats if seg.start - 0.15 <= d < x - 0.15)
            if n % 4 == 0:
                return 1.5
            if n % 2 == 0:
                return 0.75
        return 0.0

    def _leave_point(self, position: float, min_delay: float, max_wait: float) -> float | None:
        """Delay until the strongest bar line in the current song within the window (following
        the active loop), trading strength against waiting."""
        tr = self.track
        if not tr.downbeats:
            return None
        cands = []
        loop = self.loop
        for d in tr.downbeats:
            if loop and not (loop[0] - 0.05 <= d < loop[1] - 0.05):
                continue
            w = d - position
            if loop and w < 0:
                w = (loop[1] - position) + (d - loop[0])
            if min_delay <= w <= max_wait:
                cands.append((w, d))
        if loop and min_delay <= loop[1] - position <= max_wait:
            cands.append((loop[1] - position, loop[0]))      # the loop's own boundary is a phrase
        if not cands:
            return None
        best = max(cands, key=lambda c: self._strength(tr, c[1]) - 0.12 * c[0])
        return best[0]

    def _slam_hit(self, track: Track, seg: Segment, intent: Intent) -> float | None:
        """Strongest hit near the start of `seg` that lands in fight-worthy music and is loud
        straight away (a filtered/muffled lead-in before the real drop sounds weak)."""
        a, b = seg.start - 0.25, min(seg.end - 4.0, seg.start + self.tune.entry_window)
        for strength, h in sorted(((s, x) for x, s in track.hits if a <= x <= b), reverse=True):
            if not self._role_ok(track.segment_at(h + 0.3).role, intent):
                continue
            after = sum(track.energy_at(h + i) for i in range(3)) / 3
            if after < seg.energy - 0.08:
                continue
            if self.track is not None and after - self.track.energy_at(self._position or 0) > self.tune.slam_max_rise:
                continue
            return h
        return None

    def _bar_delay(self, position: float) -> float:
        """Wait for the next bar line if it's close; otherwise just the next beat. (Beat grids can
        be sparse inside loops/breakdowns, and a long wait lets the moment change under us.)"""
        d = self.track.next_bar(position) - position
        if 0.0 <= d <= 2.5:
            return d
        return max(0.0, min(2.5, self.track.next_beat(position) - position))

    def _is_high(self, intent: Intent) -> bool:
        e = max(intent.energy, getattr(self, "_raw_energy", 0.0))
        return intent.urgent or (intent.mode == "combat" and e >= self.tune.high_intensity)

    def _winddown_target(self, intent: Intent, cur: Segment) -> Segment | None:
        """This song's calmest section that still fits the moment, if it's clearly calmer."""
        bridges = [x for x in self.track.segments if x.bridge and x is not cur and x.length >= 6]
        if bridges:
            return max(bridges, key=lambda x: -abs(x.energy - (cur.energy - 0.25)))   # leave via the breakdown
        cands = [x for x in self.track.segments
                 if x is not cur and self._role_ok(x.role, intent) and x.energy < cur.energy - 0.1
                 and x.length >= 8 and not x.skip]
        def key(x):                       # calm, but still sounding like where we are
            sim = float(cur.timbre @ x.timbre) if cur.timbre is not None and x.timbre is not None else 0.0
            return x.energy + 0.4 * (1.0 - sim) / 2
        return min(cands, key=key) if cands else None

    def _next_exit(self, position: float) -> float | None:
        """Seconds until the next stop (preferred) or hit in the current song within the
        lookahead, following the active loop if there is one."""
        horizon = self.tune.exit_lookahead

        def ahead(times):
            out = []
            for x in times:
                d = x - position
                if 0.05 < d <= horizon and (not self.loop or x < self.loop[1]):
                    out.append(d)
                if self.loop:          # wrapping: loop end comes before x again
                    a, b = self.loop
                    if a <= x < b:
                        d2 = (b - position) + (x - a)
                        if 0.05 < d2 <= horizon:
                            out.append(d2)
            return out
        stops = ahead([x for x, _ in self.track.stops])
        if stops:
            return min(stops)
        hits = ahead([x for x, s in self.track.hits if s >= 8])
        return min(hits) if hits else None

    # ── helpers for in-song decisions ─────────────────────────────────────
    def _role_ok(self, role: str, intent: Intent) -> bool:
        if intent.mode == "combat":
            return role == "combat" or (role == "build" and getattr(intent, "building", False))
        return role != "combat"

    def _fits(self, seg: Segment, intent: Intent) -> bool:
        if seg.bridge:
            return intent.mode != "combat"           # a breakdown is good exploring music
        return self._role_ok(seg.role, intent) and abs(seg.energy - intent.energy) < self.tune.mismatch

    def _energy_ahead(self, position: float) -> float:
        ahead = [self.track.energy_at(position + i) for i in range(0, 8)]
        return sum(ahead) / len(ahead)

    def _next_segment(self, seg: Segment) -> Segment | None:
        segs = self.track.segments
        i = segs.index(seg)
        return segs[i + 1] if i + 1 < len(segs) else None

    def _loop_budget(self) -> float:
        """How long to hold the current loop: ~N passes, within [min, max]."""
        a, b = self.loop
        return min(self.tune.max_loop_time, max(self.tune.min_loop_time, self.tune.loop_passes * (b - a)))

    def _loop(self, seg: Segment, now: float, reason: str) -> Cue:
        self.loop, self.loop_seg, self.loop_since = seg.loop, seg, now
        self.section_used[(self.track.path, seg.start)] = now
        self.gap_since = None
        return Cue(self.track, seg.loop[0], 0.0, 0.0, reason, action="loop", loop=seg.loop)

    def _smooth(self, intent: Intent, now: float) -> Intent:
        """Exponentially smooth the energy target so per-second jitter in the game's
        parameters (enemy counts, suppression) doesn't move the music."""
        import dataclasses, math
        if self._energy_s is None or self._last_tick is None or not intent.dynamic:
            self._energy_s = intent.energy
        else:
            dt = max(0.0, now - self._last_tick)
            tau = self.tune.energy_rise_tau if intent.energy > self._energy_s else self.tune.energy_fall_tau
            if intent.urgent:
                tau = 0.5
            self._energy_s += (intent.energy - self._energy_s) * (1 - math.exp(-dt / tau))
        self._last_tick = now
        return dataclasses.replace(intent, energy=round(self._energy_s, 3))

    def skip(self) -> None:
        """Move off the current song at the next tick."""
        self._skip_requested = True

    def _jump_point(self, seg: Segment, position: float, delay: float, urgent: bool):
        """Best jump point that leaves the current song soon and lands inside `seg` (or another
        section that's just as good for the moment).
        Returns (delay, to, score) or None. Follows the active loop (the source bar may be
        reached after a wrap)."""
        if not self.track.jumps:
            return None
        wait_max = 0.6 if urgent else max(delay, self._wait_cap)
        loop = self.loop
        intent = getattr(self, "_intent", None)
        here = self.track.segment_at(position)
        ok: dict[float, bool] = {}

        def lands_well(to: float) -> bool:
            d = self.track.segment_at(to)
            if to > d.end - 8.0:
                return False
            if here.timbre is not None and d.timbre is not None and d is not here \
                    and float(here.timbre @ d.timbre) < self.tune.jump_min_sound:
                return False
            if d is seg:
                return True
            if d.start not in ok:
                ok[d.start] = (intent is not None and d is not here and d is not self.loop_seg
                               and self._role_ok(d.role, intent)
                               and abs(d.energy - seg.energy) <= 0.1)
            return ok[d.start]
        best = None
        phrased = bool(self.track.phrases)
        for frm, to, score in self.track.jumps:
            if score < self.tune.jump_min or not lands_well(to):
                continue
            if phrased and not (self.track.on_phrase(frm) and self.track.on_phrase(to, tol=0.2)):
                continue
            waits = []
            if not loop or frm < loop[1]:
                waits.append(frm - position)
            if loop and loop[0] <= frm < loop[1]:
                waits.append((loop[1] - position) + (frm - loop[0]))
            for w in waits:
                if 0.6 <= w <= wait_max:           # room for a seam crossfade to finish on the beat
                    v = (score - 0.04 * w - (0.0 if self.track.segment_at(to) is seg else 0.05)
                         + 0.08 * self._strength(self.track, frm))
                    if best is None or v > best[0]:
                        best = (v, w, to, score)
        return None if best is None else best[1:]

    def _entry_start(self, track: Track, dest: Segment, intent: Intent) -> float:
        """A new song comes in through its own build-up or breakdown leading to the part we want —
        never straight into its most intense part ("always a bridge or blended build")."""
        segs = track.segments
        cands = []
        for i, s in enumerate(segs[:-1]):
            nxt = segs[i + 1]
            leads = s.bridge or s.label == "build" or getattr(s, "user_role", None) in ("build", "bridge")
            if leads and s.length >= 6 and not s.skip and self._role_ok(nxt.role, intent):
                cands.append((abs(nxt.energy - dest.energy), s))
        if cands:
            return min(cands, key=lambda c: c[0])[1].start
        first = segs[0]
        if dest.energy >= 0.7 and first.energy < dest.energy - 0.15 and not first.skip:
            return first.start                         # at least come in from the song's opening
        return dest.start

    def _commit(self, track: Track, start: float, fade: float, delay: float, reason: str, now: float) -> Cue:
        pos = getattr(self, "_position", None)
        # Changing song mid-mission: first take the old song into its breakdown and switch during
        # it ("transition to the new song right as the breakdown happens"); the new song then comes
        # in through its build-up, blended.
        if (track is not self.track and self.track is not None and pos is not None and self._intent is not None
                and self._intent.dynamic and reason.split(":")[0] not in ("scene", "skip")
                and fade != self.tune.urgent_fade and ":entry" not in reason):
            cur = self._cur_seg()
            if cur is not None and not cur.bridge and "in-breakdown" not in reason and self._intent.mode != "combat":
                bridges = [x for x in self.track.segments if x.bridge and x is not cur and x.length >= 6]
                if bridges:
                    via = min(bridges, key=lambda x: abs(x.energy - cur.energy))
                    self._pending_song = (track, start, reason, now)
                    return self._commit(self.track, via.start, fade, self._bar_delay(pos),
                                        reason.split(":")[0] + ":exit-bridge", now)
            start = self._entry_start(track, track.segment_at(start + 0.05), self._intent)
            reason += ":entry"
        # In-song move: use a jump point (bar that sounds like this one) when there's one soon.
        curve = "power"
        jumped = False
        if (track is self.track and pos is not None and "slam" not in reason
                and "cutout" not in reason):
            self._wait_cap = self.tune.lull_max_wait if reason.startswith("lull") else self.tune.jump_wait
            jp = self._jump_point(track.segment_at(start), pos, delay, fade <= self.tune.urgent_fade)
            if jp is not None:
                delay, start, score = jp
                jumped = True          # `start` is seam-aligned already: don't snap it
                if score >= 0.85:
                    fade, curve = 0.15, "linear"
                elif score >= 0.7:
                    fade, curve = 0.3, "linear"
                else:
                    fade = min(fade, 1.0)
                reason += f"~{score:.2f}"
        if curve == "linear":
            style = "seam"
        elif ":entry" in reason and self.track is not None and self.tempo_ok(self.track, track):
            style = "blend"
            fade = min(8.0, self.tune.song_blend_bars * 4 * self.track._period())
        elif (reason.split(":")[0] in ("handover", "song-end") and self.track is not None
              and not self.tempo_ok(self.track, track)):
            # different tempos can't be faded together: commit on a big moment, change instantly
            style = "punch"
            fade = self.track._period()
            hit = self._slam_hit(track, track.segment_at(start), self._intent) if self._intent else None
            if hit is not None:
                start = hit
            reason += ":commit"
        elif reason.split(":")[0] in ("scene", "skip", "song-end", "handover"):
            style = "blend"
            if any(k in reason for k in ("fatigue", "deescalate")) and self.track is not None:
                fade = min(8.0, self.tune.song_blend_bars * 4 * self.track._period())
        elif any(k in reason for k in ("escalate", "fatigue")) and track is not self.track:
            style = "blend"
            fade = min(8.0, self.tune.song_blend_bars * 4 * self.track._period())
        elif track is self.track and "slam" not in reason and "cutout" not in reason:
            # in-song move without a jump point: a one-beat crossfade that finishes on the beat
            # ("sudden change needs a fade"), both parts full-range
            style, fade = "blend", self.track._period()
        else:
            style = "punch"
            fade = min(fade, self.tune.switch_tail if track is not self.track else self.tune.tail_max)
        # Per-section shaping (measured from the stems): how the old part wants to be left and
        # how the new part wants to come in, in beats of each song.
        fade_in = None
        if style == "punch" and self.track is not None and pos is not None:
            out_seg = self.track.segment_at(pos)
            if "slam" not in reason:
                fade = min(4.0, max(0.1, out_seg.exit_beats * self.track._period()))
            in_seg = track.segment_at(start)
            if "slam" not in reason and in_seg.entry_beats > 1.0:
                fade_in = min(4.0, in_seg.entry_beats * track._period())
            if "slam" in reason:
                fade_in = track._period()          # one-beat swell into the hit ("rough, no fade")
        if style == "blend" and track is not self.track:
            fade = max(fade, min(6.0, track.segment_at(start).entry_beats * track._period()))
        if style == "blend":
            delay = max(delay, fade)        # the crossfade finishes on the beat, so it starts earlier
        # Leave on the strongest moment nearby (section end, big hit, phrase line), not just the
        # next bar. Not for jump points (chosen already), urgent cuts or the cut-out's own stop.
        fight_entry = "slam" in reason or (reason.startswith("combat:") and track is not self.track
                                           and not any(k in reason for k in ("escalate", "fatigue", ":entry")))
        if (not jumped and pos is not None and self.track is not None and "cutout" not in reason
                and fade != self.tune.urgent_fade and reason.split(":")[0] not in ("scene", "skip")):
            urgent_slam = "slam" in reason and delay == 0.0
            if fight_entry:
                sl = self._slam_leave(pos, 1.0 if urgent_slam else None)
                if sl is not None:
                    delay = sl
            else:
                lp = self._leave_point(pos, max(0.05, delay if style == "blend" else 0.05),
                                       self.tune.slam_wait if "slam" in reason else
                                       self.tune.lull_max_wait if reason.startswith("lull") else self.tune.leave_wait)
                if lp is not None:
                    delay = lp
        # In-song moves land on a phrase start and leave on a phrase boundary (no record-skipping),
        # unless it's urgent or a fight-start slam.
        in_song = track is self.track and pos is not None and "slam" not in reason and "cutout" not in reason \
            and fade != self.tune.urgent_fade and not reason.startswith("layer")
        if in_song and not jumped and self.track.phrases:
            seg_dest = track.segment_at(start + 0.05)
            start = track.snap_start(seg_dest.start)     # section start, on its bar line
            nxt = self.track.next_phrase(pos + max(0.0, delay - 0.05), self.tune.phrase_wait)
            if nxt is not None:
                delay = nxt - pos
            jumped = True                      # start is a section (phrase) start already
        # Song changes after a lull: the old song fades over ~2 bars, the new one comes in with its intro.
        if track is not self.track and any(k in reason for k in ("fatigue", "deescalate", "handover")) \
                and self.track is not None and self._intent is not None:
            start = track.snap_start(self._intro_start(track, track.segment_at(start), self._intent))
            jumped = True
            style = "sequence"
            fade = min(6.0, 8 * self.track._period())
            if pos is not None:
                nb = self.track.next_bar(pos + fade) - pos
                delay = max(delay, nb if 0 < nb < fade + 3 else fade)
        sync = None
        if delay < 0 and fight_entry and pos is not None:
            # Catch-up: the old song's big beat (its drop) was `-delay` ago. Pretend we switched
            # right on it: leave on the old song's next beat counted from that drop, and land the
            # same number of beats into the new part — both on the beat, both in step.
            drop = pos + delay
            p_old, p_new = self.track._period(), track._period()
            k = max(1, int(-(-(pos + 0.12 - drop) // p_old)))     # first beat at least 0.12 s ahead
            delay = drop + k * p_old - pos
            start = start + k * p_new
            if track.beats:                # land on the new song's real beat, not the average-tempo guess
                start = min(track.beats, key=lambda x: abs(x - start))   # always on the new song's real beat
            sync = (drop, start - k * p_new)
            jumped = True
            reason += ":drop"                  # `start` is placed exactly; don't snap it
        # Never off the beat: land on a bar line/beat of the destination, and leave on a beat of
        # the song that's playing (the player switches sample-accurately at `at`).
        if not jumped and round(start, 3) not in {round(x, 3) for x, _ in track.hits}:
            start = track.snap_start(start)
        at = None
        pos = getattr(self, "_position", None)
        if sync is not None:
            at = self.track.beat_after(pos + delay - 0.06)   # the old song's real beat there
            delay = max(0.0, at - pos)
        elif self.track is not None and pos is not None:
            at = self.track.beat_after(pos + delay - 0.03)
            if self.loop and at >= self.loop[1]:
                a, b = self.loop
                at = self.track.beat_after(a + (at - a) % (b - a) - 0.03)
                delay = (b - pos) + (at - a)
            else:
                delay = max(0.0, at - pos)
        # A hit in the old part right on the switch: let it play out ("if we fade on a hit, let
        # it play out depending on which track the hit comes from").
        ring = False
        if style == "punch" and at is not None and ":commit" not in reason:
            if any(abs(x - at) <= 0.08 and s >= 6 for x, s in self.track.hits):
                ring = True        # the old song's hit/drop plays out under the new part
                fade = max(fade, (4 if "slam" in reason else 2) * self.track._period())
        if sync is not None or (fight_entry and style == "punch" and self.track is not None
                                and any(abs(x - (at or -9)) <= 0.08 for x, _ in self.track.hits)):
            ring, fade = True, 4 * self.track._period()    # the old song's drop rings out
        if fight_entry and track is not self.track and style == "blend":
            style = "punch"                 # a fight starting with a new song: commit on the beat
            ring = True
            fade = 4 * self.track._period()
        if style == "punch" and self.track is not None and track is not self.track \
                and not self.tempo_ok(self.track, track):
            fade = min(fade, self.track._period())   # different tempo: never overlap for long
        # Call-out bridges: the old song's own fill (or riser, going up) plays INTO the switch beat;
        # an escalation to a new song is punctuated by the old song's impact on the beat.
        bridge = None
        if self.track is not None and at is not None and pos is not None and now - self._last_bridge >= self.tune.bridge_gap:
            calls = self.callouts(self.track)
            wait = at - pos
            if "escalate" in reason:
                imp = [c for c in calls if c["type"] == "impact"]
                if imp:
                    bridge = (max(imp, key=lambda c: c["score"])["file"], 0.0)
            elif in_song or track is not self.track:
                rising = track.segment_at(start + 0.05).energy > (self._cur_seg().energy if self._cur_seg() else 0) + 0.1
                kinds = ("riser", "fill") if rising else ("fill",)
                for kind in kinds:
                    ok = [c for c in calls if c["type"] == kind and c["len"] <= wait - 0.05]
                    if ok:
                        c = max(ok, key=lambda c: c["score"])
                        bridge = (c["file"], float(c["len"]))
                        break
            if bridge:
                self._last_bridge = now
                reason += ":callout"
        if track is self.track and at is not None:
            self._dwell_until = at + self._phrase_secs(self.tune.phrase_dwell_bars)
        elif ":entry" in reason and at is not None:
            entry = track.segment_at(start + 0.05)
            self._dwell_until = at + max(self._phrase_secs(self.tune.phrase_dwell_bars), entry.end - start)
        self._layer = 1.0
        if track is not self.track:
            self.recent.append(track.path)
            self.song_since = now
        self.loop, self.loop_seg = None, None
        if track is not self.track:
            self._pending_switch = None
        self.section_used[(track.path, track.segment_at(start).start)] = now
        self.track = track
        self.last_change = now
        self.gap_since = None
        self._busy_until = now + delay + 0.15
        return Cue(track, start, max(0.04 if style == "punch" else 0.1, fade), delay, reason,
                   at=at, curve=curve, style=style, fade_in=fade_in, ring=ring, sync=None, bridge=bridge)
