"""Playback backends. The Director emits Cues; a Player makes them audible.

LocalPlayer   — full quality: sample-accurate crossfades, bar-aligned jumps inside a song.
SpotifyPlayer — plays through your Spotify app via the Web API (Premium required).
                Coarser: ~0.3–1 s latency, fades are stepped volume changes.
NullPlayer    — prints cues and simulates position; for dry runs and replays.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np

from .director import Cue


class Player:
    def play(self, cue: Cue) -> None: ...
    def position(self) -> float | None: ...
    def now_playing(self):
        """The Track currently audible (not merely scheduled)."""
        return getattr(self, "_track", None)
    def close(self) -> None: ...


class NullPlayer(Player):
    def __init__(self, clock=time.monotonic, quiet: bool = False):
        self.clock = clock
        self.quiet = quiet
        self._start_clock = None
        self._start_pos = 0.0
        self._duration = 0.0

    def play(self, cue: Cue) -> None:
        if not self.quiet:
            print("  ♪", cue.describe())
        if cue.action == "layer":
            return
        if cue.action == "loop":
            self._loop = cue.loop
            return
        self._loop = None
        self._start_clock = self.clock() + cue.delay
        self._start_pos = cue.start
        self._duration = cue.track.duration
        self._track = cue.track

    def position(self) -> float | None:
        if self._start_clock is None:
            return None
        pos = self._start_pos + max(0.0, self.clock() - self._start_clock)
        loop = getattr(self, "_loop", None)
        if loop and pos >= loop[1]:
            a, b = loop
            pos = a + (pos - a) % (b - a)
        return min(pos, self._duration)


# ── Local audio engine ──────────────────────────────────────────────────────

class _Deck:
    def __init__(self, data: np.ndarray, start_frame: int, fade_frames: int, track=None,
                 curve: str = "power"):
        self.track = track
        self.curve = curve
        self.loop: tuple[int, int] | None = None   # (a, b) frames; jump b→a seamlessly
        self.loop_xf = 0
        self.data = data
        self.pos = start_frame
        self.gain = 0.0
        self.target = 1.0
        self.step = 1.0 / max(1, fade_frames)
        self.rhythm = None       # drums+bass stem aligned with data (vertical layering), if analysed
        self.layer = 1.0         # current drums+bass level
        self.tail = False        # outgoing "tail": bass/kick removed so it can't clash with the new beat
        self.tail_k = 0.0
        self._hp = None

    def fade_out(self, fade_frames: int):
        self.target = 0.0
        self.step = max(self.gain, 1e-6) / max(1, fade_frames)

    def make_tail(self, sr: int, cutoff: float = 180.0, level: float = 0.7):
        """From now on play only what's above `cutoff` (no kick, no bass), a little quieter."""
        from scipy.signal import butter
        b, a = butter(2, cutoff / (sr / 2), "highpass")
        self._hp = (b, a, np.zeros((2, 2)), level)
        self.tail = True

    def _apply_tail(self, chunk: np.ndarray) -> np.ndarray:
        from scipy.signal import lfilter
        b, a, zi, level = self._hp
        hp = np.empty_like(chunk)
        for c in range(2):
            hp[:, c], zi[:, c] = lfilter(b, a, chunk[:, c], zi=zi[:, c])
        n = len(chunk)
        k = np.clip(self.tail_k + np.arange(1, n + 1, dtype=np.float32) / (0.02 * 44100), 0, 1)
        self.tail_k = float(k[-1])                 # 20 ms morph into the tail
        k = k[:, None]
        return chunk * (1 - k) + hp * level * k

    def render(self, frames: int) -> np.ndarray:
        chunk = self.data[self.pos:self.pos + frames]
        self.pos += frames
        if len(chunk) < frames:
            chunk = np.vstack([chunk, np.zeros((frames - len(chunk), 2), dtype=np.float32)])
        if self.tail:
            chunk = self._apply_tail(chunk)
        tgt = getattr(self, "layer_target", 1.0)
        if self.rhythm is not None and (self.layer != tgt or self.layer < 1.0):
            step = frames / max(1.0, getattr(self, "layer_frames", 44100.0))
            new = min(tgt, self.layer + step) if tgt > self.layer else max(tgt, self.layer - step)
            g = np.linspace(self.layer, new, frames, dtype=np.float32)[:, None]
            self.layer = new
            r = self.rhythm[self.pos - frames:self.pos]
            if len(r) < frames:
                r = np.vstack([r, np.zeros((frames - len(r), 2), dtype=np.float32)])
            chunk = chunk - (1.0 - g) * r
        direction = 1.0 if self.target > self.gain else -1.0
        ramp = self.gain + direction * self.step * np.arange(1, frames + 1, dtype=np.float32)
        ramp = np.clip(ramp, min(self.gain, self.target), max(self.gain, self.target))
        self.gain = float(ramp[-1])
        if self.curve == "linear":
            # loop seams join near-identical audio: equal-gain keeps loudness flat
            return chunk * ramp[:, None]
        if self.curve == "blend_in":
            # song-change crossfades: the new song comes up early…
            return chunk * np.sin(np.minimum(1.0, 1.3 * ramp) * (np.pi / 2))[:, None]
        if self.curve == "blend_out":
            # …and the old one holds late, so the overlap never dips in volume
            return chunk * np.cos(np.clip(1.3 * (1.0 - ramp) - 0.3, 0.0, 1.0) * (np.pi / 2))[:, None]
        if self.curve == "tail":
            # behind a full-volume landing: drop away quickly, then a short natural tail
            return chunk * (ramp ** 2.5)[:, None]
        # equal-power curve so crossfades between different music don't dip in loudness
        return chunk * np.sin(ramp * (np.pi / 2))[:, None]

    @property
    def finished(self) -> bool:
        return (self.target == 0.0 and self.gain <= 0.0) or self.pos >= len(self.data)


class LocalPlayer(Player):
    SR = 44100

    def __init__(self, volume: float = 0.8, cache_size: int = 8):
        import sounddevice as sd
        self.volume = volume
        self._decks: list[_Deck] = []
        self._current: _Deck | None = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._timers: list[threading.Timer] = []
        self._pending = None          # (cue, data, trigger_frame, pre_frames): start the switch at trigger
        # Mix gain on top of volume (ducking / background mute), ramped per block.
        self.trim = 1.0
        self.trim_target = 1.0
        self.trim_time = 0.2          # seconds for a full 0→1 move at the current ramp
        # 2026-11-21: stutters (a few ms repeating) while joins were being built = the audio callback
        # missing its deadline (GIL held by the join/decoding threads). Bigger buffer + faster GIL handoff.
        import sys as _sys
        _sys.setswitchinterval(0.001)
        self._stream = sd.OutputStream(samplerate=self.SR, channels=2, dtype="float32",
                                       callback=self._callback, blocksize=2048, latency="high")
        self._stream.start()

    def _decode(self, path: str) -> np.ndarray:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        import librosa
        y, _ = librosa.load(path, sr=self.SR, mono=False)
        y = np.atleast_2d(y)
        if y.shape[0] == 1:
            y = np.vstack([y, y])
        data = np.ascontiguousarray(y[:2].T, dtype=np.float32)
        data *= _level_gain(path, data, self.SR)        # every song at the same loudness
        self._cache[path] = data
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return data

    def preload(self, path: str) -> None:
        threading.Thread(target=self._decode, args=(path,), daemon=True).start()

    def _render(self, frames: int) -> np.ndarray:
        """Mix all decks, splitting the block exactly at a loop seam so it lands sample-accurately."""
        mix = np.zeros((frames, 2), dtype=np.float32)
        done = 0
        while done < frames:
            n = frames - done
            cur = self._current
            if self._pending is not None:
                cue, data, trig, pre = self._pending
                if cur is None or cur.pos >= trig or cur.pos >= len(cur.data):
                    self._pending = None
                    self._start_now(cue, data, pre)
                    continue
                n = min(n, trig - cur.pos)
            if cur is not None and cur.loop is not None and cur.target > 0:
                trigger = cur.loop[1] - cur.loop_xf
                until = trigger - cur.pos
                if until <= 0:
                    self._spawn_loop(cur)
                    continue
                n = min(n, until)
            for d in self._decks:
                mix[done:done + n] += d.render(n)
            done += n
        self._decks = [d for d in self._decks if not d.finished or d is self._current]
        return mix

    def _spawn_loop(self, cur: "_Deck") -> None:
        """At (loop end - xfade) start a twin deck at (loop start - xfade): after the crossfade
        the twin is exactly at loop start when the original would have hit loop end."""
        a, b = cur.loop
        twin = _Deck(cur.data, a - cur.loop_xf, cur.loop_xf, cur.track, curve="linear")
        twin.rhythm, twin.layer = cur.rhythm, cur.layer
        twin.layer_target, twin.layer_frames = getattr(cur, "layer_target", 1.0), getattr(cur, "layer_frames", 44100.0)
        twin.loop, twin.loop_xf = cur.loop, cur.loop_xf
        cur.curve = "linear"
        cur.fade_out(cur.loop_xf)
        cur.loop = None
        self._decks.append(twin)
        self._current = twin
        if self._pending is not None and self._pending[2] >= b - cur.loop_xf:
            cue, data, trig, pre = self._pending   # the switch point wrapped with the loop
            self._pending = (cue, data, trig - (b - a), pre)

    def set_trim(self, target: float, seconds: float) -> None:
        """Smoothly move the mix gain (ducking, background mute) to `target` over ~`seconds`."""
        self.trim_target = max(0.0, min(1.0, target))
        self.trim_time = max(0.01, seconds)

    # ── stingers: one-shots over the music (game events), with the music ducked underneath ──
    def stinger(self, path: str, duck: float = 0.9, gain: float = 1.0) -> bool:
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            print("stinger load failed:", e)
            return False
        if sr != self.SR or data.shape[1] < 2:
            data = np.repeat(data[:, :1], 2, axis=1) if data.shape[1] < 2 else data
        with self._lock:
            wait = 0
            cur = self._current
            if cur is not None and cur.track is not None:      # land on the music's next beat
                nb = cur.track.beat_after(cur.pos / self.SR + 0.03)
                wait = max(0, int((nb - cur.pos / self.SR) * self.SR))
            self._oneshots = getattr(self, "_oneshots", []) + [[data * gain, -wait]]
            self._duck = getattr(self, "_duck", 1.0)
            self._duck_target, self._duck_frames_left = duck, len(data) + wait
        return True

    def _mix_oneshots(self, frames: int, music: np.ndarray) -> np.ndarray:
        shots = getattr(self, "_oneshots", None)
        duck = getattr(self, "_duck", 1.0)
        target = getattr(self, "_duck_target", 1.0) if getattr(self, "_duck_frames_left", 0) > 0 else 1.0
        rate = frames / (0.05 * self.SR) if target < duck else frames / (0.9 * self.SR)   # fast down, slow back
        new = max(target, duck - rate) if target < duck else min(target, duck + rate)
        if new != duck or duck != 1.0:
            music = music * np.linspace(duck, new, frames, dtype=np.float32)[:, None]
        self._duck = new
        if getattr(self, "_duck_frames_left", 0) > 0:
            self._duck_frames_left -= frames
        if shots:
            for sh in shots:
                data, pos = sh
                if pos + frames > 0:
                    off = max(0, -pos)                       # still waiting for the beat
                    chunk = data[max(0, pos):pos + frames]
                    music[off:off + len(chunk)] += chunk
                sh[1] = pos + frames
            self._oneshots = [sh for sh in shots if sh[1] < len(sh[0])]
        return music

    def _callback(self, outdata, frames, _time, _status):
        with self._lock:
            mix = self._render(frames)
            mix = self._mix_oneshots(frames, mix)
        step = frames / (self.trim_time * self.SR)
        start = self.trim
        if self.trim_target > start:
            end = min(self.trim_target, start + step)
        else:
            end = max(self.trim_target, start - step)
        self.trim = end
        ramp = np.linspace(start, end, frames, dtype=np.float32)[:, None]
        np.clip(self.limit(mix * ramp * self.volume), -1.0, 1.0, out=outdata)

    def limit(self, block: np.ndarray, ceiling: float = 0.89) -> np.ndarray:
        """Smooth peak limiter (instead of hard clipping): overlapping parts and layered hits are
        turned down for a moment rather than chopped. Fast attack, ~0.25 s release."""
        g0 = getattr(self, "_lim_gain", 1.0)
        peak = float(np.abs(block).max()) if len(block) else 0.0
        want = min(1.0, ceiling / peak) if peak > 0 else 1.0
        rel = len(block) / (0.25 * self.SR)
        g1 = want if want < g0 else min(want, g0 + rel)
        self._lim_gain = g1
        return block * np.linspace(g0, g1, len(block), dtype=np.float32)[:, None] if g0 != g1 or g1 < 1.0 else block

    def set_loop(self, loop: tuple[float, float] | None, xfade: float = 0.25) -> None:
        """Hold the current deck in a loop region (seconds), or release it (None)."""
        with self._lock:
            cur = self._current
            if cur is None:
                return
            if loop is None:
                cur.loop = None
                return
            a, b = int(loop[0] * self.SR), int(loop[1] * self.SR)
            xf = min(int(xfade * self.SR), a, (b - a) // 4)
            if cur.pos >= b - xf:
                return          # already past this loop's end; the director will re-decide
            cur.loop, cur.loop_xf = (a, b), max(1, xf)

    # Transition styles (cue.style). In every one the destination is at full level ON the beat.
    #   seam  — jump between near-identical bars: short linear crossfade that finishes on the beat
    #   blend — calm hand-overs / scene changes: equal-power crossfade that finishes on the beat
    #   punch — everything else: the new part hits at full level on the downbeat; the old one
    #           becomes a bass-less tail (no double kicks) that fades out after the beat
    #   plain — legacy: fade in from the switch point (cut-outs, where the old part already stopped)
    PUNCH_PRE = 0.004

    def _pre_frames(self, cue: Cue) -> int:
        style = getattr(cue, "style", "plain")
        if style in ("seam", "blend"):
            return int(cue.fade * self.SR)
        if style == "punch":
            fi = getattr(cue, "fade_in", None)
            return int((fi if fi else self.PUNCH_PRE) * self.SR)
        return 0

    def _rhythm_for(self, track):
        """drums+bass stem for vertical layering (cache/stems/<key>_rhythm.flac), if analysed."""
        if track is None:
            return None
        key = "rhythm:" + track.path
        if key in self._cache:
            return self._cache[key]
        try:
            import os
            import soundfile as sf
            from .analysis import cache_key
            p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "stems",
                             cache_key(track.path) + "_rhythm.flac")
            r = sf.read(p, dtype="float32", always_2d=True)[0] if os.path.exists(p) else None
        except Exception:
            r = None
        self._cache[key] = r
        return r

    def set_layer(self, level: float, seconds: float = 0.6) -> None:
        with self._lock:
            for d in self._decks:
                d.layer_target, d.layer_frames = level, max(1.0, seconds * self.SR)
            self._layer_target = level

    def _start_now(self, cue: Cue, data, pre: int = 0):
        """Called with the lock held, exactly on the trigger sample (`pre` frames before the beat)."""
        if pre == -1:                       # learned join: the built audio already holds the old song
            new, s0 = data
            deck = _Deck(new, s0, 1, cue.track, curve="linear")
            deck.gain = deck.target = 1.0
            deck.rhythm = self._rhythm_for(cue.track)
            for d in self._decks:
                d.curve = "linear"
                d.fade_out(int(0.004 * self.SR))      # it continues inside the built audio
            self._decks.append(deck)
            self._current = deck
            return
        style = getattr(cue, "style", "plain")
        if style == "sequence":
            deck = _Deck(data, int(cue.start * self.SR), int(0.02 * self.SR), cue.track, curve="power")
            deck.rhythm = self._rhythm_for(cue.track)
            self._decks.append(deck)
            self._current = deck
            return
        fade = int(cue.fade * self.SR)
        start = int(cue.start * self.SR)
        pre = min(pre, start)
        new_song = self._current is None or self._current.track is not cue.track
        curve = "linear" if style == "seam" else ("blend_in" if style == "blend" and new_song else "power")
        deck = _Deck(data, start - pre, pre if style in ("seam", "blend", "punch") else fade,
                     cue.track, curve=curve)
        deck.rhythm = self._rhythm_for(cue.track)
        for d in self._decks:
            if style == "punch" and d is self._current:
                d.curve = "power" if getattr(cue, "ring", False) else "tail"
                d.fade_out(max(fade, 1))
            elif style in ("seam", "blend") and d is self._current:
                d.curve = "linear" if style == "seam" else ("blend_out" if new_song else "power")
                d.fade_out(max(pre, 1))
            elif style == "xfade" and d is self._current:
                # equal-power crossfade AFTER the beat: the new part starts exactly at its section
                # start and fades up while this one fades down over the same time (no dip, no bleed)
                d.curve = "power"
                d.fade_out(max(fade, 1))
            else:
                d.fade_out(max(fade if style != "seam" else pre, 1))
        self._decks.append(deck)
        self._current = deck

    def _route_data(self, cue: Cue, data: np.ndarray):
        """Build the learned transition (exactly what the A/B clips render: routes._join) into a
        copy of the new song, so track time stays true. Returns (data, frame to start) or None."""
        r = getattr(cue, "route", None)
        cur = self._current
        if not r or cur is None or cur.track is None or cue.at is None:
            return None
        try:
            from . import routes, policy
            pol = policy.current() or {}
            from . import playlists as _pls                          # config "stretch" (e.g. "glide") wins over
            routes.STRETCH = _pls._load().get("stretch") or pol.get("stretch", routes.STRETCH)   # your ratings' pick
            SR = self.SR
            at, t_new = float(cue.at), float(cue.start)
            if cur.loop is not None or at * SR <= cur.pos + 0.05 * SR:
                return None
            pre = cur.data[max(0, int((at - 2.0) * SR)):int(at * SR)].copy()
            out, seams = [pre], []
            tail = routes._join(out, seams, cur.data, at, data, t_new, r["dur"], r["style"], cur.track, cue.track, None)
            body = out[-1]
            s0 = int(round(t_new * SR)) - routes.LAST_SHIFT     # a tempo glide makes the blend longer/shorter
            new = data.copy()
            if s0 < 0:                                          # glide longer than the song's start: make room
                new = np.concatenate([np.zeros((-s0, new.shape[1]), new.dtype), new])
                s0 = 0
            n = min(len(body), len(new) - s0)
            new[s0:s0 + n] = body[:n]
            sp = routes._speed(cur.track, at, cue.track, t_new)
            wash = None
            if r["style"] == "djverb":
                wash = routes._reverb_wash(cur.data, at + r["dur"] * sp, body)
            elif r["style"] == "djblend":
                wash = tail
            if wash is not None:
                e = s0 + n
                m = min(len(wash), len(new) - e)
                new[e:e + m] += wash[:m]
            return new, s0
        except Exception as e:
            print("route build failed, using", r.get("fallback"), ":", e)
            return None

    def _start(self, cue: Cue):
        import time as _t
        t0 = _t.time()
        data = self._decode(cue.track.path)        # may take a moment; the switch beat may pass
        t1 = _t.time()
        built = self._route_data(cue, data)        # learned join, rendered ahead of the beat
        t2 = _t.time()
        with self._lock:
            spare = ((int(cue.at * self.SR) - self._current.pos) / self.SR
                     if self._current is not None and cue.at is not None else None)
            if built is not None and self._current is not None and cue.at is not None                     and int(cue.at * self.SR) > self._current.pos and self._current.loop is None:
                self._pending = (cue, built, int(cue.at * self.SR), -1)   # -1: hard swap onto the built audio
                # 2026-11-23: user hears A/B joins as much better than in game - log whether the join made it
                print(f"[join] {cue.route['style']} built (decode {t1 - t0:.2f}s + build {t2 - t1:.2f}s, "
                      f"{spare:.2f}s to spare)", flush=True)
                return
            if getattr(cue, "route", None):
                why = ("build failed" if built is None and (spare is None or spare > 0) else
                       "beat already passed" if spare is not None and spare <= 0 else
                       "looping" if self._current is not None and self._current.loop is not None else "no current deck")
                print(f"[join] MISSED {cue.route['style']} -> {cue.route.get('fallback') or cue.style}: {why} "
                      f"(decode {t1 - t0:.2f}s + build {t2 - t1:.2f}s, spare {spare if spare is None else round(spare, 2)}s)",
                      flush=True)
                import dataclasses
                cue = dataclasses.replace(cue, route=None, style=cue.route.get("fallback") or cue.style)
            cur = self._current
            pre = self._pre_frames(cue)
            sync = getattr(cue, "sync", None)
            if sync is not None and cur is not None:
                # catch-up: start now, already as far into the new part as the old song is past
                # its big beat, so both stay in phase as if we had switched on that beat
                import dataclasses
                late = max(0.0, cur.pos / self.SR - sync[0])
                self._start_now(dataclasses.replace(cue, start=sync[1] + late), data, 0)
                return
            if cue.at is None or cur is None or cur.track is None:
                self._start_now(cue, data, 0 if getattr(cue, "style", "") == "punch" else pre)
                return
            at = int(cue.at * self.SR)
            if getattr(cue, "style", "") == "sequence":
                # the old song fades out NOW, ending at the switch beat; the new one starts there
                cur.curve = "power"
                cur.fade_out(max(1, at - cur.pos))
                pre = 0
            br = getattr(cue, "bridge", None)
            if br:                                     # call-out that ENDS on the switch beat
                try:
                    import soundfile as sf
                    bd, _ = sf.read(br[0], dtype="float32", always_2d=True)
                    wait = at - cur.pos - int(br[1] * self.SR)
                    if wait >= 0:
                        self._oneshots = getattr(self, "_oneshots", []) + [[bd * 0.9, -wait]]
                except Exception:
                    pass
            if cur.loop is not None and at - pre <= cur.pos:
                at += cur.loop[1] - cur.loop[0]    # the switch beat is on the loop's next pass
            if at - pre <= cur.pos:                # decoding ran past it: take the next beat
                at = int(cur.track.beat_after((cur.pos + pre) / self.SR + 0.01) * self.SR)
            self._pending = (cue, data, at - pre, pre)

    def play(self, cue: Cue) -> None:
        if cue.action == "layer":
            self.set_layer(cue.layer if cue.layer is not None else 1.0,
                           0.25 if (cue.layer or 1.0) >= 1.0 else 0.8)
            return
        if cue.action == "loop":
            self.set_loop(cue.loop)
            return
        self._starting = getattr(self, "_starting", 0) + 1
        def run():
            try:
                self._start(cue)
            finally:
                self._starting -= 1
        threading.Thread(target=run, daemon=True).start()

    def busy(self) -> bool:
        """Still switching (decoding the next song, or waiting for the switch beat): the reported
        position is the OLD song's, so nobody should make decisions from it yet."""
        return getattr(self, "_starting", 0) > 0 or self._pending is not None

    def position(self) -> float | None:
        with self._lock:
            return None if self._current is None else self._current.pos / self.SR

    def now_playing(self):
        with self._lock:
            return None if self._current is None else self._current.track

    def close(self) -> None:
        for t in self._timers:
            t.cancel()
        self._stream.stop()
        self._stream.close()


# ── Spotify ─────────────────────────────────────────────────────────────────

class SpotifyPlayer(Player):
    """Controls the user's own Spotify app. Tracks need `spotify_uri` (see `conductor spotify-link`)."""

    SCOPES = "user-modify-playback-state user-read-playback-state"

    def __init__(self, client_id: str, redirect_uri: str = "http://127.0.0.1:8765/callback",
                 volume: int = 80, cache_path: str = ".spotify_token"):
        import spotipy
        from spotipy.oauth2 import SpotifyPKCE
        self.sp = spotipy.Spotify(auth_manager=SpotifyPKCE(
            client_id=client_id, redirect_uri=redirect_uri, scope=self.SCOPES, cache_path=cache_path))
        self.volume = volume
        self._pos_at = None          # (monotonic, position_s)
        self._worker: threading.Thread | None = None

    def _ramp(self, a: int, b: int, seconds: float, steps: int = 4):
        for i in range(1, steps + 1):
            try:
                self.sp.volume(int(a + (b - a) * i / steps))
            except Exception:
                pass
            time.sleep(seconds / steps)

    def _do(self, cue: Cue):
        if cue.delay > 0:
            time.sleep(cue.delay)
        half = cue.fade / 2
        self._ramp(self.volume, 15, half)
        self.sp.start_playback(uris=[cue.track.spotify_uri], position_ms=int(cue.start * 1000))
        self._pos_at = (time.monotonic(), cue.start)
        self._track = cue.track
        self._ramp(15, self.volume, half)

    def play(self, cue: Cue) -> None:
        if cue.action == "loop":
            return
        if not cue.track.spotify_uri:
            print(f"  ! no Spotify match for {cue.track.title}; skipping")
            return
        print("  ♪", cue.describe())
        self._worker = threading.Thread(target=self._do, args=(cue,), daemon=True)
        self._worker.start()

    def position(self) -> float | None:
        if self._pos_at is None:
            return None
        t0, p0 = self._pos_at
        return p0 + (time.monotonic() - t0)

_LEVELS: dict = {}
LEVEL_TARGET_DB = -11.0     # loud-part level every song is matched to (90th-percentile RMS, dBFS)


def _level_gain(path: str, data: np.ndarray, sr: int) -> float:
    """Gain that brings this song's loud parts to LEVEL_TARGET_DB (cached per song)."""
    import json, os
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "levels.json")
    if not _LEVELS and os.path.exists(cache):
        try:
            _LEVELS.update(json.load(open(cache, encoding="utf-8")))
        except (OSError, ValueError):
            pass
    if path not in _LEVELS:
        m = data.mean(axis=1)
        w = sr // 2
        n = len(m) // w
        if n < 4:
            return 1.0
        rms = np.sqrt((m[:n * w].reshape(n, w) ** 2).mean(axis=1))
        _LEVELS[path] = float(20 * np.log10(np.percentile(rms, 90) + 1e-9))
        try:
            json.dump(_LEVELS, open(cache, "w", encoding="utf-8"), indent=1)
        except OSError:
            pass
    return float(10 ** ((LEVEL_TARGET_DB - _LEVELS[path]) / 20))
