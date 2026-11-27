"""Replay a recorded mission through the StateDirector WITH the live critic attached (no audio).

    python -m conductor.critic_sim [recording.jsonl] [from_s] [to_s] [--off]

The clock runs fast while the critic is idle and at real speed while it is listening, so how often it has
heard the candidates in time is what you'd get in game. Prints every decision (who decided) + a summary.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(rec: str, t0: float = 0.0, t1: float = 1e9, use_critic: bool = True, music: str | None = None):
    music = music or __import__("conductor.playlists", fromlist=["train_music"]).train_music()
    from .events import replay
    from .library import Library
    from .players import NullPlayer
    from .profile import Profile, GameState
    from .ratings import Ratings
    from .states import StateDirector
    from .livecritic import LiveCritic
    lib = Library.build(music, os.path.join(HERE, "cache"), progress=lambda m: None, backfill=False, cached_only=True)
    prof = Profile.load(os.path.join(HERE, "profiles", "darktide.json"))
    state = GameState(prof)
    director = StateDirector(lib, ratings=Ratings(os.path.join(HERE, "cache", "ratings.json")))
    lc = None
    if use_critic:
        lc = LiveCritic()
        w0 = time.time()
        while not lc.ready and time.time() - w0 < 90 and not lc.failed:
            time.sleep(0.2)
        print(f"critic ready: {lc.ready} ({time.time() - w0:.1f}s) {lc.failed or ''}", flush=True)
        director.livecritic, director.music = lc, music
    clock = [0.0]
    player = NullPlayer(clock=lambda: clock[0])
    decisions, wall0 = [], time.time()

    def step(until):
        while clock[0] < until:
            if clock[0] >= t0:
                cue = director.tick(state.intent(clock[0]), clock[0], player.position())
                if cue:
                    by = (cue.route or {}).get("by", "hand-made")
                    decisions.append((clock[0], by))
                    player.play(cue)
                if lc is not None and lc.busy():
                    time.sleep(0.5)                      # the critic is working: game time = real time
            clock[0] += 0.5
    for ev in replay(rec):
        t = ev.data.get("t")
        if t is not None and t > t1:
            break
        if t is not None and t > clock[0]:
            step(t)
        state.apply(ev, clock[0])
    step(clock[0] + 5)
    n = len(decisions)
    crit = sum(1 for _, b in decisions if str(b).startswith("critic"))
    trees = sum(1 for _, b in decisions if str(b).startswith("trees"))
    print(f"\n{n} changes: {crit} decided by the critic, {trees} by the trees, {n - crit - trees} hand-made/other"
          f"  ({time.time() - wall0:.0f}s wall)", flush=True)
    if lc is not None:
        print("critic stats:", lc.stats, flush=True)
        lc.close()


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    rec = a[0] if a else os.path.join(HERE, "recordings", "darktide_2026-11-22_181633.jsonl")
    main(rec, float(a[1]) if len(a) > 1 else 0.0, float(a[2]) if len(a) > 2 else 1e9, "--off" not in sys.argv)
