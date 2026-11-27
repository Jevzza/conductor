"""Human-readable music feed: what the music is doing and why, for the game chat and the log.

Writes %TEMP%\\conductor_feed.txt: one "id|text" line per event (last 30 kept). The bridge echoes
lines with a newer id when "Show music changes in chat" is on.
"""
from __future__ import annotations

import os
import time

FEED_PATH = os.path.join(os.environ.get("TEMP", "/tmp"), "conductor_feed.txt")
PHASE_SETTLE = 3.0          # a phase must hold this long before it's reported (game states flicker)

SCENE_LABEL = {
    "hub": "Hub · calm", "menu": "Menu · calm", "loadout": "Loadout · building",
    "loading": "Valkyrie · building up", "intro": "Mission intro · calm", "cinematic": "Cinematic",
    "victory": "Victory", "defeat": "Defeat",
}
TIER_LABEL = {0: "Exploring · calm", 1: "Fight · skirmish", 2: "Fight · horde", 3: "Fight · peak"}

WHY = [  # first matching reason fragment wins
    ("st:mission", ""), ("st:boss", "boss!"), ("st:build", "enemies closing in"), ("st:start", "fight started"),
    ("st:up", "fight got bigger"), ("st:calm", "fight over"), ("st:down", "fight eased"), ("st:next", "next part"),
    ("via-bridge", "the fight changed"), ("via-build", "building up to it"), ("exit-bridge", "heading for a new song"),
    ("in-breakdown", "new song during the breakdown"),
    ("bridge", "mixing it up"), ("lull", "fighting eased"), ("fatigue", "time for a fresh song"), ("climb", "fight got bigger, bigger part"), ("escalate", "fight got bigger"), ("deescalate", "fight eased off"), ("handover:calm", "keeping it calm"),
    ("handover", "calm moment, new song"), ("song-end", "song ended"), ("skip", "skipped"),
    ("winddown", "winding down"), ("calm:", "keeping it calm"), ("cutout", "fight over"),
    ("slam", "fight started"), ("commit", "different tempo, hard cut"), ("tired", "fresh part"),
    ("combat:", "fight started"), ("explore:", "fight over"), ("energy:", "following intensity"),
    ("scene:", ""),
]
SECTION = {"peak": "peak", "drop": "drop", "build": "build-up", "groove": "groove", "calm": "calm part",
           "release": "calm-down"}


def _why(reason: str) -> str:
    parts = []
    for frag, text in WHY:
        if frag in reason:
            if text:
                parts.append(text)
            break
    if ":drop" in reason or ("slam" in reason and "escalate" not in reason):
        parts.append("on the drop")
    if "~" in reason:
        parts.append("seamless")
    if "via-bridge" in reason:
        parts[:] = [p for p in parts if p != "the fight changed"] + ["through the song's breakdown"]
    if ":entry" in reason:
        parts.append("in through its build")
    if ":callout" in reason:
        parts.append("with a call-out")
    return ", ".join(parts)


JOIN_NAMES = {"xfade1": "1-bar blend", "xfade2": "2-bar blend", "xfade4": "4-bar blend", "dj": "DJ swap on the drop",
              "djverb": "DJ swap + reverb tail", "djblend": "DJ overhang", "mix": "beat-mixed, basses swap",
              "echo": "echo out", "punch": "punch in", "game": "Darktide-style (game's own rules)"}


class Feed:
    def __init__(self):
        self.lines: list[str] = []
        self.id = int(time.time())          # ids keep rising across app restarts
        self._phase = None
        self._phase_pending = None
        self._song = None

    def say(self, text: str, chat: bool = True) -> None:
        """chat=False: the app log only. User 2026-11-23: chat shows only what the MUSIC did (song / part /
        layer changes); game-state lines there made it look busy when it wasn't doing anything."""
        if not chat:
            print("·", text)
            return
        now = time.time()
        if text == getattr(self, "_last_text", None) and now - getattr(self, "_last_at", 0.0) < 30:
            print("•", text, "(repeat, not shown in chat)")
            return
        self._last_text, self._last_at = text, now
        self.id += 1
        self.lines = (self.lines + [f"{self.id}|{text}"])[-30:]
        print("•", text)
        try:
            tmp = FEED_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(self.lines) + "\n")
            os.replace(tmp, FEED_PATH)
        except OSError:
            pass

    def intent(self, intent, now: float) -> None:
        if intent is None:
            return
        label = SCENE_LABEL.get(intent.scene) if not intent.dynamic else \
            TIER_LABEL.get(intent.tier if intent.mode == "combat" else 0, "Fight")
        label = label or intent.scene.capitalize()
        if label == self._phase:
            self._phase_pending = None
            return
        if self._phase_pending is None or self._phase_pending[0] != label:
            self._phase_pending = (label, now)
            if not intent.dynamic or intent.urgent:
                self._phase_pending = (label, now - PHASE_SETTLE)   # scenes & boss/urgent: say it now
        if now - self._phase_pending[1] >= PHASE_SETTLE:
            self._phase = label
            self._phase_pending = None
            self.say(label, chat=False)            # game state, not a music change: log only

    def cue(self, cue) -> None:  # noqa: C901
        if cue.action == "loop":
            return                                   # holding a loop is invisible by design
        if cue.action == "layer":
            self.say("drums & bass down  (fighting eased)" if (cue.layer or 1) < 1 else "drums & bass back in")
            return
        why = _why(cue.reason)
        r = getattr(cue, "route", None)
        if r:                                        # the learned join (policy.json)
            why = (why + " · " if why else "") + "join: " + JOIN_NAMES.get(r["style"], r["style"])
        if why.startswith("fight started") and (self._phase or "").startswith("Fight"):
            why = why.replace("fight started", "following the fight", 1)   # already fighting
        if cue.track.title != self._song:
            self._song = cue.track.title
            self.say(f"Song: {cue.track.title}" + (f"  ({why})" if why else ""))
        else:
            seg = cue.track.segment_at(cue.start)
            part = getattr(seg, "user_role", None) or SECTION.get(seg.label, seg.label)
            if seg.bridge:
                part = "breakdown"
            self.say(f"→ {part}" + (f"  ({why})" if why else ""))
