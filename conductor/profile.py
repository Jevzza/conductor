"""Game profile: turns a game's raw music states into game-agnostic musical intent.

A profile is pure data (JSON) shipped with each bridge. The core only ever reasons about
the resulting Intent: which scene we're in, how much energy the music should have, how
much tension is building, and whether a change is urgent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Intent:
    scene: str
    energy: float                 # 0..1 target intensity
    tension: float                # 0..1 "something is building"
    urgent: bool                  # change NOW, don't wait for a phrase boundary
    energy_range: tuple[float, float] = (0.0, 1.0)
    prefer: tuple[str, ...] = ()  # preferred segment labels for this scene
    dynamic: bool = False         # does energy track gameplay inside this scene?
    family: str = ""              # scenes in the same family keep the current song (e.g. menus/loading)
    mode: str = "explore"         # explore | combat — which pool of song sections fits right now
    tier: int = 0                 # game intensity: 0 calm, 1 skirmish, 2 horde, 3 peak (boss/escape)
    building: bool = False        # enemies are closing in but the fight hasn't started: build-up time

    def describe(self) -> str:
        flags = " URGENT" if self.urgent else ""
        return (f"{self.scene:<9} {self.mode:<7} tier={self.tier} energy={self.energy:.2f} "
                f"tension={self.tension:.2f}{flags}")


@dataclass
class Profile:
    game: str
    scenes: list[dict]
    energy_rules: list[dict] = field(default_factory=list)
    param_rules: list[dict] = field(default_factory=list)
    tension_rules: list[dict] = field(default_factory=list)
    urgent: list[dict] = field(default_factory=list)
    combat_when: list[dict] = field(default_factory=list)
    combat_hold: float = 0.0      # stay in combat mode this long after combat states stop (they flicker)
    combat_sustain: list[dict] = field(default_factory=list)  # states that keep an ongoing fight alive
    pre_combat: dict | None = None    # early warning that a fight is about to start
    tiers: dict | None = None         # game intensity tier rules
    local: dict | None = None         # what is fighting ME (vs the game's global horde flag)

    tuning: dict = field(default_factory=dict)   # director tuning overrides for this direction
    direction: str = "conductor"

    @classmethod
    def load(cls, path: str, direction: str = "conductor") -> "Profile":
        """`direction` picks a gameplay variant from the profile's "directions" (scenes are shared)."""
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        over = (raw.get("directions") or {}).get(direction) or {}
        raw = {**raw, **{k: v for k, v in over.items() if not k.startswith("_")}}
        return cls(
            tuning=dict(raw.get("tuning") or {}),
            direction=direction if over else "conductor",
            game=raw["game"],
            scenes=raw["scenes"],
            energy_rules=raw.get("energy_rules", []),
            param_rules=raw.get("param_rules", []),
            tension_rules=raw.get("tension_rules", []),
            urgent=raw.get("urgent", []),
            combat_when=raw.get("combat_when", []),
            combat_hold=float(raw.get("combat_hold", 0.0)),
            combat_sustain=raw.get("combat_sustain", []),
            pre_combat=raw.get("pre_combat"),
            tiers=raw.get("tiers"),
            local=raw.get("local"),
        )


class GameState:
    """Accumulates bridge events into current state and derives an Intent from a Profile."""

    def __init__(self, profile: Profile):
        self.profile = profile
        self.states: dict[str, str] = {}
        self.params: dict[str, float] = {}
        self._combat_seen: float | None = None   # last time (caller's clock) combat states were active
        self._combat_energy = 0.0                 # energy while last actually fighting
        self._pre_since: float | None = None      # when the early-warning state last switched on
        self._hot_seen: float | None = None       # last time enemies were actually fighting me

    def _combat_now(self) -> bool:
        return any(self.states.get(r["group"]) in r["values"] for r in self.profile.combat_when)

    def _pre_on(self) -> bool:
        pc = self.profile.pre_combat
        if not pc:
            return False
        if self.states.get(pc["group"]) in pc["values"]:
            return True
        return any(self.params.get(n, 0) >= th for n, th in pc.get("params", {}).items())

    def _hot(self, big: bool = False) -> bool:
        lc = self.profile.local or {}
        for g, vals in lc.get("big_states" if big else "hot_states", {}).items():
            if self.states.get(g) in vals:
                return True
        if big:
            return False
        return any(self.params.get(n, 0) >= th for n, th in lc.get("hot_params", {}).items())

    def apply(self, ev, now: float | None = None) -> bool:
        pc = self.profile.pre_combat
        was = self._pre_on()
        changed = self._apply(ev)
        if now is not None and self._hot():
            self._hot_seen = now
        if pc and now is not None and not was and self._pre_on() and not self._combat_now():
            self._pre_since = now
        if now is not None and self._combat_now():
            self._combat_seen = now
        return changed

    def _apply(self, ev) -> bool:
        """Apply an event; returns True if state changed."""
        if ev.kind == "hello":
            self.states.clear()
            self.params.clear()
            self._combat_seen = None
            self._hot_seen = None
            return True
        if ev.kind == "state":
            g, v = ev.data.get("group"), ev.data.get("value")
            if g and self.states.get(g) != v:
                self.states[g] = v
                return True
        elif ev.kind == "param":
            n, v = ev.data.get("name"), ev.data.get("value")
            if n is not None and isinstance(v, (int, float)):
                self.params[n] = float(v)
                return True
        return False

    def _scene(self) -> dict | None:
        prev = getattr(self, "_last_family", None)
        for scene in self.profile.scenes:
            if all(self.states.get(g) in vals for g, vals in scene["when"].items()):
                found = scene
                break
            # "after": also this scene when coming straight from another family (e.g. the brief
            # loading state between the hub and the loadout screen is part of the loadout)
            after = scene.get("after")
            if after and prev in after.get("families", []) and                     all(self.states.get(g) in vals for g, vals in after["when"].items()):
                found = scene
                break
        else:
            return None
        # Remember where we came from only once a scene has held for a second: state changes
        # arrive one group at a time, so in-between combinations can match the wrong scene.
        fam = found.get("family") or found["name"]
        now = getattr(self, "_now", None)
        cand = getattr(self, "_fam_cand", None)
        if cand is None or cand[0] != fam:
            self._fam_cand = (fam, now)
        elif now is not None and cand[1] is not None and now - cand[1] >= 1.0:
            self._last_family = fam
        if prev is None:
            self._last_family = fam
        return found

    def intent(self, now: float | None = None) -> Intent | None:
        self._now = now
        scene = self._scene()
        if scene is None:
            return None
        lo, hi = scene.get("energy", [0.0, 1.0])
        energy = float(scene.get("base_energy", lo))
        tension = 0.0
        if scene.get("dynamic"):
            add = 0.0
            for rule in self.profile.energy_rules:
                val = rule["values"].get(self.states.get(rule["group"]))
                if val is None:
                    continue
                if rule["mode"] == "floor":
                    energy = max(energy, val)
                else:
                    add += val
            for rule in self.profile.param_rules:
                v = self.params.get(rule["name"])
                if v is None:
                    continue
                a, b = rule["range"]
                frac = min(1.0, max(0.0, (v - a) / (b - a))) if b > a else 0.0
                add += frac * rule["max"]
            energy += add
            for rule in self.profile.tension_rules:
                val = rule["values"].get(self.states.get(rule["group"]))
                if val is not None:
                    tension = max(tension, val)
        fighting = self._combat_now()
        lc = self.profile.local
        if now is not None and self._hot():
            self._hot_seen = now
        cold = (lc is not None and now is not None and
                (self._hot_seen is None or now - self._hot_seen >= lc.get("lull_after", 10)))
        must_fight = (self.states.get("music_combat") == "boss"
                      or self.states.get("music_objective") in ("escape_event", "last_man_standing"))
        if fighting and cold and not must_fight:
            fighting = False                     # the game still says "horde", but nothing is near me
            energy = min(energy, float(scene.get("base_energy", lo)) + 0.1)
        pc = self.profile.pre_combat
        building = False
        if not fighting and pc and now is not None and self._pre_since is not None \
                and now - self._pre_since < pc.get("window", 6) and self._pre_on():
            fighting = building = True           # the fight is about to start
        # Enemies still aggroed = the fight is pausing, not over: keep the hold refreshed.
        sustained = self._combat_seen is not None and any(
            self.states.get(r["group"]) in r["values"] for r in self.profile.combat_sustain)
        if (fighting or sustained) and now is not None:
            self._combat_seen = now
        held = (not fighting and now is not None and self._combat_seen is not None
                and now - self._combat_seen < self.profile.combat_hold
                and not (cold and not must_fight))     # the game's flag alone doesn't hold a fight
        combat = bool(scene.get("dynamic")) and (fighting or held)
        # Combat flags blink off between waves; during the hold keep the fight's energy.
        if scene.get("dynamic"):
            if fighting:
                self._combat_energy = energy
            elif held:
                energy = max(energy, self._combat_energy)
        energy = min(hi, max(lo, energy))
        urgent = any(self.states.get(r["group"]) in r["values"] for r in self.profile.urgent)
        tier = 0
        if combat and self.profile.tiers:
            tier = 1
            for r in self.profile.tiers.get("states", []):
                tier = max(tier, int(r["values"].get(self.states.get(r["group"]), 0)))
            for r in self.profile.tiers.get("params", []):
                v = self.params.get(r["name"])
                if v is not None:
                    for th, tt in r["at"]:
                        if v >= th:
                            tier = max(tier, int(tt))
            if lc is not None and now is not None:
                boss = self.states.get("music_combat") == "boss"
                if tier >= 2 and not boss and (self._hot_seen is None or now - self._hot_seen >= lc.get("cool_after", 6)):
                    tier = 1                     # a horde somewhere, but not on me
                if self._hot(big=True):
                    tier = max(tier, 2)          # lots of enemies on me, whatever the game calls it
            if building:
                tier = 1
            # the energy follows the tier we settled on, not the game's raw horde floor
            cap = {1: 0.66, 2: 0.85}.get(tier)
            if cap is not None:
                energy = min(energy, cap)
            if self._hot(big=True):
                energy = max(energy, 0.74)
        # Steady the state like the game's own music: up at once, down only once it has held.
        if scene.get("dynamic") and now is not None:
            level = tier if combat else 0
            cur = getattr(self, "_level", None)
            down_hold = float((self.profile.local or {}).get("down_hold", 6.0))
            if cur is None or level >= cur:
                self._level, self._down_since = level, None
            else:
                if self._down_since is None:
                    self._down_since = now
                if now - self._down_since >= down_hold:
                    self._level, self._down_since = level, None
                else:
                    level = cur                       # still counts as the bigger state
                    combat, tier = level > 0, level
                    energy = max(energy, getattr(self, "_level_energy", energy))
            if level == self._level:
                self._level_energy = energy
        elif not scene.get("dynamic"):
            self._level, self._down_since = None, None
        return Intent(
            scene=scene["name"],
            energy=round(energy, 3),
            tension=round(tension, 3),
            urgent=urgent,
            energy_range=(lo, hi),
            prefer=tuple(scene.get("prefer", ())),
            dynamic=bool(scene.get("dynamic")),
            family=scene.get("family") or scene["name"],
            mode="combat" if combat else "explore",
            tier=tier,
            building=building,
        )
