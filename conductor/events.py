"""Bridge transport: tail the JSON-lines file a game bridge writes.

The core never talks to a game directly. A bridge (e.g. the Darktide Lua mod) appends
one JSON object per line; this module turns that into GameEvent objects. A "hello"
event means the bridge restarted and truncated the file.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterator


DEFAULT_EVENT_PATH = os.path.join(os.environ.get("TEMP", "/tmp"), "conductor_events.jsonl")


@dataclass
class GameEvent:
    kind: str                      # hello | state | param | stinger
    data: dict = field(default_factory=dict)
    received: float = 0.0          # wall-clock time the core saw it

    @classmethod
    def from_line(cls, line: str, received: float | None = None) -> "GameEvent | None":
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        kind = obj.pop("k", None)
        if not kind:
            return None
        return cls(kind=kind, data=obj, received=received if received is not None else time.time())


class FileTail:
    """Follows the bridge file, coping with truncation (bridge restart) and partial lines."""

    def __init__(self, path: str = DEFAULT_EVENT_PATH, from_start: bool = True):
        self.path = path
        self._pos = 0
        self._buf = ""
        self._from_start = from_start

    def poll(self) -> list[GameEvent]:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []
        if not self._from_start:
            self._pos, self._from_start = size, True
        if size < self._pos:          # truncated → bridge restarted
            self._pos, self._buf = 0, ""
        if size == self._pos:
            return []
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(self._pos)
            chunk = f.read()
            self._pos = f.tell()
        self._buf += chunk
        *lines, self._buf = self._buf.split("\n")
        now = time.time()
        return [e for e in (GameEvent.from_line(l, now) for l in lines if l.strip()) if e]


def replay(path: str, speed: float = 0.0) -> Iterator[GameEvent]:
    """Yield events from a recorded log. Uses the bridge's game time `t` for pacing if speed > 0."""
    last_t = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            ev = GameEvent.from_line(line)
            if not ev:
                continue
            t = ev.data.get("t")
            if speed > 0 and t is not None and last_t is not None and t > last_t:
                time.sleep((t - last_t) / speed)
            if t is not None:
                last_t = t
            yield ev
