"""Per-song ratings from the player: +1 liked, -1 banned, absent = neutral.

Stored as JSON in the cache folder, keyed by file path (title kept for readability):
    {"<music folder>/x.mp3": {"title": "x", "rating": 1}}
"""
from __future__ import annotations

import json
import os

LIKE, NEUTRAL, BAN = 1, 0, -1


class Ratings:
    def __init__(self, path: str):
        self.path = path
        self._data: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self._data = json.load(f)

    def get(self, track_path: str) -> int:
        return self._data.get(track_path, {}).get("rating", NEUTRAL)

    def set(self, track_path: str, title: str, rating: int) -> None:
        if rating == NEUTRAL:
            self._data.pop(track_path, None)
        else:
            self._data[track_path] = {"title": title, "rating": rating}
        self.save()

    def rate(self, track_path: str, title: str, value: int) -> int:
        """Apply a thumbs press. Pressing the same rating again clears it. Returns the new rating."""
        new = NEUTRAL if self.get(track_path) == value else value
        self.set(track_path, title, new)
        return new

    def items(self):
        return sorted(self._data.items(), key=lambda kv: (kv[1]["rating"], kv[1]["title"].lower()))

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=1)
        os.replace(tmp, self.path)
