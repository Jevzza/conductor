"""Playlists = named music folders, kept in config.json ("playlists": {name: folder}).

playlists.txt (next to config.json) is what the game mod reads to build its dropdown:
one line per playlist, "name|ready|total" (ready = songs fully analysed and playable).
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(HERE, "config.json")
INDEX_PATH = os.path.join(HERE, "playlists.txt")
CACHE = os.path.join(HERE, "cache")


def _load() -> dict:
    if not os.path.exists(CONFIG_PATH):      # fresh install: the app fills in defaults when it loads
        return {}
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save(cfg: dict) -> None:
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)


def all_playlists(cfg: dict | None = None) -> dict[str, str]:
    cfg = cfg or _load()
    pl = dict(cfg.get("playlists") or {})
    if not pl and cfg.get("music"):
        pl[os.path.basename(os.path.normpath(cfg["music"]))] = cfg["music"]
    return pl


def train_music() -> str:
    """The music folder the trainer and training tools work on: config "train_music" if set, else the
    current playlist, else the first one ("" when there are none)."""
    cfg = _load()
    if cfg.get("train_music"):
        return cfg["train_music"]
    pl = all_playlists(cfg)
    return pl.get(cfg.get("playlist")) or next(iter(pl.values()), "")


def train_folders() -> list[str]:
    """config "train_folders" if set, else train_music() first, then every other registered playlist folder."""
    cfg = _load()
    if cfg.get("train_folders"):
        return list(cfg["train_folders"])
    tm = train_music()
    rest = [f for f in all_playlists().values() if os.path.isdir(f)]
    return list(dict.fromkeys(([tm] if tm else []) + rest))


def register(name: str, folder: str) -> None:
    cfg = _load()
    pl = all_playlists(cfg)
    pl[name] = folder
    cfg["playlists"] = pl
    _save(cfg)


def set_current(name: str) -> None:
    cfg = _load()
    cfg["playlist"] = name
    _save(cfg)


def counts(folder: str) -> tuple[int, int]:
    """(ready, total): ready = has a merged ML analysis in the cache."""
    from .analysis import cache_key
    from .library import scan
    files = scan(folder) if os.path.isdir(folder) else []
    ready = 0
    for p in files:
        c = os.path.join(CACHE, cache_key(p) + ".json")
        if os.path.exists(c):
            try:
                with open(c, encoding="utf-8") as f:
                    if json.load(f).get("ml"):
                        ready += 1
            except (OSError, ValueError):
                pass
    return ready, len(files)


# where downloads go and pasted-in folders are found: config "playlists_dir", else playlists/ next to config.json
DATA = _load().get("playlists_dir") or os.path.join(HERE, "playlists")


def discover() -> list[str]:
    """user 2026-11-26: songs pasted into a folder under the mod's data/playlists show up in the dropdown
    by themselves (marked not processed) - picking one processes it. Returns the names it added."""
    from .library import scan
    added = []
    if not os.path.isdir(DATA):
        return added
    have = {os.path.normcase(os.path.normpath(f)) for f in all_playlists().values()}
    for d in sorted(os.listdir(DATA)):
        f = os.path.join(DATA, d)
        if os.path.isdir(f) and os.path.normcase(os.path.normpath(f)) not in have and scan(f):
            name = d
            while name in all_playlists():
                name += " (local)"
            register(name, f)
            added.append(name)
    return added


def write_index() -> list[tuple[str, int, int]]:
    try:
        discover()
    except Exception as e:
        print("playlist discover:", e)
    rows = []
    for name, folder in all_playlists().items():
        ready, total = counts(folder)
        rows.append((name, ready, total))
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for name, ready, total in rows:
            f.write(f"{name.replace('|', '/')}|{ready}|{total}\n")
    os.replace(tmp, INDEX_PATH)
    return rows
