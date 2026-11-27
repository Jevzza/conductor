"""Critic Lab: live status files for the dashboard (scratchpad lab page polls these).
Every long-running ML step calls lab.update(section, **fields) / lab.append(section, row)."""
from __future__ import annotations

import json
import os
import time

LAB = os.environ.get("CONDUCTOR_LAB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "lab"))


def _path(section):
    os.makedirs(LAB, exist_ok=True)
    return os.path.join(LAB, section + ".json")


def _read(section):
    try:
        return json.load(open(_path(section), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(section, d):
    d["updated"] = time.time()
    tmp = _path(section) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f)
    for i in range(20):                  # the dashboard may be reading it (Windows locks): retry briefly
        try:
            os.replace(tmp, _path(section))
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))


def update(section, **fields):
    d = _read(section)
    d.update(fields)
    _write(section, d)


def append(section, key, row, keep=2000):
    d = _read(section)
    d.setdefault(key, []).append(row)
    d[key] = d[key][-keep:]
    _write(section, d)


def reset(section, **fields):
    _write(section, dict(fields))
