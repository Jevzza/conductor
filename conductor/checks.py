"""Automatic checks on a rendered mission (timeline_*.json from showcase_real): the rules your
review pins taught us, measured, so a new version is checked before anyone listens to it.

    python -m conductor.checks <timeline.json> [<timeline.json> ...]
"""
from __future__ import annotations

import json
import os
import sys

from .library import Library
from .states import role_of

PHASE_STATE = {"explore": "calm", "tier1": "fight", "tier2": "horde", "tier3": "boss"}
FITS = {   # roles acceptable in each state (wider than the director's first choice)
    "calm": {"explore", "bridge", "build"},
    "fight": {"skirmish", "horde", "build"},
    "horde": {"horde", "peak", "skirmish"},
    "boss": {"peak", "horde"},
}
WRONG = {  # clearly wrong: what each pin complained about
    "calm": {"horde", "peak"},
    "fight": {"explore", "bridge"},
    "horde": {"explore", "bridge", "build"},
    "boss": {"explore", "bridge", "build", "skirmish"},
}


def run(path: str, lib: Library) -> dict:
    tl = json.load(open(path, encoding="utf-8"))
    find = {}
    for t in lib.tracks:
        find[t.title] = t
    def track(name):
        if name not in find:
            find[name] = next(t for t in lib.tracks if t.title in name or name in t.title)
        return find[name]
    phases = tl["phases"]
    def state(t):
        p = [x["phase"] for x in phases if x["t"] <= t]
        return PHASE_STATE.get(p[-1]) if p else None
    r = {"seconds": 0, "wrong": 0, "wrong_by": {}, "fit": 0, "songs": 0, "song_changes_in_fight": 0,
         "moves": 0, "late": [], "boss_seconds": 0, "boss_ok": 0}
    prev_song, last_state, changed_at, waiting = None, None, None, False
    for t, song, x in tl["pos"]:
        st = state(t)
        if st is None:
            continue
        role = role_of(track(song).segment_at(x))
        r["seconds"] += 1
        if role in WRONG[st]:
            r["wrong"] += 1
            r["wrong_by"][f"{st}<-{role}"] = r["wrong_by"].get(f"{st}<-{role}", 0) + 1
        if role in FITS[st]:
            r["fit"] += 1
        if st == "boss":
            r["boss_seconds"] += 1
            r["boss_ok"] += role in ("peak", "horde")
        if song != prev_song:
            if prev_song is not None:
                r["songs"] += 1
                if st != "calm":
                    r["song_changes_in_fight"] += 1
            prev_song = song
        if st != last_state:
            last_state, changed_at, waiting = st, t, True
        if waiting and role in FITS[st]:
            waiting = False
            if t - changed_at > 4:
                r["late"].append((round(changed_at), st, round(t - changed_at)))
    # flapping: the heard part changes more than twice within 8 s
    parts = []
    for t, song, x in tl["pos"]:
        k = (song, track(song).segment_at(x).start)
        if not parts or parts[-1][1] != k:
            parts.append((t, k))
    times = [t for t, _ in parts]
    r["flaps"] = sorted({round(times[i]) for i in range(len(times) - 2) if times[i + 2] - times[i] <= 8})
    # heard delay: game state change -> the audio's part changes
    lat = []
    for a, b in zip(phases, phases[1:]):
        if a["phase"] in PHASE_STATE and b["phase"] in PHASE_STATE and PHASE_STATE[a["phase"]] != PHASE_STATE[b["phase"]]:
            nxt = next((t for t in times if b["t"] - 0.5 <= t <= b["t"] + 15), None)
            if nxt is not None:
                lat.append(round(nxt - b["t"], 1))
    lat.sort()
    r["heard_median"] = lat[len(lat) // 2] if lat else None
    r["heard_slow"] = sum(1 for x in lat if x > 3)
    r["moves"] = sum(1 for e in tl["music"] if e["text"].startswith(("→", "Song:")))
    r["fit_pct"] = round(100 * r["fit"] / max(1, r["seconds"]))
    r["wrong_pct"] = round(100 * r["wrong"] / max(1, r["seconds"]))
    return r


def silences(mp3: str, thr_db: float = -40.0, min_s: float = 0.75) -> list:
    """Stretches where the rendered music is silent (fade-outs looped, dead air between parts)."""
    import os, subprocess, tempfile
    import numpy as np
    import soundfile as sf
    if not os.path.exists(mp3):
        return []
    wav = os.path.join(tempfile.gettempdir(), "conductor_check.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-i", mp3, "-ac", "1", "-ar", "22050", "-y", wav], check=True)
    a, sr = sf.read(wav, dtype="float32")
    w = int(0.25 * sr); n = len(a) // w
    q = 20 * np.log10(np.sqrt((a[:n * w].reshape(n, w) ** 2).mean(1)) + 1e-9) < thr_db
    out, i = [], 0
    while i < n:
        if q[i]:
            j = i
            while j < n and q[j]:
                j += 1
            if (j - i) * 0.25 >= min_s:
                out.append((round(i * 0.25, 1), round((j - i) * 0.25, 2)))
            i = j
        else:
            i += 1
    return out


def dips(mp3: str, tl: dict, drop_db: float = 8.0) -> list:
    """Music changes where the level ducks well below the music around it (fade-to-nothing joins)."""
    import os, subprocess, tempfile
    import numpy as np
    import soundfile as sf
    wav = os.path.join(tempfile.gettempdir(), "conductor_check.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-i", mp3, "-ac", "1", "-ar", "22050", "-y", wav], check=True)
    a, sr = sf.read(wav, dtype="float32")
    w = sr // 8
    n = len(a) // w
    db = 20 * np.log10(np.sqrt((a[:n * w].reshape(n, w) ** 2).mean(1)) + 1e-9)
    out = []
    for e in tl["music"]:
        if e["text"][:1] not in ("→", "S"):
            continue
        t = e["t"]
        i0, i1 = int((t - 3) * 8), int((t + 3) * 8)
        around = np.r_[db[max(0, int((t - 8) * 8)):max(0, i0)], db[i1:int((t + 8) * 8)]]
        if i0 < 0 or i1 >= n or len(around) < 8:
            continue
        ref = float(np.median(around))
        low = float(db[i0:i1].min())
        if ref > -40 and ref - low >= drop_db:
            out.append((round(t, 1), round(ref - low, 1)))
    return out


def main(paths):
    lib = Library.build(__import__("conductor.playlists", fromlist=["train_music"]).train_music(), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"), progress=lambda m: None,
                        backfill=False, cached_only=True)
    for extra in filter(None, __import__("os").environ.get("CONDUCTOR_EXTRA_LIBS", "").split("|")):   # e.g. a test with more songs
        lib.tracks += Library.build(extra, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"), progress=lambda m: None,
                                    backfill=False, cached_only=True).tracks
    for p in paths:
        r = run(p, lib)
        late = r["late"]
        print(f"\n{p}\n  fits {r['fit_pct']}%   clearly wrong {r['wrong_pct']}%  {r['wrong_by']}"
              f"\n  boss on boss music {r['boss_ok']}/{r['boss_seconds']} s   music moves {r['moves']}"
              f"   song changes {r['songs']} (mid-fight {r['song_changes_in_fight']})"
              f"\n  heard delay after game change: median {r['heard_median']} s, >3 s: {r['heard_slow']}"
              f"   flapping (3 parts in 8 s): {len(r['flaps'])} {r['flaps'][:12]}"
              f"\n  slow to react (>4 s): {len(late)}  median {sorted(x[2] for x in late)[len(late)//2] if late else 0} s"
              f"  {late[:12]}")
        import os
        d = os.path.dirname(p)
        mp3 = next((os.path.join(d, f) for f in (os.path.basename(p).replace("timeline_", "music_").replace(".json", ".mp3"),
                                                  "showcase_conductor.mp3") if os.path.exists(os.path.join(d, f))), "")
        if mp3:
            sil = silences(mp3)
            print(f"  silent gaps (>0.75 s): {len(sil)} {sil[:10]}")
            dp = dips(mp3, json.load(open(p, encoding="utf-8")))
            print(f"  level dips at changes (>8 dB): {len(dp)} {dp[:10]}")


if __name__ == "__main__":
    main(sys.argv[1:])
