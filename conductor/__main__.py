"""Conductor CLI.

  python -m conductor analyze  --music <music folder>
  python -m conductor run      --music <music folder> --profile profiles/darktide.json [--player local|spotify|none]
  python -m conductor replay   --music ... --profile ... recorded_events.jsonl
  python -m conductor spotify-link --music ... --client-id XXXX
  python -m conductor watch    [--events path]      # just print what the bridge sends
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

from .director import Director
from .events import DEFAULT_EVENT_PATH, FileTail, replay
from .ratings import BAN, LIKE, NEUTRAL, Ratings
from .library import Library
from .profile import GameState, Profile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(HERE, "cache")
DEFAULT_PROFILE = os.path.join(HERE, "profiles", "darktide.json")
# Plain text so the game bridge can read it trivially: line 1 = title, line 2 = rating (-1/0/1).
STATUS_PATH = os.path.join(os.environ.get("TEMP", "/tmp"), "conductor_status.txt")
RATING_WORD = {LIKE: "liked", NEUTRAL: "neutral", BAN: "banned"}


def _ratings(args) -> Ratings:
    return Ratings(os.path.join(args.cache, "ratings.json"))


def _write_status(track, ratings: Ratings) -> None:
    try:
        tmp = STATUS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(f"{track.title}\n{ratings.get(track.path)}\n" if track else "\n0\n")
        os.replace(tmp, STATUS_PATH)
    except OSError:
        pass


def _handle_user(ev, player, director, ratings: Ratings) -> None:
    if ev.data.get("action") != "rate":
        return
    value = ev.data.get("value")
    track = player.now_playing() or director.track
    if track is None or value not in (LIKE, BAN):
        return
    new = ratings.rate(track.path, track.title, value)
    print(f"★ {RATING_WORD[new]}: {track.title}")
    if new == BAN:
        director.skip()


def _library(args) -> Library:
    lib = Library.build(args.music, args.cache, progress=lambda m: print(m, file=sys.stderr))
    links = os.path.join(args.cache, "spotify_map.json")
    if os.path.exists(links):
        with open(links, encoding="utf-8") as f:
            m = json.load(f)
        for t in lib.tracks:
            t.spotify_uri = m.get(t.path)
    if not lib.tracks:
        sys.exit(f"No playable audio found in {args.music}")
    return lib


def cmd_analyze(args):
    lib = _library(args)
    print(lib.summary())


def cmd_watch(args):
    tail = FileTail(args.events)
    print(f"watching {args.events}")
    while True:
        for ev in tail.poll():
            print(ev.kind, ev.data)
        time.sleep(0.1)


def cmd_run(args):
    from .app import run
    run(args)


def cmd_ratings(args):
    ratings = _ratings(args)
    if args.clear:
        needle = args.clear.lower()
        hits = [(p, v) for p, v in ratings.items() if needle in v["title"].lower()]
        for p, v in hits:
            ratings.set(p, v["title"], NEUTRAL)
            print(f"cleared ({RATING_WORD[v['rating']]}): {v['title']}")
        if not hits:
            print(f"no rated song matches '{args.clear}'")
        return
    items = ratings.items()
    if not items:
        print("no ratings yet")
    for _, v in items:
        print(f"{'👍' if v['rating'] == LIKE else '👎'}  {v['title']}")


def cmd_replay(args):
    """Run a recorded bridge log through the director on a simulated clock (no audio)."""
    from .players import NullPlayer
    lib = _library(args)
    prof = Profile.load(args.profile, getattr(args, "direction", "conductor"))
    state = GameState(prof)
    from .ratings import Ratings
    director = Director(lib, ratings=Ratings(os.path.join(args.cache, "ratings.json")))
    for k, v in prof.tuning.items():
        setattr(director.tune, k, v)
    clock = [0.0]
    player = NullPlayer(clock=lambda: clock[0])
    last_intent = None

    def step(until: float):
        while clock[0] < until:
            cue = director.tick(state.intent(clock[0]), clock[0], player.position())
            if cue:
                print(f"{clock[0]:8.1f}s", end="")
                player.play(cue)
            clock[0] += 0.5

    for ev in replay(args.log):
        t = ev.data.get("t")
        if t is not None and t > clock[0]:
            step(t)
        state.apply(ev, clock[0])
        intent = state.intent(clock[0])
        if intent != last_intent and intent is not None and not args.quiet:
            print(f"{clock[0]:8.1f}s ◆ {intent.describe()}")
            last_intent = intent
    step(clock[0] + 30)


def _norm(s: str) -> str:
    s = re.sub(r"\(.*?\)|\[.*?\]|feat\..*", "", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def cmd_spotify_link(args):
    """Match each local file to a Spotify track URI by searching 'artist - title' from the filename."""
    import spotipy
    from spotipy.oauth2 import SpotifyPKCE
    sp = spotipy.Spotify(auth_manager=SpotifyPKCE(
        client_id=args.client_id or os.environ["SPOTIFY_CLIENT_ID"],
        redirect_uri="http://127.0.0.1:8765/callback",
        scope="user-read-playback-state", cache_path=os.path.join(args.cache, ".spotify_token")))
    from .library import scan
    out_path = os.path.join(args.cache, "spotify_map.json")
    mapping = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for path in scan(args.music):
        if path in mapping:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        res = sp.search(q=name.replace(" - ", " "), type="track", limit=5)["tracks"]["items"]
        want = _norm(name)
        best = next((r for r in res if _norm(r["name"]) in want), res[0] if res else None)
        if best:
            mapping[os.path.abspath(path)] = best["uri"]
            print(f"✓ {name}  →  {best['artists'][0]['name']} - {best['name']}")
        else:
            print(f"✗ {name}")
    os.makedirs(args.cache, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=1)
    print(f"saved {out_path} — edit it to fix any wrong matches")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="conductor")
    p.add_argument("--cache", default=DEFAULT_CACHE)
    sub = p.add_subparsers(dest="cmd", required=True)

    def music(sp):
        sp.add_argument("--music", required=True, help="folder of your audio files")

    a = sub.add_parser("analyze"); music(a); a.set_defaults(fn=cmd_analyze)
    w = sub.add_parser("watch"); w.add_argument("--events", default=DEFAULT_EVENT_PATH); w.set_defaults(fn=cmd_watch)
    r = sub.add_parser("run", help="play along with the game (settings from config.json)")
    r.add_argument("--music", help="folder of your audio files (default: config.json)")
    r.add_argument("--profile")
    r.add_argument("--events", default=DEFAULT_EVENT_PATH)
    r.add_argument("--player", choices=["local", "spotify", "none"])
    r.add_argument("--volume", type=float)
    r.add_argument("--client-id")
    r.add_argument("--background", action="store_true",
                   help="started by the game mod: log to cache/conductor.log, exit when the game closes")
    r.set_defaults(fn=cmd_run)
    rp = sub.add_parser("replay"); music(rp)
    rp.add_argument("--profile", default=DEFAULT_PROFILE); rp.add_argument("log")
    rp.add_argument("--quiet", action="store_true", help="only print music changes")
    rp.add_argument("--direction", default="conductor", choices=["conductor", "native"])
    rp.set_defaults(fn=cmd_replay)
    rt = sub.add_parser("ratings", help="list ratings, or --clear <title text> to reset songs to neutral")
    rt.add_argument("--clear", metavar="TITLE"); rt.set_defaults(fn=cmd_ratings)
    ig = sub.add_parser("ingest", help="download (optional) and fully analyse a playlist folder")
    ig.add_argument("--music", required=True, help="playlist folder (created if missing)")
    ig.add_argument("--spotify", help="Spotify playlist URL to download into the folder with spotDL")
    ig.add_argument("--name", help="playlist name shown in game (default: folder name)")
    ig.set_defaults(fn=lambda a: __import__("conductor.ingest", fromlist=["ingest"]).ingest(
        a.music, spotify=a.spotify, name=a.name))
    sa = sub.add_parser("standalone", help="no game: a web page with Calm/Tension/Fight/Horde/Boss + Pause buttons")
    sa.add_argument("--playlist"); sa.add_argument("--port", type=int, default=8770); sa.add_argument("--lan", action="store_true")
    sa.set_defaults(fn=lambda a: __import__("conductor.standalone", fromlist=["main"]).main(a.playlist, a.port, a.lan))
    sl = sub.add_parser("spotify-link"); music(sl); sl.add_argument("--client-id"); sl.set_defaults(fn=cmd_spotify_link)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
