"""The long-running Conductor app: started by the game mod (or by hand), plays until the game exits.

Lifecycle
  * single instance (a localhost socket acts as the lock): a second launch exits immediately
  * settings come from config.json next to the package, so the mod needs no arguments
  * writes %TEMP%\\conductor_status.txt every couple of seconds:
        line 1 now-playing title, line 2 its rating, line 3 unix time (heartbeat)
    the bridge uses the heartbeat to know the app is alive (and only then mutes game music)
  * exits when the game process is gone, so nothing is left running
  * bridge `volume` events set the playback level: master × music × level, all 0..100
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time

from .director import Director
from .events import DEFAULT_EVENT_PATH, FileTail
from .library import Library
from .profile import GameState, Profile
from .ratings import BAN, LIKE, NEUTRAL, Ratings

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(HERE, "config.json")
STATUS_PATH = os.path.join(os.environ.get("TEMP", "/tmp"), "conductor_status.txt")
LOCK_PORT = 47613

# Vanilla-style mixing. dB below normal while something important is being said.
DUCK_DIALOGUE_DB = -4.0
DUCK_VOICE_DB = -8.0
DUCK_ATTACK_S = 0.12
DUCK_RELEASE_S = 0.6
FOCUS_FADE_S = 0.25
DUCK_HOLD_S = 1.5        # the mission-giver flag drops out between words; stay ducked through gaps
RATING_WORD = {LIKE: "liked", NEUTRAL: "neutral", BAN: "banned"}

DEFAULT_CONFIG = {
    "music": "",              # none until a playlist is added
    "profile": os.path.join(HERE, "profiles", "darktide.json"),
    "game_process": "Darktide.exe",
    "player": "local",
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    else:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    return cfg


class _Log:
    """pythonw has no console: mirror output to cache/conductor.log."""

    def __init__(self, path: str):
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.f.write(f"\n===== Conductor started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    def write(self, s):
        self.f.write(s)

    def flush(self):
        self.f.flush()


RECORDINGS = os.path.join(HERE, "recordings")
KEEP_RECORDINGS = 30


def archive_sessions(game: str = "darktide") -> None:
    """Move bridge session archives (%TEMP%/conductor_events_<unix>.jsonl) into recordings/,
    named by date, and keep only the newest KEEP_RECORDINGS."""
    import glob
    import shutil
    os.makedirs(RECORDINGS, exist_ok=True)
    tmp = os.environ.get("TEMP", "/tmp")
    for src in glob.glob(os.path.join(tmp, "conductor_events_*.jsonl")):
        try:
            stamp = int(os.path.basename(src)[len("conductor_events_"):-len(".jsonl")])
        except ValueError:
            stamp = int(os.path.getmtime(src))
        dst = os.path.join(RECORDINGS, f"{game}_{time.strftime('%Y-%m-%d_%H%M%S', time.localtime(stamp))}.jsonl")
        try:
            shutil.move(src, dst)
        except OSError:
            pass
    auto = sorted(glob.glob(os.path.join(RECORDINGS, f"{game}_20*_*.jsonl")))
    for old in auto[:-KEEP_RECORDINGS]:
        try:
            os.remove(old)
        except OSError:
            pass


def _single_instance():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", LOCK_PORT))
        s.listen(1)
    except OSError:
        return None
    return s     # keep the socket alive for the process lifetime


def _game_running(image: str) -> bool:
    if not image:
        return True
    if os.name != "nt":            # Linux (Proton): the game is a Wine process whose command line names the exe
        try:
            # only the Wine process itself: its command line STARTS with a Windows path (S:\...\Darktide.exe).
            # Steam's wrappers also mention .../binaries/Darktide.exe once launch.sh skips the launcher; matching
            # them kept the app alive after the game closed, so Steam kept "running" a black screen (2026-11-24)
            # (procps pgrep: a literal backslash only matches as the bracket class [\], not \\)
            pat = r"^[A-Za-z]:[\].*[\]" + re.sub(r"([.+()$^|?*{}])", r"[\1]", image)
            out = subprocess.run(["pgrep", "-f", "-i", pat], capture_output=True, text=True, timeout=10)
            return out.returncode == 0
        except Exception:
            return True
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/NH"], capture_output=True,
                             text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=10)
        return image.lower() in out.stdout.lower()
    except Exception:
        return True   # if we can't tell, keep running


def _taper(slider: float) -> float:
    """Game volume sliders are perceptual, not linear. A square-law taper maps 50 % to about
    -12 dB (a cubic one, tried first, made 50 % far too quiet: -18 dB)."""
    v = max(0.0, min(1.0, slider / 100))
    return v ** 2


_feedback = {"id": 0, "text": ""}     # last confirmed user action, shown in game chat by the bridge


def _write_status(track, ratings: Ratings) -> None:
    """line 1 title, 2 rating, 3 heartbeat (unix), 4 '<id>|<feedback text>' for the game chat."""
    try:
        tmp = STATUS_PATH + ".tmp"
        title = track.title if track else ""
        rating = ratings.get(track.path) if track else 0
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(f"{title}\n{rating}\n{int(time.time())}\n{_feedback['id']}|{_feedback['text']}\n")
        os.replace(tmp, STATUS_PATH)
    except OSError:
        pass


def _handle_user(ev, player, director, ratings: Ratings) -> None:
    if ev.data.get("action") in ("rate_change", "rate_timing"):   # 👍/👎 the last change's join / its timing
        from . import changelog
        v = 1 if (ev.data.get("value") or 0) > 0 else -1
        msg = (changelog.rate if ev.data["action"] == "rate_change" else changelog.rate_timing)(v, ev.data.get("t"))
        print("★ " + msg, flush=True)
        _feedback["id"] += 1
        _feedback["text"] = msg
        return
    if ev.data.get("action") == "add_playlist":                  # mod options text field (2026-11-22)
        from . import addlist
        url = str(ev.data.get("url") or "").strip()
        if "spotify.com" not in url and not url.startswith("spotify:") and not addlist.is_youtube(url):
            msg = "That isn't a Spotify or YouTube Music playlist link"
        else:
            msg = "Adding the playlist - progress shows on screen" if addlist.start(url) else "Already adding a playlist"
        print("★ " + msg, url, flush=True)
        _feedback["id"] += 1
        _feedback["text"] = msg
        return
    if ev.data.get("action") != "rate":
        return
    value = ev.data.get("value")
    track = player.now_playing() or director.track
    if track is None or value not in (LIKE, BAN):
        return
    new = ratings.rate(track.path, track.title, value)
    print(f"★ {RATING_WORD[new]}: {track.title}")
    word = {LIKE: "Liked", NEUTRAL: "Back to neutral", BAN: "Banned, skipping"}[new]
    _feedback["id"] += 1
    _feedback["text"] = f"{word}: {track.title}"
    if new == BAN:
        director.skip()


STALE_S = 20.0         # seconds without any event from the game mod = the game is gone (or hung): fade out
STINGER_EVENTS = {"knocked_down": ("downed", 0.92), "dead": ("died", 0.88)}   # event, music level under it (≈ no duck)
_last_player_state = {"v": None}


def _player_event(value, player, director, feed, cfg) -> None:
    """You got downed / died: play the current song's own dramatic moment over the music."""
    prev, _last_player_state["v"] = _last_player_state["v"], value
    if value == prev or value not in STINGER_EVENTS or not hasattr(player, "stinger"):
        return
    event, duck = STINGER_EVENTS[value]
    track = player.now_playing() or director.track
    if track is None:
        return
    from .stingers import load as load_stingers
    bank = load_stingers(track.path)
    st = bank.get(event) or bank.get("died" if event == "downed" else "downed")   # a song's other stinger stands in
    try:                                    # your own pick (e.g. Miasma ~97 s) wins for every song
        user = json.load(open(os.path.join(HERE, "cache", "stingers", "user.json"), encoding="utf-8")).get("global", {})
        if user.get(event) and os.path.exists(user[event]):
            st = {"file": user[event]}
    except (OSError, ValueError):
        pass
    if st and st.get("file") and not cfg.get("rejected", {}).get(f"{track.path}|{event}")             and player.stinger(st["file"], duck=duck):
        feed.say(f"{'Downed' if event == 'downed' else 'Died'}: stinger from {track.title}")
    else:
        player.stinger_duck_only = True
        feed.say(f"{'Downed' if event == 'downed' else 'Died'} (this song has no stinger)")


def _no_console_windows() -> None:
    """Any child process (ffmpeg for tempo matching, decoders...) must never open a console window:
    from the windowless background app each one flashed a CMD window and tabbed you out of the game."""
    if os.name != "nt" or getattr(subprocess.Popen, "_no_window", False):
        return
    base = subprocess.Popen

    class Popen(base):
        _no_window = True

        def __init__(self, *a, **k):
            k["creationflags"] = k.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
            super().__init__(*a, **k)
    subprocess.Popen = Popen


def run(args) -> None:
    _no_console_windows()
    cfg = load_config()
    from . import playlists as pls
    lists = pls.all_playlists(cfg)
    current = cfg.get("playlist") if cfg.get("playlist") in lists else next(iter(lists), None)
    music = args.music or (lists.get(current) if current else cfg["music"])
    profile_path = args.profile or cfg["profile"]
    player_kind = args.player or cfg["player"]
    if args.background:
        os.makedirs(args.cache, exist_ok=True)
        sys.stdout = sys.stderr = _Log(os.path.join(args.cache, "conductor.log"))
    lock = _single_instance()
    if lock is None:
        print("Conductor is already running; exiting.")
        return
    archive_sessions()
    try:
        pls.write_index()           # the mod's playlist dropdown reads this
    except Exception as e:
        print("playlists.txt:", e)

    # Fast start: only songs already processed by `conductor ingest`; never analyse while playing.
    lib = Library.build(music, args.cache, progress=lambda m: None, backfill=False, cached_only=True)
    if not lib.tracks:
        print(f"No playable audio found in {music}")
        return
    links = os.path.join(args.cache, "spotify_map.json")
    if os.path.exists(links):
        m = json.load(open(links, encoding="utf-8"))
        for t in lib.tracks:
            t.spotify_uri = m.get(t.path)
    profile = Profile.load(profile_path)
    state = GameState(profile)
    ratings = Ratings(os.path.join(args.cache, "ratings.json"))
    from .states import StateDirector
    director = (Director(lib, ratings=ratings) if os.environ.get("CONDUCTOR_LEGACY")
                else StateDirector(lib, ratings=ratings))
    livecritic = None
    if (cfg.get("live_critic") or os.environ.get("CONDUCTOR_LIVE_CRITIC") == "1") and isinstance(director, StateDirector):
        try:                                    # 2026-11-23: listens to likely changes ahead of time (GPU, own process)
            from .livecritic import LiveCritic
            livecritic = LiveCritic()
            director.livecritic, director.music = livecritic, music
            print("live critic: starting", flush=True)
        except Exception as e:
            print("live critic unavailable:", e, flush=True)

    if player_kind == "local":
        from .players import LocalPlayer
        player = LocalPlayer(volume=args.volume if args.volume is not None else 0.8)
    elif player_kind == "spotify":
        from .players import SpotifyPlayer
        player = SpotifyPlayer(client_id=args.client_id or os.environ["SPOTIFY_CLIENT_ID"],
                               cache_path=os.path.join(args.cache, ".spotify_token"))
    else:
        from .players import NullPlayer
        player = NullPlayer()

    tail = FileTail(args.events)
    banned = sum(1 for _, v in ratings.items() if v["rating"] == BAN)
    print(f"Conductor: {len(lib.tracks)} tracks from {music}, profile '{profile.game}'")
    print(f"  ratings: {len(ratings.items()) - banned} liked, {banned} banned; listening on {args.events}")

    mix = {"focus_muted": False, "dialogue": False, "voice": False}
    playing_list = {"name": current}
    base_tune = dict(director.tune.__dict__)
    direction = {"name": "conductor"}

    def set_direction(name: str) -> None:
        nonlocal state, profile
        if name == direction["name"]:
            return
        new = Profile.load(profile_path, name)
        ns = GameState(new)
        ns.states, ns.params = state.states, state.params      # keep what the game already told us
        state, profile = ns, new
        for k, v in base_tune.items():
            setattr(director.tune, k, v)
        for k, v in new.tuning.items():
            setattr(director.tune, k, v)
        direction["name"] = new.direction
        _feedback["id"] += 1
        _feedback["text"] = ("Gameplay music: native game direction" if new.direction == "native"
                             else "Gameplay music: Conductor direction")
        print("direction →", new.direction)

    loading = {"name": None, "done": []}

    def switch_playlist(name: str) -> None:
        """Build the new library off the main loop (a first load can take minutes: tempo maps),
        so the music and the mod's heartbeat keep going; finish_switch swaps it in when ready."""
        lists_now = pls.all_playlists()
        if name in lists_now:
            _process_if_needed(name, lists_now[name])
        if name == playing_list["name"]:
            loading["name"] = None             # picked the current one again: cancel a pending switch
            return
        if name not in lists_now or name == loading["name"]:
            return
        loading["name"] = name
        import threading

        def work(path=lists_now[name]):
            try:
                lib_new = Library.build(path, args.cache, progress=lambda m: None, backfill=False,
                                        cached_only=True)
            except Exception as e:
                print("playlist load failed:", name, e)
                lib_new = Library([])
            loading["done"].append((name, lib_new))
        threading.Thread(target=work, daemon=True).start()

    processed_try = {}

    def _process_if_needed(name: str, folder: str) -> None:
        """user 2026-11-26: picking a folder that isn't processed yet processes it (progress bar on screen;
        the mod switches to it when it's done). Once per folder per session (unless songs were added)."""
        try:
            ready, total = pls.counts(folder)
        except Exception:
            return
        if total == 0 or ready >= total or processed_try.get(folder) == total:
            return
        processed_try[folder] = total
        from . import addlist
        ok = addlist.start("local:" + folder)
        _feedback["id"] += 1
        _feedback["text"] = (f"Processing {name}: {total - ready} new songs - progress shows on screen" if ok
                             else "Already processing a playlist - pick it again when that's done")
        print("*", _feedback["text"], flush=True)

    def _safe_index() -> None:
        try:
            pls.write_index()
        except Exception as e:
            print("playlists.txt:", e)

    last_index = [time.monotonic()]

    def finish_switch() -> None:
        name, new = loading["done"].pop(0)
        if name != loading["name"]:            # another choice came in meanwhile; that one wins
            return
        loading["name"] = None
        _feedback["id"] += 1
        if not new.tracks:
            _feedback["text"] = f"Playlist {name} isn't ready yet - its songs still need processing."
            print("playlist not ready:", name)
            return
        playing_list["name"] = name
        director.lib = new
        if livecritic is not None:
            director.music = pls.all_playlists().get(name, getattr(director, "music", None))
        director.skip()
        _feedback["text"] = f"Playlist: {name} ({len(new.tracks)} songs)"
        print(f"playlist → {name} ({len(new.tracks)} songs)")
        try:
            pls.set_current(name)
        except OSError:
            pass

    def apply_mix():
        if not hasattr(player, "set_trim"):
            return
        if mix["focus_muted"] or mix.get("stale"):
            player.set_trim(0.0, FOCUS_FADE_S)
            return
        db = 0.0
        if mix["dialogue"]:
            db = min(db, DUCK_DIALOGUE_DB)
        if mix["voice"]:
            db = min(db, DUCK_VOICE_DB)
        target = 10 ** (db / 20)
        player.set_trim(target, DUCK_ATTACK_S if target < player.trim else DUCK_RELEASE_S)

    last_intent = None
    from .feed import Feed
    feed = Feed()
    stinger_cfg = {}
    try:
        stinger_cfg = json.load(open(os.path.join(args.cache, "stingers", "review.json"), encoding="utf-8"))
    except (OSError, ValueError):
        pass
    last_status = 0.0
    last_game_check = time.monotonic()
    last_game_t = None
    last_event = None      # monotonic time of the last bridge event (the mod sends one at least every 5 s)
    catching_up = True     # first read replays the log so far; old presses must not re-apply
    try:
        while True:
            now = time.monotonic()
            for ev in tail.poll():
                last_event = now
                if mix.get("stale"):
                    mix["stale"] = False
                    apply_mix()
                if ev.data.get("t") is not None:
                    last_game_t = ev.data.get("t")
                if ev.kind == "user":
                    if not catching_up:
                        _handle_user(ev, player, director, ratings)
                        last_status = 0.0
                elif ev.kind == "focus":
                    mix["focus_muted"] = bool(ev.data.get("muted"))
                    apply_mix()
                elif ev.kind == "duck":
                    for k in ("dialogue", "voice"):
                        if ev.data.get(k):
                            mix[k] = True
                            mix[k + "_off"] = None
                        elif mix[k]:
                            mix[k + "_off"] = now      # release after DUCK_HOLD_S of quiet
                    apply_mix()
                elif ev.kind == "volume":
                    d = ev.data
                    if d.get("direction"):
                        set_direction(d["direction"])
                        last_status = 0.0
                    if d.get("playlist"):           # the mod's choice (also its saved one at start-up)
                        switch_playlist(d["playlist"])
                        last_status = 0.0
                    level = (_taper(d.get("master", 100)) * _taper(d.get("music", 100))
                             * (d.get("level", 100) / 100))
                    if d.get("enabled") is False:
                        level = 0.0
                    if hasattr(player, "volume") and abs(getattr(player, "volume") - level) > 1e-3:
                        player.volume = level
                        print(f"volume → {level:.2f} (master {d.get('master')}, music {d.get('music')}, level {d.get('level')})")
                else:
                    if ev.kind == "state" and ev.data.get("group") == "player_state" and not catching_up:
                        _player_event(ev.data.get("value"), player, director, feed, stinger_cfg)
                    state.apply(ev, now)
            catching_up = False
            if now - last_index[0] > 30:          # folders pasted into data/playlists show up in the dropdown
                last_index[0] = now
                import threading
                threading.Thread(target=_safe_index, daemon=True).start()
            while loading["done"]:
                finish_switch()
                last_status = 0.0
            for k in ("dialogue", "voice"):
                off = mix.get(k + "_off")
                if mix[k] and off is not None and now - off >= DUCK_HOLD_S:
                    mix[k] = False
                    mix[k + "_off"] = None
                    apply_mix()

            if now - last_status >= 2.0:
                _write_status(player.now_playing(), ratings)
                last_status = now
            # 2026-11-21: music kept playing long after the game closed. The mod's events stop at once:
            # 20 s of silence -> fade the music out; the process check then exits within seconds.
            if last_event is not None and not mix.get("stale") and now - last_event > STALE_S:
                mix["stale"] = True
                apply_mix()
            if args.background and now - last_game_check >= 3.0:
                last_game_check = now
                archive_sessions()          # pick up archives from mod reloads mid-session
                if not _game_running(cfg.get("game_process", "")):
                    print("Game closed; exiting.")
                    break

            intent = state.intent(now)
            if intent != last_intent and intent is not None:
                last_intent = intent
            feed.intent(intent, now)
            if hasattr(player, "busy") and player.busy():
                time.sleep(0.05)
                continue                     # mid-switch: the position still belongs to the old song
            prev_track, prev_pos = player.now_playing() or director.track, player.position()
            state_before = getattr(director, "state", None)
            cue = director.tick(intent, now, prev_pos)
            if cue:
                feed.cue(cue)
                player.play(cue)
                try:
                    from . import changelog
                    changelog.record(cue, prev_track, prev_pos, last_game_t, getattr(director, "state", None),
                                     prev_state=state_before)
                except Exception as e:
                    print("change log failed:", e)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        player.close()
        if livecritic is not None:
            livecritic.close()
        try:
            os.remove(STATUS_PATH)   # tells the bridge the app isn't running → game music returns
        except OSError:
            pass
        lock.close()
