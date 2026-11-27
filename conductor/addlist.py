"""Add a Spotify playlist from inside the game (mod options -> "Add playlist" text field).

    python -m conductor.addlist <spotify playlist url>

spotDL downloads it into the mod's data folder, then the usual ingest gets every song ready.
Progress goes to %TEMP%/conductor_addlist.txt, one line the mod reads for its on-screen bar:

    <fraction 0-1>|<what it's doing>|<running|done|error>|<playlist name>

When it's done the mod switches to the new playlist (the app confirms it in chat).
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = __import__("conductor.playlists", fromlist=["DATA"]).DATA     # where playlists are downloaded to
PROGRESS = os.path.join(os.environ.get("TEMP", "/tmp"), "conductor_addlist.txt")
NOWIN = 0x08000000 if os.name == "nt" else 0
AUDIO = (".mp3", ".m4a", ".flac", ".ogg", ".opus", ".wav")
# share of the bar each stage gets
STAGES = {"list": (0.0, 0.03), "download": (0.03, 0.35), "analysis": (0.35, 0.55), "gpu": (0.55, 0.85),
          "merge": (0.85, 1.0)}


def report(frac: float, text: str, state: str = "running", name: str = "") -> None:
    try:
        tmp = PROGRESS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(f"{max(0.0, min(1.0, frac)):.3f}|{text.replace('|', '/')}|{state}|{name}\n")
        os.replace(tmp, PROGRESS)
    except OSError:
        pass
    print(time.strftime("%H:%M:%S"), f"{frac:5.1%}", text, flush=True)


def stage(key: str, k: float, n: float, text: str, name: str = "") -> None:
    a, b = STAGES[key]
    report(a + (b - a) * (k / n if n else 0.0), text, name=name)


def _safe(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name).strip(" .") or time.strftime("Playlist %Y-%m-%d %H%M")


def _songs(folder: str) -> int:
    return sum(1 for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(AUDIO))


def is_youtube(url: str) -> bool:
    return bool(re.search(r"(^|//|\.)(music\.)?youtube\.com/|youtu\.be/", url))


def _ytm_download(url: str):
    """YouTube Music (or YouTube) playlist -> DATA/<playlist title>/<artist> - <title>.mp3 via yt-dlp.
    Returns the folder, or None."""
    import yt_dlp
    report(0.0, "Reading the YouTube Music playlist")
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": "in_playlist", "skip_download": True}) as y:
            info = y.extract_info(url, download=False)
    except Exception as e:
        report(0.0, f"Couldn't read that YouTube link ({str(e)[:80]})", "error")
        return None
    title = _safe((info.get("title") or "").replace("Album - ", "") or "YouTube playlist")
    total = len(info.get("entries") or []) or 1
    folder = os.path.join(DATA, title)
    os.makedirs(folder, exist_ok=True)
    st = {"done": 0}

    def hook(d):
        if d.get("status") == "finished":
            st["done"] += 1
            stage("download", st["done"], total, f"{title}: downloading {st['done']}/{total}", title)

    opts = {
        "format": "bestaudio/best", "quiet": True, "no_warnings": True, "ignoreerrors": True,
        # artist from YouTube Music's metadata; else the channel ("Artist - Topic" cleaned below)
        "outtmpl": os.path.join(folder, "%(artist,creator,uploader)s - %(track,title)s.%(ext)s"),
        "download_archive": os.path.join(folder, ".downloaded.txt"),   # re-adding only fetches what's new
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "0"},
                           {"key": "FFmpegMetadata"}],
        "progress_hooks": [hook], "noprogress": True,
    }
    stage("download", 0, total, f"{title}: downloading 0/{total}", title)
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([url])
    except Exception as e:
        print("yt-dlp:", e)
    for p in glob.glob(os.path.join(folder, "*")):       # "Artist - Topic - Song" -> "Artist - Song"
        b = os.path.basename(p)
        nb = re.sub(r" - Topic(?= - )", "", b)
        if nb != b and not os.path.exists(os.path.join(folder, nb)):
            os.replace(p, os.path.join(folder, nb))
    return folder if _songs(folder) else None


def process_folder(f: str) -> int:
    """Get every song in an existing folder ready (local files pasted in, or a finished download)."""
    name = os.path.basename(os.path.normpath(f))
    from . import ingest
    from .playlists import all_playlists
    for n, d in all_playlists().items():              # keep the dropdown's name for this folder
        if os.path.normcase(os.path.normpath(d)) == os.path.normcase(os.path.normpath(f)):
            name = n

    def prog(key, k, n, text):
        stage(key, k, n, f"{name}: {text}", name)

    ingest.ingest(f, name=name, progress=prog)
    from .playlists import counts, write_index
    ready, tot = counts(f)
    try:
        write_index()
    except Exception:
        pass
    if ready == 0:
        report(1.0, f"{name}: processing failed - see the app log", "error", name)
        return 1
    report(1.0, f"{name}: ready - {ready} songs" + (f" ({tot - ready} couldn't be processed)" if tot > ready else ""),
           "done", name)
    return 0


def main(url: str) -> int:
    url = url.strip()
    if url.startswith("local:"):
        return process_folder(url[6:])
    if is_youtube(url):
        f = _ytm_download(url)
        if f is None:
            report(0.0, "Couldn't download that playlist - is the YouTube Music link right (and public)?", "error")
            return 1
        return process_folder(f)
    report(0.0, "Warming up spotDL")
    os.makedirs(DATA, exist_ok=True)
    before = {d: os.path.getmtime(os.path.join(DATA, d)) for d in os.listdir(DATA)}
    # one pass: spotDL names the folder after the playlist ({list-name}) and skips songs already there,
    # so re-adding a playlist only fetches what's new. ("spotdl save" first took 3.5 min for 50 songs.)
    proc = subprocess.Popen([sys.executable, "-m", "spotdl", "download", url, "--output",
                             os.path.join(DATA, "{list-name}", "{artists} - {title}.{output-ext}")],
                            cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", creationflags=NOWIN,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    info = {"name": "", "total": 0, "phase": "Warming up spotDL", "last": "", "failed": 0}

    def read():
        for line in proc.stdout:
            line = line.strip()
            m = re.search(r"Found (\d+) songs in (.+?) \((?:Playlist|Album)\)", line)
            if m:
                info["total"], info["name"] = int(m.group(1)), m.group(2).strip()
                info["phase"] = f"matching {info['total']} songs on YouTube"
            elif line.startswith("Processing query"):
                info["phase"] = "finding the songs on Spotify"
            elif line.startswith("Downloaded"):
                m = re.match(r'Downloaded "(.+?)"', line)
                info["last"] = m.group(1) if m else ""
            elif line.startswith("Skipping"):
                info["last"] = "already have " + line[9:].split(" (")[0]
            elif "No results found" in line or "LookupError" in line or "AudioProviderError" in line:
                info["failed"] += 1
    threading.Thread(target=read, daemon=True).start()

    def folder():
        if info["name"] and os.path.isdir(os.path.join(DATA, _safe(info["name"]))):
            return os.path.join(DATA, _safe(info["name"]))      # re-added playlist: its folder already exists
        dirs = [d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d))]
        new = [d for d in dirs if d not in before or os.path.getmtime(os.path.join(DATA, d)) > before[d]]
        return os.path.join(DATA, max(new, key=lambda d: os.path.getmtime(os.path.join(DATA, d)))) if new else None

    while proc.poll() is None:
        f = folder()
        n = _songs(f) if f else 0
        name, total = info["name"] or "playlist", info["total"]
        if n == 0 and not info["last"]:
            text = f"{name}: {info['phase']}" if info["name"] else info["phase"]
        else:
            text = f"{name}: downloading {n}/{total or '?'}" + (f" - {info['last'][:60]}" if info["last"] else "")
            if info["failed"]:
                text += f" ({info['failed']} not found)"
        stage("download", n, total or 1, text, name)
        time.sleep(2)
    f = folder()
    if f is None and info["name"]:                       # nothing new: everything was already downloaded
        f = os.path.join(DATA, _safe(info["name"]))
    if not f or not os.path.isdir(f) or _songs(f) == 0:
        report(0.0, "Couldn't download that playlist - is the Spotify link right (and public)?", "error")
        return 1
    # 2. the usual ingest, with progress
    return process_folder(f)


def start(url: str) -> bool:
    """From the app: run it in its own process (heavy; keeps the music smooth)."""
    lock = os.path.join(HERE, "cache", "ingest.lock")
    if os.path.exists(lock):
        report(0.0, "Already adding a playlist - wait for it to finish", "error")
        return False
    report(0.0, "Starting")
    log = open(os.path.join(HERE, "cache", "addlist.log"), "a", encoding="utf-8")
    # python.exe (not the app's pythonw): a console process with CREATE_NO_WINDOW gets a hidden console that
    # spotDL's ffmpeg children share; under pythonw every ffmpeg opened its own window and tabbed you out
    exe = sys.executable.replace("pythonw.exe", "python.exe").replace("pythonw3", "python3")
    subprocess.Popen([exe, "-m", "conductor.addlist", url], cwd=HERE, stdout=log, stderr=log,
                     creationflags=NOWIN, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return True


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
