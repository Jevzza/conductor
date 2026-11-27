"""Where Conductor keeps its data.

Run from source (a git clone): the repo folder.
Drop-in mod (mods/conductor/app/conductor): mods/conductor/data.
CONDUCTOR_HOME overrides both.
"""
import os

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # .../app (or the repo)
MOD = os.path.dirname(APP)
PORTABLE = os.path.exists(os.path.join(MOD, "conductor.mod")) and os.path.isdir(os.path.join(MOD, "runtime"))
HOME = os.environ.get("CONDUCTOR_HOME") or (os.path.join(MOD, "data") if PORTABLE else APP)
RUNTIME = os.path.join(MOD, "runtime") if PORTABLE else None
PLAYLISTS = os.path.join(HOME, "playlists")       # where "Add playlist" downloads go
if HOME != APP:
    os.makedirs(HOME, exist_ok=True)              # fresh install: data/ may not exist yet
