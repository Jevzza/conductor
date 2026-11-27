# Conductor

**An adaptive soundtrack built from your own music.** Conductor plays your playlists in *Warhammer 40,000: Darktide* and steers them with the game. It stays calm while you explore and builds into fights, hordes and bosses, joining songs the way a DJ would: on the beat, layer by layer, so one song's drums carry the next song's melody in.

There's also a **standalone mode** that needs no game. You get a web page with Calm / Tension / Fight / Horde / Boss buttons, made for tabletop sessions.

- **Players:** grab the zip from [Releases](../../releases) and follow **[INSTALL.md](INSTALL.md)**. It's a drop-in mod folder with everything inside.
- **Developers:** read on.

---

## How it works

```
 Darktide ─▶ Lua mod (mod/) ─▶ events.jsonl ─▶ GameState + profile ─▶ Intent ─▶ State director ─▶ Player
            hooks the game's                    (scene, combat,        (calm / tension /   picks the song,     local engine,
            Wwise music states                   intensity, health)     fight / horde / boss) section and join   renders the blend
                                                                                                ▲
                                     your playlist ─▶ ingest: beats, sections, stems, hooks ─▶ Library
```

- **Ingest** (`conductor/ingest.py`) runs once per song. It analyses:
  - the beat and downbeat grid ([Beat This](https://github.com/CPJKU/beat_this));
  - song sections ([All-In-One](https://github.com/mir-aidj/all-in-one));
  - drums / bass / vocals / other stems ([Demucs](https://github.com/facebookresearch/demucs));
  - loudness and energy curves, loops, hits and "call-out" sounds.
- **The state director** (`conductor/states.py`) is a Darktide-style state machine. It decides *when* to change and *to what*: which song, and which section of it.
- **Routes and joins** (`conductor/routes.py`, `conductor/grid.py`) render the change:
  - cuts, crossfades and DJ-style blends;
  - multi-bar **layered blends** that hand drums, bass and melody across separately, beat-locked;
  - a vocal guard (`conductor/vocals.py`), so two singers never overlap and lyrics are never chopped mid-phrase.
- **Learning** comes from your ratings:
  - the A/B trainer page (`review/ab.html`) plays you four versions of a change and asks which works best;
  - `conductor/ab.py`, `critic.py`, `joinmodel.py` and `policy.py` train a taste model and an audio critic on those answers.

  The game bridge also sends in-game thumbs up / down and "transition good / bad".

Nothing in the core is Darktide-specific. A game plugs in with a bridge that writes JSON lines, plus a profile (`profiles/darktide.json`) that maps the game's states to musical intent.

### Bridge protocol (JSON lines, `%TEMP%\conductor_events.jsonl`)

| `k` | fields | meaning |
|---|---|---|
| `hello` | `game, protocol` | bridge (re)started |
| `state` | `group, value` | a music state group changed |
| `param` | `name, value` | a continuous music parameter (throttled) |
| `stinger` | `event` | one-shot music event |
| `user` | `action, value` | player input (ratings, playlist actions) |

---

## Running from source

Windows, Python 3.13 and an NVIDIA GPU are recommended for ingest.

```
python -m venv .venv-ml
.venv-ml\Scripts\pip install -r packaging/runtime/requirements.lock
```

madmom and beat-this have no Windows wheels for Python 3.13. Install them from their git repos.

```
python -m conductor ingest --music "<music folder>"    # analyse a folder (GPU passes run in .venv-ml)
python -m conductor run                                 # play along with the game (settings in config.json)
python -m conductor standalone [--lan]                  # the tabletop page on http://127.0.0.1:8770
python -m conductor watch                               # print live bridge events
python -m conductor ratings                             # list song ratings
```

Data lives next to the code when run from source: `config.json`, `cache/` and `review/`. All of these are git-ignored.

### Training

```
python -m conductor.review_server m0 "" 8768            # serves review/ ; the trainer is /ab.html
python -m conductor.ab quads 10 "<playlist name>"       # render 10 new four-way sets to rate
python -m conductor.ab nightly 30                       # learn from your answers, retrain, render the next sets
```

`config.json` can set `train_music` (the folder the trainer draws songs from) and `train_folders` (the folders the critic learns from). Both default to your playlists.

### Building the drop-in mod

See [packaging/README-build.md](packaging/README-build.md). Running `python packaging/build_dist.py pack` produces the folder and zip that players install.

---

## Layout

| Path | What |
|---|---|
| `conductor/` | the app: ingest, library, state director, routes and joins, players, trainer back end, standalone |
| `mod/` | the Darktide mod (Lua, for the Darktide Mod Framework) |
| `profiles/` | game profiles: game music states mapped to intent |
| `review/` | trainer and review pages (served by `conductor.review_server`) |
| `lab_site/` | Critic Lab: a training dashboard |
| `tools/` | research tools for Darktide's own Wwise score. They need your own copy of the game. |
| `packaging/` | drop-in build: boot/first-start installer, Linux launcher, pinned requirements |

---

## Credits

Conductor stands on these open-source projects:

| Project | Used for |
|---|---|
| [Beat This](https://github.com/CPJKU/beat_this) | beat and downbeat tracking |
| [All-In-One](https://github.com/mir-aidj/all-in-one) | song structure / sections |
| [Demucs](https://github.com/facebookresearch/demucs) | stem separation |
| [madmom](https://github.com/CPJKU/madmom) | audio feature extraction. Note that its bundled models are licensed CC BY-NC-SA, non-commercial only. |
| [librosa](https://librosa.org) | analysis |
| [PyTorch](https://pytorch.org) | model inference |
| [spotDL](https://github.com/spotDL/spotify-downloader) | Spotify playlist downloads |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | YouTube Music playlist downloads |
| [FFmpeg](https://ffmpeg.org) + [Rubber Band](https://breakfastquay.com/rubberband/) | decoding and time-stretching |
| [Darktide Mod Framework](https://github.com/Darktide-Mod-Framework/Darktide-Mod-Framework) | the in-game mod |

Conductor is a fan project, not affiliated with Fatshark or Games Workshop. It ships no game assets and no music. You bring your own.

## License

[MIT](LICENSE)
