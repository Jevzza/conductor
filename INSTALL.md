# Conductor: install and use

Conductor plays your own music in Warhammer 40,000: Darktide and steers it with the game. It stays calm while you explore and builds into fights, hordes and bosses, joining songs the way a DJ would.

It comes in two forms, from the same folder:

- **In-game mod**: follows Darktide automatically.
- **Standalone**: no game needed. A web page with Calm / Tension / Fight / Horde / Boss buttons, for tabletop sessions (D&D and the like).

---

## What you need

- Windows 10 or 11, 64-bit.
- An NVIDIA GPU (any recent GeForce) to get new songs ready in reasonable time. Playing songs that are already processed works on any PC.
- About 6 GB of free disk space, plus about 10 MB per song.
- An internet connection for the first start and for adding playlists.
- **In-game only:** the Darktide Mod Framework (DMF) and the Darktide Mod Loader, the same as any other Darktide mod.

---

## Install

1. Unzip `conductor.zip`. You get a folder called `conductor`.
2. Move that folder into `<your Darktide folder>\mods\`. You should end up with `...\Warhammer 40,000 DARKTIDE\mods\conductor\conductor.mod`.
3. **In-game only:** open `mods\mod_load_order.txt` and add the line `conductor` (anywhere after `dmf`).

That's the whole install. Everything Conductor uses (Python, models, ffmpeg, downloads, your settings) stays inside `mods\conductor`, and nothing is installed anywhere else.

### First start (one time)

The first time Conductor starts, in game or standalone, it downloads its Python packages. That's about 3.5 GB, mostly PyTorch for your graphics card, and can take 10 to 30 minutes depending on your connection.

- **In game:** a progress bar appears near the top of the screen in the hub or a mission. In the main menu, just give it a few minutes.
- **Standalone:** the black console window shows the progress. Leave it open.

After that, it starts in seconds.

---

## Adding music

Conductor plays from **playlists**. A playlist can come from any of three sources:

| Source | How |
|---|---|
| **Spotify** | Paste a playlist link. Conductor downloads the songs. |
| **YouTube Music / YouTube** | Paste a playlist link. Conductor downloads the songs. |
| **Your own files** | Make a new folder inside `mods\conductor\data\playlists\`, for example `data\playlists\My Battle Mix\`, and copy your songs into it (mp3, m4a, flac, ogg, opus or wav). |

Every new song gets **processed** once: Conductor finds its beats, sections and instruments. This is the slow part. Expect it to take a while for a big playlist, even on an NVIDIA GPU. A progress bar shows how far along it is, and Conductor switches to the playlist by itself when it's done. Adding the same link again later only fetches the new songs.

A folder of your own files shows up in the playlist list marked **"not processed - pick to process"**. Pick it and processing starts.

---

## Using it in game

Open **Options → Mods → Conductor**. The main settings:

- **Add playlist (Spotify link)** / **Add playlist (YouTube Music link)**: paste a link and press Enter.
- **Playlist**: which playlist to play. New playlists appear here after a game restart.
- **Next playlist** (key): switch playlists on the fly.
- **Replace game music**: silences Darktide's own score while Conductor plays. Your in-game Music volume slider then controls Conductor.
- **Conductor level**: make your music louder or quieter relative to the game.
- **Gameplay music direction**:
  - *Conductor* anticipates fights and changes songs as they escalate.
  - *Native game* follows Darktide's own music states exactly.
- **Thumbs up / Thumbs down current song** (keys):
  - Thumbs up plays a song more often.
  - Thumbs down skips the song and never plays it again.

The music app starts with the game and closes when the game closes.

---

## Using it standalone (tabletop / D&D)

1. Double-click **`Conductor Standalone.cmd`** in `mods\conductor\`. You don't need to start the game.
2. A page opens in your browser at **http://127.0.0.1:8770**.
3. No music yet? Paste a Spotify or YouTube Music playlist link into the **Add a playlist** box on the page and press **Add**. To use your own files instead, put them in a folder in `data\playlists\` (see above) and pick that folder in the dropdown. A progress bar shows the processing, and the playlist opens by itself when it's ready.
4. Press a state button when the scene calls for it:
   - **Calm**: exploring, talking, the tavern.
   - **Tension**: something's wrong, a build-up.
   - **Fight**: combat.
   - **Horde**: everything is on fire.
   - **Boss**: the big one.

**How the page behaves:**

- Music changes on the next musical beat or phrase, not mid-bar. The card under the buttons shows what's coming: from what to what, what kind of blend, and a countdown with a timeline of each layer (drums, bass, melody) fading in and out.
- **Pause** fades the music out, and **Play** fades it back in where it left off.
- **Next song** picks another song for the same state.
- Volume and the playlist dropdown are at the bottom.

**Control it from your phone or tablet:** start it with `--lan`. Either run `"Conductor Standalone.cmd" --lan` from a command prompt in that folder, or edit the last line of the .cmd file. Then open `http://<this PC's IP address>:8770` on the phone, on the same Wi-Fi. Windows may ask to allow it through the firewall.

Close the console window to stop.

You can use standalone and the game on the same install. They share playlists and processed songs.

---

## Updating

Replace the `mods\conductor` folder with the new one, but **keep `mods\conductor\data`**. That folder holds your playlists, processed songs, settings and learned preferences. Everything else can be overwritten.

## Uninstalling

Delete `mods\conductor` and remove `conductor` from `mod_load_order.txt`.

---

## Troubleshooting

| Problem | Look at |
|---|---|
| First start / app won't start | `mods\conductor\data\boot.log` |
| Music app errors | `mods\conductor\data\cache\conductor.log` |
| A playlist didn't download or process | `mods\conductor\data\cache\addlist.log` |

- **No music in game:** check that `conductor` is in `mod_load_order.txt`, that **Start Conductor with the game** is on, and that a playlist with processed songs is selected.
- **Spotify download fails for some songs:** spotDL finds each song on YouTube, and a few may not match. Those songs are skipped and the rest still work.
- **Processing is very slow:** without an NVIDIA GPU it runs on the CPU, which takes several minutes per song.
- **Standalone page won't open:** make sure the console window is still open and nothing else is using port 8770. To use another port, start it with `--port 8771`.

### Linux (Steam + Proton)

The in-game mod works on Linux. The music app runs natively with a Steam launch option. See `README.txt` in this folder for the steps. Standalone is Windows-only for now.
