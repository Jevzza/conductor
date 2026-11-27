CONDUCTOR - adaptive soundtrack for Warhammer 40,000: Darktide
================================================================

Plays your own music in Darktide and steers it with the game: calm while you explore, building
into fights, hordes and bosses, joined like a DJ would.

Full instructions (install, adding music, in-game use, standalone / tabletop use): INSTALL.md

STANDALONE (no game - tabletop / D&D)
  Double-click "Conductor Standalone.cmd" in mods\conductor. A page opens at http://127.0.0.1:8770
  with Calm / Tension / Fight / Horde / Boss buttons, Pause (fades out) and an Add playlist box.

INSTALL
  1. Needs the Darktide Mod Framework (DMF) and the mod loader, like any other Darktide mod.
  2. Drop the "conductor" folder into  <Darktide>\mods\
  3. Add "conductor" to mods\mod_load_order.txt (after dmf).
  4. Start the game.

FIRST START
  Conductor installs its own Python packages the first time (one-time download, about 3.5 GB,
  mostly the CUDA build of PyTorch). A progress bar near the top of the screen shows it
  (in the hub / a mission; in the main menu just wait a few minutes). Nothing is installed
  outside the mods\conductor folder.

ADD MUSIC
  Options -> Mods -> Conductor -> "Add playlist (Spotify link)" or "(YouTube Music link)".
  Or put your own songs in a new folder in mods\conductor\data\playlists\ and pick it in the
  Playlist list (marked not processed until you do). Paste a Spotify playlist link
  and press Enter. Conductor downloads it (spotDL, into mods\conductor\data\playlists\), gets
  every song ready (beats, sections, stems - this is the slow part) and switches to it when
  done. Re-adding the same link later only fetches new songs.

REQUIREMENTS
  - Windows 10/11, 64-bit
  - NVIDIA GPU (any recent GeForce) to process new songs in reasonable time. Playing songs
    that are already processed works on any PC.
  - ~6 GB free disk for the runtime, plus your music (~10 MB per song with its analysis)
  - Internet for the first start and for adding playlists

LINUX (Steam + Proton)
  Same install (steps 1-3). Then, because the music app runs natively on Linux, not inside Proton:
  4. Darktide > Properties > General > Launch Options:
       "<your Darktide folder>/mods/conductor/linux/launch.sh" %command%
     e.g. "/home/you/.local/share/Steam/steamapps/common/Warhammer 40,000 DARKTIDE/mods/conductor/linux/launch.sh" %command%
     (after the first launch, mods/conductor/linux/steam_launch_option.txt holds the exact line)
  5. Start the game. The first start downloads Python + PyTorch for your graphics card:
     NVIDIA (CUDA) ~3.5 GB, AMD Radeon RX 7000/9000 (ROCm) ~4 GB, otherwise CPU-only ~1 GB.
  The launch script starts Darktide directly (skipping Fatshark's launcher): with mods on, the
  launcher's empty window otherwise covers the game in Steam Big Picture / gamescope (black screen
  while the game runs). Set CONDUCTOR_KEEP_LAUNCHER=1 in the launch options to keep the launcher.
  Tested on Bazzite (gamescope / Big Picture). Working launch options there:
       LD_PRELOAD="" PROTON_NO_ESYNC=1 STEAM_SKIP_SHADER_PROCESSING=1 "<path>/launch.sh" %command%
  Mod loader on Linux: besides mods/base, it needs binaries/mod_loader and
  bundle/9ba626afa44a3aa3.patch_999 from the Darktide Mod Loader download, then the bundle patch:
  run tools/dtkit-patch.exe --patch ./bundle through Proton (or toggle_darktide_mods.bat via Wine).
  Needs: curl, ffmpeg (with ffprobe), PortAudio (libportaudio.so.2), git + gcc only if madmom
  has to be built. Logs: mods/conductor/data/launch.log and boot.log.

UNINSTALL
  Delete mods\conductor. That's everything (logs and settings live in mods\conductor\data).

TROUBLESHOOTING
  mods\conductor\data\boot.log            first-start install / app start-up
  mods\conductor\data\cache\conductor.log the music app
  mods\conductor\data\cache\addlist.log   playlist downloads + processing
