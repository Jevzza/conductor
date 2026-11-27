# Building the drop-in mod

`packaging/build_dist.py` turns this repo into the folder players drop into `<Darktide>\mods\`, and optionally zips it.

```
python packaging/build_dist.py          # -> build/conductor
python packaging/build_dist.py pack     # -> build/conductor.zip as well
```

## One-time: the runtime

The build doesn't download anything. Put these in `build/conductor/runtime/` first:

| Path | What |
|---|---|
| `runtime/python/` | The **Windows embeddable Python 3.13** (python.org, "embeddable package, 64-bit"), with pip installed. Edit its `python313._pth` to read `python313.zip`, `.`, `Lib\site-packages`, `..\..\app`, `import site`. |
| `runtime/bin/` | `ffmpeg.exe` and `ffprobe.exe`, with the rubberband filter. Any recent full build works. |
| `runtime/wheels/` | Wheels pip can't fetch for Python 3.13 on Windows: **madmom** (build it from its git repo) and **beat-this** (`pip wheel git+https://github.com/CPJKU/beat_this`). |

The build copies `packaging/runtime/requirements.lock` next to them. On the player's first start, `conductor.boot` pip-installs from that lock file into `runtime/python` (about 3.5 GB, mostly CUDA PyTorch), so the zip itself stays small.

## Trained models

If `cache/` holds trained models, the build ships them as the starting point in `app/seed`. Those models are `taste_model.pkl`, `join_model.pkl`, `live/policy.json` and `live/model_styles.json`, which the trainer and `python -m conductor.ab nightly` produce. Without them the app still works, just untrained. Release zips include the maintainer's trained models.

## Linux

`build_linux()` writes `linux/launch.sh` and `runtime/requirements-linux.lock`. On Linux, `launch.sh` sets up its own Python with uv on first launch and installs PyTorch for the graphics card it finds. See `packaging/README.txt`.
