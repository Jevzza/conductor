#!/bin/bash
# Conductor on Linux (Steam + Proton).
# Darktide > Properties > General > Launch Options:
#     "<your Darktide folder>/mods/conductor/linux/launch.sh" %command%
# (mods/conductor/linux/steam_launch_option.txt has the exact line for this install, written on first run.)
#
# Starts the Conductor app natively beside the game. The mod (inside Proton) and the app talk through
# files in the game's Windows temp folder, so the app points TEMP at that folder in the Proton prefix.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD="$(dirname "$HERE")"
RT="$MOD/runtime/linux"
DATA="$MOD/data"
mkdir -p "$DATA" "$RT"
printf '"%s" %%command%%\n' "$HERE/launch.sh" > "$HERE/steam_launch_option.txt"

PFX="${STEAM_COMPAT_DATA_PATH:-$HOME/.local/share/Steam/steamapps/compatdata/1361210}/pfx"
export TEMP="$PFX/drive_c/users/steamuser/AppData/Local/Temp"
mkdir -p "$TEMP"
export CONDUCTOR_HOME="$DATA" PYTHONPATH="$MOD/app" PYTHONIOENCODING=utf-8

(
    exec >>"$DATA/launch.log" 2>&1
    echo "== $(date '+%F %T') launch  TEMP=$TEMP"
    if [ ! -x "$RT/venv/bin/python" ]; then
        # one time: uv + its own Python 3.13 + a venv (~60 MB). The big download comes later, from boot.
        echo "  making the Python environment"
        mkdir -p "$RT/uv"
        if [ ! -x "$RT/uv/uv" ]; then
            curl -sSfL https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz \
                | tar xz -C "$RT/uv" --strip-components=1 || { echo "  uv download failed"; exit 1; }
        fi
        export UV_PYTHON_INSTALL_DIR="$RT/python" UV_CACHE_DIR="$RT/uvcache"
        "$RT/uv/uv" python install --no-bin 3.13 && "$RT/uv/uv" venv --python 3.13 "$RT/venv" || { echo "  venv failed"; exit 1; }
    fi
    # the Fatshark launcher comes first: start Conductor once the game itself is running
    for _ in $(seq 1 900); do
        pgrep -f -i '^[A-Za-z]:[\].*[\]Darktide[.]exe' >/dev/null && break   # the Wine game process only
        sleep 2
    done
    echo "  game is up, starting Conductor"
    cd "$MOD/app" && exec "$RT/venv/bin/python" -m conductor.boot
) &

# Skip Fatshark's launcher: start the game exactly as the launcher does. With mods enabled the launcher
# keeps its own (empty) window open and gamescope / Steam Big Picture shows that instead of the game -
# a black screen while the game runs behind it (lounge-pc, 2026-11-24). CONDUCTOR_KEEP_LAUNCHER=1 opts out.
GAME="$(dirname "$(dirname "$MOD")")"
if [ "${CONDUCTOR_KEEP_LAUNCHER:-0}" != "1" ] && [ -f "$GAME/binaries/Darktide.exe" ]; then
    args=()
    for a in "$@"; do
        case "$a" in
            */launcher/Launcher.exe) args+=("$GAME/binaries/Darktide.exe" -eac-untrusted --bundle-dir ../bundle --ini settings) ;;
            --skip-launcher) ;;
            *) args+=("$a") ;;
        esac
    done
    cd "$GAME/binaries" || true
    exec "${args[@]}"
fi
exec "$@"
