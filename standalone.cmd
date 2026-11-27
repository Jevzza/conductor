@echo off
rem Conductor standalone (no game): buttons for Calm / Tension / Fight / Horde / Boss + Pause.
rem Opens the control page; add --lan below to control it from a phone/tablet on your network.
cd /d "%~dp0"
start "" http://127.0.0.1:8770
python -m conductor standalone %*
