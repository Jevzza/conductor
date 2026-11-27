@echo off
rem Generations run (user, 2026-11-22): refresh with the latest answers, then 2 h of critic generations
cd /d "%~dp0"
set HF_HOME=%~dp0.hf
set PYTHONIOENCODING=utf-8
python -m conductor.ab nightly 30 > review\ab\gens_nightly.log 2>&1
.venv-ml\Scripts\python.exe -m conductor.generations 2 10 0.03 > review\ab\generations.log 2>&1
