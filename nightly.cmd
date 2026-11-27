@echo off
rem Conductor nightly training: learn from A/B answers, retrain, make tomorrow's 15 pairs.
cd /d "%~dp0"
echo ==== %date% %time% >> review\ab\nightly.log
python -m conductor.ab nightly 30 >> review\ab\nightly.log 2>&1
