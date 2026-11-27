@echo off
rem F1 qualifying GPU slot (user, 2026-11-20): game transitions -> critic -> taste -> policy -> Noisy Student
cd /d "%~dp0"
set HF_HOME=%~dp0.hf
set PYTHONIOENCODING=utf-8
.venv-ml\Scripts\python.exe -m conductor.critic build_game > review\ab\critic_game.log 2>&1
.venv-ml\Scripts\python.exe -m conductor.critic train > review\ab\critic_train.log 2>&1
.venv-ml\Scripts\python.exe -m conductor.critic taste review\ab\_taste_out.json > review\ab\critic_taste.log 2>&1
.venv-ml\Scripts\python.exe -m conductor.critic styles > review\ab\critic_styles.log 2>&1
python -m conductor.policy build > review\ab\policy.log 2>&1
.venv-ml\Scripts\python.exe -m conductor.distill train > review\ab\distill.log 2>&1
