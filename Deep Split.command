#!/bin/zsh
# BRIDGE ONE — Deep Split launcher. Double-click to start the AI stem engine.
cd "$(dirname "$0")/deepsplit" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "First-time setup: installing the AI engine (one-time, a few minutes)…"
  python3 -m venv .venv && .venv/bin/pip install --quiet --upgrade pip && .venv/bin/pip install demucs soundfile || {
    echo "install failed — check your internet connection and run this again"; read -r; exit 1; }
fi
( sleep 2; open "http://127.0.0.1:${BRIDGESPLIT_PORT:-8765}" ) &
exec .venv/bin/python server.py
