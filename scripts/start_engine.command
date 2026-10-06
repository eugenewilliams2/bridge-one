#!/bin/zsh
# Background-start the Deep Engine, then close this window. The Desktop app opens
# this via Terminal because macOS privacy (TCC) blocks a Finder-launched script app
# from reading ~/Documents; Terminal already has that access.
cd "$(dirname "$0")/../deepsplit" || exit 1
# force native arm64 — a Rosetta Terminal would run x86_64 Python against arm64 wheels
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
  nohup arch -arm64 .venv/bin/python server.py >/tmp/deepsplit.log 2>&1 &!
else
  nohup .venv/bin/python server.py >/tmp/deepsplit.log 2>&1 &!
fi
( sleep 1; osascript -e 'tell application "Terminal" to close (every window whose name contains "start_engine")' >/dev/null 2>&1 ) &!
exit 0
