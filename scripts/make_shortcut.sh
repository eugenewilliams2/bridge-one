#!/bin/bash
# Rebuild the "BRIDGE ONE.app" Desktop shortcut (icon lives in assets/AppIcon.icns).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-$HOME/Desktop}/BRIDGE ONE.app"
rm -rf "$APP"; mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$REPO/assets/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>BRIDGE ONE</string>
  <key>CFBundleDisplayName</key><string>BRIDGE ONE</string>
  <key>CFBundleIdentifier</key><string>com.gq.bridgeone</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>bridge-one</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
cat > "$APP/Contents/MacOS/bridge-one" <<LAUNCH
#!/bin/bash
REPO="$REPO"
PORT="\${BRIDGESPLIT_PORT:-8765}"
URL="http://127.0.0.1:\$PORT/app"
fail() {
  osascript -e "display dialog \"\$1\" with title \"BRIDGE ONE\" buttons {\"OK\"} with icon caution" >/dev/null 2>&1
  exit 1
}
[ -d "\$REPO" ] || fail "bridge-one folder not found at \$REPO — did it move? Re-run scripts/make_shortcut.sh"
if [ ! -x "\$REPO/deepsplit/.venv/bin/python" ]; then
  open "\$REPO/Deep Split.command"
  exit 0
fi
if ! curl -s -m 2 "http://127.0.0.1:\$PORT/health" | grep -q '"ok": true'; then
  cd "\$REPO/deepsplit" || fail "deepsplit folder missing inside \$REPO"
  if [ "\$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
    nohup arch -arm64 .venv/bin/python server.py >/tmp/deepsplit.log 2>&1 &
  else
    nohup .venv/bin/python server.py >/tmp/deepsplit.log 2>&1 &
  fi
  for i in \$(seq 1 40); do
    sleep 0.5
    curl -s -m 2 "http://127.0.0.1:\$PORT/health" | grep -q '"ok": true' && break
  done
fi
curl -s -m 2 "http://127.0.0.1:\$PORT/health" | grep -q '"ok": true' || fail "The Deep Engine didn't start. Check /tmp/deepsplit.log or run Deep Split.command."
open "\$URL"
LAUNCH
chmod +x "$APP/Contents/MacOS/bridge-one"
touch "$APP"
echo "built: $APP"
