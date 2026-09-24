#!/usr/bin/env bash
# vidgen service control: start | stop | restart | status | logs | install | uninstall
#
# No paths are hardcoded — everything resolves relative to this repo, and the
# generated launchd plist is written from the resolved values at install time.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load .env, but let variables already set by the caller win, so
# `VIDGEN_DRY_RUN=1 ./scripts/vidgen.sh restart` works whatever .env says.
if [ -f "$ROOT/.env" ]; then
  _caller_env="$(export -p | grep -E '^(declare -x|export) VIDGEN_' || true)"
  # shellcheck disable=SC1091
  set -a; . "$ROOT/.env"; set +a
  eval "$_caller_env"
  unset _caller_env
fi

PORT="${VIDGEN_PORT:-8817}"
HOST="${VIDGEN_HOST:-127.0.0.1}"
LOG_DIR="${VIDGEN_LOG_DIR:-$ROOT/logs}"
PID_FILE="$LOG_DIR/vidgen.pid"
LABEL="com.vidgen.server"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

PYTHON="${VIDGEN_PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="$(command -v python3)"; fi
fi

mkdir -p "$LOG_DIR"

running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }

start() {
  if running; then echo "vidgen already running (pid $(cat "$PID_FILE"))"; return 0; fi
  echo "starting vidgen on http://$HOST:$PORT  (python: $PYTHON)"
  cd "$ROOT"
  nohup "$PYTHON" -m vidgen --host "$HOST" --port "$PORT" >> "$LOG_DIR/vidgen.log" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 2
  if running; then echo "started (pid $(cat "$PID_FILE"))"; else
    echo "failed to start — last 20 log lines:"; tail -20 "$LOG_DIR/vidgen.log"; exit 1
  fi
}

stop() {
  if ! running; then echo "vidgen is not running"; rm -f "$PID_FILE"; return 0; fi
  local pid; pid="$(cat "$PID_FILE")"
  echo "stopping vidgen (pid $pid)"
  # SIGTERM lets the worker cancel any in-flight render and free the model.
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "stopped"
}

status() {
  if running; then
    echo "running (pid $(cat "$PID_FILE"))"
    curl -s "http://$HOST:$PORT/api/status" | python3 -m json.tool 2>/dev/null | head -30 || true
  else
    echo "not running"
  fi
}

install_agent() {
  mkdir -p "$(dirname "$PLIST")"
  cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PYTHON</string>
    <string>-m</string>
    <string>vidgen</string>
    <string>--host</string><string>$HOST</string>
    <string>--port</string><string>$PORT</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONUNBUFFERED</key><string>1</string>
    <key>VIDGEN_ENV_FILE</key><string>$ROOT/.env</string>
    <key>PATH</key><string>$(dirname "$PYTHON"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <!-- Renders are long and CPU/GPU heavy; do not let launchd throttle them,
       but do run at a nice level so the UI stays responsive. -->
  <key>Nice</key><integer>5</integer>
  <key>ProcessType</key><string>Standard</string>
  <key>StandardOutPath</key><string>$LOG_DIR/vidgen.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/vidgen.err.log</string>
</dict>
</plist>
PLIST_EOF
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load -w "$PLIST"
  echo "installed launch agent: $PLIST"
  echo "vidgen will now start at login. Manage it with:"
  echo "  launchctl unload -w $PLIST   # disable"
  echo "  launchctl load   -w $PLIST   # enable"
}

uninstall_agent() {
  launchctl unload -w "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed $PLIST"
}

case "${1:-}" in
  start)     start ;;
  stop)      stop ;;
  restart)   stop; start ;;
  status)    status ;;
  logs)      tail -f "$LOG_DIR/vidgen.log" ;;
  install)   install_agent ;;
  uninstall) uninstall_agent ;;
  *) echo "usage: $0 {start|stop|restart|status|logs|install|uninstall}"; exit 1 ;;
esac
