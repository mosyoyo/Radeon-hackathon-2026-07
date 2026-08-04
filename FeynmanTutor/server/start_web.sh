#!/bin/bash
# start_web.sh — launch the FeynmanTutor web front-end.
#
# Requires the vLLM server to already be running (server/start_vllm.sh),
# because the web app proxies model calls to http://127.0.0.1:8000/v1.
#
#   bash server/start_web.sh            # bind 127.0.0.1:8500
#   PORT=8501 bash server/start_web.sh  # different port
#
# To expose it to the internet from a Radeon Cloud instance, use rc-tunnel:
#   $HOME/.local/bin/rc-tunnel expose --port 8500

set -eu

PY="${PYTHON:-/opt/venv/bin/python}"
HOST="${WEB_HOST:-127.0.0.1}"
PORT="${WEB_PORT:-8500}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Using python: $PY"
echo "Web UI will listen on http://$HOST:$PORT"
echo "Make sure the vLLM server is up at http://127.0.0.1:8000/v1"

exec "$PY" -m uvicorn web.app:app \
  --host "$HOST" --port "$PORT" \
  --app-dir "$ROOT" \
  --log-level info
