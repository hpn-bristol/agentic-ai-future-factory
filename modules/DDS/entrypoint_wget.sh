#!/usr/bin/env bash
set -euo pipefail

SERVER_ID="${SERVER_ID:-0}"
PORT="${DISCOVERY_PORT:-11811}"
ADDR="${DISCOVERY_ADDR:-0.0.0.0}"

set +u
source ~/Fast-DDS/install/setup.bash
set -u

if command -v fastdds >/dev/null 2>&1; then
  exec fastdds discovery --server-id "${SERVER_ID}" --udp-address "${ADDR}" --udp-port "${PORT}"
fi

if [ -f /app/discovery-server-ws/install/setup.bash ]; then
  set +u
  source /app/discovery-server-ws/install/setup.bash
  set -u
fi

if command -v discovery-server >/dev/null 2>&1; then
  exec discovery-server --server-id "${SERVER_ID}" --udp-address "${ADDR}" --udp-port "${PORT}"
fi

echo "Discovery Server binary not found. Starting shell for debugging..." >&2
exec bash