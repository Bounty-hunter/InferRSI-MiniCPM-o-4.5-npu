#!/usr/bin/env bash
# Usage: serve_engine.sh <config-name|path.json> <port> [extra args...]
# Run the whole service inside ``npu run``; that scheduler supplies
# ASCEND_RT_VISIBLE_DEVICES and owns the reservation for the service lifetime.
set -euo pipefail
source "$(dirname "$0")/env.sh"
CFG=$1; PORT=$2; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$CFG" in /*|*.json) ;; *) CFG="$ROOT/configs/engine/$CFG.json" ;; esac
cd "$ROOT"
PY="${ENGINE_PYTHON:-$ROOT/.venv-engine/bin/python}"
[ -x "$PY" ] || PY=python3
: "${ASCEND_RT_VISIBLE_DEVICES:?serve_engine.sh must run inside an npu allocation}"
exec "$PY" -m engine.server --host 127.0.0.1 --port "$PORT" --config "$CFG" "$@"
