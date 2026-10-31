#!/usr/bin/env bash
# Usage: serve_vllm_omni.sh <deploy-config.yaml> <port> [extra args...]
# Run the whole service under canhazgpu/npu so the scheduler owns the device.
set -euo pipefail
# ``docker exec --user UID:GID`` intentionally avoids root, but this image has
# no passwd entry for the host UID.  Establish the writable home *before*
# env.sh derives the Hugging Face cache paths; doing it afterwards leaves
# HF_HOME at ``/.cache/huggingface`` and Transformers cannot create dynamic
# modules.
if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ] && ! getent passwd "$(id -u)" >/dev/null 2>&1; then
  case "${USER:-}" in ""|root) export USER="minicpmo-uid-$(id -u)" ;; esac
  case "${LOGNAME:-}" in ""|root) export LOGNAME="$USER" ;; esac
  export HOME="/tmp/minicpmo-baseline-home-$(id -u)"
  mkdir -p "$HOME"
  case "${HF_HOME:-}" in ""|/.cache|/.cache/*)
    unset HF_HOME HF_HUB_CACHE HF_MODULES_CACHE
    ;;
  esac
fi
source "$(dirname "$0")/env.sh"
CFG=$1; PORT=$2; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$OMNI_VLLM_REPO"
PY="${OMNI_VLLM_PYTHON:-$OMNI_VLLM_REPO/.venv/bin/python}"
[ -x "$PY" ] || PY=python3
case "$CFG" in /*) ;; *) CFG="$OMNI_VLLM_REPO/vllm_omni/deploy/$CFG" ;; esac
if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  export PYTHONPATH="$ROOT/tools/baseline_site:$OMNI_VLLM_REPO:/vllm-workspace/vllm${PYTHONPATH:+:$PYTHONPATH}"
  export MINICPMO_VLLM_ASCEND_RMSNORM_FALLBACK=1
  # torch-npu is imported explicitly by vLLM-Ascend.  Letting PyTorch also
  # discover it through device-backend entry points can recursively import
  # torch while its triton namespace is still being registered.
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export VLLM_DISABLE_COMPILE_CACHE=1
  launcher=("$PY" "$ROOT/tools/launch_vllm_baseline.py")
else
  launcher=("$PY" -m vllm_omni.entrypoints.cli.main)
fi
exec "${launcher[@]}" serve "$MODEL_PATH" --omni \
  --served-model-name openbmb/MiniCPM-o-4_5 --host 127.0.0.1 --port "$PORT" \
  --deploy-config "$CFG" --trust-remote-code "$@"
