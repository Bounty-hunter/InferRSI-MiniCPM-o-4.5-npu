#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  printf 'Usage: %s K PORT GPU_ID OUT_DIR\n' "$0" >&2
  exit 2
fi

k=$1
port=$2
gpu_id=$3
out_dir=$4
case "$k" in
  1|3) ;;
  *) printf 'r151 only permits the frozen K=1 or K=3 C16 screen\n' >&2; exit 2 ;;
esac
[[ "$port" =~ ^[1-9][0-9]*$ ]] && (( port <= 65535 )) || { printf 'invalid PORT\n' >&2; exit 2; }
[[ "$gpu_id" == 5 ]] || { printf 'r151 is frozen for physical NPU 5\n' >&2; exit 2; }
[[ "$out_dir" == /tmp/minicpmo-engine-c16-k* ]] || { printf 'invalid OUT_DIR\n' >&2; exit 2; }
[[ ! -e "$out_dir" ]] || { printf 'refusing to overwrite existing OUT_DIR: %s\n' "$out_dir" >&2; exit 1; }

metrics_audit="$out_dir/postgraph_metrics.jsonl"
/usr/local/python3.12.13/bin/python tools/capture_r151_postgraph_metrics.py \
  --url "http://127.0.0.1:${port}" \
  --out "$metrics_audit" \
  --interval 2.0 &
capture_pid=$!

cleanup() {
  local status=$?
  trap - EXIT
  kill "$capture_pid" 2>/dev/null || true
  wait "$capture_pid" 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

/usr/local/python3.12.13/bin/python -m bench.cli cell \
  --adapter engine \
  --config ascend_c16_r151_omni_counter_postgraph \
  --gpu "$gpu_id" \
  --port "$port" \
  --concurrency 16 \
  --k "$k" \
  --id-set eval \
  --kind performance \
  --mps off \
  --seed-index 0 \
  --run-order-seed 20261024 \
  --timeout 180 \
  --warmup-timeout 900 \
  --out "$out_dir" \
  --no-audio \
  --label "r151-c16-omni-counter-postgraph-k${k}"
