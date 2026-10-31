#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  printf 'Usage: %s OUT_DIR\n' "$0" >&2
  exit 2
fi

out=$1
mkdir -p "$out"
python_bin=${ENGINE_PYTHON:-/usr/local/python3.12.13/bin/python}

"$python_bin" tools/probe_engine_npu_benchmark.py \
  --concurrency 1 \
  --requests 1 \
  --max-text-tokens 16 \
  --max-audio-tokens 128 \
  --warmup-audio-tokens 64 \
  --profile-stages \
  --timeline-markers \
  --float16-flow \
  --t2w-euler-graphs \
  --t2w-hift-graphs \
  --t2w-threaded \
  --t2w-batch-wait-ms 10 \
  --t2w-cache-offload off \
  --t2w-timesteps 6 \
  --talker-device-blocks \
  --talker-event-profile \
  --talker-k 8 \
  --first-chunk 25 \
  --steady-chunk 25 \
  --talker-profile-dir "$out/profiler" \
  >"$out/result.json" 2>"$out/result.stderr"

test -s "$out/result.json"
find "$out/profiler" -type f -printf '%s %p\n' | sort -n >"$out/profile-files.txt"
test -s "$out/profile-files.txt"

