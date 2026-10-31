#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  printf 'Usage: %s OUT_DIR\n' "$0" >&2
  exit 2
fi

out=$1
python_bin=${ENGINE_PYTHON:-/usr/local/python3.12.13/bin/python}
talker_k=${POSTGRAPH_K:-8}
batch_wait_ms=${POSTGRAPH_WAIT_MS:-10}
encoder_graphs=${POSTGRAPH_ENCODER_GRAPHS:-0}
concurrency=${POSTGRAPH_CONCURRENCY:-1}
limit=${POSTGRAPH_LIMIT:-16}
graph_max_batch=${POSTGRAPH_GRAPH_MAX_BATCH:-32}
talker_pages=${POSTGRAPH_TALKER_PAGES:-128}
postgraph_enable=${POSTGRAPH_ENABLE:-1}
extra=()
if [[ "$encoder_graphs" = 1 ]]; then
  extra+=(--t2w-encoder-graphs)
fi
if [[ "$postgraph_enable" = 1 ]]; then
  extra+=(--talker-post-graphs)
fi

"$python_bin" tools/probe_engine_npu_quality.py \
  --out "$out/candidate" \
  --concurrency "$concurrency" \
  --id-set tuning \
  --limit "$limit" \
  --seed-index 0 \
  --t2w-cache-offload off \
  --t2w-batch-wait-ms "$batch_wait_ms" \
  --thinker-pages 128 \
  --talker-pages "$talker_pages" \
  --graph-max-batch "$graph_max_batch" \
  --talker-device-blocks \
  --talker-k "$talker_k" \
  --first-chunk 25 \
  --steady-chunk 25 \
  --t2w-timesteps 6 \
  "${extra[@]}" \
  >"$out.stdout" 2>"$out.stderr"

"$python_bin" - "$out/candidate/manifest.json" "$out/summary.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, summary_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
summary = {
    "schema": "minicpmo-c1-workspace-cache-postgraph-r59/v1",
    "manifest": str(manifest_path),
    "n_requested": manifest["n_requested"],
    "n_failed": manifest["n_failed"],
    "ttfa_p50_s": manifest["ttfa_p50_s"],
    "e2e_p50_s": manifest["e2e_p50_s"],
    "throughput_frames_s": 25.0 * manifest["throughput_audio_s_per_s"],
    "talker_k": manifest["talker_k"],
    "t2w_batch_wait_ms": manifest["t2w_batch_wait_ms"],
    "t2w_encoder_graphs": manifest["t2w_encoder_graphs"],
    "concurrency": manifest["concurrency"],
    "graph_max_batch": manifest["graph_max_batch"],
    "talker_pages": manifest["talker_pages"],
    "talker_post_graphs": manifest["talker_post_graphs"],
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
