#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  printf 'Usage: %s OUT_DIR\n' "$0" >&2
  exit 2
fi

out=$1
mkdir -p "$out"
python_bin=${ENGINE_PYTHON:-/usr/local/python3.12.13/bin/python}

common=(
  --concurrency 1
  --id-set tuning
  --limit 16
  --seed-index 0
  --t2w-cache-offload off
  --t2w-batch-wait-ms 10
  --thinker-pages 128
  --talker-pages 128
  --graph-max-batch 32
  --talker-k 8
)

run_case() {
  local name=$1 first=$2 steady=$3 steps=$4
  "$python_bin" tools/probe_engine_npu_quality.py "${common[@]}" \
    --out "$out/$name" \
    --first-chunk "$first" \
    --steady-chunk "$steady" \
    --t2w-timesteps "$steps" \
    >"$out/$name.stdout" 2>"$out/$name.stderr"
}

run_case f25_s25_t6 25 25 6
run_case f25_s50_t6 25 50 6
run_case f8_s50_t6 8 50 6
run_case f25_s50_t5 25 50 5
run_case f8_s50_t5 8 50 5

"$python_bin" - "$out" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
names = ("f25_s25_t6", "f25_s50_t6", "f8_s50_t6", "f25_s50_t5", "f8_s50_t5")
rows = {name: json.loads((out / name / "manifest.json").read_text()) for name in names}
base = rows[names[0]]
summary = {"schema": "minicpmo-c1-seedtts-tuning/v1", "baseline": names[0], "cases": {}}
for name in names:
    row = rows[name]
    summary["cases"][name] = {
        "n_requested": row["n_requested"],
        "n_failed": row["n_failed"],
        "ttfa_p50_s": row["ttfa_p50_s"],
        "e2e_p50_s": row["e2e_p50_s"],
        "throughput_audio_s_per_s": row["throughput_audio_s_per_s"],
        "throughput_frames_s": 25.0 * row["throughput_audio_s_per_s"],
        "versus_baseline": {
            "ttfa_speedup": base["ttfa_p50_s"] / row["ttfa_p50_s"],
            "e2e_speedup": base["e2e_p50_s"] / row["e2e_p50_s"],
            "throughput_speedup": row["throughput_audio_s_per_s"] / base["throughput_audio_s_per_s"],
        },
    }
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
