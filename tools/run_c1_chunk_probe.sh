#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  printf 'Usage: %s OUT_DIR\n' "$0" >&2
  exit 2
fi

out=$1
mkdir -p "$out"

common=(
  --concurrency 1
  --requests 16
  --max-text-tokens 16
  --max-audio-tokens 256
  --warmup-audio-tokens 128
  --float16-flow
  --t2w-euler-graphs
  --t2w-hift-graphs
  --t2w-threaded
  --t2w-batch-wait-ms 10
  --talker-k 8
  --spk-threads 2
)

python_bin=${ENGINE_PYTHON:-/usr/local/python3.12.13/bin/python}

run_case() {
  local name=$1 first=$2 steady=$3 steps=$4
  "$python_bin" tools/probe_engine_npu_benchmark.py "${common[@]}" \
    --first-chunk "$first" --steady-chunk "$steady" --t2w-timesteps "$steps" \
    >"$out/$name.json" 2>"$out/$name.stderr"
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
rows = {}
for name in names:
    raw = (out / f"{name}.json").read_text()
    # CANN may print its /ascend/log warning to stdout before Python emits the
    # benchmark JSON.  Preserve the raw receipt and parse from the first JSON
    # object instead of treating that environment warning as benchmark data.
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"{name}.json does not contain a JSON object")
    rows[name] = json.loads(raw[start:])
base = rows[names[0]]
summary = {"baseline": names[0], "cases": {}}
for name in names:
    row = rows[name]
    summary["cases"][name] = {
        "ttfa_p50_s": row["ttfa_p50_s"],
        "e2e_p50_s": row["e2e_p50_s"],
        "throughput_audio_s_per_s": row["throughput_audio_s_per_s"],
        "n_failed": row["n_failed"],
        "versus_baseline": {
            "ttfa_speedup": base["ttfa_p50_s"] / row["ttfa_p50_s"],
            "e2e_speedup": base["e2e_p50_s"] / row["e2e_p50_s"],
            "throughput_speedup": row["throughput_audio_s_per_s"] / base["throughput_audio_s_per_s"],
        },
    }
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
