#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  printf 'Usage: %s OUT_DIR\n' "$0" >&2
  exit 2
fi

out=$1
mkdir -p "$out"
python_bin=${ENGINE_PYTHON:-/usr/local/python3.12.13/bin/python}

"$python_bin" tools/probe_engine_npu_quality.py \
  --out "$out/result" \
  --concurrency 1 \
  --id-set tuning \
  --limit 2 \
  --seed-index 0 \
  --t2w-timesteps 6 \
  --first-chunk 25 \
  --steady-chunk 25 \
  --t2w-cache-offload off \
  --t2w-batch-wait-ms 10 \
  --thinker-pages 128 \
  --talker-pages 128 \
  --graph-max-batch 32 \
  --talker-k 8 \
  --profile-stages \
  --talker-profile-dir "$out/profiler" \
  >"$out/probe.stdout" 2>"$out/probe.stderr"

test -s "$out/profiler/measured.json"
"$python_bin" - "$out" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
trace = out / "profiler" / "measured.json"
doc = json.loads(trace.read_text())
events = doc.get("traceEvents", [])
complete = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0]
cats = {}
for event in complete:
    cats[event.get("cat", "")] = cats.get(event.get("cat", ""), 0) + 1
receipt = {
    "trace": str(trace),
    "trace_bytes": trace.stat().st_size,
    "complete_events": len(complete),
    "categories": dict(sorted(cats.items())),
    "manifest": str(out / "result" / "manifest.json"),
}
(out / "trace-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
