#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  printf 'Usage: %s OUT_DIR BASELINE_MANIFEST\n' "$0" >&2
  exit 2
fi

out=$1
baseline=$2
python_bin=${ENGINE_PYTHON:-/usr/local/python3.12.13/bin/python}
test -f "$baseline"

"$python_bin" tools/probe_engine_npu_quality.py \
  --out "$out/candidate" \
  --concurrency 1 \
  --id-set tuning \
  --limit 16 \
  --seed-index 0 \
  --t2w-cache-offload off \
  --t2w-batch-wait-ms 10 \
  --thinker-pages 128 \
  --talker-pages 128 \
  --graph-max-batch 32 \
  --talker-device-blocks \
  --talker-k 8 \
  --first-chunk 25 \
  --steady-chunk 25 \
  --t2w-timesteps 6 \
  >"$out.stdout" 2>"$out.stderr"

"$python_bin" - "$baseline" "$out/candidate/manifest.json" "$out/comparison.json" <<'PY'
import json
import sys
from pathlib import Path

baseline_path, candidate_path, output_path = map(Path, sys.argv[1:])
baseline = json.loads(baseline_path.read_text())
candidate = json.loads(candidate_path.read_text())

def hashes(manifest, key):
    return {row["request_id"]: row[key] for row in manifest["rows"]}

base_f32 = hashes(baseline, "audio_sha256_float32")
cand_f32 = hashes(candidate, "audio_sha256_float32")
base_wav = hashes(baseline, "audio_sha256_pcm16_wav")
cand_wav = hashes(candidate, "audio_sha256_pcm16_wav")
request_ids_match = set(base_f32) == set(cand_f32)
float32_mismatches = sorted(key for key in set(base_f32) | set(cand_f32) if base_f32.get(key) != cand_f32.get(key))
wav_mismatches = sorted(key for key in set(base_wav) | set(cand_wav) if base_wav.get(key) != cand_wav.get(key))
report = {
    "schema": "minicpmo-c1-workspace-cache-r56/v1",
    "baseline_manifest": str(baseline_path),
    "candidate_manifest": str(candidate_path),
    "request_ids_match": request_ids_match,
    "float32_mismatches": float32_mismatches,
    "wav_mismatches": wav_mismatches,
    "bitwise_audio_match": request_ids_match and not float32_mismatches and not wav_mismatches,
    "candidate": {
        "n_requested": candidate["n_requested"],
        "n_failed": candidate["n_failed"],
        "ttfa_p50_s": candidate["ttfa_p50_s"],
        "e2e_p50_s": candidate["e2e_p50_s"],
        "throughput_frames_s": 25.0 * candidate["throughput_audio_s_per_s"],
    },
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
if not report["bitwise_audio_match"]:
    raise SystemExit(1)
PY
