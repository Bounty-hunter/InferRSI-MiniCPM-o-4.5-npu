#!/usr/bin/env bash
set -euo pipefail
test "$(id -u)" = 1033
test "$(id -g)" = 1000
job=/home/dengyunyang/omni-npu-jobs/minicpmo-r337-c1B2-20261102
mkdir "$job"
exec >"$job/run.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" > "$job/exit-code"; exit "$rc"' EXIT
docker exec --user 1033:1000 --workdir /workspace/minicpmo-r182 dyy-v032-r182 bash --noprofile --norc -c '
set -euo pipefail
source /workspace/minicpmo-r182/env.sh
npu status --no-schedule
npu run --gpus 1 --timeout 10m --wait 10m --note minicpmo-r337-c1B2-20261102 -- python /workspace/minicpmo-r182/rsi-r312-r351-20261102/run_lowc.py /workspace/minicpmo-r182/rsi-r312-r351-20261102/r337-c1B2.json
'
