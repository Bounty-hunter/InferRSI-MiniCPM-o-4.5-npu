#!/usr/bin/env bash
set -euo pipefail
test "$(id -u)" = 1033
test "$(id -g)" = 1000
job=/home/dengyunyang/omni-npu-jobs/minicpmo-core-components-r4-20261103
mkdir "$job"
exec >"$job/run.log" 2>&1
trap 'rc=$?; trap - EXIT; printf "%s\n" "$rc" > "$job/exit-code"; exit "$rc"' EXIT
docker exec --user 1033:1000 --workdir /workspace/minicpmo-r182 dyy-v032-r182 bash --noprofile --norc -c '
set -euo pipefail
source /workspace/minicpmo-r182/env.sh
npu status --no-schedule
exec npu run --gpus 1 --timeout 1h --wait 20m --note minicpmo-core-components-r4-20261103 -- python /workspace/minicpmo-r182/core-causality-20261103/component_controller.py
'
