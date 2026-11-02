# Pinned latest-main rerun

Omni upstream main was fetched and fixed at `9990f10d9e898c3d38b1fa640513d55f771b8685`.
The server used the unmodified main deployment and source; the measurement client stayed at official tag
`5917730a4b02aedb44d374173cdc1b10484ccd52` to match the preceding engine and tag runs.
Both the client Git SHA/PYTHONPATH and server import path/Git SHA are recorded.

This folder contains native results, exact commands, configuration/input hashes, source/runtime identity,
the job exit receipt and the recorded harness. Absolute paths refer to the measurement workspace.
The unchanged r292 engine evidence is reused from [the earlier same-day run](../20261102-omni-v0320/engine/).
Full logs are retained in the local and remote measurement workspace.

One A2 card, UID:GID 1033:1000 in `dyy-v032-r182`, image `9fd63cd20600`, original device-group wrapper and
container-local `npu run`. New server processes/port were verified gone and the reservation released.
The original checkout/service and the separate official-tag worktree were preserved.

Flow budgets differ between implementations: main FP32/10, engine FP16/6, chunk25 for both.
The main-vs-tag comparison has the same FP32/10 budget. Single-run point estimates do not establish
statistical confidence or speech-quality equivalence. No engine release promotion is implied.
