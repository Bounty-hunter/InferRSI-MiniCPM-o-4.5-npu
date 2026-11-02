# Official-tag rerun evidence

Measured November 2, 2026 on Ascend 910B1 (A2), in `dyy-v032-r182`, image `9fd63cd20600`,
as UID:GID 1033:1000 under the original device-group wrapper and container-local `npu run`.
Both services were assigned physical card 3 sequentially. Both jobs exited with code 0;
their new service processes and listening ports were gone and card 3 was released after the runs.
The original checkout/service was preserved.

- `omni/`: clean tag `v0.32.0` at `5917730a4b02aedb44d374173cdc1b10484ccd52`, native result files,
  exact commands, deployment, model/dataset input hashes, runtime import path and versions.
- `engine/`: unchanged r292 candidate, served config and source hash, the same native client commands/results,
  server counters and runtime import audit. This benchmark candidate is not promoted to the repository default.
- `minicpmo-*/exit-code`: original job exit receipts.
- `run_baseline.py`, `run_candidate.py`, `candidate-spec.json`: recorded rerun harnesses. Their absolute paths
  refer to the measurement workspace; they require the existing frozen model, engine snapshot and environment.
- `summarize.py`: checks source/config identities, identical normalized client commands and input-token counts,
  zero errors, server exits and forbidden-import audit before deriving the published comparison.

The full logs are retained in the measurement workspace. Log review found no ERROR/Traceback lines; existing
package/version, device-environment, reference-audio truncation and shutdown warnings remain in the raw logs.
The editable package metadata still names the older development install; the imported path and clean Git SHA
identify the executed Omni source. No package metadata was rewritten to imitate a release install.

The A2-specific CI case lists C1/C4/C8; C16/C32 use the tag's generic five-tier request counts.
Each service completed 992 measured requests, plus two native warmups per tier. Flow settings differ:
Omni FP32/10 steps versus engine FP16/6 steps, both chunk25. These are single-run system measurements,
without equal-precision attribution, a frozen confidence bound or speech-quality qualification.
