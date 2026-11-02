# Current release status (2026-10-30)

This repository contains the standalone MiniCPM-o 4.5 Ascend engine snapshot used for the latest RSI profiling campaign. Runtime execution is implemented in this repository and does not import vLLM, vLLM-Omni, SGLang, or InferRSI.

The latest selected candidate used fair capture gating plus first-audio window prioritization. The native A2/Ascend 910B1 comparison used identical C1–C32 request order and two native warmups per tier. All 1,984 native requests completed without errors.

| C | vLLM-Omni: TTFP ms / RTF / req/s | Dedicated NPU engine: TTFP ms / RTF / req/s | Change vs. vLLM-Omni: TTFP / RTF / throughput |
|---:|---:|---:|---:|
| 1 | 860.24 / 0.31147 / 0.80218 | 675.38 / 0.27712 / 0.83926 | +21.49% / +11.03% / +4.62% |
| 4 | 1507.57 / 0.65669 / 1.47004 | 1665.90 / 0.63760 / 1.43391 | -10.50% / +2.91% / -2.46% |
| 8 | 2177.56 / 1.16905 / 1.72996 | 2627.45 / 1.01047 / 1.90201 | -20.66% / +13.56% / +9.95% |
| 16 | 3621.16 / 2.09091 / 1.81075 | 3539.66 / 1.32953 / 2.71455 | +2.25% / +36.41% / +49.91% |
| 32 | 14992.29 / 4.93475 / 1.44728 | 6315.01 / 2.27891 / 2.91337 | +57.88% / +53.82% / +101.30% |

The table compares r292 with the recorded Ascend vLLM-Omni baseline (vLLM runtime 0.32.0, Omni main `5b5eb49c`). Omni was measured October 26–28; the candidate was measured October 30. It is not a simultaneous rerun.

These are single-run point estimates. They are not one-sided 95% LCB results and do not meet the project target of at least 50% latency reduction and 100% throughput improvement at every tier. In the separate internal comparison against r258, C8 and C32 regressions prevent this candidate from replacing the engineering reference. The headline comparison above is against Omni, where the remaining weaknesses are C4 throughput and C4/C8 TTFP.

The development speech screen kept all comparative metrics within the configured margins, but the small 16-row screen did not pass every absolute floor. The final frozen speech gate was not run for this snapshot. A longer profiler recollection also exposed a CANN 507000 task-group update failure in the candidate diagnostic path; that failed diagnostic is retained as evidence and is not used as a performance claim.

This source snapshot keeps the implementation, tests, frozen plans, and reproducibility documentation needed to continue the work. Internal remote job orchestration scripts and generated result bundles are intentionally excluded from the public repository.

The publication copy passes Python bytecode compilation. The Windows packaging host did not have `pytest`, `torch`, CANN, or the Seed-TTS dataset, so NPU integration and speech tests must be run in the documented Linux/Ascend environment.
