# Current release status (2026-10-30)

This repository contains the standalone MiniCPM-o 4.5 Ascend engine snapshot used for the latest RSI profiling campaign. Runtime execution is implemented in this repository and does not import vLLM, vLLM-Omni, SGLang, or InferRSI.

The latest selected candidate used fair capture gating plus first-audio window prioritization. The native A2/Ascend 910B1 comparison used identical C1–C32 request order and two native warmups per tier. All 1,984 native requests completed without errors.

| Concurrency | TTFP ms | RTF | request/s | Change vs. r258 TTFP / RTF / throughput |
|---:|---:|---:|---:|---:|
| C1 | 675.38 | 0.27712 | 0.83926 | +2.11% / +0.26% / +0.37% |
| C4 | 1665.90 | 0.63760 | 1.43391 | +0.81% / +3.52% / +1.50% |
| C8 | 2627.45 | 1.01047 | 1.90201 | -3.53% / -1.75% / -3.35% |
| C16 | 3539.66 | 1.32953 | 2.71455 | +1.33% / +2.99% / +1.83% |
| C32 | 6315.01 | 2.27891 | 2.91337 | -1.64% / -0.38% / -6.07% |

These are single-run point estimates. They are not one-sided 95% LCB results and do not meet the project target of at least 50% latency reduction and 100% throughput improvement at every tier. C8 and C32 regressions prevent this candidate from replacing the engineering reference.

The development speech screen kept all comparative metrics within the configured margins, but the small 16-row screen did not pass every absolute floor. The final frozen speech gate was not run for this snapshot. A longer profiler recollection also exposed a CANN 507000 task-group update failure in the candidate diagnostic path; that failed diagnostic is retained as evidence and is not used as a performance claim.

This source snapshot keeps the implementation, tests, frozen plans, and reproducibility documentation needed to continue the work. Internal remote job orchestration scripts and generated result bundles are intentionally excluded from the public repository.

The publication copy passes Python bytecode compilation. The Windows packaging host did not have `pytest`, `torch`, CANN, or the Seed-TTS dataset, so NPU integration and speech tests must be run in the documented Linux/Ascend environment.
