# Ascend NPU optimization map

The public optimization evidence is limited to the Ascend path. The latest opt-in candidate is r305;
[RSI and six-layer profiling](rsi-profiling-20261102.zh-CN.md) records its measured points and negative experiments.
Development parity, request success and the final speech gate are separate checks; the last has not passed.

| Mechanism | NPU implementation | Current evidence |
|---|---|---|
| Paged KV and lifetime admission | `npu_paged.py` page pools and explicit Thinker/Talker commitments | Prevents dense-KV growth and late admission failures; parity and lifecycle tests pass |
| Fixed-shape graph lanes | Thinker/Talker decode, Euler/DiT and HiFT graph buckets with eager fallback | Reduces repeated launch/capture work for common NPU shapes |
| K-step Talker blocks | Device-side sampling and state update across multiple Talker frames | Keeps the Talker handoff local and reduces per-frame host control |
| Batched Token2Wav | `chunk25` windows, cached flow/HiFT shapes and streaming audio offsets | Improves the critical audio path when batch trajectory is favorable |
| Fair capture gate | Blocks new non-nested readers while a capture writer is pending | Reduced capture starvation, but some wait moved to engine readers |
| First-audio priority | Stable ordering of `sample_offset==0` windows before continuation windows | Best r292 preliminary candidate; C8/C32 regressed in matched validation |
| Prompt-lookup verification | Request prompt supplies drafts; original Thinker verifies the causal block and truncates unused KV | r305 experimental: vs same-config r292, C2/C4/C8 mean TTFP −22.32%/−25.60%/−32.03%, request/s +16.92%/+15.43%/+28.39%; speech gate NOT_RUN |

The campaign also tested HiFT/Euler startup warmup, dynamic task-update exclusion and dense page-index reuse. Warmup
was not converted into a served gain, Euler warmup failed with CANN task-group errors, task-update exclusion regressed,
and page-index reuse did not improve end-to-end metrics. These negative results stay documented so a local improvement
is not mistaken for a service improvement.

The current C1/C2/C4/C8 direct Omni comparison is in the [r305 report](rsi-profiling-20261102.zh-CN.md).
The older five-tier r292 result remains in [the archived report](current-status-2026-11-02-main.md).
Neither single-run engineering comparison completes the final K=10/LCB/speech-gate acceptance protocol.
