# Ascend NPU optimization map

当前r337候选、C1/C2/C4/C8单结果曲线及后续40轮记录已更新，详见[最新优化总结](rsi-r337-20261103.zh-CN.md)和[当前性能表](current-results.md)。下文保留相应阶段的历史证据。

The public optimization evidence is limited to the Ascend path. The latest opt-in candidate is r337.
[the full optimization journey](optimization-journey.zh-CN.md) covers early RSI adaptation and profiling/AI-derived
strategies before the last ten rounds. The [complete NPU ledger](optimization-ledger-npu.zh-CN.md) retains failures and withdrawn claims.
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
| Media graph ownership | r307/r309 warm bounded Euler/HiFT shapes on the actual audio worker/prompt stream before health readiness | Addresses observed first-use captures and preserves scratch ownership |
| Process-local resource limit | r337 uses12Cube/24Vector before model operators and graph capture | Final same-engine paired observations show positive RTF/throughput differences, while C4TTFP spans zero |
| Host query lengths | r337 reuses validated layout-builder cumulative lengths and retains device-readback fallback | Removes repeated host observation in this engine's prefill path |

The earlier r297–r306 campaign also tested HiFT/Euler startup warmup, dynamic task-update exclusion and dense page-index reuse. Warmup
was not converted into a served gain, Euler warmup failed with CANN task-group errors, task-update exclusion regressed,
and page-index reuse did not improve end-to-end metrics. These negative results stay documented so a local improvement
is not mistaken for a service improvement.

The current C1/C2/C4/C8 direct Omni comparison is in the [r337 report](rsi-r337-20261103.zh-CN.md).
The [r305 report](rsi-profiling-20261102.zh-CN.md) preserves its earlier measurement scope.
The older five-tier r292 result remains in [the archived report](current-status-2026-11-02-main.md).
Neither single-run engineering comparison completes the final K=10/LCB/speech-gate acceptance protocol.
