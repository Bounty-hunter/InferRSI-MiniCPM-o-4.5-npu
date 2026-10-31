# Ascend NPU optimization map

The public optimization claims are limited to the Ascend path. Each change was isolated, checked for parity, and
measured with the native Seed-TTS workload before it could remain in the candidate.

| Mechanism | NPU implementation | Current evidence |
|---|---|---|
| Paged KV and lifetime admission | `npu_paged.py` page pools and explicit Thinker/Talker commitments | Prevents dense-KV growth and late admission failures; parity and lifecycle tests pass |
| Fixed-shape graph lanes | Thinker/Talker decode, Euler/DiT and HiFT graph buckets with eager fallback | Reduces repeated launch/capture work for common NPU shapes |
| K-step Talker blocks | Device-side sampling and state update across multiple Talker frames | Keeps the Talker handoff local and reduces per-frame host control |
| Batched Token2Wav | `chunk25` windows, cached flow/HiFT shapes and streaming audio offsets | Improves the critical audio path when batch trajectory is favorable |
| Fair capture gate | Blocks new non-nested readers while a capture writer is pending | Reduced capture starvation, but some wait moved to engine readers |
| First-audio priority | Stable ordering of `sample_offset==0` windows before continuation windows | Best r292 preliminary candidate; C8/C32 regressed in matched validation |

The campaign also tested HiFT/Euler startup warmup, dynamic task-update exclusion and dense page-index reuse. Warmup
was not converted into a served gain, Euler warmup failed with CANN task-group errors, task-update exclusion regressed,
and page-index reuse did not improve end-to-end metrics. These negative results stay documented so a local improvement
is not mistaken for a service improvement.

The current five-tier Ascend result is in [`current-status-2026-10-30.md`](current-status-2026-10-30.md). It is a
single-run engineering comparison, not the final K=10/LCB/speech-gate acceptance result.
