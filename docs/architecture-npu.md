# Ascend NPU architecture

当前r337候选、C1/C2/C4/C8单结果曲线及后续40轮记录已更新，详见[最新优化总结](rsi-r337-20261103.zh-CN.md)和[当前性能表](current-results.md)。下文保留相应阶段的历史证据。

This document describes the runtime used for the public Ascend snapshot. The serving path is an independent
MiniCPM-o 4.5 engine; vLLM-Omni is used only as a separately launched benchmark baseline.

## Runtime ownership

| Component | Responsibility |
|---|---|
| `engine/server.py` | OpenAI-compatible `/v1/chat/completions`, request validation, streaming text/audio, health and identity endpoints |
| `engine/runtime/backend_npu.py` | Single NPU owner, admission, Thinker/Talker scheduling, KV lifetime and stage handoff |
| `engine/runtime/npu_paged.py` | Paged KV pools, batch buckets and NPU graph lanes for prefill/decode |
| `engine/runtime/thinker_lookup.py` | Optional prompt-derived drafts, original-Thinker block verification and unaccepted-KV truncation |
| `engine/runtime/talker_block_graph.py` | K-frame Talker graph replay with request-owned sampler/KV state |
| `engine/runtime/dit_fused_npu.py` | Optional NPU CFM estimator variants, with the hybrid path in the r337 configuration |
| `engine/talker_sampler.py` | Device-aware sampling, repetition history, EOS handling and per-request RNG |
| `engine/runtime/dit_graph.py` / `hift_graph.py` | Fixed-shape Euler/DiT and HiFT graph capture/replay with eager fallbacks |
| `engine/runtime/token2wav_batch.py` | Batched streaming Token2Wav windows, flow/DiT/HiFT scheduling and audio assembly |
| `engine/token2wav_chunker.py` | `chunk25` windowing and first-audio boundaries |
| `bench/` and `evals/` | Native benchmark protocol, artifact checks and WER/SIM/UTMOS quality gates |

## Request path

1. The API validates text, modality, sampling and reference audio without silently truncating the request.
2. The backend admits a request only when its Thinker and Talker KV lifetime can be committed in the paged pools.
3. Thinker and Talker state stay in the same engine runtime. The Talker is conditioned from Thinker output in place,
   and graph lanes handle fixed row/token buckets.
4. Codec tokens are sent to the batched Token2Wav worker as `chunk25` windows. Ready windows are scheduled with a
   first-audio policy while continuing streams retain playback slack.
5. Audio returns as 24 kHz mono PCM16 chunks with offsets; cancellation releases KV pages, graph state and worker state.

Every NPU workload is launched through the configured `npu run` scheduler. The runtime does not import vLLM, vLLM-Omni,
SGLang or InferRSI, and eager fallback paths remain available when a graph shape is unavailable or unsafe.

The [r337 experiment](rsi-r337-20261103.zh-CN.md) enables prompt lookup with a limit of eight draft tokens;
the backend default remains zero. A draft only proposes work: the original Thinker verifies every accepted
prediction, and KV capacity or unsupported backend paths retain ordinary decoding. This reduces serial decode
round trips on the measured speech-reading workload; it does not establish codec or speech-quality equivalence.

## Why the layout matters

The engine owns the full Thinker → Talker → Token2Wav path, so it can batch the actual audio windows, reuse paged KV,
capture the common NPU shapes and avoid handing every intermediate stage back to a generic serving scheduler. This is
the source of the potential latency and throughput advantage. It must still be validated at every concurrency with the
same request set, zero errors and the frozen speech gate; architecture alone is not a performance acceptance result.
