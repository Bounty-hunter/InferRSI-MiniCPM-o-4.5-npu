# MiniCPM-o 4.5 · Ascend 910B1

A standalone streaming speech engine for [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) on one
Ascend 910B1. Thinker and Talker use a native `torch-npu` paged-KV runtime; Token2Wav runs on the same NPU with
Ascend-safe HiFT and DiT attention. The server speaks the OpenAI-compatible `/v1/chat/completions` API with streamed
text and 24 kHz audio. It does not import or run vLLM-Omni at runtime.

Current status: the graph-enabled NPU engine is end-to-end runnable. The latest native Ascend campaign completed
1,984 requests with zero request errors, but its selected candidate is not a frozen release: regressions against its
previous internal revision at C8/C32,
single-run point estimates, an incomplete final speech gate, and a profiling-path CANN task-group failure remain.
See [`docs/current-status-2026-10-30.md`](docs/current-status-2026-10-30.md) and the frozen acceptance plan under
`evidence/20261022-ascend910b1/plan/`.

## Architecture

![MiniCPM-o 4.5 Ascend architecture](docs/figures/architecture.svg)

[Ascend NPU architecture](docs/architecture-npu.md) ·
[Ascend optimization map](docs/optimizations-npu.md) ·
[Current acceptance status](docs/current-status-2026-10-30.md) ·
[Ascend test status](docs/testing-npu.md)

## Why this engine can be faster

The advantage comes from controlling the speech pipeline as one workload instead of treating Thinker, Talker and
Code2Wav as unrelated generic stages. The design targets the dominant costs observed in the baseline traces:

| Bottleneck | Dedicated engine design | Expected effect |
|---|---|---|
| Host launch and graph overhead | Fixed-shape NPU graph buckets for Thinker/Talker/DiT/HiFT, plus K-step Talker graph blocks | Fewer Python launches and fewer shape/capture transitions per generated audio window |
| Cross-stage handoff | Thinker hidden states, Talker KV and sampler state stay in the same runtime; the Talker is updated in place | Less process/RPC/host serialization and less repeated tensor materialization |
| KV growth under concurrency | Paged KV pools with explicit lifetime commitments, page reuse and admission backpressure | Ragged concurrent requests do not require a full dense KV allocation or late eviction |
| Token2Wav launch pressure | Batched streaming windows, fixed `chunk25`, cached flow/HiFT shapes and device-side sampling where supported | More audio windows are processed per accelerator launch and the first playable window can start earlier |
| Scheduling interference | A dedicated engine owner, bounded prefill budgets, a first-chunk lane and fair graph-capture gating | Continuing streams retain playback slack while new work cannot starve a capture operation |
| CPU/device synchronization | Queued copies and ordered plan submission; no unnecessary scalar observation in the hot path | The host can enqueue the next stage while the NPU is executing the current stage |

The design targets launch-bound and queue-bound behavior in the Ascend serving path: graph-captured, batched
Token2Wav windows reduce per-window control work, while paged KV and a single runtime keep the Thinker-to-Talker
handoff local. These are architectural reasons for a potential gain, not a promise that every candidate revision wins
every tier.

The latest Ascend campaign is the honest qualification boundary. The selected r292 candidate has stronger
throughput and RTF than Omni at C16/C32, while C4 throughput and C4/C8 first-audio latency remain weaker. The public
snapshot documents the mechanisms and evidence while keeping r258 as the engineering reference; it does not claim a
uniformly faster released engine.

## Precision, parity and speech quality

Performance changes are accepted only when they preserve the model contract. The repository checks or records:

- the pinned MiniCPM-o 4.5 revision, prompt construction and reference-audio identity;
- Thinker/Talker token and KV parity tests, sampler ordering, counter-based per-request RNG, EOS handling and graph
  fallback paths;
- fixed precision, sampling, six flow-matching steps, `chunk25`, request IDs and native warmup semantics for a
  benchmark comparison;
- streamed 24 kHz mono PCM output, with WER, speaker similarity, UTMOS, silence, repetition and truncation checks.

The engine is allowed to produce codec tokens that are not bit-for-bit identical to vLLM-Omni, but that is a quality
tradeoff subject to the frozen speech gates. The Ascend candidate keeps the pinned model revision, FP16 flow, six flow-
matching steps, native sampling configuration and `chunk25` workload. Thinker/Talker token and KV parity, sampler
ordering, counter-based RNG, EOS handling and eager graph fallbacks remain part of the contract. In the latest Ascend
r292 development screen, comparative differences stayed within the configured margins, but the small 16-row screen did
not satisfy every absolute WER/UTMOS floor and the final frozen speech gate was not run. No precision reduction or
quality-threshold change is hidden behind the performance numbers.

## How RSI produced the optimizations

RSI is used as an evidence loop rather than as a list of unverified tricks:

1. Freeze the model revision, request IDs, sampling, chunk size and acceptance gates.
2. Measure the unmodified path with the repository-native benchmark and stage markers.
3. Profile the earliest demonstrated cause of a bubble: host waits, graph capture contention, copies, queueing or
   shape churn. InferRSI mechanisms are hypotheses; they are reimplemented for this engine and do not inherit an
   external speedup claim.
4. Change one mechanism at a time in an isolated engine tree, keep eager fallbacks, and run focused numerical and
   concurrency tests before serving traffic.
5. Re-run the same native workload, reverse-check promising candidates, then apply speech quality and statistical
   gates. A regression or a failed runtime is retained as evidence and rejected.

The latest ten-round profiling campaign illustrates the loop:

| RSI iteration | Hypothesis | Evidence-based decision |
|---|---|---|
| Fair capture gate | Stop new non-nested readers from starving a capture writer | Capture wait fell, but some wait moved to engine readers; retained for combination testing |
| HiFT startup warmup | Move one-time graph work out of the first request | Startup work was isolated, but served gain was not established |
| Euler startup warmup | Pre-capture common Euler shapes | CANN task-group failure and service crash; rejected |
| Dynamic task-update exclusion | Serialize a problematic update region | Host wait expanded to 13.8 s; clear regression; rejected |
| First-audio priority | Process `sample_offset==0` windows before continuation windows | Best preliminary candidate, but matched validation showed small gains and C8/C32 regressions |
| Dense page-index reuse | Cache immutable page-ID tensors | Avoided repeated constructions, but did not improve end-to-end service metrics |

This process is why the repository contains both accepted mechanisms and negative results: a faster microbenchmark,
shorter lock wait or fewer object constructions is not enough unless the unprofiled native server improves without
breaking parity or speech quality.

## Quickstart

Requires Linux, Python 3.12, CANN/torch-npu, and one Ascend 910B1. Run the complete server lifetime under the local
`npu run` scheduler so the reservation is not released while child processes are still active.

```bash
npu run --gpus 1 --timeout 8h --wait 30m --note minicpmo-ascend -- \
  bash -lc 'ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:?} \
  MODEL_PATH=/dataset/models/MiniCPM-o-4_5 \
  scripts/serve_engine.sh ascend_graph_v1 18080'
```

`scripts/serve_engine.sh <config> <port>` loads `configs/engine/<config>.json` and requires the scheduler-provided
`ASCEND_RT_VISIBLE_DEVICES`. The checkpoint is found at `MODEL_PATH` or the pinned Hugging Face snapshot.

From another terminal:

```bash
curl --fail http://127.0.0.1:18080/health
REF="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582/assets/HT_ref_audio.wav
printf '{"model": "openbmb/MiniCPM-o-4_5", "stream": true, "modalities": ["text", "audio"],
 "messages": [{"role": "user", "content": "Please read the following text out loud in English: Hello from MiniCPM-o."}],
 "chat_template_kwargs": {"use_tts_template": true, "enable_thinking": false},
 "max_tokens": 256, "temperature": 0, "seed": 1, "ref_audio": "data:audio/wav;base64,%s"}' "$(base64 -w0 "$REF")" > body.json
curl --fail-with-body -N http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @body.json
```

Text arrives as `{"modality": "text", "choices": [{"delta": {"content": ...}}]}` and audio as
`{"modality": "audio", "audio_offset_samples": o, "num_samples": n, "choices": [{"delta": {"content": <base64 WAV>}}]}`
(24 kHz, mono, 16-bit PCM). The default Talker sampling is the model card's (temperature 0.8, top-p 0.85, top-k 25,
repetition penalty 1.05 over 16 frames); `talker_sampling` overrides it per request. Image, audio and video inputs use
the usual `image_url`, `input_audio`/`audio_url` and `video_url` content parts. See [Ascend API and request flow](docs/architecture-npu.md).

## Ascend performance: dedicated engine vs. vLLM-Omni

The main comparator is **vLLM-Omni running on vLLM 0.32.0**, on Ascend 910B1 (A2). The baseline's recorded Omni
checkout is [`5b5eb49c`](https://github.com/vllm-project/vllm-omni/commit/5b5eb49cdc95af80d4d2c1270055096e32ad87da).
For version accuracy: the recorded package versions are `vllm==0.32.0+empty` and
`vllm-omni==0.1.dev1+g5b5eb49cd.npu`; **0.32.0 names the vLLM runtime, not an Omni package release**.

Both use one Ascend 910B1 per service, the same Seed-TTS English tiers (32/64/128/256/512 requests at C1/C4/C8/C16/C32),
`chunk25`, and two native warmups per tier. The displayed engine is the recorded r292 candidate measured in r295.
Omni was measured on October 26–28 and the engine on October 30, 2026; this is a comparison against a previously
frozen NPU baseline, not a new simultaneous rerun.

![Dedicated NPU engine versus vLLM-Omni: TTFP, RTF, and request throughput](docs/figures/npu-performance-curves.svg)

| C | vLLM-Omni: TTFP ms / RTF / req/s | Dedicated NPU engine: TTFP ms / RTF / req/s | Change vs. vLLM-Omni: TTFP / RTF / throughput |
|---:|---:|---:|---:|
| 1 | 860.24 / 0.31147 / 0.80218 | 675.38 / 0.27712 / 0.83926 | +21.49% / +11.03% / +4.62% |
| 4 | 1507.57 / 0.65669 / 1.47004 | 1665.90 / 0.63760 / 1.43391 | -10.50% / +2.91% / -2.46% |
| 8 | 2177.56 / 1.16905 / 1.72996 | 2627.45 / 1.01047 / 1.90201 | -20.66% / +13.56% / +9.95% |
| 16 | 3621.16 / 2.09091 / 1.81075 | 3539.66 / 1.32953 / 2.71455 | +2.25% / +36.41% / +49.91% |
| 32 | 14992.29 / 4.93475 / 1.44728 | 6315.01 / 2.27891 / 2.91337 | +57.88% / +53.82% / +101.30% |

TTFP and RTF: lower is better. Throughput: higher is better. Positive percentages mean improvement over Omni;
negative percentages show regressions. C4 is slower in TTFP and throughput, and C8 is slower in TTFP. At C16 the
engine improves TTFP / RTF / throughput by 2.25% / 36.41% / 49.91%; at C32 by 57.88% / 53.82% / 101.30%.
These are system-level differences, not the isolated gain of the latest RSI edit.

![Relative improvements over the recorded Ascend Omni baseline](docs/figures/npu-improvement-curves.svg)

Each displayed system completed 992 measured requests with zero request errors. The engine's separate matched
comparison against its previous r258 revision also completed successfully; that internal RSI comparison is retained in
the [source data](docs/data/ascend-r295-comparison.json), not used as the headline comparator.

All points are single-run estimates. No confidence intervals or frozen one-sided 95% LCB are available, the full speech
gate has not passed, and the long-profiler candidate run failed separately. These measurements do not qualify r292 as
a released/default engine. [Quality and acceptance status](docs/current-status-2026-10-30.md).

[Download PNG curves](docs/figures/npu-performance-curves.png) ·
[Download PNG improvements](docs/figures/npu-improvement-curves.png) ·
[Data and provenance](docs/data/ascend-r295-comparison.json)

To regenerate the curves with Python and Matplotlib:

```bash
python -m pip install matplotlib
python tools/plot_npu_performance.py
```

The [original frozen Ascend plan](evidence/20261022-ascend910b1/plan/acceptance.json) remains preserved for audit.
Its original baseline and K=10 protocol predate this vLLM 0.32 campaign; the curves above do not claim to complete that
frozen acceptance. All displayed performance measurements are from Ascend NPUs.

## References

- [MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5): checkpoint, processor, encoders and the
  `streaming_generate` reference ported here.
- [minicpmo-utils](https://pypi.org/project/minicpmo-utils/) (`stepaudio2`): the reference Token2Wav, CosyVoice2 flow and
  HiFT modules.
- [PyTorch](https://github.com/pytorch/pytorch) and `torch-npu`: accelerator operators and graph execution.
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) and [SGLang-Omni](https://github.com/sgl-project/sglang-omni):
  serving baselines and the speech evaluation tools (Seed-TTS WER, speaker similarity).
- [Seed-TTS eval](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval): the voice-cloning workload.
- [OmniRSI](https://github.com/ThinkFlowLab/OmniRSI): recorded optimization knowledge and evidence.

See [NOTICE](NOTICE) and the [Apache-2.0 license text](third_party_licenses/Apache-2.0.txt) of the reused code.
