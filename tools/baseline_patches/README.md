# vLLM-Omni baseline patches

`minicpmo_4_5_stage0_native_arch_diagnostic.patch` is a diagnostic-only
isolation patch. It keeps the full Omni API, orchestration, and scheduler path
but loads Stage 0 with vLLM's native `MiniCPMO` architecture. It is never valid
for a frozen baseline performance claim; its sole purpose is to distinguish the
Omni MiniCPM-o wrapper from the Omni scheduler/connector path.

`minicpmo_4_5_stage0_native_inner_diagnostic.patch` is the narrower follow-up:
it preserves the outer `MiniCPMO45OmniForConditionalGeneration`, Omni runner,
and scheduler, but replaces the wrapper's internal Thinker implementation with
vLLM's registered `MiniCPMO`. It distinguishes the custom inner Thinker from
the outer wrapper/runner integration and is likewise never baseline evidence.

`minicpmo_4_5_stage0_no_raw_tokens_diagnostic.patch` keeps the complete custom
model stack but disables its `requires_raw_input_tokens` runner capability. The
flag exists for duplex row accounting; this probe tests whether enabling that
alternate input contract in ordinary turn mode causes the token-zero failure.

`minicpmo_4_5_stage0_passthrough_inputs_diagnostic.patch` removes the wrapper's
NPU-specific unsqueeze/slice/device-copy shim. It passes the runner's packed
token rows, embeddings, and complete MRoPE position tensor to the Thinker
unchanged, matching the input contract of the working registered `MiniCPMO`.

`minicpmo_4_5_turn_standard_sampler_diagnostic.patch` keeps the model and
forward path unchanged but limits the Thinker's custom sampler to duplex mode.
Ordinary turn mode then uses vLLM's standard sampler, matching the working
registered `MiniCPMO` control while preserving the Talker's model-owned sampler.

These patches are benchmark-only and are never imported by the dedicated
engine runtime. They are applied to an isolated checkout so the shared
`/vllm-workspace/vllm-omni` tree remains untouched.

`minicpmo_4_5_code2wav_batch_buckets.patch` applies to the frozen baseline
commit `1051faa3af996b31443e8c751f33a338a94b0a94`. It adds an optional
`cfm_graph_batch_buckets` Stage-2 setting. Live CFM batches are padded upward
to a configured bucket before NPUGraph capture and replay, then padded rows
are discarded. This changes capture shapes only; live request math remains
independent along the batch dimension.

Apply it from the root of a clean, isolated vLLM-Omni checkout:

```bash
git apply /path/to/minicpmo_4_5_code2wav_batch_buckets.patch
```

`minicpmo_4_5_code2wav_frame_bucket_graph.patch` is an incremental patch and
must be applied after the batch-bucket patch. It keeps the CFM frame-padding
attention mask as an NPUGraph tensor input, allowing the frozen baseline's
existing `cfm_graph_bucket_frames` option to collapse mel-width capture shapes
instead of forcing those calls back to eager execution.

```bash
git apply /path/to/minicpmo_4_5_code2wav_frame_bucket_graph.patch
```

The probe flag `--cfm-graph-bucket-frames N` writes a thin JSON deploy overlay
beside the server log and records both the base and effective deploy paths in
the result. The generated overlay is part of the benchmark evidence.

The checkout commit, dirty diff, patch SHA-256, and launch arguments must be
recorded in any performance evidence that uses either patch.
