# Repository contract

- This is a standalone MiniCPM-o 4.5 Ascend engine. Runtime imports from vLLM,
  vLLM-Omni, SGLang, or InferRSI are forbidden; those repositories are
  references and baselines only.
- Preserve the model revision and the frozen acceptance plan under
  `evidence/20261022-ascend910b1/plan/`.
- Performance claims require one Ascend 910B1, identical model/request IDs,
  verified artifacts, and the frozen validator. GPU results copied from the
  source engine are never NPU evidence.
- NPU jobs run inside the `dyy` container through `npu run`; do not bypass the
  scheduler or modify the shared `/vllm-workspace/vllm-omni` checkout.
- Keep eager correctness fallbacks when adding fused operators or NPUGraphs.
