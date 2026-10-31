"""Real-checkpoint Talker prefill/decode parity on Ascend."""

from __future__ import annotations

import json
import os
import time

import torch

from engine.models.talker import Talker
from engine.runtime.npu_paged import NpuDecoderRunner, NpuPagedKV
from engine.runtime.paged import DecoderSpec


MODEL = "/dataset/models/MiniCPM-o-4_5"


@torch.inference_mode()
def main() -> None:
    torch.npu.set_device(0)
    started = time.perf_counter()
    model = Talker.load(MODEL, device="npu:0")
    c = model.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, False)
    kv = NpuPagedKV(spec, 4, 128, device="npu:0")
    fused_ops = os.environ.get("NPU_FUSED_OPS", "0") == "1"
    use_graphs = os.environ.get("NPU_USE_GRAPHS", "0") == "1"
    runner = NpuDecoderRunner(model.w, spec, kv, fused_ops=fused_ops, use_graphs=use_graphs)
    ids = torch.tensor([100, 200, 300, 400], device="npu:0")
    generator = torch.Generator(device="npu:0").manual_seed(7)
    reply_hidden = torch.randn(4, 4096, generator=generator, device="npu:0", dtype=torch.bfloat16)
    condition = model.condition(ids, reply_hidden)
    cache = model.new_cache(1, 32)
    rows = torch.tensor([0], device="npu:0")
    reference_logits = model.forward(condition.unsqueeze(0), cache, rows)
    hidden = runner.forward(condition, kv.layout([0], [condition.shape[0]]))
    logits = torch.nn.functional.linear(hidden[-1:], model.head).float()
    first = int(logits.argmax(-1).cpu())

    def cosine(a, b):
        return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).min().cpu())

    code = first
    decode_elapsed = []
    decode_cosines = []
    logits_next = None
    ref_next = None
    for _ in range(6):
        embedded = model.code_embed(torch.tensor([code], device="npu:0"))
        torch.npu.synchronize()
        decode_started = time.perf_counter()
        hidden_next = runner.forward(embedded, kv.layout([0], [1]))
        torch.npu.synchronize()
        decode_elapsed.append(time.perf_counter() - decode_started)
        logits_next = torch.nn.functional.linear(hidden_next, model.head).float()
        ref_next = model.forward(embedded.view(1, 1, -1), cache, rows)
        torch.npu.synchronize()
        decode_cosines.append(cosine(logits_next, ref_next))
        code = int(logits_next.argmax(-1).cpu())
    payload = {
        "device": torch.npu.get_device_name(0),
        "layers": c.num_layers,
        "fused_ops": fused_ops,
        "use_graphs": use_graphs,
        "condition_tokens": int(condition.shape[0]),
        "first_code": first,
        "prefill_logits_cosine": cosine(logits, reference_logits),
        "decode_logits_cosine": min(decode_cosines),
        "decode_steps_s": decode_elapsed,
        "decode_s": decode_elapsed[-1],
        "elapsed_s": time.perf_counter() - started,
        "npu_allocated_bytes": int(torch.npu.memory_allocated()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if min(payload["prefill_logits_cosine"], payload["decode_logits_cosine"]) < 0.999:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
