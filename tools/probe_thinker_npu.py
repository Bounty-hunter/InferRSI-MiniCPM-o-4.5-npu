"""Load the real MiniCPM-o Thinker and run one prefill plus one paged decode."""

from __future__ import annotations

import json
import os
import time

import torch

from engine.models.thinker import Thinker
from engine.runtime.npu_paged import NpuDecoderRunner, NpuPagedKV
from engine.runtime.paged import DecoderSpec


MODEL = "/dataset/models/MiniCPM-o-4_5"


@torch.inference_mode()
def main() -> None:
    torch.npu.set_device(0)
    before = time.perf_counter()
    model = Thinker.load(MODEL, device="npu:0")
    torch.npu.synchronize()
    loaded = time.perf_counter()
    c = model.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    kv = NpuPagedKV(spec, num_pages=4, page_size=128, device="npu:0")
    fused_ops = os.environ.get("NPU_FUSED_OPS", "0") == "1"
    use_graphs = os.environ.get("NPU_USE_GRAPHS", "0") == "1"
    runner = NpuDecoderRunner(model.w, spec, kv, fused_ops=fused_ops, use_graphs=use_graphs)
    ids = torch.tensor([151644, 77091, 25, 220], device="npu:0")
    reference_cache = model.new_cache(1, 16)
    rows = torch.tensor([0], device="npu:0")
    reference_hidden, _ = model.forward(model.embed(ids).unsqueeze(0), reference_cache, rows, all_logits=True)
    torch.npu.synchronize()
    prefill_started = time.perf_counter()
    layout = kv.layout([0], [len(ids)])
    hidden = runner.forward(model.embed(ids), layout)
    torch.npu.synchronize()
    prefill_elapsed = time.perf_counter() - prefill_started
    first = int(torch.nn.functional.linear(hidden[-1:], model.w["lm_head.weight"]).float().argmax(-1).cpu())
    prefilled = time.perf_counter()
    token = first
    decode_elapsed = []
    decode_cosines = []
    hidden2 = None
    for _ in range(6):
        embedded = model.embed(torch.tensor([token], device="npu:0"))
        layout = kv.layout([0], [1])
        torch.npu.synchronize()
        decode_started = time.perf_counter()
        hidden2 = runner.forward(embedded, layout)
        torch.npu.synchronize()
        decode_elapsed.append(time.perf_counter() - decode_started)
        reference_hidden2, _ = model.forward(embedded.view(1, 1, -1), reference_cache, rows)
        torch.npu.synchronize()
        decode_cosines.append(
            float(torch.nn.functional.cosine_similarity(hidden2.float(), reference_hidden2[0].float(), dim=-1).min().cpu())
        )
        token = int(torch.nn.functional.linear(hidden2, model.w["lm_head.weight"]).float().argmax(-1).cpu())
    second = token
    decoded = time.perf_counter()
    prefill_cos = torch.nn.functional.cosine_similarity(hidden.float(), reference_hidden[0].float(), dim=-1)
    payload = {
        "device": torch.npu.get_device_name(0),
        "model": MODEL,
        "layers": c.num_layers,
        "fused_ops": fused_ops,
        "use_graphs": use_graphs,
        "prefill_tokens": len(ids),
        "first_token": first,
        "second_token": second,
        "finite": bool(torch.isfinite(hidden).all() and torch.isfinite(hidden2).all()),
        "prefill_cosine_min": float(prefill_cos.min().cpu()),
        "decode_cosine_min": min(decode_cosines),
        "load_s": loaded - before,
        "prefill_s": prefill_elapsed,
        "decode_s": decode_elapsed[-1],
        "decode_steps_s": decode_elapsed,
        "decode_total_with_reference_s": decoded - prefilled,
        "npu_allocated_bytes": int(torch.npu.memory_allocated()),
        "npu_reserved_bytes": int(torch.npu.memory_reserved()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["finite"] or payload["prefill_cosine_min"] < 0.999 or payload["decode_cosine_min"] < 0.999:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
