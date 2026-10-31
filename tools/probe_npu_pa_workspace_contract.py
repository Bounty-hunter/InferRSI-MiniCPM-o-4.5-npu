"""Probe the torch-npu paged-attention workspace contract without model weights.

Run only inside an ``npu run`` reservation.  The shapes match MiniCPM-o 4.5's
talker path and the production engine's 128-page, 128-token/page KV cache.
"""

from __future__ import annotations

import json
import traceback

import torch
import torch_npu


# canhazgpu exposes the selected physical card as logical device zero inside
# the reserved process, even though ASCEND_RT_VISIBLE_DEVICES names the card.
DEVICE = "npu:0"
BATCH = 1
NUM_PAGES = 128
PAGE_SIZE = 128
NUM_HEADS = 12
NUM_KV_HEADS = 12
HEAD_DIM = 64


def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "nbytes": value.numel() * value.element_size(),
        }
    return repr(value)


def main() -> None:
    query = torch.zeros(BATCH, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    key_cache = torch.zeros(
        NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE
    )
    value_cache = torch.zeros_like(key_cache)
    output = torch.empty_like(query)
    small_blocks = torch.full((BATCH, NUM_PAGES), -1, dtype=torch.int32, device=DEVICE)
    small_blocks[:, 0] = 0
    synthetic_blocks = torch.arange(NUM_PAGES, dtype=torch.int32, device=DEVICE).repeat(BATCH, 1)
    contexts = {
        "small_cpu": torch.ones(BATCH, dtype=torch.int32),
        "small_npu": torch.ones(BATCH, dtype=torch.int32, device=DEVICE),
        "max_cpu": torch.full((BATCH,), 6144, dtype=torch.int32),
        "max_npu": torch.full((BATCH,), 6144, dtype=torch.int32, device=DEVICE),
    }
    base = {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "num_kv_heads": NUM_KV_HEADS,
        "num_heads": NUM_HEADS,
        "scale_value": HEAD_DIM**-0.5,
        "out": output,
    }
    workspaces = {}
    cases = [
        ("small_blocks_small_cpu", small_blocks, contexts["small_cpu"]),
        ("small_blocks_small_npu", small_blocks, contexts["small_npu"]),
        ("synthetic_blocks_max_cpu", synthetic_blocks, contexts["max_cpu"]),
        ("synthetic_blocks_max_npu", synthetic_blocks, contexts["max_npu"]),
    ]
    results = []
    for name, blocks, lens in cases:
        kwargs = dict(base, block_table=blocks, context_lens=lens)
        try:
            workspace = torch_npu._npu_paged_attention_get_workspace(**kwargs)
            torch.npu.synchronize()
            workspaces[name] = workspace
            results.append({"case": name, "stage": "get_workspace", "ok": True, "workspace": describe(workspace)})
        except Exception as exc:  # diagnostic tool intentionally records every case
            results.append(
                {
                    "case": name,
                    "stage": "get_workspace",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

    for workspace_name, workspace in workspaces.items():
        for context_name in ("small_cpu", "small_npu"):
            name = f"reuse_{workspace_name}_for_{context_name}"
            kwargs = dict(base, block_table=small_blocks, context_lens=contexts[context_name])
            try:
                torch_npu._npu_paged_attention(**kwargs, workspace=workspace)
                torch.npu.synchronize()
                results.append({"case": name, "stage": "paged_attention", "ok": True})
            except Exception as exc:
                results.append(
                    {
                        "case": name,
                        "stage": "paged_attention",
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
