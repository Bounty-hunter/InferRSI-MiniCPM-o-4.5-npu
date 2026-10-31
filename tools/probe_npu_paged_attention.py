"""Small, model-independent Ascend parity probe for the engine's two attention paths."""

from __future__ import annotations

import json

import torch
import torch_npu


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


@torch.inference_mode()
def main() -> None:
    torch.npu.set_device(0)
    torch.manual_seed(20261022)
    dtype, heads, kv_heads, width = torch.bfloat16, 4, 2, 128
    # Ascend 910B1's paged-attention kernel advertises block size 128.
    scale, page_size = width**-0.5, 128
    lengths = [5, 9, 13]

    keys = [torch.randn(n, kv_heads, width, device="npu", dtype=dtype) for n in lengths]
    values = [torch.randn_like(key) for key in keys]
    query = torch.randn(len(lengths), heads, width, device="npu", dtype=dtype)

    pages = len(lengths)
    key_cache = torch.zeros(pages, page_size, kv_heads, width, device="npu", dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    slot_mapping = torch.tensor(
        [page * page_size + position for page, length in enumerate(lengths) for position in range(length)],
        device="npu",
        dtype=torch.int32,
    )
    torch_npu.npu_scatter_pa_kv_cache(
        key=torch.cat(keys), value=torch.cat(values), key_cache=key_cache, value_cache=value_cache,
        slot_mapping=slot_mapping, cache_mode="Norm",
    )
    block_table = torch.arange(pages, device="npu", dtype=torch.int32).view(-1, 1)
    context_lens = torch.tensor(lengths, dtype=torch.int32)
    paged = torch.empty_like(query)
    torch_npu._npu_paged_attention(
        query=query, key_cache=key_cache, value_cache=value_cache,
        num_kv_heads=kv_heads, num_heads=heads, scale_value=scale,
        block_table=block_table, context_lens=context_lens, out=paged,
    )

    reference = []
    for row, (key, value) in enumerate(zip(keys, values)):
        reference.append(torch_npu.npu_fused_infer_attention_score(
            query=query[row:row + 1].unsqueeze(2).contiguous(),
            key=key.transpose(0, 1).unsqueeze(0).contiguous(),
            value=value.transpose(0, 1).unsqueeze(0).contiguous(),
            input_layout="BNSD", num_heads=heads, num_key_value_heads=kv_heads, scale=scale, sparse_mode=0,
        )[0].squeeze(2))
    reference = torch.cat(reference)

    prefill_lengths = [7, 11]
    q = torch.randn(sum(prefill_lengths), heads, width, device="npu", dtype=dtype)
    k = torch.randn(sum(prefill_lengths), kv_heads, width, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    cumulative = torch.tensor(prefill_lengths).cumsum(0).tolist()
    tnd_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1)
    tnd = torch_npu.npu_fused_infer_attention_score(
        query=q, key=k, value=v, input_layout="TND", atten_mask=tnd_mask, actual_seq_lengths=cumulative,
        actual_seq_lengths_kv=cumulative, num_key_value_heads=kv_heads, num_heads=heads,
        scale=scale, sparse_mode=3,
    )[0]
    per_row = []
    offset = 0
    for length in prefill_lengths:
        qq = q[offset:offset + length].transpose(0, 1).unsqueeze(0)
        kk = k[offset:offset + length].transpose(0, 1).unsqueeze(0)
        vv = v[offset:offset + length].transpose(0, 1).unsqueeze(0)
        mask = torch.ones(length, length, dtype=torch.bool, device="npu").triu(1).unsqueeze(0).unsqueeze(0)
        per_row.append(torch_npu.npu_fused_infer_attention_score(
            query=qq.contiguous(), key=kk.contiguous(), value=vv.contiguous(), input_layout="BNSD", atten_mask=mask,
            num_key_value_heads=kv_heads, num_heads=heads, scale=scale, sparse_mode=0,
        )[0].squeeze(0).transpose(0, 1))
        offset += length
    per_row = torch.cat(per_row)

    payload = {
        "device": torch.npu.get_device_name(0),
        "dtype": str(dtype),
        "paged_decode_rel_l2": rel_l2(paged, reference),
        "paged_decode_max_abs": float((paged.float() - reference.float()).abs().max()),
        "tnd_prefill_rel_l2": rel_l2(tnd, per_row),
        "tnd_prefill_max_abs": float((tnd.float() - per_row.float()).abs().max()),
        "cache_write_exact": bool(torch.equal(key_cache[0, : lengths[0]], keys[0])),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["paged_decode_rel_l2"] > 0.02 or payload["tnd_prefill_rel_l2"] > 0.02 or not payload["cache_write_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
