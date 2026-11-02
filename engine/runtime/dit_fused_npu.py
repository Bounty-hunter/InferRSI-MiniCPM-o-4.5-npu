# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional fused MiniCPM-o Code2Wav CFM body for Ascend.

This is a standalone adaptation of vLLM-Omni's MiniCPM-o NPU fused DiT body.
It deliberately has no runtime dependency on vLLM or vLLM-Omni.  The caller
keeps the reference estimator as the default and selects this function only
for explicitly qualified NPUGraph shapes.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.runtime.npu_token2wav_patch import _expand_mask


_SHIFT_MSA, _SCALE_MSA, _GATE_MSA, _SHIFT_MLP, _SCALE_MLP, _GATE_MLP, _SHIFT_CONV, _SCALE_CONV, _GATE_CONV = range(9)


def supports_fused_body(estimator: nn.Module) -> bool:
    blocks = getattr(estimator, "blocks", None)
    final_layer = getattr(estimator, "final_layer", None)
    if not blocks or not hasattr(estimator, "in_proj") or final_layer is None:
        return False
    if not isinstance(getattr(final_layer, "norm_final", None), nn.LayerNorm) or not hasattr(final_layer, "linear"):
        return False
    for block in blocks:
        attn = getattr(block, "attn", None)
        conv_block = getattr(getattr(block, "conv", None), "block", None)
        mlp = getattr(block, "mlp", None)
        if attn is None or conv_block is None or mlp is None:
            return False
        if not all(hasattr(attn, name) for name in ("to_q", "to_k", "to_v", "proj", "num_heads", "head_dim")):
            return False
        if not (isinstance(attn.q_norm, nn.LayerNorm) and isinstance(attn.k_norm, nn.LayerNorm)):
            return False
        if not all(isinstance(getattr(block, name, None), nn.LayerNorm) for name in ("norm1", "norm2", "norm3")):
            return False
        if len(conv_block) < 7 or not isinstance(conv_block[3], nn.LayerNorm) or not isinstance(conv_block[4], nn.Mish):
            return False
        for conv in (conv_block[1], conv_block[6]):
            if not isinstance(conv, nn.Conv1d) or conv.stride != (1,) or conv.dilation != (1,) or conv.groups != 1:
                return False
            if conv.padding != (0,) or conv.kernel_size[0] < 2:
                return False
        if not (hasattr(mlp, "fc1") and hasattr(mlp, "fc2") and isinstance(mlp.act, nn.GELU)):
            return False
        if not isinstance(getattr(mlp, "norm", nn.Identity()), nn.Identity):
            return False
    return True


def supports_packed_qkv(attn: nn.Module) -> bool:
    projections: list[nn.Linear] = []
    for name in ("to_q", "to_k", "to_v"):
        projection = getattr(attn, name, None)
        if type(projection) is not nn.Linear:
            return False
        projections.append(cast(nn.Linear, projection))
    reference = projections[0]
    return all(p.weight.shape == reference.weight.shape for p in projections) and (
        all(p.bias is None for p in projections) or all(p.bias is not None for p in projections)
    )


def _same_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
        and a.device == b.device
    )


def packed_qkv(attn: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    projections = (attn.to_q, attn.to_k, attn.to_v)
    cached = attn.__dict__.get("_minicpmo_npu_packed_qkv")
    rows = projections[0].out_features
    if cached is not None:
        weight, bias = cached
        if all(
            _same_view(p.weight, weight[i * rows : (i + 1) * rows])
            and (
                (p.bias is None and bias is None)
                or (p.bias is not None and bias is not None and _same_view(p.bias, bias[i * rows : (i + 1) * rows]))
            )
            for i, p in enumerate(projections)
        ):
            return cached
    if projections[0].weight.device.type == "npu" and torch.npu.is_current_stream_capturing():
        raise RuntimeError("CFM QKV weights must be packed during warmup, before graph capture")
    with torch.no_grad():
        weight = torch.cat([p.weight for p in projections])
        bias = None if projections[0].bias is None else torch.cat([p.bias for p in projections])
        for i, p in enumerate(projections):
            p.weight.data = weight[i * rows : (i + 1) * rows]
            if bias is not None:
                p.bias.data = bias[i * rows : (i + 1) * rows]
    attn.__dict__["_minicpmo_npu_packed_qkv"] = weight, bias
    return weight, bias


def packed_conv_columns(conv: nn.Conv1d) -> torch.Tensor:
    shape = (conv.out_channels, conv.kernel_size[0], conv.in_channels)
    packed = conv.__dict__.get("_minicpmo_npu_packed_conv_columns")
    if packed is not None and _same_view(conv.weight, packed.view(shape).permute(0, 2, 1)):
        return packed
    if conv.weight.device.type == "npu" and torch.npu.is_current_stream_capturing():
        raise RuntimeError("CFM convolution weights must be packed during warmup, before graph capture")
    with torch.no_grad():
        packed = conv.weight.detach().permute(0, 2, 1).reshape(conv.out_channels, -1).contiguous()
        conv.weight.data = packed.view(shape).permute(0, 2, 1)
    conv.__dict__["_minicpmo_npu_packed_conv_columns"] = packed
    return packed


def prepare_estimator(estimator: nn.Module, *, pack_convs: bool = False) -> None:
    if not supports_fused_body(estimator):
        raise ValueError("unsupported CFM estimator structure")
    for block in estimator.blocks:
        if not block.training and supports_packed_qkv(block.attn):
            packed_qkv(block.attn)
        if pack_convs:
            for conv in (block.conv.block[1], block.conv.block[6]):
                if not conv.training:
                    packed_conv_columns(conv)


def dit_modulation(estimator: nn.Module, time_embedding: torch.Tensor) -> torch.Tensor:
    embedding = time_embedding.reshape(1, 1, -1)
    table = torch.stack([block.adaLN_modulation(embedding).reshape(9, -1) for block in estimator.blocks], dim=0)
    table[:, _SCALE_MSA::3].add_(1.0)
    final_shift, final_scale = estimator.final_layer.adaLN_modulation(embedding).reshape(2, -1)
    final = table.new_zeros((1, 9, int(table.shape[2])))
    final[0, _SHIFT_MSA] = final_shift
    final[0, _SCALE_MSA] = final_scale + 1.0
    return torch.cat((table, final), dim=0).contiguous()


def _norm_affine(norm: nn.LayerNorm, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight, bias = norm.weight, norm.bias
    if weight is not None and bias is not None:
        return weight, bias
    cached = norm.__dict__.get("_minicpmo_npu_norm_affine")
    if cached is None or cached[0].device != like.device or cached[0].dtype != like.dtype:
        cached = (like.new_ones(like.shape[-1]), like.new_zeros(like.shape[-1]))
        norm.__dict__["_minicpmo_npu_norm_affine"] = cached
    return (cached[0] if weight is None else weight), (cached[1] if bias is None else bias)


def _add_layer_norm(
    x: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    fused: bool = True,
):
    op = None
    if fused and x.device.type == "npu":
        try:
            import torch_npu

            op = getattr(torch_npu, "npu_add_layer_norm", None)
        except ImportError:
            pass
    if op is None:
        added = x + branch
        return added, F.layer_norm(added, (int(added.shape[-1]),), weight, bias, eps)
    normalized, _, _, added = op(x, branch, weight, bias, epsilon=float(eps), additional_output=True)
    return added, normalized


def _causal_conv(conv: nn.Conv1d, history: torch.Tensor, frames: int, *, im2col: bool = True) -> torch.Tensor:
    if not im2col:
        return torch.nn.Conv1d.forward(conv, history.transpose(1, 2)).transpose(1, 2)
    rows, _, channels = history.shape
    taps = int(conv.kernel_size[0])
    columns = history.as_strided((rows, frames, taps * channels), (int(history.stride(0)), channels, 1))
    out = F.linear(columns.reshape(rows * frames, taps * channels), packed_conv_columns(conv), conv.bias)
    return out.view(rows, frames, -1)


def _frames_major_cache(cnn_cache: torch.Tensor, width: int):
    if width <= 0:
        return None
    return cnn_cache.transpose(-1, -2).contiguous()


def _attach_history(old: torch.Tensor | None, current: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return current
    if old is None:
        return F.pad(current, (0, 0, width, 0))
    return torch.cat((old, current), dim=1)


def _mlp(mlp: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    rows = hidden.reshape(-1, int(hidden.shape[-1]))
    inner = mlp.act(F.linear(rows, mlp.fc1.weight, mlp.fc1.bias))
    return F.linear(inner, mlp.fc2.weight, mlp.fc2.bias).view(*hidden.shape[:-1], -1)


def _attention_npu(attn, hidden, att_cache, attn_mask, kv, *, pack_qkv: bool = True):
    batch, frames, _ = hidden.shape
    if pack_qkv and supports_packed_qkv(attn):
        weight, bias = packed_qkv(attn)
        qkv = F.linear(hidden.reshape(batch * frames, -1), weight, bias).view(batch, frames, -1)
    else:
        qkv = torch.cat([attn.to_q(hidden), attn.to_k(hidden), attn.to_v(hidden)], dim=-1)
    cached = int(att_cache.shape[2])
    logical_cache = cached if attn_mask is None else int(attn_mask.shape[-1]) - frames
    cache_capacity = int(kv.shape[2]) - frames
    if logical_cache < cached or logical_cache > cache_capacity:
        raise ValueError(f"CFM cache/mask mismatch: cache={cached}, logical={logical_cache}, capacity={cache_capacity}")
    kv = kv[:, :, : frames + logical_cache]
    behind = kv[:, :, frames:]
    if cached and (att_cache.data_ptr() != behind.data_ptr() or att_cache.stride() != behind.stride()):
        behind[:, :, :cached].copy_(att_cache)
    packed = qkv.view(batch, frames, 3, int(attn.num_heads), int(attn.head_dim))
    query = attn.q_norm(packed[:, :, 0].transpose(1, 2))
    current_keys = attn.k_norm(packed[:, :, 1].transpose(1, 2))
    current_values = packed[:, :, 2].transpose(1, 2)
    head_dim = int(query.shape[-1])
    if logical_cache > cached:
        behind[:, :, cached:].zero_()
    current = kv[:, :, :frames]
    current[..., :head_dim].copy_(current_keys)
    current[..., head_dim:].copy_(current_values)
    keys, values = kv[..., :head_dim], kv[..., head_dim:]
    mask = _expand_mask(attn_mask, frames, int(keys.shape[-2]))
    if mask is not None and int(mask.shape[-1]) != int(keys.shape[-2]):
        raise ValueError(
            f"CFM attention mask/cache mismatch: mask has {mask.shape[-1]} keys, cache has {keys.shape[-2]}"
        )
    out = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
    out = out.transpose(1, 2).reshape(batch * frames, -1)
    return F.linear(out, attn.proj.weight, attn.proj.bias).view(batch, frames, -1)


def _estimator_temb_configured_npu(
    est,
    x,
    mu,
    t,
    spks,
    cond,
    cnn_cache,
    att_cache,
    mask,
    tr,
    out=None,
    *,
    pack_qkv: bool,
    add_layer_norm: bool,
    im2col_conv: bool,
):
    frames = int(x.shape[2])
    if out is None:
        out = (
            torch.empty_like(x),
            torch.empty_like(cnn_cache),
            att_cache.new_empty(att_cache.shape[:3] + (frames + att_cache.shape[3], att_cache.shape[4])),
        )
    estimator_input = torch.cat((x, mu, spks.unsqueeze(-1).expand(-1, -1, frames), cond), dim=1)
    modulation = dit_modulation(est, t[:1])
    blocks = est.blocks
    depth = len(blocks)
    conv1, conv2 = blocks[0].conv.block[1], blocks[0].conv.block[6]
    width = int(conv1.kernel_size[0]) - 1
    split = conv1.in_channels
    hidden_stream = est.in_proj(estimator_input.transpose(1, 2))
    rows, _, channels = hidden_stream.shape
    old = _frames_major_cache(cnn_cache, width)
    staged = (
        width > 0
        and conv1.in_channels == conv2.in_channels
        and all(b.conv.block[1].kernel_size == b.conv.block[6].kernel_size == conv1.kernel_size for b in blocks)
    )
    staging1 = staging2 = None
    if staged:
        offsets = torch.arange(rows, device=hidden_stream.device, dtype=tr.dtype) * (width + frames)
        rows_index = ((offsets + tr)[:, None] + torch.arange(width, device=hidden_stream.device, dtype=tr.dtype)).reshape(-1)
    first = modulation[0]
    hidden = F.layer_norm(hidden_stream, (channels,), first[_SCALE_MSA], first[_SHIFT_MSA], blocks[0].norm1.eps)
    cache_index: dict[int, torch.Tensor] = {}
    for block_index, block in enumerate(blocks):
        mod = modulation[block_index]
        x_att = _attention_npu(
            block.attn,
            hidden,
            att_cache[block_index],
            mask,
            out[2][block_index],
            pack_qkv=pack_qkv,
        )
        conv_layers = block.conv.block
        conv1, conv_norm, conv2 = conv_layers[1], conv_layers[3], conv_layers[6]
        block_width = int(conv1.kernel_size[0]) - 1
        old1 = old[block_index][..., : conv1.in_channels]
        old2 = old[block_index][..., conv1.in_channels :]
        hidden_stream, conv_in = _add_layer_norm(
            hidden_stream,
            mod[_GATE_MSA] * x_att,
            mod[_SCALE_CONV],
            mod[_SHIFT_CONV],
            block.norm3.eps,
            fused=add_layer_norm,
        )
        history = _attach_history(old1, conv_in, block_width)
        conv_hidden = _causal_conv(conv1, history, frames, im2col=im2col_conv)
        conv_weight, conv_bias = _norm_affine(conv_norm, conv_hidden)
        conv_hidden = F.mish(F.layer_norm(conv_hidden, (int(conv_hidden.shape[-1]),), conv_weight, conv_bias, conv_norm.eps))
        history2 = _attach_history(old2, conv_hidden, block_width)
        hidden_stream, mlp_in = _add_layer_norm(
            hidden_stream,
            mod[_GATE_CONV] * _causal_conv(conv2, history2, frames, im2col=im2col_conv),
            mod[_SCALE_MLP],
            mod[_SHIFT_MLP],
            block.norm2.eps,
            fused=add_layer_norm,
        )
        last = block_index + 1 == depth
        hidden_stream, hidden = _add_layer_norm(
            hidden_stream,
            mod[_GATE_MLP] * _mlp(block.mlp, mlp_in),
            modulation[block_index + 1][_SCALE_MSA],
            modulation[block_index + 1][_SHIFT_MSA],
            est.final_layer.norm_final.eps if last else blocks[block_index + 1].norm1.eps,
            fused=add_layer_norm,
        )
        if staged:
            # Under FP16 autocast the first causal history remains Half while
            # LayerNorm may promote the second to Float.  index_select(out=)
            # requires an exact dtype match, so keep two typed staging arenas
            # and cast only at the request-owned cache boundary.
            if staging1 is None:
                staging1 = history.new_empty((depth, rows * width, split))
                staging2 = history2.new_empty((depth, rows * width, split))
            torch.index_select(history.view(-1, split), 0, rows_index, out=staging1[block_index])
            torch.index_select(history2.view(-1, split), 0, rows_index, out=staging2[block_index])
        else:
            if block_width not in cache_index:
                cache_index[block_width] = tr[:, None] + torch.arange(block_width, device=tr.device)[None, :]
            idx = cache_index[block_width]
            out[1][block_index, :, : conv1.in_channels].copy_(
                history.gather(1, idx[:, :, None].expand(-1, -1, history.shape[2])).transpose(1, 2)
            )
            out[1][block_index, :, conv1.in_channels :].copy_(
                history2.gather(1, idx[:, :, None].expand(-1, -1, history2.shape[2])).transpose(1, 2)
            )
    if staged:
        cache = out[1].unflatten(2, (2, split))
        cache[:, :, 0].copy_(staging1.view(depth, rows, width, split).permute(0, 1, 3, 2))
        cache[:, :, 1].copy_(staging2.view(depth, rows, width, split).permute(0, 1, 3, 2))
    out[0].copy_(est.final_layer.linear(hidden).transpose(1, 2))
    return out


def estimator_temb_schedule_npu(est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out=None):
    """Fused schedule with native QKV, LayerNorm, and Conv1d operators."""
    return _estimator_temb_configured_npu(
        est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out,
        pack_qkv=False, add_layer_norm=False, im2col_conv=False,
    )


def estimator_temb_qkv_npu(est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out=None):
    return _estimator_temb_configured_npu(
        est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out,
        pack_qkv=True, add_layer_norm=False, im2col_conv=False,
    )


def estimator_temb_hybrid_npu(est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out=None):
    """Use packed QKV below the unstable C8 CFG bucket.

    The official C8 workload reaches 16 CFG rows.  Sustained graph replay of
    the packed projection on that bucket triggered Ascend OpExecCache error
    507000, while the native three-projection schedule is numerically exact
    and had already passed the component screen.  Shape is static at graph
    capture, so this branch is resolved once per graph rather than per replay.
    """
    return _estimator_temb_configured_npu(
        est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out,
        pack_qkv=int(x.shape[0]) < 16, add_layer_norm=False, im2col_conv=False,
    )


def estimator_temb_addln_npu(est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out=None):
    return _estimator_temb_configured_npu(
        est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out,
        pack_qkv=False, add_layer_norm=True, im2col_conv=False,
    )


def estimator_temb_conv_npu(est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out=None):
    return _estimator_temb_configured_npu(
        est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out,
        pack_qkv=False, add_layer_norm=False, im2col_conv=True,
    )


def estimator_temb_fused_npu(est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out=None):
    """Drop-in replacement for ``BatchedToken2Wav._estimator_temb``."""
    return _estimator_temb_configured_npu(
        est, x, mu, t, spks, cond, cnn_cache, att_cache, mask, tr, out,
        pack_qkv=True, add_layer_norm=True, im2col_conv=True,
    )
