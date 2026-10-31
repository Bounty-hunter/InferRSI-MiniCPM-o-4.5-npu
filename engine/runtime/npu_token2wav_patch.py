"""NPU compatibility patches for the upstream MiniCPM Token2Wav modules."""

from __future__ import annotations

import contextlib
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F


@contextlib.contextmanager
def redirect_cuda_to(device: str | torch.device):
    """Narrow construction-time redirect for an upstream package that calls ``.cuda()``."""
    target = torch.device(device)
    if target.type == "cuda":
        yield
        return
    tensor_cuda, module_cuda = torch.Tensor.cuda, torch.nn.Module.cuda

    def tensor_to(self, *args, **kwargs):
        kwargs.pop("device", None)
        return self.to(target, **kwargs)

    def module_to(self, *args, **kwargs):
        kwargs.pop("device", None)
        return self.to(target, **kwargs)

    torch.Tensor.cuda, torch.nn.Module.cuda = tensor_to, module_to
    try:
        yield
    finally:
        torch.Tensor.cuda, torch.nn.Module.cuda = tensor_cuda, module_cuda


def _linear_downsample_even_scale(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale <= 0 or scale % 2 or x.shape[-1] % scale:
        raise ValueError("NPU midpoint downsample requires an even integer scale dividing the input")
    return (x[..., scale // 2 - 1 :: scale] + x[..., scale // 2 :: scale]) * 0.5


def _f02sine_npu(self, f0_values: torch.Tensor) -> torch.Tensor:
    scale = int(self.upsample_scale)
    if self.flag_for_pulse or scale != self.upsample_scale or scale % 2 or f0_values.shape[1] % scale:
        return self._minicpmo_original_f02sine(f0_values.cpu()).to(f0_values.device)
    rad = (f0_values / self.sampling_rate) % 1
    initial = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
    initial[:, 0] = 0
    rad[:, 0, :] += initial
    rad = _linear_downsample_even_scale(rad.transpose(1, 2), scale).transpose(1, 2)
    phase = torch.cumsum(rad, dim=1) * 2 * np.pi
    phase = F.interpolate(phase.transpose(1, 2) * scale, scale_factor=scale, mode="linear").transpose(1, 2)
    return torch.sin(phase)


def patch_hift(hift: torch.nn.Module) -> None:
    if getattr(hift, "_minicpmo_npu_patched", False):
        return
    sine = hift.m_source.l_sin_gen
    sine._minicpmo_original_f02sine = sine._f02sine
    sine._f02sine = MethodType(_f02sine_npu, sine)
    hift._minicpmo_npu_patched = True


def _expand_mask(mask: torch.Tensor | None, q_len: int, kv_len: int | None = None):
    if mask is None:
        return None
    kv_len = q_len if kv_len is None else kv_len
    if mask.dim() == 3 and mask.shape[-2] == 1:
        mask = mask.expand(-1, q_len, -1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() == 4 and mask.shape[-2] == 1 and kv_len > 1:
        mask = mask.expand(-1, -1, q_len, -1)
    return mask


def _attention_forward(self, x, attn_mask):
    batch, frames, _ = x.shape
    q, k, v = self.q_norm(self.to_heads(self.to_q(x))), self.k_norm(self.to_heads(self.to_k(x))), self.to_heads(self.to_v(x))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=_expand_mask(attn_mask, frames),
                                         dropout_p=self.attn_drop.p if self.training else 0.0)
    return self.proj_drop(self.proj(out.transpose(1, 2).reshape(batch, frames, -1)))


def _attention_forward_chunk(self, x, att_cache=None, attn_mask=None):
    batch, frames, _ = x.shape
    q, k, v = self.q_norm(self.to_heads(self.to_q(x))), self.k_norm(self.to_heads(self.to_k(x))), self.to_heads(self.to_v(x))
    if att_cache is not None:
        old_k, old_v = att_cache.chunk(2, dim=3)
        k, v = torch.cat((k, old_k), 2), torch.cat((v, old_v), 2)
    cache = torch.cat((k, v), 3)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=_expand_mask(attn_mask, frames, k.shape[2]))
    return self.proj_drop(self.proj(out.transpose(1, 2).reshape(batch, frames, -1))), cache


def patch_cosyvoice_attention() -> None:
    try:
        from cosyvoice2.flow import decoder_dit
        from cosyvoice2.transformer import upsample_encoder_v2
    except ImportError:
        return
    cls = getattr(decoder_dit, "Attention", None)
    if cls is not None and not getattr(cls, "_minicpmo_npu_patched", False):
        cls.forward, cls.forward_chunk = _attention_forward, _attention_forward_chunk
        cls._minicpmo_npu_patched = True
    enc = getattr(upsample_encoder_v2, "UpsampleConformerEncoderV2", None)
    if enc is not None:
        fn = enc.forward_chunk
        original = getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "__wrapped__", fn)
        enc.forward_chunk = torch._dynamo.disable(original)


@contextlib.contextmanager
def math_sdpa():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        yield
        return
    with sdpa_kernel(SDPBackend.MATH):
        yield
