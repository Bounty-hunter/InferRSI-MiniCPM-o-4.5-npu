"""Talker codec-token sampler in the HF reference order, batched and device-resident.

Reference: ``MiniCPMTTS.generate`` / ``TTSStreamingGenerator`` in the MiniCPM-o 4.5
``modeling_minicpmo.py`` / ``utils.py`` (``gen_logits``): per step

1. ``logits / temperature``;
2. frames after the first: the repetition penalty over the last 16 generated
   frames (``CustomRepetitionPenaltyLogitsProcessorRepeat``: a token seen ``n``
   times is multiplied by ``penalty**n`` when negative, divided otherwise);
3. frames after the first: top-p (``TopPLogitsWarper``, ``min_tokens_to_keep=3``),
   then top-k (``TopKLogitsWarper``, ``k = max(top_k, 3)``);
4. EOS masked while ``t < min_new_tokens``; EOS forced at ``t == max_new_tokens - 1``;
5. softmax and one draw.

The default RNG mode is ``auto``: HF semantics use a counter-based uniform
keyed by (request seed, frame index), while Omni semantics preserve vLLM's
native per-request ``torch.multinomial`` generators.  ``counter`` can be
selected explicitly for Omni semantics to keep the same categorical
distribution while making the draw batchable.  Rows that are inactive
(padding, cancelled, finished) return ``PAD`` and never touch another row's
state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PAD = -1
M32 = 0xFFFFFFFF


@dataclass(frozen=True)
class TalkerSampling:
    temperature: float = 0.8
    top_p: float = 0.85
    top_k: int = 25
    repetition_penalty: float = 1.05
    window: int = 16
    min_new_tokens: int = 50
    max_new_tokens: int = 2048
    min_keep: int = 3
    # ``hf`` is MiniCPM-o's transformers streaming_generate contract.  The
    # vLLM-Omni baseline uses the vLLM sampler instead: top-k before top-p,
    # keeps at least one candidate, and applies the warpers to frame zero too.
    semantics: str = "hf"
    # ``auto`` preserves the historical contract: native per-request
    # multinomial generators for Omni, counter RNG for HF.  The explicit
    # counter mode is distribution-equivalent but not token-identical.
    rng: str = "auto"

    @property
    def uses_native_rng(self) -> bool:
        if self.rng not in ("auto", "native", "counter"):
            raise ValueError(f"unknown Talker RNG mode: {self.rng!r}")
        return self.rng == "native" or (self.rng == "auto" and self.semantics == "omni")


def _mix32(x: torch.Tensor) -> torch.Tensor:
    """32-bit integer finalizer (lowbias32) on int64 lanes holding values < 2**32."""
    x = x & M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & M32  # wraps modulo 2**64; the low 32 bits are exact
    return x ^ (x >> 16)


def uniform(seeds: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """U[0, 1) per row from (seed, frame index) only."""
    s = seeds.to(torch.int64)
    h = _mix32(s & M32)
    h = _mix32(h ^ ((s >> 32) & M32) ^ 0x9E3779B9)
    h = _mix32(h ^ (steps.to(torch.int64) & M32) ^ 0x632BE5AB)
    return (h >> 8).to(torch.float32) * (1.0 / (1 << 24))


def repetition_penalty(x: torch.Tensor, history: torch.Tensor, penalty: float) -> torch.Tensor:
    """``history``: [B, W] previous frames (most recent last), ``PAD`` where empty."""
    valid = history >= 0
    freq = torch.zeros_like(x).scatter_add_(1, history.clamp(min=0), valid.to(x.dtype))
    alpha = torch.pow(torch.full_like(x, penalty), freq)
    return torch.where(x < 0, x * alpha, x / alpha)


def top_p_filter(x: torch.Tensor, top_p: float, min_keep: int) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(x, descending=False, dim=-1)
    cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cum <= (1 - top_p)
    remove[..., -min_keep:] = False
    return x.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))


def top_k_filter(x: torch.Tensor, top_k: int, min_keep: int) -> torch.Tensor:
    k = min(max(top_k, min_keep), x.size(-1))
    kth = torch.topk(x, k, dim=-1).values[..., -1:]
    return x.masked_fill(x < kth, float("-inf"))


def top_p_top_k_filter(x: torch.Tensor, top_p: float, top_k: int, min_keep: int) -> torch.Tensor:
    """``top_k_filter(top_p_filter(x))`` without sorting the vocabulary.

    Top-p keeps a prefix of the descending order: rank ``j`` survives when the mass ranked above it is below ``top_p``
    (equivalently its ascending cumulative mass exceeds ``1 - top_p``), plus ``min_keep``; top-k then keeps the
    ``max(top_k, min_keep)`` largest of those.  Both are a value threshold at the last kept rank, which lies within the
    top ``k`` ranks, so ``topk`` and the full-vocabulary normaliser suffice.  The masses are summed over the top ranks
    instead of the tail, so a token within float rounding of the top-p boundary can fall the other way.
    """
    k = min(max(top_k, min_keep), x.size(-1))
    vals = torch.topk(x, k, dim=-1).values  # descending
    p = torch.exp(vals - torch.logsumexp(x, dim=-1, keepdim=True))
    above = p.cumsum(dim=-1) - p
    kept = (above < top_p).sum(dim=-1).clamp(min=min(min_keep, k), max=k)
    thr = vals.gather(1, (kept - 1)[:, None])
    return x.masked_fill(x < thr, float("-inf"))


def top_k_top_p_filter(x: torch.Tensor, top_k: int, top_p: float, min_keep: int = 1) -> torch.Tensor:
    """vLLM order: top-k, then top-p over the top-k-filtered distribution.

    This is the support-equivalent fused form of vLLM's two filters for the
    normal (untied-logit) case.  It avoids a full-vocabulary sort: after top-k,
    all probability mass is in the descending ``vals`` vector, so the last
    retained rank is obtained from its exclusive cumulative probability.
    """
    k = min(max(int(top_k), int(min_keep)), x.size(-1))
    vals = torch.topk(x, k, dim=-1).values
    p = torch.softmax(vals, dim=-1)
    above = p.cumsum(dim=-1) - p
    kept = (above < top_p).sum(dim=-1).clamp(min=min(min_keep, k), max=k)
    thr = vals.gather(1, (kept - 1)[:, None])
    return x.masked_fill(x < thr, float("-inf"))


def processed_logits(logits: torch.Tensor, history: torch.Tensor, steps: torch.Tensor, eos: int, cfg: TalkerSampling) -> torch.Tensor:
    """Prepare logits under the selected HF or vLLM-Omni sampling contract."""
    if cfg.semantics not in ("hf", "omni"):
        raise ValueError(f"unknown Talker sampling semantics: {cfg.semantics!r}")
    x = logits.float()
    col = torch.arange(x.size(-1), device=x.device)[None, :] == eos
    if cfg.semantics == "omni":
        # vLLM censors stop tokens before penalties/temperature, then applies
        # top-k -> top-p to every codec step (including frame zero).
        x = x.masked_fill(col & (steps < cfg.min_new_tokens)[:, None], float("-inf"))
        x = repetition_penalty(x, history, cfg.repetition_penalty) / cfg.temperature
        x = top_k_top_p_filter(x, cfg.top_k, cfg.top_p, min_keep=1)
    else:
        x = x / cfg.temperature
        later = steps > 0
        # Keep the branch device-resident.  Reading ``later.any()`` on the host
        # drains the launch queue once per autoregressive frame; step-zero rows
        # select the untouched logits exactly as before.
        y = top_p_top_k_filter(repetition_penalty(x, history, cfg.repetition_penalty), cfg.top_p, cfg.top_k, cfg.min_keep)
        x = torch.where(later[:, None], y, x)
        x = x.masked_fill(col & (steps < cfg.min_new_tokens)[:, None], float("-inf"))
    forced = (steps >= cfg.max_new_tokens - 1)[:, None]
    return torch.where(forced & ~col, torch.full_like(x, float("-inf")), torch.where(forced & col, torch.zeros_like(x), x))


def draw(
    x: torch.Tensor,
    seeds: torch.Tensor,
    steps: torch.Tensor,
    *,
    uniform_values: torch.Tensor | None = None,
) -> torch.Tensor:
    probs = torch.softmax(x, dim=-1)
    cdf = probs.cumsum(dim=-1)
    values = uniform(seeds, steps) if uniform_values is None else uniform_values
    u = values.to(cdf.dtype)[:, None] * cdf[:, -1:]
    return torch.searchsorted(cdf, u, right=True).squeeze(-1).clamp_(max=x.size(-1) - 1)


def draw_multinomial(x: torch.Tensor, generators: list[torch.Generator]) -> torch.Tensor:
    """One native ``torch.multinomial`` draw from each row's generator.

    This is the exact vLLM-Omni RNG contract.  The row loop is intentional:
    PyTorch accepts one generator per call, while Omni owns one independently
    seeded generator per request.  Callers must pass only live rows so a
    finished chunk does not consume another request-local random value.
    """
    if len(generators) != x.size(0):
        raise ValueError(f"expected {x.size(0)} generators, got {len(generators)}")
    probs = torch.softmax(x, dim=-1)
    return torch.stack(
        [torch.multinomial(probs[i], 1, generator=generators[i]).reshape(()) for i in range(x.size(0))]
    )


def restore_multinomial_states_after_block(
    generators: list[torch.Generator],
    states_after_step: list[list[torch.Tensor]],
    host_tokens: torch.Tensor,
) -> None:
    """Undo speculative RNG draws after a row's first PAD in a K block.

    The device block keeps decoding masked rows after codec EOS so all rows
    retain one static shape.  Native per-request generators must nevertheless
    advance only through the EOS draw, exactly like vLLM-Omni.  Each saved
    state is the post-draw state for one block step; after the single K-token
    host copy this restores a row to its last non-PAD draw.
    """
    if not states_after_step:
        return
    if len(generators) != host_tokens.size(1):
        raise ValueError("generator/token row mismatch")
    for row, generator in enumerate(generators):
        pads = (host_tokens[:, row] == PAD).nonzero()
        draws = int(pads[0]) if pads.numel() else host_tokens.size(0)
        if 0 < draws < len(states_after_step):
            generator.set_state(states_after_step[draws - 1][row])


@dataclass
class DecodeState:
    """Per-row Talker sampling state: frame index, last ``window`` frames, finished flag."""

    seeds: torch.Tensor  # [B] int64
    steps: torch.Tensor  # [B] int64, next frame index
    history: torch.Tensor  # [B, W] int64, PAD where empty
    finished: torch.Tensor  # [B] bool
    active: torch.Tensor  # [B] bool: False for padding or cancelled rows
    generators: list[torch.Generator] | None = None

    @classmethod
    def new(cls, seeds: list[int], cfg: TalkerSampling = TalkerSampling(), device="cpu") -> "DecodeState":
        b = len(seeds)
        return cls(seeds=torch.tensor(seeds, dtype=torch.int64, device=device), steps=torch.zeros(b, dtype=torch.int64, device=device),
                   history=torch.full((b, cfg.window), PAD, dtype=torch.int64, device=device),
                   finished=torch.zeros(b, dtype=torch.bool, device=device), active=torch.ones(b, dtype=torch.bool, device=device))


def step(state: DecodeState, logits: torch.Tensor, eos: int, cfg: TalkerSampling = TalkerSampling()) -> torch.Tensor:
    """Sample one frame per row and advance the state; inactive or finished rows return ``PAD`` and do not change."""
    live = state.active & ~state.finished
    prepared = processed_logits(logits, state.history, state.steps, eos, cfg)
    if cfg.uses_native_rng:
        if state.generators is None:
            raise ValueError("native Talker RNG requires one generator per row")
        tok = draw_multinomial(prepared, state.generators)
    else:
        tok = draw(prepared, state.seeds, state.steps)
    tok = torch.where(live, tok, torch.full_like(tok, PAD))
    shifted = torch.cat([state.history[:, 1:], tok[:, None]], dim=1)
    state.history = torch.where(live[:, None], shifted, state.history)
    state.steps = state.steps + live.to(torch.int64)
    state.finished = state.finished | (live & (tok == eos))
    return tok


def decode_k(state: DecodeState, logits_k: torch.Tensor, eos: int, cfg: TalkerSampling = TalkerSampling()) -> torch.Tensor:
    """K frames from precomputed per-frame logits [K, B, V]: identical to K calls of ``step``; an EOS freezes its row."""
    return torch.stack([step(state, logits_k[i], eos, cfg) for i in range(logits_k.size(0))])
