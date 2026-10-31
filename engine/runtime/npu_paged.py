"""Ascend paged-KV decoder used by the MiniCPM-o dedicated engine.

The cache layout matches ``torch_npu``'s paged-attention operators.  New K/V
tokens are written with ``npu_scatter_pa_kv_cache``.  Single-token batched
decode uses ``_npu_paged_attention`` without expanding GQA heads or gathering
the cache.  Prefill uses the fused inference-attention operator; the uncommon
"append several tokens to an existing sequence" path deliberately keeps an
eager per-row fallback so correctness does not depend on a graph shape.

This module imports ``torch_npu`` lazily.  Page accounting and layout tests can
therefore run on a CPU development machine.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from engine.runtime.fp8 import linear
from engine.runtime.paged import DecoderSpec, fuse_weights, hf_rms_norm, hf_rope, hf_rope_tables


_GRAPH_CAPTURE_ATTEMPTS = 6
# torch-npu PA returns its maximum workspace at this sequence length.  This is
# the same capture-time sentinel used by vLLM-Ascend for graph replay.
_PA_MAX_WORKSPACE_SEQ_LEN = 6144
_FIA_PRE_TOKENS = 2_147_483_647
_DECODER_OP_MODES = frozenset({"reference", "rms", "add-rms", "swiglu", "full"})


def normalize_decoder_op_mode(fused_ops: bool, fused_mode: str | None = None) -> str:
    """Resolve the decoder math path while preserving the legacy bool API.

    The named modes are intentionally single-factor apart from ``full`` so
    Ascend kernel parity can be qualified without disabling the decoder graph.
    Production callers that only pass ``fused_ops`` retain their old behavior.
    """
    mode = ("full" if fused_ops else "reference") if fused_mode is None else str(fused_mode)
    if mode not in _DECODER_OP_MODES:
        choices = ", ".join(sorted(_DECODER_OP_MODES))
        raise ValueError(f"unsupported decoder op mode {mode!r}; choose one of: {choices}")
    return mode


def _is_transient_graph_update_error(exc: RuntimeError) -> bool:
    """Return whether CANN rejected an otherwise valid first task update.

    CANN 9.1 can sporadically reject ``graph_task_update_end`` with 507000
    while constructing a fresh graph.  A new graph/update stream succeeds and
    the eager path remains usable, so retry only this narrow first-capture
    failure.  Replay failures and unrelated operator errors must still disable
    the ring and take the correctness fallback.
    """
    message = str(exc)
    return "graph_task_update_end" in message and "507000" in message


def decode_bucket_size(active: int, maximum: int = 32) -> int:
    """Smallest power-of-two graph bucket covering ``active`` rows."""
    if active < 1 or maximum < 1 or maximum & (maximum - 1) or active > maximum:
        raise ValueError("active must fit a positive power-of-two maximum")
    return 1 << (active - 1).bit_length()


@dataclass
class NpuBatchLayout:
    seqs: list[int]
    new_tokens: list[int]
    starts: list[int]
    lengths: list[int]
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    context_lens: torch.Tensor
    qo_indptr: torch.Tensor

    @property
    def last_token_index(self) -> torch.Tensor:
        return self.qo_indptr[1:].long() - 1

    @property
    def is_decode(self) -> bool:
        return all(n == 1 for n in self.new_tokens)

    @property
    def is_fresh_prefill(self) -> bool:
        return all(start == 0 for start in self.starts)


class NpuPagedKV:
    """One paged K/V pool shared by every live sequence of one decoder."""

    def __init__(
        self,
        spec: DecoderSpec,
        num_pages: int,
        page_size: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "npu:0",
    ):
        if num_pages < 1 or page_size < 1:
            raise ValueError("num_pages and page_size must be positive")
        self.spec, self.page_size, self.num_pages = spec, int(page_size), int(num_pages)
        shape = (spec.num_layers, num_pages, page_size, spec.num_kv_heads, spec.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.free: list[int] = list(range(num_pages - 1, -1, -1))
        self.pages: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}

    def pages_needed(self, seq: int, extra: int) -> int:
        if extra < 0:
            raise ValueError("extra tokens cannot be negative")
        have = len(self.pages.get(seq, ()))
        need = -(-(self.lengths.get(seq, 0) + extra) // self.page_size)
        return max(0, need - have)

    def can_fit(self, demands: dict[int, int]) -> bool:
        return sum(self.pages_needed(seq, extra) for seq, extra in demands.items()) <= len(self.free)

    def reserve(self, seq: int, extra: int) -> None:
        self.pages.setdefault(seq, [])
        self.lengths.setdefault(seq, 0)
        needed = self.pages_needed(seq, extra)
        if needed > len(self.free):
            raise MemoryError(f"KV pool exhausted: need {needed} pages, {len(self.free)} free")
        self.pages[seq].extend(self.free.pop() for _ in range(needed))

    def truncate(self, seq: int, length: int) -> None:
        if length < 0 or length > self.lengths.get(seq, 0):
            raise ValueError(f"cannot truncate seq {seq} to {length}")
        self.lengths[seq] = length

    def release(self, seq: int) -> None:
        self.free.extend(reversed(self.pages.pop(seq, ())))
        self.lengths.pop(seq, None)

    @property
    def used_pages(self) -> int:
        return self.num_pages - len(self.free)

    def layout(
        self,
        seqs: Iterable[int],
        new_tokens: Iterable[int],
        device: str | torch.device | None = None,
    ) -> NpuBatchLayout:
        seqs, new_tokens = list(seqs), list(new_tokens)
        if not seqs or len(seqs) != len(new_tokens) or any(n < 1 for n in new_tokens) or len(set(seqs)) != len(seqs):
            raise ValueError("layout needs unique sequences and one positive token count per sequence")
        for seq, count in zip(seqs, new_tokens):
            self.reserve(seq, count)
        starts = [self.lengths[seq] for seq in seqs]
        lengths = [start + count for start, count in zip(starts, new_tokens)]
        page_counts = [-(-length // self.page_size) for length in lengths]
        width = max(page_counts)
        blocks = [[*self.pages[seq][:count], *([-1] * (width - count))] for seq, count in zip(seqs, page_counts)]
        positions = [pos for start, length in zip(starts, lengths) for pos in range(start, length)]
        slots = [self.pages[seq][pos // self.page_size] * self.page_size + pos % self.page_size
                 for seq, start, length in zip(seqs, starts, lengths) for pos in range(start, length)]
        qo = [0]
        for count in new_tokens:
            qo.append(qo[-1] + count)
        for seq, length in zip(seqs, lengths):
            self.lengths[seq] = length
        target = self.k.device if device is None else torch.device(device)

        def tensor(data, dtype=torch.int32):
            return torch.tensor(data, dtype=dtype).to(target, non_blocking=True)

        return NpuBatchLayout(
            seqs=seqs,
            new_tokens=new_tokens,
            starts=starts,
            lengths=lengths,
            positions=tensor(positions, torch.int64),
            slot_mapping=tensor(slots, torch.int32),
            block_table=tensor(blocks, torch.int32),
            # torch-npu's 910B1 paged-attention API consumes context lengths on
            # the host (the vLLM-Ascend metadata builder deliberately keeps
            # this tensor on CPU); block tables and slot mappings stay on NPU.
            context_lens=torch.tensor(lengths, dtype=torch.int32),
            qo_indptr=tensor(qo, torch.int32),
        )

    def dense_layer(self, layer: int, seq: int, length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather one sequence for the append-prefill correctness fallback."""
        length = self.lengths[seq] if length is None else length
        count = -(-length // self.page_size)
        ids = torch.tensor(self.pages[seq][:count], dtype=torch.long, device=self.k.device)
        keys = self.k[layer].index_select(0, ids).reshape(-1, self.spec.num_kv_heads, self.spec.head_dim)[:length]
        vals = self.v[layer].index_select(0, ids).reshape(-1, self.spec.num_kv_heads, self.spec.head_dim)[:length]
        return keys, vals


@dataclass
class _PagedGraphOp:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    output: torch.Tensor
    handle: object
    event: object


class _PagedGraphCapture:
    """Record PA task handles so host context lengths can change on replay."""

    def __init__(self, runner: "NpuDecoderRunner", workspace: torch.Tensor):
        self.runner = runner
        self.workspace = workspace
        self.ops: list[_PagedGraphOp] = []

    def attention(self, query: torch.Tensor, layer: int, layout: NpuBatchLayout) -> torch.Tensor:
        runner = self.runner
        output = torch.empty_like(query)
        kwargs = {
            "query": query.contiguous(),
            "key_cache": runner.kv.k[layer],
            "value_cache": runner.kv.v[layer],
            "num_kv_heads": runner.spec.num_kv_heads,
            "num_heads": runner.spec.num_heads,
            "scale_value": runner.spec.head_dim ** -0.5,
            "block_table": layout.block_table.contiguous(),
            "context_lens": layout.context_lens.contiguous(),
            "out": output,
        }
        stream = torch.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        runner.npu._npu_paged_attention(**kwargs, workspace=self.workspace)
        handle = torch.npu.graph_task_group_end(stream)
        self.ops.append(
            _PagedGraphOp(
                query=kwargs["query"],
                key_cache=kwargs["key_cache"],
                value_cache=kwargs["value_cache"],
                output=output,
                handle=handle,
                event=event,
            )
        )
        return output


class _StaticFiaGraphCapture:
    """Capture paged FIA with constant host args and a device-side live mask.

    ``actual_seq_lengths_kv`` is deliberately fixed to ``capacity``.  The
    current sequence lengths are copied into ``graph.seq_lens_device`` before
    replay, and the graph turns them into ``pse_shift`` once per replay.  No
    attention layer therefore needs a ``graph_task_update_begin/end`` rebind.

    The existing ``_PagedGraphCapture`` remains the correctness fallback for
    oversized contexts and for deployments that do not opt in.
    """

    def __init__(self, graph: "NpuDecodeGraph"):
        self.graph = graph
        self.runner = graph.runner
        self.calls = 0
        self.outputs: list[torch.Tensor] = []
        self.softmax_lse: list[torch.Tensor] = []

    def prepare(self) -> None:
        graph = self.graph
        lens = graph.seq_lens_device.clamp(min=1).to(torch.int32).view(-1, 1, 1, 1)
        torch.where(graph.pse_arange >= lens, graph.pse_negative, graph.pse_zero, out=graph.pse_shift)

    def attention(self, query: torch.Tensor, layer: int, layout: NpuBatchLayout) -> torch.Tensor:
        graph, runner = self.graph, self.runner
        output = torch.empty_like(query)
        lse = torch.empty(1, dtype=query.dtype, device=query.device)
        kwargs = graph._static_attention_kwargs(query.contiguous(), layer)
        runner.npu.npu_fused_infer_attention_score.out(
            workspace=graph.workspace,
            out=[output, lse],
            **kwargs,
        )
        self.calls += 1
        self.outputs.append(output)
        self.softmax_lse.append(lse)
        return output


class NpuDecodeGraph:
    """One fixed-batch full decoder graph.

    ``static_capacity`` selects constant-shape paged FIA.  ``None`` keeps the
    original task-update PA graph, which is also the lossless fallback.
    """

    def __init__(
        self,
        runner: "NpuDecoderRunner",
        batch: int,
        max_blocks: int,
        pool=None,
        static_capacity: int | None = None,
    ):
        self.runner, self.batch, self.max_blocks = runner, int(batch), int(max_blocks)
        self.pool = pool
        spec, device = runner.spec, runner.kv.k.device
        self.static_capacity = int(static_capacity) if static_capacity is not None else None
        if self.static_capacity is not None:
            if self.static_capacity < runner.kv.page_size:
                raise ValueError("static attention capacity must cover at least one KV page")
            if self.static_capacity % runner.kv.page_size:
                raise ValueError("static attention capacity must be a multiple of the KV page size")
            if self.static_capacity > max_blocks * runner.kv.page_size:
                raise ValueError("static attention capacity exceeds the graph block table")
        self.inputs = runner.kv.k.new_zeros((batch, spec.hidden_size))
        self.positions = torch.zeros(batch, dtype=torch.int64, device=device)
        self.slot_mapping = torch.zeros(batch, dtype=torch.int32, device=device)
        self.block_table = torch.full((batch, max_blocks), -1, dtype=torch.int32, device=device)
        self.context_lens = torch.ones(batch, dtype=torch.int32)
        self.seq_lens_device = torch.ones(batch, dtype=torch.int32, device=device)
        workspace_seq_len = min(_PA_MAX_WORKSPACE_SEQ_LEN, max_blocks * runner.kv.page_size)
        # ATB paged attention consumes context_lens on the host.  Passing this
        # sizing tensor on NPU makes OperationSetup fail before a workspace is
        # returned, even though the block table and KV cache remain on NPU.
        self.workspace_context_lens = torch.full((batch,), workspace_seq_len, dtype=torch.int32)
        # Workspace setup validates that the sentinel sequence length fits the
        # supplied page table.  Use valid synthetic page ids only for sizing;
        # task updates below continue to receive the real request table.
        workspace_pages = torch.arange(max_blocks, dtype=torch.int32, device=device)
        self.workspace_block_table = workspace_pages.remainder(runner.kv.num_pages).repeat(batch, 1)
        self.layout = NpuBatchLayout(
            seqs=list(range(batch)),
            new_tokens=[1] * batch,
            starts=[0] * batch,
            lengths=[1] * batch,
            positions=self.positions,
            slot_mapping=self.slot_mapping,
            block_table=self.block_table,
            context_lens=self.context_lens,
            qo_indptr=torch.arange(batch + 1, dtype=torch.int32, device=device),
        )
        # Allocate the maximum workspace before entering an NPU graph pool.
        # Every decoder layer has the same attention shape.
        workspace_query = runner.kv.k.new_empty((batch, spec.num_heads, spec.head_dim))
        workspace_output = torch.empty_like(workspace_query)
        self.pse_shift = None
        self.pse_arange = None
        self.pse_negative = None
        self.pse_zero = None
        if self.static_capacity is None:
            # Task-update PA uses a lane-private workspace because its update
            # stream can overlap another lane's replay.
            self.workspace = runner.npu._npu_paged_attention_get_workspace(
                query=workspace_query,
                key_cache=runner.kv.k[0],
                value_cache=runner.kv.v[0],
                num_kv_heads=spec.num_kv_heads,
                num_heads=spec.num_heads,
                scale_value=spec.head_dim ** -0.5,
                block_table=self.workspace_block_table,
                context_lens=self.workspace_context_lens,
                out=workspace_output,
            )
        else:
            capacity = self.static_capacity
            self.pse_shift = runner.kv.k.new_zeros((batch, spec.num_heads, 1, capacity))
            columns = torch.arange(capacity, dtype=torch.int32, device=device).view(1, 1, 1, capacity)
            self.pse_arange = columns.expand(batch, spec.num_heads, 1, capacity).contiguous()
            self.pse_negative = torch.tensor(float("-inf"), dtype=runner.kv.k.dtype, device=device)
            self.pse_zero = torch.tensor(0.0, dtype=runner.kv.k.dtype, device=device)
            workspace_kwargs = self._static_attention_kwargs(workspace_query, 0)
            shape_key = (
                batch,
                spec.num_heads,
                spec.num_kv_heads,
                spec.head_dim,
                runner.kv.page_size,
                capacity,
                runner.kv.k.dtype,
            )
            self.workspace = runner.static_shape_workspace(
                shape_key,
                lambda: runner.npu._npu_fused_infer_attention_score_get_max_workspace(
                    **workspace_kwargs
                ),
            )
        self.graph = None
        self.output = None
        self.capture: _PagedGraphCapture | _StaticFiaGraphCapture | None = None
        self.update_stream = torch.npu.Stream(device=device)

    def _static_attention_kwargs(self, query: torch.Tensor, layer: int) -> dict:
        if self.static_capacity is None or self.pse_shift is None:
            raise RuntimeError("static attention is not configured for this graph")
        spec = self.runner.spec
        key = self.runner.kv.k[layer].view(self.runner.kv.num_pages, self.runner.kv.page_size, -1)
        value = self.runner.kv.v[layer].view(self.runner.kv.num_pages, self.runner.kv.page_size, -1)
        return {
            "query": query,
            "key": key,
            "value": value,
            "pse_shift": self.pse_shift,
            "atten_mask": None,
            "block_table": self.block_table,
            "input_layout": "TND",
            "block_size": self.runner.kv.page_size,
            "actual_seq_lengths": list(range(1, self.batch + 1)),
            "actual_seq_lengths_kv": [self.static_capacity] * self.batch,
            "num_key_value_heads": spec.num_kv_heads,
            "num_heads": spec.num_heads,
            "scale": spec.head_dim ** -0.5,
            "sparse_mode": 0,
            "pre_tokens": _FIA_PRE_TOKENS,
            "next_tokens": 0,
        }

    def _copy_inputs(self, x: torch.Tensor, layout: NpuBatchLayout) -> None:
        if x.shape != self.inputs.shape or layout.block_table.shape[1] > self.max_blocks:
            raise ValueError("decode graph input exceeds its static bucket")
        if self.static_capacity is not None and max(layout.lengths) > self.static_capacity:
            raise ValueError("decode graph input exceeds its static KV capacity")
        self.inputs.copy_(x)
        self.positions.copy_(layout.positions)
        self.slot_mapping.copy_(layout.slot_mapping)
        # Fixed-capacity FIA may issue masked loads for every declared block.
        # Give unused columns a real, zero-initialized cache page instead of
        # ``-1``; ``pse_shift`` makes those values mathematically invisible.
        self.block_table.fill_(0 if self.static_capacity is not None else -1)
        self.block_table[:, : layout.block_table.shape[1]].copy_(layout.block_table)
        if self.static_capacity is not None:
            self.block_table.clamp_min_(0)
        self.context_lens = layout.context_lens.contiguous()
        self.layout.context_lens = self.context_lens
        if self.static_capacity is not None:
            self.seq_lens_device.copy_(self.context_lens, non_blocking=True)

    def _capture(self) -> torch.Tensor:
        with self.runner.capture_context():
            return self._capture_exclusive()

    def _capture_exclusive(self) -> torch.Tensor:
        current = torch.npu.current_stream()
        stream = torch.npu.Stream(device=self.inputs.device)
        stream.wait_stream(current)
        with torch.npu.stream(stream):
            for _ in range(3):
                self.runner._forward_impl(self.inputs, self.layout)
        current.wait_stream(stream)
        current.synchronize()
        graph = torch.npu.NPUGraph()
        capture = (
            _StaticFiaGraphCapture(self)
            if self.static_capacity is not None
            else _PagedGraphCapture(self.runner, self.workspace)
        )
        self.runner._graph_capture = capture
        try:
            graph_kwargs = {"stream": stream}
            if self.pool is not None:
                graph_kwargs["pool"] = self.pool
            with torch.npu.graph(graph, **graph_kwargs):
                if isinstance(capture, _StaticFiaGraphCapture):
                    capture.prepare()
                output = self.runner._forward_impl(self.inputs, self.layout)
        finally:
            self.runner._graph_capture = None
        current.wait_stream(stream)
        current.synchronize()
        self.graph, self.output, self.capture = graph, output, capture
        captured_attention = capture.calls if isinstance(capture, _StaticFiaGraphCapture) else len(capture.ops)
        if not captured_attention:
            raise RuntimeError("decode graph captured no attention operations")
        # Task-group PA is rebound outside capture.  Static FIA has no update:
        # its captured mask refresh reads ``seq_lens_device`` during replay.
        if self.static_capacity is None:
            self._update_attention()
        self.graph.replay()
        return output

    def _attention_kwargs(self, op: _PagedGraphOp) -> dict:
        return {
            "query": op.query,
            "key_cache": op.key_cache,
            "value_cache": op.value_cache,
            "num_kv_heads": self.runner.spec.num_kv_heads,
            "num_heads": self.runner.spec.num_heads,
            "scale_value": self.runner.spec.head_dim ** -0.5,
            "block_table": self.block_table,
            "context_lens": self.context_lens,
            "out": op.output,
        }

    def _update_attention(self) -> None:
        assert isinstance(self.capture, _PagedGraphCapture)
        assert self.workspace is not None
        with torch.npu.stream(self.update_stream):
            for op in self.capture.ops:
                kwargs = self._attention_kwargs(op)
                torch.npu.graph_task_update_begin(self.update_stream, op.handle)
                # Reuse the lane-private capture-shape allocation: looking up
                # a workspace per layer/token is invariant host work, while a
                # graph-pool allocation can race another lane's replay.
                self.runner.npu._npu_paged_attention(**kwargs, workspace=self.workspace)
                torch.npu.graph_task_update_end(self.update_stream)
                op.event.record(self.update_stream)

    def run(self, x: torch.Tensor, layout: NpuBatchLayout, *, synchronize: bool = True) -> torch.Tensor:
        if self.graph is None:
            self._copy_inputs(x, layout)
            return self._capture()
        # Full graphs retain intermediate addresses.  Finish the previous
        # replay before rebinding PA task metadata, as vLLM-Ascend does.
        if synchronize and self.static_capacity is None:
            torch.npu.current_stream().synchronize()
        self._copy_inputs(x, layout)
        if self.static_capacity is None:
            self._update_attention()
        self.graph.replay()
        return self.output


class NpuDecoderRunner:
    """Fused MiniCPM decoder stack backed by Ascend paged K/V operators."""

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        spec: DecoderSpec,
        kv: NpuPagedKV,
        layers: list[dict] | None = None,
        fused_ops: bool = False,
        fused_mode: str | None = None,
        use_graphs: bool = False,
        graph_max_batch: int = 32,
        graph_max_blocks: int = 64,
        static_shape_attention: bool = False,
        static_shape_capacity: int = 512,
    ):
        try:
            import torch_npu
        except ImportError as exc:  # pragma: no cover - exercised on the Ascend runner
            raise RuntimeError("NpuDecoderRunner requires torch-npu") from exc
        self.npu = torch_npu
        self.spec, self.kv = spec, kv
        self.layers = layers if layers is not None else fuse_weights(weights, spec)
        self.final_norm = weights[f"{spec.prefix}norm.weight"]
        self.fused_mode = normalize_decoder_op_mode(fused_ops, fused_mode)
        self.fused_ops = self.fused_mode == "full"
        self.use_graphs = bool(use_graphs)
        self.graph_max_batch = int(graph_max_batch)
        self.graph_max_blocks = int(graph_max_blocks)
        self.static_shape_attention = bool(static_shape_attention)
        self.static_shape_capacity = int(static_shape_capacity)
        if self.static_shape_attention:
            if self.static_shape_capacity < kv.page_size or self.static_shape_capacity % kv.page_size:
                raise ValueError("static_shape_capacity must be a positive multiple of the KV page size")
            if self.static_shape_capacity > self.graph_max_blocks * kv.page_size:
                raise ValueError("static_shape_capacity exceeds graph_max_blocks")
        self.decode_graphs: dict[int, NpuDecodeGraph] = {}
        self.static_decode_graphs: dict[tuple[int, int], NpuDecodeGraph] = {}
        self.graph_failures: dict[int, str] = {}
        self.static_graph_failures: dict[tuple[int, int], str] = {}
        self.decode_graph_rings: dict[tuple[int, int, int], list[NpuDecodeGraph]] = {}
        self.decode_graph_ring_pools: dict[tuple[int, int, int], object] = {}
        self.graph_ring_failures: dict[tuple[int, int, int], str] = {}
        self.graph_ring_capture_retries: dict[tuple[int, int, int], int] = {}
        self.graph_ring_capture_retry_errors: dict[tuple[int, int, int], str] = {}
        self.static_shape_workspaces: dict[tuple, torch.Tensor] = {}
        self._graph_capture: _PagedGraphCapture | _StaticFiaGraphCapture | None = None
        self._graph_pad_seqs: list[int] = []
        self.capture_context = nullcontext

    def static_shape_workspace(self, key: tuple, factory) -> torch.Tensor:
        """Reuse one FIA workspace across layers and same-shape graph lanes."""
        workspace = self.static_shape_workspaces.get(key)
        if workspace is None:
            workspace = factory()
            self.static_shape_workspaces[key] = workspace
        return workspace

    def _static_capacity_for(self, layout: NpuBatchLayout) -> int | None:
        if self.static_shape_attention and max(layout.lengths) <= self.static_shape_capacity:
            return self.static_shape_capacity
        return None

    def decode_graph_bucket(self, batch: int) -> int:
        """Reuse the smallest already-captured graph that can hold ``batch``.

        Decode batches normally shrink as requests finish.  Capturing every
        smaller power-of-two shape retains another full decoder graph and can
        starve Token2Wav on a 64 GiB 910B1.  Padding into an existing larger
        graph is exact (the extra rows already use request-inert KV pages) and
        avoids both the capture latency and its persistent memory.
        """
        natural = decode_bucket_size(batch, self.graph_max_batch)
        compatible = [bucket for bucket in self.decode_graphs if bucket >= batch]
        return min(compatible, default=natural)

    def _padded_decode(self, x: torch.Tensor, layout: NpuBatchLayout, bucket: int) -> tuple[torch.Tensor, NpuBatchLayout]:
        """Pad a decode batch with request-inert rows for a graph bucket.

        Each dummy row owns one permanent KV page.  Its output is discarded;
        distinct pages avoid overlapping scatter writes inside the captured
        graph.  At most ``graph_max_batch / 2 - 1`` pages are needed because
        the bucket is the next power of two.
        """
        active = int(x.shape[0])
        missing = bucket - active
        while len(self._graph_pad_seqs) < missing:
            seq = -1 - len(self._graph_pad_seqs)
            self.kv.reserve(seq, 1)
            self.kv.lengths[seq] = 1
            self._graph_pad_seqs.append(seq)
        pads = self._graph_pad_seqs[:missing]
        if not pads:
            return x, layout
        device = x.device
        width = int(layout.block_table.shape[1])
        pages = [self.kv.pages[seq][0] for seq in pads]
        pad_blocks = torch.full((missing, width), -1, dtype=torch.int32, device=device)
        pad_blocks[:, 0] = torch.tensor(pages, dtype=torch.int32, device=device)
        padded = NpuBatchLayout(
            seqs=layout.seqs + pads,
            new_tokens=layout.new_tokens + [1] * missing,
            starts=layout.starts + [0] * missing,
            lengths=layout.lengths + [1] * missing,
            positions=torch.cat([layout.positions, torch.zeros(missing, dtype=torch.int64, device=device)]),
            slot_mapping=torch.cat([
                layout.slot_mapping,
                torch.tensor([page * self.kv.page_size for page in pages], dtype=torch.int32, device=device),
            ]),
            block_table=torch.cat([layout.block_table, pad_blocks]),
            context_lens=torch.cat([layout.context_lens, torch.ones(missing, dtype=torch.int32)]),
            qo_indptr=torch.arange(bucket + 1, dtype=torch.int32, device=device),
        )
        return torch.cat([x, x.new_zeros((missing, x.shape[1]))]), padded

    def _scatter(self, layer: int, key: torch.Tensor, value: torch.Tensor, layout: NpuBatchLayout) -> None:
        self.npu.npu_scatter_pa_kv_cache(
            key=key.contiguous(),
            value=value.contiguous(),
            key_cache=self.kv.k[layer],
            value_cache=self.kv.v[layer],
            slot_mapping=layout.slot_mapping.contiguous(),
            cache_mode="Norm",
        )

    def _paged_decode(self, query: torch.Tensor, layer: int, layout: NpuBatchLayout) -> torch.Tensor:
        if self._graph_capture is not None:
            return self._graph_capture.attention(query, layer, layout)
        out = torch.empty_like(query)
        self.npu._npu_paged_attention(
            query=query.contiguous(),
            key_cache=self.kv.k[layer],
            value_cache=self.kv.v[layer],
            num_kv_heads=self.spec.num_kv_heads,
            num_heads=self.spec.num_heads,
            scale_value=self.spec.head_dim ** -0.5,
            block_table=layout.block_table.contiguous(),
            context_lens=layout.context_lens.contiguous(),
            out=out,
        )
        return out

    def _fresh_prefill(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layout: NpuBatchLayout) -> torch.Tensor:
        cumulative = layout.qo_indptr[1:].to("cpu").tolist()
        width = max(2048, max(layout.new_tokens))
        causal_mask = torch.triu(torch.ones(width, width, device=query.device, dtype=torch.int8), diagonal=1)
        return self.npu.npu_fused_infer_attention_score(
            query=query.contiguous(),
            key=key.contiguous(),
            value=value.contiguous(),
            input_layout="TND",
            atten_mask=causal_mask,
            actual_seq_lengths=cumulative,
            actual_seq_lengths_kv=cumulative,
            num_key_value_heads=self.spec.num_kv_heads,
            num_heads=self.spec.num_heads,
            scale=self.spec.head_dim ** -0.5,
            sparse_mode=3,
        )[0]

    def _append_prefill(self, query: torch.Tensor, layer: int, layout: NpuBatchLayout) -> torch.Tensor:
        outputs = []
        offset = 0
        for seq, start, total, count in zip(layout.seqs, layout.starts, layout.lengths, layout.new_tokens):
            q = query[offset:offset + count].transpose(0, 1).unsqueeze(0)
            key, value = self.kv.dense_layer(layer, seq, total)
            key, value = key.transpose(0, 1).unsqueeze(0), value.transpose(0, 1).unsqueeze(0)
            qpos = torch.arange(start, total, device=query.device)[:, None]
            kpos = torch.arange(total, device=query.device)[None, :]
            mask = (kpos > qpos).unsqueeze(0).unsqueeze(0)
            result = self.npu.npu_fused_infer_attention_score(
                query=q.contiguous(), key=key.contiguous(), value=value.contiguous(),
                num_heads=self.spec.num_heads, num_key_value_heads=self.spec.num_kv_heads,
                input_layout="BNSD", atten_mask=mask, scale=self.spec.head_dim ** -0.5, sparse_mode=0,
            )[0]
            outputs.append(result.squeeze(0).transpose(0, 1))
            offset += count
        return torch.cat(outputs)

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int,
        layout: NpuBatchLayout,
    ) -> torch.Tensor:
        self._scatter(layer, key, value, layout)
        if layout.is_decode:
            return self._paged_decode(query, layer, layout)
        if layout.is_fresh_prefill:
            return self._fresh_prefill(query, key, value, layout)
        return self._append_prefill(query, layer, layout)

    def _forward_reference(self, x: torch.Tensor, layout: NpuBatchLayout) -> torch.Tensor:
        spec = self.spec
        nq, nk = spec.num_heads * spec.head_dim, spec.num_kv_heads * spec.head_dim
        tables = hf_rope_tables(layout.positions, spec.head_dim, spec.rope_theta, x.dtype)
        for layer, weights in enumerate(self.layers):
            hidden = hf_rms_norm(x, weights["ln1"], spec.rms_eps)
            qkv = linear(hidden, weights["qkv"])
            query = qkv[:, :nq].reshape(-1, spec.num_heads, spec.head_dim)
            key = qkv[:, nq:nq + nk].reshape(-1, spec.num_kv_heads, spec.head_dim)
            value = qkv[:, nq + nk:].reshape(-1, spec.num_kv_heads, spec.head_dim)
            if spec.qk_norm:
                query = hf_rms_norm(query.reshape(-1, spec.head_dim), weights["q_norm"], spec.rms_eps).view_as(query)
                key = hf_rms_norm(key.reshape(-1, spec.head_dim), weights["k_norm"], spec.rms_eps).view_as(key)
            query, key = hf_rope(query, key, layout.positions, spec.rope_theta, tables)
            attended = self._attention(query, key, value, layer, layout)
            x = x + linear(attended.reshape(-1, nq), weights["o"])
            hidden = hf_rms_norm(x, weights["ln2"], spec.rms_eps)
            gate, up = linear(hidden, weights["gate_up"]).chunk(2, dim=-1)
            x = x + linear(F.silu(gate) * up, weights["down"])
        return hf_rms_norm(x, self.final_norm, spec.rms_eps)

    def _forward_native(self, x: torch.Tensor, layout: NpuBatchLayout) -> torch.Tensor:
        """Run a qualified subset of Ascend-native decoder elementwise ops.

        Attention and linear projections are identical in every mode.  The
        single-factor modes make it possible to distinguish RMSNorm,
        Add+RMSNorm, and SwiGLU numerical drift while retaining NPUGraph.
        """
        spec = self.spec
        nq, nk = spec.num_heads * spec.head_dim, spec.num_kv_heads * spec.head_dim
        tables = hf_rope_tables(layout.positions, spec.head_dim, spec.rope_theta, x.dtype)
        native_rms = self.fused_mode in ("rms", "full")
        native_add_rms = self.fused_mode in ("add-rms", "full")
        native_swiglu = self.fused_mode in ("swiglu", "full")

        def rms_norm(tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            if native_rms:
                return self.npu.npu_rms_norm(tensor, weight, epsilon=spec.rms_eps)[0]
            return hf_rms_norm(tensor, weight, spec.rms_eps)

        hidden = rms_norm(x, self.layers[0]["ln1"])
        for layer, weights in enumerate(self.layers):
            qkv = linear(hidden, weights["qkv"])
            query = qkv[:, :nq].reshape(-1, spec.num_heads, spec.head_dim)
            key = qkv[:, nq:nq + nk].reshape(-1, spec.num_kv_heads, spec.head_dim)
            value = qkv[:, nq + nk:].reshape(-1, spec.num_kv_heads, spec.head_dim)
            if spec.qk_norm:
                query = rms_norm(query, weights["q_norm"])
                key = rms_norm(key, weights["k_norm"])
            query, key = hf_rope(query, key, layout.positions, spec.rope_theta, tables)
            attended = self._attention(query, key, value, layer, layout)
            attention_branch = linear(attended.reshape(-1, nq), weights["o"])
            if native_add_rms:
                hidden, _, x = self.npu.npu_add_rms_norm(
                    attention_branch,
                    x,
                    weights["ln2"],
                    epsilon=spec.rms_eps,
                )
            else:
                x = x + attention_branch
                hidden = rms_norm(x, weights["ln2"])
            gate_up = linear(hidden, weights["gate_up"])
            if native_swiglu:
                activated = self.npu.npu_swiglu(gate_up, dim=-1)
            else:
                gate, up = gate_up.chunk(2, dim=-1)
                activated = F.silu(gate) * up
            mlp_branch = linear(activated, weights["down"])
            next_norm = self.final_norm if layer + 1 == len(self.layers) else self.layers[layer + 1]["ln1"]
            if native_add_rms:
                hidden, _, x = self.npu.npu_add_rms_norm(
                    mlp_branch,
                    x,
                    next_norm,
                    epsilon=spec.rms_eps,
                )
            else:
                x = x + mlp_branch
                hidden = rms_norm(x, next_norm)
        return hidden

    def _forward_fused(self, x: torch.Tensor, layout: NpuBatchLayout) -> torch.Tensor:
        """Backward-compatible name for the fully fused Ascend path."""
        return self._forward_native(x, layout)

    def _forward_impl(self, x: torch.Tensor, layout: NpuBatchLayout) -> torch.Tensor:
        return self._forward_reference(x, layout) if self.fused_mode == "reference" else self._forward_native(x, layout)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, layout: NpuBatchLayout) -> torch.Tensor:
        batch = int(x.shape[0])
        bucket = self.decode_graph_bucket(batch) if batch <= self.graph_max_batch else batch
        graph_eligible = (
            self.use_graphs
            and layout.is_decode
            and batch <= self.graph_max_batch
            and layout.block_table.shape[1] <= self.graph_max_blocks
        )
        static_capacity = self._static_capacity_for(layout)
        static_key = (bucket, static_capacity or 0)
        if graph_eligible and static_capacity is not None and static_key not in self.static_graph_failures:
            graph_x, graph_layout = self._padded_decode(x, layout, bucket)
            graph = self.static_decode_graphs.get(static_key)
            if graph is None:
                graph = NpuDecodeGraph(
                    self,
                    bucket,
                    self.graph_max_blocks,
                    static_capacity=static_capacity,
                )
                self.static_decode_graphs[static_key] = graph
            try:
                return graph.run(graph_x, graph_layout)[:batch]
            except RuntimeError as exc:
                self.static_graph_failures[static_key] = f"{type(exc).__name__}: {exc}"
                self.static_decode_graphs.pop(static_key, None)
        graph_eligible = graph_eligible and bucket not in self.graph_failures
        if graph_eligible:
            graph_x, graph_layout = self._padded_decode(x, layout, bucket)
            graph = self.decode_graphs.get(bucket)
            if graph is None:
                graph = NpuDecodeGraph(self, bucket, self.graph_max_blocks)
                self.decode_graphs[bucket] = graph
            try:
                return graph.run(graph_x, graph_layout)[:batch]
            except RuntimeError as exc:
                self.graph_failures[bucket] = f"{type(exc).__name__}: {exc}"
                self.decode_graphs.pop(bucket, None)
        return self._forward_impl(x, layout)

    @torch.inference_mode()
    def forward_graph_lane(self, x: torch.Tensor, layout: NpuBatchLayout, lane: int, lanes: int) -> torch.Tensor:
        """Replay a decoder graph used once per host-synchronised K block.

        Each lane owns its graph inputs and dynamic paged-attention handles.
        A lane is not rebound while its previous replay can still be running:
        the caller consumes the whole block on the host before reusing lane 0.
        That removes the per-token host synchronize required by the single
        reusable graph while preserving the same sequential device work.
        """
        batch = int(x.shape[0])
        bucket = decode_bucket_size(batch, self.graph_max_batch) if batch <= self.graph_max_batch else batch
        static_capacity = self._static_capacity_for(layout)
        # Task-update PA needs one graph per lane because its update stream may
        # rebind the next replay while the current replay is still live.  Static
        # FIA has no host rebind: input copies, replay, sampling, and the next
        # input copy are all ordered on the current stream, so one graph is safe
        # to reuse for every step in the K block.
        graph_lanes = 1 if static_capacity is not None else int(lanes)
        graph_lane = 0 if static_capacity is not None else int(lane)
        key = (bucket, graph_lanes, static_capacity or 0)
        eligible = (
            self.use_graphs
            and layout.is_decode
            and batch <= self.graph_max_batch
            and layout.block_table.shape[1] <= self.graph_max_blocks
            and key not in self.graph_ring_failures
        )
        if not eligible:
            return self.forward(x, layout)
        graph_x, graph_layout = self._padded_decode(x, layout, bucket)
        ring = self.decode_graph_rings.setdefault(key, [])
        try:
            if not ring:
                self.decode_graph_ring_pools[key] = torch.npu.graph_pool_handle()
            captured_output = None
            while len(ring) <= graph_lane:
                for attempt in range(_GRAPH_CAPTURE_ATTEMPTS):
                    graph = NpuDecodeGraph(
                        self,
                        bucket,
                        self.graph_max_blocks,
                        pool=self.decode_graph_ring_pools[key],
                        static_capacity=static_capacity,
                    )
                    try:
                        captured_output = graph.run(graph_x, graph_layout, synchronize=False)
                    except RuntimeError as exc:
                        can_retry = _is_transient_graph_update_error(exc) and attempt + 1 < _GRAPH_CAPTURE_ATTEMPTS
                        if not can_retry:
                            raise
                        self.graph_ring_capture_retries[key] = self.graph_ring_capture_retries.get(key, 0) + 1
                        self.graph_ring_capture_retry_errors[key] = f"{type(exc).__name__}: {exc}"
                        # The failed task update is synchronous, but draining
                        # here prevents its discarded capture stream from
                        # racing the replacement graph/update stream.
                        try:
                            torch.npu.synchronize()
                        except RuntimeError:
                            pass
                    else:
                        ring.append(graph)
                        break
                else:  # pragma: no cover - the last failed attempt raises
                    raise RuntimeError("decode graph capture retries exhausted")
            if captured_output is not None:
                return captured_output[:batch]
            return ring[graph_lane].run(graph_x, graph_layout, synchronize=False)[:batch]
        except RuntimeError as exc:
            self.graph_ring_failures[key] = f"{type(exc).__name__}: {exc}"
            self.decode_graph_rings.pop(key, None)
            self.decode_graph_ring_pools.pop(key, None)
            return self.forward(x, layout)
