"""NPUGraph for the Talker codec head, sampler, and recurrent sampler state.

The decoder itself already runs as a graph ring.  This graph removes the
remaining per-frame eager launch train (head projection, repetition penalty,
top-p/top-k, sampling, and state updates). Counter uniforms are prepared once
per K block outside capture, using the same seed/frame hash as eager. Captured NPU matmul may make a
different rounding choice than eager projection, so this optimization is
accepted only through the frozen speech-quality gate rather than bit equality.
One graph is replayed repeatedly inside a K block.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from engine.talker_sampler import PAD, DecodeState, TalkerSampling, draw, processed_logits, uniform


class NpuTalkerPostGraphs:
    BUCKETS = (1, 2, 4, 8, 16, 32)

    def __init__(self, head: torch.Tensor, eos: int, cfg: TalkerSampling):
        self.head, self.eos, self.cfg = head, int(eos), cfg
        self.device = head.device
        self.accel = getattr(torch, self.device.type)
        self.pool = self.accel.graph_pool_handle()
        self.graphs: dict[int, tuple] = {}
        self.failures: dict[int, str] = {}
        self.replays: dict[int, int] = {}
        self.capture_context = nullcontext
        self._bound: tuple[int, int, DecodeState] | None = None
        self._uniforms: torch.Tensor | None = None
        self._step_index = 0

    def _bucket(self, n: int) -> int:
        return next(bucket for bucket in self.BUCKETS if bucket >= n)

    def _state(self, rows: int) -> dict[str, torch.Tensor]:
        z64 = lambda *shape: torch.zeros(*shape, dtype=torch.int64, device=self.device)  # noqa: E731
        zbool = lambda *shape: torch.zeros(*shape, dtype=torch.bool, device=self.device)  # noqa: E731
        return {
            "hidden": self.head.new_zeros((rows, self.head.shape[1])),
            "seeds": z64(rows),
            "uniform": torch.zeros(rows, dtype=torch.float32, device=self.device),
            "steps": z64(rows),
            "history": torch.full((rows, self.cfg.window), PAD, dtype=torch.int64, device=self.device),
            "finished": zbool(rows),
            "active": zbool(rows),
            "out": z64(rows),
            "feed": z64(rows),
        }

    def _body(self, st: dict[str, torch.Tensor]) -> None:
        live = st["active"] & ~st["finished"]
        logits = F.linear(st["hidden"], self.head).float()
        tok = draw(
            processed_logits(logits, st["history"], st["steps"], self.eos, self.cfg),
            st["seeds"],
            st["steps"],
            uniform_values=st["uniform"],
        )
        tok = torch.where(live, tok, torch.full_like(tok, PAD))
        shifted = torch.cat([st["history"][:, 1:], tok[:, None]], dim=1)
        st["out"].copy_(tok)
        st["feed"].copy_(tok.clamp_min(0))
        # Advance the recurrent state inside the captured graph.  Keeping
        # separate output state tensors required three eager D2D launches per
        # frame after replay, which recreated the launch train this graph is
        # meant to remove.  Stream ordering guarantees that all reads above
        # observe the old state before these in-place writes execute.
        st["history"].copy_(torch.where(live[:, None], shifted, st["history"]))
        st["steps"].add_(live.to(torch.int64))
        st["finished"].logical_or_(live & (tok == self.eos))

    @torch.inference_mode()
    def _capture(self, rows: int) -> tuple:
        with self.capture_context():
            st = self._state(rows)
            self._body(st)
            self.accel.synchronize()
            graph_class = getattr(self.accel, "CUDAGraph", None) or getattr(self.accel, "NPUGraph")
            graph = graph_class()
            with self.accel.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
                self._body(st)
            self.accel.synchronize()
        self.graphs[rows] = (graph, st)
        return graph, st

    @torch.inference_mode()
    def begin(self, state: DecodeState, *, frames: int = 1) -> bool:
        if frames < 1:
            raise ValueError("post-graph block must have at least one frame")
        n = int(state.steps.shape[0])
        rows = self._bucket(n)
        if rows in self.failures:
            return False
        if rows not in self.graphs:
            try:
                self._capture(rows)
            except Exception as exc:
                self.failures[rows] = f"{type(exc).__name__}: {exc}"
                return False
        _, st = self.graphs[rows]
        st["seeds"][:n].copy_(state.seeds)
        st["steps"][:n].copy_(state.steps)
        st["history"][:n].copy_(state.history)
        st["finished"][:n].copy_(state.finished)
        st["active"][:n].copy_(state.active)
        if rows > n:
            st["seeds"][n:].zero_()
            st["steps"][n:].zero_()
            st["history"][n:].fill_(PAD)
            st["finished"][n:].fill_(True)
            st["active"][n:].fill_(False)
            st["uniform"][n:].fill_(0.5)
        # The full integer counter-hash chain is not replay-equivalent on this
        # Ascend stack, even though its eager CPU/NPU results are identical.
        # Prepare the exact same uniforms once per K block outside capture.
        # No seed, distribution, or per-request frame-index rule is changed.
        offsets = torch.arange(frames, dtype=torch.int64, device=self.device)[:, None]
        self._uniforms = uniform(state.seeds[None, :], state.steps[None, :] + offsets)
        self._step_index = 0
        self._bound = (n, rows, state)
        return True

    @torch.inference_mode()
    def step(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._bound is None:
            raise RuntimeError("Talker post graph is not bound")
        n, rows, _ = self._bound
        if self._uniforms is None or self._step_index >= self._uniforms.shape[0]:
            raise RuntimeError("post graph exhausted its prepared block uniforms")
        graph, st = self.graphs[rows]
        st["hidden"][:n].copy_(hidden)
        st["uniform"][:n].copy_(self._uniforms[self._step_index])
        self._step_index += 1
        if rows > n:
            st["hidden"][n:].zero_()
        graph.replay()
        token = st["out"][:n].clone()
        feed = st["feed"][:n]
        self.replays[rows] = self.replays.get(rows, 0) + 1
        return token, feed

    @torch.inference_mode()
    def finish(self) -> None:
        if self._bound is None:
            return
        n, rows, state = self._bound
        _, st = self.graphs[rows]
        state.steps.copy_(st["steps"][:n])
        state.history.copy_(st["history"][:n])
        state.finished.copy_(st["finished"][:n])
        self._bound = None
        self._uniforms = None
        self._step_index = 0
