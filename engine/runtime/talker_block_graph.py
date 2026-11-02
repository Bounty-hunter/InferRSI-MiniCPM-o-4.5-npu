"""Experimental full K-block graph with counter uniforms prepared outside capture.

Every exact batch fitting the existing static-attention graph limit is
supported.  Exact shapes avoid padding sampler/KV state, while allowing the
service to keep dynamic 3/5/6/7-row Talker cohorts on the full K-block path.
Larger cohorts are split by the caller into exact graph-sized microbatches.
"""
from __future__ import annotations

import torch

from engine.runtime.npu_paged import NpuDecodeGraph, _StaticFiaGraphCapture
from engine.runtime.talker_post_graph import NpuTalkerPostGraphs
from engine.talker_sampler import DecodeState, uniform


class NpuTalkerBlockGraph:
    def __init__(self, runner, embed, head, eos, cfg, batch: int, frames: int):
        if not runner.static_shape_attention or not runner.use_graphs or cfg.rng != 'counter':
            raise ValueError('full block requires static graphs and the qualified counter RNG')
        if batch < 1 or batch > runner.graph_max_batch or frames < 1:
            raise ValueError('unsupported full block shape')
        self.runner, self.embed, self.frames = runner, embed, int(frames)
        self.batch, self.cfg = int(batch), cfg
        self.pool = torch.npu.graph_pool_handle()
        # Follow NpuDecodeGraph's private capture-stream discipline. This
        # keeps captured model GEMM workspaces away from other model lanes.
        self.capture_stream = torch.npu.Stream(device=head.device)
        self.view = NpuDecodeGraph(runner, batch, runner.graph_max_blocks, pool=self.pool,
                                   static_capacity=runner.static_shape_capacity)
        self.view.block_table.zero_()
        self.post = NpuTalkerPostGraphs(head, eos, cfg)
        self.st = self.post._state(batch)
        self.codes = torch.zeros(batch, dtype=torch.long, device=head.device)
        self.uniforms = torch.zeros((frames, batch), dtype=torch.float32, device=head.device)
        self.frame_positions = torch.zeros((frames, batch), dtype=torch.long, device=head.device)
        self.frame_slots = torch.zeros((frames, batch), dtype=torch.int32, device=head.device)
        self.frame_lengths = torch.zeros((frames, batch), dtype=torch.int32, device=head.device)
        self.graph = None
        self.output = None
        self.capture = None

    def _bind(self, feeds, state, starts, blocks, uniforms):
        n = self.batch
        if feeds.shape != (n,) or len(starts) != n or len(blocks) != n:
            raise ValueError('full block input cardinality mismatch')
        if min(starts) < 0 or max(starts) + self.frames > self.runner.static_shape_capacity:
            raise ValueError('full block would exceed static KV capacity')
        for row, start in zip(blocks, starts):
            needed = -(-(start + self.frames) // self.runner.kv.page_size)
            if len(row) < needed or any(page < 0 or page >= self.runner.kv.num_pages for page in row[:needed]):
                raise ValueError('full block needs valid pages for every speculative frame')
        self.codes.copy_(feeds)
        self.uniforms.copy_(uniforms)
        positions = [[start + frame for start in starts] for frame in range(self.frames)]
        slots = [[blocks[index][pos // self.runner.kv.page_size] * self.runner.kv.page_size
                  + pos % self.runner.kv.page_size for index, pos in enumerate(row)] for row in positions]
        lengths = [[pos + 1 for pos in row] for row in positions]
        self.frame_positions.copy_(torch.tensor(positions, dtype=torch.long, device=feeds.device))
        self.frame_slots.copy_(torch.tensor(slots, dtype=torch.int32, device=feeds.device))
        self.frame_lengths.copy_(torch.tensor(lengths, dtype=torch.int32, device=feeds.device))
        self.view.block_table.zero_()
        for index, row in enumerate(blocks):
            if len(row) > self.runner.graph_max_blocks:
                raise ValueError('block table exceeds graph capacity')
            self.view.block_table[index, :len(row)].copy_(torch.tensor(row, dtype=torch.int32, device=feeds.device))
        for name in ('seeds', 'steps', 'history', 'finished', 'active'):
            self.st[name].copy_(getattr(state, name))

    def _body(self):
        view = self.view
        tokens = []
        for frame in range(self.frames):
            # Match the original host-generated metadata exactly. Rebinding
            # physical pages must not rely on an unqualified captured integer
            # gather/divide chain; its replay behavior is still under audit.
            view.positions.copy_(self.frame_positions[frame])
            view.slot_mapping.copy_(self.frame_slots[frame])
            view.seq_lens_device.copy_(self.frame_lengths[frame])
            self.capture.prepare()
            hidden = self.runner._forward_impl(self.embed(self.codes), view.layout)
            # Retain the original row-selection/projection kernels. The only
            # change here is capture scope and device-owned frame metadata.
            self.st['hidden'].copy_(hidden[view.layout.last_token_index])
            self.st['uniform'].copy_(self.uniforms[frame])
            self.post._body(self.st)
            tokens.append(self.st['out'].clone())
            self.codes.copy_(self.st['feed'])
        return torch.stack(tokens)

    @torch.inference_mode()
    def run(self, feeds, state: DecodeState, starts: list[int], blocks: list[list[int]]):
        offsets = torch.arange(self.frames, dtype=torch.long, device=feeds.device)[:, None]
        values = uniform(state.seeds[None, :], state.steps[None, :] + offsets)
        if self.graph is None:
            with self.runner.capture_context():
                self.capture = _StaticFiaGraphCapture(self.view)
                previous = self.runner._graph_capture
                self.runner._graph_capture = self.capture
                try:
                    current = torch.npu.current_stream()
                    stream = self.capture_stream
                    for _ in range(2):
                        self._bind(feeds, state, starts, blocks, values)
                        stream.wait_stream(current)
                        with torch.npu.stream(stream):
                            self._body()
                        current.wait_stream(stream)
                    current.synchronize()
                    self.capture.outputs.clear()
                    self.capture.softmax_lse.clear()
                    self._bind(feeds, state, starts, blocks, values)
                    stream.wait_stream(current)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream, pool=self.pool, capture_error_mode='thread_local'):
                        output = self._body()
                    current.wait_stream(stream)
                    current.synchronize()
                    self.graph, self.output = graph, output
                finally:
                    self.runner._graph_capture = previous
        self._bind(feeds, state, starts, blocks, values)
        self.graph.replay()
        result_state = DecodeState(*(self.st[name].clone() for name in ('seeds', 'steps', 'history', 'finished', 'active')))
        return self.output.clone(), result_state
