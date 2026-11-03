"""Capture actual component inputs once; these requests are diagnostic only."""
import asyncio
import argparse
import dataclasses
import hashlib
from pathlib import Path
import threading
import time

from fixed_common import ENGINE, HERE, OUT, ROOT, cpu_tree, initialize, kv_snapshot, save_json, tensor_digest, install_resource_observers

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='fixed-inputs-r2')
args = parser.parse_args()
folder = initialize(args.tag)
import torch
from bench.workload import canonical_requests, load_seed_tts_rows
from engine.runtime.backend_npu import AscendTrackSBackend
from engine.runtime.talker_block_graph import NpuTalkerBlockGraph
from engine.runtime.token2wav_batch import BatchedToken2Wav
from engine.server import served_identity
from tools.probe_engine_npu_quality import closed_loop

active = False
captured = {}
lock = threading.Lock()


def write_case(name, payload, expected):
    payload['expected'] = cpu_tree(expected)
    path = folder / f'{name}.pt'
    torch.save(payload, path)
    record = dict(path=str(path), bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  expected_digest=tensor_digest(expected))
    with lock:
        captured[name] = record
        save_json(folder / 'manifest.json', dict(cases=captured, scope='fixed diagnostic inputs, excluded from native performance'))
    print({'captured': name, **record}, flush=True)


original_decoder = AscendTrackSBackend._decoder_forward


def decoder(self, runner, inputs, layout):
    name = 'thinker'
    eligible = (active and runner is self.trun and name not in captured
                and not layout.is_fresh_prefill and len(layout.seqs) >= 2)
    if not eligible:
        return original_decoder(self, runner, inputs, layout)
    snapshot = dict(inputs=cpu_tree(inputs), layout=cpu_tree(layout),
                    kv=kv_snapshot(self.tkv, layout.seqs),
                    batch_append_prefill=runner.batch_append_prefill,
                    selected_bucket=runner.decode_graph_bucket(int(inputs.shape[0])) if layout.is_decode else None,
                    shape=list(inputs.shape), new_tokens=list(layout.new_tokens))
    result = original_decoder(self, runner, inputs, layout)
    write_case(name, snapshot, result)
    return result


original_block = NpuTalkerBlockGraph.run


def block(self, feeds, state, starts, blocks):
    eligible = active and 'talker' not in captured and self.batch in (2, 4)
    if not eligible:
        return original_block(self, feeds, state, starts, blocks)
    seqs = [next(seq for seq, pages in self.runner.kv.pages.items() if list(pages) == list(block)) for block in blocks]
    snapshot = dict(feeds=cpu_tree(feeds), state=cpu_tree(state), starts=starts, blocks=blocks,
                    kv=kv_snapshot(self.runner.kv, seqs), batch=self.batch, frames=self.frames)
    result = original_block(self, feeds, state, starts, blocks)
    write_case('talker', snapshot, result[0])
    return result


original_decode = BatchedToken2Wav.decode


def decode(self, items, async_result=False):
    eligible = active and 'audio' not in captured and 2 <= len(items) <= 8 and not any(last for _, _, last in items)
    if not eligible:
        return original_decode(self, items, async_result=async_result)
    snapshot = dict(items=cpu_tree(items), noise=cpu_tree(self.flow.decoder.rand_noise),
                    batch=len(items), window_lengths=[len(tokens) for _, tokens, _ in items])
    result = original_decode(self, items, async_result=async_result)
    write_case('audio', snapshot, result)
    return result


AscendTrackSBackend._decoder_forward = decoder
NpuTalkerBlockGraph.run = block
BatchedToken2Wav.decode = decode
install_resource_observers(AscendTrackSBackend, folder)
import json
config = json.loads((HERE / 'config-A.json').read_text())
config.pop('backend')
backend = AscendTrackSBackend(model_dir=Path('/dataset/models/MiniCPM-o-4_5'), **config)
save_json(folder / 'info.json', served_identity(str(HERE / 'config-A.json'), 'engine.runtime.backend_npu:AscendTrackSBackend', backend))
assert torch.npu.get_device_limit(0) == {'cube_core_num': 24, 'vector_core_num': 48}


async def run():
    global active
    rows = load_seed_tts_rows(Path('/dataset/models/seed-tts-eval'))
    requests = canonical_requests(rows, [row.row_id for row in rows[:16]], seed_index=0)
    save_json(folder / 'requests.json', [req.describe() for req in requests])
    original_done = backend._done
    request_states = {}

    def done(live):
        request_states[live.req.request_id] = dict(thinker_tokens=list(live.generated), codec_tokens=list(live.codes))
        return original_done(live)

    backend._done = done
    active = True
    measured, wall = await asyncio.wait_for(closed_loop(backend, requests, 8), timeout=600)
    active = False
    clean = [{key: value for key, value in row.items() if key != 'audio'} for row in measured]
    save_json(folder / 'requests-results.json', dict(rows=clean, states=request_states, wall_s=wall,
                                                   diagnostic_only=True, stats=backend.stats()))
    assert all(row['ok'] for row in clean)
    assert set(captured) == {'thinker', 'talker', 'audio'}, captured


asyncio.run(run())
torch.npu.synchronize()
save_json(OUT / 'fixture-reference.json', dict(directory=str(folder),
          manifest_sha256=hashlib.sha256((folder / 'manifest.json').read_bytes()).hexdigest()))
print('Fixed component inputs captured successfully', flush=True)
