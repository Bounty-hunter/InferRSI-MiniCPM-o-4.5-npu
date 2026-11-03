"""Fixed-input component and serial/multistream quota counterfactuals."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import statistics
import time

from fixed_common import HERE, OUT, device_tree, initialize, restore_kv, save_json, tensor_digest, install_resource_observers

parser = argparse.ArgumentParser()
parser.add_argument('--tag', required=True)
parser.add_argument('--role', choices=('A', 'B'), required=True)
parser.add_argument('--profile', action='store_true')
args = parser.parse_args()
folder = initialize(args.tag)

import numpy as np
import torch
import torch_npu
from engine.models.token2wav import StreamState
from engine.runtime.backend_npu import AscendTrackSBackend
from engine.runtime.npu_paged import NpuDecodeGraph
from engine.server import served_identity

resource_api = Path(torch_npu.__file__).parent / 'npu/npu_config.py'
save_json(folder / 'resource-api.json', dict(torch_npu_path=torch_npu.__file__,
          set_device_limit_callable_type=type(torch.npu.set_device_limit).__name__,
          source_file=str(resource_api), source_sha256=hashlib.sha256(resource_api.read_bytes()).hexdigest()))
install_resource_observers(AscendTrackSBackend, folder)
config = json.loads((HERE / f'config-{args.role}.json').read_text())
config.pop('backend')
backend = AscendTrackSBackend(model_dir=Path('/dataset/models/MiniCPM-o-4_5'), **config)
save_json(folder / 'info.json', served_identity(str(HERE / f'config-{args.role}.json'),
          'engine.runtime.backend_npu:AscendTrackSBackend', backend))
limit = torch.npu.get_device_limit(0)
expected_limit = {'cube_core_num': 24, 'vector_core_num': 48} if args.role == 'A' else {'cube_core_num': 12, 'vector_core_num': 24}
assert limit == expected_limit, limit
save_json(folder / 'device-limit.json', limit)

fixtures = {}
fixture_reference = json.loads((OUT / 'fixture-reference.json').read_text())
fixture_folder = Path(fixture_reference['directory'])
assert hashlib.sha256((fixture_folder / 'manifest.json').read_bytes()).hexdigest() == fixture_reference['manifest_sha256']
for name in ('thinker', 'talker', 'audio'):
    path = fixture_folder / f'{name}.pt'
    fixtures[name] = torch.load(path, map_location='cpu', weights_only=False)
save_json(folder / 'input-files.json', {name: hashlib.sha256((fixture_folder / f'{name}.pt').read_bytes()).hexdigest()
                                      for name in fixtures})

thinker = fixtures['thinker']
talker = fixtures['talker']
audio = fixtures['audio']
with torch.inference_mode():
    restore_kv(backend.tkv, thinker['kv'])
    restore_kv(backend.akv, talker['kv'])
    thinker_inputs = device_tree(thinker['inputs'])
    thinker_layout = device_tree(thinker['layout'])
    thinker_layout.context_lens = thinker['layout'].context_lens
    backend.trun.batch_append_prefill = thinker['batch_append_prefill']
    if thinker['selected_bucket'] is not None:
        bucket = thinker['selected_bucket']
        backend.trun.decode_graphs[bucket] = NpuDecodeGraph(backend.trun, bucket, backend.trun.graph_max_blocks)
    feeds = device_tree(talker['feeds'])
    state = device_tree(talker['state'])
    talker_graph = backend.talker_full_block_graphs[(talker['batch'], talker['frames'])]
    template_items = device_tree(audio['items'])
    backend.t2w.flow.decoder.rand_noise.copy_(audio['noise'].to(backend.device))
torch.npu.synchronize()
engine_stream = torch.npu.current_stream()
audio_stream = backend.t2w_stream


def thinker_task():
    with torch.npu.stream(engine_stream), backend._graph_gate.launch():
        return backend.trun.forward(thinker_inputs, thinker_layout)


def talker_task():
    with torch.npu.stream(engine_stream), backend._graph_gate.launch():
        return talker_graph.run(feeds, state, talker['starts'], talker['blocks'])[0]


def audio_task():
    # The source decode replaces every cache entry. Rebuild request-owned
    # dictionaries while reusing immutable initial tensors for each trial.
    items = [(StreamState(st.prompt, dict(st.flow_cache), dict(st.hift_cache), st.chunks, st.noise_seed), list(tokens), last)
             for st, tokens, last in template_items]
    with backend._t2w_lock, torch.npu.stream(audio_stream), backend._graph_gate.launch():
        return backend.t2w_batch.decode(items)


def summary(values):
    return dict(n=len(values), mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                min_ms=min(values), max_ms=max(values), samples_ms=values)


def compare(actual, expected):
    if isinstance(actual, torch.Tensor):
        a = actual.detach().cpu().float().numpy()
        b = expected.float().numpy()
    else:
        a = np.concatenate([item.reshape(-1) for item in actual])
        b = np.concatenate([item.reshape(-1) for item in expected])
    return dict(exact=bool(np.array_equal(a, b)), shape=list(a.shape),
                max_abs=float(np.max(np.abs(a-b))),
                relative_l2=float(np.linalg.norm((a-b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-30)))


def measure(function, stream, repeats):
    samples = []
    device_elapsed = []
    for _ in range(repeats):
        torch.npu.synchronize()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record(stream)
        result = function()
        end.record(stream)
        torch.npu.synchronize()
        samples.append((time.perf_counter()-started)*1000)
        device_elapsed.append(begin.elapsed_time(end))
    return dict(host_completion=summary(samples), stream_elapsed=summary(device_elapsed)), result


def pair(mode, engine_loops):
    for _ in range(engine_loops):
        thinker_task()
        talker_task()
    if mode == 'serial':
        engine_stream.synchronize()
    audio_task()
    torch.npu.synchronize()


results = dict(role=args.role, quota=limit, diagnostic_only=True,
               scope='Frozen real inputs, reused prefix/cache and fixed batch. Single host submitter; not native service latency.',
               shapes=dict(thinker=thinker['shape'], thinker_new_tokens=thinker['new_tokens'],
                           thinker_bucket=thinker['selected_bucket'], talker_batch=talker['batch'],
                           talker_frames=talker['frames'], audio_batch=audio['batch'],
                           audio_windows=audio['window_lengths']), isolated={}, matrix={})
tasks = dict(thinker=(thinker_task, engine_stream), talker=(talker_task, engine_stream), audio=(audio_task, audio_stream))

with torch.inference_mode():
    frozen_inputs_before = tensor_digest((thinker_inputs, feeds, state, template_items))
    for name, (function, stream) in tasks.items():
        for _ in range(5):
            function()
            torch.npu.synchronize()
        first = function()
        torch.npu.synchronize()
        first_digest = tensor_digest(first)
        measurements, last = measure(function, stream, repeats=3 if args.profile else 20)
        last_digest = tensor_digest(last)
        results['isolated'][name] = dict(**measurements, same_input_repeated_exact=first_digest == last_digest,
                                        digest=last_digest, versus_capture=compare(last, fixtures[name]['expected']))
        save_json(folder / 'results.json', results)
        assert first_digest == last_digest, (name, 'Repeated fixed-input output changed')
    for loops in (1, 4):
        for mode in ('serial', 'parallel'):
            pair(mode, loops)
        samples = {'serial': [], 'parallel': []}
        for repeat in range(2 if args.profile else 12):
            order = ('serial', 'parallel') if repeat % 2 == 0 else ('parallel', 'serial')
            for mode in order:
                torch.npu.synchronize()
                started = time.perf_counter()
                pair(mode, loops)
                samples[mode].append((time.perf_counter()-started)*1000)
        results['matrix'][str(loops)] = {mode: summary(values) for mode, values in samples.items()}
        save_json(folder / 'results.json', results)
    results['input_templates_unchanged'] = frozen_inputs_before == tensor_digest((thinker_inputs, feeds, state, template_items))
    assert results['input_templates_unchanged'], 'Input cache mutation invalidates fixed replay'
    if args.profile:
        experimental = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True, profile_memory=False, with_stack=False,
            experimental_config=experimental,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(folder / 'profile')),
        ) as profiler:
            for mode in ('serial', 'parallel'):
                for repeat in range(3):
                    with torch.profiler.record_function(f'core-study:{mode}:engine4:repeat{repeat}'):
                        pair(mode, 4)
                    profiler.step()
        results['profile'] = dict(path=str(folder / 'profile'), with_stack=False,
                                  record_shapes=True, aic_metrics='PipeUtilization', owner='main diagnostic thread')
    results['stats_after'] = backend.stats()
    save_json(folder / 'results.json', results)
print({'finished': args.tag, 'quota': limit}, flush=True)
