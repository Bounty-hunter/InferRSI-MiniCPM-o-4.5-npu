"""Diagnostic-only fixed-input snapshots and replay setup for the quota study."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path('/workspace/minicpmo-r182')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts' / HERE.name
ENGINE = ROOT / 'rsi-r312-r351-20261102/engine-r320'


def initialize(tag):
    assert os.getuid() == 1033 and os.getgid() == 1000
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    folder = OUT / tag
    folder.mkdir(exist_ok=False)
    os.environ.update(MINICPMO_ENGINE_ROOT=str(ENGINE), MINICPMO_RESULT_DIR=str(folder),
                      MODEL_PATH='/dataset/models/MiniCPM-o-4_5')
    sys.path.insert(0, str(ENGINE))
    # Reuse the runtime-import audit without starting the public HTTP server.
    entry = ROOT / 'run_engine_standalone.py'
    prefix, separator, _ = entry.read_text().partition('from engine.server import main')
    assert separator
    exec(compile(prefix, str(entry), 'exec'), {'__name__': 'core_study_audit', '__file__': str(entry)})
    return folder


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **{f.name: cpu_tree(getattr(value, f.name)) for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def device_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to('npu:0')
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **{f.name: device_tree(getattr(value, f.name)) for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {key: device_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [device_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(device_tree(item) for item in value)
    return value


def kv_snapshot(kv, seqs):
    import torch
    mapping = {seq: list(kv.pages[seq]) for seq in seqs}
    pages = sorted({page for block in mapping.values() for page in block})
    index = torch.tensor(pages, dtype=torch.long, device=kv.k.device)
    return dict(pages=pages, mapping=mapping, lengths={seq: kv.lengths[seq] for seq in seqs},
                key=kv.k.index_select(1, index).cpu(), value=kv.v.index_select(1, index).cpu())


def restore_kv(kv, payload):
    import torch
    assert not kv.pages, 'Fresh warmed backend must have no live request KV'
    kv.pages = {int(seq): list(pages) for seq, pages in payload['mapping'].items()}
    kv.lengths = dict(payload['lengths'])
    owned = set(payload['pages'])
    kv.free = [page for page in range(kv.num_pages - 1, -1, -1) if page not in owned]
    index = torch.tensor(payload['pages'], dtype=torch.long, device=kv.k.device)
    kv.k.index_copy_(1, index, payload['key'].to(kv.k.device))
    kv.v.index_copy_(1, index, payload['value'].to(kv.v.device))


def tensor_digest(value):
    import numpy as np
    import torch
    digest = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str((tensor.shape, tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            digest.update(field.name.encode())
            digest.update(tensor_digest(getattr(value, field.name)).encode())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            digest.update(tensor_digest(value[key]).encode())
    elif isinstance(value, (list, tuple)):
        for item in value:
            digest.update(tensor_digest(item).encode())
    else:
        digest.update(repr(value).encode())
    return digest.hexdigest()


def resource_record():
    import torch
    stream = torch.npu.current_stream()
    record = dict(thread=threading.current_thread().name, tid=threading.get_native_id(),
                  device_limit=torch.npu.get_device_limit(0), stream=str(stream))
    if hasattr(torch.npu, 'get_stream_limit'):
        try:
            record['stream_limit'] = torch.npu.get_stream_limit(stream)
        except Exception as error:
            record['stream_limit_error'] = repr(error)
    return record


def install_resource_observers(backend_class, folder):
    import torch
    for method_name, label in (('_run', 'engine'), ('_run_t2w', 'audio'), ('_warm_media_setup_graphs', 'prompt')):
        original = getattr(backend_class, method_name)

        def observed(self, *args, _original=original, _label=label, **kwargs):
            torch.npu.set_device(0)
            stream = self.prompt_stream if _label == 'prompt' else (
                self.t2w_stream if _label == 'audio' else torch.npu.current_stream())
            with torch.npu.stream(stream):
                save_json(folder / f'resource-{_label}.json', resource_record())
            return _original(self, *args, **kwargs)

        setattr(backend_class, method_name, observed)
