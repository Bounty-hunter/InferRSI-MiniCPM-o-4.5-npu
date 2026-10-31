"""Keep temporary profiling instrumentation out of the production Talker loop."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / "engine" / "runtime" / "backend_npu.py"


def _method_source(name: str) -> str:
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"method {name!r} not found")


def test_talker_block_has_no_runtime_profiler_instrumentation():
    block = _method_source("_talker_decode_block")
    implementation = _method_source("_talker_decode_block_impl")
    hot_path = block + implementation
    assert "cProfile" not in hot_path
    assert "record_function" not in hot_path
    assert "torch.npu.Event" not in hot_path
    assert "elapsed_time" not in hot_path


def test_backend_constructor_does_not_expose_temporary_talker_profilers():
    constructor = _method_source("__init__")
    assert "talker_profile_dir" not in constructor
    assert "talker_event_profile" not in constructor


def test_talker_post_graph_requires_counter_rng_not_non_omni_semantics():
    constructor = _method_source("__init__")
    assert "if self.cfg.uses_native_rng" in constructor
    assert "talker_post_graphs do not preserve native per-request RNG state" in constructor
    assert 'if talker_sampling_semantics == "omni"' not in constructor


def test_t2w_batch_wait_survives_the_c16_tail_without_affecting_c1():
    source = SOURCE.read_text(encoding="utf-8")
    worker = _method_source("_run_t2w")
    constructor = _method_source("__init__")

    assert "_T2W_BATCH_WAIT_MIN_ACTIVE = 8" in source
    assert "t2w_batch_wait_min_active: int = _T2W_BATCH_WAIT_MIN_ACTIVE" in constructor
    assert "t2w_batch_wait_min_active must be positive" in constructor
    assert "t2w_batch_min_ready: int | None = None" in constructor
    assert "t2w_batch_min_ready must be positive when set" in constructor
    assert "target >= self.t2w_batch_wait_min_active" in worker
    assert "wait_target = min(target, self.t2w_batch_min_ready or target)" in worker
    assert "len(ready) < wait_target" in worker


def test_prompt_setup_batching_is_opt_in_and_preserves_shared_scratch_locking():
    constructor = _method_source("__init__")
    prepare = _method_source("_prepare")
    setup = _method_source("_setup_prompts")
    coordinator = _method_source("_run_prompt_setups")

    assert "prompt_setup_batching: bool = False" in constructor
    assert "prompt_setup_wait_ms: float = 0.0" in constructor
    assert "prompt_setup_wait_ms must be non-negative" in constructor
    assert "self.t2w.features(ref, defer_speaker=True)" in prepare
    assert "if not self.prompt_setup_batching" in prepare
    assert "self.prompt_setup_inbox.put(item)" in prepare
    assert "with self._t2w_lock" in setup
    assert "self._graph_gate.launch()" in setup
    assert "self.t2w_batch.setup_many(feats_list)" in setup
    assert "ROW_LIMIT" in coordinator and "SETUP_T" in coordinator
    assert "self.prompt_setup_batches[len(part)]" in coordinator
    assert "item.ready.set()" in coordinator

