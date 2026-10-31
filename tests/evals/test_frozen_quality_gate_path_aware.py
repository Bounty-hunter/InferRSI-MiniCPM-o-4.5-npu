from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "run_frozen_quality_gate_path_aware.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("path_aware_quality_gate_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verified_dref_adapter_checks_frozen_rows_without_source_traversal(tmp_path):
    runner = load_runner()
    module = SimpleNamespace(DIRECT_DREF_SCHEMA="ascend_direct_d_ref/v1")
    runner.install_verified_dref_adapter(module)
    dref = tmp_path / "dref.json"
    dref.write_text(json.dumps({
        "schema": "ascend_direct_d_ref/v1",
        "role": "truncation",
        "concurrency": 32,
        "gpu_class": {"gpu_sku": "Ascend910B1", "gpu_driver": "x", "gpu_memory_mib": 64},
        "model": {"repo": "model", "revision": "rev"},
        "dataset": {"repo": "data", "revision": "rev"},
        "seed_policy": "request_seed",
        "sources": [{"cell": "/intentionally/not/visited"}],
        "rows": {"row-a": 1.5, "row-b": 2.5},
    }))
    expect = {
        "role": "truncation",
        "concurrency": 32,
        "gpu_class": {"gpu_sku": "Ascend910B1", "gpu_driver": "x", "gpu_memory_mib": 64},
        "model": {"repo": "model", "revision": "rev"},
        "dataset": {"repo": "data", "revision": "rev"},
        "seed_policy": "request_seed",
        "rows": ["row-a", "row-b"],
    }
    problems, rows = module.verify_dref(dref, expect, check_sources=True)
    assert problems == []
    assert rows == {"row-a": 1.5, "row-b": 2.5}


def test_verified_dref_adapter_fails_closed_on_mismatch(tmp_path):
    runner = load_runner()
    module = SimpleNamespace(DIRECT_DREF_SCHEMA="ascend_direct_d_ref/v1")
    runner.install_verified_dref_adapter(module)
    dref = tmp_path / "dref.json"
    dref.write_text(json.dumps({
        "schema": "ascend_direct_d_ref/v1",
        "role": "truncation",
        "concurrency": 1,
        "gpu_class": {},
        "model": {},
        "dataset": {},
        "seed_policy": "request_seed",
        "rows": {"row-a": -1.0},
    }))
    problems, rows = module.verify_dref(dref, {"concurrency": 32, "rows": ["row-a", "row-b"]})
    assert rows == {}
    assert any("non-positive" in problem for problem in problems)
    assert any("concurrency" in problem for problem in problems)
    assert any("row-b" in problem for problem in problems)


def test_candidate_qref_binding_is_checked_per_source_cell(tmp_path):
    runner = load_runner()
    qref = tmp_path / "qref.json"
    qref.write_text('{"frozen": true}')
    expected = runner.sha256(qref)
    cells = []
    for seed, digest in ((0, expected), (1, "0" * 64)):
        cell = tmp_path / f"cell-{seed}"
        cell.mkdir()
        (cell / "experiment.json").write_text(json.dumps({
            "experiment": {"frozen_refs": {"speech_qref": {"sha256": digest}}}
        }))
        cells.append((seed, cell))
    verified = SimpleNamespace(report={
        "generation": {"candidate_digest": "headline"},
        "seeds": {str(seed): {"cell": str(cell)} for seed, cell in cells},
    })
    problems = runner.binding_problems(verified, qref, "headline")
    assert len(problems) == 1
    assert "seed 1" in problems[0]


def test_candidate_qref_binding_checks_headline_digest(tmp_path):
    runner = load_runner()
    qref = tmp_path / "qref.json"
    qref.write_text('{}')
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "experiment.json").write_text(json.dumps({
        "experiment": {"frozen_refs": {"speech_qref": {"sha256": runner.sha256(qref)}}}
    }))
    verified = SimpleNamespace(report={
        "generation": {"candidate_digest": "other"},
        "seeds": {"0": {"cell": str(cell)}},
    })
    assert runner.binding_problems(verified, qref, "headline") == [
        "candidate report does not name the headline digest"
    ]
