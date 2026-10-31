from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "run_frozen_quality_pipeline_r149.py"
PREPARER = ROOT / "tools" / "prepare_remote_r149_quality_postprocess.sh"


def load_runner():
    spec = importlib.util.spec_from_file_location("quality_pipeline_r149_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_json_new_never_rewrites_an_audit(tmp_path):
    runner = load_runner()
    target = tmp_path / "audit.json"
    runner.write_json_new(target, {"first": True})
    with pytest.raises(FileExistsError):
        runner.write_json_new(target, {"first": False})
    assert json.loads(target.read_text()) == {"first": True}


def test_candidate_requires_a_matching_baseline_dref_audit(tmp_path):
    runner = load_runner()
    dref = tmp_path / "dref.json"
    dref.write_text('{"rows":{"a":1.0}}')
    freeze = tmp_path / "freeze.json"
    freeze.write_text('{}')
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{}')
    frozen = tmp_path / "validator.py"
    frozen.write_text('# frozen')
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    ctx = SimpleNamespace(
        quality_freeze={"baseline_source": {"container_root": str(baseline)}},
        quality_freeze_path=freeze,
        runtime_path=runtime,
        frozen_paths={"validator.py": frozen},
    )
    audit = tmp_path / "dref-audit.json"
    audit.write_text(json.dumps({
        "schema": runner.DREF_AUDIT_SCHEMA,
        "ok": True,
        "source_root": str(baseline),
        "dref": runner.file_record(dref),
        "quality_freeze": runner.file_record(freeze),
        "runtime_manifest": runner.file_record(runtime),
        "frozen_files": {"validator.py": runner.sha256(frozen)},
    }))
    receipt = runner.require_baseline_dref_audit(ctx, dref, audit)
    assert receipt["audit"]["sha256"] == runner.sha256(audit)

    changed = json.loads(audit.read_text())
    changed["source_root"] = str(tmp_path / "candidate")
    audit.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="baseline source root"):
        runner.require_baseline_dref_audit(ctx, dref, audit)


def test_dref_audit_hash_binds_the_exact_dref(tmp_path):
    runner = load_runner()
    dref = tmp_path / "dref.json"
    dref.write_text('{"rows":{"a":1.0}}')
    freeze = tmp_path / "freeze.json"
    freeze.write_text('{}')
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{}')
    frozen = tmp_path / "validator.py"
    frozen.write_text('# frozen')
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    ctx = SimpleNamespace(
        quality_freeze={"baseline_source": {"container_root": str(baseline)}},
        quality_freeze_path=freeze,
        runtime_path=runtime,
        frozen_paths={"validator.py": frozen},
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "schema": runner.DREF_AUDIT_SCHEMA,
        "ok": True,
        "source_root": str(baseline),
        "dref": runner.file_record(dref),
        "quality_freeze": runner.file_record(freeze),
        "runtime_manifest": runner.file_record(runtime),
        "frozen_files": {"validator.py": runner.sha256(frozen)},
    }))
    dref.write_text('{"rows":{"a":2.0}}')
    with pytest.raises(RuntimeError, match="path or hash"):
        runner.require_baseline_dref_audit(ctx, dref, audit)


def test_r149_postprocess_waits_for_reference_and_never_bypasses_npu_scheduler():
    text = PREPARER.read_text()
    assert "minicpmo-npu-20261024-quality-reference-r149.ready" in text
    assert "minicpmo-quality-reference-bundles-r149" in text
    assert text.count("docker exec --user 1033:1000 dyy") >= 10
    assert "npu run" not in text
    assert "build-dref" in text and "export-bundle" in text
    for suffix in ("r149a", "r149b", "r149c", "r149d", "r149e", "r149f"):
        assert suffix in text
    assert "baseline-c1-bundle/manifest.json" in text
    assert "baseline-c32-bundle/manifest.json" in text
