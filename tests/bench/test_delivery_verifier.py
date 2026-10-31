import json
import importlib.util
import io
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_verifier():
    path = ROOT / "outputs" / "verify-minicpmo-r152-delivery.py"
    spec = importlib.util.spec_from_file_location("delivery_verifier_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delivery_verifier_exit_status_is_final_acceptance_signal():
    verifier = ROOT / "outputs" / "verify-minicpmo-r152-delivery.py"
    proc = subprocess.run(
        [sys.executable, str(verifier)],
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(proc.stdout)
    assert proc.returncode == (0 if report["final_acceptance_passed"] else 1)
    assert report["checks"]["standalone_runtime_imports"] is True
    assert report["forbidden_runtime_imports"] == []
    assert report["final_acceptance_passed"] is (
        report["performance_delivery_passed"] and report["quality_gate"]["passed"]
    )


def _write_quality_fixture(root: Path, verifier, *, bad_audit: bool = False) -> None:
    for name in verifier.QUALITY_SUPPORT_FILES:
        (root / name).write_text("ready\n")
    for concurrency, name in verifier.QUALITY_GATE_FILES.items():
        gate = {
            "overall": "pass",
            "absolute_floors_passed": True,
            "scope": "final",
            "candidate_id": f"candidate-c{concurrency}",
        }
        gate_path = root / name
        gate_path.write_text(json.dumps(gate))
        audit = {
            "schema": "minicpmo-frozen-path-aware-quality-gate-audit/v1",
            "gate_output": {"sha256": verifier.sha256(gate_path)},
            "baseline": {"ok": True, "problems": []},
            "baseline_dref": {"ok": True, "problems": []},
            "candidate_report": {"ok": True, "problems": []},
            "candidate_gate": {
                "ok": not bad_audit,
                "problems": [] if not bad_audit else ["source verification failed"],
                "source_cells": ["seed0", "seed1", "seed2"],
            },
        }
        (root / verifier.QUALITY_GATE_AUDIT_FILES[concurrency]).write_text(json.dumps(audit))


def test_quality_status_requires_complete_path_aware_evidence(tmp_path):
    verifier = _load_verifier()
    assert verifier.quality_status(tmp_path)["status"] == "pending"
    (tmp_path / verifier.QUALITY_GATE_FILES["1"]).write_text("{}")
    assert verifier.quality_status(tmp_path)["status"] == "incomplete"


def test_quality_status_accepts_only_passing_gates_and_audits(tmp_path):
    verifier = _load_verifier()
    _write_quality_fixture(tmp_path, verifier)
    result = verifier.quality_status(tmp_path)
    assert result["status"] == "passed" and result["passed"] is True


def test_quality_status_fails_closed_on_path_aware_audit_failure(tmp_path):
    verifier = _load_verifier()
    _write_quality_fixture(tmp_path, verifier, bad_audit=True)
    result = verifier.quality_status(tmp_path)
    assert result["status"] == "failed" and result["passed"] is False


def test_engine_tree_rejects_forbidden_runtime_imports(tmp_path):
    verifier = _load_verifier()
    archive = tmp_path / "source.tar.gz"
    files = {
        "engine/allowed.py": b"import torch\n",
        "engine/direct.py": b"import vllm_omni\nfrom sglang import Engine\n",
        "engine/dynamic.py": b"import importlib\nimportlib.import_module('vllm.worker')\n",
    }
    with tarfile.open(archive, "w:gz") as bundle:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    count, _digest, cache, syntax, forbidden = verifier.engine_tree(archive)
    assert count == 3 and cache == [] and syntax == []
    assert forbidden == [
        "engine/direct.py: sglang",
        "engine/direct.py: vllm_omni",
        "engine/dynamic.py: vllm",
    ]
