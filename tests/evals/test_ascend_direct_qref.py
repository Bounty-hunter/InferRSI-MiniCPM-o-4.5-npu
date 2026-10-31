"""Ascend single-baseline quality evidence, independent of the L20X campaign."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.speech import qref as sq
from evals.speech import verify_report as vrm
from evals.speech.suite import Collection, GenRow
from tools.speech_quality_bundle import compare_manifest_to_collection, export_bundle, sha256, validate_manifest


GPU = "Ascend 910B1"
UUID = "ascend-pci-0000:c1:00.0"
MODEL = {"repo": "openbmb/MiniCPM-o-4_5", "revision": "503e754207c94da6bb26850b4469f367c9ea3582"}
DATASET = {"repo": "zhaochenyang20/seed-tts-eval", "revision": "8f5e1aa2a35d42f42e940074c1983358b9491f89"}


def acceptance(tmp_path: Path) -> Path:
    p = tmp_path / "acceptance.json"
    p.write_text(json.dumps({
        "schema": 1,
        "model": MODEL,
        "hardware": {"accelerator": "Ascend910B1", "same_physical_device_required": True},
        "baseline": {"name": "vllm-omni"},
        "candidate": {"name": "minicpmo-ascend-engine"},
        "concurrencies": [1, 16, 32],
        "required_gates": {"speech_quality_gate": "pass"},
    }))
    return p


def metrics():
    return [{"wer_corpus": 0.0, "catastrophic_count": 0, "sim_mean": 60.0, "utmos_mean": 4.3,
             "silence_rate": 0.0, "repetition_rate": 0.0, "truncation_rate": 0.0} for _ in range(3)]


def fake_report(path: Path, adapter: str, concurrency: int, digest: str):
    path.write_text("{}")
    path.with_suffix(".rows.jsonl").write_text("{}\n")
    candidate = {"adapter": adapter, "candidate_id": f"{adapter}:{digest[:12]}", "digest": digest}
    identity = {"gpu_sku": GPU, "gpu_driver": "25.5.0", "gpu_memory_mib": 65536, "concurrency": concurrency,
                "id_set": "full", "measured_ids_sha256": "ids", "dataset": DATASET, "model": MODEL,
                "seed_policy": "request-id", "seeds": [0, 1, 2], "rows_per_seed": {"0": 1088, "1": 1088, "2": 1088},
                "evaluator_digest": "scorer", "scope": "final"}
    report = {"scope": "final", "generation": {"candidate": candidate, "concurrency": concurrency, "model": MODEL},
              "rows_sha256": "rows", "d_ref": {"path": "dref"},
              "external_scoring": {"receipt": {"sha256": "receipt"}},
              "evaluator": {"digest": "scorer"},
              "seeds": {str(s): {"gpu_uuid": UUID, "cell": f"cell{s}"} for s in range(3)}}
    summary = {"seed_indices": [0, 1, 2], "per_seed": metrics(), "absolute_floors_pooled": {"passed": True}}
    return SimpleNamespace(report=report, identity=identity, summary=summary,
                           rows=[SimpleNamespace(truncation=False)], problems=[], ok=True)


def test_direct_qref_rebuild_and_gate(tmp_path, monkeypatch):
    plan = acceptance(tmp_path)
    refs = {}
    for c in (1, 32):
        p = tmp_path / f"baseline_c{c}.json"
        refs[str(p)] = fake_report(p, "vllm-omni", c, "b" * 64)
    engine = tmp_path / "engine_c32.json"
    refs[str(engine)] = fake_report(engine, "engine", 32, "e" * 64)
    monkeypatch.setattr(sq, "_verified", lambda p, check_sources=True: refs[str(Path(p))])
    qref = tmp_path / "qref.json"
    sq.freeze_direct(GPU, [str(tmp_path / "baseline_c1.json"), str(tmp_path / "baseline_c32.json")], plan, "scorer", qref,
                     check_sources=False)
    assert sq.verify_qref(str(qref), check_sources=False)["selection"]["physical_gpu_uuid"] == UUID
    result = sq.gate(str(qref), str(engine), "e" * 64, check_sources=False)
    assert result["overall"] == "pass" and result["absolute_floors_passed"]
    edited = json.loads(qref.read_text())
    edited["values"]["utmos_mean"] = 1.0
    bad = tmp_path / "edited.json"
    bad.write_text(json.dumps(edited))
    with pytest.raises(sq.QRefError):
        sq.verify_qref(str(bad), check_sources=False)


def _cell(tmp_path: Path, seed: int, adapter: str = "vllm-omni") -> tuple[Path, dict]:
    cell = tmp_path / f"cell_{seed}_{adapter}"
    (cell / "run1").mkdir(parents=True)
    (cell / "run1" / "requests.jsonl").write_text(
        "".join(json.dumps({"request_id": rid, "duration_s": dur}) + "\n" for rid, dur in (("r1", 2.0), ("r2", 3.0))))
    (cell / "result.json").write_text(json.dumps({"experiment_digest": f"exp{seed}", "artifacts": {"run1": {"sha": f"a{seed}"}}}))
    spec = {"kind": "quality", "id_set": "full", "k": 1, "seed_index": seed, "concurrency": 1,
            "model": MODEL, "dataset": DATASET, "measured_ids": ["r1", "r2"], "sampling": {"seed_policy": "request-id"},
            "candidate": {"adapter": adapter, "candidate_id": f"{adapter}:base", "digest": "b" * 64},
            "hardware": {"gpu_sku": GPU, "driver": "25.5.0", "memory_mib": 65536, "gpu_uuid": UUID}}
    return cell, spec


def test_direct_dref_is_acceptance_and_source_bound(tmp_path, monkeypatch):
    plan = acceptance(tmp_path)
    made = [_cell(tmp_path, s) for s in range(3)]
    specs = {str(p): spec for p, spec in made}
    monkeypatch.setattr(vrm, "verify_cell", lambda p: SimpleNamespace(ok=True, problems=[], spec=specs[str(Path(p))]))
    out = tmp_path / "dref.json"
    doc = vrm.build_direct_dref([p for p, _ in made], out, plan)
    assert doc["schema"] == vrm.DIRECT_DREF_SCHEMA and len(doc["rows"]) == 2
    problems, rows = vrm.verify_dref(out)
    assert not problems and rows == {"r1": 2.0, "r2": 3.0}
    tampered = json.loads(out.read_text())
    tampered["rows"]["r1"] = 9.0
    bad = tmp_path / "dref_bad.json"
    bad.write_text(json.dumps(tampered))
    assert vrm.verify_dref(bad)[0]
    wrong = [_cell(tmp_path / "wrong", s, adapter="engine") for s in range(3)]
    specs.update({str(p): spec for p, spec in wrong})
    with pytest.raises(vrm.ReportError):
        vrm.build_direct_dref([p for p, _ in wrong], tmp_path / "wrong.json", plan)


def test_portable_bundle_and_external_receipt_are_hash_bound(tmp_path):
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    ref = seed_root / "ref.wav"
    wav = tmp_path / "generated.wav"
    ref.write_bytes(b"reference")
    wav.write_bytes(b"generated")
    row = GenRow("0:r1", "r1", 0, "target", str(ref), str(wav), "target", 1.0)
    col = Collection([row], {"candidate_id": "engine:x", "concurrency": 1}, {"0": {"cell": "c"}})
    dref = tmp_path / "dref.json"
    dref.write_text("{}")
    manifest = export_bundle(col, dref, seed_root, tmp_path / "bundle")
    _, loaded = validate_manifest(manifest, seed_root)
    assert [r.key for r in loaded] == ["0:r1"]
    assert not compare_manifest_to_collection(manifest, col, dref, seed_root)

    receipt = {
        "schema": "speech-quality-scored/v1",
        "bundle": {"sha256": sha256(manifest)},
        "rows": 1,
        "row_keys_sha256": __import__("hashlib").sha256(b"0:r1").hexdigest(),
        "evaluator": {"digest": "scorer"},
        "execution_evidence": {"asr": {}},
        "tasks": {"asr": {"rows": {"0:r1": {"hyp_text": "h", "ref_norm": "a", "hyp_norm": "a", "wer": 0.0,
                                                       "substitutions": 0, "deletions": 0, "insertions": 0, "hits": 1}}},
                  "sim": {"rows": {"0:r1": 60.0}}, "utmos": {"rows": {"0:r1": 4.3}}},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    quality_row = SimpleNamespace(seed_index=0, row_id="r1", hyp_text="h", ref_norm="a", hyp_norm="a", wer=0.0,
                                  substitutions=0, deletions=0, insertions=0, hits=1, sim=60.0, utmos=4.3)
    report = {"evaluator": receipt["evaluator"], "execution_evidence": receipt["execution_evidence"],
              "external_scoring": {"receipt": {"path": str(receipt_path), "sha256": sha256(receipt_path)},
                                     "bundle": {"path": str(manifest), "sha256": sha256(manifest)}}}
    assert not vrm._external_scoring_problems(report, [quality_row])
    receipt["tasks"]["utmos"]["rows"]["0:r1"] = 1.0
    receipt_path.write_text(json.dumps(receipt))
    report["external_scoring"]["receipt"]["sha256"] = sha256(receipt_path)
    assert vrm._external_scoring_problems(report, [quality_row])

    (manifest.parent / json.loads(manifest.read_text())["rows"][0]["wav_file"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        validate_manifest(manifest, seed_root)
