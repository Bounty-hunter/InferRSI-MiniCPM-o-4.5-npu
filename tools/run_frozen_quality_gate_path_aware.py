#!/usr/bin/env python3
"""Run the frozen speech gate across independently rooted measured stacks.

The formal baseline and candidate cells intentionally record different absolute
``qwen3_omni`` roots.  Each side is therefore source-verified in its own fresh
Python process.  Only after those checks pass do we call the frozen Q_ref gate
with source traversal disabled; the candidate Q_ref declaration is checked
explicitly against every already source-verified experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType


EXPECTED = {
    "evals/speech/qref.py": "9fb0c2f91e5d0911c03edec52a0a461237f2d1163e0dbd141f29104a69dea523",
    "evals/speech/verify_report.py": "23b80087bda8d406c8754d1c81bac4527b4ca478a0b26c76f5708b74db32c742",
    "tools/speech_quality_bundle.py": "0d106ab190110505ba211562de5ff9dd1c3a15b90f5d041883ee7bf6a330d151",
    "tools/export_speech_quality_bundle.py": "99f8ac2229253bad5736db8d4535a214dc04612b7ae4e9db05196145c2375893",
    "tools/score_speech_quality_bundle.py": "6be026e74bae6e5c8b1b9bb2452484265101f9c3e8a42b86a683db511426c61e",
    "tools/assemble_speech_quality_report.py": "941aa132e907508111ef14cc754ec6828a1443bd84a24cb6b2e6b5a3bf73c80b",
}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_paths(freeze_root: Path, quality_freeze: Path) -> dict[str, Path]:
    freeze_doc = json.loads(quality_freeze.read_text())
    declared = freeze_doc.get("validator_files") or {}
    if declared != EXPECTED:
        raise RuntimeError(f"quality freeze names different validator files: {declared}")
    paths = {rel: freeze_root / rel for rel in EXPECTED}
    bad = {
        rel: {"expected": EXPECTED[rel], "actual": sha256(path) if path.is_file() else None}
        for rel, path in paths.items()
        if not path.is_file() or sha256(path) != EXPECTED[rel]
    }
    if bad:
        raise RuntimeError(f"frozen quality validator files are missing or changed: {bad}")
    return paths


def load_validator(freeze_root: Path, source_root: Path, quality_freeze: Path) -> tuple[ModuleType, ModuleType]:
    paths = checked_paths(freeze_root.resolve(), quality_freeze.resolve())
    source_root = source_root.resolve()
    if not (source_root / "bench" / "__init__.py").is_file():
        raise RuntimeError(f"source root has no bench package: {source_root}")
    sys.path.insert(0, str(source_root))

    # Bind package __path__ to the measured checkout before qref.py temporarily
    # prepends its own frozen file location to sys.path.
    import bench  # noqa: F401
    import evals
    import evals.speech
    import tools  # noqa: F401

    verify_report = load_module("evals.speech.verify_report", paths["evals/speech/verify_report.py"])
    setattr(evals.speech, "verify_report", verify_report)
    qref = load_module("evals.speech.qref", paths["evals/speech/qref.py"])
    setattr(evals.speech, "qref", qref)

    freeze_text = str(freeze_root.resolve())
    sys.path[:] = [entry for entry in sys.path if str(Path(entry).resolve()) != freeze_text]
    sys.path.insert(0, str(source_root))
    return qref, verify_report


def report_cells(report_path: Path) -> list[Path]:
    report = json.loads(report_path.read_text())
    return [Path(item["cell"]) for item in (report.get("seeds") or {}).values()]


def qref_reports(qref_path: Path) -> list[Path]:
    qref = json.loads(qref_path.read_text())
    return [Path(item["quality"]) for item in (qref.get("references") or {}).values()]


def quality_dref(quality_path: Path) -> Path:
    report = json.loads(quality_path.read_text())
    record = report.get("d_ref") or {}
    path = Path(record.get("path", ""))
    if not path.is_file() or sha256(path) != record.get("sha256"):
        raise RuntimeError(f"quality report names a missing or changed D_ref: {record}")
    return path.resolve()


def dref_cells(dref_path: Path) -> list[Path]:
    doc = json.loads(dref_path.read_text())
    return [Path(item["cell"]) for item in (doc.get("sources") or [])]


def dref_expectation(quality_path: Path) -> dict:
    report = json.loads(quality_path.read_text())
    generation = report.get("generation") or {}
    sidecar = quality_path.with_suffix(".rows.jsonl")
    rows = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
    return {
        "role": "truncation",
        "concurrency": generation.get("concurrency"),
        "gpu_class": {key: generation.get(key) for key in ("gpu_sku", "gpu_driver", "gpu_memory_mib")},
        "model": generation.get("model"),
        "dataset": generation.get("dataset"),
        "seed_policy": generation.get("seed_policy"),
        "rows": sorted({row["row_id"] for row in rows}),
    }


def declared_repo_path(cell: Path, name: str) -> str | None:
    declaration = json.loads((cell / "experiment.json").read_text())
    manifest_repos = (declaration.get("manifest") or {}).get("repos") or {}
    fingerprint_repos = ((declaration.get("experiment") or {}).get("fingerprints") or {}).get("repos") or {}
    value = (manifest_repos.get(name) or {}).get("path") or (fingerprint_repos.get(name) or {}).get("path")
    return str(Path(value).resolve()) if value else None


def set_declared_repo_environment(cells: list[Path]) -> dict[str, str]:
    mapping = {"vllm-omni": "OMNI_VLLM_REPO", "sglang-omni": "OMNI_SGLANG_REPO"}
    result = {}
    for repo, env_name in mapping.items():
        paths = {path for cell in cells if (path := declared_repo_path(cell, repo))}
        if len(paths) > 1:
            raise RuntimeError(f"source cells declare several {repo} roots: {sorted(paths)}")
        if paths:
            result[env_name] = next(iter(paths))
            os.environ[env_name] = result[env_name]
    return result


def install_verified_dref_adapter(verify_report: ModuleType) -> None:
    """Avoid cross-root traversal after a separate baseline D_ref verification.

    The frozen report verifier still checks the recorded D_ref hash before this
    hook, then uses the returned rows to re-check every candidate row and all
    derived truncation flags.  Only source traversal is delegated to the D_ref
    worker running under the baseline source root.
    """

    def verified_rows(path: str | Path, expect: dict | None = None, check_sources: bool = True):
        doc = json.loads(Path(path).read_text())
        problems = []
        if doc.get("schema") != verify_report.DIRECT_DREF_SCHEMA or doc.get("role") != "truncation":
            problems.append(f"not a {verify_report.DIRECT_DREF_SCHEMA} truncation reference")
        rows = doc.get("rows") or {}
        if not rows or any(not (math.isfinite(float(value)) and float(value) > 0) for value in rows.values()):
            problems.append("duration reference rows are empty, non-positive, or non-finite")
        if expect:
            for key in ("role", "concurrency", "gpu_class", "model", "dataset", "seed_policy"):
                if key in expect and doc.get(key) != expect[key]:
                    problems.append(f"reference {key}={doc.get(key)!r}, required {expect[key]!r}")
            missing = sorted(set(expect.get("rows") or []) - set(rows))
            if missing:
                problems.append(f"reference lacks {len(missing)} required rows (e.g. {missing[:3]})")
        return problems, (dict(rows) if not problems else {})

    verify_report.verify_dref = verified_rows


def report_worker(args: argparse.Namespace) -> int:
    quality = Path(args.quality).resolve()
    cells = report_cells(quality)
    repo_env = set_declared_repo_environment(cells)
    _, verify_report = load_validator(Path(args.freeze_root), Path(args.source_root), Path(args.quality_freeze))
    install_verified_dref_adapter(verify_report)
    verified = verify_report.verify_report(quality, check_sources=True, check_audio=True)
    payload = {
        "worker": "report",
        "quality": str(quality),
        "source_root": str(Path(args.source_root).resolve()),
        "repo_environment": repo_env,
        "ok": verified.ok,
        "problems": verified.problems,
        "candidate_id": (verified.report.get("generation") or {}).get("candidate_id"),
        "candidate_digest": (verified.report.get("generation") or {}).get("candidate_digest"),
        "concurrency": (verified.report.get("generation") or {}).get("concurrency"),
        "source_cells": [str(cell.resolve()) for cell in cells],
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if verified.ok else 1


def dref_worker(args: argparse.Namespace) -> int:
    quality = Path(args.quality).resolve()
    dref_path = quality_dref(quality)
    cells = dref_cells(dref_path)
    repo_env = set_declared_repo_environment(cells)
    _, verify_report = load_validator(Path(args.freeze_root), Path(args.source_root), Path(args.quality_freeze))
    problems, rows = verify_report.verify_dref(dref_path, dref_expectation(quality), check_sources=True)
    payload = {
        "worker": "dref",
        "quality": str(quality),
        "dref": str(dref_path),
        "dref_sha256": sha256(dref_path),
        "source_root": str(Path(args.source_root).resolve()),
        "repo_environment": repo_env,
        "source_cells": [str(cell.resolve()) for cell in cells],
        "rows": len(rows),
        "ok": not problems,
        "problems": problems,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if not problems else 1


def qref_worker(args: argparse.Namespace) -> int:
    qref_path = Path(args.qref).resolve()
    reports = qref_reports(qref_path)
    cells = [cell for report in reports for cell in report_cells(report)]
    repo_env = set_declared_repo_environment(cells)
    qref, _ = load_validator(Path(args.freeze_root), Path(args.source_root), Path(args.quality_freeze))
    doc = qref.verify_qref(qref_path, check_sources=True)
    payload = {
        "worker": "qref",
        "qref": str(qref_path),
        "qref_sha256": sha256(qref_path),
        "source_root": str(Path(args.source_root).resolve()),
        "repo_environment": repo_env,
        "ok": True,
        "sources": doc.get("sources"),
        "selection": doc.get("selection"),
        "source_reports": [str(report.resolve()) for report in reports],
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def binding_problems(verified, qref_path: Path, headline_digest: str) -> list[str]:
    problems = []
    generation = verified.report.get("generation") or {}
    if generation.get("candidate_digest") != headline_digest:
        problems.append("candidate report does not name the headline digest")
    qref_sha = sha256(qref_path)
    for seed, source in (verified.report.get("seeds") or {}).items():
        cell = Path(source["cell"])
        declaration = json.loads((cell / "experiment.json").read_text())
        spec = declaration.get("experiment") or {}
        declared = (((spec.get("frozen_refs") or {}).get("speech_qref") or {}).get("sha256"))
        if declared != qref_sha:
            problems.append(f"seed {seed}: {cell} declares speech_qref {declared!r}, expected {qref_sha}")
    return problems


def gate_worker(args: argparse.Namespace) -> int:
    quality = Path(args.quality).resolve()
    qref_path = Path(args.qref).resolve()
    cells = report_cells(quality)
    repo_env = set_declared_repo_environment(cells)
    qref, verify_report = load_validator(Path(args.freeze_root), Path(args.source_root), Path(args.quality_freeze))
    install_verified_dref_adapter(verify_report)
    verified = verify_report.verify_report(quality, check_sources=True, check_audio=True)
    problems = list(verified.problems)
    if verified.ok:
        problems += binding_problems(verified, qref_path, args.headline_digest)
    result = None
    if not problems:
        # Baseline source traversal was completed by the independent qref worker;
        # candidate traversal and the frozen Q_ref binding were completed above.
        result = qref.gate(qref_path, quality, args.headline_digest, check_sources=False)
    payload = {
        "worker": "gate",
        "quality": str(quality),
        "qref": str(qref_path),
        "source_root": str(Path(args.source_root).resolve()),
        "repo_environment": repo_env,
        "source_cells": [str(cell.resolve()) for cell in cells],
        "ok": not problems and result is not None,
        "problems": problems,
        "gate": result,
    }
    print(json.dumps(payload, sort_keys=True))
    passed = bool(result and result.get("overall") == "pass" and result.get("absolute_floors_passed"))
    return 0 if not problems and passed else 1


def run_worker(kind: str, args: argparse.Namespace) -> tuple[int, dict, str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        kind,
        "--freeze-root",
        str(Path(args.freeze_root).resolve()),
        "--quality-freeze",
        str(Path(args.quality_freeze).resolve()),
        "--source-root",
        str(Path(args.baseline_source_root if kind in ("qref", "dref") else args.candidate_source_root).resolve()),
    ]
    if kind == "qref":
        command += ["--qref", str(Path(args.qref).resolve())]
    else:
        command += ["--quality", str(Path(args.quality).resolve())]
        if kind == "gate":
            command += ["--qref", str(Path(args.qref).resolve()), "--headline-digest", args.headline_digest]
    completed = subprocess.run(command, text=True, capture_output=True, env=os.environ.copy())
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{kind} worker did not return JSON: returncode={completed.returncode}, "
            f"stdout={completed.stdout[-2000:]!r}, stderr={completed.stderr[-4000:]!r}"
        ) from exc
    return completed.returncode, payload, completed.stderr


def write_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def top_level_gate(args: argparse.Namespace) -> int:
    freeze_doc = json.loads(Path(args.quality_freeze).read_text())
    expected_baseline = Path(freeze_doc["baseline_source"]["container_root"]).resolve()
    expected_candidate = Path(freeze_doc["candidate_source"]["container_root"]).resolve()
    if Path(args.baseline_source_root).resolve() != expected_baseline:
        raise RuntimeError(f"baseline source root differs from the quality freeze: {expected_baseline}")
    if Path(args.candidate_source_root).resolve() != expected_candidate:
        raise RuntimeError(f"candidate source root differs from the quality freeze: {expected_candidate}")

    qref_code, qref_audit, qref_stderr = run_worker("qref", args)
    if qref_code != 0 or not qref_audit.get("ok"):
        raise RuntimeError(f"baseline Q_ref source verification failed: {qref_audit}, stderr={qref_stderr[-4000:]}")
    dref_code, dref_audit, dref_stderr = run_worker("dref", args)
    if dref_code != 0 or not dref_audit.get("ok"):
        raise RuntimeError(f"baseline D_ref source verification failed: {dref_audit}, stderr={dref_stderr[-4000:]}")
    report_code, report_audit, report_stderr = run_worker("report", args)
    if report_code != 0 or not report_audit.get("ok"):
        raise RuntimeError(f"candidate report source verification failed: {report_audit}, stderr={report_stderr[-4000:]}")
    gate_code, gate_audit, gate_stderr = run_worker("gate", args)
    if gate_audit.get("gate") is None:
        raise RuntimeError(f"candidate gate preconditions failed: {gate_audit}, stderr={gate_stderr[-4000:]}")

    gate_result = gate_audit["gate"]
    write_new(Path(args.out), gate_result)
    audit = {
        "schema": "minicpmo-frozen-path-aware-quality-gate-audit/v1",
        "quality_freeze": {"path": str(Path(args.quality_freeze).resolve()), "sha256": sha256(args.quality_freeze)},
        "freeze_root": str(Path(args.freeze_root).resolve()),
        "frozen_files": EXPECTED,
        "baseline": qref_audit,
        "baseline_dref": dref_audit,
        "candidate_report": report_audit,
        "candidate_gate": {key: value for key, value in gate_audit.items() if key != "gate"},
        "gate_output": {"path": str(Path(args.out).resolve()), "sha256": sha256(args.out)},
    }
    write_new(Path(args.audit_out), audit)
    print(json.dumps({"out": str(Path(args.out).resolve()), "audit": str(Path(args.audit_out).resolve()), **gate_result}, indent=2))
    return gate_code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-root", required=True)
    ap.add_argument("--quality-freeze", required=True)
    ap.add_argument("--source-root")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--worker", choices=("report", "qref", "dref", "gate"))
    ap.add_argument("--qref")
    ap.add_argument("--quality")
    ap.add_argument("--headline-digest")
    ap.add_argument("--baseline-source-root")
    ap.add_argument("--candidate-source-root")
    ap.add_argument("--out")
    ap.add_argument("--audit-out")
    args = ap.parse_args()

    if args.self_check:
        if not args.source_root:
            ap.error("--self-check requires --source-root")
        qref, verify_report = load_validator(Path(args.freeze_root), Path(args.source_root), Path(args.quality_freeze))
        import bench.cli
        import bench.verify
        import evals.speech.suite
        print(json.dumps({
            "schema": "minicpmo-frozen-quality-validator-self-check/v1",
            "source_root": str(Path(args.source_root).resolve()),
            "freeze_root": str(Path(args.freeze_root).resolve()),
            "frozen_files": EXPECTED,
            "loaded": {
                "qref": qref.__file__,
                "verify_report": verify_report.__file__,
                "bench_cli": bench.cli.__file__,
                "bench_verify": bench.verify.__file__,
                "speech_suite": evals.speech.suite.__file__,
                "bench_root": str(bench.cli.ROOT),
            },
        }, indent=2, sort_keys=True))
        return 0
    if args.worker == "report":
        return report_worker(args)
    if args.worker == "qref":
        return qref_worker(args)
    if args.worker == "dref":
        return dref_worker(args)
    if args.worker == "gate":
        return gate_worker(args)
    required = ("qref", "quality", "headline_digest", "baseline_source_root", "candidate_source_root", "out", "audit_out")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        ap.error(f"top-level gate requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
    return top_level_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
