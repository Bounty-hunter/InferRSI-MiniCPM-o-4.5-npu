#!/usr/bin/env python3
"""Prepare hash-bound r149 speech artifacts without changing the frozen gate.

Each invocation binds one measured source root (baseline or candidate), loads
the frozen r34 quality modules, and writes a new audit beside the requested
artifact.  Candidate-side bundle/report assembly may consume the baseline
D_ref only after a baseline-root verification audit has been supplied.  The
final path-aware gate still repeats that baseline verification independently.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


DREF_AUDIT_SCHEMA = "minicpmo-frozen-dref-verification-audit/v1"
PIPELINE_AUDIT_SCHEMA = "minicpmo-frozen-quality-pipeline-audit/v1"


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


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json_new(path: str | Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def file_record(path: str | Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


@dataclass
class Context:
    runtime_path: Path
    runtime: dict
    quality_freeze_path: Path
    quality_freeze: dict
    gate_runner_path: Path
    gate_runner: ModuleType
    freeze_root: Path
    source_root: Path
    side: str
    repo_environment: dict[str, str]
    qref: ModuleType
    verify_report: ModuleType
    frozen_paths: dict[str, Path]

    def base_audit(self, operation: str) -> dict:
        return {
            "schema": PIPELINE_AUDIT_SCHEMA,
            "operation": operation,
            "created_unix": time.time(),
            "side": self.side,
            "source_root": str(self.source_root),
            "repo_environment": self.repo_environment,
            "runtime_manifest": file_record(self.runtime_path),
            "quality_freeze": file_record(self.quality_freeze_path),
            "gate_runner": file_record(self.gate_runner_path),
            "freeze_root": str(self.freeze_root),
            "frozen_files": {rel: sha256(path) for rel, path in sorted(self.frozen_paths.items())},
        }


def _expected_source_root(freeze: dict, side: str) -> Path:
    key = "baseline_source" if side == "baseline" else "candidate_source"
    return Path(freeze[key]["container_root"]).resolve()


def prepare_context(args: argparse.Namespace, cells: list[Path]) -> Context:
    runtime_path = Path(args.runtime_manifest).resolve()
    runtime = read_json(runtime_path)
    if runtime.get("schema") != "minicpmo-ascend-quality-validator-runtime/v1":
        raise RuntimeError(f"not an r149 quality runtime manifest: {runtime_path}")

    quality_freeze_path = Path(args.quality_freeze).resolve()
    expected_freeze = runtime.get("quality_freeze") or {}
    if sha256(quality_freeze_path) != expected_freeze.get("sha256"):
        raise RuntimeError("quality freeze hash differs from the runtime manifest")
    quality_freeze = read_json(quality_freeze_path)

    gate_runner_path = Path(args.gate_runner).resolve()
    expected_runner = runtime.get("runner") or {}
    if sha256(gate_runner_path) != expected_runner.get("sha256"):
        raise RuntimeError("path-aware gate runner hash differs from the runtime manifest")
    gate_runner = load_module("r149_path_aware_gate_runtime", gate_runner_path)

    freeze_root = Path(args.freeze_root).resolve()
    if freeze_root != Path(runtime["freeze_root"]).resolve():
        raise RuntimeError(f"freeze root differs from the runtime manifest: {runtime['freeze_root']}")
    frozen_paths = gate_runner.checked_paths(freeze_root, quality_freeze_path)
    actual_frozen = {rel: sha256(path) for rel, path in frozen_paths.items()}
    if actual_frozen != runtime.get("frozen_files"):
        raise RuntimeError("runtime manifest and checked frozen files disagree")

    source_root = _expected_source_root(quality_freeze, args.side)
    if args.source_root and Path(args.source_root).resolve() != source_root:
        raise RuntimeError(f"{args.side} source root differs from the quality freeze: {source_root}")
    repo_environment = gate_runner.set_declared_repo_environment(cells) if cells else {}
    qref, verify_report = gate_runner.load_validator(freeze_root, source_root, quality_freeze_path)
    return Context(
        runtime_path=runtime_path,
        runtime=runtime,
        quality_freeze_path=quality_freeze_path,
        quality_freeze=quality_freeze,
        gate_runner_path=gate_runner_path,
        gate_runner=gate_runner,
        freeze_root=freeze_root,
        source_root=source_root,
        side=args.side,
        repo_environment=repo_environment,
        qref=qref,
        verify_report=verify_report,
        frozen_paths=frozen_paths,
    )


def dref_expectation_from_doc(path: Path) -> dict:
    doc = read_json(path)
    return {
        "role": doc.get("role"),
        "concurrency": doc.get("concurrency"),
        "gpu_class": doc.get("gpu_class"),
        "model": doc.get("model"),
        "dataset": doc.get("dataset"),
        "seed_policy": doc.get("seed_policy"),
        "rows": sorted((doc.get("rows") or {}).keys()),
    }


def write_dref_audit(
    ctx: Context,
    dref: Path,
    audit_out: Path,
    operation: str,
    source_cells: list[Path],
) -> dict:
    problems, rows = ctx.verify_report.verify_dref(
        dref, dref_expectation_from_doc(dref), check_sources=True
    )
    audit = {
        **ctx.base_audit(operation),
        "schema": DREF_AUDIT_SCHEMA,
        "dref": file_record(dref),
        "source_cells": [str(path.resolve()) for path in source_cells],
        "rows": len(rows),
        "ok": not problems,
        "problems": problems,
    }
    write_json_new(audit_out, audit)
    if problems:
        raise RuntimeError(f"D_ref verification failed: {problems[:4]}")
    return audit


def require_baseline_dref_audit(ctx: Context, dref: Path, audit_path: str | Path) -> dict:
    audit_path = Path(audit_path).resolve()
    audit = read_json(audit_path)
    expected_root = _expected_source_root(ctx.quality_freeze, "baseline")
    failures = []
    if audit.get("schema") != DREF_AUDIT_SCHEMA or not audit.get("ok"):
        failures.append("not a successful frozen D_ref verification audit")
    if Path(audit.get("source_root", "")).resolve() != expected_root:
        failures.append("D_ref was not verified under the frozen baseline source root")
    record = audit.get("dref") or {}
    if Path(record.get("path", "")).resolve() != dref.resolve() or record.get("sha256") != sha256(dref):
        failures.append("D_ref path or hash differs from its verification audit")
    if (audit.get("quality_freeze") or {}).get("sha256") != sha256(ctx.quality_freeze_path):
        failures.append("D_ref audit names another quality freeze")
    if (audit.get("runtime_manifest") or {}).get("sha256") != sha256(ctx.runtime_path):
        failures.append("D_ref audit names another runtime manifest")
    if audit.get("frozen_files") != {rel: sha256(path) for rel, path in sorted(ctx.frozen_paths.items())}:
        failures.append("D_ref audit names another frozen validator set")
    if failures:
        raise RuntimeError(f"cannot consume cross-root D_ref: {failures}")
    return {"audit": file_record(audit_path), "verification": audit}


def load_frozen_tool(ctx: Context, rel: str, name: str) -> ModuleType:
    if rel != "tools/speech_quality_bundle.py":
        # Bind the canonical dependency before loading export/assemble modules.
        if "tools.speech_quality_bundle" not in sys.modules:
            bundle = load_module(
                "tools.speech_quality_bundle",
                ctx.frozen_paths["tools/speech_quality_bundle.py"],
            )
            import tools

            setattr(tools, "speech_quality_bundle", bundle)
    return load_module(name, ctx.frozen_paths[rel])


def build_dref(args: argparse.Namespace) -> int:
    if args.side != "baseline":
        raise RuntimeError("a direct D_ref can only be built under the baseline source root")
    cells = [Path(path).resolve() for path in args.cell]
    ctx = prepare_context(args, cells)
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx.verify_report.build_direct_dref(cells, out, Path(args.acceptance).resolve())
    audit = write_dref_audit(ctx, out, Path(args.audit_out).resolve(), "build-dref", cells)
    print(json.dumps({"dref": file_record(out), "audit": file_record(args.audit_out), "rows": audit["rows"]}, indent=2))
    return 0


def verify_dref(args: argparse.Namespace) -> int:
    if args.side != "baseline":
        raise RuntimeError("full D_ref source verification must run under the baseline source root")
    dref = Path(args.d_ref).resolve()
    cells = [Path(item["cell"]).resolve() for item in (read_json(dref).get("sources") or [])]
    ctx = prepare_context(args, cells)
    audit = write_dref_audit(ctx, dref, Path(args.audit_out).resolve(), "verify-dref", cells)
    print(json.dumps({"dref": file_record(dref), "audit": file_record(args.audit_out), "rows": audit["rows"]}, indent=2))
    return 0


def install_cross_root_adapter_if_needed(
    ctx: Context, dref: Path, verified_dref_audit: str | None
) -> dict | None:
    if ctx.side == "baseline":
        return None
    if not verified_dref_audit:
        raise RuntimeError("candidate-side D_ref consumption requires --verified-dref-audit")
    receipt = require_baseline_dref_audit(ctx, dref, verified_dref_audit)
    ctx.gate_runner.install_verified_dref_adapter(ctx.verify_report)
    return receipt


def export_bundle(args: argparse.Namespace) -> int:
    cells = [Path(path).resolve() for path in args.cell]
    dref = Path(args.d_ref).resolve()
    ctx = prepare_context(args, cells)
    dref_receipt = install_cross_root_adapter_if_needed(ctx, dref, args.verified_dref_audit)
    module = load_frozen_tool(ctx, "tools/export_speech_quality_bundle.py", "r149_frozen_export_bundle")
    col = module.collect(cells)
    module.load_dref(str(dref), col)
    manifest = module.export_bundle(col, dref, Path(args.seed_tts_root).resolve(), Path(args.out).resolve())
    audit = {
        **ctx.base_audit("export-bundle"),
        "inputs": {"dref": file_record(dref), "cells": [str(path) for path in cells]},
        "verified_dref_audit": dref_receipt,
        "output": {"manifest": file_record(manifest), "rows": len(col.rows)},
    }
    write_json_new(args.audit_out, audit)
    print(json.dumps({"manifest": file_record(manifest), "audit": file_record(args.audit_out), "rows": len(col.rows)}, indent=2))
    return 0


def assemble_report(args: argparse.Namespace) -> int:
    cells = [Path(path).resolve() for path in args.cell]
    dref = Path(args.d_ref).resolve()
    ctx = prepare_context(args, cells)
    dref_receipt = install_cross_root_adapter_if_needed(ctx, dref, args.verified_dref_audit)
    module = load_frozen_tool(ctx, "tools/assemble_speech_quality_report.py", "r149_frozen_assemble_report")
    argv = [
        "assemble_speech_quality_report.py",
        *[value for cell in cells for value in ("--cell", str(cell))],
        "--d-ref", str(dref),
        "--manifest", str(Path(args.manifest).resolve()),
        "--scored", str(Path(args.scored).resolve()),
        "--seed-tts-root", str(Path(args.seed_tts_root).resolve()),
        "--expected-evaluator-digest", args.expected_evaluator_digest,
        "--out", str(Path(args.out).resolve()),
    ]
    previous = sys.argv
    try:
        sys.argv = argv
        code = module.main()
    finally:
        sys.argv = previous
    if code != 0:
        raise RuntimeError(f"frozen report assembler returned {code}")
    out = Path(args.out).resolve()
    sidecar = out.with_suffix(".rows.jsonl")
    report = read_json(out)
    audit = {
        **ctx.base_audit("assemble-report"),
        "inputs": {
            "dref": file_record(dref),
            "manifest": file_record(args.manifest),
            "scored": file_record(args.scored),
            "cells": [str(path) for path in cells],
        },
        "verified_dref_audit": dref_receipt,
        "output": {
            "report": file_record(out),
            "sidecar": file_record(sidecar),
            "rows": report.get("n_rows"),
            "floors_passed": ((report.get("summary") or {}).get("absolute_floors_pooled") or {}).get("passed"),
        },
    }
    write_json_new(args.audit_out, audit)
    print(json.dumps({"report": file_record(out), "sidecar": file_record(sidecar), "audit": file_record(args.audit_out)}, indent=2))
    return 0


def freeze_qref(args: argparse.Namespace) -> int:
    if args.side != "baseline":
        raise RuntimeError("Q_ref must be frozen under the baseline source root")
    reports = [Path(path).resolve() for path in args.reference]
    cells = [
        Path(source["cell"])
        for report in reports
        for source in (read_json(report).get("seeds") or {}).values()
    ]
    ctx = prepare_context(args, cells)
    out = Path(args.out).resolve()
    doc = ctx.qref.freeze_direct(
        args.gpu,
        [str(path) for path in reports],
        Path(args.acceptance).resolve(),
        args.evaluator_digest,
        out,
        check_sources=True,
    )
    audit = {
        **ctx.base_audit("freeze-qref"),
        "inputs": {"reports": [file_record(path) for path in reports], "acceptance": file_record(args.acceptance)},
        "output": {"qref": file_record(out), "selection": doc.get("selection"), "sources": doc.get("sources")},
    }
    write_json_new(args.audit_out, audit)
    print(json.dumps({"qref": file_record(out), "audit": file_record(args.audit_out), "selection": doc.get("selection")}, indent=2))
    return 0


def self_check(args: argparse.Namespace) -> int:
    ctx = prepare_context(args, [])
    payload = {
        **ctx.base_audit("self-check"),
        "loaded": {
            "qref": str(Path(ctx.qref.__file__).resolve()),
            "verify_report": str(Path(ctx.verify_report.__file__).resolve()),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--runtime-manifest", required=True)
    ap.add_argument("--quality-freeze", required=True)
    ap.add_argument("--gate-runner", required=True)
    ap.add_argument("--freeze-root", required=True)
    ap.add_argument("--side", choices=("baseline", "candidate"), required=True)
    ap.add_argument("--source-root", help="optional assertion; must equal the source root frozen for --side")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    sc = sub.add_parser("self-check")
    add_common(sc)
    sc.set_defaults(func=self_check)

    bd = sub.add_parser("build-dref")
    add_common(bd)
    bd.add_argument("--cell", action="append", required=True)
    bd.add_argument("--acceptance", required=True)
    bd.add_argument("--out", required=True)
    bd.add_argument("--audit-out", required=True)
    bd.set_defaults(func=build_dref)

    vd = sub.add_parser("verify-dref")
    add_common(vd)
    vd.add_argument("--d-ref", required=True)
    vd.add_argument("--audit-out", required=True)
    vd.set_defaults(func=verify_dref)

    eb = sub.add_parser("export-bundle")
    add_common(eb)
    eb.add_argument("--cell", action="append", required=True)
    eb.add_argument("--d-ref", required=True)
    eb.add_argument("--verified-dref-audit")
    eb.add_argument("--seed-tts-root", required=True)
    eb.add_argument("--out", required=True)
    eb.add_argument("--audit-out", required=True)
    eb.set_defaults(func=export_bundle)

    ar = sub.add_parser("assemble-report")
    add_common(ar)
    ar.add_argument("--cell", action="append", required=True)
    ar.add_argument("--d-ref", required=True)
    ar.add_argument("--verified-dref-audit")
    ar.add_argument("--manifest", required=True)
    ar.add_argument("--scored", required=True)
    ar.add_argument("--seed-tts-root", required=True)
    ar.add_argument("--expected-evaluator-digest", required=True)
    ar.add_argument("--out", required=True)
    ar.add_argument("--audit-out", required=True)
    ar.set_defaults(func=assemble_report)

    fq = sub.add_parser("freeze-qref")
    add_common(fq)
    fq.add_argument("--gpu", required=True)
    fq.add_argument("--reference", action="append", required=True)
    fq.add_argument("--acceptance", required=True)
    fq.add_argument("--evaluator-digest", required=True)
    fq.add_argument("--out", required=True)
    fq.add_argument("--audit-out", required=True)
    fq.set_defaults(func=freeze_qref)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
