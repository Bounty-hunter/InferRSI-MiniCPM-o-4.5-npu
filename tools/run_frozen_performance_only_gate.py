#!/usr/bin/env python3
"""Apply the frozen K=10 performance rules without claiming speech PASS.

This is a transition gate for starting the expensive speech campaign.  The
final claim still uses the frozen ``validator.compare`` entry point after the
real speech gate exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def declared_repo_path(cell: Path, name: str) -> str | None:
    declaration = json.loads((cell / "experiment.json").read_text())
    manifest_repos = (declaration.get("manifest") or {}).get("repos") or {}
    fingerprint_repos = ((declaration.get("experiment") or {}).get("fingerprints") or {}).get("repos") or {}
    path = (manifest_repos.get(name) or {}).get("path") or (fingerprint_repos.get(name) or {}).get("path")
    return str(Path(path).resolve()) if path else None


def verify_worker(cell: Path, loader_path: Path, freeze_root: Path, source_root: Path) -> int:
    loader = load_module("_minicpmo_frozen_loader_worker", loader_path.resolve())
    loader.load_validator(freeze_root.resolve(), source_root.resolve())
    verify = sys.modules["bench.verify"]
    checked = verify.verify_cell(cell)
    payload = {
        "cell": str(cell.resolve()),
        "source_root": str(source_root.resolve()),
        "ok": checked.ok,
        "problems": checked.problems,
        "spec": checked.spec,
        "measured_summaries": checked.measured_summaries,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if checked.ok else 1


def verify_in_subprocess(cell: Path, loader_path: Path, freeze_root: Path, fallback_source_root: Path) -> dict:
    source_root = Path(declared_repo_path(cell, "qwen3_omni") or fallback_source_root).resolve()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--verify-worker",
        str(cell),
        "--loader",
        str(loader_path),
        "--freeze-root",
        str(freeze_root),
        "--source-root",
        str(source_root),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, env=os.environ.copy())
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"frozen verification worker failed for {cell}: returncode={completed.returncode}, "
            f"stdout={completed.stdout[-2000:]!r}, stderr={completed.stderr[-2000:]!r}"
        ) from exc
    payload["worker_returncode"] = completed.returncode
    payload["worker_stderr"] = completed.stderr
    return payload


def means_from_verified(verified: dict, reported_metrics: tuple[str, ...]) -> dict:
    rows = verified["measured_summaries"]
    result = {}
    for metric in reported_metrics:
        values = [float(row[metric]) for row in rows if metric in row]
        result[metric] = sum(values) / len(values) if len(values) == len(rows) and values else None
    result["error_rate"] = sum(int(row.get("n_failed", 0)) for row in rows) / max(
        1, sum(int(row["n_requests"]) for row in rows)
    )
    return result


def ensure_frozen_comparison_inputs(candidate: dict, baseline: dict) -> None:
    for side, verified in (("engine", candidate), ("baseline", baseline)):
        spec = verified["spec"]
        if spec.get("kind") != "performance" or spec.get("id_set") != "eval":
            raise RuntimeError(f"{side}: expected performance/eval cell")
        if spec.get("k_updates") or int(spec.get("k", 0)) != 10:
            raise RuntimeError(f"{side}: final confirmation needs a predeclared fixed K=10 without extensions")
    cs, bs = candidate["spec"], baseline["spec"]
    for field in ("model", "dataset", "measured_ids", "seed_index", "timeout_s"):
        if cs.get(field) != bs.get(field):
            raise RuntimeError(f"engine and baseline experiment specs differ on {field}")
    if (cs.get("sampling") or {}).get("canonical") != (bs.get("sampling") or {}).get("canonical"):
        raise RuntimeError("engine and baseline experiment specs use different canonical sampling")
    if (cs.get("sampling") or {}).get("seed_policy") != (bs.get("sampling") or {}).get("seed_policy"):
        raise RuntimeError("engine and baseline experiment specs use different seed policy")
    if (cs.get("adapter_options") or {}).get("streaming") != (bs.get("adapter_options") or {}).get("streaming"):
        raise RuntimeError("engine and baseline use different streaming protocols")
    if (cs.get("hardware") or {}).get("gpu_uuid") != (bs.get("hardware") or {}).get("gpu_uuid"):
        raise RuntimeError("engine and baseline experiment specs name different physical GPUs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", nargs="?")
    ap.add_argument("--loader", required=True)
    ap.add_argument("--freeze-root", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--out")
    ap.add_argument("--verify-worker")
    args = ap.parse_args()

    if args.verify_worker:
        return verify_worker(Path(args.verify_worker), Path(args.loader), Path(args.freeze_root), Path(args.source_root))
    if not args.manifest or not args.out:
        ap.error("manifest and --out are required outside --verify-worker mode")

    manifest = json.loads(Path(args.manifest).read_text())
    repo_env = {
        "vllm-omni": "OMNI_VLLM_REPO",
        "sglang-omni": "OMNI_SGLANG_REPO",
    }
    repo_paths: dict[str, set[str]] = {name: set() for name in repo_env}
    for pair in manifest["cells"].values():
        for side in ("candidate", "baseline"):
            declaration = json.loads((Path(pair[side]) / "experiment.json").read_text())
            manifest_repos = (declaration.get("manifest") or {}).get("repos") or {}
            fingerprint_repos = ((declaration.get("experiment") or {}).get("fingerprints") or {}).get("repos") or {}
            for name in repo_env:
                path = (manifest_repos.get(name) or {}).get("path") or (fingerprint_repos.get(name) or {}).get("path")
                if path:
                    repo_paths[name].add(path)
    for name, env_name in repo_env.items():
        paths = repo_paths[name]
        if len(paths) > 1:
            raise RuntimeError(f"frozen cells disagree on {name} path: {sorted(paths)}")
        if paths:
            os.environ[env_name] = next(iter(paths))

    loader_path = Path(args.loader).resolve()
    freeze_root = Path(args.freeze_root).resolve()
    fallback_source_root = Path(args.source_root).resolve()
    first_pair = manifest["cells"]["1"]
    stats_source_root = Path(declared_repo_path(Path(first_pair["candidate"]), "qwen3_omni") or fallback_source_root)
    loader = load_module("_minicpmo_frozen_loader", loader_path)
    compare = loader.load_validator(freeze_root, stats_source_root)
    stats = sys.modules["bench.stats"]

    failures: list[str] = []
    results = {}
    for concurrency in (1, 16, 32):
        pair = manifest["cells"][str(concurrency)]
        candidate = Path(pair["candidate"])
        baseline = Path(pair["baseline"])
        cv = verify_in_subprocess(candidate, loader_path, freeze_root, fallback_source_root)
        bv = verify_in_subprocess(baseline, loader_path, freeze_root, fallback_source_root)
        if not cv["ok"]:
            failures.append(f"C{concurrency}: candidate verification failed: {cv['problems'][:8]}")
            continue
        if not bv["ok"]:
            failures.append(f"C{concurrency}: baseline verification failed: {bv['problems'][:8]}")
            continue
        if cv["spec"].get("candidate", {}).get("adapter") != "engine":
            failures.append(f"C{concurrency}: candidate adapter is not engine")
            continue
        if bv["spec"].get("candidate", {}).get("adapter") != "vllm-omni":
            failures.append(f"C{concurrency}: baseline adapter is not vllm-omni")
            continue
        ensure_frozen_comparison_inputs(cv, bv)
        item = {
            "candidate": means_from_verified(cv, tuple(compare.REPORTED)),
            "baseline": means_from_verified(bv, tuple(compare.REPORTED)),
            "ratios": {},
            "verification": {
                "candidate_source_root": cv["source_root"],
                "baseline_source_root": bv["source_root"],
                "frozen_verify": True,
            },
        }
        for metric in compare.HEADLINE:
            estimate = stats.estimate(
                cv["measured_summaries"],
                bv["measured_summaries"],
                metric,
                final=True,
                new_k=int(cv["spec"]["k"]),
                base_k=int(bv["spec"]["k"]),
            )
            item["ratios"][metric] = estimate.__dict__
            if estimate.verdict != "pass":
                failures.append(
                    f"C{concurrency} {metric}: {estimate.verdict}, "
                    f"point={estimate.point:.4f}, lcb95={estimate.lcb95:.4f}"
                )
        if item["candidate"]["error_rate"] != 0:
            failures.append(f"C{concurrency}: candidate error rate {item['candidate']['error_rate']:.6f}")
        results[str(concurrency)] = item

    report = {
        "schema": "minicpmo-frozen-performance-only-gate/v1",
        "performance_overall": "pass" if not failures else "fail",
        "final_rules": True,
        "speech_quality_evaluated": False,
        "cells": results,
        "failures": failures,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    Path(args.out).write_text(encoded)
    print(encoded, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
