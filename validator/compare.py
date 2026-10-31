"""Frozen final comparison entry point.

Input manifest::

  {"cells": {"1": {"candidate": "/...", "baseline": "/..."}, ...},
   "quality_gate": "/path/to/gate.json"}

Every performance cell is reverified from raw artifacts by ``bench.verify``.
An absent/failed quality gate can never produce PASS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.stats import compare_cells
from bench.verify import verify_cell


CONCURRENCIES = (1, 16, 32)
HEADLINE = ("ttfa_p50_s", "throughput_frames_s")
REPORTED = ("ttfa_p50_s", "ttfa_p95_s", "e2e_p50_s", "e2e_p95_s", "throughput_frames_s", "rtf_mean")


def means(cell: Path) -> dict:
    verified = verify_cell(cell)
    if not verified.ok:
        raise ValueError(f"cell {cell} is not verified: {verified.problems[:8]}")
    rows = verified.measured_summaries
    result = {}
    for metric in REPORTED:
        values = [float(row[metric]) for row in rows if metric in row]
        result[metric] = float(np.mean(values)) if len(values) == len(rows) and values else None
    result["error_rate"] = sum(int(row.get("n_failed", 0)) for row in rows) / max(1, sum(int(row["n_requests"]) for row in rows))
    return result


def validate(manifest: dict, final: bool) -> dict:
    cells, results, failures = manifest.get("cells", {}), {}, []
    for concurrency in CONCURRENCIES:
        pair = cells.get(str(concurrency)) or {}
        if not pair.get("candidate") or not pair.get("baseline"):
            failures.append(f"C{concurrency}: candidate/baseline cell missing")
            continue
        candidate, baseline = Path(pair["candidate"]), Path(pair["baseline"])
        candidate_verified, baseline_verified = verify_cell(candidate), verify_cell(baseline)
        if candidate_verified.spec.get("candidate", {}).get("adapter") != "engine":
            failure = f"C{concurrency}: candidate cell is not the standalone engine"
            failures.append(failure)
            results[str(concurrency)] = {"error": failure}
            continue
        if baseline_verified.spec.get("candidate", {}).get("adapter") != "vllm-omni":
            failure = f"C{concurrency}: baseline cell is not vLLM-Omni"
            failures.append(failure)
            results[str(concurrency)] = {"error": failure}
            continue
        item = {"candidate": means(candidate), "baseline": means(baseline), "ratios": {}}
        for metric in HEADLINE:
            estimate = compare_cells(candidate, baseline, metric, final=final)
            item["ratios"][metric] = estimate.__dict__
            if estimate.verdict != "pass":
                failures.append(f"C{concurrency} {metric}: {estimate.verdict}, point={estimate.point:.4f}, lcb95={estimate.lcb95:.4f}")
        if item["candidate"]["error_rate"] != 0:
            failures.append(f"C{concurrency}: candidate error rate {item['candidate']['error_rate']:.6f}")
        results[str(concurrency)] = item
    quality_path = manifest.get("quality_gate")
    quality = json.loads(Path(quality_path).read_text()) if quality_path else None
    if not quality or quality.get("overall") != "pass":
        failures.append("speech quality gate is absent or not pass")
    return {"overall": "pass" if not failures else ("inconclusive" if not results else "fail"),
            "final": final, "cells": results, "quality_gate": quality, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--development", action="store_true", help="use K=5..10 development rules instead of frozen K=10")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = validate(json.loads(Path(args.manifest).read_text()), final=not args.development)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(encoded + "\n")
    print(encoded)
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
