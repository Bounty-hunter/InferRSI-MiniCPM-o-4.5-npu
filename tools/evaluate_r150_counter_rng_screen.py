#!/usr/bin/env python3
"""Evaluate the pre-registered r150 C16 K=1/K=3 screening rules.

This is deliberately not the frozen acceptance validator.  It only decides
whether the counter-RNG candidate is worth a new formal K=10 campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _summary(root: Path, run: int) -> dict:
    path = root / f"run{run}" / "summary.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing summary: {path}") from exc
    required = ("valid", "n_failed", "n_requests", "ttfa_p50_s", "throughput_frames_s")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    return data


def _finite_positive(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not 0.0 < number < float("inf"):
        raise ValueError(f"{label} must be finite and positive: {number!r}")
    return number


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric list")
    return sum(values) / len(values)


def evaluate(plan: dict, stage_name: str, baseline_root: Path, candidate_root: Path) -> dict:
    try:
        stage = next(item for item in plan["stages"] if item["name"] == stage_name)
    except (KeyError, StopIteration) as exc:
        raise ValueError(f"unknown stage: {stage_name}") from exc

    formal_runs = [int(run) for run in stage["formal_runs"]]
    probe_run = int(stage["probe_run"])
    expected_formal = list(range(1, int(stage["k"]) + 1))
    if formal_runs != expected_formal or probe_run != 0:
        raise ValueError(
            f"stage {stage_name} must use probe run0 and formal runs {expected_formal}, "
            f"got probe={probe_run} formal={formal_runs}"
        )

    candidate = {run: _summary(candidate_root, run) for run in [probe_run, *formal_runs]}
    baseline = {run: _summary(baseline_root, run) for run in formal_runs}
    invalid = [run for run, summary in candidate.items() if summary["valid"] is not True]
    failed = sum(int(summary["n_failed"]) for summary in candidate.values())
    requests = sum(int(summary["n_requests"]) for summary in candidate.values())
    if requests <= 0:
        raise ValueError("candidate summaries contain no requests")

    paired: list[dict] = []
    for run in formal_runs:
        base = baseline[run]
        cand = candidate[run]
        if base["valid"] is not True or int(base["n_failed"]) != 0:
            raise ValueError(f"paired baseline run{run} is not valid and error-free")
        base_ttfa = _finite_positive(base["ttfa_p50_s"], f"baseline run{run} ttfa_p50_s")
        cand_ttfa = _finite_positive(cand["ttfa_p50_s"], f"candidate run{run} ttfa_p50_s")
        base_throughput = _finite_positive(
            base["throughput_frames_s"], f"baseline run{run} throughput_frames_s"
        )
        cand_throughput = _finite_positive(
            cand["throughput_frames_s"], f"candidate run{run} throughput_frames_s"
        )
        paired.append(
            {
                "run": run,
                "ttfa_speedup": base_ttfa / cand_ttfa,
                "throughput_speedup": cand_throughput / base_throughput,
                "candidate_throughput_frames_s": cand_throughput,
            }
        )

    error_rate = failed / requests
    checks = {
        "candidate_error_rate": error_rate == float(stage["promotion"]["candidate_error_rate"]),
        "all_runs_valid": not invalid,
    }
    if stage_name == "r150a-k1":
        checks.update(
            {
                "ttfa_speedup": paired[0]["ttfa_speedup"]
                >= float(stage["promotion"]["ttfa_speedup_vs_paired_r147c_run1_min"]),
                "throughput_frames_s": paired[0]["candidate_throughput_frames_s"]
                >= float(stage["promotion"]["throughput_frames_s_min"]),
            }
        )
    elif stage_name == "r150b-k3":
        ttfa_point = _mean([row["ttfa_speedup"] for row in paired])
        throughput_point = _mean([row["throughput_speedup"] for row in paired])
        checks.update(
            {
                "ttfa_speedup_point_estimate": ttfa_point
                >= float(stage["promotion"]["ttfa_speedup_point_estimate_min"]),
                "throughput_speedup_point_estimate": throughput_point
                >= float(stage["promotion"]["throughput_speedup_point_estimate_min"]),
                "each_run_throughput_speedup": all(
                    row["throughput_speedup"]
                    >= float(stage["promotion"]["each_run_throughput_speedup_min"])
                    for row in paired
                ),
            }
        )
    else:  # guarded above, but keep future plan edits fail-closed
        raise ValueError(f"unsupported stage rules: {stage_name}")

    return {
        "schema": "minicpmo-ascend-performance-screen-result/v1",
        "stage": stage_name,
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "probe_run": probe_run,
        "formal_runs": formal_runs,
        "candidate_failed_requests": failed,
        "candidate_requests_including_probe": requests,
        "candidate_error_rate": error_rate,
        "candidate_invalid_runs": invalid,
        "paired": paired,
        "checks": checks,
        "pass": all(checks.values()),
        "acceptance_evidence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("r150a-k1", "r150b-k3"), required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        result = evaluate(plan, args.stage, args.baseline_root, args.candidate_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid screen inputs: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
