#!/usr/bin/env python3
"""Freeze the exact r152 candidate source selected by the r151 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from bench.fingerprint import tree_sha256


CONCURRENCIES = (1, 16, 32)
BOUND_FILES = (
    "engine/talker_sampler.py",
    "engine/runtime/backend_npu.py",
    "engine/runtime/talker_post_graph.py",
    "configs/engine/ascend_c1_r152_omni_counter_postgraph.json",
    "configs/engine/ascend_c16_r152_omni_counter_postgraph.json",
    "configs/engine/ascend_c32_r152_omni_counter_postgraph.json",
    "evidence/20261022-ascend910b1/plan/performance-compare-r152-k10.json",
    "tools/run_remote_engine_cell_r152.sh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--screen-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.source_root.resolve()
    screen_path = args.screen_report.resolve()
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if screen.get("schema") != "minicpmo-ascend-performance-screen-result/v1":
        raise RuntimeError("unexpected r151 screen schema")
    if screen.get("stage") != "r151b-k3" or screen.get("pass") is not True:
        raise RuntimeError("r151b K=3 screen has not passed")
    if screen.get("acceptance_evidence") is not False:
        raise RuntimeError("the r151 screen must remain non-acceptance evidence")

    missing = [rel for rel in BOUND_FILES if not (root / rel).is_file()]
    if missing:
        raise RuntimeError(f"missing source-freeze files: {missing}")

    configs = {}
    for concurrency in CONCURRENCIES:
        rel = f"configs/engine/ascend_c{concurrency}_r152_omni_counter_postgraph.json"
        config = json.loads((root / rel).read_text(encoding="utf-8"))
        if config.get("talker_sampling_rng") != "counter":
            raise RuntimeError(f"C{concurrency} does not select counter RNG")
        if config.get("talker_post_graphs") is not True:
            raise RuntimeError(f"C{concurrency} does not select Talker post-graph")
        configs[str(concurrency)] = {
            "path": rel,
            "sha256": sha256(root / rel),
        }

    git_head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    payload = {
        "schema": "minicpmo-ascend-candidate-source-freeze/v1",
        "campaign": "r152-fixed-k10-gpu5-counter-postgraph",
        "source_root": str(root),
        "git_head": git_head,
        "git_status": git_status,
        "engine_tree_sha256": tree_sha256(root, ("engine",)),
        "screen": {
            "path": str(screen_path),
            "sha256": sha256(screen_path),
            "stage": screen["stage"],
            "pass": True,
        },
        "configs": configs,
        "bound_files": {rel: sha256(root / rel) for rel in BOUND_FILES},
        "formal_evidence": False,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
