#!/usr/bin/env python3
"""Run the frozen Ascend performance validator without modifying a measured checkout.

The three acceptance modules are loaded from ``--freeze-root`` while their
ordinary package dependencies are imported from ``--source-root``.  Keeping
``bench.cli`` rooted at the measured checkout is important: cell manifests bind
that absolute repository path, so copying the whole repository to a verifier
directory would otherwise create a false identity mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType


EXPECTED = {
    "validator/compare.py": "cc3028d94f497f1a1499b9c2e9fdef0bb9b951f1a5ffd5dbef7146dcf258bf39",
    "bench/stats.py": "1000b20cbceafd250cdb505f187454d7ad8979686015218e9c9471cee1122218",
    "bench/verify.py": "636fa79df886eb5e6f6302a51b0c94dfbc794847e90a83388d326f735373662c",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_paths(freeze_root: Path) -> dict[str, Path]:
    paths = {rel: freeze_root / rel for rel in EXPECTED}
    bad = {
        rel: {"expected": EXPECTED[rel], "actual": _sha(path) if path.is_file() else None}
        for rel, path in paths.items()
        if not path.is_file() or _sha(path) != EXPECTED[rel]
    }
    if bad:
        raise RuntimeError(f"frozen validator files are missing or changed: {bad}")
    return paths


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _declared_vllm_omni_root(cell_paths: list[str]) -> str | None:
    roots = set()
    for cell in cell_paths:
        declaration = json.loads((Path(cell) / "experiment.json").read_text())
        repo = (((declaration.get("manifest") or {}).get("repos") or {}).get("vllm-omni") or
                (((declaration.get("experiment") or {}).get("fingerprints") or {}).get("repos") or {}).get("vllm-omni"))
        if repo and repo.get("path"):
            roots.add(str(Path(repo["path"]).resolve()))
    if len(roots) > 1:
        raise RuntimeError(f"cells declare several vLLM-Omni roots: {sorted(roots)}")
    return next(iter(roots), None)


def _manifest_cells(path: str) -> list[str]:
    manifest = json.loads(Path(path).read_text())
    return [pair[role] for pair in (manifest.get("cells") or {}).values() for role in ("candidate", "baseline") if pair.get(role)]


def load_validator(freeze_root: Path, source_root: Path) -> ModuleType:
    paths = _checked_paths(freeze_root)
    source_root = source_root.resolve()
    if not (source_root / "bench" / "__init__.py").is_file():
        raise RuntimeError(f"source root has no bench package: {source_root}")
    sys.path.insert(0, str(source_root))
    import bench

    stats = _load("bench.stats", paths["bench/stats.py"])
    setattr(bench, "stats", stats)
    verify = _load("bench.verify", paths["bench/verify.py"])
    setattr(bench, "verify", verify)
    return _load("_frozen_ascend_validator_compare", paths["validator/compare.py"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("--freeze-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--verify-cell", action="append", default=[])
    parser.add_argument("--vllm-omni-root")
    parser.add_argument("--out")
    args = parser.parse_args()

    freeze_root = Path(args.freeze_root).resolve()
    source_root = Path(args.source_root).resolve()
    cell_paths = list(args.verify_cell)
    if args.manifest:
        cell_paths += _manifest_cells(args.manifest)
    vllm_omni_root = args.vllm_omni_root or _declared_vllm_omni_root(cell_paths)
    if vllm_omni_root:
        os.environ["OMNI_VLLM_REPO"] = vllm_omni_root
    compare = load_validator(freeze_root, source_root)
    if args.self_check:
        result = {
            "schema": "minicpmo-frozen-performance-validator-self-check/v1",
            "freeze_root": str(freeze_root),
            "source_root": str(source_root),
            "vllm_omni_root": vllm_omni_root,
            "files": EXPECTED,
            "loaded": {
                "compare": str(Path(compare.__file__).resolve()),
                "stats": str(Path(sys.modules["bench.stats"].__file__).resolve()),
                "verify": str(Path(sys.modules["bench.verify"].__file__).resolve()),
            },
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.verify_cell:
        verify = sys.modules["bench.verify"]
        cells = []
        for cell in args.verify_cell:
            checked = verify.verify_cell(cell)
            cells.append({"cell": str(Path(cell).resolve()), "ok": checked.ok, "problems": checked.problems})
        result = {"schema": "minicpmo-frozen-performance-cell-verification/v1", "cells": cells}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if all(cell["ok"] for cell in cells) else 1
    if not args.manifest:
        parser.error("manifest is required unless --self-check or --verify-cell is used")
    report = compare.validate(json.loads(Path(args.manifest).read_text()), final=not args.development)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(encoded + "\n")
    print(encoded)
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
