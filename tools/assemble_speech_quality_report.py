"""Join frozen CUDA scores back to verified NPU cells and write a final report."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from evals.speech.suite import build_quality, collect, load_dref, write_report
from tools.speech_quality_bundle import SCORED_SCHEMA, compare_manifest_to_collection, sha256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--d-ref", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--seed-tts-root", required=True)
    ap.add_argument("--expected-evaluator-digest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    started = time.time()
    col = collect([Path(c) for c in a.cell])
    dref = load_dref(a.d_ref, col)
    bad = compare_manifest_to_collection(a.manifest, col, a.d_ref, a.seed_tts_root)
    if bad:
        raise ValueError(f"portable scorer manifest differs from the verified NPU cells on {bad}")
    receipt_path = Path(a.scored).resolve()
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != SCORED_SCHEMA:
        raise ValueError(f"not a {SCORED_SCHEMA} receipt")
    if (receipt.get("bundle") or {}).get("sha256") != sha256(a.manifest):
        raise ValueError("scoring receipt names a different bundle")
    keys = hashlib.sha256("\n".join(sorted(r.key for r in col.rows)).encode()).hexdigest()
    if receipt.get("rows") != len(col.rows) or receipt.get("row_keys_sha256") != keys:
        raise ValueError("scoring receipt row identity differs from the source cells")
    if (receipt.get("evaluator") or {}).get("digest") != a.expected_evaluator_digest:
        raise ValueError("scoring receipt used a different evaluator")
    scored = {k: receipt[k] for k in ("evaluator", "execution_evidence", "tasks")}
    quality = build_quality(col.rows, scored, dref)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"quality report already exists: {out}")
    rep = write_report(out, "final", col, scored, quality, str(Path(a.d_ref).resolve()), started)
    rep["external_scoring"] = {"receipt": {"path": str(receipt_path), "sha256": sha256(receipt_path)},
                               "bundle": {"path": str(Path(a.manifest).resolve()), "sha256": sha256(a.manifest)}}
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({"out": str(out), "rows": rep["n_rows"], "candidate": rep["generation"]["candidate_id"],
                      "evaluator": rep["evaluator"]["digest"], "floors": rep["summary"]["absolute_floors_pooled"]["passed"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
