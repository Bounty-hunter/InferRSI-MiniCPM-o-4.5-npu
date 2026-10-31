"""Verify NPU cells and export their generated WAVs as a portable scorer bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.speech.suite import collect, load_dref
from tools.speech_quality_bundle import export_bundle, sha256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--d-ref", required=True)
    ap.add_argument("--seed-tts-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    col = collect([Path(c) for c in a.cell])
    load_dref(a.d_ref, col)
    manifest = export_bundle(col, a.d_ref, a.seed_tts_root, a.out)
    print(json.dumps({"manifest": str(manifest), "sha256": sha256(manifest), "rows": len(col.rows),
                      "candidate": col.generation["candidate_id"], "concurrency": col.generation["concurrency"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
