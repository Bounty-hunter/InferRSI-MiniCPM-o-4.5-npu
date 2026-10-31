"""Score a portable NPU speech bundle with the frozen CUDA evaluator."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evals.speech.suite import score
from tools.speech_quality_bundle import SCORED_SCHEMA, sha256, validate_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed-tts-root", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-whisper", action="store_true")
    a = ap.parse_args()
    manifest = Path(a.manifest).resolve()
    _, rows = validate_manifest(manifest, a.seed_tts_root)
    started = time.time()
    scored = score(rows, a.gpu, whisper=not a.no_whisper)
    doc = {"schema": SCORED_SCHEMA, "bundle": {"path": str(manifest), "sha256": sha256(manifest)},
           "rows": len(rows), "row_keys_sha256": __import__("hashlib").sha256("\n".join(sorted(r.key for r in rows)).encode()).hexdigest(),
           "evaluator": scored["evaluator"], "execution_evidence": scored["execution_evidence"], "tasks": scored["tasks"],
           "wall_s": time.time() - started}
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"scored output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({"out": str(out), "sha256": sha256(out), "rows": len(rows), "evaluator": doc["evaluator"]["digest"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
