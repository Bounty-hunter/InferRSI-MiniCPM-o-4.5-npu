"""Score one or more exported NPU quality slices with the frozen CUDA evaluator.

This is a development screen, not the final 1088-row/three-seed quality gate.
It reuses the pinned Qwen3-ASR, WavLM and UTMOS workers from
``evals.speech.suite`` and records their evaluator fingerprint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def load_slice(label: str, root: Path, seed_tts_root: Path) -> list[object]:
    from evals.speech.suite import GenRow

    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for row in manifest["rows"]:
        request_id = row["request_id"]
        ref = Path(row["ref_wav"])
        try:
            rel = ref.relative_to("/dataset/models/seed-tts-eval")
        except ValueError as exc:
            raise ValueError(f"unexpected NPU reference path: {ref}") from exc
        wav = (root / "wav" / f"{request_id}.wav").resolve()
        ref_wav = (seed_tts_root / rel).resolve()
        if not wav.is_file() or not ref_wav.is_file():
            raise FileNotFoundError(f"missing generated/reference WAV: {wav} / {ref_wav}")
        rows.append(
            GenRow(
                key=f"{label}:{request_id}",
                row_id=request_id,
                seed_index=int(manifest["seed_index"]),
                target_text=row["target_text"],
                ref_wav=str(ref_wav),
                wav=str(wav),
                thinker_text=row["thinker_text"],
                duration_s=float(row["duration_s"]),
            )
        )
    if len(rows) != int(manifest["n_ok"]) or manifest["n_failed"]:
        raise ValueError(f"slice {label} is incomplete")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    parser.add_argument("--seed-tts-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--with-whisper", action="store_true")
    args = parser.parse_args()

    from evals.speech.gates import absolute_floors
    from evals.speech.suite import build_quality, score, summarize

    groups = {label: load_slice(label, Path(path).resolve(), args.seed_tts_root.resolve()) for label, path in args.slice}
    if len(groups) != len(args.slice):
        parser.error("slice labels must be unique")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        physical_gpu = int(visible.split(",", 1)[0])
    except ValueError as exc:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must start with a physical GPU id, got {visible!r}") from exc

    started = time.time()
    scored = score([row for rows in groups.values() for row in rows], physical_gpu, whisper=args.with_whisper)
    reports = {}
    for label, rows in groups.items():
        quality = build_quality(rows, scored, d_ref=None)
        summary = summarize(quality)
        reports[label] = {
            "n_rows": len(rows),
            "summary": summary,
            "absolute_floors": absolute_floors(summary["pooled"]),
            "rows": [row.to_json() for row in quality],
        }
    result = {
        "schema": "npu-quality-screen/v1",
        "development_only": True,
        "visible_devices": visible,
        "evaluator": scored["evaluator"],
        "execution_evidence": scored["execution_evidence"],
        "slices": reports,
        "wall_s": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({label: data["summary"]["pooled"] for label, data in reports.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
