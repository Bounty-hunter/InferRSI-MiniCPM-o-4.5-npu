"""Portable, hash-bound speech rows for scoring away from the NPU host."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from evals.speech.suite import Collection, GenRow

BUNDLE_SCHEMA = "speech-quality-bundle/v1"
SCORED_SCHEMA = "speech-quality-scored/v1"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wav_name(row: GenRow) -> str:
    return hashlib.sha256(row.key.encode()).hexdigest() + ".wav"


def row_record(row: GenRow, seed_tts_root: Path) -> dict:
    ref = Path(row.ref_wav).resolve()
    try:
        ref_rel = ref.relative_to(seed_tts_root.resolve())
    except ValueError as e:
        raise ValueError(f"reference audio is outside the frozen Seed-TTS root: {ref}") from e
    wav = Path(row.wav).resolve()
    return {
        "key": row.key,
        "row_id": row.row_id,
        "seed_index": row.seed_index,
        "target_text": row.target_text,
        "thinker_text": row.thinker_text,
        "duration_s": row.duration_s,
        "wav_file": f"wav/{wav_name(row)}",
        "wav_sha256": sha256(wav),
        "ref_rel": ref_rel.as_posix(),
        "ref_sha256": sha256(ref),
    }


def manifest_for(col: Collection, dref_path: str | Path, seed_tts_root: str | Path) -> dict:
    dref = Path(dref_path).resolve()
    root = Path(seed_tts_root).resolve()
    return {
        "schema": BUNDLE_SCHEMA,
        "generation": col.generation,
        "seeds": col.seeds,
        "d_ref": {"path": str(dref), "sha256": sha256(dref)},
        "rows": [row_record(r, root) for r in sorted(col.rows, key=lambda x: x.key)],
    }


def export_bundle(col: Collection, dref_path: str | Path, seed_tts_root: str | Path, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"bundle output already exists: {out}")
    (out / "wav").mkdir(parents=True)
    manifest = manifest_for(col, dref_path, seed_tts_root)
    by_key = {r.key: r for r in col.rows}
    for rec in manifest["rows"]:
        shutil.copyfile(by_key[rec["key"]].wav, out / rec["wav_file"])
        if sha256(out / rec["wav_file"]) != rec["wav_sha256"]:
            raise RuntimeError(f"copied WAV changed for {rec['key']}")
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path


def validate_manifest(path: str | Path, seed_tts_root: str | Path) -> tuple[dict, list[GenRow]]:
    path = Path(path).resolve()
    doc = json.loads(path.read_text())
    if doc.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"not a {BUNDLE_SCHEMA} manifest")
    root = Path(seed_tts_root).resolve()
    rows, keys = [], set()
    for rec in doc.get("rows") or []:
        if rec["key"] in keys:
            raise ValueError(f"duplicate bundle row {rec['key']}")
        keys.add(rec["key"])
        wav = (path.parent / rec["wav_file"]).resolve()
        ref = (root / rec["ref_rel"]).resolve()
        if not wav.is_file() or sha256(wav) != rec["wav_sha256"]:
            raise ValueError(f"generated WAV is missing or changed for {rec['key']}")
        if not ref.is_file() or sha256(ref) != rec["ref_sha256"]:
            raise ValueError(f"reference WAV is missing or changed for {rec['key']}")
        rows.append(GenRow(rec["key"], rec["row_id"], int(rec["seed_index"]), rec["target_text"], str(ref), str(wav),
                           rec["thinker_text"], float(rec["duration_s"])))
    if not rows:
        raise ValueError("bundle has no rows")
    return doc, rows


def compare_manifest_to_collection(manifest_path: str | Path, col: Collection, dref_path: str | Path, seed_tts_root: str | Path) -> list[str]:
    """Recompute every decisive manifest field from the verified source cells."""
    recorded = json.loads(Path(manifest_path).read_text())
    expected = manifest_for(col, dref_path, seed_tts_root)
    return [k for k in ("schema", "generation", "seeds", "d_ref", "rows") if recorded.get(k) != expected.get(k)]
