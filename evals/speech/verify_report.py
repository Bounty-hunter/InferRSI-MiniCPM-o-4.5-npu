"""Authoritative verification of speech quality reports and duration references.

Every consumer (qualify, freeze, gate) goes through ``verify_report``: it
re-derives membership, counts, seeds and metrics from the hash-checked row
sidecar, re-verifies each seed's source cell and measured run, and reconciles
the report's generation identity and every row's generation facts with that
run.  Per-row detector flags are recomputed, never read.  Final scope needs the
canonical 1088-row set for each of exactly three seeds and a verified duration
reference.  Metadata in the report is only accepted where it equals what the
evidence yields.

A duration reference (D_ref, schema ``d_ref/v1``) is built by ``build_dref``
from verified cells of the frozen throughput winner of one (GPU class, C);
``verify_dref`` re-verifies the winner declaration and every source cell,
checks the role's run policy and coverage, and recomputes each per-row mean.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from bench.experiment import digest
from bench.verify import verify_cell
from bench.workload import load_seed_tts_rows, make_id_sets
from evals.speech.detectors import (
    CATASTROPHIC_WER,
    RowQuality,
    SilenceProfile,
    repetition_flag,
    silence_profile,
    truncation_flag,
    word_recall,
)
from evals.speech.suite import FULL_ROWS, summarize

FINAL_SEEDS = 3
DREF_SCHEMA = "d_ref/v1"
DIRECT_DREF_SCHEMA = "ascend_direct_d_ref/v1"
#: Run policy per D_ref role: the output-duration guard comes from the winner's evaluation runs; the
#: truncation detector at full scope comes from the winner's full-set runs under the 3-seed policy.
DREF_ROLES = {
    "guard": {"kind": "performance", "id_set": "eval", "seeds": [0], "min_k": 5},
    "truncation": {"kind": "quality", "id_set": "full", "seeds": [0, 1, 2], "min_k": 1},
}
CAND_IDENTITY = ("adapter", "config", "extra", "env", "deploy_files_sha256", "mps")


class ReportError(ValueError):
    pass


@dataclass
class VerifiedReport:
    path: Path
    report: dict
    rows: list[RowQuality]
    summary: dict
    identity: dict
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def _evaluator_consistent(ev: dict) -> bool:
    body = {k: v for k, v in ev.items() if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == ev.get("digest")


def measured_ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def candidate_problems(c: dict) -> list[str]:
    """A candidate record must carry the digest of its own identity fields."""
    try:
        ident = {k: c[k] for k in CAND_IDENTITY}
    except KeyError as e:
        return [f"candidate record lacks {e}"]
    out = []
    if digest(ident) != c.get("digest"):
        out.append("candidate digest does not match the candidate identity")
    if not str(c.get("candidate_id", "")).endswith(str(c.get("digest", ""))[:12]):
        out.append("candidate_id does not carry the candidate digest")
    return out


def gpu_class(hardware: dict) -> dict:
    return {"gpu_sku": hardware.get("gpu_sku"), "gpu_driver": hardware.get("driver"), "gpu_memory_mib": hardware.get("memory_mib")}


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def _row_flag_problems(r: RowQuality, dref: dict | None) -> list[str]:
    """Recompute every derived per-row flag from the row's recorded measurements."""
    out = []
    want_trunc = truncation_flag(word_recall(r.hits, r.substitutions, r.deletions), r.duration_s, dref.get(r.row_id) if dref is not None else None)
    derived = {"catastrophic": r.wer > CATASTROPHIC_WER, "repetition": repetition_flag(r.hyp_norm, r.ref_norm), "truncation": want_trunc,
               "thinker_exact_match": r.thinker_text.strip() == r.target_text.strip(),
               "silence": SilenceProfile(r.leading_silence_s, r.trailing_silence_s, r.max_internal_silence_s, r.silent_fraction).flagged}
    for k, want in derived.items():
        if getattr(r, k) != want:
            out.append(f"row {r.seed_index}:{r.row_id}: {k}={getattr(r, k)!r} but the recorded measurements give {want!r}")
    return out


def _reconcile_rows(rows: list[RowQuality], run_dir: Path, targets: dict, problems: list[str], check_audio: bool) -> None:
    reqs = {r["request_id"]: r for r in _jsonl(run_dir / "requests.jsonl")}
    traces = {t["request_id"]: t for t in _jsonl(run_dir / "traces.jsonl")}
    for r in rows:
        req, tr = reqs.get(r.row_id), traces.get(r.row_id)
        if req is None or tr is None:
            problems.append(f"row {r.seed_index}:{r.row_id} has no request in the source run")
            continue
        if not _close(req["duration_s"], r.duration_s):
            problems.append(f"row {r.seed_index}:{r.row_id}: duration {r.duration_s} != source run {req['duration_s']}")
        if tr.get("text") != r.thinker_text:
            problems.append(f"row {r.seed_index}:{r.row_id}: Thinker text differs from the source run")
        if targets.get(r.row_id) != r.target_text:
            problems.append(f"row {r.seed_index}:{r.row_id}: target text differs from the dataset")
        if check_audio:
            pcm, sr = sf.read(str(run_dir / "wav" / f"{r.row_id}.wav"), dtype="float32", always_2d=False)
            prof = silence_profile(np.asarray(pcm.mean(axis=1) if pcm.ndim > 1 else pcm, dtype=np.float32), sr)
            for k, got in (("leading_s", r.leading_silence_s), ("trailing_s", r.trailing_silence_s), ("max_internal_s", r.max_internal_silence_s),
                           ("silent_fraction", r.silent_fraction)):
                if not _close(getattr(prof, k), got):
                    problems.append(f"row {r.seed_index}:{r.row_id}: silence {k} {got} differs from the source audio")
                    break


def _external_scoring_problems(rep: dict, rows: list[RowQuality]) -> list[str]:
    """Bind portable CUDA scores to their immutable receipt and row metrics."""
    ext = rep.get("external_scoring")
    if not ext:
        return []
    out: list[str] = []
    receipt_rec, bundle_rec = ext.get("receipt") or {}, ext.get("bundle") or {}
    receipt_path = Path(receipt_rec.get("path", ""))
    bundle_path = Path(bundle_rec.get("path", ""))
    if not receipt_path.is_file() or _sha(receipt_path) != receipt_rec.get("sha256"):
        return ["external scoring receipt is missing or changed"]
    if not bundle_path.is_file() or _sha(bundle_path) != bundle_rec.get("sha256"):
        return ["external scoring bundle manifest is missing or changed"]
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"external scoring receipt is unreadable: {e}"]
    if receipt.get("schema") != "speech-quality-scored/v1":
        out.append("external scoring receipt has the wrong schema")
    if (receipt.get("bundle") or {}).get("sha256") != bundle_rec.get("sha256"):
        out.append("external scoring receipt names another bundle")
    if receipt.get("evaluator") != rep.get("evaluator") or receipt.get("execution_evidence") != rep.get("execution_evidence"):
        out.append("report evaluator or execution evidence differs from the scoring receipt")
    keys = sorted(f"{r.seed_index}:{r.row_id}" for r in rows)
    if receipt.get("rows") != len(rows) or receipt.get("row_keys_sha256") != hashlib.sha256("\n".join(keys).encode()).hexdigest():
        out.append("external scoring receipt row identity differs from the report")
    tasks = receipt.get("tasks") or {}
    asr = (tasks.get("asr") or {}).get("rows") or {}
    sim = (tasks.get("sim") or {}).get("rows") or {}
    utmos = (tasks.get("utmos") or {}).get("rows") or {}
    for r in rows:
        key = f"{r.seed_index}:{r.row_id}"
        a = asr.get(key)
        if not a or key not in sim or key not in utmos:
            out.append(f"external scoring receipt lacks row {key}")
            break
        fields = {"hyp_text": r.hyp_text, "ref_norm": r.ref_norm, "hyp_norm": r.hyp_norm,
                  "substitutions": r.substitutions, "deletions": r.deletions, "insertions": r.insertions, "hits": r.hits}
        if any(a.get(k) != v for k, v in fields.items()) or not _close(a.get("wer"), r.wer) \
                or not _close(sim.get(key), r.sim) or not _close(utmos.get(key), r.utmos):
            out.append(f"external scoring receipt metrics differ for row {key}")
            break
    return out


def verify_report(path: str | Path, check_sources: bool = True, check_audio: bool = True) -> VerifiedReport:
    path = Path(path)
    rep = json.loads(path.read_text())
    problems: list[str] = []
    side = path.with_suffix(".rows.jsonl")
    if not side.exists() or _sha(side) != rep.get("row_sidecar_sha256"):
        raise ReportError(f"{side} is missing or does not match its recorded hash")
    rows = [RowQuality(**r) for r in _jsonl(side)]
    keys = [(r.seed_index, r.row_id) for r in rows]
    if len(set(keys)) != len(keys):
        problems.append("sidecar has duplicate (seed, row) entries")
    by_seed: dict[int, list[str]] = {}
    for s, rid in keys:
        by_seed.setdefault(s, []).append(rid)
    seeds = sorted(by_seed)
    if [str(s) for s in seeds] != sorted(rep.get("seeds", {}), key=int):
        problems.append(f"sidecar seeds {seeds} != report seeds {sorted(rep.get('seeds', {}))}")
    if {str(s): len(v) for s, v in by_seed.items()} != rep.get("rows_per_seed"):
        problems.append("rows_per_seed claimed by the report differs from the sidecar")
    rsha = hashlib.sha256("\n".join(sorted(f"{s}:{rid}" for s, rid in keys)).encode()).hexdigest()
    if rsha != rep.get("rows_sha256"):
        problems.append("rows_sha256 claimed by the report differs from the sidecar")
    if not _evaluator_consistent(rep.get("evaluator") or {}):
        problems.append("evaluator record does not match its digest")
    gen = rep.get("generation") or {}
    problems += [f"generation.candidate: {p}" for p in candidate_problems(gen.get("candidate") or {})]
    if (gen.get("candidate") or {}).get("digest") != gen.get("candidate_digest") or (gen.get("candidate") or {}).get("candidate_id") != gen.get("candidate_id"):
        problems.append("generation candidate fields disagree with the candidate record")
    if check_sources:
        targets = {r.row_id: r.target_text for r in load_seed_tts_rows()}
        for s in seeds:
            src = (rep.get("seeds") or {}).get(str(s)) or {}
            v = verify_cell(src.get("cell", ""))
            if not v.ok:
                problems.append(f"seed {s}: source cell {src.get('cell')} fails verification: {v.problems[:2]}")
                continue
            spec = v.spec
            res = json.loads((Path(src["cell"]) / "result.json").read_text())
            measured = {f"run{i}" for i in range(1, int(spec["k"]) + 1)}
            if src.get("run") not in measured:
                problems.append(f"seed {s}: source run {src.get('run')!r} is not a measured run (run1..run{spec['k']})")
                continue
            if res.get("experiment_digest") != src.get("experiment_digest") or res["artifacts"].get(src.get("run")) != src.get("artifacts"):
                problems.append(f"seed {s}: source binding (digest/artifacts) changed since scoring")
            if src.get("gpu_uuid") != spec["hardware"].get("gpu_uuid"):
                problems.append(f"seed {s}: recorded device {src.get('gpu_uuid')} is not the source cell's ({spec['hardware'].get('gpu_uuid')})")
            if int(spec["seed_index"]) != s:
                problems.append(f"seed {s}: source cell declares seed {spec['seed_index']}")
            if sorted(by_seed[s]) != sorted(spec["measured_ids"]):
                problems.append(f"seed {s}: sidecar rows differ from the source cell's declared IDs")
            want = {"candidate": spec["candidate"], "candidate_digest": spec["candidate"]["digest"], "candidate_id": spec["candidate"]["candidate_id"],
                    **gpu_class(spec["hardware"]), "concurrency": spec["concurrency"], "id_set": spec["id_set"], "dataset": spec["dataset"],
                    "model": spec["model"], "measured_ids_sha256": measured_ids_sha256(spec["measured_ids"]),
                    "seed_policy": spec["sampling"].get("seed_policy")}
            for k, w in want.items():
                if gen.get(k) != w:
                    problems.append(f"generation.{k}={gen.get(k)!r} contradicts source cell ({w!r})")
            _reconcile_rows([r for r in rows if r.seed_index == s], Path(src["cell"]) / src["run"], targets, problems, check_audio)
    if rep.get("scope") == "final":
        canon = sorted(make_id_sets(load_seed_tts_rows()).full())
        if len(seeds) != FINAL_SEEDS:
            problems.append(f"final scope needs exactly {FINAL_SEEDS} seeds, got {len(seeds)}")
        for s in seeds:
            if sorted(by_seed[s]) != canon:
                problems.append(f"final scope: seed {s} does not cover the canonical {FULL_ROWS}-row set")
        if gen.get("id_set") != "full":
            problems.append("final scope needs full-set generation")
        if not rep.get("d_ref"):
            problems.append("final scope needs a verified duration reference (truncation)")
    dref = None
    if rep.get("d_ref"):
        dpath = Path(rep["d_ref"]["path"])
        if not dpath.exists() or _sha(dpath) != rep["d_ref"]["sha256"]:
            problems.append("duration reference file is missing or changed")
        else:
            expect = {"role": "truncation", "concurrency": gen.get("concurrency"), "gpu_class": {k: gen.get(k) for k in ("gpu_sku", "gpu_driver", "gpu_memory_mib")},
                      "model": gen.get("model"), "dataset": gen.get("dataset"), "seed_policy": gen.get("seed_policy"), "rows": sorted({r.row_id for r in rows})}
            dproblems, verified = verify_dref(dpath, expect, check_sources=check_sources)
            problems += [f"d_ref: {p}" for p in dproblems]
            if not dproblems:
                dref = verified
                bad = [r.row_id for r in rows if r.d_ref_s != dref.get(r.row_id)]
                if bad:
                    problems.append(f"{len(bad)} rows use a D_ref not in the verified reference (e.g. {bad[:3]})")
    elif any(r.d_ref_s is not None for r in rows):
        problems.append("rows carry D_ref values without a duration reference")
    if rep.get("d_ref") and dref is None:
        dref = {}  # an unverifiable reference yields no truncation flags; the problem above already fails the report
    for r in rows:
        problems += _row_flag_problems(r, dref if rep.get("d_ref") else None)[:1]
    problems += _external_scoring_problems(rep, rows)
    if bool(rep.get("truncation_evaluated")) != (bool(rows) and all(r.truncation is not None for r in rows)):
        problems.append("truncation_evaluated does not match the rows")
    summary = summarize(rows) if rows else {}
    identity = {"gpu_sku": gen.get("gpu_sku"), "gpu_driver": gen.get("gpu_driver"), "gpu_memory_mib": gen.get("gpu_memory_mib"),
                "concurrency": gen.get("concurrency"), "id_set": gen.get("id_set"),
                "measured_ids_sha256": gen.get("measured_ids_sha256"), "dataset": gen.get("dataset"), "model": gen.get("model"),
                "seed_policy": gen.get("seed_policy"),
                "seeds": seeds, "rows_per_seed": {str(s): len(v) for s, v in by_seed.items()}, "evaluator_digest": (rep.get("evaluator") or {}).get("digest"),
                "scope": rep.get("scope")}
    return VerifiedReport(path=path, report=rep, rows=rows, summary=summary, identity=identity, problems=problems)


# --- duration references -------------------------------------------------------------------------
def _cell_durations(cell: Path, runs: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for run in runs:
        for r in _jsonl(cell / run / "requests.jsonl"):
            out.setdefault(r["request_id"], []).append(float(r["duration_s"]))
    return out


def _source_problems(specs: list[dict], role: str) -> list[str]:
    policy = DREF_ROLES[role]
    out = []
    for spec in specs:
        if spec["kind"] != policy["kind"] or spec["id_set"] != policy["id_set"]:
            out.append(f"{role} D_ref needs {policy['kind']}/{policy['id_set']} cells, got {spec['kind']}/{spec['id_set']}")
        if int(spec["k"]) < policy["min_k"]:
            out.append(f"{role} D_ref needs K >= {policy['min_k']}, got {spec['k']}")
    if sorted(int(s["seed_index"]) for s in specs) != policy["seeds"]:
        out.append(f"{role} D_ref needs seeds {policy['seeds']}, got {sorted(int(s['seed_index']) for s in specs)}")
    for k in ("concurrency", "model", "dataset"):
        if len({json.dumps(s[k], sort_keys=True) for s in specs}) > 1:
            out.append(f"source cells disagree on {k}")
    if len({json.dumps(gpu_class(s["hardware"]), sort_keys=True) for s in specs}) > 1:
        out.append("source cells span several hardware classes")
    if len({s["candidate"]["digest"] for s in specs}) > 1:
        out.append("a duration reference comes from a single candidate")
    if len({json.dumps(s["measured_ids"]) for s in specs}) > 1:
        out.append("source cells measured different rows")
    if len({s["sampling"].get("seed_policy") for s in specs}) > 1:
        out.append("source cells use different seed policies")
    return out


def _winner_problems(winners_path: str, winners_sha: str, spec: dict, check_sources: bool) -> list[str]:
    from evals.campaign import WinnerError, verify_winners

    if not Path(winners_path).exists() or _sha(Path(winners_path)) != winners_sha:
        return ["winner declaration is missing or changed"]
    try:
        w = verify_winners(winners_path, check_sources=check_sources)
    except WinnerError as e:
        return [f"winner declaration does not verify: {e}"]
    out = []
    cell = (w["cells"].get(str(spec["concurrency"])) or {}).get("throughput") or {}
    try:
        wspec = json.loads((Path(cell["cell"]) / "experiment.json").read_text())["experiment"]
        for k, a, b in (("model", wspec["model"], spec["model"]), ("dataset", wspec["dataset"], spec["dataset"]),
                        ("seed policy", wspec["sampling"].get("seed_policy"), spec["sampling"].get("seed_policy"))):
            if a != b:
                out.append(f"the winner's tuning cell and the reference sources differ in {k}")
    except (KeyError, OSError, json.JSONDecodeError):
        out.append("the winner declaration does not name a readable tuning cell")
    if cell.get("candidate_digest") != spec["candidate"]["digest"]:
        out.append(f"C={spec['concurrency']} throughput winner is {cell.get('candidate_id')}, not {spec['candidate']['candidate_id']}")
    if w["gpu_class"] != gpu_class(spec["hardware"]):
        out.append(f"winners were selected on {w['gpu_class']}, the reference was measured on {gpu_class(spec['hardware'])}")
    return out


def _norm_hardware_name(value: object) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _direct_dref_problems(doc: dict, specs: list[dict]) -> list[str]:
    """Bind an Ascend duration reference directly to the frozen acceptance baseline.

    The generic campaign path binds D_ref to a tuning-selected throughput winner.
    This standalone engine campaign has exactly one named baseline, vLLM-Omni, so
    the frozen acceptance plan is the selection authority instead.
    """
    out: list[str] = []
    sel = doc.get("selection") or {}
    if sel.get("mode") != "direct_acceptance_baseline":
        out.append("direct duration reference lacks direct_acceptance_baseline selection")
    acc = doc.get("acceptance") or {}
    raw_path = acc.get("path")
    path = Path(raw_path) if raw_path else None
    if path is None or not path.is_file() or _sha(path) != acc.get("sha256"):
        return out + ["frozen acceptance plan is missing or changed"]
    try:
        plan = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return out + [f"frozen acceptance plan is unreadable: {e}"]
    if plan.get("schema") != 1:
        out.append("direct duration reference needs acceptance schema 1")
    baseline = (plan.get("baseline") or {}).get("name")
    if baseline != "vllm-omni":
        out.append(f"acceptance baseline is {baseline!r}, not 'vllm-omni'")
    for spec in specs:
        if (spec.get("candidate") or {}).get("adapter") != baseline:
            out.append(f"direct D_ref source is not the acceptance baseline {baseline!r}")
        if spec.get("concurrency") not in (plan.get("concurrencies") or []):
            out.append(f"source concurrency {spec.get('concurrency')} is outside the acceptance plan")
        model = plan.get("model") or {}
        if any((spec.get("model") or {}).get(k) != model.get(k) for k in ("repo", "revision")):
            out.append("source model does not match the frozen acceptance model")
        want_hw = (plan.get("hardware") or {}).get("accelerator")
        got_hw = (spec.get("hardware") or {}).get("gpu_sku")
        if _norm_hardware_name(got_hw) != _norm_hardware_name(want_hw):
            out.append(f"source hardware {got_hw!r} does not match acceptance hardware {want_hw!r}")
    uuids = {(s.get("hardware") or {}).get("gpu_uuid") for s in specs}
    if None in uuids or len(uuids) != 1:
        out.append("direct D_ref sources must use one stable physical accelerator UUID")
    return out


def build_dref(cells: list[str | Path], out: Path, role: str, winners: str | Path) -> dict:
    """Per-row mean output duration over every measured run of the frozen throughput winner's cells."""
    if role not in DREF_ROLES:
        raise ReportError(f"unknown D_ref role {role!r}")
    cells = [Path(c).resolve() for c in cells]
    if len(set(cells)) != len(cells):
        raise ReportError("the same source cell was given more than once")
    specs, sources, durations = [], [], {}
    for c in cells:
        v = verify_cell(c)
        if not v.ok:
            raise ReportError(f"{c} fails verification: {v.problems[:2]}")
        res = json.loads((c / "result.json").read_text())
        runs = [f"run{i}" for i in range(1, int(v.spec["k"]) + 1)]
        for rid, d in _cell_durations(c, runs).items():
            durations.setdefault(rid, []).extend(d)
        specs.append(v.spec)
        sources.append({"cell": str(c), "experiment_digest": res["experiment_digest"], "runs": runs,
                        "artifacts": {r: res["artifacts"][r] for r in runs}, "seed_index": v.spec["seed_index"]})
    problems = _source_problems(specs, role)
    winners = Path(winners).resolve()
    problems += _winner_problems(str(winners), _sha(winners), specs[0], check_sources=True)
    if problems:
        raise ReportError(f"cannot build a {role} D_ref: {problems[:4]}")
    s0 = specs[0]
    doc = {"schema": DREF_SCHEMA, "role": role, "concurrency": s0["concurrency"], "gpu_class": gpu_class(s0["hardware"]),
           "candidate_digest": s0["candidate"]["digest"], "candidate_id": s0["candidate"]["candidate_id"], "model": s0["model"],
           "dataset": s0["dataset"], "seed_policy": s0["sampling"].get("seed_policy"), "winners": {"path": str(winners), "sha256": _sha(winners)},
           "sources": sources, "rows": {k: float(np.mean(v)) for k, v in sorted(durations.items())}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    return doc


def build_direct_dref(cells: list[str | Path], out: Path, acceptance: str | Path) -> dict:
    """Build a full-set truncation D_ref from the frozen vLLM-Omni baseline.

    Unlike ``build_dref``, selection is not delegated to the legacy multi-stack
    L20X tuning campaign.  The immutable acceptance plan names the sole baseline.
    Every source cell and row duration remains hash-bound and is re-verified.
    """
    cells = [Path(c).resolve() for c in cells]
    if not cells:
        raise ReportError("a direct duration reference needs source cells")
    if len(set(cells)) != len(cells):
        raise ReportError("the same source cell was given more than once")
    acceptance = Path(acceptance).resolve()
    if not acceptance.exists():
        raise ReportError(f"acceptance plan does not exist: {acceptance}")
    specs, sources, durations = [], [], {}
    for c in cells:
        v = verify_cell(c)
        if not v.ok:
            raise ReportError(f"{c} fails verification: {v.problems[:2]}")
        res = json.loads((c / "result.json").read_text())
        runs = [f"run{i}" for i in range(1, int(v.spec["k"]) + 1)]
        for rid, values in _cell_durations(c, runs).items():
            durations.setdefault(rid, []).extend(values)
        specs.append(v.spec)
        sources.append({"cell": str(c), "experiment_digest": res["experiment_digest"], "runs": runs,
                        "artifacts": {r: res["artifacts"][r] for r in runs}, "seed_index": v.spec["seed_index"]})
    problems = _source_problems(specs, "truncation")
    shell = {"selection": {"mode": "direct_acceptance_baseline"},
             "acceptance": {"path": str(acceptance), "sha256": _sha(acceptance)}}
    problems += _direct_dref_problems(shell, specs)
    if problems:
        raise ReportError(f"cannot build a direct truncation D_ref: {problems[:4]}")
    s0 = specs[0]
    doc = {"schema": DIRECT_DREF_SCHEMA, "role": "truncation", "concurrency": s0["concurrency"],
           "gpu_class": gpu_class(s0["hardware"]), "candidate_digest": s0["candidate"]["digest"],
           "candidate_id": s0["candidate"]["candidate_id"], "model": s0["model"], "dataset": s0["dataset"],
           "seed_policy": s0["sampling"].get("seed_policy"), **shell, "sources": sources,
           "rows": {k: float(np.mean(v)) for k, v in sorted(durations.items())}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    return doc


def verify_dref(path: str | Path, expect: dict | None = None, check_sources: bool = True) -> tuple[list[str], dict]:
    """(problems, verified row map).  ``expect`` binds role, C, GPU class, model, dataset and required rows."""
    doc = json.loads(Path(path).read_text())
    problems: list[str] = []
    direct = doc.get("schema") == DIRECT_DREF_SCHEMA
    if doc.get("schema") not in (DREF_SCHEMA, DIRECT_DREF_SCHEMA) or doc.get("role") not in DREF_ROLES:
        return [f"not a {DREF_SCHEMA} or {DIRECT_DREF_SCHEMA} duration reference with a known role"], {}
    if direct and doc.get("role") != "truncation":
        return [f"{DIRECT_DREF_SCHEMA} supports only the truncation role"], {}
    srcs = doc.get("sources") or []
    cells = [str(Path(s["cell"]).resolve()) for s in srcs]
    if not srcs or len(set(cells)) != len(cells):
        problems.append("duration reference needs distinct source cells")
    specs, durations = [], {}
    for src in srcs:
        v = verify_cell(src["cell"])
        if not v.ok:
            problems.append(f"source {src['cell']} fails verification")
            continue
        res = json.loads((Path(src["cell"]) / "result.json").read_text())
        if res["experiment_digest"] != src["experiment_digest"] or any(res["artifacts"].get(r) != a for r, a in src["artifacts"].items()):
            problems.append(f"source {src['cell']} changed since the reference was built")
        if src["runs"] != [f"run{i}" for i in range(1, int(v.spec["k"]) + 1)] or sorted(src["artifacts"]) != sorted(src["runs"]):
            problems.append(f"source {src['cell']}: reference must use every measured run")
        specs.append(v.spec)
        for rid, d in _cell_durations(Path(src["cell"]), src["runs"]).items():
            durations.setdefault(rid, []).extend(d)
    if specs and len(specs) == len(srcs):
        problems += _source_problems(specs, doc["role"])
        s0 = specs[0]
        for k, w in (("concurrency", s0["concurrency"]), ("gpu_class", gpu_class(s0["hardware"])), ("candidate_digest", s0["candidate"]["digest"]),
                     ("model", s0["model"]), ("dataset", s0["dataset"]), ("seed_policy", s0["sampling"].get("seed_policy"))):
            if doc.get(k) != w:
                problems.append(f"{k}={doc.get(k)!r} contradicts the source cells ({w!r})")
        if direct:
            problems += _direct_dref_problems(doc, specs)
        else:
            problems += _winner_problems((doc.get("winners") or {}).get("path", ""), (doc.get("winners") or {}).get("sha256", ""), s0, check_sources)
        want_n = sum(int(s["k"]) for s in specs)
        if sorted(durations) != sorted(s0["measured_ids"]) or any(len(v) != want_n for v in durations.values()):
            problems.append("source runs do not cover every declared row in every measured run")
    recomputed = {k: float(np.mean(v)) for k, v in sorted(durations.items())}
    if any(not (math.isfinite(d) and d > 0) for d in recomputed.values()):
        problems.append("duration reference has a non-positive or non-finite duration")
    if recomputed != doc.get("rows"):
        problems.append("per-row durations do not match the source cells")
    if expect:
        for k in ("role", "concurrency", "gpu_class", "model", "dataset", "seed_policy"):
            if k in expect and doc.get(k) != expect[k]:
                problems.append(f"reference {k}={doc.get(k)!r}, required {expect[k]!r}")
        missing = sorted(set(expect.get("rows") or []) - set(doc.get("rows") or {}))
        if missing:
            problems.append(f"reference lacks {len(missing)} required rows (e.g. {missing[:3]})")
    return problems, (dict(doc.get("rows") or {}) if not problems else {})
