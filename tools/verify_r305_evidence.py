"""Verify public r305 measurement arithmetic, workload identity and source binding.

CPU-only artifact verification. This is not the frozen statistical or speech gate.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/20261102-r305"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def command_workload(path):
    args = read(path)
    normalized = []
    i = 0
    while i < len(args):
        if args[i] in ("--port", "--result-dir", "--result-filename", "--metric-percentiles"):
            i += 2
        else:
            normalized.append(args[i])
            i += 1
    return normalized


def main():
    data = read(ROOT / "docs/data/ascend-r305-20261102.json")
    manifest = read(EVIDENCE / "source-manifest.json")
    assert [r["concurrency"] for r in data["rows"]] == [1, 2, 4, 8]
    assert len(data["rounds"]) == 10
    assert data["speech_gate"] == "NOT_RUN" and not data["quality_equivalence"]
    assert not data["promoted_to_default"]
    for row in data["rows"]:
        c = row["concurrency"]
        omni = read(EVIDENCE / "native" / f"omni-c{c}.json")
        engine = read(EVIDENCE / "native" / f"r305-c{c}.json")
        assert omni == row["omni"] and engine == row["engine"]
        assert omni["max_concurrency"] == engine["max_concurrency"] == c
        assert omni["completed"] == engine["completed"] == row["requests"]
        assert omni["failed"] == engine["failed"] == 0
        assert omni["total_input_tokens"] == engine["total_input_tokens"]
        assert command_workload(EVIDENCE / "native" / f"omni-c{c}-command.json") == command_workload(
            EVIDENCE / "native" / f"r305-c{c}-command.json"
        )
        for field, metric in (("ttfp", "mean_audio_ttfp_ms"), ("rtf", "mean_audio_rtf"), ("request_s", "request_throughput")):
            gain = 100 * (engine[metric] / omni[metric] - 1) if field == "request_s" else 100 * (1 - engine[metric] / omni[metric])
            assert math.isclose(gain, row["mean_gains"][field], abs_tol=1e-9)
        for raw in (omni, engine):
            for field in ("mean_audio_ttfp_ms", "p99_audio_ttfp_ms", "mean_audio_rtf", "p99_audio_rtf", "request_throughput"):
                assert math.isfinite(raw[field]) and raw[field] > 0
    for row in data["incremental"]:
        c = row["concurrency"]
        assert row["control"] == read(EVIDENCE / "native" / f"r292-c{c}.json")
        assert row["screen"] == read(EVIDENCE / "native" / f"r305-screen-c{c}.json")
        for name, metric in (("ttfp_pct", "mean_audio_ttfp_ms"), ("rtf_pct", "mean_audio_rtf"), ("req_s_pct", "request_throughput")):
            ratio = row["candidate"][metric] / row["control"][metric]
            gain = 100 * (ratio - 1) if name == "req_s_pct" else 100 * (1 - ratio)
            assert math.isclose(gain, row["mean_gains"][name], abs_tol=1e-9)
    for row in data["rounds"]:
        folder = EVIDENCE / "rounds" / row["artifact"]
        assert (folder / "exit-code.txt").read_text().strip() == "0"
        assert read(folder / "spec.json")["engine_sha256"] == row["source_sha256"]
        assert all(r["failed"] == 0 for r in read(folder / "summary.json"))
        assert (ROOT / row["patch"]).exists()
    assert (EVIDENCE / "rounds/r306-first-attempt-exit-code.txt").read_text().strip() == "1"
    for tag in ("omni-lowc", "r305-omni-confirm", "r305-c1-complete"):
        folder = EVIDENCE / "receipts" / tag
        assert (folder / "exit-code.txt").read_text().strip() == "0"
        server_exit = read(folder / ("server-exit-code.json" if tag == "omni-lowc" else "native-server-exit.json"))
        assert server_exit.get("returncode", server_exit.get("exit_code")) == 0 and not server_exit["forced"]

    # Reconstruct the measured tree's hash without modifying the public server.
    exact_server = (ROOT / manifest["exact_measured_server"]).read_bytes()
    assert hashlib.sha256(exact_server).hexdigest() == manifest["measured_server_sha256"]
    digest = hashlib.sha256()
    actual_files = {
        p.relative_to(ROOT).as_posix(): p for p in (ROOT / "engine").rglob("*")
        if p.is_file() and p.suffix in (".py", ".sh", ".yaml", ".json") and "__pycache__" not in p.parts
    }
    assert set(actual_files) == set(manifest["runtime_files"])
    for name, path in sorted(actual_files.items()):
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == manifest["runtime_files"][name], name
        measured = exact_server if name == "engine/server.py" else content
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(measured)
        digest.update(b"\0")
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(content)):
                modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else (
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                )
                assert not any(m.split(".")[0].lower() in ("vllm", "vllm_omni", "sglang", "inferrsi") for m in modules), name
    assert digest.hexdigest() == data["measured_engine_tree_sha256"] == manifest["measured_engine_tree_sha256"]
    assert hashlib.sha256((ROOT / manifest["experimental_config"]).read_bytes()).hexdigest() == data["measured_config_sha256"]
    journey_path = ROOT / "docs/data/optimization-journey.json"
    if journey_path.exists():
        journey = read(journey_path)
        raw = ROOT / "evidence/optimization-journey"
        for index, filename in ((0, "r187-native.json"), (1, "r239-native.json")):
            for row in read(raw / filename):
                c = str(row["concurrency"])
                saved = journey["versions"][index]["rows"][c]
                result = row["result"]
                assert saved["ttfp_ms"] == result["mean_audio_ttfp_ms"]
                assert saved["rtf"] == result["mean_audio_rtf"]
                assert saved["req_s"] == result["request_throughput"]
        assert journey["r295_matched"] == read(raw / "r295-matched-comparison.json")
        for row in journey["r295_matched"]:
            assert journey["versions"][2]["rows"][str(row["concurrency"]) ] == row["candidate"]
        for phase in journey["phase_endpoint_comparisons"]:
            for row in phase["rows"]:
                assert row["before"]["failed"] == row["after"]["failed"] == 0
                for metric in ("ttfp_ms", "rtf", "req_s"):
                    ratio = row["after"][metric] / row["before"][metric]
                    expected = 100 * (ratio - 1) if metric == "req_s" else 100 * (1 - ratio)
                    assert math.isclose(expected, row["mean_gains"][metric], abs_tol=1e-9)
        assert journey["six_layer_incremental"] == data["incremental"]
        assert journey["old_protocol_r153"] == read(raw / "r153-old-protocol-report.json")
        assert journey["old_protocol_r153"]["performance_overall"] == "pass"
        assert journey["old_protocol_r153"]["speech_quality_evaluated"] is False
    links = 0
    docs = [ROOT / "README.md", ROOT / "docs/rsi-profiling-20261102.zh-CN.md", EVIDENCE / "README.md"]
    if journey_path.exists():
        docs.extend([ROOT / "docs/optimization-journey.zh-CN.md", ROOT / "docs/optimization-ledger-npu.zh-CN.md"])
    for doc in docs:
        for link in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if link.startswith(("http://", "https://", "#")):
                continue
            assert (doc.parent / link.split("#", 1)[0]).exists(), (doc.name, link)
            links += 1
    print(json.dumps(dict(direct_comparison_cells=4,incremental_cells=3,rounds=10,local_links=links,
                          measured_source_hash_matches=True,speech_gate="NOT_RUN",quality_certified=False)))


if __name__ == "__main__":
    main()
