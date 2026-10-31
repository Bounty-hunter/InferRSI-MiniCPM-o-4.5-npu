import json

from tools.evaluate_r151_counter_postgraph_screen import evaluate


def _write_summary(root, run, *, throughput, ttfa, valid=True, failed=0):
    path = root / f"run{run}" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "valid": valid,
                "n_failed": failed,
                "n_requests": 320,
                "throughput_frames_s": throughput,
                "ttfa_p50_s": ttfa,
            }
        ),
        encoding="utf-8",
    )


def _write_metrics(path, *snapshots):
    path.write_text(
        "".join(json.dumps({"captured_at_unix": i, "metrics": metrics}) + "\n" for i, metrics in enumerate(snapshots)),
        encoding="utf-8",
    )


def _metrics(*, replays=None, failures=None):
    return {
        "talker_post_graph_replays": {} if replays is None else replays,
        "talker_post_graph_failures": {} if failures is None else failures,
    }


def _plan():
    return {
        "stages": [
            {
                "name": "r151a-k1",
                "k": 1,
                "probe_run": 0,
                "formal_runs": [1],
                "promotion": {
                    "candidate_error_rate": 0.0,
                    "all_runs_valid": True,
                    "ttfa_speedup_vs_paired_r147c_run1_min": 2.0,
                    "throughput_frames_s_min": 145.0,
                    "runtime_path_guard": True,
                },
            },
            {
                "name": "r151b-k3",
                "k": 3,
                "probe_run": 0,
                "formal_runs": [1, 2, 3],
                "promotion": {
                    "candidate_error_rate": 0.0,
                    "all_runs_valid": True,
                    "ttfa_speedup_point_estimate_min": 2.0,
                    "throughput_speedup_point_estimate_min": 2.1,
                    "each_run_throughput_speedup_min": 2.0,
                    "runtime_path_guard": True,
                },
            },
        ]
    }


def _passing_k1(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    audit = tmp_path / "postgraph_metrics.jsonl"
    _write_summary(baseline, 1, throughput=70.0, ttfa=22.0)
    _write_summary(candidate, 0, throughput=1.0, ttfa=999.0)
    _write_summary(candidate, 1, throughput=148.0, ttfa=10.0)
    _write_metrics(audit, _metrics(), _metrics(replays={"16": 320}))
    return baseline, candidate, audit


def test_r151_k1_requires_positive_postgraph_replay_and_no_failures(tmp_path):
    baseline, candidate, audit = _passing_k1(tmp_path)
    result = evaluate(_plan(), "r151a-k1", baseline, candidate, audit)
    assert result["pass"] is True
    assert result["runtime_path_audit"]["max_replay_total"] == 320

    _write_metrics(audit, _metrics(replays={"16": 320}, failures={"16": 1}))
    result = evaluate(_plan(), "r151a-k1", baseline, candidate, audit)
    assert result["pass"] is False
    assert result["checks"]["talker_post_graph_failures_empty"] is False

    _write_metrics(audit, _metrics())
    result = evaluate(_plan(), "r151a-k1", baseline, candidate, audit)
    assert result["pass"] is False
    assert result["checks"]["talker_post_graph_replays_positive"] is False


def test_r151_k1_uses_formal_run1_and_probe_only_for_validity(tmp_path):
    baseline, candidate, audit = _passing_k1(tmp_path)
    result = evaluate(_plan(), "r151a-k1", baseline, candidate, audit)
    assert result["formal_runs"] == [1]
    assert result["paired"][0]["run"] == 1

    _write_summary(candidate, 0, throughput=1.0, ttfa=999.0, valid=False)
    result = evaluate(_plan(), "r151a-k1", baseline, candidate, audit)
    assert result["pass"] is False
    assert result["checks"]["all_runs_valid"] is False


def test_r151_k3_requires_every_paired_throughput_speedup(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    audit = tmp_path / "postgraph_metrics.jsonl"
    _write_summary(candidate, 0, throughput=1.0, ttfa=999.0)
    for run in (1, 2, 3):
        _write_summary(baseline, run, throughput=70.0, ttfa=22.0)
        _write_summary(candidate, run, throughput=148.0, ttfa=10.0)
    _write_metrics(audit, _metrics(replays={"16": 320}))

    result = evaluate(_plan(), "r151b-k3", baseline, candidate, audit)
    assert result["pass"] is True

    _write_summary(candidate, 2, throughput=139.0, ttfa=10.0)
    result = evaluate(_plan(), "r151b-k3", baseline, candidate, audit)
    assert result["pass"] is False
    assert result["checks"]["each_run_throughput_speedup"] is False
