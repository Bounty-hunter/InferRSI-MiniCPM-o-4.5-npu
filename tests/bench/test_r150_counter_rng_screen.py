import json

from tools.evaluate_r150_counter_rng_screen import evaluate


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


def _plan():
    return {
        "stages": [
            {
                "name": "r150a-k1",
                "k": 1,
                "probe_run": 0,
                "formal_runs": [1],
                "promotion": {
                    "candidate_error_rate": 0.0,
                    "all_runs_valid": True,
                    "ttfa_speedup_vs_paired_r147c_run1_min": 2.0,
                    "throughput_frames_s_min": 145.0,
                },
            },
            {
                "name": "r150b-k3",
                "k": 3,
                "probe_run": 0,
                "formal_runs": [1, 2, 3],
                "promotion": {
                    "candidate_error_rate": 0.0,
                    "all_runs_valid": True,
                    "ttfa_speedup_point_estimate_min": 2.0,
                    "throughput_speedup_point_estimate_min": 2.1,
                    "each_run_throughput_speedup_min": 2.0,
                },
            },
        ]
    }


def test_r150_k1_uses_formal_run1_and_probe_only_for_validity(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_summary(baseline, 1, throughput=70.0, ttfa=22.0)
    _write_summary(candidate, 0, throughput=1.0, ttfa=999.0)
    _write_summary(candidate, 1, throughput=148.0, ttfa=10.0)

    result = evaluate(_plan(), "r150a-k1", baseline, candidate)

    assert result["pass"] is True
    assert result["formal_runs"] == [1]
    assert result["paired"][0]["run"] == 1

    _write_summary(candidate, 0, throughput=1.0, ttfa=999.0, valid=False)
    result = evaluate(_plan(), "r150a-k1", baseline, candidate)
    assert result["pass"] is False
    assert result["checks"]["all_runs_valid"] is False


def test_r150_k3_requires_every_paired_throughput_speedup(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_summary(candidate, 0, throughput=1.0, ttfa=999.0)
    for run in (1, 2, 3):
        _write_summary(baseline, run, throughput=70.0, ttfa=22.0)
        _write_summary(candidate, run, throughput=148.0, ttfa=10.0)

    result = evaluate(_plan(), "r150b-k3", baseline, candidate)
    assert result["pass"] is True

    _write_summary(candidate, 2, throughput=139.0, ttfa=10.0)
    result = evaluate(_plan(), "r150b-k3", baseline, candidate)
    assert result["pass"] is False
    assert result["checks"]["each_run_throughput_speedup"] is False
