import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_r152_configs_are_exact_r125_derivatives():
    for concurrency in (1, 16, 32):
        base = _load(f"configs/engine/ascend_c{concurrency}_r125_omni_native_rng.json")
        candidate = _load(
            f"configs/engine/ascend_c{concurrency}_r152_omni_counter_postgraph.json"
        )
        expected = dict(base)
        expected["talker_sampling_rng"] = "counter"
        expected["talker_post_graphs"] = True
        assert candidate == expected


def test_r152_plan_binds_fresh_fixed_k10_roots():
    plan = _load("evidence/20261022-ascend910b1/plan/performance-compare-r152-k10.json")
    assert plan["schema"] == "minicpmo-ascend-performance-compare/v1"
    assert plan["campaign"] == "r152-fixed-k10-gpu5-counter-postgraph"
    assert set(plan["cells"]) == {"1", "16", "32"}
    for concurrency, suffixes in {
        "1": ("r152a", "r152b"),
        "16": ("r152c", "r152d"),
        "32": ("r152e", "r152f"),
    }.items():
        baseline_suffix, candidate_suffix = suffixes
        pair = plan["cells"][concurrency]
        assert f"baseline-c{concurrency}-k10" in pair["baseline"]
        assert pair["baseline"].endswith(baseline_suffix)
        assert f"engine-c{concurrency}-k10" in pair["candidate"]
        assert pair["candidate"].endswith(candidate_suffix)


def test_r152_runner_is_frozen_to_k10_device5_and_isolated_root():
    runner = (ROOT / "tools/run_remote_engine_cell_r152.sh").read_text(encoding="utf-8")
    assert '[[ "$k" == 10 ]]' in runner
    assert '[[ "$gpu_id" == 5 ]]' in runner
    assert "engine_root=/tmp/inferrsi-minicpmo-npu-engine-r152-formal" in runner
    assert 'config="ascend_c${concurrency}_r152_omni_counter_postgraph"' in runner
