from bench.verify import declared_run_labels


def test_quality_single_pass_declares_only_the_measured_run():
    assert declared_run_labels(
        {
            "kind": "quality",
            "id_set": "full",
            "k": 1,
            "cache_policy": "restart",
            "quality_single_pass": True,
        }
    ) == ["run1"]


def test_standard_schedule_retains_discarded_run0():
    assert declared_run_labels({"kind": "performance", "k": 2}) == ["run0", "run1", "run2"]
