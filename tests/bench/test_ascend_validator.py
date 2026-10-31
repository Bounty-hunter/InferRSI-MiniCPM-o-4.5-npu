from types import SimpleNamespace

import validator.compare as compare
from validator.compare import CONCURRENCIES, HEADLINE


def test_frozen_ascend_gate_scope():
    assert CONCURRENCIES == (1, 16, 32)
    assert HEADLINE == ("ttfa_p50_s", "throughput_frames_s")


def test_validator_rejects_a_non_engine_candidate_before_scoring(monkeypatch):
    monkeypatch.setattr(compare, "verify_cell", lambda path: SimpleNamespace(spec={"candidate": {"adapter": "vllm-omni"}}))
    cells = {str(c): {"candidate": f"candidate-{c}", "baseline": f"baseline-{c}"} for c in CONCURRENCIES}
    report = compare.validate({"cells": cells}, final=True)
    assert report["overall"] == "fail"
    assert all(f"C{c}: candidate cell is not the standalone engine" in report["failures"] for c in CONCURRENCIES)
