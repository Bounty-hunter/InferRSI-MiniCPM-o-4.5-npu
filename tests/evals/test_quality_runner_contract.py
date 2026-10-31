import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _tool(name: str) -> str:
    return (ROOT / "tools" / name).read_text(encoding="utf-8")


def test_direct_quality_runners_share_request_order_and_bind_qref():
    baseline = _tool("run_remote_vllm_omni_quality_cell.sh")
    engine = _tool("run_remote_engine_quality_bound_r75.sh")

    order = '--run-order-seed "$((20261025 + seed_index))"'
    assert order in baseline
    assert order in engine
    assert '--frozen-ref "speech_qref=$qref_path"' in engine
    assert "engine_root=/tmp/inferrsi-minicpmo-npu-engine-final-r71" in engine
    assert 'config="ascend_c${concurrency}_r71"' in engine


def test_candidate_preparation_uses_the_verified_personal_container_identity():
    prepare = _tool("prepare_remote_r75_quality_candidate.sh")

    assert prepare.count("docker exec --user 1033:1033 dyy") == 2
    assert "1033:1000" not in prepare


def test_r81_candidate_is_bound_to_r83_on_reference_device_one():
    baseline = _tool("run_remote_vllm_omni_quality_cell.sh")
    engine = _tool("run_remote_engine_quality_bound_r81.sh")
    prepare = _tool("prepare_remote_r81_quality_candidate_gpu1.sh")

    order = '--run-order-seed "$((20261025 + seed_index))"'
    assert order in baseline
    assert order in engine
    assert '--frozen-ref "speech_qref=$qref_path"' in engine
    assert "engine_root=/tmp/inferrsi-minicpmo-npu-engine-final-r83" in engine
    assert 'config="ascend_c${concurrency}_r71"' in engine
    assert "reference-r79.ready" in prepare
    assert "physical_device=1" in prepare
    assert "'$qref_path' 1" in prepare
    assert prepare.count("docker exec --user 1033:1033 dyy") == 2
    assert "1033:1000" not in prepare


def test_r137_quality_chain_is_bound_to_the_r136_candidate_on_device_two():
    baseline = _tool("run_remote_vllm_omni_quality_cell.sh")
    engine = _tool("run_remote_engine_quality_bound_r137.sh")
    reference = _tool("prepare_remote_r137_quality_reference_gpu2.sh")
    candidate = _tool("prepare_remote_r137_quality_candidate_gpu2.sh")

    order = '--run-order-seed "$((20261025 + seed_index))"'
    assert order in baseline
    assert order in engine
    assert '--frozen-ref "speech_qref=$qref_path"' in engine
    assert "engine_root=/tmp/inferrsi-minicpmo-npu-engine-9674df1" in engine
    assert 'config="ascend_c${concurrency}_r125_omni_native_rng"' in engine
    assert "engine-c32-k10-r136f" in reference
    assert "physical_device=2" in reference
    assert "'$qref_path' 2" in candidate
    assert candidate.count("docker exec --user 1033:1000 dyy") == 2


def test_r137_quality_freeze_hashes_every_bound_file():
    freeze = json.loads(
        (
            ROOT
            / "evidence"
            / "20261022-ascend910b1"
            / "plan"
            / "quality-validator-freeze-r137.json"
        ).read_text(encoding="utf-8")
    )

    bindings = {
        **freeze["validator_files"],
        **freeze["generation_runners"],
        freeze["acceptance_plan"]["path"]: freeze["acceptance_plan"]["sha256"],
    }
    for relative_path, expected in bindings.items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected, relative_path
