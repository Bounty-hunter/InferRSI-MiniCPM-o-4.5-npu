from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "work" / "r149-quality-scoring" / "run-score.sh"


def test_r149_cuda_runner_uses_frozen_evaluator_and_scheduler():
    text = RUNNER.read_text()
    assert "must run as dyy" in text
    assert "gpu run --gpus 1 --timeout 24h --wait 2h" in text
    assert "CUDA_VISIBLE_DEVICES" not in text
    assert "/data/dyy/code/minicpmo-npu-quality-final-r36" in text
    assert "630c4def88696b272e901377be31b384a673f24a186213118e511da97de2a5a0" in text
    assert "6be026e74bae6e5c8b1b9bb2452484265101f9c3e8a42b86a683db511426c61e" in text
    assert "score_speech_quality_bundle.py" in text
    assert 'receipt["rows"] == 3264' in text
    assert "speech-quality-scored/v1" in text


def test_r149_cuda_runner_accepts_only_the_four_frozen_roles():
    text = RUNNER.read_text()
    assert "^(baseline|candidate)-c(1|32)$" in text
    assert "/data/dyy/logs/minicpmo-npu-quality-r149-inputs/*" in text
