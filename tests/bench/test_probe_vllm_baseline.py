import json
from pathlib import Path

import pytest

from bench.adapters.vllm_omni import stage_sampling
from tools import probe_vllm_baseline as probe
from tools.probe_vllm_baseline import (
    FROZEN_REQUEST_TIMEOUT_S,
    make_stage_overrides,
    materialize_deploy_overlay,
    omni_timing_count,
    resolve_warmup_timeout,
    wait_for_omni_timings,
)


def test_frozen_talker_sampling_preserves_codec_eos():
    params = stage_sampling(probe.request(0, max_text_tokens=16, max_audio_tokens=128))

    assert params[1]["stop_token_ids"] == [6561]


def test_start_server_disables_recursive_torch_backend_autoload(monkeypatch, tmp_path: Path):
    captured = {}

    class _Proc:
        returncode = None

        @staticmethod
        def poll():
            return None

    def popen(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return _Proc()

    monkeypatch.setattr(probe.subprocess, "Popen", popen)
    monkeypatch.setattr(probe, "healthy", lambda _url: True)

    chat_template = tmp_path / probe.CHAT_TEMPLATE_RELATIVE
    chat_template.parent.mkdir(parents=True)
    chat_template.write_text("{{ messages }}\n", encoding="utf-8")

    proc, _startup_s = probe.start_server(
        port=18100,
        log_path=tmp_path / "server.log",
        deploy_config=tmp_path / "deploy.yaml",
        baseline_root=tmp_path,
    )

    assert isinstance(proc, _Proc)
    assert captured["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    cmd = captured["args"][0]
    assert cmd[cmd.index("--chat-template") + 1] == str(chat_template)
    assert cmd[cmd.index("--chat-template-content-format") + 1] == "openai"


def test_harness_launcher_disables_recursive_torch_backend_autoload():
    launcher = Path(__file__).parents[2] / "scripts" / "serve_vllm_omni.sh"
    source = launcher.read_text(encoding="utf-8")

    assert "export TORCH_DEVICE_BACKEND_AUTOLOAD=0" in source
    assert '! getent passwd "$(id -u)"' in source
    assert 'export HOME="/tmp/minicpmo-baseline-home-$(id -u)"' in source
    assert "unset HF_HOME HF_HUB_CACHE HF_MODULES_CACHE" in source
    assert source.index('export HOME="/tmp/minicpmo-baseline-home-$(id -u)"') < source.index('source "$(dirname "$0")/env.sh"')


def test_start_server_requires_frozen_native_chat_template(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="native.*chat template"):
        probe.start_server(
            port=18100,
            log_path=tmp_path / "server.log",
            deploy_config=tmp_path / "deploy.yaml",
            baseline_root=tmp_path,
        )


def test_write_result_is_immediately_readable(tmp_path: Path):
    output = tmp_path / "nested" / "result.json"

    probe.write_result(output, {"n_ok": 1, "text": "spoken"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"n_ok": 1, "text": "spoken"}
    assert not output.with_name(output.name + ".tmp").exists()


def test_materialize_deploy_overlay_records_frame_bucket(tmp_path: Path):
    base = tmp_path / "minicpmo_4_5_turn.yaml"
    base.write_text("session_mode: turn\n", encoding="utf-8")
    output = tmp_path / "probe.deploy.json"

    resolved = materialize_deploy_overlay(base, output, 16)

    assert resolved == output
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "base_config": str(base.resolve()),
        "connectors": {
            "connector_of_shared_memory": {
                "extra": {"cfm_graph_bucket_frames": 16},
            },
        },
    }


def test_materialize_deploy_overlay_is_noop_without_bucket(tmp_path: Path):
    base = tmp_path / "minicpmo_4_5_turn.yaml"
    output = tmp_path / "probe.deploy.json"

    assert materialize_deploy_overlay(base, output, None) == base
    assert not output.exists()


def test_resolve_warmup_timeout_preserves_frozen_request_timeout_by_default():
    assert FROZEN_REQUEST_TIMEOUT_S == 180.0
    assert resolve_warmup_timeout(None, FROZEN_REQUEST_TIMEOUT_S) == 180.0
    assert resolve_warmup_timeout(600.0, FROZEN_REQUEST_TIMEOUT_S) == 600.0


def test_omni_timing_count_handles_missing_and_partial_lines(tmp_path: Path):
    log = tmp_path / "server.log"
    assert omni_timing_count(log) == 0

    log.write_bytes(b"prefix [OmniTiming] one\n[OmniTiming] two\npartial [Omni")
    assert omni_timing_count(log) == 2


def test_wait_for_omni_timings_returns_observed_barrier(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text("[OmniTiming] first\n[OmniTiming] second\n", encoding="utf-8")

    result = wait_for_omni_timings(log, expected=2, timeout_s=0.1)

    assert result["expected_timings"] == 2
    assert result["observed_timings"] == 2
    assert result["elapsed_s"] >= 0


def test_wait_for_omni_timings_rejects_incomplete_drain(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text("[OmniTiming] only\n", encoding="utf-8")

    with pytest.raises(TimeoutError, match=r"observed 1/2"):
        wait_for_omni_timings(log, expected=2, timeout_s=0)


def test_stage2_graph_limit_is_scoped_to_code2wav_additional_config():
    overrides = make_stage_overrides(
        stage2_max_num_seqs=32,
        stage0_max_model_len=32768,
        stage0_enforce_eager=True,
        stage0_generation_config_vllm=True,
        stage0_async_scheduling=False,
        stage2_max_npu_graphs=32,
        stage2_graph_batch_buckets=[1, 2, 4, 8, 16, 32],
    )

    assert overrides["2"]["max_num_seqs"] == 32
    assert overrides["0"]["max_model_len"] == 32768
    assert overrides["0"]["enforce_eager"] is True
    assert overrides["0"]["generation_config"] == "vllm"
    assert overrides["0"]["async_scheduling"] is False
    assert overrides["2"]["additional_config"]["code2wav_max_npu_graphs"] == 32
    assert overrides["2"]["additional_config"]["cfm_graph_batch_buckets"] == [1, 2, 4, 8, 16, 32]
    assert "code2wav_max_npu_graphs" not in overrides["0"]["additional_config"]
    assert "code2wav_max_npu_graphs" not in overrides["1"]["additional_config"]
    assert "cfm_graph_batch_buckets" not in overrides["0"]["additional_config"]
    assert "cfm_graph_batch_buckets" not in overrides["1"]["additional_config"]
