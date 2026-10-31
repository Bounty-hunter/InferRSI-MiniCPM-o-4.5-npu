from tools.probe_vllm_omni_text import build_payload


def test_text_only_probe_keeps_omni_scheduler_and_skips_audio_stages():
    payload = build_payload()

    assert payload["modalities"] == ["text"]
    assert payload["stream"] is False
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 16
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False,
        "use_tts_template": True,
    }
    assert "ref_audio" not in payload
    assert "sampling_params_list" not in payload
