"""``t2w_hift_decode_tf32`` values and accelerator precision contexts (CPU)."""

from types import SimpleNamespace

import pytest
import torch

from engine.models.token2wav import PARTS, decoder_tf32, tf32_stages


def test_tf32_stage_selection():
    every = {(i, q) for i in range(3) for q in PARTS}
    assert tf32_stages(True, 3) == every
    assert tf32_stages(False, 3) == tf32_stages(None, 3) == tf32_stages([], 3) == frozenset()
    assert tf32_stages([1, 2, "0:src", "0:res"], 3) == every - {(0, "ups")}  # the B24 repair
    assert tf32_stages(["2"], 3) == {(2, q) for q in PARTS}
    for bad in ([3], ["0:conv"], [-1]):
        with pytest.raises(AssertionError):
            tf32_stages(bad, 3)


def test_decoder_tf32_toggles_npu_hf32(monkeypatch):
    conv = SimpleNamespace(allow_hf32=False)
    matmul = SimpleNamespace(allow_hf32=False)
    fake_npu = SimpleNamespace(is_available=lambda: True, conv=conv, matmul=matmul)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    with decoder_tf32():
        assert conv.allow_hf32 is True
        assert matmul.allow_hf32 is True
    assert conv.allow_hf32 is False
    assert matmul.allow_hf32 is False
