from __future__ import annotations

import pytest

from engine.runtime.npu_paged import normalize_decoder_op_mode


@pytest.mark.parametrize(
    ("fused_ops", "fused_mode", "expected"),
    [
        (False, None, "reference"),
        (True, None, "full"),
        (False, "rms", "rms"),
        (False, "add-rms", "add-rms"),
        (False, "swiglu", "swiglu"),
        (True, "reference", "reference"),
    ],
)
def test_normalize_decoder_op_mode(fused_ops, fused_mode, expected):
    assert normalize_decoder_op_mode(fused_ops, fused_mode) == expected


def test_normalize_decoder_op_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported decoder op mode"):
        normalize_decoder_op_mode(True, "mystery")
