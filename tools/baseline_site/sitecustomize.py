"""Compatibility shim used only by the frozen vLLM-Omni NPU baseline.

The pinned vLLM-Ascend checkout selects its private AddRmsNormBias custom op
on 910B1 whenever the extension imports successfully.  This image's CANN
``libopapi.so`` does not export that operator, although the public
``torch_npu.npu_add_rms_norm`` operation is available.  vLLM-Omni starts each
stage with multiprocessing ``spawn``, so a gated sitecustomize hook is the
smallest way to install the same fallback in every stage without editing the
shared baseline checkout.
"""

from __future__ import annotations

import os
import warnings


if os.environ.get("MINICPMO_VLLM_ASCEND_RMSNORM_FALLBACK") == "1":
    from vllm_ascend.ops.layernorm import AscendRMSNorm

    def _forward_oot_without_private_add_rms_norm_bias(self, x, residual=None):
        import torch_npu

        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(
                x,
                residual,
                self.weight,
                self.variance_epsilon,
            )
            if self.bias_loaded:
                x.add_(self.bias)
            return x, residual

        x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
        if self.bias_loaded:
            x.add_(self.bias)
        return x

    AscendRMSNorm.forward_oot = _forward_oot_without_private_add_rms_norm_bias
    warnings.warn(
        "Installed MiniCPM-o baseline RMSNorm compatibility fallback: "
        "torch_npu.npu_add_rms_norm replaces unavailable aclnnAddRmsNormBias.",
        RuntimeWarning,
        stacklevel=1,
    )
