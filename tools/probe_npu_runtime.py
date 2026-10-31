"""Minimal scheduled Ascend runtime probe with no engine-framework imports."""

from __future__ import annotations

import json
import os

import torch
import torch_npu


def main() -> None:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    count = int(torch.npu.device_count())
    available = bool(torch.npu.is_available())
    if not visible:
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES is not set by the scheduler")
    if count != 1 or not available:
        raise RuntimeError(f"invalid scheduled NPU visibility: visible={visible!r} count={count} available={available}")

    tensor = torch.ones(4, dtype=torch.float32, device="npu")
    observed = float(tensor.sum().cpu().item())
    if observed != 4.0:
        raise RuntimeError(f"unexpected NPU result: {observed}")

    print(
        json.dumps(
            {
                "visible_devices": visible,
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "device_count": count,
                "available": available,
                "sum": observed,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
