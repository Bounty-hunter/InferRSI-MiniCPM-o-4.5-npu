"""Load and exercise the request-owned Token2Wav path on Ascend."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from engine.models.token2wav import Token2Wav


ASSETS = "/dataset/models/MiniCPM-o-4_5/assets"


@torch.inference_mode()
def main() -> None:
    torch.npu.set_device(0)
    started = time.perf_counter()
    model = Token2Wav(f"{ASSETS}/token2wav", device="npu:0")
    loaded = time.perf_counter()
    prompt = model.prepare(f"{ASSETS}/HT_ref_audio.wav")
    prepared = time.perf_counter()
    state = model.new_stream(prompt, noise_seed=7)
    # A legal 28-frame streaming window; the last flag exercises the full flush.
    wav = model.decode(state, [100 + i for i in range(28)], last=True)
    torch.npu.synchronize()
    payload = {
        "device": torch.npu.get_device_name(0),
        "load_s": loaded - started,
        "prepare_s": prepared - loaded,
        "decode_s": time.perf_counter() - prepared,
        "samples": int(wav.size),
        "finite": bool(np.isfinite(wav).all()),
        "peak": float(np.abs(wav).max(initial=0.0)),
        "npu_allocated_bytes": int(torch.npu.memory_allocated()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["finite"] or payload["samples"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
