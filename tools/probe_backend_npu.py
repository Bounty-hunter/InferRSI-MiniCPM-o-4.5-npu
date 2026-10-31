"""End-to-end in-process smoke of the dedicated backend on one Ascend NPU."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np

from engine.runtime.backend_npu import AscendTrackSBackend
from engine.server import AudioChunk, Done, EngineRequest, TextDelta


MODEL = Path("/dataset/models/MiniCPM-o-4_5")


async def run() -> None:
    started = time.perf_counter()
    full = os.environ.get("SMOKE_FULL_KV") == "1"
    graphs = os.environ.get("NPU_USE_GRAPHS") == "1"
    backend = AscendTrackSBackend(model_dir=MODEL, max_live=32 if full else 1,
                                  thinker_pages=512 if full else 64, talker_pages=768 if full else 128,
                                  page_size=128, use_graphs=graphs)
    ready = time.perf_counter()
    request = EngineRequest(
        request_id="npu-smoke-0",
        messages=[{"role": "user", "content": "Please say hello."}],
        modalities=["text", "audio"],
        max_tokens=4,
        temperature=0.0,
        seed=7,
        chat_template_kwargs={"use_tts_template": True, "enable_thinking": False},
        ref_audio=(MODEL / "assets" / "HT_ref_audio.wav").read_bytes(),
        talker_sampling={"temperature": 0.8, "top_p": 0.85, "top_k": 25, "repetition_penalty": 1.05,
                         "repetition_window": 16, "min_new_tokens": 0, "max_new_tokens": 28},
        stream=True,
    )
    text, audio, first_audio, done = [], [], None, None
    async with asyncio.timeout(300):
        async for event in backend.generate(request):
            if isinstance(event, TextDelta):
                text.append(event.text)
            elif isinstance(event, AudioChunk):
                first_audio = first_audio or time.perf_counter()
                audio.append(event.samples)
            elif isinstance(event, Done):
                done = event
    ended = time.perf_counter()
    wav = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
    payload = {
        "backend_load_s": ready - started,
        "request_s": ended - ready,
        "ttfa_s": None if first_audio is None else first_audio - ready,
        "text": "".join(text),
        "audio_events": len(audio),
        "audio_samples": int(wav.size),
        "audio_finite": bool(np.isfinite(wav).all()),
        "done": done is not None,
        "stats": backend.stats(),
        "thinker_graph_failures": backend.trun.graph_failures,
        "talker_graph_failures": backend.arun.graph_failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["done"] or not payload["audio_finite"] or payload["audio_samples"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(run())
