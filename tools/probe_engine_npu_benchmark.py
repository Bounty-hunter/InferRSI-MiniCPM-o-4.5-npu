"""Development C1/C16/C32 benchmark for the standalone Ascend backend.

This keeps one backend alive for a same-shape warmup wave and one measured
closed-loop wave.  It is intentionally smaller than the frozen K=5/K=10
campaign, but uses the same latency and audio-throughput definitions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from engine.runtime.backend_npu import AscendTrackSBackend
from engine.server import AudioChunk, Done, EngineRequest


MODEL = Path("/dataset/models/MiniCPM-o-4_5")
REF_AUDIO = MODEL / "assets/HT_ref_audio.wav"


def request(index: int, max_text_tokens: int, max_audio_tokens: int) -> EngineRequest:
    return EngineRequest(
        request_id=f"dev-{index:04d}",
        messages=[{"role": "user", "content": "The dedicated engine is measuring streaming speech latency."}],
        modalities=["text", "audio"],
        max_tokens=max_text_tokens,
        temperature=0.0,
        seed=20261022 + index,
        chat_template_kwargs={"use_tts_template": True, "enable_thinking": False},
        ref_audio=REF_AUDIO.read_bytes(),
        talker_sampling={
            "temperature": 0.8,
            "top_p": 0.85,
            "top_k": 25,
            "repetition_penalty": 1.05,
            "repetition_window": 16,
            "min_new_tokens": 50,
            "max_new_tokens": max_audio_tokens,
        },
        stream=True,
    )


async def one(backend: AscendTrackSBackend, req: EngineRequest) -> dict:
    started = time.perf_counter()
    first_audio = None
    samples = 0
    audio_digest = hashlib.sha256()
    done = None
    try:
        async for event in backend.generate(req):
            if isinstance(event, AudioChunk):
                first_audio = first_audio or time.perf_counter()
                chunk = np.asarray(event.samples, dtype=np.float32)
                samples += int(chunk.size)
                audio_digest.update(np.ascontiguousarray(chunk).tobytes())
            elif isinstance(event, Done):
                done = event
        ended = time.perf_counter()
        return {
            "ok": done is not None and samples > 0,
            "ttfa_s": None if first_audio is None else first_audio - started,
            "e2e_s": ended - started,
            "audio_s": samples / 24000.0,
            "samples": samples,
            "audio_sha256": audio_digest.hexdigest(),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "ttfa_s": None,
            "e2e_s": time.perf_counter() - started,
            "audio_s": 0.0,
            "samples": 0,
            "audio_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def wave(backend, concurrency: int, max_text_tokens: int, max_audio_tokens: int, offset: int,
               n_requests: int | None = None) -> dict:
    count = concurrency if n_requests is None else n_requests
    reqs = [request(offset + i, max_text_tokens, max_audio_tokens) for i in range(count)]
    started = time.perf_counter()
    rows = await asyncio.gather(*(one(backend, req) for req in reqs))
    wall = time.perf_counter() - started
    ttfa = sorted(row["ttfa_s"] for row in rows if row["ttfa_s"] is not None)
    e2e = sorted(row["e2e_s"] for row in rows)
    audio_s = sum(row["audio_s"] for row in rows)
    return {
        "concurrency": concurrency,
        "n_ok": sum(row["ok"] for row in rows),
        "n_failed": sum(not row["ok"] for row in rows),
        "wall_s": wall,
        "audio_s": audio_s,
        "throughput_audio_s_per_s": audio_s / wall,
        "ttfa_s": ttfa,
        "ttfa_p50_s": statistics.median(ttfa) if ttfa else None,
        "e2e_s": e2e,
        "e2e_p50_s": statistics.median(e2e),
        "rtf": [row["e2e_s"] / row["audio_s"] if row["audio_s"] else None for row in rows],
        "failures": [row for row in rows if not row["ok"]],
        "audio_sha256": [row["audio_sha256"] for row in rows],
    }


async def run(args) -> dict:
    cache_offload = args.concurrency >= 16
    if args.t2w_cache_offload != "auto":
        cache_offload = args.t2w_cache_offload == "on"
    loaded_at = time.perf_counter()
    backend = AscendTrackSBackend(
        model_dir=MODEL,
        float16_flow=args.float16_flow,
        max_live=32,
        # Sized for C32 of this 16-text/128-audio development workload rather
        # than reserving the production maximum and starving Token2Wav caches.
        thinker_pages=128,
        talker_pages=128,
        page_size=128,
        fused_decoder_ops=True,
        use_graphs=True,
        graph_max_batch=32,
        talker_k=args.talker_k,
        talker_k_max=8,
        first_chunk=args.first_chunk,
        steady_chunk=args.steady_chunk,
        spk_threads=args.spk_threads,
        t2w_cache_offload=cache_offload,
        t2w_euler_graphs=args.t2w_euler_graphs,
        t2w_hift_graphs=args.t2w_hift_graphs,
        t2w_encoder_graphs=args.t2w_encoder_graphs,
        t2w_threaded=args.t2w_threaded,
        t2w_batch_wait_ms=args.t2w_batch_wait_ms,
        t2w_timesteps=args.t2w_timesteps,
        talker_device_blocks=args.talker_device_blocks,
        timeline_markers=args.timeline_markers,
        profile_stages=args.profile_stages,
    )
    ready_at = time.perf_counter()
    warmup = await wave(backend, args.concurrency, args.max_text_tokens, args.warmup_audio_tokens, 10000)
    if warmup["n_failed"]:
        raise RuntimeError(f"warmup failed: {warmup['failures']}")
    if args.profile_stages:
        backend.tick_profile.clear()
        backend.t2w_batch.profile.clear()
        backend.prompt_profile.clear()
        backend.t2w.feature_profile.clear()
    if args.timeline_markers:
        backend.timelines.clear()
    measured = await wave(backend, args.concurrency, args.max_text_tokens, args.max_audio_tokens, 0, args.requests)
    measured.update({
        "engine": "minicpmo-ascend-dedicated",
        "model": str(MODEL),
        "backend_load_s": ready_at - loaded_at,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "warmup_wall_s": warmup["wall_s"],
        "stats": backend.stats(),
        "thinker_graph_failures": backend.trun.graph_failures,
        "talker_graph_failures": backend.arun.graph_failures,
        "stage_profile": backend.tick_profile,
        "token2wav_profile": backend.t2w_batch.profile,
        "prompt_profile": backend.prompt_profile,
        "feature_profile": backend.t2w.feature_profile,
        "timelines": backend.timelines,
    })
    return measured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, choices=(1, 16, 32), required=True)
    parser.add_argument("--requests", type=int, default=None,
                        help="total measured requests; defaults to one closed-loop wave of CONCURRENCY requests")
    parser.add_argument("--max-text-tokens", type=int, default=16)
    parser.add_argument("--max-audio-tokens", type=int, default=128)
    parser.add_argument("--warmup-audio-tokens", type=int, default=64)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--t2w-euler-graphs", action="store_true")
    parser.add_argument("--t2w-hift-graphs", action="store_true")
    parser.add_argument("--t2w-encoder-graphs", action="store_true")
    parser.add_argument("--t2w-threaded", action="store_true")
    parser.add_argument("--t2w-batch-wait-ms", type=float, default=0.0)
    parser.add_argument("--t2w-cache-offload", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--t2w-timesteps", type=int, default=10)
    parser.add_argument("--talker-device-blocks", action="store_true")
    parser.add_argument("--talker-k", type=int, default=4)
    parser.add_argument("--spk-threads", type=int, default=2)
    parser.add_argument("--first-chunk", type=int, default=8)
    parser.add_argument("--steady-chunk", type=int, default=25)
    parser.add_argument("--float16-flow", action="store_true")
    parser.add_argument("--timeline-markers", action="store_true")
    args = parser.parse_args()
    if args.requests is not None and args.requests < args.concurrency:
        parser.error("--requests must be at least --concurrency")
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["n_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
