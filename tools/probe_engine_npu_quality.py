"""Generate a frozen Seed-TTS quality slice with the standalone NPU backend.

This is the fast quality screen used before a full three-seed/1088-row cell.
The row split, prompt, sampling parameters, and per-request seeds come from
``bench.workload``.  WAVs are written through the public PCM16 wire encoder so
the scorer sees exactly the audio an OpenAI-compatible client receives.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.audio import encode_wav_pcm16
from bench.workload import CanonicalRequest, canonical_requests, load_seed_tts_rows, make_id_sets
from engine.server import AudioChunk, Done, EngineRequest, TextDelta

if TYPE_CHECKING:
    from engine.runtime.backend_npu import AscendTrackSBackend


MODEL = Path("/dataset/models/MiniCPM-o-4_5")
SEED_TTS = Path("/dataset/models/seed-tts-eval")


def engine_request(req: CanonicalRequest) -> EngineRequest:
    return EngineRequest(
        request_id=req.request_id,
        messages=req.messages(),
        modalities=["text", "audio"],
        max_tokens=req.thinker.max_tokens,
        temperature=req.thinker.temperature,
        seed=req.seed,
        chat_template_kwargs={"use_tts_template": req.use_tts_template, "enable_thinking": req.enable_thinking},
        ref_audio=req.ref_audio_bytes(),
        talker_sampling={
            "temperature": req.talker.temperature,
            "top_p": req.talker.top_p,
            "top_k": req.talker.top_k,
            "repetition_penalty": req.talker.repetition_penalty,
            "repetition_window": req.talker.repetition_window,
            "min_new_tokens": req.talker.min_new_tokens,
            "max_new_tokens": req.talker.max_new_tokens,
        },
        stream=req.stream,
    )


async def one(backend: "AscendTrackSBackend", canonical: CanonicalRequest) -> dict:
    started = time.perf_counter()
    first_audio = None
    chunks: list[np.ndarray] = []
    text: list[str] = []
    done = None
    error = None
    try:
        async for event in backend.generate(engine_request(canonical)):
            if isinstance(event, TextDelta):
                text.append(event.text)
            elif isinstance(event, AudioChunk):
                first_audio = first_audio or time.perf_counter()
                chunks.append(np.asarray(event.samples, dtype=np.float32))
            elif isinstance(event, Done):
                done = event
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    finite = bool(np.isfinite(audio).all())
    ok = done is not None and audio.size > 0 and finite and error is None
    return {
        "request_id": canonical.request_id,
        "seed": canonical.seed,
        "target_text": canonical.row.target_text,
        "ref_text": canonical.row.ref_text,
        "ref_wav": canonical.row.ref_wav_path,
        "thinker_text": "".join(text),
        "ok": ok,
        "ttfa_s": None if first_audio is None else first_audio - started,
        "e2e_s": ended - started,
        "samples": int(audio.size),
        "duration_s": audio.size / 24000.0,
        "audio_sha256_float32": hashlib.sha256(np.ascontiguousarray(audio).tobytes()).hexdigest() if audio.size else None,
        "finish_reason": done.finish_reason if done is not None else None,
        "usage": dict(done.usage) if done is not None else {},
        "error": error if error else (None if ok else "missing Done or finite audio"),
        "audio": audio,
    }


async def closed_loop(backend: "AscendTrackSBackend", requests: list[CanonicalRequest], concurrency: int) -> tuple[list[dict], float]:
    queue: asyncio.Queue[CanonicalRequest] = asyncio.Queue()
    for req in requests:
        queue.put_nowait(req)
    results: list[dict] = []

    async def worker() -> None:
        while True:
            try:
                req = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await one(backend, req))
            queue.task_done()

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(requests)))))
    return results, time.perf_counter() - started


def build_backend(args) -> "AscendTrackSBackend":
    from engine.runtime.backend_npu import AscendTrackSBackend

    # Preserve stage-0's transposed convolution in FP32.  This is the
    # repository's bounded HiFT reduced-precision variant (AC-3.4).
    hift_hf32 = [1, 2, "0:src", "0:res"] if args.t2w_hift_hf32 else False
    return AscendTrackSBackend(
        model_dir=MODEL,
        float16_flow=True,
        max_live=32,
        thinker_pages=args.thinker_pages,
        talker_pages=args.talker_pages,
        page_size=128,
        fused_decoder_ops=args.fused_decoder_ops == "on",
        thinker_decoder_op_mode=args.thinker_decoder_op_mode,
        talker_decoder_op_mode=args.talker_decoder_op_mode,
        use_graphs=args.use_graphs == "on",
        graph_max_batch=args.graph_max_batch,
        talker_static_shape_attention=args.talker_static_shape_attention,
        talker_static_kv_capacity=args.talker_static_kv_capacity,
        talker_k=args.talker_k,
        talker_k_max=args.talker_k,
        first_chunk=args.first_chunk,
        steady_chunk=args.steady_chunk,
        spk_threads=2,
        t2w_cache_offload=args.t2w_cache_offload == "on",
        t2w_euler_graphs=True,
        t2w_hift_graphs=True,
        t2w_hift_decode_tf32=hift_hf32,
        t2w_encoder_graphs=args.t2w_encoder_graphs,
        t2w_threaded=True,
        t2w_batch_wait_ms=args.t2w_batch_wait_ms,
        t2w_timesteps=args.t2w_timesteps,
        talker_device_blocks=args.talker_device_blocks,
        talker_post_graphs=args.talker_post_graphs,
        talker_sampling_semantics=args.talker_sampling_semantics,
        profile_stages=args.profile_stages,
        stage_trace_dir=str(args.stage_trace_dir) if args.stage_trace_dir is not None else None,
    )


async def run(args) -> dict:
    rows = load_seed_tts_rows(SEED_TTS)
    ids = make_id_sets(rows)
    selected = ids.measured(args.id_set, args.concurrency)[: args.limit]
    if args.request_id:
        available = {row.row_id for row in rows}
        missing = [request_id for request_id in args.request_id if request_id not in available]
        if missing:
            raise ValueError(f"request IDs are not in Seed-TTS: {missing}")
        selected = list(dict.fromkeys(args.request_id))
    measured = canonical_requests(rows, selected, seed_index=args.seed_index)
    warm_ids = ids.warmup(args.concurrency)
    warmup = canonical_requests(rows, warm_ids, seed_index=args.seed_index)

    loaded_at = time.perf_counter()
    backend = build_backend(args)
    ready_at = time.perf_counter()
    warm_rows, warm_wall = await closed_loop(backend, warmup, args.concurrency)
    warm_failures = [r for r in warm_rows if not r["ok"]]
    if warm_failures:
        raise RuntimeError(f"warmup failed: {warm_failures[:3]}")
    if args.profile_stages:
        backend.tick_profile.clear()
        backend.prompt_profile.clear()
        backend.t2w_batch.profile.clear()
        backend.t2w.feature_profile.clear()
        backend.timelines.clear()
    if args.talker_profile_dir is None:
        results, wall = await closed_loop(backend, measured, args.concurrency)
    else:
        from torch_npu.profiler import ProfilerActivity, profile, schedule

        args.talker_profile_dir.mkdir(parents=True, exist_ok=False)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
            with_stack=True,
        )
        prof.start()
        profile.enable_profiler_in_child_thread(with_stack=True)
        try:
            results, wall = await closed_loop(backend, measured, args.concurrency)
        finally:
            profile.disable_profiler_in_child_thread()
            prof.step()
            prof.stop()
            prof.export_chrome_trace(str(args.talker_profile_dir / "measured.json"))

    args.out.mkdir(parents=True, exist_ok=False)
    wav_dir = args.out / "wav"
    wav_dir.mkdir()
    manifest = []
    for row in sorted(results, key=lambda x: x["request_id"]):
        audio = row.pop("audio")
        wav = encode_wav_pcm16(audio, 24000)
        if row["ok"]:
            (wav_dir / f"{row['request_id']}.wav").write_bytes(wav)
        row["wav"] = str((wav_dir / f"{row['request_id']}.wav").resolve()) if row["ok"] else None
        row["audio_sha256_pcm16_wav"] = hashlib.sha256(wav).hexdigest() if row["ok"] else None
        manifest.append(row)

    ttfa = sorted(r["ttfa_s"] for r in manifest if r["ttfa_s"] is not None)
    e2e = sorted(r["e2e_s"] for r in manifest)
    audio_s = sum(r["duration_s"] for r in manifest)
    report = {
        "schema": "npu-quality-slice/v1",
        "engine": "minicpmo-ascend-dedicated",
        "model": str(MODEL),
        "dataset": {
            "repo": "zhaochenyang20/seed-tts-eval",
            "revision": "8f5e1aa2a35d42f42e940074c1983358b9491f89",
            "root": str(SEED_TTS),
            "meta_sha256": hashlib.sha256((SEED_TTS / "en/meta.lst").read_bytes()).hexdigest(),
        },
        "id_set": args.id_set,
        "seed_index": args.seed_index,
        "concurrency": args.concurrency,
        "t2w_timesteps": args.t2w_timesteps,
        "first_chunk": args.first_chunk,
        "steady_chunk": args.steady_chunk,
        "t2w_cache_offload": args.t2w_cache_offload,
        "t2w_batch_wait_ms": args.t2w_batch_wait_ms,
        "t2w_encoder_graphs": args.t2w_encoder_graphs,
        "t2w_hift_hf32": args.t2w_hift_hf32,
        "thinker_pages": args.thinker_pages,
        "talker_pages": args.talker_pages,
        "graph_max_batch": args.graph_max_batch,
        "fused_decoder_ops": args.fused_decoder_ops,
        "thinker_decoder_op_mode": args.thinker_decoder_op_mode,
        "talker_decoder_op_mode": args.talker_decoder_op_mode,
        "use_graphs": args.use_graphs,
        "talker_static_shape_attention": args.talker_static_shape_attention,
        "talker_static_kv_capacity": args.talker_static_kv_capacity,
        "talker_device_blocks": args.talker_device_blocks,
        "talker_k": args.talker_k,
        "talker_post_graphs": args.talker_post_graphs,
        "talker_sampling_semantics": args.talker_sampling_semantics,
        "profile_stages": args.profile_stages,
        "stage_trace_dir": str(args.stage_trace_dir) if args.stage_trace_dir is not None else None,
        "talker_profile_dir": str(args.talker_profile_dir) if args.talker_profile_dir is not None else None,
        "n_requested": len(measured),
        "n_ok": sum(r["ok"] for r in manifest),
        "n_failed": sum(not r["ok"] for r in manifest),
        "wall_s": wall,
        "audio_s": audio_s,
        "throughput_audio_s_per_s": audio_s / wall,
        "ttfa_p50_s": statistics.median(ttfa) if ttfa else None,
        "e2e_p50_s": statistics.median(e2e) if e2e else None,
        "backend_load_s": ready_at - loaded_at,
        "warmup_wall_s": warm_wall,
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "stats": backend.stats(),
        "rows": manifest,
    }
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 16, 32), default=16)
    parser.add_argument("--id-set", choices=("tuning", "eval", "full"), default="tuning")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--request-id", action="append", default=[], help="Run these exact Seed-TTS row IDs instead of the leading slice")
    parser.add_argument("--t2w-timesteps", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--first-chunk", type=int, default=8)
    parser.add_argument("--steady-chunk", type=int, default=25)
    parser.add_argument("--t2w-cache-offload", choices=("on", "off"), default="on")
    parser.add_argument("--t2w-batch-wait-ms", type=float, default=0.0)
    parser.add_argument("--t2w-encoder-graphs", action="store_true")
    parser.add_argument("--t2w-hift-hf32", action="store_true")
    parser.add_argument("--thinker-pages", type=int, default=128)
    parser.add_argument("--talker-pages", type=int, default=128)
    parser.add_argument("--graph-max-batch", type=int, choices=(1, 2, 4, 8, 16, 32), default=32)
    parser.add_argument("--fused-decoder-ops", choices=("on", "off"), default="on")
    parser.add_argument("--thinker-decoder-op-mode", choices=("reference", "rms", "add-rms", "swiglu", "full"))
    parser.add_argument("--talker-decoder-op-mode", choices=("reference", "rms", "add-rms", "swiglu", "full"))
    parser.add_argument("--use-graphs", choices=("on", "off"), default="on")
    parser.add_argument("--talker-static-shape-attention", action="store_true")
    parser.add_argument("--talker-static-kv-capacity", type=int, default=512)
    parser.add_argument("--talker-device-blocks", action="store_true")
    parser.add_argument("--talker-k", type=int, choices=range(4, 17), default=8)
    parser.add_argument("--talker-post-graphs", action="store_true")
    parser.add_argument("--talker-sampling-semantics", choices=("hf", "omni"), default="hf")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--stage-trace-dir", type=Path)
    parser.add_argument("--talker-profile-dir", type=Path)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.first_chunk <= 0:
        parser.error("--first-chunk must be positive")
    if args.steady_chunk < 25 or args.first_chunk > args.steady_chunk:
        parser.error("--steady-chunk must be >= 25 and >= --first-chunk")
    if args.t2w_batch_wait_ms < 0:
        parser.error("--t2w-batch-wait-ms must be non-negative")
    if args.thinker_pages <= 0 or args.talker_pages <= 0:
        parser.error("--thinker-pages and --talker-pages must be positive")
    if args.talker_static_kv_capacity <= 0:
        parser.error("--talker-static-kv-capacity must be positive")
    result = asyncio.run(run(args))
    print(json.dumps({k: result[k] for k in ("n_requested", "n_ok", "n_failed", "wall_s", "audio_s", "throughput_audio_s_per_s", "ttfa_p50_s", "e2e_p50_s")}, indent=2))
    if result["n_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
