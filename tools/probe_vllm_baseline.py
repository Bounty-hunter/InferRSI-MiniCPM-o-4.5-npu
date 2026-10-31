"""Start the frozen vLLM-Omni NPU baseline and measure one closed-loop wave.

This is a development probe, not a substitute for the frozen K=5/K=10
campaign.  It exists so stage gaps can be measured before the Seed-TTS assets
and full harness are available on a runner.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from bench.adapters.vllm_omni import VllmOmniAdapter
from bench.client import make_session, send_request
from bench.metrics import request_metrics
from bench.workload import (
    CanonicalRequest,
    SeedTTSRow,
    TalkerSampling,
    ThinkerSampling,
    canonical_requests,
    load_seed_tts_rows,
    make_id_sets,
)


BASELINE = Path("/vllm-workspace/vllm-omni")
MODEL = Path("/dataset/models/MiniCPM-o-4_5")
DEPLOY = BASELINE / "vllm_omni/deploy/minicpmo_4_5_turn.yaml"
CHAT_TEMPLATE_RELATIVE = Path("vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja")
REF_AUDIO = MODEL / "assets/HT_ref_audio.wav"
BASELINE_SITE = Path(__file__).with_name("baseline_site")
FROZEN_REQUEST_TIMEOUT_S = 180.0


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def omni_timing_count(log_path: Path) -> int:
    """Return the number of requests that fully left the Omni pipeline."""
    try:
        return log_path.read_bytes().count(b"[OmniTiming]")
    except FileNotFoundError:
        return 0


def wait_for_omni_timings(log_path: Path, expected: int, timeout_s: float) -> dict:
    """Wait until timed-out warmups have actually drained server-side.

    aiohttp cancelling a timed-out streaming response does not reliably cancel
    the corresponding vLLM-Omni stage work.  Starting the measured wave while
    those requests are still running silently turns it into a warmup tail
    measurement.  Omni emits exactly one ``[OmniTiming]`` record after a
    request leaves the complete pipeline, so use that as the drain barrier.
    """
    started = time.monotonic()
    deadline = started + timeout_s
    observed = omni_timing_count(log_path)
    while observed < expected and time.monotonic() < deadline:
        time.sleep(0.5)
        observed = omni_timing_count(log_path)
    elapsed = time.monotonic() - started
    if observed < expected:
        raise TimeoutError(
            f"baseline warmup did not drain after {elapsed:.1f}s: "
            f"observed {observed}/{expected} [OmniTiming] records in {log_path}"
        )
    return {"elapsed_s": elapsed, "expected_timings": expected, "observed_timings": observed}


def resolve_warmup_timeout(warmup_timeout: float | None, request_timeout: float) -> float:
    """Keep the measured request timeout frozen while allowing slow warmup."""
    return request_timeout if warmup_timeout is None else warmup_timeout


def materialize_deploy_overlay(
    deploy_config: Path,
    output_path: Path,
    cfm_graph_bucket_frames: int | None,
) -> Path:
    """Create a thin, recorded deploy overlay for CFM frame bucketing."""
    if cfm_graph_bucket_frames is None:
        return deploy_config
    overlay = {
        "base_config": str(deploy_config.resolve()),
        "connectors": {
            "connector_of_shared_memory": {
                "extra": {"cfm_graph_bucket_frames": cfm_graph_bucket_frames},
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def make_stage_overrides(
    stage2_max_num_seqs: int | None = None,
    stage0_max_model_len: int | None = None,
    stage0_enforce_eager: bool = False,
    stage0_generation_config_vllm: bool = False,
    stage0_async_scheduling: bool | None = None,
    stage0_gpu_memory_utilization: float | None = None,
    stage1_gpu_memory_utilization: float | None = None,
    stage2_gpu_memory_utilization: float | None = None,
    stage2_max_npu_graphs: int | None = None,
    stage2_graph_batch_buckets: list[int] | tuple[int, ...] | None = None,
) -> dict[str, dict]:
    stage_overrides = {
        str(stage): {
            "compilation_config": {"pass_config": {"fuse_norm_quant": False}},
            "additional_config": {"ascend_compilation_config": {"fuse_norm_quant": False}},
        }
        for stage in range(3)
    }
    if stage2_max_num_seqs is not None:
        stage_overrides["2"]["max_num_seqs"] = stage2_max_num_seqs
    if stage0_max_model_len is not None:
        stage_overrides["0"]["max_model_len"] = stage0_max_model_len
    if stage0_enforce_eager:
        stage_overrides["0"]["enforce_eager"] = True
    if stage0_generation_config_vllm:
        stage_overrides["0"]["generation_config"] = "vllm"
    if stage0_async_scheduling is not None:
        stage_overrides["0"]["async_scheduling"] = stage0_async_scheduling
    if stage0_gpu_memory_utilization is not None:
        stage_overrides["0"]["gpu_memory_utilization"] = stage0_gpu_memory_utilization
    if stage1_gpu_memory_utilization is not None:
        stage_overrides["1"]["gpu_memory_utilization"] = stage1_gpu_memory_utilization
    if stage2_gpu_memory_utilization is not None:
        stage_overrides["2"]["gpu_memory_utilization"] = stage2_gpu_memory_utilization
    if stage2_max_npu_graphs is not None:
        stage_overrides["2"]["additional_config"]["code2wav_max_npu_graphs"] = stage2_max_npu_graphs
    if stage2_graph_batch_buckets is not None:
        stage_overrides["2"]["additional_config"]["cfm_graph_batch_buckets"] = list(stage2_graph_batch_buckets)
    return stage_overrides


def start_server(
    port: int,
    log_path: Path,
    deploy_config: Path = DEPLOY,
    stage2_max_num_seqs: int | None = None,
    stage0_max_model_len: int | None = None,
    stage0_enforce_eager: bool = False,
    stage0_generation_config_vllm: bool = False,
    stage0_async_scheduling: bool | None = None,
    stage0_gpu_memory_utilization: float | None = None,
    stage1_gpu_memory_utilization: float | None = None,
    stage2_gpu_memory_utilization: float | None = None,
    stage2_max_npu_graphs: int | None = None,
    stage2_graph_batch_buckets: list[int] | None = None,
    baseline_root: Path = BASELINE,
) -> tuple[subprocess.Popen, float]:
    env = dict(os.environ)
    # Every vLLM-Omni stage uses multiprocessing spawn.  Prepending this
    # baseline-only sitecustomize directory installs the CANN compatibility
    # fallback in the API process and all spawned stage processes.
    source_paths = [str(BASELINE_SITE), str(baseline_root), "/vllm-workspace/vllm"]
    env["PYTHONPATH"] = os.pathsep.join(source_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["MINICPMO_VLLM_ASCEND_RMSNORM_FALLBACK"] = "1"
    # PyTorch's device-backend entry-point auto-loader can recursively import
    # torch_npu while torch itself is still registering the ``triton``
    # namespace.  The Ascend stack imports torch_npu explicitly, so disabling
    # the redundant auto-load path prevents a non-deterministic double
    # TORCH_LIBRARY registration without changing the selected backend.
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    # The vLLM cache key does not include this compatibility method body.  A
    # graph produced before the shim would otherwise keep calling the missing
    # private operator even though every spawned process installed the patch.
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    stage_overrides = make_stage_overrides(
        stage2_max_num_seqs=stage2_max_num_seqs,
        stage0_max_model_len=stage0_max_model_len,
        stage0_enforce_eager=stage0_enforce_eager,
        stage0_generation_config_vllm=stage0_generation_config_vllm,
        stage0_async_scheduling=stage0_async_scheduling,
        stage0_gpu_memory_utilization=stage0_gpu_memory_utilization,
        stage1_gpu_memory_utilization=stage1_gpu_memory_utilization,
        stage2_gpu_memory_utilization=stage2_gpu_memory_utilization,
        stage2_max_npu_graphs=stage2_max_npu_graphs,
        stage2_graph_batch_buckets=stage2_graph_batch_buckets,
    )
    chat_template = baseline_root / CHAT_TEMPLATE_RELATIVE
    if not chat_template.is_file():
        raise FileNotFoundError(
            "frozen MiniCPM-o 4.5 native chat template is missing: "
            f"{chat_template}"
        )
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("launch_vllm_baseline.py")),
        "serve",
        str(MODEL),
        "--omni",
        "--served-model-name",
        "openbmb/MiniCPM-o-4_5",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--deploy-config",
        str(deploy_config),
        "--trust-remote-code",
        # This is part of the model's documented serving contract.  The
        # native template matches HF ``chat(omni_mode=True)`` exactly; even an
        # injected newline changes the Thinker hidden states handed to Talker
        # and can prevent codec EOS from being sampled.
        "--chat-template",
        str(chat_template),
        "--chat-template-content-format",
        "openai",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0,"audio":0}',
        "--stage-overrides",
        json.dumps(stage_overrides, separators=(",", ":")),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("wb")
    started = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=baseline_root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = started + 1800
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"baseline exited with {proc.returncode}; see {log_path}")
        if healthy(f"http://127.0.0.1:{port}/health"):
            return proc, time.monotonic() - started
        time.sleep(1)
    raise TimeoutError(f"baseline was not healthy after 1800 seconds; see {log_path}")


def request(index: int, max_text_tokens: int, max_audio_tokens: int) -> CanonicalRequest:
    row = SeedTTSRow(
        row_id=f"dev-{index:04d}",
        ref_text="This is a reference voice.",
        ref_wav_path=str(REF_AUDIO),
        target_text="The dedicated engine is measuring streaming speech latency.",
    )
    return CanonicalRequest(
        row=row,
        seed=20261022 + index,
        thinker=ThinkerSampling(max_tokens=max_text_tokens),
        talker=TalkerSampling(min_new_tokens=50, max_new_tokens=max_audio_tokens),
    )


async def measure(
    port: int,
    concurrency: int,
    requests: list[CanonicalRequest],
    timeout: float,
    *,
    control_sampling: bool = True,
    include_traces: bool = False,
) -> dict:
    adapter = VllmOmniAdapter(control_sampling=control_sampling)
    payloads = [adapter.build_payload(req) for req in requests]
    traces = [None] * len(requests)
    next_index = 0
    lock = asyncio.Lock()

    async def worker(session) -> None:
        nonlocal next_index
        while True:
            async with lock:
                if next_index >= len(requests):
                    return
                index = next_index
                next_index += 1
            traces[index] = await send_request(
                session,
                f"http://127.0.0.1:{port}",
                adapter,
                requests[index],
                timeout_s=timeout,
                payload=payloads[index],
            )

    async with make_session(connections=max(64, 4 * concurrency)) as session:
        await asyncio.gather(*(worker(session) for _ in range(min(concurrency, len(requests)))))
    traces = [trace for trace in traces if trace is not None]
    metrics = [request_metrics(trace) for trace in traces]
    wall_start = min(trace.t_dispatch for trace in traces)
    wall_end = max(trace.t_complete or trace.t_dispatch for trace in traces)
    audio_s = sum(metric.duration_s for metric in metrics)
    ttfa = sorted(metric.ttfa_s for metric in metrics if metric.ttfa_s is not None)
    e2e = sorted(metric.e2e_s for metric in metrics if metric.e2e_s is not None)
    result = {
        "concurrency": concurrency,
        "n_ok": sum(metric.ok for metric in metrics),
        "n_failed": sum(not metric.ok for metric in metrics),
        "wall_s": wall_end - wall_start,
        "audio_s": audio_s,
        "throughput_audio_s_per_s": audio_s / (wall_end - wall_start),
        "ttfa_s": ttfa,
        "ttfa_p50_s": statistics.median(ttfa) if ttfa else None,
        "e2e_s": e2e,
        "e2e_p50_s": statistics.median(e2e) if e2e else None,
        "rtf": [metric.rtf for metric in metrics],
        "failures": [asdict(metric) for metric in metrics if not metric.ok],
        "request_ids": [req.request_id for req in requests],
        "request_ids_sha256": hashlib.sha256("\n".join(req.request_id for req in requests).encode()).hexdigest(),
    }
    if include_traces:
        result["requests"] = [req.describe() for req in requests]
        result["traces"] = [trace.to_json() for trace in traces]
    return result


def write_result(path: Path | None, result: dict) -> None:
    """Persist a completed wave before server shutdown can delay stdout."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=120)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--max-text-tokens", type=int, default=16)
    parser.add_argument("--max-audio-tokens", type=int, default=128)
    parser.add_argument("--warmup-audio-tokens", type=int, default=64)
    parser.add_argument(
        "--warmup-timeout",
        type=float,
        help=(
            "Per-request timeout for the unmeasured warmup wave. Defaults to "
            "--timeout; set this higher when baseline graph capture is slow."
        ),
    )
    parser.add_argument("--timeout", type=float, default=FROZEN_REQUEST_TIMEOUT_S)
    parser.add_argument(
        "--warmup-drain-timeout",
        type=float,
        default=900,
        help="Maximum extra seconds to wait for client-timed-out warmups to leave every Omni stage.",
    )
    parser.add_argument("--server-log", type=Path, default=Path("/tmp/minicpmo-vllm-baseline.log"))
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Write the completed measurement atomically before stopping the server.",
    )
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="Include request text, usage, audio contributions, and event timing diagnostics in the result JSON.",
    )
    parser.add_argument("--baseline-root", type=Path, default=BASELINE)
    parser.add_argument("--deploy-config", type=Path, default=DEPLOY)
    parser.add_argument("--stage2-max-num-seqs", type=int)
    parser.add_argument("--stage0-max-model-len", type=int)
    parser.add_argument(
        "--stage0-enforce-eager",
        action="store_true",
        help="Diagnostic compatibility mode: disable compilation and graph replay for the Thinker stage.",
    )
    parser.add_argument(
        "--stage0-generation-config-vllm",
        action="store_true",
        help="Ignore the checkpoint generation_config.json for the Thinker stage.",
    )
    parser.add_argument(
        "--stage0-sync-scheduling",
        action="store_true",
        help="Diagnostic only: use OmniARScheduler instead of OmniARAsyncScheduler for Stage 0.",
    )
    parser.add_argument("--stage0-gpu-memory-utilization", type=float)
    parser.add_argument("--stage1-gpu-memory-utilization", type=float)
    parser.add_argument("--stage2-gpu-memory-utilization", type=float)
    parser.add_argument(
        "--stage2-max-npu-graphs",
        type=int,
        help="Cap the vLLM-Omni Code2Wav CFM NPUGraph cache; unseen shapes fall back to eager execution.",
    )
    parser.add_argument(
        "--stage2-graph-batch-buckets",
        type=int,
        nargs="+",
        help="Round live Code2Wav batches up to these capture-key buckets.",
    )
    parser.add_argument(
        "--cfm-graph-bucket-frames",
        type=int,
        help="Round CFM mel widths to this frame multiple through a recorded deploy overlay.",
    )
    parser.add_argument(
        "--allow-warmup-failures",
        action="store_true",
        help=(
            "Record warmup failures and continue into the measured probe. "
            "The frozen harness treats warmup as unmeasured and likewise "
            "continues; measured requests must still all pass."
        ),
    )
    parser.add_argument("--seed-tts-root", type=Path)
    parser.add_argument("--id-set", choices=("tuning", "eval"), default="tuning")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument(
        "--native-stage-sampling",
        action="store_true",
        help=(
            "Diagnostic only: omit sampling_params_list and let vLLM-Omni use "
            "the frozen deploy defaults for non-comprehension stages."
        ),
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.warmup_timeout is not None and args.warmup_timeout <= 0:
        parser.error("--warmup-timeout must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.warmup_drain_timeout <= 0:
        parser.error("--warmup-drain-timeout must be positive")
    if args.stage2_max_npu_graphs is not None and args.stage2_max_npu_graphs <= 0:
        parser.error("--stage2-max-npu-graphs must be positive")
    if args.stage0_max_model_len is not None and args.stage0_max_model_len <= 0:
        parser.error("--stage0-max-model-len must be positive")
    if args.stage2_graph_batch_buckets is not None:
        if any(bucket <= 0 for bucket in args.stage2_graph_batch_buckets):
            parser.error("--stage2-graph-batch-buckets values must be positive")
        if args.stage2_graph_batch_buckets != sorted(set(args.stage2_graph_batch_buckets)):
            parser.error("--stage2-graph-batch-buckets values must be sorted and unique")
    if args.cfm_graph_bucket_frames is not None and args.cfm_graph_bucket_frames <= 1:
        parser.error("--cfm-graph-bucket-frames must be greater than 1")
    for stage, utilization in (
        (0, args.stage0_gpu_memory_utilization),
        (1, args.stage1_gpu_memory_utilization),
        (2, args.stage2_gpu_memory_utilization),
    ):
        if utilization is not None and not (0.0 < utilization <= 1.0):
            parser.error(f"--stage{stage}-gpu-memory-utilization must be in (0, 1]")

    dataset = None
    if args.seed_tts_root is not None:
        rows = load_seed_tts_rows(args.seed_tts_root)
        ids = make_id_sets(rows)
        measured_ids = ids.measured(args.id_set, args.concurrency)
        if args.limit is not None:
            measured_ids = measured_ids[: args.limit]
        warmup_requests = canonical_requests(rows, ids.warmup(args.concurrency), seed_index=args.seed_index)
        measured_requests = canonical_requests(rows, measured_ids, seed_index=args.seed_index)
        dataset = {
            "repo": "zhaochenyang20/seed-tts-eval",
            "revision": "8f5e1aa2a35d42f42e940074c1983358b9491f89",
            "root": str(args.seed_tts_root),
            "id_set": args.id_set,
            "seed_index": args.seed_index,
        }
    else:
        warmup_requests = [
            request(10000 + i, args.max_text_tokens, args.warmup_audio_tokens) for i in range(args.concurrency)
        ]
        measured_requests = [request(i, args.max_text_tokens, args.max_audio_tokens) for i in range(args.concurrency)]

    proc = None
    try:
        effective_deploy_config = materialize_deploy_overlay(
            args.deploy_config,
            args.server_log.with_suffix(".deploy.json"),
            args.cfm_graph_bucket_frames,
        )
        proc, startup_s = start_server(
            port=args.port,
            log_path=args.server_log,
            deploy_config=effective_deploy_config,
            stage2_max_num_seqs=args.stage2_max_num_seqs,
            stage0_max_model_len=args.stage0_max_model_len,
            stage0_enforce_eager=args.stage0_enforce_eager,
            stage0_generation_config_vllm=args.stage0_generation_config_vllm,
            stage0_async_scheduling=False if args.stage0_sync_scheduling else None,
            stage0_gpu_memory_utilization=args.stage0_gpu_memory_utilization,
            stage1_gpu_memory_utilization=args.stage1_gpu_memory_utilization,
            stage2_gpu_memory_utilization=args.stage2_gpu_memory_utilization,
            stage2_max_npu_graphs=args.stage2_max_npu_graphs,
            stage2_graph_batch_buckets=args.stage2_graph_batch_buckets,
            baseline_root=args.baseline_root,
        )
        timing_count_before_warmup = omni_timing_count(args.server_log)
        warmup_timeout = resolve_warmup_timeout(args.warmup_timeout, args.timeout)
        warmup = asyncio.run(
            measure(
                args.port,
                args.concurrency,
                warmup_requests,
                warmup_timeout,
                control_sampling=not args.native_stage_sampling,
                include_traces=args.include_traces,
            )
        )
        if warmup["n_failed"] and not args.allow_warmup_failures:
            raise RuntimeError(f"warmup failed: {warmup['failures']}")
        warmup_drain = wait_for_omni_timings(
            args.server_log,
            timing_count_before_warmup + len(warmup_requests),
            args.warmup_drain_timeout,
        )
        result = asyncio.run(
            measure(
                args.port,
                args.concurrency,
                measured_requests,
                args.timeout,
                control_sampling=not args.native_stage_sampling,
                include_traces=args.include_traces,
            )
        )
        result.update(
            {
                "engine": "vllm-omni",
                "model": str(MODEL),
                "deploy": str(effective_deploy_config),
                "base_deploy": str(args.deploy_config),
                "baseline_root": str(args.baseline_root),
                "chat_template": str(args.baseline_root / CHAT_TEMPLATE_RELATIVE),
                "chat_template_sha256": hashlib.sha256(
                    (args.baseline_root / CHAT_TEMPLATE_RELATIVE).read_bytes()
                ).hexdigest(),
                "startup_s": startup_s,
                "warmup_wall_s": warmup["wall_s"],
                "warmup_timeout_s": warmup_timeout,
                "request_timeout_s": args.timeout,
                "warmup_n_ok": warmup["n_ok"],
                "warmup_n_failed": warmup["n_failed"],
                "warmup_drain_s": warmup_drain["elapsed_s"],
                "warmup_drain_expected_timings": warmup_drain["expected_timings"],
                "warmup_drain_observed_timings": warmup_drain["observed_timings"],
                "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "control_sampling": not args.native_stage_sampling,
                "stage2_max_num_seqs": args.stage2_max_num_seqs,
                "stage0_max_model_len": args.stage0_max_model_len,
                "stage0_enforce_eager": args.stage0_enforce_eager,
                "stage0_generation_config_vllm": args.stage0_generation_config_vllm,
                "stage0_async_scheduling": False if args.stage0_sync_scheduling else None,
                "stage0_gpu_memory_utilization": args.stage0_gpu_memory_utilization,
                "stage1_gpu_memory_utilization": args.stage1_gpu_memory_utilization,
                "stage2_gpu_memory_utilization": args.stage2_gpu_memory_utilization,
                "stage2_max_npu_graphs": args.stage2_max_npu_graphs,
                "stage2_graph_batch_buckets": args.stage2_graph_batch_buckets,
                "cfm_graph_bucket_frames": args.cfm_graph_bucket_frames,
                "dataset": dataset,
                "warmup_request_ids_sha256": warmup["request_ids_sha256"],
            }
        )
        if args.include_traces:
            result["warmup_requests"] = warmup.get("requests", [])
            result["warmup_traces"] = warmup.get("traces", [])
        write_result(args.result_json, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if result["n_failed"]:
            raise SystemExit(1)
    finally:
        if proc is not None:
            stop(proc)


if __name__ == "__main__":
    main()
