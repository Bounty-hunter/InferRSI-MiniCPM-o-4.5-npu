#!/usr/bin/env python3
"""Isolate MiniCPM-o Thinker correctness from the Omni pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


MODEL = Path("/dataset/models/MiniCPM-o-4_5")
CHAT_TEMPLATE_RELATIVE = Path("vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja")
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech.\n"
    "For this request you act as a text-to-speech engine with zero-shot voice cloning: "
    "the API provides reference audio and its transcript (ref_audio, ref_text) and task_type Base. "
    "The user message is the exact text you must speak. "
    "Synthesize natural speech in the same language as that user text, "
    "matching the timbre, prosody, and speaking style of the reference audio while reading the new content clearly."
)


def healthy(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def stop(proc: subprocess.Popen[bytes]) -> None:
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
    parser.add_argument("--port", type=int, default=18186)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument(
        "--async-scheduling",
        action="store_true",
        help="Diagnostic parity with the full Omni stage-0 scheduler mode.",
    )
    args = parser.parse_args()

    chat_template = args.baseline_root / CHAT_TEMPLATE_RELATIVE
    env = dict(os.environ)
    source_paths = [
        str(Path(__file__).with_name("baseline_site")),
        str(args.baseline_root),
        "/vllm-workspace/vllm",
    ]
    env["PYTHONPATH"] = os.pathsep.join(source_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["MINICPMO_VLLM_ASCEND_RMSNORM_FALLBACK"] = "1"
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["VLLM_DISABLE_COMPILE_CACHE"] = "1"

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("launch_vllm_baseline.py")),
        "serve",
        str(MODEL),
        "--served-model-name",
        "openbmb/MiniCPM-o-4_5",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--chat-template",
        str(chat_template),
        "--chat-template-content-format",
        "openai",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0,"audio":0}',
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.32",
        "--enforce-eager",
        "--generation-config",
        "vllm",
    ]
    if args.async_scheduling:
        cmd.append("--async-scheduling")
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    log = args.server_log.open("wb")
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=args.baseline_root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = started + 1800
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Stage 0 server exited with {proc.returncode}; see {args.server_log}")
            if healthy(f"http://127.0.0.1:{args.port}/health"):
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"Stage 0 server was not healthy after 1800 seconds; see {args.server_log}")

        payload = {
            "model": "openbmb/MiniCPM-o-4_5",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "This generation saw strong console wars."}],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 16,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False, "use_tts_template": True},
        }
        request = Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        dispatched = time.monotonic()
        with urlopen(request, timeout=300) as response:
            raw = response.read()
            status = response.status
        completed = time.monotonic()
        body = json.loads(raw)
        result = {
            "baseline_root": str(args.baseline_root),
            "async_scheduling": args.async_scheduling,
            "elapsed_s": completed - dispatched,
            "engine": "vllm-stage0-only",
            "http_status": status,
            "payload": payload,
            "response": body,
            "startup_s": dispatched - started,
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        }
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        stop(proc)


if __name__ == "__main__":
    main()
