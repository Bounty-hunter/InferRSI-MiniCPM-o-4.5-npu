#!/usr/bin/env python3
"""Probe the Thinker through the full vLLM-Omni API with text-only output.

This keeps Omni's request construction and scheduler in the path while skipping
the Talker and Code2Wav stages.  It distinguishes a stage-0 Omni regression from
an audio-stage handoff or response-aggregation regression.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

from tools.probe_vllm_baseline import DEPLOY, start_server, stop
from tools.probe_vllm_stage0 import SYSTEM_PROMPT


def build_payload() -> dict:
    return {
        "model": "openbmb/MiniCPM-o-4_5",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": "This generation saw strong console wars."}],
            },
        ],
        "modalities": ["text"],
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False, "use_tts_template": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18187)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--deploy-config", type=Path, default=DEPLOY)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--stage0-enforce-eager", action="store_true")
    parser.add_argument(
        "--stage0-sync-scheduling",
        action="store_true",
        help="Diagnostic only: use the synchronous Omni Stage 0 scheduler.",
    )
    args = parser.parse_args()

    proc = None
    started = time.monotonic()
    try:
        proc, startup_s = start_server(
            port=args.port,
            log_path=args.server_log,
            deploy_config=args.deploy_config,
            baseline_root=args.baseline_root,
            stage0_enforce_eager=args.stage0_enforce_eager,
            stage0_generation_config_vllm=True,
            stage0_async_scheduling=False if args.stage0_sync_scheduling else None,
        )
        payload = build_payload()
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
            "deploy": str(args.deploy_config),
            "elapsed_s": completed - dispatched,
            "engine": "vllm-omni-text-only",
            "http_status": status,
            "payload": payload,
            "response": body,
            "stage0_enforce_eager": args.stage0_enforce_eager,
            "stage0_generation_config_vllm": True,
            "stage0_async_scheduling": False if args.stage0_sync_scheduling else None,
            "startup_s": startup_s,
            "total_wall_s": completed - started,
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        }
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        if proc is not None:
            stop(proc)


if __name__ == "__main__":
    main()
