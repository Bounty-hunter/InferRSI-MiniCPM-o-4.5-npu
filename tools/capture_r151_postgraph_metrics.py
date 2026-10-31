#!/usr/bin/env python3
"""Continuously persist successful engine /metrics snapshots as JSONL."""

from __future__ import annotations

import argparse
import json
import signal
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metrics_url = args.url.rstrip("/") + "/metrics"
    with args.out.open("a", encoding="utf-8", buffering=1) as sink:
        while running:
            started = time.monotonic()
            try:
                with urllib.request.urlopen(metrics_url, timeout=min(1.0, args.interval)) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and isinstance(payload, dict):
                    sink.write(
                        json.dumps(
                            {
                                "captured_at_unix": time.time(),
                                "metrics": payload,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            except (OSError, ValueError, urllib.error.URLError):
                pass
            remaining = args.interval - (time.monotonic() - started)
            if running and remaining > 0:
                time.sleep(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
