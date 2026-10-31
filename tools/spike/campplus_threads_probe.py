"""Measure CAMPPlus CPU thread scaling and numerical drift on one reference.

This probe intentionally avoids loading any accelerator model.  It feeds the
same normalized Kaldi fbank to independent ONNX Runtime sessions so timing and
output differences only reflect the session configuration under test.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime
import torch
import torchaudio.compliance.kaldi as kaldi


def session(model: Path, threads: int):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return onnxruntime.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/dataset/models/MiniCPM-o-4_5"))
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    wav, _ = librosa.load(args.model_dir / "assets/HT_ref_audio.wav", sr=16000, mono=True)
    audio = torch.from_numpy(wav)
    feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    feat -= feat.mean(dim=0, keepdim=True)
    feed = feat.unsqueeze(0).numpy()
    model = args.model_dir / "assets/token2wav/campplus.onnx"

    outputs = {}
    rows = []
    for threads in args.threads:
        ort = session(model, threads)
        name = ort.get_inputs()[0].name
        for _ in range(args.warmup):
            ort.run(None, {name: feed})
        times = []
        output = None
        for _ in range(args.repeats):
            started = time.perf_counter()
            output = ort.run(None, {name: feed})[0]
            times.append(time.perf_counter() - started)
        outputs[threads] = output
        rows.append({
            "threads": threads,
            "median_s": statistics.median(times),
            "min_s": min(times),
            "max_s": max(times),
            "all_s": times,
        })

    reference = outputs[args.threads[0]].astype(np.float64)
    denom = np.linalg.norm(reference)
    for row in rows:
        value = outputs[row["threads"]].astype(np.float64)
        row["relative_l2_vs_first"] = float(np.linalg.norm(value - reference) / denom)
        row["max_abs_vs_first"] = float(np.max(np.abs(value - reference)))
    print(json.dumps({"shape": list(reference.shape), "results": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
