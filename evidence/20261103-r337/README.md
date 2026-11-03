# Ascend r337 source and measurement evidence

The candidate runtime is published as an opt-in experimental configuration. `source-manifest.json` binds the exact
measured engine hash, configuration, preserved public-server integration difference and control309 reconstruction.

## Measurements and figures

The `campaign` directory retains the C1 supplement, C2/C4/C8 confirmations, both raw candidate measurements,
one aggregated candidate curve, the pinned Omni reference, nine PNG/SVG/PDF exports, and the parent patches
for all 40 algorithm attempts r312–r351. It also includes scheduler receipts and complete saved logs for the
seven rejected/failed algorithm jobs. Historical `finalization.json` fields describe the measurement-time state.

The figure convention takes the arithmetic mean of both candidate run-level values. P99 figures likewise average
the two run-level P99 values. Request counts are32/32/64/128 per source run, with2warmups per tier. C1 was collected
later with unchanged source/configuration. Zero request errors applies to the successful comparison runs.

## Reproduction

From the repository root, run `python tools/plot_r337_performance.py` and `python tools/verify_r337_evidence.py`.
The former needs matplotlib and a Chinese font. The latter uses the standard library and verifies source hashes,
workload commands, all original comparison points, aggregation, 40-round receipts/patches and local document links.

For an exact measured-tree reconstruction in a separate checkout, restore `engine/server.py` from
`source/engine-server.py.txt`. `source-manifest.json` lists the additional file overrides for control309. The r305
source manifest records its own archived files and continues to support its historical verifier.

## Quality and scope

Omni uses Flow FP32/10 steps, the engine FP16/6 steps and counter RNG. The paired current speech gate is NOT_RUN.
NPU scorer preparation is recorded separately from completed performance evidence. The frozen statistical/speech
acceptance plan retains its thresholds. Large waveforms, model weights and raw device traces remain in the audit store.

- [Full Chinese account](../../docs/rsi-r337-20261103.zh-CN.md)
- [All40rounds](campaign/持续优化实验记录.md)
- [Forward/reverse controls and full mean/P99](campaign/正反序确认与Omni对照.md)
- [CPU and language analysis](campaign/CPU瓶颈与语言选择.md)
- [r307–r309 follow-up](followup/六层profiling继续优化-r307-r309.md)
- [Source manifest](source-manifest.json)
