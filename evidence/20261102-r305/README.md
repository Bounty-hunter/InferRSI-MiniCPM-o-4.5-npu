# Measured Ascend r305 evidence

Native benchmark outputs and workload commands are included for C1/C2/C4/C8, along with r292 controls,
r305 screening points, ten-round summaries, per-round source patches, receipts and compact profiling analyses.
No audio, model weights or large device trace is published. Full logs/WAVs/raw trace remain in the original audit store.

`source-manifest.json` binds the measured runtime. The published server preserves its existing optional CPU-affinity
feature; its other engine files match the measured snapshot. For an exact measured-tree reconstruction, use an isolated
checkout and replace `engine/server.py` with `source/engine-server.py.txt`. The two runtime files superseded by r337 are archived under `source/engine/runtime/` and listed in `archived_publication_files` in the manifest. Restore those files as well to reconstruct r305. The experimental JSON is byte-identical to
the measured configuration. Do not describe the integrated publication tree as a new timed NPU run.

All numbers are exploratory points. The original frozen validator plan is unchanged; statistical and final speech
acceptance remain incomplete. The failed r306 API attempt and repaired r306b results are retained separately.
