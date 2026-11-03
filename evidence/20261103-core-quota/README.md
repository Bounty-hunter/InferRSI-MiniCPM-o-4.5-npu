# Fixed-work core-quota study, 2026-11-03

[中文报告](../../docs/core-quota-causality.zh-CN.md)

This bundle contains fixed-input component comparisons and a quota × serial/multistream 2×2 replay.
The submitting host thread is single-threaded. A uses 24 Cube/48 Vector and B uses 12 Cube/24 Vector.
Three fresh processes per role run in A1/B1/B2/A2/A3/B3 order. The diagnostic profile pair is separate.

- `summary.json` aggregates the three process medians for each role.
- `trials/` retains individual samples, input hashes, output comparisons, resource observations and exits.
- `capture/` identifies the captured inputs and diagnostic request set.
- `fixed-work-analysis.json` retains the three paired comparisons.
- `profile-kernel-analysis.json` records matched shape groups and remote CSV paths/hashes.
- `profiles/` contains compact profile-run metadata and exits.
- `source-manifest.json` reconstructs the measured r320 runtime through explicit source overrides.
- `scripts/` preserves the executed component harness and analysis scripts.
- `bundle-sha256.json` binds every other file in this directory.

The large `.pt` inputs and raw profiles remain on the NPU host at the recorded paths. The local verifier
checks compact evidence consistency and file hashes. Its checks of remote large files are limited to their
recorded identities. Profile aggregates are diagnostic measurements with profiler overhead.

Cross-quota outputs differ reproducibly. Speech-quality acceptance remains `NOT_RUN`.
These fixed-work timings measure component/task-package completion. Native-service performance remains
documented separately in the existing benchmark reports.

```sh
python tools/verify_core_quota_study.py
```
