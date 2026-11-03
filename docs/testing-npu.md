# Ascend NPU test status

## r337 publication checks — November 3, 2026

The current publication passes 26 focused CPU tests covering prompt verification, NPU hot-path contracts, and host-length provenance/fallback cases. The artifact verifier reconstructs measured r337 and control309, verifies all 40 round receipts/patches, and checks the four-tier aggregation and local document links. The historical r305 verifier also passes after reconstruction from archived runtime files.

The published runtime includes the existing optional CPU-affinity server API. This integrated tree has source binding to the measured NPU snapshot. Full current speech/statistical gates remain NOT_RUN.

[Current checks](../evidence/20261103-r337/publication-checks.json) · [Current evidence](../evidence/20261103-r337/README.md)

## r305 publication checks — November 2, 2026

The [experimental r305 update](rsi-profiling-20261102.zh-CN.md) includes 35 passing focused CPU tests for draft
acceptance/rejection, graph-gate coordination, the retained CPU-affinity API, server entry and stream lifecycle.
Five NPU hotpath AST checks and compileall pass. The evidence verifier checks four direct Omni cells, three
same-config incremental cells, ten rounds, commands, source/config hashes, receipts and local documentation links.

A broader local run also encountered seven environment failures: six require the absent Seed-TTS snapshot and
one requires local torch. Those seven were excluded from the focused rerun, not converted into passes.
No new NPU benchmark was run for the integrated publication tree; measured-source reconstruction is explicitly bound
in the [source manifest](../evidence/20261102-r305/source-manifest.json).
See [publication-checks.json](../evidence/20261102-r305/publication-checks.json). These checks do not pass the final
frozen statistical or speech-quality gates.

## Earlier snapshot

The earlier release snapshot was bytecode-compiled on its packaging host. That host did not have `pytest`, `torch`,
CANN or Seed-TTS. NPU integration and speech tests require the documented Linux/Ascend environment.

The earlier native Ascend r295 comparison used two fresh services, identical C1→C4→C8→C16→C32
order, two native warmups per tier, 992 requests per service, 1,984 total, and zero request errors. The candidate and
reference source/input identities and runtime import audit were verified. The final frozen speech gate and one-sided
95% LCB were not completed for this snapshot.
