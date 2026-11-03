# Prompt-lookup removal

On November 3, 2026, the user requested removal of the prompt-lookup optimization.
The serving runtime now uses ordinary original-model Thinker decoding. The draft/verification module,
its scheduling branch and counters have been removed. Both r305/r337 named configurations have had
the lookup setting removed. A legacy constructor value of zero is accepted. Nonzero values fail before
device initialization so an old configuration receives an explicit migration error.

The current runtime has no new NPU benchmark or speech-quality acceptance result.
Published r305/r337 tables and evolution curves remain historical measurements of their recorded source
and configuration, including lookup. Their values cannot be assigned to the modified runtime.

`source-map.json` binds the exact pre-removal public backend, lookup module and two configurations to
archived bytes. Historical artifact verifiers reconstruct those snapshots along with their existing
measured-server/control overrides. Original frozen results, manifests and evidence hashes are preserved.

To reconstruct a historical measurement, use its existing source manifest and overrides, additionally
restoring the files listed in this source map to their recorded original paths. The r305 manifest's own
backend override takes precedence over the public backend archived here. Historical configurations
with lookup enabled require the corresponding historical runtime.

```sh
python -m unittest discover -s tests/engine_api -p test_lookup_removed.py -v
python tools/verify_r337_evidence.py
python tools/verify_r305_evidence.py
python tools/verify_core_quota_study.py
```
