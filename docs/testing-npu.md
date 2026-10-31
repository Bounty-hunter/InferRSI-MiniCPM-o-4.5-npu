# Ascend NPU test status

The release snapshot was bytecode-compiled on the packaging host. The host did not have `pytest`, `torch`, CANN or the
Seed-TTS dataset, so NPU integration and speech tests must run inside the documented Linux/Ascend environment.

The latest native Ascend r295 comparison is the relevant end-to-end evidence: two fresh services, identical C1→C4→C8→C16→C32
order, two native warmups per tier, 992 requests per service, 1,984 total, and zero request errors. The candidate and
reference source/input identities and runtime import audit were verified. The final frozen speech gate and one-sided
95% LCB were not completed for this snapshot.
