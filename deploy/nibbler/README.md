# Nibbler branch deployment

Local branch: `nibbler/prefill-and-conversation-cache`, based on upstream main
`99f3dbd0b21d1401b3769e0c0d963913607f380b`. No GitHub fork/push required.

This hardware-specific config retains the installed IQ3_S model, sharp tokenizer,
262144 context, INT8 KV with 32768 resident cells, MTP and GPU vision. It enables:

- Native fused prompt experts: `STRATA_PF_FUSED=1`.
- Single-GPU fused tail balancing: `STRATA_PREFILL_BALANCE_TAIL=1`.
- A 4096 MiB host conversation cache, at most four parked entries, and a 4096 MiB
  physical RAM floor. The budget is not four guaranteed full-context slots.

The branch fixes speculative output-cap/EOS overcommit in both serving and CLI.
Verification windows are bounded by remaining output/context, and only inputs
producing the actually emitted prefix enter persistent state. Unit tests live in
`src/spec/output_limits_test.cpp`.

Tail balancing is opt-in and keeps the existing stream/fused threshold. A 9171-token
run uses 8147 + 1024 rather than 8192 + 979, avoiding the MMQ tail fallback without
adding chunks or exceeding the borrowed arena. PLE read-ahead uses the same schedule;
logical positions, progress, checkpoints and draft callbacks use actual chunk lengths.
Layer splits and peer-device execution retain their previous schedule. Unit tests
live in `src/prefill/chunk_schedule_test.cpp`. As with other chunk/kernel changes,
bitwise identity with the old floating-point path is not promised.

Per-run prefill profiling now prints deltas for host staging and PLE, instead of
mixing cumulative counters with local CUDA-event/wall timings.

## Build and launch

Use nibbler's existing CUDA 13.3/GCC 15 Release build configuration. Enable the
standalone tests with `-DSTRATA_BUILD_CONVERSATION_TESTS=ON`. Build target `strata`;
install a copy as `engine/strata-nibbler`, not over the original `engine/strata`.
Set executable permission on `deploy/nibbler/start-strata`.

llama-swap still invokes `/opt/native-inference/bin/start-strata-sharp-medium`.
That script delegates to this branch's `deploy/nibbler/start-strata PORT` after
validation. Original production JSON remains unchanged. Requests retain the same
model name and aliases. The branch engine log is `/data/llm/Strata/strata-nibbler.log`.

Do not run upstream `update.sh`/setup on this deployment without reviewing its
checkout/build/config rewriting behavior. Fetch upstream and merge/rebase this branch
intentionally; the external launcher expects this branch's deployment files.

## Rollback

Pre-change binary, config and launcher were saved in
`/data/llm/Strata-tests/implementation-20261003/`:
`strata.before`, `production-config.before.json`, `launcher.before`.

1. Wait for idle or arrange a maintenance window; unload only Strata:
   `curl -X POST http://127.0.0.1:8088/api/models/unload/qwen3.8-flash-next-iq3_s`.
2. Restore `launcher.before` to `/opt/native-inference/bin/start-strata-sharp-medium`
   and make it executable. The original `engine/strata` and production JSON have
   not been replaced; those are what the old launcher uses.
3. Reload with a request through llama-swap, and verify `/running` and backend health.
   Switch the checkout to the original main commit while unloaded if desired.

Validation evidence and limitations are recorded in `validation.md` when deployment
is completed. Performance/quality probes are local smoke gates, not proof that every
model workload has unchanged quality. No true parallel serving or shared KV pool is
introduced by this branch.
