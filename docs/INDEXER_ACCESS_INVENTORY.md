# QSA indexer access inventory for incremental storage

Status: source audit for task `str-a2y.1`; no runtime storage change is implemented here.

## CPU arithmetic contract added in I1

`include/strata/core/indexer_extent.hpp` defines the checked CPU-only contract used by later storage work. Its
parameterized geometry is for arithmetic tests and planning only; it does not imply GPU kernel support beyond the
current native kernel geometry (`idx_block = 4`, `idx_dim = 128`).

- **Ceiling rows:** current contiguous allocation capacity is `floor(ceiling_cells / idx_block) + 2`. This is the
  formula in `src/core/layer.cpp`'s `kv_plan` and is intentionally not a `session_bytes` validation.
- **Completed rows:** accepted, committed history owns `floor(cells / idx_block)` completed pooled keys.
- **Valid rows:** non-empty history also has a sequence-local moving spare at `cells / idx_block`, so valid rows are
  `cells > 0 ? floor(cells / idx_block) + 1 : 0`.
- **Score read rows:** reported as a conservative envelope across old and new scorer paths. The old `qsa_index` path
  reads the spare from `pooled[n_bid]`; newer block scorers branch to `dead` for `b == n_bid`, so this is not an exact
  physical read requirement for every optimized scorer.
- **Append required rows:** reserve for the whole append/provisional range before launch. Position zero writes `dead`
  and `pooled[0]`; every block completion `b` writes `pooled[b]` and the next spare `pooled[b + 1]`. A rejected
  verification window still needs the rows it might write before rollback.
- **Snapshot/restore rows:** canonical snapshots store zero rows for an empty sequence, otherwise `tokens / idx_block +
  1` rows so the spare is included. Checkpoint restore writes the saved spare back to `pooled[tokens / idx_block]`.
- **Physical bytes:** bytes are reported as raw bytes per layer, raw bytes for all QSA layers, granularity-rounded bytes
  per layer, and the sum of independently rounded per-layer ranges. The helper does not query device granularity.
  `indexer_byte_extent_for_rows` is a raw slab/accounting utility and deliberately permits row counts beyond the
  logical ceiling; sequence helpers (`history`, `append`, `snapshot`) enforce ceiling bounds.

The helpers validate geometry, negative extents where applicable, integer overflow, and sequence ceiling bounds into
temporary outputs first; a rejected call leaves the caller's output object unchanged.

## Source-grounded access inventory

| Category | Source references | Accesses that incremental storage must cover | Backend uncertainty |
| --- | --- | --- | --- |
| Allocation and current capacity | `src/core/layer.cpp` `kv_plan` (`pooled_rows = max_cells / idx_block + 2`), `qsa_state_bytes`, `qsa_state_init`; `include/strata/core/layer.hpp` `QsaState::idx_pooled_rows` | The legacy contiguous private allocation owns the whole ceiling. Later incremental storage must preserve the same virtual row capacity while separately tracking committed/valid rows. | CUDA path is authoritative. SYCL mirror has the same formula in `sycl/src/core/layer.cpp`; any interface change must be ported or explicitly rejected there. HIP gfx906 compiles the CUDA sources through compat headers. |
| Zero/reset | `src/core/layer.cpp` `qsa_state_zero` zeroes `idx_tail`, `idx_dead`, all `idx_pooled_rows`, and `idx_block_pos`; `sycl/src/core/layer.cpp` mirrors it | Incremental mode may zero only mapped rows, but must clear validity metadata so a shorter new history cannot inherit rows from an older longer one. Legacy disabled mode must keep full zeroing. | CUDA/HIP/SYCL all have reset code. Future incremental mode needs a backend-specific mapped-byte zero primitive or a clear unsupported-backend diagnostic. |
| Single-token append writer | `src/core/layer.cpp` `qsa_layer` builds `QsaIndexerBuffers` and calls `native_qsa_indexer_append` or `indexer_key_append`; `src/kernels/cuda/qsa.cu` `indexer_key_append_kernel`; `src/kernels/cuda/native_qsa_indexer.cu` `append` | Position zero writes the raw tail slot, `dead`, and `pooled[0]`. A non-completing token writes only a tail slot. A completing token writes completed row `b`, spare row `b + 1`, and `idx_block_pos`. Reserve the write high-water before launch. | CUDA source covers both reference and native paths. HIP gfx906 uses the CUDA source. SYCL has `.dp.cpp` equivalents and must not be assumed safe from CUDA-only changes. |
| Batched/chunk prefill writer | `src/prefill/prefill.cpp` around `native_qsa_indexer_append_batch`; fallback `STRATA_INDEXER_PER_TOKEN` calls `native_qsa_indexer_append`; `src/kernels/cuda/native_qsa_indexer.cu` `append_first`, `append_blocks`, `append_tail` | The prompt chunk is appended before scoring. The batch path writes position-zero spare when `p0 == 0`, every completed block whose last cell falls in `[p0, p0 + T)`, only the final spare after the last completed block, and tail slots for the last partial block. Reserve for the entire chunk, not each later query. | Tensor-core and SIMT scoring read the same produced rows. SYCL prefill mirrors the same calls in `sycl/src/prefill/prefill.cpp`; backend parity is unverified here. |
| Verification/speculative writer | `src/core/verify.cpp` around `native_qsa_indexer_append_steps` for batched decode windows and per-slot `native_qsa_indexer_append` fallback; later MTP append paths in the same file | The verification window may complete rows that are later rejected. Capacity must cover the full provisional window, including spare row writes, before any append launch. Rollback can reduce valid history, but it cannot make an earlier out-of-bounds write safe. | Batch-recursive slots use each slot's own `QsaState`; ordinary batched verification can write one shared state for the window. SYCL has mirrored verify code. |
| Scoring readers | `src/core/layer.cpp` `qsa_block_scores` / `qsa_index_step`; `src/core/verify.cpp` `qsa_block_scores` and `qsa_block_topk`; `src/prefill/prefill.cpp` tensor-core `qsa_block_scores_tc` and SIMT fallback; `src/kernels/cuda/qsa_select.cu` `block_scores_multi_kernel` | Scorers run grids sized by ceiling/capacity but guard by per-query `n_bid`. The old `qsa_index` path reads the tail/spare key from `pooled[n_bid]`; the newer block scorers in `qsa_select.cu` branch to `dead` when `b == n_bid`. Incremental storage should use the helper's valid-row read envelope until each enabled scorer/debug path has an exact committed/read contract. | Tensor-core path has an `active` bound in prefill but still needs an audited mapped-row contract. HIP WMMA path is optional and unverified by this task. |
| Snapshot K/V payload copy | `src/core/conversation_snapshot.cpp` `layout` computes `pooled = index && upto > 0 ? upto / idx_block + 1 : 0`; `pools()` includes `st.idx_pooled`; save/restore copy the canonical logical payload | Snapshot bytes must include the spare row but not unused ceiling rows. Restore must validate the whole image before destructive writes and then have capacity for the canonical payload rows. | Snapshot code is CUDA-oriented through `conversation_copy`; SYCL has a separate mirror. This task does not change canonical format. |
| Checkpoint save/restore running state | `src/core/conversation_state.cpp` `conversation_checkpoint_save` copies `idx_tail`, `idx_dead`, `idx_block_pos`; `conversation_checkpoint_restore` copies them back and writes `idx_pooled + ids.size() / idx_block` from saved `dead` | Restore writes a spare row at the restored boundary. Capacity must exist before the copy; empty checkpoints do not write a spare. Checkpoint save does not copy completed pooled rows. | Used by `src/program/generate.cpp` prompt cache, parking, slot transfer and disk restore flows. Split-stage checkpoints save each stage separately. |
| Slot/session transfer | `src/program/generate.cpp` `copy_to_slot` / `copy_from_slot` save a `ConversationCheckpoint`, restore it into another session, and call `conversation_kv_restore` for K/V payloads | Transfers combine checkpoint running-state restore with K/V/snapshot pooled payload restore. Incremental storage must prepare destination indexer extent before copying or restoring into slot sessions. | Multi-stage and draft paths have separate sessions. This audit names the call sites but does not prove all transfer ordering under future incremental allocation. |
| Diagnostics and hashing | `src/program/generate.cpp` diagnostic hash near `pooled=` / `pooled_full=` / `dead=`; `src/prefill/prefill.cpp` `STRATA_IDX_FP16_CHECK` rounds `st.idx_pooled_rows`; debug dumps in `src/core/layer.cpp`; overlap diagnostics in prefill | Diagnostics must read only mapped/valid rows. `pooled=` is completed rows; `pooled_full=` adds the spare. `STRATA_IDX_FP16_CHECK` currently rounds full allocated rows and would need a valid/committed extent before incremental mode is enabled. | Diagnostic parity for SYCL mirrors is uncertain. Debug-only paths must either respect incremental extents or refuse the mode. |
| Native validation spans | `include/strata/kernels/native_qsa_indexer.hpp`; `src/kernels/cuda/native_qsa_indexer.cu` validates pooled span as `max_cells / R + 1` rows | Native wrappers currently validate enough contiguous memory for completed rows plus one spare, while session allocation provides `+2`. Incremental storage must reconcile wrapper span validation with the ceiling/valid rows it commits. | Validation is host-side CUDA/HIP source. SYCL `.dp.cpp` mirror may have drifted and needs separate audit before backend claims. |

## Backend and measurement limits for I1

This task adds CPU arithmetic and an inventory only. It does not allocate VMM ranges, query a real device allocation
granularity, validate `session_bytes` at runtime, alter graph captures, or access GPUs/model endpoints. CUDA, HIP and
SYCL backend behavior remains unchanged and must be separately built/audited when later tasks touch their interfaces.
