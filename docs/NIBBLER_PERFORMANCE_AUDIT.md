# Nibbler performance audit — 2026-10-11

Read-only inspection of the deployment and source at `5f9d4752` (`kv-unified`).
No model requests, benchmarks, builds, reloads, configuration changes or host tuning
were performed. Findings below identify optimization opportunities, not measured
speedups. The Beads descriptions contain implementation constraints and acceptance
criteria.

## Verified baseline

Commands used from the homelab checkout (mandatory preflight passed):

```sh
scripts/agent-preflight.sh
bin/hl peek host nibbler summary
bin/hl peek host nibbler path read /data/llm/Strata-run/nibbler/config.json
bin/hl peek host nibbler path read /data/llm/Strata/engine/BUILD.json
```

Read-only HTTP GETs used `http://nibbler.local.keen.land:8088`:

- `/running`: GPTQ model and pocket-tts both `ready`.
- `/metrics`: router host/swap metrics.
- `/upstream/qwen3.8-flash-next-nvfp4-gptq/metrics`: engine configuration,
  request summaries, parking counters and hardware. Snapshot `time=1791751971.2637427`.
- `/upstream/qwen3.8-flash-next-nvfp4-gptq/props`: two slots, vision enabled,
  unified/incremental KV, 262144 backing and 32768 GPU-resident cells.

| Property | Observed |
|---|---|
| CPU | AMD Ryzen 9 9900X, 12 cores / 24 threads |
| GPU | RTX 5060 Ti, 17103323136 bytes reported VRAM, PCIe width 8, maximum generation 5 |
| RAM | 132481556480 bytes total; host summary available 19384717312 bytes (~18.1 GiB) |
| Swap | Router reports 16566452224 used / 25785532416 bytes total |
| Engine | 0.1.41; BUILD.json arch 120, src `0622acf1cb5e6154` |
| Model | Non-abliterated GPTQ NVFP4, froggeric v22.5 tokenizer |
| Host expert arena | 64800 MiB |
| GPU expert cache | 1361 slots / 3588 MiB |
| CPU expert pool | 11 workers plus host (default source behavior) |
| PCIe fraction | 0.25 |
| Serving | Two slots, batch MTP, vision, 262144 context, INT8 KV |
| Parking | 8192 MiB budget, four entries, 4096 MiB available-RAM floor |
| GPU reserve | Configured 2048 MiB; engine startup free-VRAM field 1954 MiB |

Runtime config matched the tracked `deploy/nibbler/config.json`. Manifest provenance
was read, not verified by hashing/rebuilding the live executable. GPU utilization
was zero when inspected; idle PCIe generation 1 is not evidence of an underperforming
active link. Swap occupancy alone is not evidence of current swap thrashing.

## New Beads

| Bead | Priority | Finding |
|---|---|---|
| `str-pk5` | P2 | `STRATA_ADAPT_FETCH=2` is requested but `generate.cpp:7791–7803` excludes every `--batch` configuration. Report effective state and assess coherent opt-in batch admission; do not simply remove the guard. Ordinary adaptive swaps are not disabled by this finding. |
| `str-inn` | P2 | Runtime PCIe balancing requires `o.batch <= 0` (`generate.cpp:2707–2710`). Production instead retains a 0.25 NVFP4 split whose reference measurement used a 5090/9950X3D. Calibrate the split and worker count on the actual shared 9900X/5060 Ti. |
| `str-2bg` | P2 | Linux arena startup lacks the Windows direct-load implementation (`pinned.cu:523–667`); buffered reads copy through a temporary buffer into the 63.3 GiB arena. Historical reload trials exhausted swap. Add and measure an opt-in memory-conscious Linux loader with reliable fallback. |
| `str-5p1` | P3 | A fixed 2048 MiB reserve materially competes with the small 3588 MiB expert cache. Measure safe reserve/cache sizing with vision, full context and both slots; do not assume the reserve can safely be reduced. |

Latest twelve request summaries in the metrics snapshot report expert-cache hit
rates 38.8–49.4% and decode 27.3–29.7 tok/s. These are historical production rows
with differing prompts, not matched measurements or evidence of a regression.

## Existing issues and limits

- `str-7ze.30` already owns parking churn/over-capacity validation. Added audit
  evidence rather than creating a duplicate: four parked entries, 3107518412 bytes
  (~2.89 GiB), 55 parks, 14 restores, 34 evictions. The snapshot corroborates the
  slot-pressure concern but does not identify the cause of each eviction. Increasing
  entry count should be evaluated separately from allocating more RAM.
- `str-kp7.6` already covers finer prefill interleaving. No duplicate was filed.
- Added a caveated note to closed `str-7ze.18`: all twelve latest request rows have
  zero draft counters, but cumulative counters show 74457 offered / 51928 accepted.
  Target-only handoffs may legitimately suppress drafts. This is not proof of a
  regression; correlate native counters before reopening or treating MTP as off.
- The `hl path read` engine-log probe truncates at the first 65536 bytes. Those
  bytes contain old IQ3_S/0.1.38 records, not the current NVFP4 log tail. They were
  excluded from current-performance evidence. A bounded tail/offset path probe is
  needed for current native-log attribution; no raw SSH fallback was used.
- Root-level router `/props` and `/status` returned 404; the explicit model upstream
  route provided the read-only backend endpoints.
- No live CPU topology/CCD map, DRAM bandwidth, active PCIe bandwidth, hugepage
  backing, per-process swap attribution or worst-case VRAM was measured. No BIOS,
  hugepage, clocks, affinity or swap changes are recommended without that evidence.
- AVX-512 NVFP4 CPU kernels and sm_120 GPU targeting already exist; blindly adding
  compiler flags is not an established improvement. NVFP4 prefill `w4a4x2` is
  already selected in production.

All proposed performance experiments require a newly authorized quiet window,
matched alternating runs, correctness gates and a rollback. Nibbler is shared
production infrastructure, not a benchmark-only machine.
