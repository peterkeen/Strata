# Q4_K_XL performance audit — nibbler, 2026-10-13

Status: **inspection and prioritized Beads only**. Epic **str-s1w**. No source-code,
production configuration, service, model residency or benchmark workload was changed.
No speedup below is a measured candidate improvement.

## Target and evidence boundary

Read-only inspection used the homelab's `bin/hl peek host nibbler path read/ls`
probes, not ad hoc SSH. The launcher is important: `/data/llm/Strata` is an older
checkout and is **not** the current Q4 serving source directory.

`/opt/native-inference/bin/start-strata` currently selects:

- cwd: `/data/llm/Strata-kv-unified-c763cf3a`
- executable: that directory's `build/strata`
- configuration: `/data/llm/Strata-run/ud-q4_k_xl-unified/config.json`
- engine log: `/data/llm/Strata-run/ud-q4_k_xl-unified/engine.log`

The config, launcher, AGENTS.md, CMakeCache.txt, `src/prefill/moe_mmq.cu` and
`src/kernels/cpu/native_expert.cpp` were read in full from nibbler. The last two
source files exactly match this local checkout. The first 65,536 bytes of
`src/prefill/prefill.cpp` and `src/core/verify.cpp` also match locally.

**Probe limitation:** path reads stop at 65,536 bytes. Larger source files and the
engine log were truncated; this is not a full deployed-tree hash audit or a log
snapshot through the end of service activity. For later portions of large files,
analysis uses local `kv-unified` at `8888253c`, with the relevant changes against
`c763cf3a` checked. It does not establish that every remote byte matches that
commit. Before implementation/benchmarking, obtain full identity via an approved
ranged-read/hash probe (extend the probe if necessary), rather than bypassing it
with raw SSH. Source line numbers below refer to the local audit revision; symbol
names are the stable references.

No raw conversation contents are included here. Log observations below are from
existing shared production traffic, not controlled trials. The logs do not prove
exclusive device use, physical SSD traffic, or counterfactual candidate gains.

## Observed configuration and startup

| Item | Observed value |
| --- | --- |
| Model | Unsloth UD-Q4_K_XL, native GGUF shards, compat pack |
| Build | Release, CUDA architecture 120, `STRATA_MMQ_KQUANTS:BOOL=ON` |
| Serving | `parallel: 2`, unified KV, one configured GPU |
| Context | 262,144 logical cells; 32,768 GPU-resident KV cells; INT8 KV |
| Expert RAM | 64 GiB budget; startup confirms all 64 GiB pinned/mapped |
| Expert GPU cache | 1,790 slots, 5.21 GiB |
| Non-GPU expert complement | 66.52 GiB; 64 GiB kept by profile rank |
| CPU | 12 participating pool threads, 36 tasks/phase |
| PCIe probe | 28.9 GB/s H2D; selected fraction 0.55 |
| Draft | `--mtp ... --spec 4 --spec-min-p 0.5`; 835 MiB main draft layer |
| Batch draft | No `--batch-mtp` in config/startup argv; no enable witness in inspected log prefix |
| Parking | No `--conversation-cache-mib` in config; source default is zero |
| Prefill | `auto`; actual startup selected 9,728 tokens, 96-slot ring |
| File staging | Sample reported 100% page-cache residency, selected RAM stager profile |
| Free VRAM | Startup after load: 1,523 MiB; later graph captures consume more |

The host's documented CPU/GPU pairing is 9900X / RTX 5060 Ti (see
[NIBBLER_CALIBRATION.md](NIBBLER_CALIBRATION.md)); no fresh hardware inventory was
performed here. Parent-inherited environment was not exhaustively inventoried:
config absence alone is not proof that an environment-controlled option is off.
Reconfirm effective options in the measurement manifest.

Q4_K_XL is **mixed format**, not uniformly Q4_K: gate/up are mostly Q4_K with
Q5_K in layer 2; down is Q5_1 with several Q8_0 layers. This matters for dispatch,
blob sizes and fast-path eligibility. See [UNSLOTH_Q4.md](UNSLOTH_Q4.md).

### Existing traffic observations

Examples from the inspected log prefix:

| Work | Observed cost |
| --- | --- |
| 31 new prompt tokens, 20,135 reused | 790 ms, 39.2 read tok/s |
| 42 new tokens, 1,536 reused | 1,057 ms, 39.7 read tok/s |
| 60 new tokens, 1,439 reused | 1,413 ms, 42.5 read tok/s |
| 66 new tokens, 9,159 reused | 1,486 ms, 44.4 read tok/s |
| 1,967 prompt tokens | 3,424 ms, 574.4 read tok/s |
| 10,246 new tokens | 8,620 ms, 1,188.6 read tok/s |
| 18,699 new tokens | 11,826 ms, 1,581.2 read tok/s |
| 10,179 generated tokens | 351,196 ms, 29.0 tok/s |
| 2,966 generated tokens | 86,584 ms, 34.3 tok/s |
| 18,639 generated tokens | 625,780 ms, 29.8 tok/s |

Many ordinary decode examples are around 25–34 tok/s; this is not a distribution
estimate. The sharp short-suffix penalty is a stronger actionable signal than
assuming all prefill sizes share the same bottleneck.

One early, **six-window** batch sample reports 74.65 ms run time, 63.93 ms CPU
experts, 29.25 ms GPU-reach wait and 6.51 ms adaptive wait, with 43.7% of routed
entries on CPU, 40.0% PCIe and 16.3% VRAM. CPU/GPU timings overlap; **do not sum
them** or extrapolate six startup windows to steady state. Longer requests show
much warmer GPU caches, with the reported decode-cache ratio often near 60–65%;
that ratio's denominator is not necessarily all routed entries.

The file-tier byte counter is logical data served/assembled. Even hundreds of GB
there does **not** prove hundreds of GB physically read from SSD. The warm-cache
startup evidence argues against treating this as the old 64-GB-PC SSD bottleneck
without `/proc/self/io`/fault evidence.

## Prioritized work

Priority reflects expected impact and confidence, not promised percentages.
Within P1, start with the measurement plan, then the short-prefill and dispatch
work. Existing memory work remains a major complementary opportunity.

| Priority | Bead | Outcome / likely benefit |
| --- | --- | --- |
| P1 | **str-s1w.1** | Q4-specific dry-run measurement matrix and stage-cost report; prerequisite for comparable results |
| P1 | **str-s1w.2** | Fix short-continuation path selection using total verify vs batched cost; targets repeated ~1-second TTFT costs |
| P1 | **str-s1w.3** | Safely reuse the CPU expert pool for small prefills with two slots; reduces sparse-expert GPU streaming |
| P1 | **str-s1w.4** | Slot-aware CPU/PCIe decode balancing; removes a measured-cost optimization's batch exclusion |
| P1 | **str-s1w.5** | Qualify existing batch MTP specifically for Q4; potentially substantial overlapping-decode benefit, offset by draft VRAM cost |
| P1 | **str-s1w.10** | Model demand-allocated batch sessions and reversible expert-cache budgeting; avoid paying the ~1 GiB slot cost during solo use |
| P1, existing | **str-a2y** | Incremental/pooled indexer storage and actual cache-capacity brokerage; do not duplicate this epic |
| P2 | **str-s1w.6** | Reuse bounded GGUF fetch workers instead of creating/joining threads for each demand staging call |
| P2 | **str-s1w.7** | Prove size-class ownership rotation for mixed-format adaptive exchanges; removes a host memcpy eligibility gap |
| P2 | **str-s1w.8** | Bounded Zen5 Q4_K gate/up prototype, benchmarked against the actual compiled ggml path |
| P3 | **str-s1w.9** | Bounded CUDA fused Q4_K prefill gate/up experiment, compared to already-enabled MMQ |
| Existing review | **str-7ze.30** | Revisit parking/replay economics for the Q4 config; previous NVFP4 budget recommendation is not transferable |

### Short continuation path selection

`src/program/generate.cpp` defaults `Options::short_read` to 64 (~679).
`windows_ok` (~11865) chooses verify windows for each sufficiently short segment;
it does not use measured cost or total loan/refill overhead. Its comment assumes
~16 ms/token versus ~480 ms batched setup/refill from another configuration.
Nibbler's residual-token observations make that crossover worth revisiting.

A 66-token request can contain several checkpoint/header segments; do not infer
that all 66 tokens took the batched path. Instrument actual segments first. Compare
existing `--short-read` settings before writing a policy. Preserve image rows,
checkpoint exactness, draft provenance, cancellation and reused prefixes.

### CPU prefill share is excluded by slots, not merely disabled in the config

`generate.cpp` (~7323) defines `share_pool = o.batch <= 0 && !o.no_pool`; later
`set_cpu_pool` is conditional on it. `prefill.cpp:196–258` already has a measured
CPU-share policy and busy guard. With `parallel:2`, setting an environment switch
alone cannot attach the pool. The improvement requires a proven quiescent lease
from the scheduler, **not** deleting the guard and racing the decode pool.

This has a plausible large effect on 256–2,048-token sparse prefills. Existing
comments contain measurements on other hardware, not a nibbler speedup claim.

### Decode split ignores the available measured balancer in batch mode

`generate.cpp:3052` requires `o.batch <= 0` for `pcie_balance()`.
`PcieModel` in `include/strata/core/expert_source.hpp` and verifier timing in
`src/core/verify.cpp` already support cost-based balancing elsewhere. A link probe
cannot determine Q4 CPU kernel throughput or the best split as window width changes.

First compare fixed fractions using the Q4 matrix; then opt in to a batch-aware
model only if the headroom is real. Keep samples coherent across slots/window
widths and preserve explicit operator overrides. Do not copy the previous NVFP4
calibration's optimum.

### Batch MTP: existing code, different Q4 economics

The fork already supports unified batch MTP and has correctness/pressure tools;
closed `str-ro1` and `str-7ze` tasks cover the prior NVFP4 work. Current Q4 config
only requests ordinary MTP/speculation. Qualification, not a new implementation,
is needed. Test **actually overlapping** slots, real offer/accept/reject counts,
restored target-only fallbacks and total expert-cache loss from per-slot drafts.
A higher acceptance count is not a win if the smaller expert cache slows every
verify step. Deterministic proposal hooks are correctness tools, not benchmarks.

### File demand workers: a concrete overhead, not an assumed disk problem

`FileExpertSource::prefetch` calls `fill_many` before CPU jobs are built in expert
dispatch. `fill_many` (~1519–1582) creates and joins `std::thread`s each call,
including for page-cache-warm GGUF copies. A persistent bounded team can remove
this overhead without changing the existing completion barrier. Test inline
small jobs, shutdown, failures and stage-buffer fences. A more ambitious async
compute-before-fill pipeline is deliberately outside this first task.

### Mixed-size exchange rotation cannot currently activate for Q4

`FileExpertSource::reserve_exchanges` (~3250) requires all layer blob sizes to
match the maximum before enabling ownership rotation. Q4's mixed formats fail
that predicate. `commit_copies` and the copy branch of `commit_exchanges` then
copy each evicted blob into the complement. The log's cumulative exchange count
reaches 738,157 in the inspected prefix, so even a small per-exchange cost repeats
often; actual time saved still needs measurement.

A pure CPU size-class ownership model now exists in `exchange_storage.hpp`, with
tests in `exchange_storage_test`. It does not change `expert_source.cpp` or the
runtime `STRATA_EXCHANGE_ROTATE` eligibility gate yet. The existing uniform model
is kept for equal-size slots; the new model describes the later mixed-size case:
resident experts keep their compact offsets and each supported blob size gets its
own spare queue. A commit between same-size experts swaps metadata only, recycles
`in`'s old host/device alias as the same class's spare, and counts the avoided
host-copy payload. Unsupported swaps (missing class, cross-size `in`/`out`, wrong
staged pointer, exhausted class capacity, bad alignment/range) fail before
changing residency, so engine integration can retain the current memcpy path for
those cases.

The real UD-Q4_K_XL routed-expert blob sizes are derived from the packer's rule in
`tools/iq_pack.py`: `blob = gate_per_expert + up_per_expert + down_per_expert`.
The format mix is documented in `docs/UNSLOTH_Q4.md`: gate/up are Q4_K except
layer 2 Q5_K; down is Q5_1 except layers 2, 4, 30, 46 and 47 Q8_0. The expert
shapes used by the prompt/expert code are gate/up as two 640 x 2560 projections
and down as one 2560 x 640 projection; gguf block sizes are Q4_K 144 bytes per
256 values, Q5_K 176 per 256, Q5_1 24 per 32 and Q8_0 34 per 32 (also asserted by
`moe_fused_iq.cu`). Therefore:

| Layer formats | Per-expert bytes |
| --- | ---: |
| Q4_K gate/up + Q5_1 down | `2 * 640 * 2560 / 256 * 144 + 2560 * 640 / 32 * 24 = 3,072,000` |
| Q4_K gate/up + Q8_0 down | `2 * 640 * 2560 / 256 * 144 + 2560 * 640 / 32 * 34 = 3,584,000` |
| Q5_K gate/up + Q8_0 down | `2 * 640 * 2560 / 256 * 176 + 2560 * 640 / 32 * 34 = 3,993,600` |

Bounded memory proof: the model stores one atomic slot id per expert, one immutable
host/device pair per resident or spare buffer, one byte count per expert only in
the classed model, and one small record per class. Spare payload is the configured
sum of per-class buffers, e.g. one spare for each UD-Q4_K_XL class is 10,649,600
bytes rather than `3 * 3,993,600 = 11,980,800` bytes, and no rule pads the 64-GiB
resident complement to the maximum blob. Wiring points for a later integration are
limited to building a class table from `layer_blob_bytes_` in `reserve_exchanges`,
allocating mapped exchange buffers per class, making `exchange_buffer(q)` and
`stage_exchange` address a `(bytes, q)` class queue, and keeping the existing copy
path whenever class setup or commit validation fails.

### Kernel experiments come after dispatch/memory opportunities

- `native_expert.cpp:168–176` has an opt-in AVX2 Q4 multi-token kernel, but no
  dedicated Strata AVX-512 Q4 gate/up dispatch. Its default ggml fallback may
  **already** use AVX-512; compare actual compiled code. Existing KQ256 comments
  explicitly report little/no engine benefit at the usual small group sizes.
  Prototype one format/operation, not an entire mixed-format kernel family.
- `moe_fused_iq.cu:1076–1088` only exposes the Q4/Q5/Q8 mixed-format fused path
  under HIP guards. CUDA Q4 cannot gain it through `STRATA_PF_FUSED_KQ=1`.
  A standalone Q4 gate/up experiment can test intermediate-buffer and borrowing
  savings. Compare against K-quant MMQ already active on nibbler; do not claim
  the older FP16-to-MMQ speedup a second time. Q5_K/down support and production
  integration are follow-ups only if the experiment pays.

### Reuse the existing cache/indexer and parking work

`str-a2y` / [INDEXER_MEMORY_PLAN.md](INDEXER_MEMORY_PLAN.md) already owns the
incremental/pooled indexer design. At 262,144 cells, one private completed-key
payload is about 384 MiB; multiple sessions duplicate it. Recovering actual
expert-cache slots is the valuable outcome, not merely observing free VRAM.
The entire 1.00-GiB slot-session overhead in the log is **not** all indexer memory.
Do not reduce logical context or reserve safety to manufacture a gain.

The current Q4 config omits a parking budget, unlike prior NVFP4 deployments.
`str-7ze.30` already owns measurement of replay versus parking costs. Add a Q4
arm and test more independent conversation chains than live slots before deciding
whether RAM should go to parking or the expert complement. Existing small/long
prefix reuse does not prove parking is working; in-slot reuse is separate.
Do not blindly restore an 8–12 GiB parking budget onto a pinned 64-GiB Q4 arena.

## Follow-up: actual ggml Q4 fallback (2026-10-14)

For `str-s1w.8`, a temporary invocation-local increase of the existing path-read
probe limit to 2 MiB allowed complete reads of the deployed build's
`compile_commands.json`, ggml `ggml-cpu.c`, and x86 `quants.c`. The existing
allowlist/audit logging remained active; no repository or host config changed.
This resolves the source/compile-command question for this fallback, not a full
binary disassembly or deployed-tree identity audit.

- Build source is `/data/llm/Strata/third_party/llama.cpp/ggml`, selected by
  `STRATA_GGML_DIR`, not a llama.cpp tree under the newer Q4 checkout.
- The generated command for `ggml-cpu/arch/x86/quants.c` is GCC 15, `-O3`,
  **`-march=native`**, with no later ISA override. Thus cache entries such as
  `GGML_AVX512=OFF` do not mean the native build excludes AVX-512.
- `ggml-cpu.c:311–313` maps Q4_K's `.vec_dot` to `ggml_vec_dot_q4_K_q8_K`.
- In the host's x86 `quants.c:2038–2214`, that function selects an **AVX2**
  implementation, then AVX, then generic. Its hot loop uses 256-bit
  `_mm256_maddubs_epi16`, `_mm256_madd_epi16` and FMA. There is **no explicit
  AVX-512/512-bit or VNNI-specialized Q4_K dot-product branch**.
- The Q5_K dot similarly uses an AVX2 implementation. Q5_1 and Q8_0 down dots
  differ: their 256-bit helper can select AVX-512 VNNI+VL via
  `_mm256_dpbusd_epi32` (`mul_sum_us8_pairs_float`, ~106). AVX-512 ISA use and
  512-bit vector width are distinct; do not label all fallback projections AVX2-only.

Conclusion: `.8` is not just enabling a missing build flag. It can explore a
genuinely different Q4_K gate/up dot/reuse implementation versus an already
optimized AVX2/FMA baseline compiled natively. Compiler-emitted instructions have
not been disassembled, so this is not a claim that the resulting function contains
zero AVX-512-encoded instructions. No speedup is established.

## Follow-up: defer parallel VRAM and qualify parallel MTP

[LAZY_PARALLEL_MEMORY.md](LAZY_PARALLEL_MEMORY.md) records the full-source
investigation. Two extra ~512-MiB sessions are allocated before expert-cache sizing,
even while the main session handles a single request. Lazy allocation is feasible
but requires shared-runtime/verifier binding changes and safe cache capacity
exchange; the existing resize command refuses this resident-complement mode.
P1 `str-s1w.10` owns the resource model and implementation breakdown, coordinated
with `str-a2y` (the indexer savings overlap, not add).

Existing parallel MTP shares the main draft weights and owns only private slot
state/buffers. The current Q4 config meets the basic gates; qualification remains
`str-s1w.5`, independent of lazy resources. Neither feature was enabled live.

## Ruled out as simple fixes

- **Enable K-quant MMQ:** already enabled in deployed build cache.
- **Fix SSD stager oversubscription:** this run already detects warm GGUF pages
  and selects the RAM profile. Recheck under later pressure, but not a missing fix.
- **Pin the resident arena:** already fully pinned/mapped.
- **Enable CUDA fused Q4 via environment:** coverage is absent on CUDA.
- **Enable exchange rotation via environment:** mixed-size eligibility blocks it.
- **Increase resident RAM blindly:** startup complement gap is ~2.52 GiB, but
  prefill also borrows GPU-only experts and the process shares RAM with live
  services. Startup's warning about a hypothetical 32K loan is not the actual
  selected 9,728-token loan. Larger budgets are not automatically better overall.
- **Switch quantization, skip experts, reduce context:** outside the requested
  goal of retaining Q4 quality and current serving capability.

## Validation and dependencies

`str-s1w.2/.3/.4/.5` depend on the Q4 measurement-plan task `.1`. Worker and
standalone ownership/kernel prototypes can proceed independently; final engine
acceptance still requires the shared matrix. Existing epics are referenced, not
made artificial implementation blockers.

Each task includes scope, exclusions, acceptance criteria and commands. Tests
named as new deliverables must be added by implementation; listing them is not
claiming they exist or passed. Benchmark promotion requires:

1. Fingerprinted same-day baseline/candidate, identical input IDs and sampling,
   comparable cache warmth, interleaved repeats and bounded request lengths.
2. Prompt-read and TTFT separately, accepted output tokens/sec, per-client tails,
   effective CPU/PCIe shares, draft acceptance and actual expert slots.
3. Logical file bytes separated from physical I/O; RAM/swap/VRAM high-water marks,
   idle-vs-busy device state, and stop floors from a freshly approved safety plan.
4. Exact default-path behavior; for opt-in reordered arithmetic, established
   logit/quality gates plus native lifecycle/cancel/pressure/model tests. Build
   every affected backend and disclose unavailable checks.
5. No rollout from this audit. Live tests, reloads, budget changes and deployment
   need separate authorization. In particular, revalidate or resolve **str-inn.5**
   before repeating reload-based calibration.

## Checks and remaining uncertainty

- Homelab agent preflight: **passed**.
- `bin/hl peek triage nibbler --json`: **inspection_complete=true**, zero observed
  unhealthy containers. Separate Gatus `host-reboot/nibbler` check was failing;
  no reboot or remediation attempted. Container health is not inference health.
- Target launcher/config/build/source and existing log prefix: inspected read-only
  with the completeness limits above.
- No compilation, GPU/model parity test, fresh inference request or performance
  experiment was run: this is a planning-only change.
- Historical `str-inn.5` swap/recovery blocker remains open. Current swap health
  was not established by this audit; startup available-RAM logs are not a current
  memory safety clearance.
- No candidate speedup, optimal split, quality equivalence or end-to-end deployed
  source identity is claimed without the missing measurements/identity checks.
