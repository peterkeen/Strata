# Demand-allocated parallel sessions and parallel MTP

Investigation: 2026-10-14. **Design recommendation, not implemented or deployed.**
Related: `str-s1w`, `str-s1w.5`, `str-a2y`, and
[Q4 performance audit](Q4_NIBBLER_PERFORMANCE_AUDIT.md).

## Answers

1. **Yes, slot-specific VRAM can in principle be deferred until actual concurrent
   work arrives. No current setting implements this.** The current two-slot
   engine eagerly allocates two extra sessions even though one request alone
   uses the main session. Startup logs show about **1.00 GiB** for those sessions.
2. **Yes, MTP already supports parallel decoding in this fork.** `--batch-mtp`
   enables one private proposal per active slot per window, with shared draft
   weights and private draft state. The current Q4 config meets the basic gates
   but does not contain that flag. It needs Q4-specific qualification, not a new
   speculative-decoding architecture.
3. The useful feature is **elastic parallel resources plus expert-cache capacity
   exchange**, not merely moving a `cudaMalloc` later. Leaving a permanent 1 GiB
   empty reserve would avoid an allocation but would not restore solo expert-cache
   capacity or its potential speed benefit.

No live setting was changed or inference request/benchmark submitted.

## Live-source confirmation

The launcher and `/data/llm/Strata-run/ud-q4_k_xl-unified/config.json` were re-read
and are unchanged from the Q4 audit. Serving cwd remains
`/data/llm/Strata-kv-unified-c763cf3a`, `parallel:2`, main `--mtp` / `--spec 4`,
262144 logical cells and 32768 resident main-KV cells.

The full deployed `src/program/generate.cpp` (1,020,578 bytes), `session.cpp`,
`mtp.cpp` and `verify.cpp` were retrieved through the existing path-read probe
with its invocation-local 2 MiB read limit; no path-allowlist or host change.
The latter three match the local source exactly. The local-vs-deployed generate
comparison shows conversation-cache changes, not changes to the allocation,
MTP gating or batch paths discussed here. References below use local symbol names
and approximate line numbers, not an assertion of identical whole-tree revision.

## What is allocated before a second client appears

`src/program/generate.cpp`, batch setup (~4704–4908):

- The main session `ss` already exists.
- For every configured batch slot, `session_bytes(..., share_kv=&ss)` is priced,
  `cudaMalloc` is called, and `session_init` / `session_zero` run.
- This happens **before** `--expert-cache auto` is sized. Thus slot VRAM is
  directly subtracted from the expert-cache opportunity at startup.
- `SharedKvRuntime` is then constructed with `{main, slot0, slot1}`. Its
  constructor requires real non-null, compatible sessions, initializes their
  private addressing, and publishes every sequence's bindings.
- `Verifier::init_slots` (~2610 in `verify.cpp`) also requires real sessions and
  allocates batch commit staging and indexer-tail snapshots. Batch CUDA graphs
  are subsequently captured on demand and retain session addresses.
- With `--batch-mtp`, `MtpDrafter::load(..., &main_drafter)` also runs eagerly for
  each slot, followed by per-slot bind/logit/residual buffers. The verifier is
  initialized for the eight-row maximum in the batch-MTP case.

Unified KV already shares the expensive **physical main K/V pools and RoPE**.
The extra sessions are not duplicate full KV caches. `session_bytes` still
includes private GDN recurrence, indexer keys/address tables, session workspaces,
MoE/block buffers and PLE history.

Source-derived canonical geometry, not a new allocation measurement:

- Private pooled indexer payload: about **384 MiB per session** at 262144 cells
  (`12 * (262144/4 + 2) * 128 * 4` bytes).
- Private GDN recurrent/conv state: about **112.22 MiB per session**
  (`36 * (128*48*128 + 10240*3) * 4` bytes).
- Private mappings, workspaces, alignment and other buffers add to that.

This explains the observed roughly **512 MiB per slot / 1 GiB for two slots**.
The indexer work in `str-a2y` attacks much of this same overhead; its savings and
lazy allocation savings must **not** be added as if independent.

## Recommended lifecycle

Retain `parallel:2` as a **logical concurrency ceiling**, separate from the number
of physically allocated batch sessions. Start with only main-session resources.
Keep shared KV physical ownership/capacity unchanged.

### Solo -> concurrent

At the existing HANDOFF/admission boundary, before transferring the first stream:

1. Pause at a safe engine boundary; finish commits, drain adaptive transfers and
   ensure no prefill loan or graph still owns memory to be reclaimed.
2. Compute the complete activation budget: both batch sessions, any batch-only
   verifier workspace, graph-capture margin, optional slot drafts, temporary
   handoff images and unchanged driver safety reserve.
3. Give back sufficient **expert-cache** physical segments. Before eviction,
   establish a valid CPU/file-tier source for every displaced expert; publish
   residency changes in the correct fenced order.
4. Allocate/init both slot sessions and attach stable sequence IDs to the shared
   KV runtime and verifier. Under the current main-admission design two clients
   need two batch sessions, not just one: main remains the prompt admission owner.
5. If batch MTP is requested, create the private slot drafters sharing the main
   weights, then transfer the exact coherent draft history where available.
6. Publish readiness only when all allocations/bindings are valid; perform the
   existing handoff and admit the waiting request.

If required slot allocation fails, leave the first request intact and keep the
second queued. Do not advertise a broken slot or turn allocation pressure into a
process exit. Optional draft allocation can degrade to target-only batching if
cleanup/provenance is proven safe. Failure recovery must cover partially mapped
cache growth/shrink, not assume those APIs are transactional.

### Concurrent -> solo

When only one stream remains and no request waits:

1. Hand it back through the existing BHANDOFF path, including coherent private
   draft history when possible.
2. Preserve reusable idle conversation state before removing its GPU resources,
   or apply an explicit tested replay/eviction policy. A finished slot is also a
   warm conversation cache; `active=false` is **not** permission to discard it.
3. Fence and destroy/invalidate graphs that reference slot or draft addresses,
   detach the bindings, then free private allocations. Never free the main
   session's shared KV pools or a sibling's referenced pages.
4. Return physical capacity to the expert cache, refill valid expert placements,
   refresh prefill loan sizing and reports.

Use hysteresis / an idle grace policy to avoid allocation and expert-refill churn
on rapid tool turns. The exact timeout is a measurement/product choice, not a
magic constant inferred from this audit. A simpler first milestone can allocate
on first concurrency and retain the resources until an idle boundary, but must
be labeled **one-time lazy allocation**, not full recovery of solo performance.

## Why this is not a small flag change

- `SharedKvRuntime` assumes all sessions exist at construction; it needs safe
  attach/detach or a deferred-binding design with stable logical sequence IDs.
- `Verifier::init_slots` is not a detach/reinitialize API. Graphs embed pointers;
  allocation generations and graph invalidation or stable virtual addresses are
  required. Do not repeatedly call init_slots without explicit cleanup.
- `ExpertCache` already has segmented VMM grow/shrink support, but production Q4
  does not enable it. More importantly, `generate.cpp`'s `vram_command` (~9875)
  explicitly refuses **resident low-RAM mode** because the GPU holds experts the
  compact RAM complement omits. This is exactly the current Q4 configuration.
- The existing resize command operates **between requests**, not during the live
  solo-to-parallel handoff. Reusing its primitives requires a new safe scheduling
  boundary; deleting its guard is not sufficient.
- Source lifetime, RAM-complement/adaptive exchange ownership, mixed-size cache
  slots, prefill loans and residency tables must remain coherent when the cache
  changes size. Coordinate with `str-a2y`'s budget broker rather than creating a
  competing VRAM owner.
- A one-request engine with configured slots also loses some optimization
  eligibility (`o.batch` guards CPU prefill sharing and PCIe balancing). Lazy
  memory alone does not remove those exclusions; `str-s1w.3/.4` own them.

The expected solo opportunity is reclaiming **roughly 1 GiB of eager session
allocations**, plus separately accounted batch-only overhead, for potentially
hundreds of Q4 expert slots. Actual capacity/performance recovery remains
unmeasured and depends on alignment, profile, safety reserve and indexer changes.
First overlap pays allocation/initialization, cache eviction and graph warmup
latency, so report that latency separately from steady parallel throughput.

## CPU resource-policy model added by str-s1w.10

`include/strata/core/parallel_resource_policy.hpp` is a GPU-free contract model,
not runtime integration. It names the resource classes that a future lazy
implementation must price and publish:

| Class | Accounting rule |
| --- | --- |
| Shared main pools / weights | Baseline solo resources. The lazy lifecycle never frees or double-charges the main shared KV pools, RoPE, target weights, main draft weights or shared draft head storage. |
| Private slot sessions | Per physical batch session private non-indexer bytes: GDN recurrent/conv state, per-session maps, workspaces, MoE/block buffers and PLE history. |
| Brokered slot indexer | Per-slot private indexer bytes owned by `str-a2y`'s indexer/cache budget broker. The policy ledger reports these separately so indexer pooling/incremental savings and lazy-session savings are not added twice. |
| Batch verifier and graphs | Batch-only verifier staging, tail snapshots, commit staging and graph-capture margin. Graphs are tied to an allocation generation. |
| Slot draft rings/buffers | Optional batch-MTP private mutable draft state only. Shared draft weights are baseline, not duplicated per slot. |
| Transition temporaries | Handoff images, exact-prefix/draft-transfer scratch and activation-only graph/copy margin. |
| Safety reserve | Driver floor that remains unavailable to both expert-cache and session growth. A hidden permanent reserve is not counted as recovered solo cache capacity. |
| Expert-cache capacity | The elastic source/sink for physical bytes. Giving capacity back must create coherent cache misses or valid refill targets, never stale residency. |

The model keeps `parallel=2` as a **logical** advertised capacity while physical
batch sessions can be absent in solo mode. Under the current main-admission
architecture activation still requires two physical batch sessions for two
clients, because the main session remains the prompt-admission owner. The pure
state machine is:

1. **solo**: main stream is valid; no private slot sessions, batch verifier,
   batch graphs or slot-draft buffers are published.
2. **activating**: a second stream is queued. At a safe boundary with commits,
   DMA and adaptive transfers drained and no prefill loan outstanding, the
   runtime asks the cache/indexer broker for the full activation budget. Only
   after sufficient expert-cache capacity is coherently evicted or reclaimed may
   it allocate slot sessions, bind logical slots, initialize the batch verifier,
   capture generation-tagged graphs and optionally allocate slot drafters.
3. **parallel**: publication happens only after every required allocation,
   binding and graph generation is valid. Optional draft allocation failure may
   fall back to target-only batch mode if partial draft state is released and
   draft provenance stays explicit.
4. **draining**: after one stream remains and the idle-grace hook allows
   reclamation, the runtime first preserves the finished slot's warm
   conversation or evicts it under an explicit replay policy, invalidates graphs,
   detaches bindings, then frees private resources. Only after private pointers
   are unreachable may capacity return to the expert cache.
5. **solo**: main stream continues. If expert-cache grow/refill fails after
   private resources are released, the engine is still in a coherent solo state
   with fewer/hotter-missing cache entries; it must report that separately from
   a clean refill.

The CPU test `src/core/parallel_resource_policy_test.cpp` injects failures at
cache shrink, partial slot mapping, binding/publication, graph invalidation,
draft allocation and cache grow. The checked behavior is that the first stream
remains intact, the second stream remains queued unless explicitly cancelled,
logical advertised capacity stays consistent, and rollback releases only owned
private slot/draft/verifier resources. The model deliberately has no callback
that can free the shared main KV pools.

The graph contract is generation-based: every successful activation captures
batch graphs against one allocation generation; reclamation must invalidate those
graphs before detach/free. A failed invalidation blocks detach so stale graph
pointers cannot reference freed sessions. Runtime integration can satisfy this
with graph destruction, stable virtual addresses, or another reviewed mechanism,
but it must expose the same generation/fence semantics.

The cache/indexer broker contract is intentionally aligned with
[INDEXER_MEMORY_PLAN.md](INDEXER_MEMORY_PLAN.md): str-s1w.10 does not implement
private-indexer VMM, pooled keys or a competing budget broker. A future runtime
must submit the brokered indexer portion of the activation budget to that broker,
respect its victim exclusions (in-flight copies, prefill loans, adaptive
publication, retained pointers and unsupported multi-GPU cases), and treat
partial publication failures as coherent cache misses rather than re-advertising
stale experts. Because the live resident-complement configuration makes
`vram_command` refuse resize when `src.complement_ready()`, the future feature
needs a new safe handoff boundary and resident-complement-aware broker path; it
cannot reuse the command by deleting that guard.

This model proves the lifecycle shape and failure semantics only. It does **not**
allocate runtime sessions, change serving behavior, recover VRAM, enable batch
MTP, resize the expert cache, or benchmark first-overlap latency.

## L1 implementation note: stable logical slot registry

`SharedKvLogicalSlots` in `include/strata/core/shared_kv_runtime.hpp` is a
small serving-thread registry primitive. Its logical IDs are created at
construction and are independent of whether a physical session pointer is
present. Attach runs an initialization callback before publishing the pointer;
a failed callback leaves the slot empty and does not advance its generation.
Detach runs its binding-removal callback before clearing the pointer; failure
leaves the existing binding intact. Successful attach, detach and reattach each
advance that slot's generation, which is the key consumers use to invalidate
allocation-dependent state.

`src/core/shared_kv_runtime_lazy_test.cpp` exercises this CPU-only registry
contract, including initialization failure, detach failure, and stable-ID
reattach. The existing CUDA `SharedKvRuntime` session/page-pool implementation
remains on its eager constructor path; this registry primitive does not itself
add runtime CUDA attach/detach calls or prove GPU pool safety. Therefore actual
runtime attach/detach integration and the canary GPU gate remain outstanding.

## Follow-up implementation tasks proposed by str-s1w.10

These are proposed Beads, not implemented work:

| Proposed task | File scope | Acceptance | Verification |
| --- | --- | --- | --- |
| L1: deferred logical-slot registry | `include/strata/core/shared_kv_runtime.hpp`, `src/core/shared_kv_runtime.cpp` if split exists, focused tests under `src/core/` | Stable logical slot IDs can exist before physical sessions; attach publishes only fully initialized sessions; detach removes bindings without touching shared main KV. | CPU attach/detach/failure tests; CUDA/HIP/SYCL default builds still compile; isolated GPU gate exercises attach, detach and reattach with canary sequence IDs. |
| L2: verifier slot attach/detach and graph generations | `include/strata/core/verify.hpp`, `src/core/verify.cpp`, batch graph tests | Replace one-shot `init_slots` assumptions with explicit slot attach/detach cleanup; graph keys include allocation generation or use stable addresses; failed graph invalidation blocks free. | CPU graph-registry/failure model; CUDA build and graph replay/destroy canary; HIP/SYCL build disclosure for touched declarations. |
| L3: activation budget bridge to str-a2y broker | `include/strata/core/indexer_budget_broker.hpp`, `include/strata/core/parallel_resource_policy.hpp`, runtime planning seam only | One activation plan prices private non-indexer bytes, brokered indexer bytes, verifier/graph/draft bytes, temporaries and reserve; no hidden empty reserve; rejected/partial broker plans keep first stream and queued waiter safe. | CPU broker-policy integration matrix; backend builds for changed headers; isolated GPU low-memory negative with no stale expert residency. |
| L4: expert-cache shrink/grow at serving handoff boundary | `src/program/generate.cpp` admission/handoff boundaries plus `include/strata/core/expert_cache.hpp` use sites only | Shrink happens only after commits, DMA/adaptive drain and no prefill loan; resident-complement mode uses a new coherent path rather than the refused `vram_command`; grow/refill after reclaim reports degraded-cache failures. | CPU scheduler-state test for safe-point gates; CUDA single-GPU shrink/grow/refill canary with resident complement; HIP/SYCL builds or explicit unsupported diagnostics. |
| L5: private slot-session lazy allocation | `src/program/generate.cpp`, `src/core/session.cpp`, `include/strata/core/session.hpp` | Startup with configured parallel=2 does not allocate private slot sessions; cold first overlap allocates both current batch sessions; allocation/map/bind failure queues second request and preserves first request tokens/state/cancellation. | CPU fake-allocation lifecycle test based on this model; CUDA actual session allocation/free canary; model-serving smoke on isolated GPU for cold first overlap, cancellation and reactivation. |
| L6: warm-conversation preservation or replay eviction | `src/program/generate.cpp`, conversation cache/session-save helpers (`include/strata/core/conversation_cache.hpp`, `src/core/conversation_*`) | Finished slots are either preserved for exact reuse or intentionally evicted with a replay record; `active=false` alone never discards a warm conversation. | CPU conversation policy tests; backend builds for touched helpers; isolated GPU handoff/restore test with exact-prefix and target-only restore cases. |
| L7: optional lazy batch-MTP private state | `src/core/mtp.cpp`, MTP headers, `src/program/generate.cpp`, `include/strata/core/batch_mtp_policy.hpp` | Batch MTP remains optional; slot drafters share main weights, transfer only exact coherent prefix state, release partial draft allocations on failure and fall back to target-only when configured optional. | Existing and new CPU batch-MTP policy/draft-coherence tests; CUDA two-stream accept/reject/cancel gate; Q4 performance qualification remains str-s1w.5. |
| L8: lifecycle telemetry and first-overlap report | CLI/reporting in `src/program/generate.cpp`, docs | Report logical capacity, physical allocated slots, expert-cache bytes/slots recovered, brokered indexer bytes, first-overlap activation latency, refill latency and degraded-cache state separately. | CPU formatting tests or log parser fixtures; CUDA isolated measurement harness from str-s1w.1; no production rollout without fresh safety approval. |

## Parallel MTP: what exists and what must be qualified

`generate.cpp`'s `batch_mtp` gate (~4729) requires `--serve`, at least two slots,
`--mtp` and `--spec >= 2`; incompatible split/group/helper cases are rejected or
reported disabled. Current one-GPU Q4 serving satisfies the basic conditions.
The explicit candidate configuration addition is **`--batch-mtp` in `args`**;
this document is not authorization to apply it. Use one group for unified KV.

Each slot verifies `[current token, one draft token]`. With two coherent ready
slots, that is four verifier rows. On an accepted draft it commits the additional
verified token; on rejection it commits the correct target prefix. Solo execution
retains its ordinary speculative window. This is not four speculative proposals
per slot just because the solo config says `--spec 4`.

The drafters **share loaded weights and draft head storage**, not their mutable
KV history. Therefore enabling batch MTP does **not** duplicate the entire
~835-MiB main draft load for every slot. Private ring/buffers, logits, graph and
verifier costs still matter and should themselves be demand-allocated in the lazy
parallel design. Default `--mtp-window=32768` with context 262144 uses a bounded
private ring; preserve the window rather than reducing draft context to make an
unfair memory comparison. No exact incremental memory figure is claimed for
this live Q4 geometry without the candidate allocation log.

Evidence that this is implemented rather than conjecture:

- `src/core/mtp.cpp:245`: optional `shared` drafter borrows the existing weights.
- `generate.cpp:4893`: constructs per-slot drafters and calls `set_max_drafts(1)`.
- `generate.cpp:10334`: batch windows insert proposal rows per coherent slot.
- `batch_mtp_choose_window` reserves mandatory target rows first; optional-row
  shortage falls back to target-only rather than escalating pressure itself.
- Exact live-prefix handoffs can transfer draft state using
  `batch_draft_copy_to_slot/main`. Older checkpoints, incoherent clones and
  target-only restores cannot safely resume speculative offers.
- Prior **NVFP4**, short-context isolated nibbler gates (`str-ro1.4`) measured
  batch-MTP aggregate wall throughput improvements of **5.4–9.8%** over dual
  target-only in three repetitions. This is not a Q4 speedup forecast.
- `docs/BATCHING.md` also records Q4 gains on a different two-GPU machine. That
  is useful feasibility evidence, not validation of this single-GPU deployment.

Qualification stays in **str-s1w.5**. Run real two-stream accept/reject,
limits/cancel/pressure and coherent/incoherent handoff gates, then hook-free
same-day Q4 throughput/TTFT/memory A/B. Lazy slots and batch MTP are independent
axes; compare them independently before combining. A draft gain can be offset by
expert-cache loss or first-overlap setup cost.

## Acceptance for a future implementation

- Trace solo -> second arrival -> two streams -> one stream -> idle -> repeat,
  including failures and rapid bursts, with physical VRAM and actual expert slots.
- Before first overlap, no eager slot/draft allocation and no hidden permanent
  reserve consuming the claimed savings; base shared KV remains valid.
- Logical capacity remains two; allocation failure queues safely without losing
  first-stream state, tokens, ownership or cancellation semantics.
- Warm-prefix and exact draft-coherence contracts survive activation/reclamation.
- Neither CUDA graphs nor DMA accesses released/unmapped memory; CPU failure
  models, affected backend builds and isolated GPU gates required.
- Demonstrate solo recovery after cooldown, bounded first-overlap latency, and
  correct fixed-context mixed-format resident-complement operation.
- No live rollout or model unload without explicit authorization; revalidate the
  previously recorded memory/swap safety concern before isolated GPU work.
