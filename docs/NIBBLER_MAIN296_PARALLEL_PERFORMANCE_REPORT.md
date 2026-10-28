# Nibbler UD-Q4_K_XL: parallel inference performance investigation

Report written: 2026-10-16 UTC. Live observations were collected during the preceding investigation, around 2026-10-15 22:25–22:35 in the host's displayed time.

## Executive summary

The goal is to make `parallel: 2` as fast as `parallel: 1` on nibbler's current hardware and UD-Q4_K_XL model, ideally without doubling private memory. A series of small wins is acceptable.

There are several credible opportunities without starting with unified KV. In the investigated upstream version, enabling parallel 2 changes more than request scheduling:

1. It allocates two additional full sessions, displacing about **1.9 GiB of cached experts**.
2. That can push the model from a complete resident expert complement into **file-backed expert misses**.
3. It disables default routed-expert tail skipping and runtime PCIe-share refinement, including while only one request is active.
4. Concurrent decoding uses one MTP proposal per slot, sequential blocking drafter calls, synchronous batch commit, and some per-row operations where solo execution uses multi-row operations.

The first targets should be the expert-residency threshold and these solo-versus-batch execution differences. They are independently testable and do not require a unified-KV architecture.

**Feasibility assessment, not a measured result:** recovering single-request performance with parallel 2 enabled looks plausible. Matching parallel-1 aggregate throughput with two active requests is a sensible next target. Giving each of two simultaneous requests the full parallel-1 speed is substantially harder and is not established by this inspection.

### Scope and non-goals

- Investigated the configuration on nibbler and upstream source revision `296b59d2f73c8f04a805df29c2fa97655adace78`.
- Did not use the `kv-unified` branch as the implementation baseline or as the basis for the recommendations.
- No production configuration changes, restarts, builds, deployments, or inference benchmarks were performed.
- No proposed speedup percentages in this report should be interpreted as measured gains on nibbler.
- This is a fresh investigation of the selected upstream build, not an endorsement of earlier branch designs or benchmark conclusions.

## 1. Define the performance goal precisely

Three distinct measurements are needed:

| Case | Configuration | Active requests | Question |
| --- | --- | ---: | --- |
| A | parallel 1 | 1 | What is the current single-request baseline? |
| B | parallel 2 | 1 | What does merely enabling parallel service cost? |
| C | parallel 2 | 2 | What are aggregate throughput and each request's latency under concurrency? |

Success should be evaluated in stages:

1. B approaches A: parallel capacity does not penalize the common single-request case.
2. C's aggregate throughput approaches or exceeds A, with improved waiting time.
3. Each request in C approaches A's per-request speed, if hardware utilization and expert reuse allow it.

The third target implies roughly twice the aggregate throughput of A. It cannot be assumed to follow from eliminating configuration overhead.

Private host memory, pinned host memory, file-backed residency, GPU memory, and system memory consumption must be reported separately. They are not interchangeable measurements.

## 2. Verified deployment and hardware

### Launch and configuration

The live launcher `/opt/native-inference/bin/start-strata` selects:

```text
cwd:    /data/llm/Strata-main-296b59d2
server: /data/llm/Strata/.venv/bin/python -u -m serve.server
config: /data/llm/Strata-run/ud-q4_k_xl-main296/config.json
engine: /data/llm/Strata-main-296b59d2/build/strata
```

The router's model identifier still contains `nvfp4-gptq`, but the upstream Strata API and configuration identify the actual model as `qwen3.8-flash-next-ud-q4_k_xl`. The router label is not evidence of the loaded quantization.

| Item | Observed state |
| --- | --- |
| Engine API version | `0.1.44.1` |
| Source/build directory | `/data/llm/Strata-main-296b59d2` |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| GPU memory | 16,311 MiB reported total |
| GPU link | x8; maximum reported generation 5 |
| CPU | AMD Ryzen 9 9900X, 12 cores / 24 threads |
| Usable system RAM | About 123.4 GiB |
| Model expert pack | `/data/llm/models/unsloth/strata-ud-q4_k_xl-expbin/pack` |
| Native model | UD-Q4_K_XL GGUF shard set |
| Serving/requested concurrency | 1 / 1 |
| Resident expert budget | 65 GiB |
| Reported arena | 66,042 MiB, about 64.5 GiB |
| GPU expert cache | 2,482 experts / 7,407 MiB |
| Context limit | 262,144 tokens |
| KV format | INT8 |
| Resident KV cells | 32,768 |
| KV growth | Explicitly disabled |
| CPU pool | 11 workers plus participating host thread |
| Reported initial PCIe fraction | 0.55 |
| Speculative configuration | `--spec 4 --spec-min-p 0.5`, MTP enabled |
| Engine-reported speculative capacities | `spec=6`, `mtp_max=4`, `lookup=3`; not simply the configured MTP argument |
| Adaptation | `--adapt-async 1` |
| Expert profile | Learned profile, also configured as save destination |
| Conversation parking | 8,192 MiB, four slots, 4,096-MiB minimum-free guard |
| Short-read threshold | 32 tokens |
| Prefill | `auto` |
| Requested VRAM reserve | 1,024 MiB |
| Vision | Disabled |
| Shared reasoning setting | `medium` |

The GPU reported generation 1 while idle. This is not evidence that inference runs at generation 1; link generation must be measured under load before drawing a bandwidth conclusion.

### Current memory headroom

- Engine startup metadata reported **497 MiB free VRAM**.
- A live API snapshot reported 15,873 MiB used out of 16,311 MiB, approximately **438 MiB free**.
- The host summary reported about 29.3 GB available system memory at its observation time.

GPU headroom is tight. Do not reduce the reserve merely to increase expert-cache capacity. Check peak memory after graph capture, both-slot activation, long prompts, and parking activity. The historical log's low-VRAM warning is a warning, not proof that Linux driver paging was occurring.

### Current smoke result

The live process had one recorded completed request:

- Prompt: 180 tokens, no reused tokens.
- Output: 184 tokens.
- Prefill: 1,291.5 ms, 139.4 tok/s.
- Decode: 4,361.2 ms, **42.2 tok/s**.
- Drafts: 127 accepted of 143 offered.
- Reported cache hit rate: 50.8%; PCIe share: 27.5%.
- File-tier expert reads: **0**.

This confirms that the current configuration can serve and that this request did not need file-tier expert reads. It does not establish representative throughput, long-context behavior, or steady-state concurrency performance.

## 3. Build identity and source confidence

The selected local upstream revision is:

```text
296b59d2f73c8f04a805df29c2fa97655adace78
```

Build evidence on nibbler records:

- Release build.
- GCC/G++ 15.3.0.
- CUDA 13.3.73, architecture 120.
- Native experts and K-quant MMQ support enabled.
- GGML CPU compilation with `-march=native`.
- GGML commit recorded by configuration: `0ef1fc0fb`.
- GGML source directory: `/data/llm/Strata/third_party/llama.cpp`.

Stored build-time executable SHA-256:

```text
7a5637c23474c5320295ee9bba6c2f38467be92ac0534bfcbda37abc2417ccc4
```

This hash was read from `evidence/binary.sha256`; the investigation did not independently hash the current executable.

The source was inspected from an archive of the upstream commit in `/tmp/strata-audit-296b59d2`. Deployed source comparisons found:

| File | Comparison |
| --- | --- |
| `src/core/session.cpp` | Complete deployed file matched upstream archive |
| `src/kernels/cpu/native_expert.cpp` | Complete deployed file matched upstream archive |
| `src/program/generate.cpp` | Returned prefix matched; live probe truncated at 64 KiB |
| `src/core/layer.cpp` | Returned prefix matched; live probe truncated at 64 KiB |

Thus the directory identity, version, build evidence, and sampled source agree. This is not complete byte-for-byte attestation of the deployed source tree or executable.

All source line references below refer to **upstream `296b59d2`**, not the working branch of the repository containing this report. Canonical source root:

<https://github.com/Niko1221/Strata/tree/296b59d2f73c8f04a805df29c2fa97655adace78>

## 4. Highest-priority finding: the expert-residency threshold

The live `native_experts.txt` declares 77,017,907,200 bytes, approximately **71.73 GiB**, of routed expert weights.

Today's approximately 7.23-GiB GPU expert cache and 64.5-GiB resident host complement are consistent with covering that model. A short live smoke request reported no expert file reads.

### Historical two-slot evidence from this build

The beginning of the same runtime log records an earlier startup with parallel 2, batch MTP, a 64-GiB resident budget, and explicit PCIe fraction 0.40:

```text
--batch: 2 slot sessions on CUDA0 (0.95 GiB each)
--batch 2: the slot sessions take 1.90 GiB of VRAM ...
--batch-mtp: 2 slot drafters ... 50 MiB ... each
expert cache 1788 slots, 5.20 GiB of VRAM
RAM budget 64.00 GiB ... 66.52 GiB of experts the GPU cache does not hold
```

That startup subsequently recorded file-tier expert reads. It also reported only 491 MiB free VRAM after loading.

These are historical observations, not a controlled A/B against the current configuration. The expert profile, budget, adaptation settings, PCIe split, request mix, and process lifetime differ.

### Interpretation

Enabling parallel 2 with today's 65-GiB budget is likely to leave some expert weights outside both the GPU cache and pinned host complement. The precise deficit must be calculated from the resulting cache allocation and profile, not copied blindly from the historical run.

A file-tier expert read does not necessarily mean a physical SSD read: it may hit the OS page cache. Nevertheless, pinning the full complement avoids reliance on reclaimable cache and can preserve access to the pinned GPU-offload path.

### First experiment

Test parallel 2 with a budget sufficient to hold its actual complement. **67 GiB is a reasonable initial candidate**, subject to measured host headroom and startup accounting.

Acceptance checks:

- Confirm actual cache size and complement allocation.
- Confirm steady decode file-tier reads and physical storage reads separately.
- Check host headroom with both slots active and conversation parking populated.
- Measure prefill separately; a complete decode complement does not imply file-free prefill.

This is a few-GiB adjustment, not duplication of the model.

### Prefill loan-region gap

In `FileExpertSource::pin_cache_complement`, explicit budget mode ranks only the non-GPU complement and sets `lend_from_slot = -1`. The single-GPU prefill loan region is therefore not retained by that path, even if some configured budget remains unused.

When prefill borrows GPU expert-cache slots, their experts may have to come from the file mapping. This also affects decoding interleaved with the prompt while those slots are unavailable.

A bounded improvement would let spare budget retain selected loan-region experts, while maintaining a hard total budget and preserving exchange correctness.

**Source:** `src/core/expert_source.cpp:2634–2790`.

## 5. Hidden default differences between parallel 1 and parallel 2

### 5.1 Routed-expert tail skipping

The investigated CUDA serve path defaults to skipping low-ranked missed experts when no batch slots are configured. With `o.batch != 0`, that default is disabled.

The skip removes contributions without renormalizing the remaining expert weights. It is an accuracy-affecting optimization, not identical arithmetic executed faster.

Implications:

- Parallel 2 can lose speed even with one active request.
- Comparing default settings does not isolate batching overhead.
- Upstream speed claims for this feature are not nibbler measurements.

Benchmark both configurations with `STRATA_ROUTE_TAIL_SKIP=0` to quantify the architectural difference. Also retain the current default configuration as the real operational baseline; making the baseline slower is not itself a performance win.

A source change could preserve the existing solo policy when a parallel-enabled engine is genuinely executing a solo request. Such a change needs graph/state transition and quality testing.

Do not simply force global skipping on for concurrent windows: the existing decision considers all rows in the window, so another request can affect which expert contributions survive. A batch implementation needs an explicit per-sequence policy if independence is required.

**Sources:** `src/program/generate.cpp:8618–8664`; `src/core/expert_source.cpp:3725–3775`.

### 5.2 Runtime PCIe-share refinement

The runtime PCIe-share estimator is gated on `o.batch == 0`. A parallel-enabled engine loses that refinement even when it executes one request alone.

Two active requests can change the number of distinct misses, CPU work, and transfer reuse. Their best CPU/GPU split need not equal the solo split.

Recommended progression:

1. Sweep explicit PCIe fractions in controlled experiments.
2. Tune jointly with CPU worker count.
3. Preserve solo refinement in a parallel-enabled engine.
4. Introduce separate batch estimates instead of blindly mixing solo and batch samples.

**Source:** `src/program/generate.cpp:8669–8705`.

## 6. Batch speculative decoding and synchronization

### Current execution

- Each batch slot drafter is limited to **one proposal**.
- Batch window assembly allocates two rows per active slot: current token plus proposal.
- After verification, batch commit synchronizes the CUDA stream.
- The engine loops through active slots and calls `draft()` for each.
- `draft()` synchronizes its stream before returning.

Distinct drafter streams do not by themselves provide concurrent submission when the host waits for one before launching the next.

### Candidate A: submit drafts before waiting

Split draft execution into enqueue and completion phases. Submit all active slots' draft work before collecting results.

Potential benefit: reduce host serialization and expose overlap. The GPU may still be bandwidth-bound, so this is not a guaranteed throughput increase.

### Candidate B: event-based batch commit

Replace the unconditional commit wait with explicit completion events and overlap independent draft work where legal.

Correctness requirements include:

- No next verification reads uncommitted recurrence or indexer state.
- Draft inputs and PLE history are ready before their consumers.
- Cancellation, slot reuse, admissions, and error paths drain or order in-flight work.
- Shared buffers are not reused prematurely.

### Candidate C: longer per-slot speculation

Test two or three proposals per slot. Two slots with three proposals each consume the existing eight-row verification capacity.

The verifier already has contiguous slot-row groups and per-slot accepted-prefix commits. The scheduler and drafter setup still need generalization; this is not a flag-only change.

Measure accepted output tokens per millisecond. More proposals may increase wasted expert computation and transfers when acceptance is poor.

**Sources:**

- `src/program/generate.cpp:4631–4652, 9950–10110`
- `src/core/mtp.cpp:1516–1675`
- `src/core/verify.cpp:3129–3166`

## 7. Recover grouped operations in batch execution

The solo QSA path uses multi-row operations in places where the batch path loops over rows:

- KV appends.
- Indexer appends.
- Block scores and top-k selection.
- KV resolution and attention.

For batch MTP, adjacent rows belonging to the same slot can potentially use grouped operations without sharing state between conversations.

Batch commit also uses per-layer/per-slot GDN operations, whereas solo commit has a batched GDN commit optimization.

Recommended small patches:

1. Group contiguous same-slot KV and indexer appends.
2. Group same-slot attention work where causal semantics permit.
3. Batch independent operations through slot-pointer tables if profiling justifies the added complexity.
4. Adapt the solo commit batching approach to multiple slot destinations.

Preserve per-slot recurrence, accepted-prefix behavior, causal visibility, and streamed-host KV mirrors. Test both short and long context: kernel-launch overhead and attention traffic will have different weights.

**Source:** `src/core/verify.cpp:1290–1430, 2790–2860`.

## 8. Memory accounting and improvements without unified KV

Parallel 2 currently means **one main session plus two additional slot sessions**. It does not mean two sessions total.

Approximate major allocations per main-model session at 262,144 context / INT8 / 32,768 resident cells:

| Allocation | Approximate size | Notes |
| --- | ---: | --- |
| Authoritative streamed host KV | 3.09 GiB pinned RAM | Full configured context |
| Resident KV | 396 MiB VRAM | 32K INT8 cells across 12 QSA layers |
| Pooled indexer | 384 MiB VRAM | FP32, full configured context |
| RoPE tables | 64 MiB VRAM | Already shared within a session, not across sessions |
| Persistent GDN state | 112 MiB VRAM | Includes recurrent and convolution state |
| Scratch and metadata | Additional | Full session allocator includes execution buffers |

The two extra sessions alone add about **6.19 GiB of pinned host KV**, before draft host KV and other buffers. These figures are allocation-derived estimates, not measured process-private memory deltas.

### 8.1 Share immutable RoPE tables

Share compatible RoPE tables across the main and slot sessions, with explicit lifetime ownership.

Estimated saving at parallel 2: **128 MiB VRAM**. Preserve geometry, context capacity, and scaling compatibility.

### 8.2 Separate persistent slot state from scratch

Slots are carved using the full session allocator, including GDN, QSA, MoE, and block execution buffers. Audit actual consumers and retain only private persistent state where the verifier already owns the required scratch.

Do not assume every buffer is unused: admission, prefill, MTP, restore, and graph-captured pointers must all be traced.

### 8.3 Direct slot-state transfers

`copy_to_slot` and `copy_from_slot` currently use conversation checkpoint/KV save and restore functions, temporary images, and repeated device synchronization.

Introduce dedicated transfers for compatible in-process layouts:

- Device-to-device for GPU-resident state.
- Host-to-host for authoritative streamed KV where possible.
- Explicit handling of destination residency maps and indexer state.
- No serialization of state merely to move it between live sessions.

This targets transition latency and transient allocations, particularly at long context. It does not require shared ownership of conversational KV.

### 8.4 Incremental allocation

Larger independent options:

- Grow each slot's pinned host KV with actual context.
- Grow pooled indexer storage with actual context.
- Allocate unused slots lazily.

Stable virtual addressing or explicit chunk lookup is needed where captured GPU graphs retain pointers. Host registration, resizing, reclamation, and error paths need their own design.

Lazy allocation alone only removes idle overhead. When both slots become active, the memory still has to come from somewhere. Recovering expert-cache capacity after slots become idle requires a safe reclamation mechanism, not just delayed allocation.

**Sources:** `src/core/session.cpp:56–128`; `src/core/layer.cpp:669–852`; `src/program/generate.cpp:9867–9948`.

## 9. Configuration experiments

Run changes individually before combining winners.

| Experiment | Rationale | Risk or tradeoff |
| --- | --- | --- |
| Sufficient resident expert budget, initially test 67 GiB | Avoid partial complement after slot allocations | Host memory pressure; actual need depends on cache allocation |
| `--kv-resident 24576`, then `20480` | Free VRAM for experts without changing precision | More KV transfers at long context |
| English/code draft vocabulary | Smaller draft head and more room for experts | Weaker drafting for excluded languages |
| CPU workers 7, 9, 11 | Balance compute against shared RAM bandwidth | Fewer workers may lose CPU throughput |
| Pool tasks around 24–96 | Improve balance for batch expert jobs | More scheduling overhead |
| Explicit PCIe fraction sweep | Different batch CPU/DMA balance | Interacts with workers and expert hit rate |
| Smaller prefill chunks during contention | Reduce uninterrupted pauses and cache loans | Lower prompt throughput |
| Adaptation cadence and swap count | Reduce bandwidth spent moving experts | Lower hit rate if adaptation becomes too slow |

### Resident KV calculation

Changing 32,768 resident cells to 20,480 saves approximately **148.5 MiB per main-model session**, or **445.5 MiB across the main session and two slot sessions**.

The inspected source enforces a minimum resident-cell count of 20,480. A lower requested number should not be treated as additional savings.

### Draft vocabulary calculation

The historical log records a 106,299-token draft head occupying 275.7 MiB. Using the 40,525-token English/code subset would save roughly **171 MiB** at the same row format, plus a small vocabulary-index difference.

This is an allocation estimate based on that observed head, not a fresh live measurement. Verify the selected vocabulary and resulting head size at restart. Draft weights/head are already shared with slot drafters, so this saving is not multiplied by the number of slots.

### Precision policy

Keep INT8 KV for the first experiments. K8V4 can be evaluated later as an explicit quality/performance tradeoff. It is not an arithmetic-preserving memory optimization.

### Prefill and concurrent decode

While prefill borrows expert-cache slots, interleaved decode sees those experts as unavailable in VRAM. The default batch decode share is 0.5 times the preceding prompt chunk's duration.

Changing chunk size or decode share trades prompt latency against inter-token latency. It cannot make both disappear. Do not use `--no-prefill-borrow` indiscriminately: permanently reserved prefill buffers also reduce expert-cache space.

**Source:** `src/program/generate.cpp:11554–11625`.

## 10. Two tempting switches are not straightforward wins

### 10.1 Exchange rotation

`STRATA_EXCHANGE_ROTATE=1` requires uniform expert blob sizes and a fully mapped/pinned complement.

The live pack has three sizes:

```text
3,072,000 bytes
3,584,000 bytes
3,993,600 bytes
```

Simply enabling the existing switch will fail its uniform-size eligibility check and keep the copy path.

A fresh size-class exchange implementation is a plausible optimization:

- Keep separate spare-buffer ownership by blob size.
- Exchange metadata instead of copying host blobs when sizes match.
- Preserve all residency and in-flight-reader invariants.
- Avoid expanding every expert to the largest size.

Async adaptation hides some waiting but does not remove host memcpy bandwidth. This makes rotation worth considering after measuring adaptation traffic.

**Source:** `src/core/expert_source.cpp:3270–3440`.

### 10.2 Multi-token K-quant CPU kernels

`STRATA_KQ256=1` enables multi-token AVX2 paths for Q4_K gate/up and Q5_1/Q8_0 down. It does not cover every gate/up format in the pack; layer 2 uses Q5_K.

Upstream comments report no clear engine-level win in their tested workload. The current GGML build is already compiled with `-march=native` for the host. Therefore this is not an obvious missing-AVX-512 fix.

The AVX2 kernel's exactness comments refer to its reference arithmetic and build assumptions. Verify against the actual native GGML build rather than assuming those comments prove parity on this deployment.

Before developing new kernels, measure:

- CPU expert time per layer/window.
- Distinct missed experts.
- Tokens sharing each expert.
- Distribution of group widths, especially for unrelated concurrent requests.
- Memory bandwidth and the cost of activation quantization/task dispatch.

A same-prompt concurrency benchmark can exaggerate expert reuse relative to real independent requests.

**Sources:** `src/kernels/cpu/native_expert.cpp:125–270`; `src/kernels/cpu/kq_avx2.cpp`; `CMakeLists.txt:1296–1345`.

## 11. Recommended implementation and experiment order

| Priority | Work | Expected kind of benefit | Status |
| --- | --- | --- | --- |
| 1 | Establish A/B/C baselines with matched policies | Correct attribution | Not run |
| 2 | Keep parallel-2 expert complement resident | Avoid file-tier dependence | Candidate |
| 3 | Tune resident KV, CPU workers, PCIe fraction | Small configuration wins | Candidate |
| 4 | Share RoPE; optimize slot transfers | VRAM and transition overhead | Source work |
| 5 | Enqueue drafts together; event-based commit | Remove serialization | Source work |
| 6 | Group batch QSA/commit operations | Reduce launch/work overhead | Source work |
| 7 | Longer per-slot speculation | More useful output per window | Source work, acceptance-sensitive |
| 8 | Incremental slot allocation or size-class rotation | Memory or host bandwidth | Profile-directed design work |

Preserve the existing learned-profile configuration and async adaptation as initial controls. Do not stack unrelated switches and then attribute the result to one of them.

## 12. Benchmark and correctness contract

### Workloads

Include:

1. One request with parallel 1.
2. One request with parallel 2 configured.
3. Two unrelated concurrent decode-heavy requests.
4. A new long prompt arriving during an existing decode.
5. Repeated turns that exercise prompt/slot/conversation reuse.
6. Short, medium, and long contexts, including the upper range actually used in production.
7. Greedy and representative sampled requests.
8. Languages representative of real use if changing draft vocabulary.

### Controls

- Same binary for config-only comparisons.
- Same initial expert profile, preserved as a fixed fixture for each arm.
- Explicitly distinguish fresh and warm caches.
- Record model, prompt, output-token limits, sampling, and reasoning settings.
- Use matched tail-skip policy for architectural comparisons.
- Retain today's operational default as a separate baseline.
- Interleave or randomize arm order and repeat across restarts.
- Record other host load; nibbler is shared production infrastructure.
- Do not silently reduce supported context to claim equal performance.

### Measurements

- Aggregate useful output tokens per wall-clock second.
- Per-request decode tok/s and inter-token latency distribution.
- Time to first token and total completion latency.
- Prompt throughput and pauses imposed on existing decodes.
- Accepted drafts and useful tokens per verify window.
- CPU expert time, GPU-reach wait, commit, draft/emit, and adaptation wait.
- Expert-cache capacity and routing shares.
- File-tier logical reads separately from physical storage reads.
- KV hit rate and bytes transferred.
- Peak private host memory, pinned memory, file-backed residency, and VRAM.
- Memory after warm-up, both-slot activation, graph capture, parking, cancellation, and reuse.

`STRATA_DECODE_TIMING=1` and `STRATA_VERIFY_PROFILE=1` can identify bottlenecks. GPU event profiling adds overhead; collect final throughput without it.

### Correctness gates

Existing relevant test surfaces include:

- `tools/batch_test.py`
- `tools/batch_interleave_test.py`
- `tools/parking_test.py`
- `tools/early_close_test.py`
- `serve/test_parallel.py`
- `tests/core/batch_rows_test.cpp`
- `src/kernels/kv_stream_parity.cpp`
- Conversation checkpoint, memory, snapshot, and validation tests.
- Native expert parity/bench tools for CPU-kernel changes.

For arithmetic-preserving changes, use fixed-placement/exactness settings where required, including explicitly disabling tail skipping. Check sampled behavior separately. Test admission, restore, cancellation, slot reuse, end-of-context handling, and accepted-prefix correctness.

The investigated batch path does not apply repetition/frequency/presence penalties in the same way as solo execution: `Verifier::sample_rows` sets `penalty_last_n = 0`. Keep that behavioral difference visible when choosing benchmark requests and judging semantic parity.

Build and test all affected backends before proposing a cross-backend patch for adoption. No such builds or tests were performed in this investigation.

## 13. Evidence collection and limitations

### Checks performed

- Homelab agent preflight: all required checks passed.
- `bin/hl peek host nibbler summary`.
- Read-only path probes for launcher, runtime config, shared settings, service unit, build evidence, source samples, model metadata, and the beginning of the engine log.
- Router `GET /running`.
- Strata `GET /v1/status` and `GET /metrics` through the router's upstream route.
- Read-only upstream source archive inspection and selected deployed-source comparisons.

Useful API routes used:

```text
http://nibbler:8088/running
http://nibbler:8088/upstream/qwen3.8-flash-next-nvfp4-gptq/v1/status
http://nibbler:8088/upstream/qwen3.8-flash-next-nvfp4-gptq/metrics
```

The router identifier above is historical; the upstream API identified the loaded Q4 model.

### Limitations

- The sanctioned path-read probe returns only the first 64 KiB of large files and offers no offset/tail option.
- The inspected historical log prefix is not the current process's startup tail.
- The available host probes did not provide process-private/pinned memory accounting.
- API RAM figures are system-wide, not engine-private memory.
- No under-load PCIe link/bandwidth, memory-bandwidth, GPU-stage, or CPU-pool profiling was performed.
- No complete deployed-source or current executable rehash was performed.
- The exact GGML dependency was identified from stored build evidence, not fully audited.
- No fresh performance or correctness tests were run.
- No production changes were made.

Before benchmarking, add or authorize narrowly scoped read-only process-memory and log-tail inspection. Do not silently bypass the host inspection policy with ad hoc SSH or process execution.

### Temporary local investigation artifacts

These may be removed by normal temporary-directory cleanup; they are not the durable report:

```text
/tmp/strata-audit-296b59d2/
/tmp/strata-current-metrics.json
/tmp/strata-main296-engine.log
/tmp/strata-model-native-probe.json
```

## Conclusion

There is no need to begin by doubling model memory or adopting a unified-KV architecture.

The most concrete first steps are:

1. Avoid falling off the resident-expert threshold when enabling two slots.
2. Separate true batching overhead from the optimization policies disabled by batch configuration.
3. Remove sequential draft waits and unnecessary batch synchronization.
4. Recover grouped operations and modest immutable/scratch-memory savings.
5. Use profiles to decide whether longer speculation, incremental allocation, or size-class exchange rotation should follow.

These are opportunities, not yet demonstrated gains. A controlled A/B/C baseline and direct private-memory measurements are the next prerequisites for deciding which ones pay on nibbler.
