# Nibbler Tier-A Benchmark — Session Record

**Host:** paseo-nibbler (Debian libvirt VM on nibbler)  
**Date:** 2026-10-07 / 2026-10-08  
**Scope:** Single-session CLI fresh-process triage.  Single GPU, no unified KV,
no handoff, no vision, no concurrency/batching, no steady-state decode.

---

## Contents

1. [Purpose](#1-purpose)
2. [Hardware and software state](#2-hardware-and-software-state)
3. [Binaries under test](#3-binaries-under-test)
4. [Build provenance](#4-build-provenance)
5. [Pre-flight parity gates](#5-pre-flight-parity-gates)
6. [Harness methodology](#6-harness-methodology)
7. [Prompt corpus](#7-prompt-corpus)
8. [Smoke run: 1 rep, 32/64 tokens (completed)](#8-smoke-run-1-rep-3264-tokens-completed)
9. [Full matrix: 3 rep, 256/512 tokens (pending)](#9-full-matrix-3-rep-256512-tokens-pending)
10. [Eligibility notes per comparison](#10-eligibility-notes-per-comparison)
11. [Interpretation guide](#11-interpretation-guide)
12. [Raw artifact paths](#12-raw-artifact-paths)

---

## 1. Purpose

Tier-A measures whether a candidate build of the Strata engine produces
materially different numbers from the pinned production build for a specific set
of comparisons, on a specific machine, using the CLI fresh-process mode.

The comparisons are:

| Group | Signal | Primary metric |
|-------|--------|----------------|
| G1–G3 | baseline exe vs candidate exe (short code / long doc / story) | decode_tok_s (cold-graph triage); prefill_tok_s |
| G4 | `--prefill auto` vs `--prefill 4096`, candidate, long doc | prefill_tok_s |
| G5 | `--adapt-every 0/1/2`, candidate, story | decode_tok_s |
| G6 | `STRATA_GR_DOWN_MAX4` off vs on, candidate, story | decode_tok_s (likely no-op confirmation) |
| G7 | `STRATA_IQ256_GATHER` — **not applicable on this host** (see §10) | skipped |

---

## 2. Hardware and software state

Captured: `2026-10-07T11:45:37-04:00`

```
CPU:    AMD Ryzen 9 9900X 12-Core Processor
        24 logical CPUs, 1 socket, 12 cores/socket
        ISA flags: AVX2, AVX-512F/DQ/BW/VL/VBMI/VBMI2/IFMA/VNNI/BF16,
                   AVX_VNNI, SHA-NI, GFNI, VAES, VPCLMULQDQ
RAM:    ~123 GiB physical, ~95 GiB available at bench time
GPU:    NVIDIA GeForce RTX 5060 Ti, 16311 MiB VRAM
        driver 610.57.04, pstate P8 at bench start (2 MiB used, 3.91 W)
OS:     Debian (libvirt VM on nibbler bare-metal host)
        kernel 7.2.8+deb14-amd64
```

**Inference service state during benchmarking:**  
`/opt/native-inference/bin/start-strata-sharp-medium` was **mode 644**
(not executable — intentional launcher guard, no service loaded).  
`llama-swap.service` active/running; `wyoming-openai.service` active/running.  
llama-swap health: `{"status":"ok","backend":"cpu","models":2}` — CPU-only,
no GPU contention from inference service during bench.

---

## 3. Binaries under test

| Role | Path | SHA-256 |
|------|------|---------|
| **baseline** | `/data/llm/Strata-tests/tier-a-20261007/production.strata` | `40e8305c8e09e657e22f520255185bee29faaa1`… (see §12) |
| **candidate** | `/data/llm/Strata-tier-a/build-tier-a/strata` | `47a999a8ac7505952553b5f575618340cd8b189`… |

Both binaries pinned before any benchmark arm ran.
Full hashes in [`build-and-snapshot.sha256`](#12-raw-artifact-paths).

**Source lineage:**

- Baseline: built from commit `9921eeca` (upstream ggml) / Strata production tree,
  snapshotted to `/data/llm/Strata-tests/tier-a-20261007/`
- Candidate: built from `strata-tier-a` working tree, commit `60a081a`
  (`Add isolated nibbler Tier A benchmark harness and measurement gates`)
- The two trees share **no common git history** (independent forks/snapshots).
  This is a whole-pinned-tree comparison, not an isolated kernel port.

---

## 4. Build provenance

Both binaries built with the same CMake configuration:

```
Build type:     Release
CXX compiler:   g++-15 (GNU 15.3.0)
C compiler:     gcc-15 (GNU 15.3.0)
CUDA compiler:  nvcc 13.3.73 (CUDA Toolkit 13.3)
                host compiler: GNU 15.3.0
CUDA arch:      120 (RTX 5060 Ti / Blackwell sm_120)
GGML_CUDA:      OFF  (Strata uses its own CUDA engine, not ggml-cuda)
GGML_CUDA_FA:   ON
STRATA_NATIVE_EXPERTS: ON
ggml version:   0.24.0  (commit 9921eeca)
Strata version: 0.1.39
```

**Build result:** 145 objects, 0 errors.  
**Warnings:** 20 in candidate build (`-Wformat` `%llu`/`%lu` on `uint64_t` fields
in `weights.cpp` under gcc-15 — pre-existing; not related to the changes under
test). Full build log at [`build-gcc15.log`](#12-raw-artifact-paths).

**Also snapshotted** (hashes in `build-and-snapshot.sha256`):
- `production.config.json`
- `production.server.py`
- `production.launcher`
- `third_party/llama.cpp/ggml/src/ggml-quants.c`
- `third_party/llama.cpp/ggml/include/ggml.h`

---

## 5. Pre-flight parity gates

All gates run against the candidate build before any live arm.

### 5.1 Targeted parity suite (18 tests)

Run: `ctest` from `build-tier-a`, selecting targeted kernel tests.  
Result: **18/18 passed** in 10.73 s.

| # | Test | Result |
|---|------|--------|
| 28 | gr_parity | PASS 0.46 s |
| 29 | gdn_parity | PASS 0.49 s |
| 31 | sampler_parity | PASS 0.92 s |
| 33 | sampler_parity_one_block | PASS 1.25 s |
| 34 | sampler_parity_old | PASS 3.15 s |
| 75 | iq_avx2_parity | PASS 0.12 s |
| 76 | iq_avx2_parity_iq3s_mt1 | PASS 0.12 s |
| 81 | native_expert_parity_refuses_q6_K | PASS 0.24 s |
| 82–91 | native_expert_parity (8 quant combos) | PASS 0.32–0.52 s each |

Full log: [`targeted-gates.log`](#12-raw-artifact-paths).

### 5.2 GR/fused-GR parity (`gr_parity`)

Exercised: mixed vs reference, inject vs reference, FP32 activations, real dims
2560/4/320, fused GR read + q8_1 (STRATA_QFUSE), multi max-T=8 LDS launch,
CUDA graph capture and replay, scalar gamma BF16/FP32/native-pinned variants.

```
mixed vs reference:   worst 3.594e-08  (mean |ref| 0.2105)
inject vs reference:  rel   2.845e-08
FP32 activations:     pass  (mixed 7.823e-08, inject 1.138e-07)
real dims (BF16):     pass  (mixed 7.056e-08, inject 1.986e-07)
real dims (FP32):     pass  (mixed 1.620e-07, inject 2.817e-07)
native pinned GR:     pass  (mixed 1.669e-07, inject 1.359e-07)
gr_read/gr_write:     0 failures
```

### 5.3 IQ AVX2 parity — STRATA_IQ256_GATHER=0 (default)

CPU: AMD Ryzen 9 9900X (avx2=1, avx512=1).  
Variants this CPU runs at gather=0: `gather+vnni`; this thread's variant: `vnni`.

All quant types (iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_xs, iq4_nl, q2_0),
1–8 tokens, all kernel variants: **0 failures**.

Note: `iq2_xxs/xs/s/iq3_xxs/iq3_s/iq4_xs` — ggml dot below 2 tokens rounds
differently from the engine (#152); this is a known pre-existing divergence in
the reference, not a candidate regression.

### 5.4 IQ AVX2 parity — STRATA_IQ256_GATHER=1 (gather forced on)

Same CPU. Variants with gather=1: `gather+vnni`; this thread's: `gather+vnni`.  
All quant types, 1–8 tokens, all variants: **0 failures**.

Context: on the 9900X, IQ3_S decode in production routes through
`native_gu_rows` (AVX-512) before `cpu_gather_fast` is reached (see §10).
This parity test exercises the standalone gather path in isolation;
it passes cleanly even though the gather path is not active in end-to-end runs
on this host.

Full logs: [`iq-gather-off-parity.log`](#12-raw-artifact-paths),
[`iq-gather-on-parity.log`](#12-raw-artifact-paths).

### 5.5 Harness offline unit tests

79/79 tests pass (bare Python 3.11 + regex from uv cache).  
Covers: `--tokens-file` flag, mkstemp cleanup, FIXED_FLAGS, counterbalancing,
summarise_group eligibility, decode header validation, prompt echo validation,
performance_eligible field, fail-fast, dry-run artifacts.

---

## 6. Harness methodology

**Mode:** CLI fresh-process. Each arm launches a fresh OS process:

```
/path/to/strata \
  --pack /data/llm/Strata-data/packs/iq3_s \
  --native <GGUF shard 1> --ple-gguf <GGUF shard 2> \
  --expert-profile <profile.bin> \
  --mtp /data/llm/Strata-data/mtp/rt \
  --expert-cache 2000 \
  --pcie-frac 0 \
  --max-context 262144 --kv-resident 32768 --kv int8 \
  --spec 4 --spec-min-p 0.5 \
  --vram-reserve-mib 2048 \
  --stats --stop-eos \
  --adapt-every 0 \
  --prefill auto \
  --tokens-file <tmp>.ids \
  --max-new <N>
```

**Env (both arms):** `STRATA_PF_FUSED=1  STRATA_IQ_MT_MIN=1`  
**Controlled vars** (stripped from ambient env, re-injected per arm only):
`STRATA_GR_DOWN_MAX4`, `STRATA_IQ256_GATHER`, `STRATA_NO_AVXVNNI`,
`STRATA_IQ_MT_MIN`, `STRATA_PF_FUSED`, `STRATA_PF_FUSED_NATIVE`,
`STRATA_NO_MULTI_GR`

**Expert cache:** 2000 slots — matches the production setup that showed
2582 resident slots in the upstream smoke run.

**CUDA graph cold-graph caveat (read before comparing decode numbers):**

> Every `decode_tok_s` figure in this harness includes CUDA graph compilation
> for the first occurrence of each verify-window size T encountered in that
> process. This is NOT steady-state throughput. Even matched token budgets do
> not guarantee equal graph-capture cost between arms: capture cost depends on
> the actual T distribution encountered, which can differ if outputs diverge.
> Verify `prompt_hash` equality before comparing decode numbers across arms.
> `prefill_tok_s` is unaffected (prefill uses a separate batched kernel, not
> captured into the decode graph).

**Performance eligibility:**  
Arms where `max_new ≥ 128` and `n_output < 200` are flagged
`performance_eligible=false` with reason `short_output`. Their raw data is
preserved; they are excluded from group medians. Short-output is not a
harness error and does not trigger `--fail-fast`.

**Counterbalancing:** within each group the baseline/candidate (or variant)
ordering is rotated across reps to reduce systematic first-process advantages.

**Fail-fast:** `--fail-fast` aborts the matrix on the first real arm error
(non-zero exit, timeout, launch failure, parse error). Short-output does not
trigger it.

**Per-arm validation:**
- Decode header count (`decode N tokens`) must equal `len(output_ids)`
- Engine's `prompt  :` echo must match the supplied token IDs
- Temp file created with `mkstemp`, closed before child opens it, unlinked in
  `finally` on every path (success, error, timeout, dry_run)

---

## 7. Prompt corpus

| Key | Description | Tokens (real tokeniser) |
|-----|-------------|------------------------|
| `short_code` | Short Python/C++ algorithm snippet with Q&A wrapper | 42 |
| `long_doc` | Multi-section algorithm survey (Timsort, Introsort, Heapsort, Radix…) with appendices | **8887** |
| `story` | Narrative creative-writing prompt | 80 |

`long_doc` encodes to 8887 tokens — verified by the real tokeniser during the
smoke run. The harness asserts ≥ 8192 before any live arm starts.

Long prompt prefill exercises 8886 tokens of KV (8887 − 1 context token);
this is the primary signal for G2 and G4.

---

## 8. Smoke run: 1 rep, 32/64 tokens (completed)

Run: `2026-10-07` · 1 rep · max_new=32 (short), max_new=64 (long/story) ·
Groups 1–3 only · elapsed 121 s · **6/6 arms clean, 0 errors**.

All six arms returned `performance_eligible=true` (smoke budgets are below the
128-token threshold, so short-output eligibility rules do not apply).

Prompt hashes confirmed identical across baseline/candidate within each group.

### G1 — short code (42 tokens prompt, max_new=32)

| arm | decode_tok_s | prefill_tok_s | draft_rate | output_hash |
|-----|-------------|---------------|------------|-------------|
| baseline  | 47.33 | 68.2  | 0.880 (22/25) | `3ce6c2acab68091e` |
| candidate | 47.22 | 69.6  | 0.885 (23/26) | `3ce6c2acab68091e` |

Output hashes **identical**. Cold-graph decode numbers within noise (1 rep,
no median). Prefill within noise.

### G2 — long doc (8887 tokens prompt, max_new=64)

| arm | decode_tok_s | prefill_tok_s | ttft_ms | draft_rate | output_hash |
|-----|-------------|---------------|---------|------------|-------------|
| baseline  | 38.96 | 1473.8 | 6564.7 | 0.685 (37/54) | `997bd5d005adb31e` |
| candidate | 39.69 | 1476.7 | 6531.4 | 0.684 (39/57) | `997bd5d005adb31e` |

Output hashes **identical**. Prefill ~1474–1477 tok/s (1 rep each;
difference < noise). TTFT dominated by 8887-token prefill (~6.5 s).

### G3 — story (80 tokens prompt, max_new=64)

| arm | decode_tok_s | prefill_tok_s | draft_rate | output_hash |
|-----|-------------|---------------|------------|-------------|
| baseline  | 31.23 | 103.3 | 0.529 (27/51) | `bde23c3fa23955c3` |
| candidate | 33.61 | 106.2 | 0.529 (27/51) | `bde23c3fa23955c3` |

Output hashes **identical**. Story prompt has lower draft acceptance (0.529)
than short code (0.880) — expected: creative text is less predictable. Decode
difference visible but within cold-graph variability at n=1.

**Summary (smoke, cold-graph triage — 1 rep, not median):**

> Output hashes match on all 3 prompt types. Prefill and decode numbers are in
> the same range for both binaries. No regressions visible at smoke budget.
> Full-run repeated measurements needed before drawing conclusions on decode
> speed; prefill numbers are more reliable even at n=1.

---

## 9. Full matrix: 3 rep, 256/512 tokens (pending)

**Status:** Matrix launched with `--fail-fast` at time of writing.
Full artifacts will be copied to `/data/llm/Strata-tests/tier-a-20261007/`
when the run completes. This section will be filled in on parent follow-up.

Expected arms: 35 live + 1 skip (G7).  
Full-run budgets: max_new=256 (short code), max_new=512 (long doc / story).  
Short-output threshold: n_output < 200 with budget ≥ 128 → `performance_eligible=false`.

Parent note: candidate short-code at 256 budget returned 256 tokens (full
budget used). Story at 512 budget returned 321 tokens (EOS at 321 — natural).
Both clean; no prompt rewrite needed.

### Placeholder: group summaries

*(To be filled when full matrix artifacts arrive.)*

| Group | Variants | Primary metric | n | Result |
|-------|----------|----------------|---|--------|
| G1 short_code | baseline / candidate | decode_tok_s median | 3 rep | **PENDING** |
| G2 long_doc   | baseline / candidate | prefill_tok_s median | 3 rep | **PENDING** |
| G3 story      | baseline / candidate | decode_tok_s median | 3 rep | **PENDING** |
| G4 prefill    | auto / fixed4096     | prefill_tok_s median | 3 rep | **PENDING** |
| G5 adapt      | off / every1 / every2 | decode_tok_s median | 3 rep | **PENDING** |
| G6 GR_DOWN_MAX4 | off / on           | decode_tok_s (1 pair) | 1 rep | **PENDING** |
| G7 IQ256_GATHER | not_applicable     | — | — | SKIPPED (see §10) |

---

## 10. Eligibility notes per comparison

### IQ256_GATHER — not applicable on this host

The AMD Ryzen 9 9900X has AVX-512. `native_expert.cpp` routes IQ3_S through
`native_gu_rows` (AVX-512) before the `cpu_gather_fast` path is reached.
`cpu_gather_fast` probes Intel AVX-VNNI only — the gather control knob has
no effect on this host in end-to-end runs.

The standalone `iq_avx2_parity` test (§5.3–5.4) does exercise the gather path
directly and passes with 0 failures at both STRATA_IQ256_GATHER=0 and =1.
That confirms the gather path is numerically correct in isolation on AMD;
it is simply not the active path for IQ3_S in production on the 9900X.

G7 is recorded as `variant=not_applicable` and skipped.

### GR_DOWN_MAX4 — likely no-op (one pair recorded for confirmation)

`fused_gr.cu` lines 884–910: the `max4` branch guards `ct<=4 &&
gr_down_max4()`. With `STRATA_NO_MULTI_GR` unset (default), exact-T
specialised kernels (`ct==1..6`) are selected first for T=1..6, covering
all single-token and small-multi-token decode. The `max4` generic path
is unreachable in normal decode. G6 runs one off/on pair to confirm the
reading is consistent with this analysis.

---

## 11. Interpretation guide

**What these numbers are:**  
Cold-graph triage. Every `decode_tok_s` includes CUDA graph compilation
overhead. Useful for catching regressions (a large unexplained drop) and
confirming output hash parity. Not suitable for absolute throughput claims.

**What they are not:**  
Steady-state throughput. Steady-state requires a persistent `--serve` harness
with per-arm warmup generations. That is out of scope for this session.

**prefill_tok_s is the more reliable metric in this harness** because the
prefill path uses a separate batched kernel not captured into the decode graph.
For G2 (long doc, 8887 tokens) and G4 (prefill mode comparison),
`prefill_tok_s` is the primary signal.

**Comparing across arms:**  
Always check `prompt_hash` equality first. Two arms with the same budget but
different output text may hit different CUDA graph T-values in different orders,
making their cold-graph decode numbers non-comparable. Arms within the same
group in this harness are given the same tokenised prompt (same `prompt_hash`).

**Do not claim speed gains before repeated evidence.**  
Smoke run numbers (n=1) are indicative only. Full-run medians (n=3 reps) are
the intended evidence base. Even then, note the cold-graph caveat above.

---

## 12. Raw artifact paths

All artifacts pinned to `/data/llm/Strata-tests/tier-a-20261007/` on nibbler.
A working copy of the smoke artifacts was also present at
`/tmp/strata-tier-a-artifacts/` on paseo-nibbler at bench time.

| File | Contents |
|------|----------|
| `build-and-snapshot.sha256` | SHA-256 of both binaries + 5 source snapshots |
| `build-gcc15.log` | Full candidate build log (145 objects, 20 format warnings) |
| `configure-gcc15.log` | CMake configure output |
| `production.CMakeCache.txt` | Full CMakeCache for both builds |
| `targeted-gates.log` | 18-test parity suite (ctest output) |
| `iq-gather-off-parity.log` | IQ AVX2 parity at STRATA_IQ256_GATHER=0 |
| `iq-gather-on-parity.log` | IQ AVX2 parity at STRATA_IQ256_GATHER=1 |
| `gr-max4-parity.log` | GR/fused-GR parity (gr_parity test detail) |
| `host.bench-state.txt` | CPU/GPU/memory/service state at bench time |
| `smoke/tier_a_results.json` | Full smoke JSON (6 arm records) |
| `smoke/arms/<label>/` | Per-arm: `argv.txt`, `stdout.txt`, `stderr.txt`, `record.json`, `prompt_ids.txt`, `output_ids.txt` |
| `smoke.log` | Harness stdout for the smoke run |
| *(full matrix)* | `full/tier_a_results.json`, `full/arms/<label>/` — **pending** |

---

*This document records measurements taken; it does not constitute a release
decision. Conclusions about the full matrix are reserved for the parent after
reviewing the complete artifact set.*
