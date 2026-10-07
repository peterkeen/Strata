# Nibbler Tier-A Benchmark — Session Record

**Bench host:** nibbler (bare-metal AMD 9900X — live GPU harness ran here)
**Orchestration VM:** paseo-nibbler (Debian libvirt VM on nibbler — offline unit tests and harness orchestration ran here)
**Date:** 2026-10-07
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
9. [Full matrix: 3 rep, 256/512 tokens (completed)](#9-full-matrix-3-rep-256512-tokens-completed)
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
OS:     nibbler bare-metal — kernel 7.2.8+deb14-amd64 (captured via SSH uname)
        paseo-nibbler orchestration VM: Debian libvirt
```

**Inference service state during benchmarking:**
`/opt/native-inference/bin/start-strata-sharp-medium` was **mode 644**
(not executable — intentional launcher guard, no service loaded).
`llama-swap.service` active/running (port 8088); `wyoming-openai.service` active/running.
llama-swap health endpoint returned `OK` (exact plain-text body) — router up.
Audio stack on port 5800: `audiocpp_server` (PocketTTS + Parakeet) — not a
whisper or wyoming server; not the LLM inference path.
`wyoming-openai.service` active on port 10301 (separate from the audio stack on 5800).
Qwen model intentionally unloaded; no LLM GPU contention during bench.

---

## 3. Binaries under test

| Role | Path | SHA-256 |
|------|------|---------|
| **baseline** | `/data/llm/Strata-tests/tier-a-20261007/production.strata` | `40e8305c8e09e657e22f520255185bee29faaa1`… (see §12) |
| **candidate** | `/data/llm/Strata-tier-a/build-tier-a/strata` | `47a999a8ac7505952553b5f575618340cd8b189`… |

Both binaries pinned before any benchmark arm ran.
Full hashes in [`build-and-snapshot.sha256`](#12-raw-artifact-paths).

**Source lineage:**

- Baseline engine: Strata source commit `9921eecae9d442878de5bfe2b2cd25a71fbdb4ed`,
  version **0.1.39**. Binary snapshotted to
  `/data/llm/Strata-tests/tier-a-20261007/production.strata`.
- Candidate engine: `strata-tier-a` source commit
  `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a`, version **0.1.40.2**.
  Built to `/data/llm/Strata-tier-a/build-tier-a/strata`.
- Harness code: commit `60a081a` (`Add isolated nibbler Tier A benchmark
  harness and measurement gates`); adapt-replay additions at commit `fb1cc15`
  (`run_adapt_replay.py` + 131-test suite) — both are measurement tooling
  commits, not compiled engines.
- `third_party/llama.cpp` in both trees has no embedded `.git`; the git ref
  `9921eeca` is a Strata parent commit, not a ggml commit. ggml version
  reported as 0.24.0 by CMake configure; no ggml git ref is attributed.
- The two engine trees share **no common git history** (independent
  forks/snapshots). This is a whole-pinned-tree comparison, not an isolated
  kernel port. The saved local ref `refs/remotes/fork/main` points to
  `13be4c37fdd979097f58285c533d38030bed0cb3` (no GitHub verification performed).

---

## 4. Build provenance

Both binaries built with the same selected toolchain:

```
Build type:     Release
CXX compiler:   g++-15 (GNU 15.3.0)
C compiler:     gcc-15 (GNU 15.3.0)
CUDA compiler:  nvcc 13.3.73 (CUDA Toolkit 13.3)
                host compiler: GNU 15.3.0
CUDA arch:      120 (RTX 5060 Ti / Blackwell sm_120)
```

Candidate-specific CMake settings (from `configure-gcc15.log` /
`build-gcc15.log`; candidate `build-tier-a` directory):
```
GGML_CUDA:             OFF  (Strata uses its own CUDA engine)
GGML_CUDA_FA:          ON
STRATA_NATIVE_EXPERTS: ON
ggml version:          0.24.0  (reported by CMake; no ggml git ref attributed)
Strata version:        0.1.40.2
```

Baseline CMake settings recorded in `production.CMakeCache.txt` (baseline
build only; not the candidate cache). Baseline Strata version: **0.1.39**.
The two CMakeCache files are separate; only the baseline cache was snapshotted.

**Candidate build result:** 145 objects, 0 errors.
**Warnings:** 20 in candidate build — pre-existing style/portability warnings,
none related to the changes under test. Breakdown by `grep` of
`build-gcc15.log`:

- 9 `-Wformat` on `src/core/weights.cpp:168` (`%llu`/`%lld` vs `uint64_t`/`int64_t` under gcc-15).
- 5 `-Wmisleading-indentation` in `src/core/layer.cpp`.
- 6 `-Wunused-variable` / `-Wunused-but-set-variable` in `src/core/layer.cpp`,
  `src/core/verify.cpp`, `src/program/generate.cpp`, `src/kernels/gdn_parity.cpp`.

Full build log at [`build-gcc15.log`](#12-raw-artifact-paths).

**Also snapshotted** (hashes in `build-and-snapshot.sha256`):
- `production.config.json`
- `production.server.py`
- `production.launcher`
- `third_party/llama.cpp/ggml/src/ggml-quants.c`
- `third_party/llama.cpp/ggml/include/ggml.h`

All original source/binary/config/server/launcher content hashes confirmed
unchanged at time of harness run.

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
| 82–91 | native_expert_parity (10 quant combos) | PASS 0.24–0.52 s each |

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

### 5.3 IQ AVX2 parity — STRATA_IQ256_GATHER=0 (gather forced off)

CPU: AMD Ryzen 9 9900X (avx2=1, avx512=1).
Variants this CPU runs with gather=0: `gather+vnni`; this thread's variant: `vnni`.

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

Full logs: [`iq-gather-off-parity.log`](#12-raw-artifact-paths) (gather=0),
[`iq-gather-on-parity.log`](#12-raw-artifact-paths) (gather=1).

### 5.5 Harness offline unit tests

131/131 tests pass, no skips.
Run on nibbler host (existing `/data/llm/Strata/.venv`) and on paseo-nibbler VM
(system Python 3.11 + regex module from uv cache); both environments pass.
Covers: `--tokens-file` flag, mkstemp cleanup, FIXED_FLAGS, counterbalancing,
summarise_group eligibility, decode header validation, prompt echo validation,
performance_eligible field, fail-fast, dry-run artifacts; adapt-replay
reference loading, canonical snapshot isolation, `--spec-follow` path
provenance, argmax disagreement parsing, arm construction, output mismatch
eligibility, dry-run end-to-end, invalid reference rejection.

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
| `short_code` | "Write a Python function `merge_sorted(a, b)` that merges two sorted lists into one sorted list. Include a docstring and two test cases." | 42 |
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

Smoke budgets (max_new=32/64) are below the 128-token full-run threshold;
short-output eligibility rules do not apply. The older smoke harness JSON
does not carry a `performance_eligible` field; eligibility is derived here
from the budget/output counts, not read from the recorded artifact.

Prompt hashes confirmed identical across baseline/candidate within each group.

### G1 — short code (42 tokens prompt, max_new=32)

| arm | decode_tok_s | prefill_tok_s | draft_rate | output_hash |
|-----|-------------|---------------|------------|-------------|
| baseline  | 47.33 | 68.2  | 0.880 (22/25) | `3ce6c2acab68091e` |
| candidate | 47.22 | 69.6  | 0.885 (23/26) | `3ce6c2acab68091e` |

Output hashes **identical**. Cold-graph decode numbers are descriptive only (1 rep,
no median); no statistical significance established. Prefill likewise descriptive.

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
than short code (0.880). Decode difference visible; descriptive only (n=1,
no statistical significance established).

**Summary (smoke, cold-graph triage — 1 rep, not median):**

> Output hashes match on all 3 prompt types. Prefill and decode numbers are in
> the same range for both binaries. No regressions visible at smoke budget.
> Full-run repeated measurements needed before drawing conclusions on decode
> speed; prefill numbers are more reliable even at n=1.

---

## 9. Full matrix: 3 rep, 256/512 tokens (completed)

**Status:** Completed. 35 eligible arms (1 skip: G7), elapsed **859 s**,
**0 errors, 0 ineligible arms**.
Artifacts at `/data/llm/Strata-tests/tier-a-20261007/full/` on nibbler.

Full-run budgets: max_new=256 (short code), max_new=512 (long doc / story).

To re-run from scratch on nibbler (use a fresh `--out` dir to avoid clobbering):

```
cd /data/llm/Strata-tier-a && \
/data/llm/Strata/.venv/bin/python tools/tier_a/run_matrix.py \
  --out /data/llm/Strata-tests/tier-a-20261007/full-rerun \
  --fail-fast
```

### G1 — short code (42 tokens prompt, max_new=256, n=3 reps)

Primary metric: `decode_tok_s` (cold-graph triage; see §6 and §11).

| variant | decode_tok_s (median) | prefill_tok_s (median) |
|---------|-----------------------|------------------------|
| baseline  | 46.91 | — |
| candidate | 46.41 | — |
| **Δ candidate − baseline** | **−1.1%** | |

Output hashes identical within each rep. Short-code prefill (42 tokens) is
not the primary signal. The −1.1% decode difference is descriptive; this
three-repetition cold-process probe does not establish statistical significance
or a performance regression.

### G2 — long doc (8887 tokens prompt, max_new=512, n=3 reps)

Primary metric: `prefill_tok_s` (most reliable in cold-graph mode; see §6 and §11).

| variant | decode_tok_s (median) | prefill_tok_s (median) |
|---------|-----------------------|------------------------|
| baseline  | 41.63 | 1471.13 |
| candidate | 42.98 | 1475.97 |
| **Δ candidate − baseline** | **+3.2%** | **+0.3%** |

The measured prefill difference is +0.3% (1471→1476 tok/s); statistical
significance is not established. Cold-graph decode +3.2% for candidate.
Output hashes identical within each rep.

### G3 — story (80 tokens prompt, max_new=512, n=3 reps)

Primary metric: `decode_tok_s`.

| variant | decode_tok_s (median) | prefill_tok_s (median) |
|---------|-----------------------|------------------------|
| baseline  | 34.97 | — |
| candidate | 36.17 | — |
| **Δ candidate − baseline** | **+3.4%** | |

Cold-graph decode +3.4% for candidate. Output hashes identical within each rep.

### G4 — prefill mode (long doc, candidate, max_new=512, n=3 reps)

Primary metric: `prefill_tok_s`.

| variant | prefill_tok_s (median) |
|---------|------------------------|
| `--prefill auto`  | 1474.17 |
| `--prefill 4096`  | 1410.33 |
| **Δ fixed4096 − auto** | **−4.3%** |

`--prefill auto` produces ~4.5% higher prefill throughput than fixed 4096
(1474 vs 1410 tok/s, (auto−fixed)/fixed); equivalently, fixed 4096 is ~4.3%
lower than auto ((fixed−auto)/auto). Both arms use the candidate binary.

### G5 — adapt-every: forced-continuation replay (story prompt, n=3 reps)

**Natural free-generation results (full matrix, story prompt, G5 arms, n=3 reps):**
Output lengths and hashes diverge by adapt variant; direct throughput comparison
is not meaningful without controlling the output sequence.

| adapt variant | decode_tok_s (median) | n_output | output_hash |
|---------------|-----------------------|----------|-------------|
| off (every=0) | 36.33 | 321 | `c44b95579fbfdc3a` |
| every1 (swaps=80) | 50.12 | 314 | `d7ea25e026394580` |
| every2 (swaps=80) | 50.40 | 303 | `3d5c59e7ff2b2945` |

Because output sequences differ, the above values are not a controlled
comparison; the forced-replay below controls for output sequence.

**Method:** adapt-replay harness (`run_adapt_replay.py`), commit `fb1cc15`.
All 9 arms forced to emit the same 321 output token IDs via `--spec-follow`.
Reference hash: `c44b95579fbfdc3a`; prompt hash: `06a101a92d88121e`.
9 arms: adapt-every off/every1/every2 × 3 reps, G5 rotation.
Elapsed **193 s**, 0 errors, all 9 arms performance-eligible.
All arms verified to emit exact reference IDs (parent-confirmed).
Canonical argv paths verified for all 9 arms.

To re-run (use a fresh `--out` dir; reference from full run output):

```
cd /data/llm/Strata-tier-a && \
/data/llm/Strata/.venv/bin/python tools/tier_a/run_adapt_replay.py \
  --reference /data/llm/Strata-tests/tier-a-20261007/full/arms/g5_rep0_off/output_ids.txt \
  --out /data/llm/Strata-tests/tier-a-20261007/adapt-replay-rerun \
  --fail-fast
```

| adapt variant | decode_tok_s (median, n=3) | argmax differ (median, of 321) |
|---------------|---------------------------|--------------------------------|
| off (every=0) | 34.63 | 0 |
| every1 (swaps=80) | 50.06 | 6 |
| every2 (swaps=80) | 49.90 | 4 |
| **Δ every1 vs off** | **+44.5%** | |

Argmax-differ values were stable across all 3 reps for each variant.

**Critical caveats for G5:**

- **Forced continuation, not free generation.** All arms emit the same 321
  IDs; this measures processing cost under a controlled output sequence.
  Output quality parity is **not** established by this method.
- **Equal output sequence does NOT guarantee equal verify-window T
  distribution or graph-capture cost** across arms. The +44.5% is a
  throughput difference (decode_tok_s is higher for adapt variants) under
  forced continuation only; it is not a free-generation or quality-parity result.
- **`follow_differ` is an argmax disagreement count** (positions where the
  natural argmax differed from the forced continuation). It is **not** a
  direct expert routing metric.
- **Production default is `--adapt-every 4 --adapt-swaps 96`**, which was
  not tested in this session. `every1` and `every2` (both at 80 swaps) have
  not been compared against the actual production setting. Do not promote
  either over existing production on the basis of these numbers.

### G6 — GR_DOWN_MAX4 (story prompt, candidate, 1 pair)

The saved stderr reports selection of the staged hyper-connection path.
With STRATA_NO_MULTI_GR unset, the source dispatch in
`src/kernels/cuda/fused_gr.cu` (lines 887–898) chooses exact-T staged
kernels for ct=1..6 before the generic max4 branch. Therefore no
kernel-dispatch change is expected from STRATA_GR_DOWN_MAX4 for this
selected path. This combines a startup diagnostic with source-based
inference; it is not a runtime trace of every kernel launch.
off=36.79 tok/s, on=35.62 tok/s; n=1 per arm; identical output hash
`c44b95579fbfdc3a`; descriptive only, not statistical proof.

### G7 — IQ256_GATHER (skipped)

`variant=not_applicable` — see §10. Not run.

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

### GR_DOWN_MAX4 — expected no-op in selected staged path (dispatch analysis)

`fused_gr.cu` lines 884–910: the `max4` branch guards `ct<=4 &&
gr_down_max4()`. With `STRATA_NO_MULTI_GR` unset (default), the source
dispatch in `src/kernels/cuda/fused_gr.cu` (lines 887–898) chooses
exact-T staged kernels for ct=1..6 before the generic max4 branch. The
`max4` generic path is not reached for this selected staged path.

**Result (G6):** The saved stderr reports selection of the staged
hyper-connection path. With STRATA_NO_MULTI_GR unset, the source dispatch
in `src/kernels/cuda/fused_gr.cu` (lines 887–898) chooses exact-T staged
kernels for ct=1..6 before the generic max4 branch. Therefore no
kernel-dispatch change is expected from STRATA_GR_DOWN_MAX4 for this
selected path. This combines a startup diagnostic with source-based
inference; it is not a runtime trace of every kernel launch.
off=36.79 tok/s, on=35.62 tok/s; n=1 per arm; identical output hash
`c44b95579fbfdc3a`; descriptive only, not statistical proof.

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

### Session recommendation

Based on the completed cold-graph triage:

- **Keep `--prefill auto`** (G4 shows ~4.5% higher prefill throughput than
  fixed 4096 on the 8887-token long-doc prompt on this host).
- **No `STRATA_GR_DOWN_MAX4` change** — expected no dispatch change for the
  selected staged path; source dispatch + staged startup diagnostic confirm
  exact-T branch is chosen before max4 (see §10 and G6 result).
- **No `STRATA_IQ256_GATHER` change** — gather path is not active for IQ3_S
  on this host; AVX-512 routes via `native_gu_rows` before `cpu_gather_fast`
  is reached. Standalone parity passes, but the knob has no end-to-end effect
  (see §10).
- **Upstream decode gains are modest cold-graph triage only:** G2/G3 decode
  +3.2%/+3.4% (candidate better); G1 decode −1.1%; G2 prefill +0.3%
  (descriptive; significance unestablished). None of these constitute a
  compelling case for Tier B/C promotion on their own.
- **Do not promote adapt-every 1 or 2 over production.** The ∼44.5% higher
  forced-continuation throughput (every1 vs off) is not a free-generation or
  quality-parity claim. Production default is every=4/swaps=96 — neither
  tested here. No quality certification or steady-state comparison is
  established.
- **No engine changes, no Tier B/C escalation, no production restoration
  needed** from this session. Original HEAD `9921eec` and saved fork/main
  ref `13be4c37` are unchanged. No further GitHub calls are made.

### Known limitations

Two harness follow-ups are tracked separately and not addressed in this
session: (1) direct CLI cancellation cleanup — arm processes are not
cancelled via the harness in the current implementation (sta-d76); (2)
prefill token-count validation outside this epic scope (sta-0a7). Neither
affects the results reported here: all arms ran under supervised
control-group conditions, and the prefill metrics used (`prefill_tok_s`)
are valid for the prompts where they are the primary signal.

---

## 12. Raw artifact paths

All artifacts pinned to `/data/llm/Strata-tests/tier-a-20261007/` on nibbler.
Working copies also at `/tmp/strata-tier-a-artifacts/` on paseo-nibbler.

| File | Contents |
|------|----------|
| `build-and-snapshot.sha256` | SHA-256 of both binaries + 5 source snapshots |
| `build-gcc15.log` | Full candidate build log (145 objects, 20 build warnings: 9 format, 5 indentation, 6 unused) |
| `configure-gcc15.log` | CMake configure output |
| `production.CMakeCache.txt` | CMakeCache snapshot — **baseline build only** |
| `targeted-gates.log` | 18-test parity suite (ctest output) |
| `iq-gather-off-parity.log` | IQ AVX2 parity at STRATA_IQ256_GATHER=0 |
| `iq-gather-on-parity.log` | IQ AVX2 parity at STRATA_IQ256_GATHER=1 |
| `gr-max4-parity.log` | GR/fused-GR parity (gr_parity test detail) |
| `host.bench-state.txt` | CPU/GPU/memory/service state at bench time |
| `smoke/tier_a_results.json` | Full smoke JSON (6 arm records) |
| `smoke/arms/<label>/` | Per-arm: `argv.txt`, `stdout.txt`, `stderr.txt`, `record.json`, `prompt_ids.txt`, `output_ids.txt` |
| `smoke.log` | Harness stdout for the smoke run |
| `full/tier_a_results.json` | Full matrix JSON (36 records: 35 eligible + 1 skip, 859 s) |
| `full/arms/<label>/` | Per-arm artifacts (same structure as smoke) |
| `full.log` | Harness stdout for the full matrix run |
| `adapt-replay-canonical/adapt_replay_results.json` | Adapt-replay JSON (9 arm records, forced continuation, 193 s) |
| `adapt-replay-canonical/arms/<label>/` | Per-arm artifacts for adapt-replay |
| `adapt-replay-canonical/reference_ids.txt` | Canonical 321-id reference snapshot (hash `c44b95579fbfdc3a`) |
| `adapt-replay-canonical/reference_meta.json` | Reference provenance (source path, hash, dry_run=false) |
| `adapt-replay-canonical.log` | Adapt-replay harness stdout |
| `harness-replay-canonical.sha256` | SHA-256 of replay harness Python source files |
| `harness-tests-131.log` | 131/131 unit test run log (nibbler host run only) |
| `final-state.txt` | Post-run host state (GPU 2 MiB, launcher 644, no timers, source SHA unchanged) |

Additional VM-side test log (not in host artifact tree):
`/tmp/tier-a-tests-131.log` on paseo-nibbler — same 131/131 result, separate run.

---

*This document records measurements taken. All full matrix and adapt-replay
artifacts are complete. See §11 (Session recommendation) for the session
conclusion. No engine changes, Tier B/C escalation, or production restoration
are recommended from this session.*
