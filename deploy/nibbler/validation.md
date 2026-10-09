# Nibbler validation and deployment

## Port status — `kv-unified`

**Port in progress; being validated. No port deployment is claimed.**
Source/config inspection used upstream main base `fb58e0db`, port commits
`1092339f` and `537f8b35`, and the real `src/program/generate.cpp` parser.
Despite the 0.1.40.x lineage description, both the base and current
`CMakeLists.txt:11` say `project(strata VERSION 0.1.41 LANGUAGES CXX)`.
The fork snapshot supplying these deployment files was `9921eec`.

This section records local, read-only source checks and the config/doc edits
for task `str-7ze.6`. No build, engine launch, model inference, SSH, network or
live deployment inspection was performed. All dated measurements, test counts,
rollout hashes and operational statements in the remainder of this file describe
**the historical fork lineage**, not this port or the present live host.
Historical publish/launch/rollback instructions are retained as records, not
current port instructions or authorization to change infrastructure.

### Flag-by-flag parser check

All 18 flags in `config.json`'s `args` still exist with compatible arity.
The corresponding fork-tip parser entries were also inspected: no configured
flag was found to be removed or repurposed at the parsing level. Values below
are accepted option syntax; model files and actual memory availability have
not been checked.

| Flag / config key | Config value | Arity and result | Port parser line(s) |
|---|---|---|---|
| `--pack` | IQ3_S pack path | One path; compatible | 1650 |
| `--native` | IQ3_S GGUF shard path | One path; compatible | 1725 |
| `--ple-gguf` | PLE GGUF shard path | One path; compatible | 1698 |
| `--expert-profile` | Expert-profile path | One path; compatible | 1924 |
| `--expert-cache` | `auto` | One integer or `auto`; `auto` still selects sizing from free VRAM | 1771–1774 |
| `--prefill` | `auto` | One size or `auto`/`auto:N`; compatible | 1787–1796 |
| `--spec` | `4` | One integer; verification-window setting retained | 1803 |
| `--spec-min-p` | `0.5` | One floating-point value; draft extension threshold retained | 1825 |
| `--mtp` | MTP path | One path; ordinary MTP retained, not batch MTP | 1819 |
| `--max-context` | `262144` | One integer; logical ceiling and unified aggregate backing budget | 1670 |
| `--kv` | `int8` | One format; `int8` accepted by format validation | 1705, 2211–2216 |
| `--kv-resident` | `32768` | One integer; GPU-resident cells per attention layer | 1706, 2225–2231 |
| `--kv-unified` | Enabled | No value; shared-pool mode, subject to restrictions below | 1804 |
| `--vision` | Enabled | No value; image input enabled | 1833 |
| `--vram-reserve-mib` | `2048` | One integer, MiB; explicit VRAM reserve retained | 1784 |
| `--conversation-cache-mib` | `8192` | One nonnegative integer, MiB; optional parking budget | 1835–1848 |
| `--conversation-cache-slots` | `4` | One nonnegative integer; parked-entry cap, not active slots | 1835–1848 |
| `--conversation-cache-min-free-mib` | `4096` | One nonnegative integer, MiB; physical available-RAM floor | 1835–1848 |
| `"parallel"` (server config) | `2` | Supported config key; becomes one-value `--batch 2` | Engine 1805; `serve/server.py:2552–2570` |
| `--serve` (server-added) | Enabled | No value; added when the server launches the engine | Engine 1832; `serve/server.py:651` |

**`--parallel` is not an engine flag**, in either the fork-tip or ported parser.
It would hit the unknown-argument error at `generate.cpp:1963–1969`. The tracked
config correctly uses `"parallel": 2` instead. A local AST-extracted execution
of the actual `parallel_args` function returned `["--batch", "2"]`, without
importing or starting the server. Every tracked argument/value boundary and
the conversation-cache integer ranges were checked locally.

Unified-mode startup checks (`generate.cpp:1971–1992`) require `--serve`,
`--batch 2..8`, no `--layer-split` (even `auto`), and `--batch-groups 1`, not
`auto`. The latter defaults to 1. They explicitly reject `--batch-mtp` and
`STRATA_BATCH_MTP=1` (also any nonempty value not beginning with `0`). They
also reject elastic K/V (`--kv-grow`/`--kv-elastic`, or enabled
`STRATA_KV_GROW`): the unified and elastic pools are mutually exclusive.
The config selects only GPU 0 and requests none of those incompatible options.
Failure to allocate at least two unified slot sessions is fatal, not a silent
independent/solo fallback (`generate.cpp:4007–4011`).

Thus the server-generated command has compatible parser syntax and satisfies
the static unified prerequisites **provided the inherited environment does not
enable batch MTP or elastic KV**. This is not an executed startup test: loading
GGUF shards/profiles, GPU allocation and other runtime gates can still fail.
`--prefill auto`/`--expert-cache auto` remain automatic choices on the new
upstream engine; compatibility does not establish byte-identical output,
identical memory/chunk sizing, or the fork's measured performance.

### Environment and JSON check

`serve/server.py:2703–2722` copies the parent environment and overlays
`config.env`. The port reads `STRATA_PF_FUSED` in
`src/prefill/moe_fused.cu:657,668`; value `1` explicitly requests native fused
experts on supported devices/formats.

No reader for `STRATA_PREFILL_FUSED_TAIL` exists in the port's `src/` or
`include/` (nor in serving/SYCL sources). Its old config value `0` was inert and
has been removed; the resulting env block is:

```json
"env": {
  "STRATA_PF_FUSED": "1"
}
```

The fork's `STRATA_PREFILL_BALANCE_TAIL` path is also absent. Neither experiment
is available just because historical records below describe it. Engine flags,
slot count, memory budgets, model/tokenizer/vision paths, hosts and other config
fields are unchanged.

Validation command and output: `python3 -m json.tool deploy/nibbler/config.json`
printed the complete formatted JSON object with the env block above, no error
output, and **exit status 0**. A separate source/config check passed all 18
argument arities, the server's parallel translation, static unified prerequisites,
conversation-cache value ranges and the remaining environment reader.

### Old claims that do not carry over

- The old current branch names and "validated/deployed" status describe fork
  builds. They are not the port's status, and the old SHA-256 values do not
  identify a tested port binary/frontend/config.
- The old README's tail-experiment availability and its
  `src/spec/output_limits_test.cpp` / `src/prefill/chunk_schedule_test.cpp`
  references are wrong for this checkout: those knobs/tests are absent.
  Historical timings and checks are preserved below, with their fork dates.
- `deploy/nibbler/start-strata` is absent from this checkout. External launcher
  delegation, runtime-config refresh and frontend/proxy port integration are
  therefore not ready/verified for the port. No launcher was added by this task.
- Configured `/data/llm/...` artifacts, CUDA 13.3 libraries, ownership/settings,
  live service/alias/host behavior, saved rollback files and the documented
  hardware/toolchain have not been inspected on the live host here.
- No port build/backend, GPU parity, long-context pressure/handoff, model quality,
  HTTP/vision or deployment gate has been run in this task. Historical successes
  and limitations must not be promoted into port validation results.

## Historical fork validation and deployment — 2026-10-03/04

> The branch, binary and source statements in this section describe the fork's
> 2026-10-03/04 rollout. Later dated sections describe subsequent fork rollouts;
> none establishes the port's deployment or current live-host state.

Branch: `nibbler/prefill-and-conversation-cache`. Upstream was fetched and verified
at `99f3dbd0b21d1401b3769e0c0d963913607f380b`. No fork or upstream push was made.
Hardware: RTX 5060 Ti 16 GB / Ryzen 9 9900X; existing CUDA 13.3/GCC 15 Release build.

## Enabled production changes

1. Native fused full-chunk prompt experts (`STRATA_PF_FUSED=1`).
2. Host conversation parking: 4096 MiB budget, four-entry cap, 4096 MiB available-RAM
   floor. Large snapshots can be rejected; this is not four guaranteed full-context
   slots or a hard OS memory reservation.
3. Speculative verification/commit is bounded by remaining output/context and first
   EOS, in both CLI and serving. Hidden accepted tokens after length/EOS no longer
   enter persistent model state.

The small-tail optimization was implemented and tested, but **not enabled**: neither
rebalancing nor routed native fused tails demonstrated a reliable end-to-end win.
Production retains the existing MMQ small-tail fallback. Both experimental knobs
remain available for controlled comparisons; `STRATA_PREFILL_FUSED_TAIL=0` is
explicit in the deployment config and balancing is off by default.

## Matched prefill timings

Four fresh alternating A/B samples per size, median engine `prompt_ms` (not just
CUDA kernel/timeline time), same model/template/context/speculation/vision. All
performance samples reported zero cached tokens. Parking and adaptive swaps were
disabled for these comparisons; decode PCIe fraction was zero and the one-row IQ
GPU path was forced with `STRATA_IQ_MT_MIN=1`. Production keeps its original adaptive
settings and does not force that diagnostic knob.

| Prompt tokens | Original upstream MMQ | Final branch defaults | Less prefill time |
|---|---:|---:|---:|
| 9,177 | 5.909 s | 5.468 s | 7.5% |
| 18,177 | 11.079 s | 10.264 s | 7.4% |

This is approximately 8% higher prefill throughput. It is not an all-workload
throughput or decode-speed claim.

The original fused-only control had a 5.451 s 9k median. Tail prototypes measured:
rebalanced chunks 6.044 s; routed per-expert fused 5.654 s; contiguous batch-8
5.503 s; batch-8 spanning inactive ids 5.718 s; pipelined batches using the already
allocated large ring 5.479 s. The final prototype was within roughly 0.5% of the
control, not a convincing improvement. No slower experiment was enabled.

## Quality and correctness evidence

- Ten matched quality fixtures cover arithmetic, Python/code, binary search, SQL,
  JSON extraction, French, Chinese, two code prompt lengths, and GPU vision.
  **9/10 correct in both upstream and final branch; no new quality failures.**
- The added 8,720-token code fixture expects 463 but returns 232 in original MMQ,
  fused-only, and tail candidates. Its expectation was not changed; it is explicitly
  reported as a pre-existing failure, not counted as a pass. The 28,560-token code
  fixture and vision fixture pass.
- Seven standalone output-limit/chunk-schedule/conversation tests pass.
- Native fused CUDA reference tests pass for all six exercised format pairs.
  Launch partitions 1/8/32/128 match the original fused per-pair outputs exactly in
  those tests, including inactive interior/end ids with null blob pointers.
- Model-level caps 1..8, natural EOS, consumed-state lengths, and immediate live-prefix
  reuse pass on the final binary.
- A/B/A parking passes byte-exact main-model-state and output checks for one-token
  windows; speculative-window-4 A/B/A passes output/reuse checks. Byte-exact state is
  not claimed for the window-4 comparison. The parity harness excludes irrelevant
  padding/stale cells and the drafter's uncomputed final cell.
- Byte-pressure (350 MiB) and oversized (1 MiB) fallback scenarios pass output and
  byte-exact main-model-state checks.
- Cold raw-engine boundary probes cover batched tails 63/64/512/978/1023/1024 with
  tail fusion off/on; output and logical consumed length agree. These truncated-data
  probes naturally emit EOS and are boundary/coherence checks, not a quality score.
- Python suite: 143 tests, five skipped, no failures, using nibbler's existing venv.
  The quality validator now rejects boolean-as-number answers and malformed schemas.

Per-run host staging/PLE telemetry now reports deltas rather than cumulative totals.
These smoke/reference gates are not proof of equivalent quality on all workloads.
Maximum-context, mixed long-running concurrent-client, broad stochastic-quality,
and other-GPU/HIP validation remain outside this run.

## Live deployment verification

llama-swap still uses its original model name/aliases and launcher path; the launcher
now delegates to this branch. Only Strata was unloaded/reloaded. The original binary
`engine/strata` and original production JSON were verified byte-identical to backups.

The first switch rolled back automatically because the service user could not create
a log in the root-owned checkout. The corrected deployment keeps mutable files in
`/data/llm/Strata-run/nibbler/`, owned by `native-inference` with mode 0700. Existing
shared Chat defaults were copied byte-identically; restarts refresh the config but
not the shared settings. Tracked source/config files do not become service-writable.

Live proxy A/B/A returned 391 / Lin / 391. A-again restored 182 parked checkpoint
tokens; its prompt time was 161.5 ms versus 987.0 ms for cold A. Native log confirms
RAM restoration in 15.6 ms. Snapshots were approximately 228 MiB. Health confirms
loaded, context 262144, and images enabled; the running command uses
`engine/strata-nibbler` and all three parking limits.

Validated deployed binary SHA-256:
`ec160772e1f34d0dc5c5730f82287312c1cbb494e5924012857b99dac76143a4`.
Native source last changed at `6cf61ad`; later commits adjust test tooling,
deployment files, and documentation.

## Evidence and rollback

Remote evidence: `nibbler:/data/llm/Strata-tests/implementation-20261003/`.
Local results/report/bundle: `/home/pete/strata-implementation-20261003/`.
Important records: `upstream-full.results.json`, `branch-final-default.results.json`,
`final-default-native-tests.log`, `final-default-fused-reference.log`,
`final-checks.log`, `final-python-tests.log`, `tail-boundaries.results.json`, the
`final-spec-state` / `final-cache-*` directories, and `production-branch-smoke.json`.
The upstream ten-fixture record combines its original nine-fixture run and the
separate targeted rerun of the newly added code fixture.

Original launcher, binary and JSON are saved as `launcher.before`, `strata.before`,
and `production-config.before.json`. Follow `README.md` for an idle-window rollback.
The router config and unrelated pocket-tts service were not changed.

To publish later, create your GitHub fork, add it as a separate remote, and push
`nibbler/prefill-and-conversation-cache`. Do not use upstream's auto-update script
blindly on this deployment; build/install the branch binary deliberately.

## Historical fork deployment — 2026-10-06

Native source: `feature/kv-warm-handoff` at `3e98524`. Live artifacts, re-hashed
on nibbler with a root shell:

- `engine/strata-nibbler`: `d318cd7baf926c2b1a3d3eebfd94a615d5e9fe909fd560dcd556d61174771cac` (unchanged)
- `serve/server.py`: `262f00a67b7056b3e2d7f2e0192dd4f923f865ea55f64cf9aa84a0fff0e8bb26`, byte-identical to `3e98524`
- `deploy/nibbler/config.json` and `/data/llm/Strata-run/nibbler/config.json`: `8178be74bc7b08ec609d754291dc576c322b333cbd169f65bf02d6e08f33fa27` from this commit. Production ran `e806bfeb934c1fe7350ec305d49af66ed8089bbeb8d5198b032967e236b38166` before it, whose extra `nibbler.local.keen.land` host entry was committed nowhere - `deploy/nibbler/start-strata` copies the tracked config over the runtime config at every model start, so any `git checkout`/`git pull` on the host would have dropped that host and answered those requests with 403.
- Shared Chat settings: `b61e757d0d94ca1f1c7c17cf180f98e779cbd976d53658a88ae91d870d9c3e2a` (unchanged)

The deployed binary's `generate.cpp` differs from this commit's only in three
readability hunks (`store(true)` / `store(false)` / an implicit bool instead of
`!= 0`); that file hashes to `1339145eb288e5b68bead93cb8828bc545adb340c5b7e064359eced4d9627af1`,
frozen in `/data/llm/Strata-tests/handoff-20261005/native-v2.candidate.sha256`.
Comparing all 388 tracked source/config files on the host against `3e98524` found
those hunks, the config above and three stale frontend test files - nothing else.

Deployment hygiene that came with this commit:

- The host checkout sat at `00a7289f` with the deployed frontend and the unified-KV
  headers as uncommitted edits, and it is now aligned to this commit: a rebuild
  reproduces the deployed source, and the tree's own tests match `serve/server.py`.
  The stale `serve/test_unified_lifecycle.py` had errored 4 of 15 against the
  deployed frontend (`blocked_resume() got an unexpected keyword argument
  'max_new'`), and `serve/test_kv_handoff.py` / `serve/test_incremental_kv.py`
  were not on the host at all.
- `/etc/systemd/system/strata.service` and `strata.service.d/` are removed. That
  dead unit started the pre-unified-KV stack (`engine/strata`, port 8088,
  `--vram-reserve-mib 700`, no `--kv-unified`), so `systemctl start strata` would
  have put a second, older engine on the GPU beside the live one. The live stack
  is llama-swap's `qwen3.8-flash-next-iq3_s`:
  `/opt/native-inference/bin/start-strata-sharp-medium ${PORT}` calls
  `deploy/nibbler/start-strata` (frontend 5801, proxy 8088).
- Parking raised to `--conversation-cache-mib 8192`. `--conversation-cache-slots 4`
  and the 4096 MiB available-RAM floor are unchanged: the raise is about the
  measured 2.0-2.4 GiB parked images, and nibbler has 123 GiB total with 35 GiB
  available at idle and `Max locked memory` unlimited for both processes.

Measured before the raise (engine log window of 11 h, 0 `ERR` lines): 2,777
requests; prompt reuse 180,131 of 181,577, 182,654 of 185,136 and 186,211 of
186,819 tokens; 169 `TARGET_ONLY slot clone` and 293 `TARGET_ONLY decode`; drafts
ran on 2,260 of 2,777 requests at 67.7% acceptance (1,382,206 of 2,041,742
offered); 22 pressure events, all parked, no budget/RAM-floor/allocation miss.

Applied on 2026-10-06: the host checkout moved to this commit as branch
`nibbler-deployed` (the worktree it replaced is saved under
`/data/llm/Strata-tests/audit-20261006/`: `worktree.before.patch`,
`untracked.before.tar`, `config.tracked.before`, `config.runtime.pre-8192`),
`strata.service` and `strata.service.d/` were removed, and llama-swap was
restarted at 07:27 EDT - `/health` 200 after 30 s. The runtime config copy is now
`8178be74…` and the engine runs `--conversation-cache-mib 8192
--conversation-cache-slots 4 --conversation-cache-min-free-mib 4096`, which
`/metrics` reports as `conversation_cache_mib=8192`. After the reload: the host's
own frontend tests pass (11 handoff, 18 incremental KV, 26 unified lifecycle, 13
unified reporting, 11 parallel, 8 lifecycle, 10 monitor); `Host:
nibbler.local.keen.land` answers 200 while an unknown host gets 403; a two-turn
chat answered correctly with the second turn reusing 183 of 207 prompt tokens;
pocket-tts came back.

The binary was still `d318cd7b…` at that point, built from the three-hunk variant of
`generate.cpp`; the rebuild below replaced it.

### Serving-path fix deployment — 2026-10-06

`feature/kv-warm-handoff` at `9b0c8e2` (the three serving-path fixes and their tests)
was built and deployed. The host checkout moved to that commit on branch
`nibbler-deployed`; because those commits do not touch `serve/server.py`, the frontend
stayed byte-identical.

Build: `cmake --build build -j 4` in the deployment's own `/data/llm/Strata/build`
(Ninja Release, CUDA 13.3, arch 120, g++-15, `STRATA_BUILD_CONVERSATION_TESTS=ON`), 33
steps, exit 0, with only the two pre-existing `-Wunused` warnings (1023 `argmax`, 6421
`w`). This build compiles this commit's `generate.cpp` (`bece8f5b…`) directly, so the
three readability hunks that separated the previous build from its source are gone.

Exclusive-GPU gates, production unloaded for the window, artifacts in
`/data/llm/Strata-tests/servingfix-20261006/`: `ctest-final-exclusive.log` **81/81
passed** (100.61 s, same three exclusions as the previous rollout), plus
`warm-small-v2.json`, `warm-streamed-v2.json` (35000/35004-token histories, 3 cycles),
`cancellation-pressure-v2.json`, `coherence-v2.json` (pressure restore, 4096-MiB
parking), `private-http.json`, `private-vision.json` and `private-http-warm.json`, all
`passed: true`. Each gate restored production hash-identical before the next one
(`*.production-unchanged.log`).

`deploy-servingfix.sh` then installed the frozen candidate after unloading only Qwen and
verified live properties. Deployed hashes (`rollout.sha256`):

- `engine/strata-nibbler`: `be37cae8df1cb73879b7aace36dc6bd4709d38bfc86f4d9d6abb0ad0c5aa4ef3` (was `d318cd7b…`)
- `serve/server.py`: `262f00a67b7056b3e2d7f2e0192dd4f923f865ea55f64cf9aa84a0fff0e8bb26` (unchanged)
- `deploy/nibbler/config.json` and `/data/llm/Strata-run/nibbler/config.json`: `8178be74bc7b08ec609d754291dc576c322b333cbd169f65bf02d6e08f33fa27` (unchanged)
- Shared Chat settings: `b61e757d0d94ca1f1c7c17cf180f98e779cbd976d53658a88ae91d870d9c3e2a` (unchanged)

Live after the rollout: two slots, 262144 backing, 32768 residency, vision,
`kv_handoff=1 kv_incremental=1 kv_reserve_ahead=256`, `conversation_cache_mib=8192`,
`/health` 200, `Host: nibbler.local.keen.land` 200 with an unknown host 403,
llama-swap active and pocket-tts ready. The host's own frontend suites pass (12
`test_kv_handoff` - the deployed source carries the test added in this work - 18
incremental-KV, 26 unified lifecycle, 13 unified reporting, 11 parallel, 8 lifecycle, 10
monitor; log in the gate directory as `frontend-tests.log`), and a two-turn chat through
8088 reused 212 of 240 prompt tokens on its second turn.

Rollback: `bash /data/llm/Strata-tests/servingfix-20261006/rollback-handoff.sh` restores
`baseline.strata` (the pre-fix `d318cd7b…`) and `baseline.server.py` after hash checks,
reloads only Qwen, and leaves configs, shared settings, router and TTS untouched. Not
executed.

Remaining uncertainty: the shortage-parking cost of the third fix is still unmeasured;
no parked event has yet fallen back to replay. The 8 GiB parking budget, by contrast, was
exercised on 2026-10-06 by an operator-run two-stream capacity probe against the live
engine: parking held `parked=3 bytes=8343518772` (7.77 GiB) and, in the first phase,
`bytes=8568791496` (7.98 GiB) - inside the raised budget and roughly twice the previous
4096 MiB one, so under the old value one of the two large images would have been evicted
and its stream forced to replay. The probe also measured the aggregate-capacity wall
(250k cells co-resident; 300k forcing one stream to re-prefill its history) and the
idle-claim restore limitation behind that replay; see
`docs/INCREMENTAL_UNIFIED_KV.md`,"Two-stream aggregate-capacity probe (2026-10-06)".
Evidence: `/data/llm/Strata-tests/overcap-20261006/` and
`/data/llm/Strata-tests/overcap-clean-20261006/`.
### Reclaim-restore rollout — 2026-10-06

`feature/kv-warm-handoff` at `6cad2c7` (epic `str-kp7`: reclaim a recoverable claim instead
of replaying; see `docs/INCREMENTAL_UNIFIED_KV.md`, "Reclaiming a claim instead of
replaying"). The host checkout moved to that commit on branch `nibbler-deployed`;
`serve/server.py` is untouched by these commits, so the frontend stayed byte-identical.

Exclusive-GPU gates on the frozen candidate, production unloaded for the window, artifacts
in `/data/llm/Strata-tests/kvrestore3-20261006/`: `ctest-final-exclusive.log` **81/81
passed** (101 s), plus `warm-small-v2`, `warm-streamed-v2`, `cancellation-pressure-v2`,
`coherence-v2`, `private-http`, `private-vision` and `private-http-warm`, all
`passed: true`; production restored hash-identical after every window.

`coherence-v2` failed an earlier revision of this change and is worth keeping in the gate
set for it: the first deferral rule pushed a 1,015-token restore back in the small-backing
FIFO stage (`another slot holds 104 pages`) and answered a zero-token main-path pressure
continuation where the harness expects the engine to proceed ("pressure must be a nonempty,
nonfinal batch attempt"). The rule now requires the avoided replay to be at least 8,192
tokens and the busy owner's remaining allowance to satisfy `R * 45 <= restore_end`; that
stage now logs zero deferrals and zero reclaims.

Capacity probe (three phases: 2x125k, then 2x150k, then 2x170k), candidate on a private
endpoint with production-equivalent settings, then again against the live stack:

- private: 0 `skip restore`, 1 deferral, 3 reclaims, 0 `ERR`; the 300,004-cell phase's
  second stream restored (`prompt 150002 tokens = 125003 reused + 24999 read in 16137 ms`)
  where the pre-fix binary replayed 150,000 tokens in 82,527 ms, and the 340,003-cell phase
  kept both streams warm (27.6 s / 14.1 s wall).
- live (`rollout.sha256` binary, 8088 -> 5801): 0 `skip restore`, 1 deferral, 1 reclaim,
  1 restore, 0 `ERR`; the same phase went from 100.5 s wall / 82.5 s read to **34.1 s wall /
  16.1 s read**, and the other stream stayed warm at 17.4 s.
- live limitation, recorded not hidden: in the 340,003-cell phase the partner stream
  replayed 169,998 tokens in 93.9 s because its parked image had been evicted (the window's
  evictions reached 3 before its turn) and its slot claim had already been released by the
  other stream's pressure path. The pre-fix binary would have replayed that turn too; the
  mitigations (pin the image, refuse the release, prefer superseded drops, or raise the
  budget) are unmeasured. Live probe evidence: `live-probe/` in the gate directory.

Deployed hashes (`rollout.sha256`):

- `engine/strata-nibbler`: `1d71f758c08ca8992f80a3ca7ff4b245db4398d8d003f26ddb47ec27dddcf7de` (was `be37cae8…`)
- `serve/server.py`: `262f00a67b7056b3e2d7f2e0192dd4f923f865ea55f64cf9aa84a0fff0e8bb26` (unchanged)
- `deploy/nibbler/config.json` and `/data/llm/Strata-run/nibbler/config.json`: `8178be74bc7b08ec609d754291dc576c322b333cbd169f65bf02d6e08f33fa27` (unchanged)
- Shared Chat settings: `b61e757d0d94ca1f1c7c17cf180f98e779cbd976d53658a88ae91d870d9c3e2a` (unchanged, verified before and after)

Live after the rollout: `engine/strata-nibbler` running with
`--conversation-cache-mib 8192 --batch 2`, `kv_capacity_cells=262144`,
`conversation_cache_mib=8192`, `kv_handoff=1 kv_incremental=1 kv_reserve_ahead=256`,
`/health` 200, `Host: nibbler.local.keen.land` 200 with an unknown host 403, and the host's
own frontend suites pass (`frontend-tests.log`: 12 handoff, 18 incremental-KV, 26 unified
lifecycle, 13 unified reporting, 11 parallel, 8 lifecycle, 10 monitor).

Rollback: `bash /data/llm/Strata-tests/kvrestore3-20261006/rollback-kvrestore.sh` restores
`baseline.strata` (`be37cae8…`) and `baseline.server.py` after hash checks, reloads only
Qwen, and leaves configs, shared settings, router and TTS untouched. Not executed.

#### Eviction-order follow-up — rolled out in two steps

`fa7b977` makes `make_room` give up a copy the incoming chain covers before the oldest-first
order. `drop_superseded`'s exact test missed every park in both probe runs (the client re-renders
its last reply), so the live 340,003-cell phase evicted a conversation's only 150k image and its
partner replayed 169,998 tokens in 93.9 s.

Step 1, `f5f8b42a` (source `fa7b977`): `ctest` 81/81 and all seven gates passed on the frozen
candidate (artifacts in `/data/llm/Strata-tests/evict-20261006/`; CPU `conversation_cache_test`
4,205), the private probe was clean, and it was deployed — but the live probe **reproduced the
replay** (108.5 s wall, `no image for a 169999-token prompt (parked=2 bytes=6250821344
evictions=3; slot -1 holds 0)`). `park_current` and `park_slot_target` reserve room with
`make_room(estimate, held)` before they capture, and those calls had no incoming chain, so they
evicted oldest-first and `put`'s chain-aware choice never ran.

Step 2, `40e8305c…` (source `f7fe1cd`): both reservation sites now pass the chain they are about
to park. `ctest` 81/81 and all seven gates passed again (artifacts in
`/data/llm/Strata-tests/evict2-20261006/`; `conversation_cache_test` 4,208), the candidate was
deployed, and the live three-phase probe now logs 0 `skip restore`, 1 deferral, 2 reclaims,
2 restores, 0 `ERR`: `reclaimed idle slot 0 (42503 pages) for a canonical restore` →
`restored 149999 tokens (live) in 123.8 ms` → `prompt 169999 tokens = 149999 reused + 20000 read
in 12979 ms`, 27.7 s wall against the 108.5 s replay, with the other stream warm at 14.2 s and
both slots ending idle at ~170k tokens.

Deployed hashes (`rollout.sha256` in the step-2 directory): `engine/strata-nibbler`
`40e8305c8e09e657e22f520255185bee29faaa10b988b5b0b09500d98529d191`; `serve/server.py`
`262f00a6…`; tracked and runtime config `8178be74…`; shared Chat settings `b61e757d…` (all
unchanged). Rollback for either step restores the preceding binary after hash checks
(`rollback-evict.sh` → `1d71f758…`, `rollback-evict2.sh` → `f5f8b42a…`), reloading only Qwen and
leaving configs, shared settings, router and TTS untouched. Neither was executed.

Gate note for future windows: the model unload acknowledges before the engine's VRAM returns, and
`prefill_fused_iq_test` then fails with `cudaMalloc: out of memory` while nothing else is running.
`native-gates.sh` now waits for under 2 GiB GPU memory used before `ctest`.

## 2026-10-09 — `kv-unified` port with NVFP4 and batch MTP deployed to production

Authorized by the owner after the port's model gates were run on isolated hardware. The engine is
the integrated `kv-unified` line (unified KV port + NVFP4 kernels and expert cache + opt-in unified
batch MTP), built in `build-deploy-validated` in the host checkout at commit `7256bdaa`.

- Config (`deploy/nibbler/config.json`, copied to the runtime config by the launcher on every start):
  the `huihui-nvfp4` pack with `--batch-mtp`, `--spec 4 --spec-min-p 0.5`, `--max-context 65536`,
  `--kv int8 --kv-resident 32768 --kv-unified --no-kv-grow`, `--vram-reserve-mib 2048`, conversation
  cache 0, `parallel: 2`, no vision section (the NVFP4 pack ships no projector).
- Deployed `engine/strata-nibbler` sha256 `25cec204db80104c9f0049ee958cbf5792884e27f4dcdbff14eaf76b87bfd836`
  (`--version` 0.1.41); `engine/strata-vision` left as it was.
- Backup and rollback (not executed): `/data/llm/Strata-backups/deploy-20261009T192028Z/` with the
  previous engine, vision binary, `BUILD.json` and runtime config plus `rollback.sh`.
- Live confirmation: two concurrent completions through `llama-swap` produced
  `strata batch_mtp_stats slot=0 windows=33 offered=33 accepted=27 rejected=6` and
  `slot=1 windows=34 offered=34 accepted=29 rejected=5`, both with zero fallbacks — one proposal per
  window on each active slot.
- Model gates against the deployed artifact: default target-only baseline 26/26; correctness
  (`--batch-mtp`, hook build) offered 17 / accepted 13 / rejected 4 with exact token-ID parity;
  limits; lifecycle; tails (reused 105/106/103, branch depths 8/8/8, clone offers 0); pressure
  (exhaustion, first `BDONE pressure`, 2048-token target-only park, sibling cancel, canonical
  restore `2048 == 2048`, last returned token unfed, exact continuation IDs); optional; handoff
  (reused 1538 == 1538 with 30/30 resumed MAIN offers). Records live in the bead trail and the
  per-task evidence files under `/data/llm/Strata-tests/deploy-gates-20261009/`.
- Known limitations at this date: the pressure gate requires a declared native substitute for its
  rejection witness when the fixture accepts every proposal (this NVFP4 fixture does so
  deterministically), the paused-prefill ordering witness in the baseline smoke is sensitive to the
  build layout, and the API reports `draft_n = 0` for concurrent batch requests while the native
  per-slot counters are positive.
- Operational note: with the NVFP4 model loaded the host keeps ~32 GiB RAM available with swap
  nearly full, so gate or benchmark runs must unload the model and use an exclusive GPU.
