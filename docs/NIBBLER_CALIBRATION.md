# Nibbler calibration — 2026-10-11

## Status: paused, no valid tuning measurements

Work covers `str-inn` (CPU/PCIe split and worker count) and `str-5p1` (VRAM
reserve). Keep production **pcie_frac 0.25, 11 pool workers and 2048 MiB reserve**.
This is a keep-current decision because measurement gates have not passed, not
evidence that those settings are optimal. No production config/binary was edited
or installed. No Beads, commits, agents or worktrees were changed by this helper.

The owner authorized model-only unload/reload in a quiet window, leaving the
router, CPU audio, VMs and other services running. The primary subsequently
**paused all benchmark and candidate-transfer operations for independent safety
review**. This pause supersedes the earlier experiment authorization. The runner
below is frozen review material, **not yet approved for another live run**.

## Observed deployment and host

Read the [performance audit](NIBBLER_PERFORMANCE_AUDIT.md) for the earlier snapshot.
This session used homelab `bin/hl peek` for equivalent host/path inspection,
including the complete saved `perf-fullctx-20261009/perf_driver.py` and
`task40-newpack-20261009/perf-guarded-launcher.sh`. Native topology, GPU consumers,
process identity, current native tail and execution have no equivalent `peek`
probe. The owner/primary explicitly authorized the existing homelab `RemoteShell`
transport for those needs: strict host-key checking, hostname
`nibbler.ts.keen.land`, `retries: 0`. No raw SSH fallback or probe-framework changes
were made.

The first native read-only snapshot showed:

- Two idle production slots, no running/queued backend work; Qwen and pocket-tts
  both ready. Same GPTQ NVFP4 pack and froggeric v22.5 tokenizer/template, GPU
  vision with 1024-token cap, batch MTP, shared 262144 backing / 32768 resident
  cells, INT8 KV, 8192 MiB / four-entry conversation parking budget.
- MemAvailable 15375156 KiB (14.66 GiB), SwapFree 9023768 KiB (8.61 GiB).
  These are time-local observations, not worst-case guarantees.
- GPU used/free 14057 / 1833 MiB. Only the installed vision encoder and engine
  appeared as compute consumers. Idle link generation 1, width 8; restored live
  request observations later showed generation 5, width 8. Neither measures PCIe
  transfer bandwidth.
- `lscpu -e=CPU,CORE,SOCKET,NODE,CACHE,ONLINE`: physical cores 0–5 and SMT siblings
  12–17 share L3 id 0; cores 6–11 / siblings 18–23 share L3 id 1. All are socket
  0, NUMA node 0. This records the reported cache topology; no affinity/CCD/BIOS
  tuning was applied.

## Invalid attempts, retained in full

[Bounded sanitized raw artifacts](../bench/results/2026-10-11-nibbler-calibration/)
contain before-state, outcome, sampled resources and restoration proof. Host
originals are under
`/data/llm/Strata-tests/perf-audit-20261011/baseline-control-{1,2,3}/`.

| Attempt | Stop before private model load | Outcome |
| --- | --- | --- |
| control-1 | Strict JSON parser rejected successful plaintext unload response | Finally restored production; no private process spawned |
| control-2 | Empty-body-only fix still rejected the nonempty `OK` response | Finally restored production; no private process spawned |
| control-3 | Private launcher could not find `runuser` in the deliberately restricted child PATH | Finally restored production; no private process spawned |

Every `result.json` has **`passed: false`, `runs: []`, `restored: true`**.
None is a valid baseline, loader timing, correctness parity or tuning trial.
The third command had already been dispatched when the primary's pause arrived.
Its tool transport was interrupted, but subsequent `hl` reads proved that the
host's finally block completed and wrote `restoration.json`; current independent
read-only checks also proved ready state. This successful recovery does **not**
prove general disconnect/signal cleanup. Review identified durable restoration
as a necessary gate before another model cycle.

The unload parser now has a specific plaintext-success path for the model
control API; metrics retain strict JSON parsing. Both empty and `OK`-style
responses have local tests. Independent re-audit marked the original three
Blockers and memlock High in `str-inn.1` **addressed**. It found H1 (no first-class
neutral preflight) and M1 (kernel child-list support not probed before readiness).
The scoped `str-inn.2` fix below has offline coverage and awaits the same reviewer's
re-audit. **No target preflight or live control is authorized by this fix.**

### Restoration proof

Current checks after control-3 used authorized `RemoteShell` for process/hash
inspection and GETs for `/running` and audio `/health`:

- `qwen3.8-flash-next-nvfp4-gptq: ready` and `pocket-tts: ready`.
- Short streamed router completion in `baseline-control-3/restoration.json`
  ended with `CALIBRATION READY`, `finish: stop`, five model output tokens.
  Router loading banners precede the answer; their chunks are **not model TTFT**
  and this request is labelled health-only, not performance data.
- Audio `status: ok`, `backend: cpu`. Router PID/start ticks
  `1435174 / 18877666`; audio `1643128 / 54686907`, identical to the captured
  pre-window identities. Audio did not restart in these attempts.
- All six baseline artifact hashes unchanged: installed engine, installed vision
  binary, BUILD.json, tracked production config, runtime production config and
  shared settings. Shared-settings contents were not read or exported.
- No private model process found; GPU consumers were only the installed
  `/data/llm/Strata/engine/strata-vision` (PID 2881000) and `strata-nibbler`
  (PID 2881053). The short health request and subsequent GET are both required
  evidence, not an inference from cleanup code.
- At this independent check MemAvailable was 24144977920 bytes (22.49 GiB),
  SwapFree 8699682816 bytes (8.10 GiB).

## Tooling and local tests

Files:

- `tools/nibbler_perf.py`: dry-run-default, bounded one-dimension runner with an
  explicit neutral `--run --preflight-only` execution mode.
- `tools/nibbler_perf_test.py`: CPU-only plan/parser/preflight/restoration tests;
  also imports the lifecycle fixture suite so the main test command covers both.
- `tools/nibbler_lifecycle.py`: separate Linux subreaper, durable ownership journal,
  scoped pidfd reaper, independent deadline and serialized recovery.
- `tools/nibbler_lifecycle_test.py`: disposable subprocesses and a loopback fake
  control API. Never contacts production or loads a model.

```sh
python3 tools/nibbler_perf.py --fractions baseline --repeats 1
python3 -W error tools/nibbler_perf_test.py -v
python3 -W error tools/nibbler_lifecycle_test.py -v
python3 -W error -m py_compile tools/nibbler_perf.py tools/nibbler_perf_test.py \
  tools/nibbler_lifecycle.py tools/nibbler_lifecycle_test.py
git diff --check
```

Current offline result: **58 combined tests passed** (25 original/updated runner
checks, 20 lifecycle fixtures, 11 preflight fixtures and two restoration
fixtures), with warnings treated as errors. Python compilation and whitespace
checks passed. All prior 51 tests remain green; seven scoped checks were added
for `str-inn.2`. The earlier 22-test run predated the lifecycle change. These
results establish the fixture contract below, **not target-host PAM readiness,
GPU cleanup, model-output parity, or live approval**.

Historical review hashes (the pre-fix scaffold, retained for comparison):

```text
a29bf335293bfe27494e73151d31f942b0921198d94f78e18c473efe73d0a578  tools/nibbler_perf.py
ecf9d6d780cedf77d8025489dbc18b50948a7d04e8536b4dd1f10b08c42491ef  tools/nibbler_perf_test.py
```

The runner's intended safeguards and records are:

- Dry run performs no HTTP, process launch, GPU probe, directory creation or
  unload. `--run` is explicit, but does not itself grant operator authorization.
- Root coordinator on nibbler, single operator lock, private copied binary,
  frontend source, config, cwd/logs, `native-inference` owner and loopback port
  18089. Absolute `runuser`/`ldd` launch tools, absolute venv Python pathname
  (do not resolve its symlink away from venv packages), and an exact-identity,
  exact-environment no-model preflight for **every prepared arm before unload**.
  The actual launch repeats that check after `runuser`/PAM, then execs the frontend.
  Both soft and hard effective child memlock must be unlimited; the coordinator's
  infinity is not proof and a finite limit refuses the window. Retain configured
  CUDA libraries and default affinity; still require successful full-arena
  `cudaHostRegister PORTABLE ok`, not unpinned fallback.
- Capture idle slots/queued counters immediately before the sole model unload;
  preserve router/audio PID and start-time identity and artifact hashes.
  Wait for GPU memory used **< 2048 MiB** before any private load or unloaded-model
  restore; an already-ready original model needs only health verification.
- Sample memory, swap, GPU use/free, active link state and load every second.
  Stop on MemAvailable **< 10 GiB**, SwapFree **< 1 GiB**, errors, invalid output,
  log/output bounds, private startup timeout, window deadline, or production
  Qwen unexpectedly reappearing. No parallel model loads or global cache/clock/
  BIOS/swap changes. One-second sampling cannot establish sub-second allocation
  peaks.
- Capture per-arm binary/config/tokenizer/template and published model-manifest
  hashes. Manifest identity is **not a new full model payload verification**;
  rehashing the 63 GiB expert payload would itself perturb cache state.
- Actual prompt/output token counts come from the API, not bytes/words. TTFT is
  first nonempty output chunk; inter-chunk p95 is not per-token p95. Nearest-rank
  descriptive p95 with very small n is not a confidence interval.
- Preserve native phase/hit/PCIe/draft/parking lines, loader time to ready,
  samples and all failed outcomes. Event-based GPU profiling is not enabled in
  speed trials. Batch-phase visibility of the current engine is not assumed.

**Known gaps:** the current scaffold does not yet drive full-context pressure/
canonical park/restore/handoff gates. `--reserve-gates` runs a synthetic image,
then explicitly stops because those gates are missing; lower reserves are
prohibited. It also does not establish native token/logit parity, actual active
PCIe byte bandwidth, per-process swap/RSS/cache attribution, PSI or audio latency
under expert CPU load. Do not mistake implemented flags for passed acceptance
criteria.

## `str-inn.2` neutral preflight — pending independent re-audit

Only H1/M1 are addressed here; the guardian protocol and measurement gates are
not redesigned. The default invocation remains a side-effect-free printed plan.
A future **separately authorized**, root-on-nibbler neutral execution uses:

```sh
python3 tools/nibbler_perf.py --run --preflight-only --fractions baseline \
  --tag neutral-preflight-20261011
```

Use a fresh tag for each execution; an existing tag refuses rather than overwrites.
This command was **not run on target** by the implementation helper.

Flag semantics:

- Neither flag: dry-run only, with no directories/processes/HTTP/bytecode.
- `--preflight-only` without `--run`: parser refusal (exit 2), no execution.
- `--run --preflight-only`: prepare/check every selected arm, then stop. It creates
  private preparation/evidence directories and runs neutral Python/import/ELF
  dependency probes through the exact post-PAM `runuser` identity/environment.
  Defaults remain 11 workers, 2048 MiB reserve and loopback port 18089; the example
  selects one baseline arm. Without `--fractions baseline`, all eight default
  arms are prepared and checked. Existing one-dimension and bounds rules apply.
- `--reserve-gates` is incompatible with neutral mode, and lower-reserve schedules
  still refuse. Repeat/output/context options retain plan validation but execute
  **no repetitions, prompts, vision gates or measurements** in neutral mode.
- `--run` without `--preflight-only` retains the supervised-window path and requires
  its own later authorization. Neutral success never grants that authorization.

Neutral execution checks `/running` for already-ready Qwen/audio **before** any
upstream idle GET, because a router upstream request could autoload an absent
model. It permits readiness/idle/audio GETs, host resource queries, hashes and
local preparation only. It verifies kernel direct-child-list availability in the
coordinator and post-PAM probes; prepares every arm; then rechecks ready/idle/audio
state, identities, hashes and memory limits. It returns **before Supervisor
construction/arming**, unload, restoration, model completion, private frontend,
native model executable or GPU allocation. No experiment monitor is started.
The supervisor independently probes the kernel child-list file again at startup;
absence/unreadability returns a clear negative readiness response before any
journal or mutation. The final empty-child-list restoration witness is unchanged.

Evidence in the fresh tag directory:

- `result.json` is labelled `mode: "preflight-only"`, with **`passed: false` and
  `runs: []` on both success and refusal**. `ready: true` / exit 0 means neutral
  readiness checks succeeded, not performance passed. `ready: false` / exit 1
  means execution refused. Parser/initial host/config/output-directory refusals
  can occur before a result file exists and still make no model mutation.
- `before.json` records readiness facts, PID/start identities, hashes (never
  shared-settings contents), resources, coordinator limits and kernel support.
  Raw slots, metrics/engine payloads, audio extras and process command lines are
  not exported in neutral evidence.
- `run-NN/preflight.json` records each prepared arm's effective uid/gid, child
  limits, Python/config identity and kernel/dependency readiness. These `run-*`
  directories are **preparation only**, not model runs. A refused child records
  structured identity/limit facts when available; arbitrary child stderr or
  exception payloads are not exported. JSON artifacts retain the 8 MiB bound.
- A passing result records final ready states and unchanged router/audio identities
  and production hashes. A failed/incomplete check does not assert unchanged
  production. Neither outcome attempts restoration: neutral mode never unloads.
  No `lifecycle.json`, unload/private-start artifacts or restoration proof exists.

New offline fixtures cover a three-arm positive execution, real `(0,0)` post-PAM
limit refusal, missing kernel support, already-unloaded Qwen refusal before any
upstream GET, incompatible CLI flags and no-bytecode refusal. Fixtures forbid
supervisor/monitor construction, unload/restoration/completion/private-exec calls
on success **and** failure, and reject secret-like extras in evidence. A separate
subprocess fixture simulates an unsupported kernel and verifies a clear readiness
refusal before journal creation or any fake-API mutation. All target PAM, real
native descendant/GPU-release and guardian-failure limitations below still apply.

Current code freeze for the same independent reviewer
(`a98dca6f-d019-4f8c-b3eb-e29b0cbfbdfb`):

```text
1852695e3e6d962567d24264f51c053a37836b63eefc9d86357452c5141041b3  tools/nibbler_perf.py
978f43961e26b67294e2937c646d98d2c41f45c1f9c49b3af3b35766612ee899  tools/nibbler_perf_test.py
3af88b22dcacd9dc3c6cd8c6c26613bb85ebcab93527a1a4fb8140943e55b177  tools/nibbler_lifecycle.py
5fcb1b8b45d824911d3d63e944faf982296f08a51195afbb2d1df433c8a95fee  tools/nibbler_lifecycle_test.py
```

Primary owns Beads and re-audit. A target neutral preflight and a single supervised
baseline are **separate later authorizations**; neither was performed here.

## `str-inn.1` reviewed offline lifecycle contract

The paused broad helper was replaced for this task only. The implementation did
not change Beads, production configuration/artifacts, native/frontend source,
prior failed evidence, agents or worktrees. No SSH, target API, GPU or model
requests were made for this fix. Native build/review work belongs to the primary's
separate helper; it does not authorize a calibration cycle.

### Before any unload

1. Validate the baseline config, root/on-host requirement, operator lock, private
   port collision check, host memory/swap thresholds, production idle/ready API
   response shapes, router/audio identities, hashes and model manifests.
2. Prepare **all** private config/binary/work directories before mutation. Freeze
   the reviewed runner and supervisor explicitly, even if the installed tools
   tree lacks them. Resolve launch tools outside the deliberately restricted child
   PATH; preserve the absolute venv Python path. Do not change system limits or
   PAM configuration to make a refusal disappear.
3. Launch a no-model probe through the exact `runuser` command, uid/gid, cwd and
   environment. Assert effective memlock **after PAM**; import Python/frontend
   dependencies; validate readable model/config/tokenizer/library paths,
   searchable directories, writable cwd/HOME/TMPDIR/log directory, executable
   copied binaries and `ldd` dependency resolution under the engine library env.
   The probe opens/checks files and parses the template, but neither runs native
   model code nor reads the expert payload. Refusals occur before unload intent.
4. Start the separate supervisor in its own session with detached standard I/O,
   establish Linux child-subreaper status and test real kernel pidfd support.
   Wait for its readiness acknowledgement and durable PID/start/boot identity.
   It retains the operator lock independently of the coordinator.
5. Immediately recheck idle state. The supervisor fsyncs unload intent and ACKs it;
   **only the supervisor-owned, journal-guarded callback sends the unload POST**.
   That callback rechecks identities/hashes, idle and memory. A stopped coordinator
   cannot resume a late unload after watchdog restoration. Empty/plaintext 2xx
   control bodies remain accepted; metrics and readiness retain strict JSON.

### Ownership, interruption and restoration

- Catch coordinator SIGTERM/INT/HUP. Also handle coordinator SIGKILL, transport
  EOF, malformed/lost RPC responses and a coordinator that remains stopped past
  the window deadline without relying on coordinator `finally`.
- The supervisor is the ancestor and Linux subreaper of every private launch.
  Before spawning it fsyncs spawn intent; a child blocks at an exec gate until its
  PID/start/boot identity is durably recorded. Lost spawn ACKs do not lose ownership.
  No threaded `preexec_fn`, frontend-only parent-death signal, or unverified PGID
  kill is used. Detached/double-fork descendants are adopted by the supervisor.
- Cleanup repeatedly censuses descendants and sends SIGKILL through pidfds with
  start-identity rechecks. Stat identity and parent relationship come from the
  same snapshot, with ancestor-identity checks. After reaping, the kernel's direct
  child list must be empty; `/proc` enumeration alone is not sufficient to close
  fork/reparent races. A surviving/unobservable tree blocks restoration after a
  ten-second reaper deadline rather than guessing PIDs or loading in parallel.
- The independent window deadline shares the original monitor's remaining budget;
  exhausted preflight refuses without unload. A model-control callback is bounded
  to 45 seconds, reaping to ten seconds, and each restoration callback to 720
  seconds, with at most two attempts. These are process/I/O bounds, not a hard
  real-time scheduling promise. A timed-out restoration is killed/reaped and
  reported blocked; it does not continue unlimited retries in the background.
- After the private tree is empty, the owner invokes `--restore-journal` under a
  trusted, armed journal in `restoring` phase. The living parent's PID/start/boot
  must equal the recorded supervisor. Restoration checks baseline identities and
  hashes before requesting the original model, retains the 10 GiB/1 GiB stop
  thresholds and 2 GiB empty-GPU gate for an unloaded model, and records the short
  original-model completion, both ready states, audio health, identities/hashes,
  memory and GPU state. A retry finding Qwen already ready does not wait for its
  GPU to become empty or unload it again. Re-entry uses separate sampler files.
- The journal ends `restored`, `unarmed` or `blocked`, retaining the trigger and
  attempt count even if the coordinator cannot write `result.json`. A supervisor
  deadline/error does not count as a passed measurement just because recovery
  succeeded. The previous three failed controls remain failed.

### Offline proof and limits

The process fixtures run on the coding VM with disposable Python children and a
local fake API whose state changes from ready to unloaded to restored. The fake
restorer checks that frontend/child/grandchild identities are gone **before** the
restore request. Coverage includes graceful TERM/INT/HUP, hard kill, SIGSTOP,
exited frontend plus detached descendants, an ignored TERM, ordinary close,
transport loss, unload taking effect despite a dropped API response, lost
readiness/spawn ACKs, malformed RPC, missing cwd/executable,
exec-gate EOF, reused PID identities (including a race after pidfd opening), an
unrelated live process, retained operator lock, deadline, successful retry, two
failed retries and a hanging recovery callback. Parser/measurement gates and
fresh-directory dry-run no-file/no-bytecode behavior remain covered.

Preflight fixtures use a fake `runuser`/PAM boundary with exact launch env checks,
small fake frontend modules and real ELF `/usr/bin/true` binaries. The successful
unlimited-limit case is **synthetic**. The finite-limit refusal sets `(0, 0)` in a
real disposable child while the mocked coordinator reports infinity; it proves
that parent infinity is not accepted as evidence. Missing launcher/Python/binary,
imports, config/path permissions, ELF libraries, unavailable memlock capability
and occupied port all refuse without unload. Actual exec-path finite limits and
wrong identity refuse before model/config access or frontend exec. Separate
restoration fakes exercise the real restoration function twice with the same
journal attempt and reject changed production identities before any load request.

**Unsupported / not established:** supervisor death or suspension, whole-session
or whole-cgroup killing that also reaches the independent supervisor, OOM,
host reboot, kernel/procfs/storage failure, unkillable kernel tasks, privileged
processes deliberately escaping the ownership tree, or unavailable/unsafe
production restoration dependencies. These scenarios can leave production down;
blocked evidence is not a recovery guarantee. A dead supervisor's journal is
**not** permission for blind manual cleanup: the guarded recovery entry refuses a
foreign/reused parent, and this task does not implement post-supervisor-death
adoption. Port-bind checks remain TOCTOU; a later bind failure is cleaned up by
the owner, not declared impossible. A quiet window is still required: this fix
cannot prevent an unrelated router client from autoloading production during a
private run. Target PAM, actual native/vision descendant behavior and real GPU
release remain untested until independently reviewed and explicitly authorized.

Historical `str-inn.1` freeze independently reviewed before the H1/M1 patch:

```text
62e6c85ffed5ee684d707c80a9f8ad7a7e12d4392b3b4cedde1127ca890e6198  tools/nibbler_perf.py
b1eb12683d4b02b969e20e455ebd3855db5a64ff4e0c3abff43357eca11cbb35  tools/nibbler_perf_test.py
30ab09b0745d881ea54e5eac571bc060775142dcc0f4bf243d23ab2c0a906b72  tools/nibbler_lifecycle.py
00a8112ddbef2f2dd483689aa2b4b6502890ab64dfc10dcb13b3ece778c30f98  tools/nibbler_lifecycle_test.py
```

Primary owns Beads completion and independent re-audit. All Blocker/High findings
must be independently rechecked against this freeze; only then may primary give
fresh explicit authorization for **one** deployed-engine control. No live trial
or candidate transfer is authorized by these offline results.

## Approved experiment design, pending new go-ahead

After safety review and a fresh explicit authorization:

1. Verify all launch dependencies, process limits, production idle counters,
   rollback and unrelated services **before** unload. Run the existing deployed
   engine control first, same GPTQ pack/features/config.
2. At 11 workers and 2048 reserve, sweep 0.10 / 0.25 / 0.40 / 0.55, alternating
   order and repeated controls. Default split control retains the startup link
   probe; explicit splits skip that native probe and must be labelled as such.
   Alternate solo/dual order and short/long, fresh/exact-warm prompts. Require
   measured overlapping decode, not merely two submitted HTTP requests. Record
   warm prefix reuse and adaptive cache state. Word counts only generate input;
   actual rendered token counts govern context claims.
3. Choose a split only from matched valid throughput/TTFT/correctness trials;
   then compare 11 / 9 / 7 (possibly 5) workers at that split. No combinatorial
   matrix. Keep pool task count and prefetch settings unchanged. Audio health
   alone is not a latency/responsiveness measurement.
4. Reserve stays 2048 until full-context dual-MTP, 1024-token GPU vision and
   pressure/park/restore/handoff gates establish peak margin for other services.
   Only a new cautious approved phase can test 1536, then 1024. Failed gates or
   sampler stops prohibit proceeding downward. Startup free-VRAM is not peak
   headroom and hit rate alone can be inflated by raising PCIe share.
5. Compare native token IDs/logits separately from asynchronous production-like
   speed trials; identical answer-text hashes alone do not prove that contract.
   End every window with original-model short completion, `/running`, unchanged
   router/audio identities, audio health and artifact hashes.

## Earlier candidate source/build plan — separate from `str-inn.1`

The earlier measurement report recorded frozen loader/reporting source and paused
transfer. Subsequent primary-authorized native staging/build work belongs to its
separate helper and is not audited by this lifecycle task. The earlier plan below
is retained as context, **not authorization for a live candidate/model run**.
After timing and a new go-ahead, prepare a **sanitized source copy, not a git worktree** at
`/data/llm/Strata-tests/perf-audit-20261011/candidate-src` with source manifest.
Include `src`, `include`, `cmake`, `CMakeLists.txt`, required `tests`, `tools` and
`serve`; omit `.git`, `.beads`, model files, logs, credentials, `.venv` and build
outputs. Use existing `/data/llm/Strata/third_party/llama.cpp` read-only rather than
fetching/downloading. No compilation during timed trials.

Use a fresh build directory, CUDA 13.3 / GCC 15, Release, arch120,
`STRATA_NATIVE_EXPERTS=ON`, `STRATA_BUILD_TESTS=ON`,
`STRATA_BUILD_CONVERSATION_TESTS=ON`, `STRATA_BUILD_ARENA_LOAD_TESTS=ON`, then
`cmake --build ... --target strata linux_arena_load_test linux_arena_parity -j2`.
See [Linux arena loader](LINUX_ARENA_LOAD.md) and
[batch-fetch assessment](BATCH_FETCH_ASSESSMENT.md) for exact fixture contracts.

Inspect HIP/SYCL compiler availability; record unavailable backends as skipped,
not passed. Do not install multi-GB toolchains on production. Target native arena
parity requires actual registration and direct I/O, not fallback. Full pack
geometry must be verified before using the snapshot 67947724800 bytes / 48 layers.
A single arena may be reused sequentially for buffered/direct per-layer and whole
hash parity, never two simultaneous 63 GiB arenas. Both private correctness
server paths must use `STRATA_VERIFY_ARENA=1`; only private
`STRATA_UNBUFFERED_LOAD` changes for loader A/B. Production stays buffered/default.
Naturally first/warm OS-cache states are observed, never manufactured with drops.

The three failed controls ran none of native arena parity, full model gates,
token/logit parity or loader A/B. Separate source builds do not satisfy those
measurement gates. These remain for primary review; no candidate install or
configuration recommendation follows from this lifecycle fix.
