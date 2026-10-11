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

## `str-inn.3` authorized neutral target outcome — 2026-10-11 22:29:39 UTC

**One neutral preflight passed. No benchmark, control or private model window ran.**
This timestamped outcome supersedes earlier “not run on target” wording for the
neutral mode only; **live baseline remains paused** and needs separate fresh
primary authorization plus human presence/manual recovery readiness.

Primary separately authorized exactly one root-on-nibbler neutral invocation of
commit `fa8c13f099cf3b4db47f150b8011150d2c05fbb1`. This owner's homelab mandatory
`scripts/agent-preflight.sh` passed all required checks. History inspection confirmed no unknown target dispatch before the
provider-disconnect resumption. No second neutral invocation was made.

Only the frozen runner/lifecycle module were staged in a new root-owned test leaf
(created mode 0700, files uid 0 / mode 0444), not candidate source/build or installed
production tools. Their exact SHA-256 was verified after write and again before
execution:

```text
1852695e3e6d962567d24264f51c053a37836b63eefc9d86357452c5141041b3  nibbler_perf.py
3af88b22dcacd9dc3c6cd8c6c26613bb85ebcab93527a1a4fb8140943e55b177  nibbler_lifecycle.py
```

Exact root command, with a ten-minute budget:

```sh
/usr/bin/python3 /data/llm/Strata-tests/perf-audit-20261011/neutral-preflight-20261011T222638Z-tools/nibbler_perf.py \
  --run --preflight-only --fractions baseline --repeats 1 \
  --tag neutral-preflight-20261011T222638Z --window-minutes 10
```

Staging/execution used existing homelab `RemoteShell.exec_stdin`, hostname
`nibbler.ts.keen.land`, strict host keys, `retries: 0`, no raw-SSH fallback.
The exclusive invocation marker and execution receipt each record one invocation.
The command returned **0**, `mode: preflight-only`, **`ready: true`, `passed: false`,
`runs: []`**, and `production_mutations: none`. The prepared arm remained baseline
split, 11 workers, 2048 MiB reserve, port 18089. The same second-resolution start
and outcome timestamp is not a loader-time or performance result.

### Target launch facts and unchanged production

- Actual post-runuser/PAM child **uid 996 / gid 979**, matching native-inference.
  Effective soft/hard memlock **`[-1, -1]` (unlimited)**; coordinator also unlimited.
  No PAM/limit-policy change was made to obtain this result.
- Absolute launcher `/usr/sbin/runuser`, dependency probe `/usr/bin/ldd`, venv
  Python `/data/llm/Strata/.venv/bin/python`; `dependencies_resolved: true`.
  Kernel direct-child-list support was true in coordinator and post-PAM child.
  Prepared config hash:
  `e40b838be023cae1c69016fa52332e8556baad711cd67a7c864269c1575a79e4`.
- GET `/running`: Qwen and pocket-tts both **ready**, before and after. Audio GET
  `/health`: **`status: ok`, `backend: cpu`**, before and after. Neutral idle GETs
  verified two idle slots only after the early already-ready gate.
- Unchanged router PID/start **1435174 / 18877666**; audio
  **1643128 / 54686907**; installed production frontend **2880993 / 65583967**.
- Only GPU compute consumers before and after were installed vision
  **2881000 / 65583992** and engine **2881053 / 65584100**, using reported
  1268/12560 MiB respectively. No new private frontend/native/vision or GPU
  consumer was found. Total GPU used/free was 13981/1909 MiB in both snapshots.
- MemAvailable **23388459008 → 23332052992 bytes** (21.78 → 21.73 GiB), SwapFree
  **8785260544 → 8787292160 bytes** (8.18 GiB). Snapshots are not peak headroom.
- All six installed engine/vision/BUILD/tracked-config/runtime-config/shared-settings
  hashes were identical before/after; exact values are in the bounded operator
  snapshots and evidence README. Shared-settings contents/config env, production
  logs, process arguments and binary copies were not exported into results.
- All nine operator verification checks were true. No supervisor/sampler,
  private frontend/native model, GPU fixture, completion, unload/reload/control
  POST or restoration was invoked. No corresponding start/unload/recovery
  artifacts exist. Frozen tool code, production config, services, PAM and global
  memory policies were left unchanged.

### Bounded proof retrieval and its local warning caveat

[Neutral evidence](../bench/results/2026-10-11-nibbler-calibration/neutral-preflight-20261011T222638Z/)
contains the complete captured `before.json`, `result.json`, arm proof, operator
before/after/verification, exclusive invocation marker and execution receipt.
All **eight captured** host-file responses from existing `hl peek path_read` were
**`truncated: false`**, with local SHA verified against execution-receipt hashes
where available. `retrieval.json` preserves exact commands, paths, bytes and hashes;
`probe-audit.json` preserves only this task's local audit entries. Old failed
control artifacts were not changed.

The unmodified hl CLI used an invocation-local RemoteShell constructor adapter to
force strict host keys and zero retries (its library defaults do not enforce
these). Representative retrieval, not another preflight invocation:

```sh
cd /home/pete/devel/homelab
RUBYOPT=-W0 bundle exec ruby /tmp/str-inn.3-peek-strict.rb peek host nibbler path read \
  /data/llm/Strata-tests/perf-audit-20261011/neutral-preflight-20261011T222638Z/result.json
```

The first `stage-manifest.json` probe succeeded (`status: 0`, `error: null`,
141 ms, 22:29:40 UTC), but the initial local collector discarded stdout when
Ruby/Bundler compatibility warnings appeared on stderr. That successful path was
**not re-read**, and its content/truncated field is not claimed as retained.
Local audit established this was not a target transport failure. The remaining
**eight distinct** authorized reads completed with local Ruby warning suppression
only; nine task-scoped path probes all had status 0 / no error. No target
transport/probe retry, raw-SSH fallback, frozen-code/host repair, second preflight
or additional model operation occurred.

Key captured evidence SHA-256:

```text
6a9003e065900ed1dc862e23bbba446e7ddb79264508bec5a5cda890289c6fc3  result.json
3d07582e742762c36712e9926cda0442332e90dd42968f3687f229b360a2d9ae  arm-00-preflight.json
7b831af4452c6136f0c2dfd48c0d8ab1526f7c3cc73595769c612406332ad7b8  operator-before.json
2b69b8c9c52527eef22a74e816020d35bdb546539d585c28b65bede1df07c5d6  operator-after.json
750f7b178f050153ef6d1612fb2101886dd270f4822b677259c609a2dee662c7  operator-verification.json
```

**Limits:** this establishes target neutral launch readiness for this frozen
invocation, not actual private native/vision descendant recovery, private GPU
release, arena registration, loader timing, token/logit parity, full-context
pressure/park/handoff or performance. Published manifest identity is not a fresh
63 GiB payload verification. Previous failed controls remain failed; no tuning
claim or production-setting recommendation follows. Primary owns Beads and any
separate future baseline authorization.

## STAGE 1 authorized supervised baseline — 2026-10-11 22:55:05 UTC

**Exactly one installed-engine control passed and restored production.** This
new dated result supersedes earlier “no valid baseline / live baseline paused”
wording for control-4 only. The sweep, worker/reserve changes, candidate loader,
GPU parity fixtures and all further stages remain paused pending separate primary
review and authorization. Keep production **0.25 / 11 workers / 2048 MiB reserve**;
this one control is not a tuning conclusion. Old failed controls are unchanged.

Pete authorized the live Qwen unload and was present for manual recovery. Only
STAGE 1 was authorized; this owner remained the sole target operator. Homelab
`scripts/agent-preflight.sh` passed. Local HEAD was
`fb13c698a75a0b5657dc713d541e298f8cbdaf32`; all four frozen tool hashes still matched
the `str-inn.2` freeze above. No frozen code was patched.

The two reviewed modules were staged under a fresh root-owned mode-0700 leaf,
files uid 0/mode 0444, using existing homelab `RemoteShell.exec_stdin` with strict
host keys and zero retries. SHA was checked after write and immediately before
execution. Exact root command on nibbler:

```sh
/usr/bin/python3 /data/llm/Strata-tests/perf-audit-20261011/baseline-control-4-20261011T225505Z-tools/nibbler_perf.py \
  --run --fractions baseline --repeats 1 \
  --tag baseline-control-4-20261011T225505Z --window-minutes 30
```

The exclusive invocation marker records **one invocation**. Defaults remained
11 workers, 2048 reserve, port 18089, max-tokens 96 and long-words 8192. No
`--reserve-gates`, lower reserve, other fraction, candidate or GPU fixture ran.
The wrapper launched the coordinator in its own session with DEVNULL stdin and
private-file stdout/stderr; no timeout or group-kill wrapper surrounded it. The
reviewed supervisor independently detached in its own session. Coordinator
PID/start/session was **3172811 / 66200737 / 3172811**; supervisor PID/start was
**3172842 / 66200771**. Final result/restoration file mtimes were **22:56:53 UTC**;
operator verification ran at **23:00:41 UTC**. Detached coordinator exit status
was not captured; its exit, final `passed: true`, `restored: true`, and journal
phase **restored** were confirmed. No numeric exit status is claimed.

### Actual launch, normal cleanup and restoration proof

Before launch, GET `/running` showed Qwen and pocket-tts ready; audio health was
ok/CPU. Two slots were idle, queued/running/waiting/outside_slots all zero. The
frozen coordinator and owner mutation callback repeat idle checks immediately
before unload. Prepared and actual pre-exec post-PAM proofs both recorded uid
**996**, gid **979**, effective soft/hard memlock **[-1, -1]**, resolved ELF
libraries and supported kernel child lists, with identical private config SHA
`6a61ad85ae9a6427e26faa692a8bdc2307fc0bf13e4e1509b7f67f10af2cf805`.

The owner journal ended `restored`, `children: []`, restoration attempt **1**.
Its trigger `EOFError: coordinator transport disconnected` is the expected normal
finish socket close, **not an injected transport failure**. The verified source
requires its kernel empty-child witness before restoration. Both coordinator and
supervisor exited; no private processes/GPU consumers remained and private port
18089 was closed. All **15** operator post-window checks passed.

| Process | Before PID / start ticks | Restored PID / start ticks |
| --- | --- | --- |
| Router | 1435174 / 18877666 | 1435174 / 18877666 |
| Audio | 1643128 / 54686907 | 1643128 / 54686907 |
| Qwen frontend | 2880993 / 65583967 | 3176081 / 66208703 |
| Qwen vision | 2881000 / 65583992 | 3176084 / 66208726 |
| Qwen engine | 2881053 / 65584100 | 3176141 / 66208830 |

After reload, Qwen and pocket-tts were ready and audio was ok/CPU. The frontend
uses the original installed runtime config and working directory. All six
production hashes were unchanged; actual `/proc/<PID>/exe` hashes of both new
native processes match the originals. Published pack-manifest/tokenizer/template
identity also matched; no fresh 63 GiB payload verification was performed. Only
installed vision/engine were GPU consumers, uid 996, 1268/12538 MiB respectively.
No router/audio restart, manual cleanup, global memory policy/clock/cache change,
installation or production config/binary edit occurred.

The original-model health completion finished `stop`, five output tokens, ending
with **CALIBRATION READY**. Restoration health-request wall time was
**28.171366464928724 s**, including reload and router loading banners. Its first
banner is not model TTFT, and its timings are not benchmark measurements.

### Descriptive control numbers — n=1 window, twelve requests

Measured on nibbler's installed engine **0.1.41**, unchanged published GPTQ NVFP4
pack identity, 9900X / RTX 5060 Ti hardware documented in the performance audit.
Private launch-to-metrics-ready was **27.02275056799408 s**. OS cache was not
flushed: an observed first private load, not proven cold. Native startup recorded
**cudaHostRegister PORTABLE ok**, 64800 MiB arena, buffered loader, unavailable
MAP_HUGETLB and preexisting THP always state. The retained startup PCIe probe
reported **28.9 GB/s host→device** (best of 28.9/28.9/28.8/28.8), selecting the
unchanged effective **pcie_frac 0.25**. This startup probe is not sustained
inference transfer bandwidth.

Actual rendered prompt lengths were **178** and **8243** tokens, not the input
word counts. Every HTTP request returned **96** completion tokens. Aggregate wall
throughput includes prefill/admission/queue time; API decode TPS is a separate
native-reported field. Dual overlap is the intersection of first-to-last output
intervals, not a guarantee of continuously simultaneous GPU work.

| Prompt tokens | Streams | State | Aggregate wall tok/s | Client TTFT s, slot order | API decode tok/s, slot order | Dual output overlap s |
| --- | --- | --- | --- | --- | --- | --- |
| 178 | 1 | fresh | 26.71 | 1.314 | 41.2 | — |
| 178 | 1 | warm | 52.21 | 0.189 | 56.9 | — |
| 178 | 2 | fresh | 33.49 | 2.714 / 1.287 | 32.7 / 22.5 | 2.926 |
| 178 | 2 | warm | 44.44 | 0.566 / 0.170 | 26.2 / 24.2 | 3.665 |
| 8243 | 1 | fresh | 11.35 | 6.937 | 61.6 | — |
| 8243 | 1 | warm | 53.83 | 0.209 | 59.5 | — |
| 8243 | 2 | fresh | 10.39 | 16.133 / 7.045 | 61.6 / 9.3 | 1.333 |
| 8243 | 2 | warm | 42.77 | 0.616 / 0.227 | 25.5 / 23.5 | 3.787 |

Request nearest-rank descriptive p95 TTFT/wall was **16.133391922921874 /
18.473981243907474 s** across these twelve requests, not a service percentile or
confidence bound. Reported native/API expert-cache hits ranged **30.0–68.0%**,
PCIe-routed share **7.8–15.0%**; configured PCIe fraction is a different quantity.
Warm repeats reused **171 / 8236** prompt tokens; cache counters reported six of
twelve requests reused, four parked entries, thirteen parks and four evictions.

**Counter/path caveats:** native timings can cover the last decode segment,
not the whole HTTP request: long dual-fresh has one `engine_generated: 57` versus
96 output tokens and includes admission/prefill pauses. Warm dual logs
`TARGET_ONLY` MTP suppression, zero drafts offered and incoherent fallbacks;
it is not the same effective MTP path as fresh dual. Preserved launch args request
`--spec 4`, while initial engine metadata reports `spec: 6`, `mtp_max: 4`.
These are recorded observations, not regression or tuning conclusions. Integer
presence/overlap checks do not prove native token/logit parity, full-context,
vision or pressure/park/handoff gates. No patch or rerun followed these findings.

At approximately one-second sampling, the private window's **78** samples had
minimum MemAvailable **22476390400 bytes (20.93 GiB)** and SwapFree
**8786137088 bytes (8.18 GiB)**. Restoration's **28** samples had minimum
MemAvailable **25060810752 bytes (23.34 GiB)** and SwapFree
**8078467072 bytes (7.52 GiB)**. Private sampled peak GPU used / minimum free
was **14005 / 1885 MiB**. No built-in stop fired. These are sampled observations,
not sub-second peaks or permission to lower the 2048 reserve. Audio latency,
whole-fleet performance and fault-injected native recovery remain unmeasured.

### Bounded evidence and unchanged freeze

[Control-4 evidence](../bench/results/2026-10-11-nibbler-calibration/baseline-control-4-20261011T225505Z/)
contains result, final journal, samples, restoration, sanitized before, operator
before/after/verification, both launch proofs, measurements and bounded scrubbed
private synthetic log tails. **24 target-derived files** were captured with
**34 successful existing hl `path_read` responses**, all **`truncated: false`**,
strict keys and zero retries; no fallback or target retry occurred. Result and
arm measurements exceeded the single-probe limit, so six additive base64 export
chunks per file (24576 raw bytes per full chunk) were read and reconstructed.
Whole original byte counts and SHA-256 matched; no original artifact was edited.
The sanitized before export drops production slot/engine payloads and extra
audio fields, retaining original-source hash provenance. Log exports are at most
32768 source bytes / 160 lines; no production log/config env/shared-settings body
or binary was exported. `retrieval.json` preserves exact commands and per-response
hashes; `SHA256SUMS` covers all local evidence files except itself.

Key exact evidence SHA-256:

```text
496ca0979bfde0cb151325c576b72930cd60831c45f24e4c9b4aefe17b9d04ac  result.json
2ad3d0102deec501fbbe13fb5d0227e23ff8925c4d8a666cd085ea32ce1b93b3  lifecycle.json
8d6a21a26b4e3baefff6bcb95477ed025f2b4b2fbe58871dfe729465a80e42ca  restoration.json
0e50dab184ac77f26ea3682089047178e742c12ffccd6a8dd966f44e648e8c21  operator-verification.json
```

All four frozen tool hashes remain unchanged. Only this append-only doc section
and the fresh evidence leaf changed in the shared checkout; no Beads/git mutations,
agents or worktrees were made. **STAGE 1 is complete. Do not start the sweep or
any further model window without separate primary authorization.**

## STAGE 2 authorized split sweep — stopped, 2026-10-11 23:26:07 UTC

**One sweep invocation failed at the second arm's pre-exec port check; production
was restored. `passed: false`, `restored: true`. No patch, rerun or further stage.**
This dated outcome records the separately authorized STAGE 2 after primary
committed STAGE 1 at `2139033df3d89ffe461aff91f6400fe32803b1c7`. Pete's live-window
authorization and manual recovery presence continued. This owner remained sole
target operator; no manual recovery was needed.

Homelab agent preflight passed; initial checkout was clean and all four frozen
tool hashes matched. Local side-effect-free dry run confirmed default eight-arm
order, two repeats, 11 workers, 2048 reserve, port 18089, max-tokens 96 and
long-words 8192. Conservative estimate `8 × 2 × 108 s = 1728 s = 28.8 min` was
below the requested 45-minute cutoff, so repeats 2 remained the default. This
was an estimate, not a runtime guarantee; the frozen independent 50-minute stop
remained authoritative.

Fresh root-owned staging leaf mode 0700, module files uid 0/mode 0444, transferred
via existing homelab RemoteShell with strict host keys and zero retries. SHA was
verified after write and immediately before execution. Exact root command:

```sh
/usr/bin/python3 /data/llm/Strata-tests/perf-audit-20261011/split-sweep-20261011T232607Z-tools/nibbler_perf.py \
  --run --tag split-sweep-20261011T232607Z --window-minutes 50
```

One exclusive invocation marker. Coordinator PID/start/session
**3268495 / 66386919 / 3268495** used its own session, DEVNULL stdin and private
file stdout/stderr; no timeout/group-kill wrapper. Reviewed supervisor separately
detached, PID/start **3268685 / 66387089**. Launch was **23:26:07 UTC**; first
sampler stop **23:28:17.464086 UTC**, phase **run-01-load**. Final result/restoration
mtimes were **23:28:46 UTC**; independent recovery proof **23:29:41 UTC**. Both
processes exited; detached coordinator numeric exit status was not retained.

### Exact refusal and arm coverage

Baseline arm 00 completed **16 groups / 24 requests**, both repeats. Arm 01's
0.10 Python launcher then refused in `launch_check` **before frontend exec,
native model execution or GPU allocation**:

```text
nibbler_perf.py line 440: s.bind(("127.0.0.1", cfg["port"]))
OSError: [Errno 98] Address already in use
```

The generic stop reason `allocation/native error in server.log` was the monitor's
Traceback gate; it is not evidence of a CUDA allocation failure. Exact trace is
in the captured arm-01 server tail, and sampler-stop records run-01-load. No
port-holder/TCP-state snapshot was captured; do not claim a live listener or
TIME_WAIT as established cause. No port clearing, code/host patch or retry was
attempted. Initial `runs: 1` status was refined by this attempted-launch evidence:
no explicit-fraction **frontend/native model/measurement** ran, but the arm-01
pre-exec launcher did. All eight neutral preparation proofs exist, uid 996/gid
979, unlimited effective memlock and kernel child lists; preparation is not model
execution or performance success.

| Arm | Fraction label | Execution / measurements | Load s | GPU peak MiB | Client TTFT p95 s | API decode tok/s range | API hit % / PCIe share % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | baseline, probed effective 0.25 | 24 requests recorded; baseline arm complete inside failed sweep | 27.023 | 14043 | 16.049 | 9.2–62.0 | 34.4–67.5 / 6.1–13.3, latest 12 only |
| 01 | explicit 0.10 | pre-exec bind refused; no frontend/native model/measurements | — | — | — | — | — |
| 02 | explicit 0.40 | not started; preparation only | — | — | — | — | — |
| 03 | explicit 0.55 | not started; preparation only | — | — | — | — | — |
| 04 | explicit 0.55 | not started; preparation only | — | — | — | — | — |
| 05 | explicit 0.40 | not started; preparation only | — | — | — | — | — |
| 06 | explicit 0.10 | not started; preparation only | — | — | — | — | — |
| 07 | explicit 0.25 | not started; preparation only | — | — | — | — | — |

### Retained baseline numbers and observed within-arm variation

Descriptive data from the **failed window**, not a successful split comparison.
Measured on nibbler's installed 0.1.41 engine, unchanged published GPTQ NVFP4 pack
identity, 9900X / RTX 5060 Ti. Buffered loader, 64800 MiB arena,
`cudaHostRegister PORTABLE ok`; observed cache state, not proven cold. Retained
native startup PCIe probe reported **28.9 GB/s host→device** (best of
28.9/28.8/28.9/28.8), effective fraction **0.25**. No worker/reserve, candidate
loader/binary, reserve-gate, parity-fixture, clock/affinity/cache policy change.

Within each repeat, short then long inputs, fresh then exact warm repeat. Repeat
0 runs solo before dual; repeat 1 reverses that order. The same word/stream/state
buckets below pair those two different order/cache histories. Actual rendered
prompts **178 / 8243 tokens**, 96 output tokens each. Slash separates requests;
semicolon separates repeat 0 from repeat 1. Mean averages two group wall rates,
not a pooled token/time rate. Repeat difference is absolute wall-rate difference
and percent of their mean; these are descriptive ranges, not confidence bounds.

| Prompt tokens | Streams | State | Wall tok/s r0 / r1 | Mean wall tok/s | Repeat difference tok/s (% mean) | Client TTFT s r0 ; r1 | API decode tok/s r0 ; r1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 178 | 1 | fresh | 27.39 / 34.92 | 31.15 | 7.528 (24.16%) | 1.297 ; 1.230 | 42.6 ; 61.7 |
| 178 | 1 | warm | 52.01 / 56.48 | 54.25 | 4.465 (8.23%) | 0.191 ; 0.190 | 56.7 ; 62.0 |
| 178 | 2 | fresh | 33.74 / 33.81 | 33.77 | 0.071 (0.21%) | 1.289 / 2.688 ; 2.640 / 1.234 | 22.8 / 32.8 ; 32.2 / 22.3 |
| 178 | 2 | warm | 44.29 / 45.30 | 44.80 | 1.010 (2.25%) | 0.170 / 0.564 ; 0.153 / 0.486 | 24.1 / 26.1 ; 36.7 / 39.0 |
| 8243 | 1 | fresh | 11.31 / 11.32 | 11.31 | 0.011 (0.10%) | 6.957 ; 6.956 | 61.2 ; 61.6 |
| 8243 | 1 | warm | 55.08 / 54.65 | 54.86 | 0.427 (0.78%) | 0.211 ; 0.199 | 61.2 ; 60.2 |
| 8243 | 2 | fresh | 10.39 / 10.38 | 10.38 | 0.014 (0.14%) | 16.101 / 7.025 ; 6.978 / 16.049 | 59.7 / 9.3 ; 9.2 / 58.5 |
| 8243 | 2 | warm | 42.38 / 42.37 | 42.38 | 0.004 (0.01%) | 0.212 / 0.603 ; 0.206 / 0.602 | 36.2 / 37.1 ; 23.2 / 25.0 |

Across 24 requests, nearest-rank descriptive p95 TTFT/wall was
**16.04890423209872 / 18.476183643913828 s**. API/native hit/PCIe ranges in the
arm table cover **latest twelve requests, repeat 1 only**, not all 24. Native
counters can cover last segments, and warm dual target-only paths/variable cache
state remain caveated as in STAGE 1. API decode TPS is not full-request wall TPS.
Warm repeats reused 171/8236 prompt tokens; cache snapshot records twelve reused
requests, twenty-seven parks, fourteen evictions and two restores. These counters
do not establish canonical full-context park/restore/handoff gates.

**Which split differences exceed repeat variation: not assessable. Which do not:
also not assessable, not evidence of equivalence.** Planned repeated-arm pairs
01/06 (0.10), 02/05 (0.40), 03/04 (0.55), and probed/explicit 0.25 endpoints did
not produce paired model measurements. The only observed repeat variation is
inside baseline arm 00, **0.004–7.528 wall tok/s** by bucket, with reversed
solo/dual order and changing cache state. Do not use it to manufacture a split
ranking, statistical claim or production recommendation.

Main window's 126 approximately one-second samples had minimum MemAvailable
**22932238336 bytes (21.36 GiB)**, SwapFree **8145944576 bytes (7.59 GiB)**,
GPU peak/minimum free **14043 / 1847 MiB**. Restoration's 28 samples had minimum
MemAvailable **24910196736 bytes (23.20 GiB)** and SwapFree
**7619219456 bytes (7.10 GiB)**. Resource thresholds did not fire; the launcher
traceback did. Samples are not sub-second peaks, full-context/vision safety,
audio responsiveness or token/logit parity proof.

### Original production restored, failed sweep preserved

Final journal **restored**, `children: []`, restoration attempt **1**. All
**thirteen production-restoration checks passed**. Two sweep completion/success
checks failed as expected: not all eight arms ran and `passed` is false.
`operator-verification.json` therefore retains **verified: false**, not a false
all-clear for the failed experiment. Its expected-schedule check includes eight
completed runs; the actual CLI/default plan was correct, not misconfigured.

Qwen and pocket-tts were ready; audio ok/CPU. Router/audio identities and six
production hashes were unchanged, and actual new native executable hashes match
the originals. Restored frontend uses original installed cwd/runtime config.
Published manifest/tokenizer/template identity matches, not a new payload hash.
No private processes/GPU consumers remain; port 18089 is closed.

| Process | Before PID / start ticks | Restored PID / start ticks |
| --- | --- | --- |
| Router | 1435174 / 18877666 | 1435174 / 18877666 |
| Audio | 1643128 / 54686907 | 1643128 / 54686907 |
| Qwen frontend | 3176081 / 66208703 | 3275561 / 66399925 |
| Qwen vision | 3176084 / 66208726 | 3275568 / 66399950 |
| Qwen engine | 3176141 / 66208830 | 3275635 / 66400061 |

Only installed vision/engine were post-window GPU consumers, uid 996,
1268/12538 MiB. Health completion ended CALIBRATION READY, finish stop, five
output tokens; restoration health wall time **28.142920011887327 s**, including
reload/loading banners, not model TTFT or benchmark data. No manual cleanup or
live model operation outside the single approved invocation/built-in recovery.

### Evidence and freeze

[Stopped split-sweep evidence](../bench/results/2026-10-11-nibbler-calibration/split-sweep-20261011T232607Z/)
contains **33 target-derived files**, captured in **53 successful existing hl
path_read responses**, all **truncated: false**, strict keys, retries 0. The
receipt helper's planned-count display had a redundant +1; actual recorded probe
count is 53, not 54. Result/arm measurements were reconstructed from 12/10
additive host base64 chunks (24576 raw bytes per full chunk); whole original
byte counts/SHA matched. No original runner artifact was edited. Before is an
allowlisted sanitized export retaining original SHA provenance; private synthetic
logs are at most 32768 source bytes / 160 lines, credential-like lines scrubbed.
No production log/config env/shared-settings body or binary exported. Exact
paths/commands/hashes are in retrieval.json; SHA256SUMS covers local evidence.

```text
d1c7a0af963bc20195780f24b2d18106919981f0339effd2519115526bb3fee6  result.json
6684d94affe8958e90ef81c8a9863f2b7c65f82d3eca62eb5ba8ae68fe6de586  lifecycle.json
faa79e3046cd25e2afe73537c6fa48c8c71cc5a1382ca68959cd7d43bf502244  restoration.json
a8b262b4e249440810af51259d8cb22df1ca6caab58ccd4ec04b5a1dfe0d1691  arm-01-server.log.tail.txt
```

Local parsed-evidence assertions passed, including failed/restored flags, baseline
24 outputs/16 groups, positive dual overlaps, all eight neutral memlock proofs
and the thirteen production checks. All frozen tool hashes are unchanged; only
this append and the fresh evidence leaf changed. No code/host patch, rerun,
Beads/git mutation, agents or worktrees. **Stopped after the first refusal.
Production stays 0.25 / 11 workers / 2048 reserve. No further stage is authorized
by this report; primary decides any future action.**
