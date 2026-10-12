# Stage 2 rerun — measurements complete, recovery BLOCKED

**Overall window FAILED. No retry, repair, manual cleanup or further stage.**
The eight arms completed, but automatic recovery refused the SwapFree safety
floor. Raw `result.json` says `passed: true`, `restored: null` and records a
restoration error. Do not treat the raw measurement flag as window success.

## Authorization and invocation

- HEAD `b876248c20ed97210cba353787cb48b419d97bc8`, clean before the invocation.
- User-authorized exactly one rerun after accepted port-probe fix `fb31fba6`.
- Required staged SHA256 values verified on receipt and immediately before exec:
  - `nibbler_perf.py`: `4b85a3adfee6e341831a9d4ff7a7f015e0e5f1a9a2c6a5c69d2e0f563f400b12`
  - `nibbler_lifecycle.py`: `3af88b22dcacd9dc3c6cd8c6c26613bb85ebcab93527a1a4fb8140943e55b177`
- Fresh tools dir `/data/llm/Strata-tests/perf-audit-20261011/split-sweep-20261011T235149Z-tools`.
- Fresh output dir `/data/llm/Strata-tests/perf-audit-20261011/split-sweep-20261011T235149Z`.
- Dry plan validated. Conservative estimate 28.8 minutes < 45; repeats 2 retained.
- Qwen/pocket-tts ready, audio ok/CPU, two idle Qwen slots, no private orphan and
  `ss -H -ltnp 'sport = :18089'` empty before launch. Reusable bind probe passed.
- Marker and receipt: one detached coordinator invocation, uid 0. Supervisor
  separately detached; both sessions observed alive during the window.

```sh
/usr/bin/python3 /data/llm/Strata-tests/perf-audit-20261011/split-sweep-20261011T235149Z-tools/nibbler_perf.py --run --tag split-sweep-20261011T235149Z --window-minutes 50
```

Launch 2026-10-11 23:51:49 UTC; terminal result/journal mtime 2026-10-12
00:09:10 UTC, approximately 17m21s. Sequence:
`baseline / .10 / .40 / .55 / .55 / .40 / .10 / .25`, workers 11, reserve 2048.
No reserve gates or experimental loader. Installed 0.1.41 GPTQ engine on nibbler
9900X / RTX 5060 Ti. All eight arms: 16 groups, 24 requests, outputs 96 tokens;
128 groups / 192 private requests total. Every dual group had output overlap.
All eight post-PAM preflight and exec proofs show uid/gid 996/979, memlock
`[-1,-1]`, kernel child-list support. Baseline probe 28.9 GB/s -> effective .25;
explicit arms skip the probe and report their requested fraction.

## Recovery error and independent current-state verification

- Journal `phase: blocked`, `restore_attempt: 2`, `children: []`.
- Error: `LifecycleError: bounded original-model restoration attempts failed`.
- First restore monitor stopped at 00:08:53.558 UTC, `production-restore-load`,
  `SwapFree < 1 GiB`, recording 1,000,034,304 bytes free swap.
- Final `restoration-error.json`: `SwapFree < 1 GiB`, free swap 4,001,792 bytes
  (~3.82 MiB). There is no successful `restoration.json`; no restoration health
  completion text, token count or wall time is claimed.
- Private benchmark samples stayed above resource floors: min MemAvailable
  21,015,420,928 bytes (19.57 GiB), min SwapFree 2,598,244,352 (2.42 GiB), GPU
  peak/minimum free 14045/1845 MiB. Restore monitor has 12 successful samples;
  the failing sample is separately in its sampler-stop file, not its JSONL.
- At independent snapshot **00:11:01 UTC**: Qwen/pocket-tts ready, audio ok/CPU;
  six hashes and router/audio identities unchanged; original frontend cwd/config
  and actual installed native binary hashes match. No private processes/GPU
  consumers, port 18089 closed; coordinator and supervisor exited.
- But free swap was only 114,180,096 bytes (~109 MiB), below the safety floor;
  MemAvailable 32,729,063,424 (30.48 GiB). **Readiness is not a safety all-clear.**
- Installed GPU consumers (uid 996): vision PID 3396020 / start ticks 66642307,
  1268 MiB; engine PID 3396048 / start ticks 66642413, 12538 MiB. Restored
  frontend PID 3396017 / start ticks 66642283. No surviving private identities.
- Operator verifier: 13/15 predicates true, recovery predicates false,
  `verified: false`. Do not relabel the blocked journal as restored.

The journal's generic EOF coordinator-trigger text is retained; it is not proof
of an SSH transport failure. Error evidence identifies the SwapFree refusal.
No per-process attribution establishes why swap declined, and no new allocation,
OOM, arena parity, audio latency or production-performance claim is made.
Pete was immediately warned of the blocked recovery and low-swap snapshots.
No live action followed beyond read-only verification and evidence retrieval.

## Results and limits

[RESULTS.md](RESULTS.md) has the full per-arm paired wall/TTFT/API-decode table,
native/API hit/PCIe ranges, cohort versus baseline repeat-band comparisons, and
between-reload drift. [metrics-summary.json](metrics-summary.json) retains
unrounded derived values, actual failed status and source-result SHA256.

Each split is compared to first-arm startup-probed .25, not pooled with last-arm
explicit .25. Mean differences are compared to (1) the larger maximum within-arm
r0/r1 difference and (2) the larger full observed range across repeated arms.
These are descriptive bands, not significance tests or confidence intervals.
Raw native/API counters are latest 12 request rows (repeat 1 only), and timings
can be segment scoped. Fixed order, cache/order changes, memory/swap drift,
integer fixture and n=1 window prohibit a general ranking or equivalence claim.
The failed recovery additionally prevents treating this as a safely restored
calibration window. Production remains .25 / 11 workers / 2048 reserve; no
production recommendation or further-stage authorization.

## Evidence integrity

62 target-derived files, 238 successful existing `hl peek path_read` responses;
all `truncated: false`, strict host keys, retries 0, no raw SSH fallback. Whole
export/source byte counts and SHA256 verified after base64 chunk assembly where
needed (24576 raw bytes per chunk). `retrieval.json` records exact target paths
and local probe commands. `after-receipt.json` describes target export hashes.
Before export is allowlisted/sanitized with original source-hash provenance.
Private synthetic log tails are bounded/redacted. No production logs, raw
config/env/shared-settings bodies, model payload or binary were exported.

`SHA256SUMS` covers all local evidence/derived reports except itself. Old evidence
and original runner artifacts were untouched. Only this fresh leaf and an
append-only calibration doc section changed in the repository. No frozen tool,
code/host, Beads, commit, agent or worktree mutation.
