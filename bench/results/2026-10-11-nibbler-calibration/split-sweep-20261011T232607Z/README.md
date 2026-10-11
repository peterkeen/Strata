# STAGE 2 split sweep — stopped and restored

**One authorized sweep invocation. `passed: false`, `restored: true`. STOP; no
patch, host change, rerun or further stage.** Pete's authorization and manual
recovery presence continued. No manual recovery was needed.

## Command, budget and outcome

Local HEAD `2139033df3d89ffe461aff91f6400fe32803b1c7`; all four frozen tool hashes
matched. Homelab agent preflight passed, checkout initially clean. Local dry run
confirmed eight default arms and repeats 2. Conservative estimate
`8 × 2 × 108 s = 1728 s = 28.8 min`, below the 45-minute cutoff, so defaults were
retained. The estimate was not a guarantee; the frozen 50-minute deadline remained
in force.

Exact root command on nibbler:

```sh
/usr/bin/python3 /data/llm/Strata-tests/perf-audit-20261011/split-sweep-20261011T232607Z-tools/nibbler_perf.py \
  --run --tag split-sweep-20261011T232607Z --window-minutes 50
```

No explicit fractions/repeats override: default order was baseline, 0.10, 0.40,
0.55, 0.55, 0.40, 0.10, 0.25; repeats 2, workers 11, reserve 2048, port 18089,
max-tokens 96, long-words 8192. No reserve gates, worker/reserve change, candidate
binary/loader or GPU parity fixture.

Fresh root-owned tools leaf mode 0700, module files uid 0/mode 0444, SHA checked
after transfer and again before execution. Existing homelab RemoteShell,
`nibbler.ts.keen.land`, strict keys, retries 0; no raw-SSH fallback. Coordinator
launched in its own session, DEVNULL stdin, private-file stdout/stderr, no timeout
or group-kill wrapper. Reviewed supervisor separately detached.

- Launch **23:26:07 UTC**; coordinator PID/start/session
  **3268495 / 66386919 / 3268495**.
- Supervisor PID/start **3268685 / 66387089**.
- First recorded stop **23:28:17.464086 UTC**, `run-01-load`.
- Final result/restoration mtimes **23:28:46 UTC**.
- Independent post-window proof **23:29:41 UTC**.
- Detached coordinator exit status was not retained. Exit was confirmed; no
  numeric status is claimed. Final result explicitly reports failed/restored.

**Arm 00 completed 16 groups / 24 private requests (both repeats). Arm 01's
pre-exec Python launcher then refused before frontend exec/native model load:**

```text
launch_check, nibbler_perf.py line 440:
    s.bind(("127.0.0.1", cfg["port"]))
OSError: [Errno 98] Address already in use
```

The monitor's generic `allocation/native error in server.log` reason was its
Traceback gate, not evidence of a CUDA allocation failure. The exact refusal is
in `arm-01-server.log.tail.txt`; sampler-stop identifies `run-01-load`. Underlying
port-holder/TCP state was not captured: no assertion about a live listener or
TIME_WAIT, and no attempt to repair/clear the port. No explicit-fraction frontend,
native model or measurements ran. Initial status based on `runs: 1` was refined
after retrieving this attempted-launch evidence. All eight no-model preparation
proofs exist, but these are not eight model runs or performance passes.

## Production recovered; sweep still failed

Final lifecycle phase **restored**, `children: []`, restoration attempt **1**.
Expected EOF owner-socket close initiated recovery, not an injected transport
fault. Both coordinator and supervisor exited. All **13 production restoration
checks** passed. The two sweep-completion/success checks failed as expected, so
`operator-verification.json` has **`verified: false`**. This does not mean
production restoration failed; it prevents relabelling the failed sweep passed.

Qwen and pocket-tts are ready; audio is ok/CPU. No private processes or GPU
consumers remain; port 18089 is closed. Router/audio identities and all six
production hashes are unchanged. The restored frontend uses the original
installed working directory/runtime config; actual `/proc/<PID>/exe` hashes
match both original native binaries. Published pack-manifest/tokenizer/template
identity matches; expert payload was not rehashed.

| Process | Before PID / start ticks | After PID / start ticks |
| --- | --- | --- |
| Router | 1435174 / 18877666 | 1435174 / 18877666 |
| Audio | 1643128 / 54686907 | 1643128 / 54686907 |
| Qwen frontend | 3176081 / 66208703 | 3275561 / 66399925 |
| Qwen vision | 3176084 / 66208726 | 3275568 / 66399950 |
| Qwen engine | 3176141 / 66208830 | 3275635 / 66400061 |

Only installed vision/engine consume compute GPU memory after recovery,
uid 996, 1268/12538 MiB. Restoration health finished `stop`, five output tokens,
ending `CALIBRATION READY`; wall time **28.142920011887327 s** includes reload and
loading banners, not benchmark/model TTFT.

## Descriptive retained baseline data, not a valid split comparison

See `metrics-summary.json` and the appended calibration doc tables. Installed
engine 0.1.41, unchanged published GPTQ pack identity, nibbler 9900X / RTX 5060 Ti.
Only baseline (startup-probed effective 0.25) produced measurements:

- Load **27.02327433100436 s**, observed OS cache state, not proven cold.
- Native PCIe probe **28.9 GB/s host→device**, best of 28.9/28.8/28.9/28.8.
- Native `cudaHostRegister PORTABLE ok`; buffered loader, 64800 MiB arena.
- 24 HTTP completions, 96 tokens each; actual prompts 178 / 8243 tokens.
- Client nearest-rank descriptive p95 TTFT **16.04890423209872 s**, wall
  **18.476183643913828 s**, across these 24 requests, not service percentiles.
- Latest twelve native/API rows cover **repeat 1 only**: expert hit rates
  **34.4–67.5%**, PCIe-routed share **6.1–13.3%**. Native counters can reflect last
  segments; do not equate API decode rates with full-request throughput.
- Baseline repeat-0/repeat-1 wall rates differ by **0.004–7.528 tok/s**, depending
  on prompt/streams/state. Repetition reverses solo/dual order, so these are
  observed cache/order-dependent ranges, not controlled noise/confidence bands.
- Planned repeated-arm pairs 0.10 (01/06), 0.40 (02/05), 0.55 (03/04), and probed
  versus explicit 0.25 endpoints are **unavailable**. No split difference can be
  classified as exceeding or not exceeding between-arm repeat variation. This
  is missing evidence, not evidence that fractions are equivalent.

Main-window 126 samples: minimum MemAvailable **22932238336 bytes** and SwapFree
**8145944576 bytes**; sampled peak GPU **14043 MiB**, minimum free **1847 MiB**.
Restoration 28 samples: minimum MemAvailable **24910196736 bytes**, SwapFree
**7619219456 bytes**, sampled peak GPU **13939 MiB**. Thresholds did not fire;
the launch traceback did. One-second sampling is not instantaneous peak proof.

Integer presence and overlapping output intervals are not token/logit parity,
full-context/vision/pressure gates, or permission to lower reserve. Warm dual
logs include target-only paths; API segment timings and changing cache state
remain caveated as in STAGE 1. No production setting recommendation follows.

## Bounded evidence and hashes

**33 target-derived files**, **53 successful existing hl path_read responses**,
all `truncated: false`, strict keys, zero retries. The local receipt helper's
planned-count display had a redundant +1; actual `retrieval.json` records 53
responses, not 54. Result (273680 bytes) and arm measurements (244081 bytes) were
reconstructed from 12/10 additive host base64 chunks, 24576 raw bytes per full
chunk. Per-chunk and whole-source SHA/byte counts match. No original artifact
was modified. Exact paths/commands/hashes are in `retrieval.json` and target
source descriptors in `after-receipt.json`.

`before.json` is an allowlisted export with raw production slot/engine payloads
and extra audio fields removed, retaining original SHA provenance. Logs are
private synthetic tails, at most 32768 source bytes / 160 lines, credential-like
lines scrubbed. No production logs, private config env, shared-settings body or
binary exported. `SHA256SUMS` covers all local files except itself.

Six unchanged production hashes:

```text
25cec204db80104c9f0049ee958cbf5792884e27f4dcdbff14eaf76b87bfd836  installed engine
01e24b62a8df26a90bc78b07fca2488dab6f6bd61a7d44c215dfc525c4f996eb  installed vision
ee7ddf6de2a1864b803296ff8b40b857999ffb18fbe412808a6f6035ce91810c  BUILD.json
116b04ecf89fb730f649a921d909305cce0b0b3e95bdddd47c0bddd14ec877d0  tracked config
116b04ecf89fb730f649a921d909305cce0b0b3e95bdddd47c0bddd14ec877d0  runtime config
b61e757d0d94ca1f1c7c17cf180f98e779cbd976d53658a88ae91d870d9c3e2a  shared-settings hash only
```

Local parsed-evidence checks preserve failed/restored status, 24 baseline
completions, 16 groups, positive dual overlaps, eight no-model unlimited-memlock
proofs, and the thirteen passing production checks. Frozen code remains unchanged;
no code/host patch, rerun, manual cleanup, Beads/git mutation, agent or worktree.
**Stopped. Production remains 0.25 / 11 workers / 2048 reserve. Primary owns review
and any separate future authorization.**
