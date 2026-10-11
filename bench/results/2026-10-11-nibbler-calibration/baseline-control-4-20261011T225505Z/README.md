# One supervised installed-engine baseline — 2026-10-11

**STAGE 1 only: one invocation, one arm, `passed: true`, `restored: true`.**
Pete authorized the Qwen unload and was present for manual recovery. No manual
recovery was needed. No second control, sweep, candidate loader/binary, lower
reserve, native parity fixture, installation or production configuration change
was performed. Further stages require separate primary authorization.

## Command and timestamps

Root on nibbler, local HEAD `fb13c698a75a0b5657dc713d541e298f8cbdaf32`:

```sh
/usr/bin/python3 /data/llm/Strata-tests/perf-audit-20261011/baseline-control-4-20261011T225505Z-tools/nibbler_perf.py \
  --run --fractions baseline --repeats 1 \
  --tag baseline-control-4-20261011T225505Z --window-minutes 30
```

- Launch: **22:55:05 UTC**. Final result/restoration file mtimes:
  **22:56:53 UTC**. Operator post-window proof: **23:00:41 UTC**.
- Defaults retained: 11 workers, 2048 MiB reserve, loopback port 18089,
  96 output tokens, 8192 long-input words. No `--reserve-gates`.
- Existing homelab `RemoteShell`, `nibbler.ts.keen.land`, strict host keys,
  zero retries, no raw-SSH fallback. Mandatory agent preflight passed.
- Fresh root-owned staging directory mode 0700; two tool files uid 0/mode 0444.
  SHA verified after write and immediately before execution.
- One exclusive invocation marker. Coordinator launched using
  `subprocess.Popen(..., start_new_session=True, stdin=DEVNULL,
  stdout=private_log, stderr=STDOUT)`, with no timeout/group-kill wrapper.
  Coordinator PID/start/session **3172811 / 66200737 / 3172811**.
  Reviewed supervisor separately starts its own session; its journal records
  PID/start **3172842 / 66200771**. Both exited.
- The supervisor's expected normal-finish trigger is
  `EOFError: coordinator transport disconnected`: the coordinator deliberately
  closes its owner socket. **No actual transport-loss fault was injected.**
- Detached coordinator exit status was not retained. Its exit was independently
  confirmed; the complete final result and journal establish the passed/restored
  contract. No numeric exit status is claimed.

## Readiness and restoration

Before launch, Qwen and pocket-tts were ready; audio was ok/CPU; both production
slots were idle, with queued/running/waiting/outside_slots all zero. The frozen
coordinator and mutation callback independently repeat the idle check immediately
before unload. No-model preflight and actual pre-exec post-PAM proof both record
uid **996**, gid **979**, unlimited memlock **[-1, -1]**, available kernel child
lists, and the same private config SHA
`6a61ad85ae9a6427e26faa692a8bdc2307fc0bf13e4e1509b7f67f10af2cf805`.

All **15** operator verification checks passed. The final lifecycle phase is
**restored**, `children: []`, with one restoration attempt. Qwen and pocket-tts
are ready; audio is ok/CPU. Router and audio PID/start ticks are unchanged:

| Process | Before PID/start | After PID/start |
| --- | --- | --- |
| Router | 1435174 / 18877666 | 1435174 / 18877666 |
| Audio | 1643128 / 54686907 | 1643128 / 54686907 |
| Qwen frontend | 2880993 / 65583967 | 3176081 / 66208703 |
| Qwen vision | 2881000 / 65583992 | 3176084 / 66208726 |
| Qwen engine | 2881053 / 65584100 | 3176141 / 66208830 |

After reload the frontend uses `/data/llm/Strata-run/nibbler/config.json` from the
installed Strata directory. Hashing `/proc/<PID>/exe` confirms both newly running
native binaries match the pre-window installed hashes. Published pack manifests,
tokenizer and template identity are unchanged; the 63 GiB payload was not
rehashed. No private processes or GPU consumers remain; port 18089 is closed.
The only post-window compute consumers are installed vision/engine, uid 996,
reported at 1268/12538 MiB. This is not an audio latency or whole-fleet check.

The restoration health request ended with `CALIBRATION READY`, finish `stop`,
five output tokens. Its **28.171366464928724 s** wall time includes original-model
reload and router loading banners. The health stream's first banner is **not
model TTFT**; this request is not performance evidence.

## Descriptive measurements, n=1 window

See [metrics-summary.json](metrics-summary.json) and the full source artifacts.
Nibbler's installed 0.1.41 engine, unchanged GPTQ NVFP4 pack, 9900X/5060 Ti
hardware documented in the performance audit; no affinity/clock/memory-policy
changes. Private binaries match the installed engine/vision hashes.

- Private launch-to-metrics-ready: **27.02275056799408 s**. OS cache was not
  flushed; this is an observed first private load, not proven cold.
- Native arena registration: **cudaHostRegister PORTABLE ok**, arena 64800 MiB.
  Buffered loader; MAP_HUGETLB unavailable, preexisting THP always state.
- Preserved startup PCIe probe: **28.9 GB/s host→device**, reported best of
  28.9/28.9/28.8/28.8; effective **pcie_frac 0.25**.
- Eight groups, twelve requests, actual prompts **178 / 8243 tokens**, each
  yielding 96 output tokens. Warm requests reused 171/8236 prompt tokens.
- Client request nearest-rank p95 TTFT **16.133391922921874 s**, wall
  **18.473981243907474 s**. Descriptive across these twelve requests, not a
  confidence bound or representative service p95.
- Reported native/API expert-hit rates **30.0–68.0%**, PCIe-routed share
  **7.8–15.0%**. These source counters can describe native subsegments rather
  than complete HTTP requests; configured PCIe fraction is a different quantity.
- Sampled private-window minimum MemAvailable **22476390400 bytes (20.93 GiB)**,
  SwapFree **8786137088 bytes (8.18 GiB)**, 78 samples. Restoration minimum
  MemAvailable **25060810752 bytes (23.34 GiB)**, SwapFree
  **8078467072 bytes (7.52 GiB)**, 28 samples. No built-in stop fired.
- Sampled private peak GPU used/free minimum **14005 / 1885 MiB**. Approximately
  one-second sampling is not an instantaneous peak or full-context safety gate.

Important source observations: long dual-fresh contains admission/prefill pauses;
its last native segment reports 57 generated versus 96 HTTP output tokens. Warm
dual logs `TARGET_ONLY` MTP suppression, zero drafts offered and incoherent
fallbacks, unlike fresh dual. Initial engine metadata reports `spec: 6`,
`mtp_max: 4`, while the preserved launch args request `--spec 4`. These are
recorded observations, not new tuning or regression conclusions. Do not equate
native per-segment decode rates with end-to-end aggregate HTTP throughput.
Integer-presence and overlapping-output-interval checks are not token/logit
parity, full-context, vision, pressure/park/handoff or reserve-sizing proof.

## Retrieval and hashes

**24 target-derived files**, retrieved through **34 successful existing hl
`path_read` responses**, all **`truncated: false`**, strict host keys, zero
retries. Result (143521 bytes) and arm measurements (128067 bytes) each exceed
the probe bound: six additive host export chunks per file, 24576 raw bytes per
full chunk, base64 JSON below 65536 bytes per probe. Ordered chunk bytes/hashes
and the reconstructed complete source SHA were checked. No original runner
artifact was modified. `retrieval.json` records every path, command and response
hash; `after-receipt.json` records source/export byte counts and whole hashes.

Representative exact retrieval:

```sh
cd /home/pete/devel/homelab
RUBYOPT=-W0 bundle exec ruby /tmp/str-inn.3-peek-strict.rb peek host nibbler path read \
  /data/llm/Strata-tests/perf-audit-20261011/baseline-control-4-20261011T225505Z-tools/evidence/result.json.part-0000.json
```

The invocation-local adapter forces RemoteShell strict keys/retries zero without
editing homelab code/config. `before.json` is an allowlisted export excluding raw
production slot/engine payloads and extra audio fields; its provenance retains
the untouched original SHA. Log exports are private synthetic tails, at most
32768 source bytes / 160 lines, with credential-like lines scrubbed. Empty
recovery/mutation log tails contain one newline; originals were zero bytes.
No production log, shared-settings body, config env or binary was exported.

`SHA256SUMS` lists every local file except itself. The doc hash is reported
separately because it lives outside this directory. All six production hashes
were unchanged:

```text
25cec204db80104c9f0049ee958cbf5792884e27f4dcdbff14eaf76b87bfd836  installed engine
01e24b62a8df26a90bc78b07fca2488dab6f6bd61a7d44c215dfc525c4f996eb  installed vision
ee7ddf6de2a1864b803296ff8b40b857999ffb18fbe412808a6f6035ce91810c  BUILD.json
116b04ecf89fb730f649a921d909305cce0b0b3e95bdddd47c0bddd14ec877d0  tracked config
116b04ecf89fb730f649a921d909305cce0b0b3e95bdddd47c0bddd14ec877d0  runtime config
b61e757d0d94ca1f1c7c17cf180f98e779cbd976d53658a88ae91d870d9c3e2a  shared-settings hash only
```

Keep production 0.25 / 11 workers / 2048 reserve. **No tuning conclusion and no
further-stage authorization follow from this one baseline.** Old failed controls
remain unchanged. No Beads/git mutations, agents or worktrees were created.

Local verification after retrieval: 58 combined tests and 20 standalone lifecycle
tests passed with warnings treated as errors; four frozen tools compiled into a
temporary directory. Local result/readiness/overlap assertions and SHA checks
passed. The calibration doc's entire pre-append byte prefix retains SHA
`ae0d18b1942eebfcf8b3a61bd19d7ba96b03cac6d9c320a819aa1adc65cd358c`.
