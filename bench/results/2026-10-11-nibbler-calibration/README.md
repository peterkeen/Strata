# Nibbler calibration artifacts — 2026-10-11

**No valid calibration trial completed. Keep existing settings.** All three
control attempts stopped before any private model was spawned; their
`result.json` records `passed: false`, `runs: []`, `restored: true`.

See [calibration record](../../../docs/NIBBLER_CALIBRATION.md) for scope, failure
causes, safety pause, restoration verification, pending gates and experiment plan.

Files below are complete bounded host artifacts retrieved with homelab
`bin/hl peek host nibbler path read`; every probe returned `truncated: false`.
No model payloads, production request logs/prompts, secrets, shared-settings
contents, native binaries or private frontend source were copied into these
results. `before.json` contains only current idle/backend facts, hardware state,
artifact hashes and process identities. Restoration uses synthetic health input;
its stream includes router loading banners and is **not a performance measure**.

Host originals:
`nibbler:/data/llm/Strata-tests/perf-audit-20261011/baseline-control-{1,2,3}/`.

- `before.json`: pre-mutation idle/resource/configuration/identity snapshot.
- `result.json`: invalid outcome and exact stop reason.
- `samples.jsonl`: host samples before and through unload; no private model load.
- `restoration.json`: completed short request, both models ready, audio health,
  unchanged artifact hashes and router/audio PID/start-time identities.
- `restore-monitor/samples.jsonl` (controls 2/3): continuous production reload
  sampling. Control 1 had no restoration sampler; this missing check is retained,
  not retroactively claimed.
- `unloaded.json` (control 3): the successful model-only unload state before the
  missing-`runuser` launch failure.

Observed status after control 3 was independently rechecked: `/running` ready for
Qwen and pocket-tts, audio health OK, router PID 1435174/start 18877666 and audio
PID 1643128/start 54686907 unchanged, six production hashes unchanged, no private
model processes. GPU consumers were only the installed vision encoder and engine.
No additional recovery load was needed.

Sampler cadence is approximately one second; these observations do not establish
sub-second peak allocation. No PCIe/worker/reserve speedup, loader improvement,
native parity, full-context or vision acceptance result is supported here.
