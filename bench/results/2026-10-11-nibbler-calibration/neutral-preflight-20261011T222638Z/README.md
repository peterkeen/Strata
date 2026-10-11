# One neutral target preflight — 2026-10-11 22:29:39 UTC

**Readiness passed; no calibration measurement.** Exactly one authorized
`--run --preflight-only` invocation returned 0 with `ready: true`, `passed: false`,
`runs: []`, `production_mutations: "none"`. All nine read-only before/after
verification checks passed. No live baseline is authorized by this outcome.

## Authorization, staging and exact invocation

Homelab `scripts/agent-preflight.sh` passed all required checks in this owner's
context. Source was commit `fa8c13f099cf3b4db47f150b8011150d2c05fbb1`; all four
reviewed tool hashes and the original calibration doc hash matched before staging.
The owner confirmed no unknown target dispatch from the earlier provider disconnect.

Only the runner and lifecycle module were staged, under a new root-owned test leaf
created mode 0700. Staged files were uid 0 / mode 0444, with SHA-256 verified after
write and again immediately before invocation:

```text
1852695e3e6d962567d24264f51c053a37836b63eefc9d86357452c5141041b3  nibbler_perf.py
3af88b22dcacd9dc3c6cd8c6c26613bb85ebcab93527a1a4fb8140943e55b177  nibbler_lifecycle.py
```

Stage:
`/data/llm/Strata-tests/perf-audit-20261011/neutral-preflight-20261011T222638Z-tools`.
Output:
`/data/llm/Strata-tests/perf-audit-20261011/neutral-preflight-20261011T222638Z`.
Neither candidate source/build nor installed production tools was replaced.

Executed once as root on nibbler:

```sh
/usr/bin/python3 /data/llm/Strata-tests/perf-audit-20261011/neutral-preflight-20261011T222638Z-tools/nibbler_perf.py \
  --run --preflight-only --fractions baseline --repeats 1 \
  --tag neutral-preflight-20261011T222638Z --window-minutes 10
```

The invocation marker and execution receipt both record `neutral_invocations: 1`.
The plan was one baseline arm, 11 workers, 2048 MiB reserve, loopback port 18089.
The identical invocation/outcome timestamps are second-resolution observations,
**not a loader-time or performance measurement**.

Staging/execution used homelab `RemoteShell.exec_stdin(["/usr/bin/python3", "-"], ...)`
with hostname `nibbler.ts.keen.land`, `strict_host_key_checking: true`, `retries: 0`,
`log_output: false`; no raw-SSH fallback. An exclusive invocation marker prevented
re-entry, and the outer neutral process was bounded to 600 seconds.

## Actual post-PAM launch proof

`arm-00-preflight.json` records:

- native-inference uid **996**, gid **979**, matching the host account lookup.
- Effective soft/hard memlock **`[-1, -1]` (unlimited)**, measured inside the actual
  post-runuser/PAM child. Coordinator memlock was also `[-1, -1]`.
- Venv Python `/data/llm/Strata/.venv/bin/python`.
- Kernel direct-child-list support **true** in both coordinator and child.
- `dependencies_resolved: true`, `ready: true`, `mode: "preflight-only"`.
- Private prepared-config SHA-256:
  `e40b838be023cae1c69016fa52332e8556baad711cd67a7c864269c1575a79e4`.

The operator snapshots record absolute launcher `/usr/sbin/runuser` and dependency
probe `/usr/bin/ldd`. Neutral Python imports and read-only ELF dependency probes
completed. No native model executable, private frontend, GPU fixture, model
completion, unload/reload/control POST or restoration was invoked. No supervisor
or sampler, private server log, launch-readiness, unload or restoration artifact
was created. No host/PAM/service/global-memory policy or production config changed.

## Before/after proof

The read-only wrapper and neutral runner used:

- GET `http://127.0.0.1:8088/running` — both Qwen and pocket-tts **ready** before/after.
- GET `http://127.0.0.1:5800/health` — **`status: ok`, `backend: cpu`** before/after.
- Neutral runner idle GETs to `/upstream/qwen3.8-flash-next-nvfp4-gptq/slots` and
  `/metrics`, only after the already-ready check — two idle slots verified.
- `/proc/<PID>/stat` start-time identities, scoped frontend/native process
  enumeration, and SHA-256 of the same six installed artifacts/config files.
  Shared-settings contents, config env, production logs and process arguments
  were not exported.
- Read-only GPU queries:

```sh
nvidia-smi --id=0 --query-gpu=memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits
```

All process identities below were unchanged:

| Process | PID | Start ticks |
| --- | ---: | ---: |
| Router | 1435174 | 18877666 |
| CPU audio | 1643128 | 54686907 |
| Installed production frontend | 2880993 | 65583967 |
| Installed vision / GPU consumer | 2881000 | 65583992 |
| Installed engine / GPU consumer | 2881053 | 65584100 |

Only the installed vision and engine were GPU compute consumers, before and
after; there were no new private model processes or GPU consumers. Total GPU
used/free was 13981/1909 MiB in both snapshots; reported consumer memory was
1268 MiB vision and 12560 MiB engine. MemAvailable was 23388459008 → 23332052992
bytes; SwapFree 8785260544 → 8787292160 bytes. These are snapshots, not peak
resource/headroom or GPU-release evidence.

All six before/after SHA-256 values matched:

```text
25cec204db80104c9f0049ee958cbf5792884e27f4dcdbff14eaf76b87bfd836  /data/llm/Strata/engine/strata-nibbler
01e24b62a8df26a90bc78b07fca2488dab6f6bd61a7d44c215dfc525c4f996eb  /data/llm/Strata/engine/strata-vision
ee7ddf6de2a1864b803296ff8b40b857999ffb18fbe412808a6f6035ce91810c  /data/llm/Strata/engine/BUILD.json
116b04ecf89fb730f649a921d909305cce0b0b3e95bdddd47c0bddd14ec877d0  /data/llm/Strata/deploy/nibbler/config.json
116b04ecf89fb730f649a921d909305cce0b0b3e95bdddd47c0bddd14ec877d0  /data/llm/Strata-run/nibbler/config.json
b61e757d0d94ca1f1c7c17cf180f98e779cbd976d53658a88ae91d870d9c3e2a  /data/llm/Strata-run/nibbler/config.shared-settings.json
```

## Retrieval and complete captured artifacts

The existing homelab `hl peek path_read` CLI retrieved the eight allowlisted
JSON files below. Each response had **`truncated: false`**. Each local SHA was
checked against the hash captured by the target execution receipt where available.
`retrieval.json` retains exact argv, source paths, byte counts and hashes.

Invocation-local transport adapter `/tmp/str-inn.3-peek-strict.rb` only forced the
existing `RemoteShell` constructor to strict host keys and zero retries, then
loaded the unmodified `lib/hl/entrypoint.rb`. No persistent config/probe changes.
Representative exact retrieval command (other exact paths are in retrieval.json):

```sh
cd /home/pete/devel/homelab
RUBYOPT=-W0 bundle exec ruby /tmp/str-inn.3-peek-strict.rb peek host nibbler path read \
  /data/llm/Strata-tests/perf-audit-20261011/neutral-preflight-20261011T222638Z/result.json
```

```text
1205a181a03cf5fe9dfb1f8a62113c03267b79f9a6c6869af1abbd593ff8fd2a  invocation-start.json
7b831af4452c6136f0c2dfd48c0d8ab1526f7c3cc73595769c612406332ad7b8  operator-before.json
2b69b8c9c52527eef22a74e816020d35bdb546539d585c28b65bede1df07c5d6  operator-after.json
750f7b178f050153ef6d1612fb2101886dd270f4822b677259c609a2dee662c7  operator-verification.json
1b2bd80b25fba687f6d429ec796b75873c8849f0c7e4833ca3e175f3b32a0e74  before.json
6a9003e065900ed1dc862e23bbba446e7ddb79264508bec5a5cda890289c6fc3  result.json
3d07582e742762c36712e9926cda0442332e90dd42968f3687f229b360a2d9ae  arm-00-preflight.json
b119cd01fa6ea2c41b9743be3b4b2e3c4254fcb66fa9f461323ee2816e435ad4  execution-receipt.json
```

Retrieval caveat: the first `stage-manifest.json` read succeeded according to the
local homelab audit (`status: 0`, `error: null`, 141 ms, 22:29:40 UTC), but the
initial local collector discarded its stdout when Ruby/Bundler compatibility
warnings appeared on stderr. That successful path was **not re-read**, and its
content/truncated field is not retained or claimed here. The remaining eight
**distinct** authorized artifact reads completed with local Ruby warning
suppression only (`RUBYOPT=-W0`). No target transport/probe failure, retry, raw-SSH
fallback, second preflight, host/code repair or control operation occurred.
Task-scoped local peek audit entries are preserved in `probe-audit.json`.

## Remaining limits / stop line

Actual target post-PAM identity/memlock, kernel child-list availability and neutral
launch dependency readiness are established for this one frozen invocation. This
is not actual native/vision descendant lifecycle, private GPU release, arena
registration, loader timing, correctness parity or performance evidence. Manifest
identity is not a fresh 63 GiB payload verification. Previous three failed
controls remain failed. Any future private-model baseline needs **separate fresh
primary authorization and human presence/manual recovery readiness**.
