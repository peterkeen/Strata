# Nibbler deployment candidate — `kv-unified`

Status: **port in progress; being validated, not a validated port deployment**.
The unified-KV fork lineage (copied from fork tip `9921eec`) is now ported onto
upstream main `fb58e0db` on branch `kv-unified`. The initial port commits are
`1092339f` (runtime/session/serving) and `537f8b35` (CLI integration).
This is the newer upstream lineage following 0.1.40.x; both the upstream base
and this checkout's `CMakeLists.txt` identify the engine as **0.1.41**.

The [unified KV work log](../../docs/MULTI_SLOT_UNIFIED_KV.md),
[incremental allocation work log](../../docs/INCREMENTAL_UNIFIED_KV.md) and
[deployment record](validation.md) retain the fork's dated measurements and
rollouts. Those results are evidence for the fork builds tested on
2026-10-03 through 2026-10-06, **not measurements or deployment verification
of this port**. No port performance or quality result is claimed here.

## Preserved configuration and architecture

`config.json` retains the nibbler IQ3_S model and sharp tokenizer paths,
GPU vision, and these budgets:

- Two serving slots (`"parallel": 2`) with `--kv-unified`, one shared
  **262144-cell authoritative pinned host backing pool** (K/V backing per
  attention layer), and **32768 GPU-resident cells per attention layer**.
  Neither physical budget is multiplied by the slot count.
- A logical **262144-cell ceiling per request**, not half that ceiling per slot.
  Aggregate capacity still limits which histories can coexist.
- Private recurrence, PLE history, QSA indexer and logical maps. Private MTP
  ring/draft KV and captured-graph buffers are not part of the shared main
  attention pool. Ordinary batch windows do not run per-slot MTP drafts.
- Reference-counted prefix sharing with copy-on-write (COW), last-reference
  invalidation, and shared GPU CLOCK residency.
- Incremental admission: known prompt plus at most **256 output cells** of
  rolling headroom; exact target-write/COW preflight before inference. Optional
  headroom shortage alone does not preempt a request.
- Pressure continuation: actual backing exhaustion can safely park/release an
  owner, then continue the same frontend stream using original prompt IDs plus
  **all** returned tokens and the remaining allowance. Optional parking failure
  or eviction falls back to replay. Logical context and omitted-output-limit
  semantics are unchanged by the reservation policy.
- Warm `HANDOFF`/`BHANDOFF` transfers preserve valid completed-prefill prefixes;
  actual `STOP`/`BSTOP` cancellation still releases ownership. Target-only
  transfers suppress private MTP/suffix proposals until coherent full residual
  prompt replay rebuilds draft history; the private ring remains allocated.
- **2048 MiB VRAM reserve**, **8192 MiB conversation parking budget**, at most
  **four parked entries**, and a **4096 MiB available-RAM floor**. Parking is
  separate from authoritative host KV and does not guarantee four full-context
  images or reserve OS RAM. These are retained settings, not newly measured
  safe headroom on the port.

## Port restrictions and config acceptance

The port's unified serving path is single-GPU. The config selects GPU 0;
keep it single-GPU and do not add a layer split or helper-GPU configuration.
The real parser in `src/program/generate.cpp` enforces:

- `--serve`, `--batch 2..8`, and `--batch-groups 1` (the default); group `auto`
  is rejected.
- **No layer split**, including `--layer-split auto`. Rejection happens before
  split resolution, even if `auto` would eventually choose only one GPU.
- **No `--batch-mtp` or `STRATA_BATCH_MTP=1`**. Any nonempty environment value
  whose first character is not `0` also requests batch MTP and is rejected.
  This does not reject ordinary `--mtp`/`--spec` for coherent solo execution.
- **Unified and elastic KV are mutually exclusive**: no `--kv-grow`,
  `--kv-elastic`, or enabled `STRATA_KV_GROW`. This concerns elastic K/V, not
  expert caching. An explicit elastic-KV flag is rejected even if the
  environment says `STRATA_KV_GROW=0`.
- Failure to allocate at least two slot sessions is fatal in unified mode;
  it must not silently fall back to independent/solo KV.

`"parallel": 2` is a **server config key**, not an engine `--parallel` flag.
`serve/server.py` translates it to `--batch 2` and starts the engine with
`--serve`. Passing `--parallel` directly to the engine is an unknown-argument
error. Every flag in the tracked `args` has a compatible parser entry; the
flag-by-flag source check is recorded in [validation.md](validation.md).

The environment now contains only `STRATA_PF_FUSED=1`, which the ported fused
expert code reads. The fork's `STRATA_PREFILL_FUSED_TAIL` and
`STRATA_PREFILL_BALANCE_TAIL` experiments **are not in this port**; the inert
`STRATA_PREFILL_FUSED_TAIL=0` config entry has been removed. Their measured
2026-10-03/04 comparisons remain in the historical deployment record. Do not
infer their availability, fallback behavior or timings for the upstream prompt
path. Likewise, the old README's `src/spec/output_limits_test.cpp` and
`src/prefill/chunk_schedule_test.cpp` references do not exist in this checkout.
Upstream's `--prefill auto` and `--expert-cache auto` retain compatible syntax,
but flag acceptance is not proof of the fork's exact chunk sizing, output bytes
or performance on this newer engine.

The server inherits its parent environment before overlaying `config.env`.
Ensure `STRATA_BATCH_MTP` and `STRATA_KV_GROW` are unset or disabled when
launching this candidate; the tracked config does not override them.

## Deployment prerequisites — not completed

This task checks source/config compatibility only. JSON can be checked locally:

```sh
python3 -m json.tool deploy/nibbler/config.json
```

It does not establish that the configured model, tokenizer, expert profile,
CUDA libraries, engine/vision binaries or runtime directories exist on nibbler,
or that the port starts and passes GPU/model/HTTP gates there.

The old fork used a CUDA 13.3/GCC 15 Release build, installed a separate
`engine/strata-nibbler`, and enabled standalone conversation tests with
`-DSTRATA_BUILD_CONVERSATION_TESTS=ON`. The target and test option still exist,
but that historical toolchain/build evidence must be repeated for the port.
Do not overwrite the original engine or install this candidate before the
required validation and an explicitly authorized rollout.

**Launcher integration is missing from this checkout:**
`deploy/nibbler/start-strata`, referenced by the fork's external
`/opt/native-inference/bin/start-strata-sharp-medium`, has not been ported here.
The old claim that the launcher refreshes the runtime config on every start
therefore is not a verified property of this branch. Restoring/reviewing that
integration and confirming frontend/proxy ports are deployment prerequisites,
not actions performed by this documentation task.

The historical layout kept writable logs/config/shared Chat settings under
`/data/llm/Strata-run/nibbler`, owned by `native-inference`, rather than in the
root-owned checkout. Preserve existing shared Chat settings; do not replace
user settings on each restart. The candidate retains the old model name,
loopback host, allowed-host list and engine-log path. No live launcher, alias,
ownership or service state has been inspected for this port. Keep the server
on `127.0.0.1`; exposing it elsewhere requires `--api-key`.

Do not run upstream `update.sh`/setup blindly against this hardware-specific
candidate: review its checkout/build/config rewriting first.

## Historical rollback evidence

The dated fork rollouts, exact hashes, saved binaries/frontends/configs and
rollback scripts remain in [validation.md](validation.md) and the linked work
logs. They are **historical fork recovery procedures**, not a prepared rollback
for a port rollout. Their paths, hashes and availability have not been rechecked
here. Save and verify the actual deployment baseline and prepare a rollout-specific
rollback before any authorized installation; preserve shared Chat settings and
avoid changing unrelated services. Nothing in these documents authorizes live
changes.
