# Several requests at once (batch slots)

By default Strata serves **one request at a time**: the others wait in the server's queue. With `"parallel": N`
(the engine's `--batch N`, also spelled `--slots N`) the engine keeps up to N conversations open and decodes them
**together**: every verify window then carries one token of each conversation, so the dense weights, the shared
expert, the head and every routed expert two conversations share are read once per window for all of them.
Combined with a layer split across several GPUs and `--batch-groups`, the cards also work on different
conversations at the same time instead of waiting for each other.

It is opt-in and changes nothing when the options are absent (#465; the engine part is PR #559).

## Turning it on

One GPU: add `"parallel": 2` to the model's config (`strata-<model>.json`) and restart, or run setup with
`--parallel 2`. Setup recommends it only where it does not cost speed (below); any number you ask for is kept as
asked, with a note when it is more than setup would recommend.

```
"parallel": 2
```

On one GPU with MTP (`--mtp` and `--spec`), `--batch-mtp` (in the config's `args`, or `STRATA_BATCH_MTP=1` in the
server's environment) lets each batch slot verify **one MTP proposal per window**: the slot's own private drafter
proposes the token after its current one, and the verifier commits both rows when they agree. It is opt-in; without it
the batch behaviour described below is exactly the one without MTP, and the solo path is unchanged. It needs VRAM per
slot for the draft state and its buffers, so check the engine's free-memory log before using it on a smaller card. It
runs when all of these hold, and otherwise the engine says why and batches as usual:

- `--serve` with `--batch 2..8` (a window holds at most eight rows), `--mtp` and `--spec T` with `T >= 2`;
- **one GPU**: no `--layer-split` and no helper GPU. It has not been validated with a layer split;
- `--kv-unified --batch-groups 1` for the shared-pool window reservation described in
  [INCREMENTAL_UNIFIED_KV.md](INCREMENTAL_UNIFIED_KV.md). Without unified KV a slot still verifies one proposal per
  window, but there is no shared-pool reservation to prove, so the optional-row fallback below does not apply.

Each window reserves its **mandatory** target row first; only then does it ask once for the optional two-row extent
that carries the proposal. A **shortage** of that optional row is a target-only window (`fallback_reserve` in the
slot's counter line) with no reclaim retry and no pressure escalation; only the mandatory row's shortage, after its
own reclaim retry, parks an owner. That guarantee is about the shortage path only: a *successful* optional
reservation is an ordinary two-row write, so it maps shared pages and can contribute to a later mandatory shortage
which that path may then reclaim or park for.

One target-only window also ends speculation for that slot's whole lifecycle: the slot's draft coherence is
invalidated (`batch_draft_after_commit(..., drafter_advanced=false, ...)`) and every later window is target-only too
(`fallback_incoherent` in the counter line) until the slot is re-admitted. The output stays exact (target-only rows
still commit verified target tokens; the tails, limits and lifecycle gates matched their solo references), but the
one-proposal gain is gone for that lifecycle - measured in the pressure run: slot 1 reported `fallback_reserve=1`
together with `fallback_incoherent=4`.

Each slot keeps its own **bounded private draft ring**, sized from the MTP window:
`mtp_kv_ring_cells(window, max_cells, max_t) = window + 4*max_t + 64` when `0 < window < max_cells`, and `-1`
(private, fully resident draft K/V) otherwise, where `window` is `--mtp-window` (default 32768 cells) and `max_cells`
the slot's logical context. A default window is at least the context on a small pool, so the resident path applies and
the bounded ring needs an explicit smaller window. Measured on an RTX 5060 Ti at `--max-context 4096 --mtp-window 128`
with `max_t = 2`: 200 cells requested and 200 allocated, against 1536/1540-cell prompts and 2048 consumed cells.

The per-slot counter line (stderr, once per slot MTP lifecycle) is:

```
strata batch_mtp_stats slot=N windows=N offered=N accepted=N rejected=N discarded=N fallback_attempts=N
    fallback_incoherent=N fallback_not_ready=N fallback_limits=N fallback_capacity=N fallback_reserve=N
```

`offered == accepted + rejected + discarded`, a target-only window has exactly one reason counter, and `offered`
counts proposal rows passed to a successful verifier call. RTX PRO 5000 owners measured +31%
to +39% total throughput with 2 to 4 clients (a RX R9700 run too); it has not been validated with a layer split.

With a layer split, the engine options go into the config's `args`:

```
"args": [ ..., "--batch", "8", "--batch-groups", "4", "--trim-stage-weights" ],
"layer_split": "12,24,36"
```

| Option | What it does |
| --- | --- |
| `"parallel": N` / `--batch N` / `--slots N` (2..8 normally) | up to N conversations have batch slots; more requests wait for a free slot. Each slot gets its own state (a session carved like the stage's own: GDN recurrence, QSA K/V and indexer, PLE history) on every GPU of the split. With grouped MTP, more than 8 slots can rotate through eight-row windows if memory permits. |
| `--batch-groups G` | with a layer split (default: auto, below): the N slots in G groups that flow through the GPUs as a pipeline (GPU k runs one group while GPU k+1 runs another). G must divide N. 1 = all slots in one window, GPU after GPU. |
| `--batch-groups auto` | the default on a layer split since 0.1.41 (give no `--batch-groups`): the engine pipelines one group per GPU stage (the most that divide the slots; 8 slots on 4 GPUs = 4 groups of 2) and says so (`INFO batch_groups=G`). `--batch-groups 1` turns it off (all slots in one window, GPU after GPU). Measured, 8 clients, total tok/s against one group: 4 x R9700 166 against 86, 2 GPUs 109 against 78, 3 GPUs 110 against 78. |
| `--trim-stage-weights` | with an **explicit** `--layer-split` (e.g. `12,24,36`, not `auto`): every GPU loads only the dense weights of its own layers instead of the whole model's (the same as `STRATA_STAGE_TRIM=1`, PR #639). The VRAM this frees goes to the expert cache. Useful without `--batch` too. |

The engine never refuses a count it cannot run: it says so in its log and runs what it can - at most 8 slots by
default (a window holds 8 rows), as many as fit in VRAM, or none (one request at a time) when not two fit. The server reads
the count the engine reports (`INFO batch_slots=N`), and `GET /v1/status` says it (`concurrency.serving`).

### Elastic shared KV capacity (opt-in)

Add `--kv-unified` to the engine's `args` beside `"parallel": N` (or native `--batch N`). Attention KV pages then come
from **one shared pool** instead of one pool per slot; recurrent state, PLE history, the QSA indexer and the MTP ring
stay private to each slot. Cached prefixes share pages with copy-on-write, so two slots reading the same history do not
store it twice.

The pool has an aggregate host backing of `--max-context` cells **for the whole engine** - it is not divided between
slots - and `--kv-resident` cells per attention layer stay on the GPU while the rest of each layer streams from pinned
RAM. `--max-context` is also each request's logical ceiling, so 262144 logical context with 32768 GPU-resident cells
is a supported configuration, not an approximation.

Admission is incremental: a request reserves its known prompt plus a bounded headroom (256 cells) instead of reserving
its whole possible output, and every write or copy-on-write extent is preflighted exactly. A headroom shortage does not
preempt anyone. Real exhaustion releases a safely parked owner and hands the stream back to the frontend, which
continues the same request through its original plus returned tokens (`pressure` completion); a warm conversation can
also hand its cache back for a solo/batch transition (`handoff`) instead of replaying its history. Conversation parking
(`--conversation-cache-mib`, `--conversation-cache-slots`) keeps those idle caches in pinned RAM.

Unified KV requires a single session GPU: layer split and pipeline groups (`--layer-split`, `--batch-groups`) are
rejected rather than half-supported. It is opt-in and orthogonal to expert streaming and caching.

See [MULTI_SLOT_UNIFIED_KV.md](MULTI_SLOT_UNIFIED_KV.md) for the shared-pool design and
[INCREMENTAL_UNIFIED_KV.md](INCREMENTAL_UNIFIED_KV.md) for the incremental admission policy, the measured validation
and the remaining limits. All per-slot memory estimates below describe the default independent-pool mode; with
`--kv-unified` the KV footprint is one pool plus the resident window, not `N` times a slot's session.

### What a slot costs, and what setup recommends

Every slot's session takes VRAM that the expert cache would otherwise hold: 0.56 GiB at a 32K context with 8-bit
KV, more with a longer context unless the KV cache streams (`--kv-resident`: then only the attention's 32K window
stays in VRAM, and each slot's whole KV cache takes pinned RAM - 1.6 GB at 128K). On a card whose experts mostly run
on the CPU, a batch also reads about as many distinct experts as the requests one by one (different conversations
route to different experts), so the gain is in **latency** (nobody waits for a whole answer), and a request alone
runs slower (the smaller expert cache): 11-24 % on a 12 GB card, see the measurements below.

So setup recommends `"parallel"` **only where the experts mostly fit in VRAM**: the expert cache (each card's VRAM
less ~5 GB, every card of a layer split counted) must still hold at least half of the model's experts beside the
slots, and the slots may take at most a fifth of it, up to 4 slots. With Q2_0 at 32K that is 3 slots on a 24 GB
card, 4 from 32 GB or on a split such as 2 x 16 GB; IQ3_S needs 32 GB or a split. Everywhere else (any 12 or 16 GB
card alone) it stays at one at a time and setup says: "parallel N reduces waiting for several users but costs
about 10-25% speed per request on this card". `--parallel N` is honoured as asked either way.

## How the server uses the slots

- **One request alone** runs on the usual solo path (verify windows with MTP drafts): the fastest single stream.
- **When a second request arrives**, the first hands off (`HANDOFF` with unified `kv_handoff=1`, otherwise the legacy
  `STOP`) and continues in a batch slot with its prompt plus what it generated so far. A valid prefix is kept for the
  admission instead of being treated as a cancellation, so nothing is read again, and the new request is admitted next
  to it. By default, a request in a slot decodes **without MTP drafts** (one token per window).
  With `--batch-mtp`, each slot verifies one MTP proposal alongside its current token.
- **A request left alone in a slot** (the others finished, nobody waits) goes back to the solo path: the slot is
  handed off (`BHANDOFF` with unified `kv_handoff=1`, otherwise the legacy `BSTOP`), the engine copies its sessions
  back and decodes with MTP drafts again (at most twice per request; with `--prompt-cache 0` it stays in the slot;
  `STRATA_PARALLEL_SOLO=0` turns it off). With unified KV the slot clone-back is target-only: private MTP/suffix
  proposals stay suppressed until a full residual prompt replay rebuilds coherent draft history, so it never uses an
  unrelated conversation's draft K/V. The engine says so on stderr
  (`TARGET_ONLY slot clone` / `TARGET_ONLY restore`, then `TARGET_ONLY decode: T=1, MTP/suffix proposals disabled`) and
  the private ring stays allocated. Measured with `--batch-mtp` (context 4096, 1536-cell prompt, cap 64):
  `BDONE 0 3 handoff` with a nonterminal 3-token prefix, then a MAIN continuation that reused 1538 of its 1539 prompt
  cells (1536 + 3, the last returned token unfed), reproduced the same-binary solo token IDs exactly
  (3 + 61 = 64) and resumed **30 accepted MAIN draft offers** - a coherent full-slot transfer, not a target-only clone.
  The same run's solo reference reused a turn checkpoint instead of the live slot and stayed target-only (the
  documented checkpoint behaviour). Without unified KV,
  the draft layer's own K/V was built for another conversation then, but measured it accepted as many drafts
  (140 of 172) as a draft layer that read the conversation (140 of 173).
- **More requests than slots** wait for a free one (`/metrics` -> `live.slots` shows each slot: idle, reading or
  decoding, its tokens and tok/s; `live.running` the requests in flight).
- **Each admission** reads the request's prompt through the usual prompt path (prompt cache and conversation
  checkpoints included) and produces its first token there; the state is then copied into the slot. Admissions
  are taken one at a time, and **the slots decode between the prompt's chunks**: after each chunk (`--prefill`,
  2048-8192 tokens) they decode for half as long as the chunk took (`STRATA_BATCH_DECODE_SHARE`, default 0.5), so
  a long prompt slows the others down instead of stopping them. The chunks are the ones one uninterrupted read
  takes, so the prompt's arithmetic is unchanged.
- **A long prompt gives way to a short one** (#656's cooperative preemption): when a request with a prompt under
  half as long is waiting, the server sends `BYIELD`; at its next chunk boundary the long read stops, the part read
  so far is copied into a slot, the short request is admitted, and the long one then goes on from its slot with the
  same chunks (at most twice per request).
- **Each slot is a conversation cache.** A finished slot keeps what it holds (the prompt, the answer, and the
  checkpoint at the prompt's last turn boundary); the next turn of that conversation goes to that slot and the
  engine copies its state back (50-60 ms for a short conversation) instead of reading the history again - also for
  a client that drops the reply's thinking from the history (the checkpoint matches up to the new turn). A new
  conversation takes an empty slot, else the one used longest ago.
- Slots are assigned so that consecutive requests land in different pipeline groups (`--batch-groups`).
- A client that disconnects stops its slot (`BSTOP`); the others go on.

## Exactness

A batch row's arithmetic is the single-token window's, so with greedy decoding **every conversation of a batch
produces exactly the tokens it produces alone** - verified token by token for 8 concurrent conversations of 150
tokens, with and without the pipeline (`tools/batch_test.py`), and on one RTX 5070 for 4 conversations, for a long
prompt read while two others decode, for a prompt that gave way and went on, for a next turn continued from its
slot (from all it holds, and from its turn checkpoint without the reply's thinking), and for a request stopped in
its slot and continued on the solo path (`tools/batch_interleave_test.py`). These
settings make the comparison exact:

- `STRATA_IQ_MT_MIN=1` (the multi-token CPU expert kernels for every group, as for the solo path's own
  exactness tests: by default an expert's rows round differently alone than in a group, so the output depends on
  how many rows of a window share an expert - which differs between a batch and a request alone),
- `--pcie-frac 0`: the PCIe share of the missed experts is chosen per window from the window's misses, so the
  same expert can run on the GPU in one window and on the CPU in another, which rounds differently, and
- `--adapt-every 1000000` (the VRAM tier fixed), and `--no-prefill-borrow` while a prompt is read beside decoding
  slots: their windows then see the expert cache without the slots the prompt borrowed (those experts run on the
  CPU).

With the default settings the outputs stay coherent but drift apart after some tokens, as two solo runs whose
windows differ can. Measured on the RTX 5070 (Q2_0, 4 conversations of 200 tokens, `tools/batch_test.py` without
the settings above): one equal to its solo run, the others apart from token 0, 99 and 174 - the first token already
differed for one, because the adaptive VRAM tier had moved experts between the solo runs and the batch; all four
texts read as well as their solo runs.

Sampled requests (temperature, top_p, top_k, min_p, seed) are drawn row by row with the solo window's
counter-based draw (Philox(seed, position)).

## Limits (for now)

- By default, batch windows carry no MTP drafts: a conversation in a slot decodes one token per window (the solo
  path keeps its drafts, which is why a request alone is not put in a slot, and goes back to it when left alone).
- `--batch-mtp` opts each slot into one MTP proposal per window (above). It is **one proposal, not a chain**, and it
  still requires one GPU (`--serve`, `--batch 2..8`, `--mtp`, `--spec T >= 2`, no `--layer-split` or helper GPU).
- Repetition / frequency / presence penalties are not applied in batch windows.
- A prompt shorter than one chunk is read in one piece (the slots wait for it); a read gives way only at a chunk
  boundary, and not for pictures.
- Admissions are one at a time: two new long prompts are read one after the other.
- `--batch-groups` needs every stage on its own GPU. A pipelined slot is a conversation cache again (0.1.41): a request left alone in its slot goes back to the solo path with its drafts, as on one GPU.
- The slot sessions take VRAM (above) and, with KV streaming, pinned RAM.

## Measured

One RTX 5070 (12 GB), Ryzen 5 7600, 64 GB DDR5, Q2_0, 32K context, through the HTTP server: C different requests
sent at once (an 800-word essay each, 256 tokens per answer, greedy, thinking off), median of 3 rounds:

| Concurrent | Setting | Total tok/s | vs one at a time | Per request tok/s | First token: median / last of the round |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | one at a time (default; what setup recommends on this card) | 74.3 | - | 83.6 | 0.4 s / 0.4 s |
| 1 | `"parallel": 2` | 67.3 | -9 % | 74.8 | 0.4 s / 0.4 s |
| 1 | `"parallel": 4` | 57.9 | -22 % | 63.8 | 0.4 s / 0.4 s |
| 2 | one at a time | 71.6 | - | 77.3 | 2.1 s / 4.1 s |
| 2 | `"parallel": 2` | 61.0 | -15 % | 32.6 | 0.5 s / 0.9 s |
| 2 | `"parallel": 4` | 54.6 | -24 % | 28.7 | 0.6 s / 0.7 s |
| 4 | one at a time | 70.7 | - | 79.6 | 6.0 s / 11.2 s |
| 4 | `"parallel": 2` | 61.2 | -13 % | 32.2 | 4.5 s / 9.3 s |
| 4 | `"parallel": 4` | 63.1 | -11 % | 16.9 | 1.0 s / 1.8 s |

On this card the slots buy **waiting time, not speed**: the fourth of four requests starts after 1.8 s instead of
11.2 s, but together they decode 11-24 % slower than one after the other, and a request alone loses 11 % (2 slots)
or 24 % (4 slots), because the slots' sessions (0.56 GiB each) come out of the expert cache and most experts run
on the CPU: a batch window over 4 conversations reads 24 CPU experts per layer against ~8 for one, so it costs about
what the 4 tokens cost one after the other (`strata batch:` in the engine log: 54 ms per 4-row window, ~20 ms per
1-row window). This is why setup leaves a 12 GB card at one at a time. Cards that hold most experts in VRAM, and a
layer split, are where the slots also add speed (below).

A 4-GPU layer split (4 x 16 GB, PCIe Gen3), IQ3_S, `--batch 8 --batch-groups 4 --trim-stage-weights`, through
the HTTP server, 400 tokens per answer, temperature 0.7 (PR #559):

| Concurrent requests | Per request | Total |
| ---: | ---: | ---: |
| 1 | 123 tok/s (solo path) | 120 tok/s |
| 2 | 57 tok/s | 113 tok/s |
| 4 | 51 tok/s | 205 tok/s |
| 8 | 45 tok/s | 360 tok/s |

With the patches below on engine 0.1.38 and parking on, through the service: 8 requests at temperature 0 -> 369
tok/s, at 0.7 -> 358 tok/s.

`--trim-stage-weights` alone raised the share of experts held in VRAM on that machine from 76-85 % to 84-100 %
per card.

### Batch MTP with unified KV (opt-in)

One RTX 5060 Ti, greedy, two concurrent 1024-cell prompts of 320 tokens each (the slot-count control is one stream),
`--max-context 4096`, `--kv-unified` (4096 cells), 4096 MiB parking, `--spec 2 --mtp-max-t 2`, `--pcie-frac 0`,
identical expert placement, private subprocesses (not the HTTP server), 3 repetitions per arm in an ABCCBA+ABC order.
This is a synthetic generation workload, not a quality benchmark:

| Arm | Aggregate wall tok/s | Decode-window tok/s (last BDONE - first BADM) | Sum of per-stream tok/s |
| --- | ---: | ---: | ---: |
| (a) two target-only streams, batch MTP off | 18.26 / 18.32 / 18.33 | 24.63 / 24.75 / 24.82 | 31.31 / 31.49 / 31.61 |
| (b) two streams, `--batch-mtp` | 20.09 / 20.11 / 19.33 | 28.13 / 28.16 / 26.64 | 37.52 / 37.60 / 34.84 |
| (c) one target-only stream (slot-count control) | 14.33 / 14.31 / 14.35 | 24.17 / 24.09 / 24.24 | - |

`--batch-mtp` was faster in every repetition: **+5.4 % to +9.8 %** aggregate wall throughput and **+7.3 % to +14.2 %**
on the dual decode window (and +10.2 % to +19.9 % on the sum of per-stream rates). The margin narrows in the third
repetition of arm (b), which is also the only one with rejections, and is reported as measured rather than averaged
away. Real proposals in arm (b): 318 offered / 318 accepted / 0 rejected, the same again, and 320 offered / 317
accepted / 3 rejected - **99.7 % accepted over the arm** - with 2, 2 and 1 target-only windows
(`fallback_attempts`). Both arms used one proposal per window (no multi-draft comparison). The single-slot control
shows how little two *target-only* streams buy here (24.6-24.8 against 24.1-24.2 tok/s decode window, about +2 %);
the MTP arm is what lifts it.

What this A/B does **not** measure: the batch slot sessions' own cost. All three arms run `--batch 2 --kv-unified`,
so the ~1.00 GiB of VRAM the native reports for the two slot sessions (and the expert-cache reduction it causes) is
present in every arm. Observed expert cache: 6407 MiB / 2430 slots in arm (a) and (c), 6386 MiB / 2422 slots in arm
(b), so the MTP feature itself costs ~21 MiB and ~8 expert slots here. No steady batch CPU fraction, COW internals,
park/restore timing or output quality was measured in this A/B.

### Full-context measurement (262144 cells, 2026-10-09)

The same feature measured at the context the engine is meant to serve, on the deployed NVFP4 build
(`engine/strata-nibbler` sha256 `25cec204…`, pack `huihui-nvfp4`), one RTX 5060 Ti, greedy temperature 0,
`--max-context 262144 --kv int8 --kv-resident 32768 --kv-unified --no-kv-grow --spec 4 --spec-min-p 0.5
--prefill auto:8192 --batch 2`. The **only** knob under test is `--batch-mtp`; both sides use one proposal per
slot. Three repetitions per arm in an alternating order, every repetition in the evidence record:

| Arm | MTP on | MTP off | Change |
| --- | ---: | ---: | ---: |
| Solo stream, 125 000-cell prompt, 128 output tokens | decode **29.81** tok/s, prefill 1827 tok/s | decode **23.62** tok/s, prefill 1888 tok/s | decode **+26.2 %**, prefill −3.2 % |
| Two concurrent 125 000-cell streams, 128 each | Σ per-stream **38.19**, dual-decode window **60.21** tok/s | Σ **29.82**, window **47.48** tok/s | **+28.0 % / +26.8 %** |
| Short prompt, 256 output tokens | decode **30.53** tok/s | decode **23.89** tok/s | **+27.8 %** (wall −18.9 %) |
| 250 000-cell prompt (one run per side) | decode 29.22 tok/s | decode 23.08 tok/s | +26.6 % |

Draft acceptance on these prompts is 63/63, 63/63 with one 64/63/1, 127/127 and 31/31 - near total, because the
workloads are greedy and highly predictable; a chat workload will accept fewer. Native evidence:
`strata batch: 64 windows, avg 1.98 rows` with MTP against `127 windows, avg 1.00 rows` without, i.e. one proposal
per window. No pressure or park event occurred in any run. The prefill rate is 3.2 % lower with MTP at 125k because
the feature's own VRAM keeps 1845 expert slots against 1890 and the engine picks a smaller adaptive prefill chunk
(1656 against 1693 borrowed slots) - a cost of the feature, not a settings difference. Expert-cache hit rate during
decode was 19-21 %. The aggregate wall rate is flat because a 125k prompt is prefill-dominated; the decode figures
above are the ones the drafts affect. Host note: two repetitions of the dual arm had to be re-run because swap was
exhausted mid-run (the record includes both discarded attempts and the per-run RAM/swap state); no failed-run number
is quoted.

## Together with conversation parking

`--conversation-cache-mib N --conversation-cache-slots K` (DETAILS.md) works with the layer split too: a request
whose conversation was parked is restored on every stage before its admission, so an agent and its sub-agents, or
several chats that alternate, come back without reading their history again. Measured on the same 4-GPU split,
two long conversations alternating through the HTTP server: the first turns took 3.9 s and 5.7 s to the first token
(their prompts read), the follow-ups 0.53 s and 0.46 s.

## Testing

Four scripts drive a built engine or a running server; each exits non-zero on a failure. `serve/test_parallel.py`
tests the server's side with a scripted engine (no GPU).

| Script | What it checks |
| --- | --- |
| `tools/batch_test.py` | the same prompts alone (`GEN`) and together in the batch slots (`BGEN`): every slot's greedy tokens equal its solo tokens; prints the aggregate rate. `--batch-groups` in `--extra` tests the pipeline, `--keys "temperature=0.7"` the sampled rows. |
| `tools/batch_interleave_test.py` | a long prompt read while two slots decode, a prompt that gives way (`BYIELD`) and goes on, and a next turn continued from its slot: each equal to its solo tokens. |
| `tools/parking_test.py` | a follow-up to a conversation decodes the same tokens whether its state stayed live or came back from the parking cache (with a layer split: every stage's image). |
| `tools/early_close_test.py` | a client that stops reading a streamed answer early (alone, and with a second request running) does not leave its tokens to the next request (server). |

For exact comparisons pass `--pcie-frac 0 --adapt-every 1000000` (and the scripts set `STRATA_IQ_MT_MIN=1`):

```
python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 8 --n 8 \
    --extra "--layer-split 12,24,36 --trim-stage-weights --batch-groups 4 --pcie-frac 0 --adapt-every 1000000"
python3 tools/batch_interleave_test.py --exe engine/strata --config strata-<model>.json \
    --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
python3 tools/parking_test.py --exe engine/strata --config strata-<model>.json \
    --extra "--layer-split 12,24,36 --conversation-cache-mib 8192 --conversation-cache-slots 4 --pcie-frac 0"
STRATA_KEY=<key> python3 tools/early_close_test.py http://127.0.0.1:8080
```

For single-GPU `--batch-mtp`, use a model with `rt/draft_vocab.bin` to exercise admission with a
shared draft-vocabulary head. This compares both slots' output against solo decoding:

```
python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 2 --n 2 --max-new 64 \
    --extra "--batch-mtp --pcie-frac 0 --adapt-every 1000000"
```

The two model-backed gates for the opt-in path drive the engine directly and write a JSON record plus stderr sidecars;
each refuses an existing output path and reports what it did **not** cover as `UNTESTED`:

```
# acceptance/rejection/parity with the compile-time test hook (validation build only; never for measurements)
python3 tools/unified_kv_batch_mtp_smoke.py --exe <hook-on strata> --config <model.json> \
    --output /tmp/mtp-core.json --mode correctness --context 1024 --gpu 0
# terminal target-only restore at partial-page offsets 1/2/3 (the same tool also has --mode limits and --mode lifecycle)
python3 tools/unified_kv_batch_mtp_smoke.py --exe <hook-on strata> --config <model.json> \
    --output /tmp/mtp-tails.json --mode tails [--tail-suffix new-user-turn] --context 1024 --gpu 0
# shared-pool pressure park/restore (the same tool also has --gate optional and --gate handoff)
python3 tools/unified_kv_batch_mtp_pressure_smoke.py --exe <strata> --config <model.json> \
    --output /tmp/mtp-pressure.json --gate pressure --context 4096 --cache-mib 4096 --mtp-window 128 --gpu 0
```

Run them on an isolated GPU with the model stopped: each starts its own private process, and `--batch-mtp` must be
set explicitly (the gates pass `--batch-mtp` plus `STRATA_BATCH_MTP=1`). The measured commands, hashes and results are
recorded in [INCREMENTAL_UNIFIED_KV.md](INCREMENTAL_UNIFIED_KV.md).

## Engine protocol (`--serve`)

On top of `GEN` / `GENI`:

| Line | Direction | Meaning |
| --- | --- | --- |
| `BGEN <slot> <max_new> [keys] <ids>` | in | read the prompt (as `GEN 1`), then continue in `<slot>` |
| `BGENI <slot> <max_new> [keys] <file> <ids>` | in | the same with images |
| `BADM <slot> <1/0>` | out | after the admission's `DONE`: 1 = it continues in the slot, 0 = it ended |
| `BT <slot> <id>` | out | a token of that slot |
| `BDONE <slot> <generated> <stop/length/cancel/pressure/handoff> <ms>` | out | the slot is free again; unified natural completion or handoff may keep an idle cache, while cancellation or pressure releases the backing |
| `BSTOP <slot>` | in | cancel that slot at its next window; in unified mode it releases active or paused ownership before `BDONE`, and a late stop of an already-finished owner owes no second acknowledgement |
| `HANDOFF` | in | with `INFO kv_handoff=1`: end solo decode with `DONE ... handoff ...`, keeping a valid completed-prefill cache; an incomplete read still cancels |
| `BHANDOFF <slot>` | in | with `INFO kv_handoff=1`: end active slot decode with `BDONE ... handoff ...`, keeping a target-only idle cache and checkpoints; a late handoff owes no second acknowledgement |
| `BYIELD <slot>` | in | the prompt being read gives way at its next chunk boundary; its part read waits in `<slot>` (the admission's own, or a free slot for a solo request) |
| `YIELDED <slot> <tokens>` | out | before the `DONE cancel` of a read that gave way: the request is sent again later and goes on from there |
| `INFO ... batch_slots=N` | out | the slots the engine runs (only with `--batch`) |

`tools/batch_test.py` drives the engine directly: the same prompts alone, then together, compared token by token,
and the aggregate rate.
