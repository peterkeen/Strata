#!/usr/bin/env python3
"""Private, model-backed SHARED STREAMING KV / parking regression.

Primary operator only, on an available GPU (never uses an existing service):
  python3 tools/unified_kv_streaming_smoke.py --exe /path/to/private/strata \
      --config /path/to/actual-model.json --output /tmp/streaming-262k.json --gpu 0
  python3 tools/unified_kv_streaming_smoke.py --exe /path/to/private/strata \
      --config /path/to/actual-model.json --output /tmp/streaming-long.json \
      --prompt-cells 20000 --max-new 256 --prefill 512 --vram-reserve-mib 2048 --gpu 0
  python3 tools/unified_kv_streaming_smoke.py --exe /path/to/private/strata \
      --config /path/to/actual-model.json --output /tmp/streaming-pressure.json \
      --context 24576 --resident 20480 --pressure --gpu 0

Defaults: --kv-unified --batch 2 --max-context 262144 --kv-resident 32768
--conversation-cache-mib 4096; fixed prefill defaults to 512. Actual config model
paths, KV format, environment and libraries are retained; expert placement and
greedy sampling are fixed. --vram-reserve-mib explicitly overrides only that base
setting (e.g. 2048 for capture headroom). --kv can select another format. K8V4 is
rejected before loading, not silently changed. Resident must be >=20480 and
strictly smaller than context; four-cell-aligned capacities avoid rounding.

Scenarios: solo/interleaved unequal-prompt parity; main A->B->A canonical parking
and restore with POSITIVE native stderr diagnostics; divergent idle-slot cached
prefixes ending at offsets 1/2/3; BSTOP acknowledgement then healthy same-slot
readmission. Optional --pressure tests LOGICAL HOST-BACKING reservation pressure
with an active output reservation >half the pool and a queued waiting admission.
It can be expensive: use the smaller context example, not 262K, for that stage.
--prompt-cells N is long mode: distinct N/N+4 prompts, cap 256 by default, and a
required live-prompt footprint greater than the GPU cache but fitting backing.
The cap must exceed ceil(second-prompt/prefill); this is ONLY a sizing check.
Actual first-slot survival through second BADM and subsequent BT is REQUIRED;
natural EOS or extra scheduler windows that prevent overlap fail, not skip.

Imports only reusable protocol/process/tokenizer helpers from unified_kv_smoke;
it does NOT run its resident-only scenarios or INFO assertions. Raw native stdout
uses a reader thread/Queue and absolute stage deadlines; stdin and cleanup are
bounded too. Every ERR fails. JSON retains raw protocol, commands, IDs, stderr
byte ranges/diagnostics, capacity evidence and failures; full stderr is beside
it as <output>.stderr.log. Existing evidence files are refused.

LIMITATIONS: short default prompts do not saturate the GPU cache or host backing.
Long mode exercises backing addresses beyond GPU slot capacity under the shared
allocator's uniqueness contract; no native backing-ID trace is available. Larger
--prompt-a-cells / --prompt-b-cells can also exercise longer model histories,
but native counters expose lookups/misses/approximate RAM reads, NOT CLOCK victim
or eviction counts. Cold loads, COW, restore and recycling also produce misses.
We collect these diagnostics but NEVER claim verified GPU CLOCK churn. Forced
cross-sequence CLOCK churn is covered by src/kernels/shared_kv_stream_parity.cpp,
not established by this model harness. Two rows also have a small sparse QSA
selection union relative to the resident minimum. Not a production/headroom gate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_kv_smoke as common

PAGE = common.PAGE_CELLS
MIN_RESIDENT = 20480
require = common.require
Request = common.Request
Protocol = common.Protocol
normal = common.normal
parity = common.parity


# Exact native diagnostics from generate.cpp; do not confuse cold misses with
# CLOCK eviction, or slot-prefix copies with canonical parked-image restores.
PATTERNS = {
    'allocation': re.compile(r'unified KV: (\d+) total backing cells shared by admission and (\d+) slots, '
                             r'(\d+) GPU-resident cells per layer( \(host-backed\))?'),
    'pinned': re.compile(r'KV streaming: (\d+) of (\d+) cells per QSA layer in VRAM, the K/V in '
                         r'([\d.]+) GiB of pinned RAM'),
    'park': re.compile(r'conversation cache: parked (\d+) tokens in ([\d.]+) ms; parked=(\d+) bytes=(\d+) '
                       r'evictions=(\d+) snapshot_bytes=(\d+)(?: reused_kv_bytes=(\d+))?'),
    'restore': re.compile(r'conversation cache: restored (\d+) tokens \((live|checkpoint)\) in ([\d.]+) ms; '
                          r'parked=(\d+) bytes=(\d+)'),
    'slot_copy': re.compile(r'strata batch: slot (\d+) gave back (\d+) tokens of this conversation '
                            r'\((all it holds|its turn checkpoint)\) in ([\d.]+) ms'),
    'stream_counters': re.compile(r'KV streaming: ([\d.]+)% of (\d+) block reads hit VRAM, ([\d.]+) MiB read '
                                  r'from RAM( - OVERFLOW \(too few resident cells\))?'),
}


def diagnostics(text):
    records = []
    for number, line in enumerate(text.splitlines(), 1):
        for kind, pattern in PATTERNS.items():
            match = pattern.search(line)
            if not match:
                continue
            f = match.groups()
            record = {'kind': kind, 'line': number, 'raw': line}
            if kind == 'allocation':
                record.update(backing_cells=int(f[0]), slots=int(f[1]), resident_cells=int(f[2]), host_backed=bool(f[3]))
            elif kind == 'pinned':
                record.update(resident_cells=int(f[0]), logical_cells=int(f[1]), pinned_gib=float(f[2]))
            elif kind == 'park':
                record.update(tokens=int(f[0]), ms=float(f[1]), parked=int(f[2]), bytes=int(f[3]),
                              evictions=int(f[4]), snapshot_bytes=int(f[5]), reused_kv_bytes=int(f[6] or 0))
            elif kind == 'restore':
                record.update(tokens=int(f[0]), source=f[1], ms=float(f[2]), parked=int(f[3]), bytes=int(f[4]))
            elif kind == 'slot_copy':
                record.update(slot=int(f[0]), tokens=int(f[1]), source=f[2], ms=float(f[3]))
            else:
                record.update(hit_percent=float(f[0]), lookups=int(f[1]), reported_ram_mib=float(f[2]), overflow=bool(f[3]))
            records.append(record)
            break
    return records


def verify_info(info, context, resident, cache_mib, kv=None):
    expected = {'context': str(context), 'kv_unified': '1', 'kv_capacity_cells': str(context),
                'kv_resident': str(resident), 'kv_resident_capacity_cells': str(resident), 'batch_slots': '2',
                'conversation_cache_mib': str(cache_mib), 'conversation_cache_slots': '4',
                'slot_cache': '1', 'lookup': '0', 'mtp_max': '1', 'spec': '2', 'pcie_frac': '0.00'}
    for key, wanted in expected.items():
        require(info.get(key) == wanted, f'streaming INFO requires {key}={wanted}, got {info.get(key)!r}')
    require(info.get('kv') in ('fp16', 'int8', 'q4_0'), 'INFO reports an unsupported streaming KV format')
    if kv:
        require(info['kv'] == kv, 'INFO KV format differs from actual configuration')
    return {'logical_context_cells_per_request': context, 'aggregate_shared_backing_cells': context,
            'aggregate_shared_gpu_cells_per_layer': resident, 'batch_slots': 2,
            'capacity_is_not_multiplied_or_partitioned_by_slots': True,
            'source': 'native INFO, cross-checked against host-backed allocation stderr'}


def verify_allocation(records, context, resident):
    allocation = [d for d in records if d['kind'] == 'allocation']
    require(len(allocation) == 1, 'required single unified backing/resident allocation diagnostic is missing/duplicated')
    a = allocation[0]
    require(a['host_backed'] and a['slots'] == 2 and a['backing_cells'] == context and a['resident_cells'] == resident,
            'native allocation diagnostic does not describe one shared host-backed streaming pool')
    pinned = [d for d in records if d['kind'] == 'pinned']
    require(len(pinned) == 1 and pinned[0]['resident_cells'] == resident and
            pinned[0]['logical_cells'] == context and pinned[0]['pinned_gib'] > 0,
            'missing positive pinned-host backing / GPU residency diagnostic')
    return {'allocation': a, 'pinned': pinned[0]}


def settings(cfg, exe, context, resident, cache_mib, gpu=None, prefill=512, kv=None, min_free_mib=None,
             vram_reserve_mib=None):
    """Independent streaming arguments, preserving the actual model and environment."""
    values = {'--batch': '2', '--batch-groups': '1', '--max-context': str(context), '--max-new': '1',
              '--kv-resident': str(resident), '--conversation-cache-mib': str(cache_mib),
              '--conversation-cache-slots': '4', '--pcie-frac': '0', '--adapt-every': '1000000',
              '--adapt-swaps': '0', '--peer-adapt-swaps': '0', '--spec': '2', '--mtp-max-t': '1',
              '--spec-min-p': '0', '--suffix-draft': '0', '--prompt-cache': '6', '--prefill': str(prefill),
              '--short-read': '64'}
    if kv:
        values['--kv'] = kv
    if min_free_mib is not None:
        values['--conversation-cache-min-free-mib'] = str(min_free_mib)
    if vram_reserve_mib is not None:
        values['--vram-reserve-mib'] = str(vram_reserve_mib)
    removed_values = {'--layer-split', '--split-device', '--expert-profile-save', '--expert-profile-save-every'}
    removed_flags = {'--serve', '--kv-unified', '--no-prefill-borrow', '--trim-stage-weights'}
    original = list(cfg['args'])
    args, i = [], 0
    while i < len(original):
        arg = original[i]
        if arg in values or arg in removed_values:
            require(i + 1 < len(original), f'config option missing a value: {arg}')
            i += 2
        elif arg in removed_flags:
            i += 1
        else:
            args.append(arg)
            i += 1
    args += ['--serve', '--kv-unified', '--no-prefill-borrow']
    for option, value in values.items():
        args.extend((option, value))
    effective_kv = 'fp16'
    for i, arg in enumerate(args):
        if arg == '--kv':
            require(i + 1 < len(args), 'missing --kv value')
            effective_kv = args[i + 1]
    effective_kv = 'q4_0' if effective_kv == 'q4' else effective_kv
    require(effective_kv in ('fp16', 'int8', 'q4_0'),
            'shared streaming cannot use configured K8V4/unknown KV; explicitly select --kv int8, q4_0 or fp16')
    cwd = str(Path(cfg.get('cwd') or os.getcwd()).expanduser().resolve())
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (cfg.get('env') or {}).items()})
    env['STRATA_IQ_MT_MIN'] = '1'
    devices = cfg.get('gpu')
    if gpu is None:
        gpu = cfg.get('hip_ordinal') if cfg.get('backend') == 'hip' else None
        if gpu is None and devices is not None:
            gpu = devices[0] if isinstance(devices, list) and devices else devices
    visibility = 'HIP_VISIBLE_DEVICES' if cfg.get('backend') == 'hip' else 'CUDA_VISIBLE_DEVICES'
    if gpu is not None and gpu != []:
        require(',' not in str(gpu), 'configuration must select one GPU (or explicitly pass --gpu)')
        env[visibility] = str(gpu)
    elif env.get(visibility):
        env[visibility] = env[visibility].split(',')[0]
    if visibility == 'CUDA_VISIBLE_DEVICES':
        env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    libs = [common.resolve_path(d, cwd) for d in cfg.get('lib_dirs') or []]
    if libs:
        var = 'PATH' if os.name == 'nt' else 'LD_LIBRARY_PATH'
        env[var] = os.pathsep.join(libs + ([env[var]] if env.get(var) else []))
    return [str(Path(exe).expanduser().resolve()), *args], cwd, env, effective_kv


class StreamingEngine(common.NativeEngine):
    def start(self, command, cwd, env, context, resident, cache_mib, kv, timeout):
        # Reuse bounded raw readers/writers/cleanup, NOT resident-only start().
        self.log = open(self.stderr_path, 'xb')
        self.p = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, start_new_session=(os.name == 'posix'))
        self.evidence['pid'] = self.p.pid
        self.reader = threading.Thread(target=self._read_stdout, name='streaming-stdout', daemon=True)
        self.reader.start()
        deadline = time.monotonic() + timeout
        info = None
        while True:
            event = self.next_event(deadline)
            if event['kind'] == 'INFO':
                require(info is None, 'duplicate startup INFO')
                info = event['fields']
                self.evidence['info'] = info
            elif event['kind'] == 'READY':
                require(info is not None, 'READY without shared streaming INFO')
                self.evidence['capacity_evidence'] = verify_info(info, context, resident, cache_mib, kv)
                require(event['context'] == context, 'READY logical context differs')
                self.evidence['startup_allocation'] = verify_allocation(
                    diagnostics(Path(self.stderr_path).read_text(errors='replace')), context, resident)
                return
            else:
                require(event['kind'] == 'OTHER', f'unexpected startup protocol: {event}')


class StreamingSuite(common.Suite):
    def run(self, name, requests, cancel=False, capacity_wait=False):
        self.engine.stage = name
        start = Path(self.engine.stderr_path).stat().st_size
        stage = {'name': name, 'passed': False, 'requests': []}
        self.evidence['stages'].append(stage)
        protocol = Protocol(requests)
        deadline = time.monotonic() + self.timeout
        stopped = False
        try:
            self.engine.send(*(r.command() for r in requests), deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                protocol.consume(event)
                trigger = (event['kind'] == 'BT' if capacity_wait else
                           event['kind'] == 'BADM' and event['continues'])
                if cancel and not stopped and trigger and event['slot'] == requests[0].slot:
                    if capacity_wait:
                        require(protocol.pending and protocol.pending[0] is requests[1] and not requests[1].tokens,
                                'logical pressure cancellation never reached waiting admission')
                    self.engine.send(f'BSTOP {requests[0].slot}', deadline=deadline)
                    stage['bstop_after_seq'] = event['seq']
                    stopped = True
            if cancel:
                req = requests[0]
                require(stopped and req.completion['kind'] == 'BDONE' and req.completion['finish'] == 'cancel',
                        'BSTOP did not cancel an active streamed slot (fixture may have ended on EOS)')
                require(stage['bstop_after_seq'] < req.completion['seq'] and len(req.tokens) < req.cap,
                        'stop acknowledgement did not release a remaining output reservation')
            return stage
        finally:
            stage['requests'] = [r.record() for r in requests]
            data = Path(self.engine.stderr_path).read_bytes()
            stage['stderr_byte_range'] = [start, len(data)]
            stage['diagnostics'] = diagnostics(data[start:].decode('utf-8', errors='replace'))
            self.save()


def verify_park_restore(park_stage, restore_stage, restored, reference, prefix, cache_mib):
    parks = [d for d in park_stage['diagnostics'] if d['kind'] == 'park' and d['tokens'] == prefix]
    require(parks and all(0 < d['bytes'] <= cache_mib * 1024 * 1024 and d['snapshot_bytes'] > 0 for d in parks),
            'A was not positively parked in canonical host snapshot storage before B')
    reused = restored.admission_done['reused']
    require(reused is not None and reused >= prefix, 'A main prefix was not actually reused after B')
    restores = [d for d in restore_stage['diagnostics'] if d['kind'] == 'restore' and d['tokens'] == reused]
    require(restores, 'missing positive canonical parked-conversation restore diagnostic for A (replay is not a pass)')
    parity(restored, reference)
    return {'park': parks[-1], 'restore': restores[-1], 'reused_prefix_cells': reused,
            'no_BGEN_slots_used_in_this_A_B_A_sequence': True}


def verify_cached_branch(stage, req, reference, shared_cells, source_slot):
    parity(req, reference)
    require(shared_cells % PAGE in (1, 2, 3), 'branch is not a partial-page prefix')
    require(req.admission_done['reused'] is not None and req.admission_done['reused'] >= shared_cells,
            'divergent cached prefix was not actually reused')
    copies = [d for d in stage['diagnostics'] if d['kind'] == 'slot_copy' and
              d['slot'] == source_slot and d['tokens'] >= shared_cells]
    require(copies, 'branch was replayed/restored from parking instead of cloning the cached idle-slot prefix')
    return {'shared_prefix_cells': shared_cells, 'partial_page_offset': shared_cells % PAGE, 'slot_copy': copies[-1]}


def make_fixtures(tok, context, a_cells, b_cells, cap):
    def chat(question):
        return tok.encode(f'<|im_start|>user\n{question}<|im_end|>\n'
                          '<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
    filler = tok.encode(' alpha beta gamma delta epsilon zeta eta theta')
    require(filler, 'empty filler tokenizer result')

    def padded(label, cells):
        # Place data BEFORE the assistant header, not after a completed reply.
        head = tok.encode(f'<|im_start|>user\n{label}: Read this data, then list numbered observations '
                          'from 1 through 1000 in full sentences. Keep listing; do not summarize.\nData:\n',
                          parse_special=True)
        tail = tok.encode('\nUsing the data above, list numbered observations 1 through 1000. '
                          'Write a full sentence for every item. Do not summarize or stop early.\n'
                          '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
        require(cells >= len(head) + len(tail), 'requested prompt size is too small for the chat fixture')
        return head + (filler * (cells // len(filler) + 1))[:cells - len(head) - len(tail)] + tail

    a, b = padded('Streaming A animals', a_cells), padded('Streaming B mathematics', b_cells)
    require(len(a) != len(b), 'interleaved prompts must have unequal lengths')
    require(sum((len(p) + cap + PAGE - 1) // PAGE for p in (a, b)) <= context // PAGE,
            'interleaved reservations cannot fit aggregate logical backing; reduce prompt sizes')
    for p in (a, b, padded('Parking A astronomy', 192), padded('Parking B cooking', 288)):
        require(len(p) + cap + 8 <= context, 'fixture exceeds logical context')
    return chat, padded, filler, a, b


def long_plan(a, b, cap, prefill, context, resident):
    """Conservative unique written-prefix bound, not a GPU eviction counter."""
    chunks = (len(b) + prefill - 1) // prefill
    require(cap > chunks, 'long-mode max-new must exceed ceil(second-prompt/prefill) to leave overlap headroom')
    common_prefix = 0
    for left, right in zip(a, b):
        if left != right:
            break
        common_prefix += 1
    shared_upper_bound = ((common_prefix + PAGE - 1) // PAGE) * PAGE
    written_lower_bound = len(a) + len(b) - shared_upper_bound
    reserved_pages = sum((len(p) + cap + PAGE - 1) // PAGE for p in (a, b))
    require(written_lower_bound > resident,
            'long-mode distinct live prompt footprint must exceed shared GPU capacity, not just output reservations')
    require(reserved_pages <= context // PAGE, 'long-mode reservations exceed logical backing')
    return {'prompt_cells': [len(a), len(b)], 'second_prompt_chunks_estimate': chunks, 'max_new': cap,
            'common_prompt_cells_upper_bound': shared_upper_bound, 'unique_live_prompt_cells_lower_bound': written_lower_bound,
            'aggregate_reservation_cells': reserved_pages * PAGE, 'gpu_resident_cells': resident,
            'shared_backing_cells': context, 'backing_ids_above_gpu_slot_capacity_required_by_allocator_contract': True,
            'native_backing_id_trace_available': False, 'clock_churn_verified': False,
            'overlap_sizing_limit': 'Multiple decode windows per prefill chunk and natural EOS are possible; '
                                   'protocol evidence, not this chunk estimate, must prove actual overlap.'}


def verify_interleaving(pair, references):
    for req, reference in zip(pair, references):
        parity(req, reference)
        require(req.badm and req.badm['continues'], 'fixture ended at admission; interleaved decode was not tested')
    first, second = pair
    require(first.badm['seq'] < second.badm['seq'] < first.completion['seq'],
            'first slot completed before second admission; actual overlap was not observed')
    progress = [e for e in first.token_events if e['kind'] == 'BT' and e['seq'] > second.badm['seq']]
    require(progress, 'first slot did not progress after second admission')
    return {'first_badm_seq': first.badm['seq'], 'second_badm_seq': second.badm['seq'],
            'first_bdone_seq': first.completion['seq'], 'first_slot_tokens_after_second_admission': len(progress),
            'actual_protocol_overlap_confirmed': True}


def run_suite(suite, tok, cap, a_cells, b_cells, pressure=False, long_mode=False, prefill=512):
    context = suite.context
    chat, padded, filler, a, b = make_fixtures(tok, context, a_cells, b_cells, cap)

    # Run BEFORE any BGEN: a slot cannot accidentally supply the A restore.
    pa, pb = padded('Parking A astronomy', 192), padded('Parking B cooking', 288)
    first = suite.solo('parking-reference-A-seed', pa, 1)
    continuation = pa + first.tokens + tok.encode('\nContinue the numbered observations:\n')
    ref = suite.solo('parking-reference-A-continuation', continuation, cap)
    reset = suite.solo('parking-A-main-seed', pa, 1)
    parity(reset, first)
    suite.solo('parking-switch-to-B', pb, cap)
    park_stage = suite.evidence['stages'][-1]
    restored = suite.solo('parking-return-to-A', continuation, cap)
    restore_stage = suite.evidence['stages'][-1]
    restore_stage['passed'] = False  # Solo completion alone is not the parking regression gate.
    try:
        restore_stage['parking_evidence'] = verify_park_restore(park_stage, restore_stage, restored, ref, len(pa),
                                                              suite.evidence['cache_mib'])
        restore_stage['passed'] = True
    finally:
        suite.save()

    solo_a = suite.solo('solo-interleave-A', a, cap)
    solo_b = suite.solo('solo-interleave-B', b, cap)
    pair = [Request('interleaved-A', a, cap, 0), Request('interleaved-B', b, cap, 1)]
    stage = suite.run('shared-streaming-long-interleaved-parity' if long_mode else
                      'shared-streaming-interleaved-parity', pair)
    stage['overlap_evidence'] = verify_interleaving(pair, (solo_a, solo_b))
    if long_mode:
        stage['long_prompt_evidence'] = long_plan(a, b, cap, prefill, context, suite.evidence['resident'])
    stage['passed'] = True
    suite.save()

    for offset in (1, 2, 3):
        # cap 8 leaves exactly seven consumed outputs in the idle slot.
        size = 128 + (offset - 7 - 128) % PAGE
        seed = padded(f'Cached seed offset {offset}', size)
        seed_ref = suite.solo(f'solo-cached-seed-{offset}', seed, 8)
        require(len(seed_ref.tokens) == 8, 'cached seed ended early; no partial-tail fixture established')
        shared = seed + seed_ref.tokens[:-1]
        branches = [shared + tok.encode(text) for text in ('\nContinue with even numbers:\n',
                                                          '\nContinue with odd numbers:\n')]
        refs = [suite.solo(f'solo-branch-{offset}-{i}', p, 16) for i, p in enumerate(branches)]
        seeded = Request(f'cached-slot-seed-{offset}', seed, 8, 0)
        stage = suite.run(f'populate-cached-slot-{offset}', [seeded])
        parity(seeded, seed_ref)
        require(seeded.badm['continues'], 'cached seed did not populate the batch slot')
        stage['passed'] = True
        for i, (prompt, reference) in enumerate(zip(branches, refs)):
            req = Request(f'cached-branch-{offset}-{i}', prompt, 16, 1 - i)
            stage = suite.run(f'shared-streaming-partial-tail-{offset}-{i}', [req])
            stage['cached_prefix_evidence'] = verify_cached_branch(stage, req, reference, len(shared), 0)
            stage['passed'] = True
            suite.save()

    cancelled = Request('streamed-cancel-active', a, 512, 0)
    stage = suite.run('streamed-BSTOP', [cancelled], cancel=True)
    stage['passed'] = True
    suite.save()
    healthy = Request('healthy-same-slot-after-stop', b, cap, 0)
    stage = suite.run('streamed-healthy-readmission', [healthy])
    parity(healthy, solo_b)
    require(cancelled.completion['seq'] < healthy.token_events[0]['seq'], 'readmission preceded stop acknowledgement')
    stage['previous_bdone_seq'] = cancelled.completion['seq']
    stage['passed'] = True
    suite.save()

    if pressure:
        # Pressure is on authoritative backing, not on the smaller GPU cache.
        base = chat('Distinct pressure waiter. Read this data and list numbered facts.\n')
        require(len(base) < context // 2, 'pressure waiter base is too large')
        waiter = base + (filler * (context // len(filler) + 1))[:context // 2 - len(base)]
        reference = suite.solo('solo-logical-pressure-waiter', waiter, 16)
        active = Request('logical-pressure-active', a, 3 * context // 4, 0)
        waiting = Request('logical-pressure-waiting', waiter, 16, 1)
        require(len(a) + active.cap + 8 <= context, 'pressure active prompt too large; reduce --prompt-a-cells')
        stage = suite.run('shared-host-backing-pressure-BSTOP', [active, waiting], cancel=True, capacity_wait=True)
        stage['pressure_evidence'] = common.verify_pressure(active, waiting, context, True)
        parity(waiting, reference)
        stage['passed'] = True
        suite.save()


def finalize_diagnostics(evidence, stderr_path, require_upload=False):
    records = diagnostics(Path(stderr_path).read_text(errors='replace')) if Path(stderr_path).exists() else []
    evidence['native_diagnostics'] = records
    counters = [d for d in records if d['kind'] == 'stream_counters']
    evidence['streaming_observations'] = {
        'counter_samples': len(counters), 'max_reported_lookups': max((d['lookups'] for d in counters), default=0),
        'max_reported_ram_mib': max((d['reported_ram_mib'] for d in counters), default=0),
        'host_reads_reported': any(d['reported_ram_mib'] > 0 for d in counters),
        'clock_churn_verified': False,
        'counter_scope': 'native cumulative shared-cache summaries; RAM MiB rounded and estimated by native engine',
        'limitation': 'Misses can be cold loads, COW, restore or recycled backing IDs. No native CLOCK victim/eviction '
                      'counter is reported. Short prompts/two sparse-selection rows do not establish GPU cache churn; '
                      'forced churn is covered by the shared_kv_stream_parity kernel test, not this harness.'}
    require(counters, 'no native KV streaming counter summaries were emitted')
    require(not any(d['overflow'] for d in counters), 'native KV streaming reported resident selection OVERFLOW')
    if require_upload:
        require(evidence['streaming_observations']['host_reads_reported'],
                'no positive native RAM-read diagnostic; remove --require-upload or enlarge prompt fixtures')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exe', required=True, type=Path)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path, help='NEW JSON evidence file')
    ap.add_argument('--context', type=int, default=262144)
    ap.add_argument('--resident', type=int, default=32768)
    ap.add_argument('--cache-mib', type=int, default=4096)
    ap.add_argument('--min-free-mib', type=int, help='explicit parking RAM floor; default retains actual config/default')
    ap.add_argument('--kv', choices=('int8', 'q4_0', 'fp16'), help='default retains actual config KV format')
    ap.add_argument('--gpu', help='one physical CUDA/HIP ordinal/UUID; default first configured GPU')
    ap.add_argument('--vram-reserve-mib', type=int, help='override base config capture headroom, e.g. 2048')
    ap.add_argument('--prefill', type=int, default=512)
    ap.add_argument('--max-new', type=int, help='default 32; 256 in --prompt-cells long mode')
    ap.add_argument('--prompt-cells', type=int, help='long mode: distinct N/N+4 prompts exceeding shared GPU capacity')
    ap.add_argument('--prompt-a-cells', type=int, help='individual prompt size, default 160; excludes --prompt-cells')
    ap.add_argument('--prompt-b-cells', type=int, help='individual prompt size, default 416; excludes --prompt-cells')
    ap.add_argument('--pressure', action='store_true', help='additional LOGICAL backing pressure; expensive at 262K')
    ap.add_argument('--require-upload', action='store_true', help='require positive RAM-read diagnostic, NOT CLOCK churn')
    ap.add_argument('--startup-timeout', type=float, default=900)
    ap.add_argument('--stage-timeout', type=float, default=600)
    ap.add_argument('--cleanup-timeout', type=float, default=15)
    a = ap.parse_args(argv)
    if not MIN_RESIDENT <= a.resident < a.context <= 2147483639 or a.context % PAGE or a.resident % PAGE:
        ap.error('four-aligned context/resident required, with 20480 <= resident < context < 2^31-8')
    if a.cache_mib <= 0 or (a.min_free_mib is not None and a.min_free_mib < 0):
        ap.error('cache-mib must be positive; min-free-mib must be nonnegative')
    if a.vram_reserve_mib is not None and a.vram_reserve_mib < 0:
        ap.error('vram-reserve-mib must be nonnegative')
    long_mode = a.prompt_cells is not None
    if long_mode:
        if a.prompt_a_cells is not None or a.prompt_b_cells is not None:
            ap.error('--prompt-cells excludes --prompt-a-cells / --prompt-b-cells')
        a.prompt_a_cells, a.prompt_b_cells = a.prompt_cells, a.prompt_cells + PAGE
    else:
        a.prompt_a_cells = 160 if a.prompt_a_cells is None else a.prompt_a_cells
        a.prompt_b_cells = 416 if a.prompt_b_cells is None else a.prompt_b_cells
    if a.max_new is None:
        a.max_new = 256 if long_mode else 32
    if not 64 <= a.prefill <= 4096 or a.prefill % 64 or not 2 <= a.max_new <= 512:
        ap.error('prefill must be 64..4096 in multiples of 64; max-new must be 2..512')
    if min(a.prompt_a_cells, a.prompt_b_cells) < 96:
        ap.error('prompt sizes must be at least 96 tokens')
    if long_mode and a.max_new <= (a.prompt_b_cells + a.prefill - 1) // a.prefill:
        ap.error('long-mode max-new must exceed ceil(second-prompt/prefill); use e.g. --max-new 256 --prefill 512')
    if a.gpu is not None and (not a.gpu.strip() or ',' in a.gpu):
        ap.error('--gpu must select exactly one device')
    if any(not 0 < t < float('inf') for t in (a.startup_timeout, a.stage_timeout, a.cleanup_timeout)):
        ap.error('timeouts must be positive and finite')
    output = a.output.expanduser().resolve()
    stderr_path = Path(str(output) + '.stderr.log')
    if output.exists() or stderr_path.exists():
        ap.error('evidence/log already exists; use a new --output')
    with output.open('x', encoding='utf-8') as f:
        f.write('{}\n')
    evidence = {'schema': 1, 'harness': 'shared-streaming', 'passed': False, 'config': str(a.config.resolve()),
                'exe': str(a.exe.resolve()), 'context': a.context, 'resident': a.resident, 'cache_mib': a.cache_mib,
                'page_cells': PAGE, 'stderr_path': str(stderr_path), 'commands': [], 'stdout': [], 'stages': [],
                'logical_backing_pressure_requested': a.pressure, 'long_prompt_mode_requested': long_mode,
                'workload': {'prompt_cells': [a.prompt_a_cells, a.prompt_b_cells], 'max_new': a.max_new,
                             'prefill': a.prefill, 'vram_reserve_mib_override': a.vram_reserve_mib},
                'timeouts': {'startup_s': a.startup_timeout, 'stage_s': a.stage_timeout, 'cleanup_s': a.cleanup_timeout}}
    engine = StreamingEngine(evidence, stderr_path, a.cleanup_timeout)
    suite = StreamingSuite(engine, evidence, output, a.stage_timeout, a.context)
    success = False
    try:
        cfg = json.loads(a.config.read_text(encoding='utf-8'))
        command, cwd, env, kv = settings(cfg, a.exe, a.context, a.resident, a.cache_mib, a.gpu,
                                        a.prefill, a.kv, a.min_free_mib, a.vram_reserve_mib)
        evidence.update(command=command, cwd=cwd, config_env=cfg.get('env') or {},
                        device_env={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                                                       'CUDA_DEVICE_ORDER', 'STRATA_IQ_MT_MIN') if k in env})
        suite.save()
        tok = common.tokenizer(common.resolve_path(cfg['tokenizer'], cwd))
        _, _, _, pa, pb = make_fixtures(tok, a.context, a.prompt_a_cells, a.prompt_b_cells, a.max_new)
        if long_mode:
            evidence['long_prompt_plan'] = long_plan(pa, pb, a.max_new, a.prefill, a.context, a.resident)
            suite.save()
        engine.start(command, cwd, env, a.context, a.resident, a.cache_mib, kv, a.startup_timeout)
        run_suite(suite, tok, a.max_new, a.prompt_a_cells, a.prompt_b_cells, a.pressure, long_mode, a.prefill)
        require(evidence['stages'] and all(s['passed'] for s in evidence['stages']), 'incomplete streaming stages')
        success = True
    except (Exception, KeyboardInterrupt) as exc:
        evidence['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'stage': engine.stage}
    finally:
        try:
            engine.close(success)
            if success:
                cleanup = evidence['cleanup']
                require(cleanup.get('returncode') == 0 and not cleanup.get('quit_timeout'), 'native QUIT did not exit cleanly')
                require(cleanup.get('reader_stopped') and cleanup.get('writer_stopped'), 'native pipe thread did not stop')
                require(not cleanup.get('protocol_errors'), 'malformed native shutdown protocol')
                require(not any(e.get('kind') == 'ERR' for e in evidence['stdout']), 'unexpected trailing native ERR')
                trailing = evidence['stdout'][cleanup['stdout_start_seq']:]
                require(all(e.get('kind', 'OTHER') == 'OTHER' for e in trailing), 'protocol after final completion')
                finalize_diagnostics(evidence, stderr_path, a.require_upload)
            elif stderr_path.exists():
                evidence['native_diagnostics'] = diagnostics(stderr_path.read_text(errors='replace'))
        except (Exception, KeyboardInterrupt) as exc:
            success = False
            evidence['cleanup_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        evidence['passed'] = success
        suite.save()
    print(f'{"PASS" if success else "FAIL"}: {output}; stderr: {stderr_path}', flush=True)
    if not success:
        print(json.dumps(evidence.get('failure') or evidence.get('cleanup_failure')), file=sys.stderr)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
