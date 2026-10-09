#!/usr/bin/env python3
"""Independent incremental unified-KV actual-model gates; PRIVATE subprocess only.

Operator examples (fresh output names; no build/deploy performed):
  python3 tools/incremental_kv_smoke.py --exe /private/strata --config model.json \
      --output /tmp/inc-1024-UNIQUE.json --context 1024 --cache-mib 0
  # Repeat disabled parking with a genuinely insufficient optional snapshot budget:
  python3 tools/incremental_kv_smoke.py --exe /private/strata --config model.json \
      --output /tmp/inc-tiny-UNIQUE.json --context 1024 --cache-mib 1 --expect-replay
  # Positive canonical parking (requires enough RAM), same frozen target math:
  python3 tools/incremental_kv_smoke.py --exe /private/strata --config model.json \
      --output /tmp/inc-restore-UNIQUE.json --gate parking --context 32768 --cache-mib 4096
  # Large backing/streaming overlap without an expensive 262K exhaustion run:
  python3 tools/incremental_kv_smoke.py --exe /private/strata --config model.json \
      --output /tmp/inc-stream-UNIQUE.json --gate overlap --context 262144 \
      --resident 32768 --cache-mib 4096 --vram-reserve-mib 2048 \
      --prompt-a-cells 35000 --prompt-b-cells 35004 --prefill 1024 --prefix 128

--gate late-stop is a focused transport ownership audit: two cap-16 solo
references, ONE finite pressure pair, then late BSTOP + immediate same-slot
healthy BGEN after pressure and again after natural completion. No stopped
owner remains, so neither late BSTOP may emit another BDONE (NoAckOwned).
It deliberately does not rerun full pressure-output parity or FIFO resumption;
those are separate gates. Select a fresh output when testing a rebuilt engine.

--prefill defaults to 64 to preserve small/yield fixtures; an explicit 1024
uses the same native prefill path with the actual config's fused-prefill env.
Allowed fixed chunks are 64..4096 in multiples of 64, matching the streaming
harness. Larger chunks change scheduling; protocol evidence must still prove
both rows emitted concurrently. They do not change output allowances or caps.

Imports existing process/tokenizer/config helpers, not existing reservation-pressure
scenarios. INFO must explicitly say kv_incremental=1 kv_reserve_ahead=256.
Temperature (default greedy 0), positive seed, IQ math and expert placement are
fixed across references and replay. --temperature 0.3 --seed 12345 enables an
actual sampled-pressure parity gate; it never adds rng_offset.
--require-pressure-restore adds a separate source-pressure / cold small GEN /
MAIN GEN restore probe, requiring TARGET_ONLY restore and decode T=1 diagnostics.
FIFO retries may legitimately skip optional restores and are NOT required to
restore. --coherence-mtp-max 2 (or 4) explicitly enables a GREEDY MTP audit: solo
references must offer drafts, while restored MAIN GEN must offer zero drafts.
Default model/parity settings still freeze mtp_max=1. A pressure BDONE counts admission T + BT for THIS attempt, releases
backing, and is NOT a logical completion. Replay includes the unfed last output;
absolute input position drives RNG, so there is deliberately NO rng_offset.

Overlap advertises remaining context (minus native's eight-cell guard), consumes
only a prefix, then BSTOPs. Pressure uses FINITE targets, actual SMALL backing,
and long-count prompts. EOS before pressure, absent overlap, absent positive
restore, or absence of pressure/resume for either logical request FAILS, never
skips. Pressure owners requeue immediately in FIFO command order, including when
another admission is pending. Two reversed admission orders are exercised. Both
owners must pressure-stop, replay and finally complete in EACH stage; every
emitted token is checked against its frozen solo reference.

At 512 cells two 256-cell headrooms may not fit; fixture preflight rejects that
configuration rather than crediting serialized admission as an exhaustion test.
1024 is the default. --pressure-target / prompt sizes can select another fitting
SMALL layout. No diagnostics are promoted into a claim of GPU CLOCK eviction.
JSON + raw stdout, command IDs, per-attempt replay histories and stderr byte
ranges remain on failure. Timeouts are absolute per stage, including resumes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_kv_smoke as common
import unified_kv_streaming_smoke as streaming

require = common.require
PAGE = common.PAGE_CELLS
AHEAD = 256
SEED = 12345
GUARD = 8


def pages(cells):
    require(cells >= 0, 'negative cell count')
    return (cells + PAGE - 1) // PAGE


def remaining_context(context, prompt):
    budget = context - len(prompt) - GUARD
    require(budget > 1, 'prompt leaves no logical output budget')
    return budget


def growth_plan(prompts, targets, context, ahead=AHEAD):
    """Page arithmetic is sizing evidence, NOT proof that pressure was exercised."""
    require(len(prompts) == len(targets) == 2, 'two prompts/targets required')
    require(context % PAGE == 0 and ahead > 0, 'invalid rolling capacity')
    for prompt, target in zip(prompts, targets):
        require(prompt and 1 < target <= remaining_context(context, prompt), 'target exceeds logical budget')
    initial = [pages(len(p) + min(t, ahead)) for p, t in zip(prompts, targets)]
    nominal = [pages(len(p) + t) for p, t in zip(prompts, targets)]
    require(sum(initial) <= context // PAGE, 'prompt + initial headrooms do not fit; use 1024 or resize fixtures')
    require(sum(nominal) > context // PAGE, 'nominal histories do not exhaust backing')
    return {'pool_pages': context // PAGE, 'initial_pages_upper_bound': initial,
            'nominal_pages': nominal, 'reserve_ahead_cells': ahead,
            'initial_histories_and_headroom_fit': True, 'combined_nominal_exceeds_pool': True,
            'pressure_requires_native_BDONE_not_reservation_arithmetic': True}


def verify_info(info, context, resident, cache_mib, mtp_max=1):
    expected = {'kv_unified': '1', 'kv_incremental': '1', 'kv_reserve_ahead': str(AHEAD),
                'context': str(context), 'kv_capacity_cells': str(context), 'batch_slots': '2',
                'kv_resident': str(resident), 'conversation_cache_mib': str(cache_mib),
                'slot_cache': '1', 'lookup': '0', 'mtp_max': str(mtp_max),
                'spec': str(max(2, mtp_max)), 'pcie_frac': '0.00'}
    if resident:
        expected['kv_resident_capacity_cells'] = str(resident)
    for key, wanted in expected.items():
        require(info.get(key) == wanted, f'INFO requires {key}={wanted}, got {info.get(key)!r}')


def settings(cfg, exe, context, resident, cache_mib, gpu=None, vram_reserve_mib=2048,
             coherence_mtp_max=1, prefill=64):
    # This helper preserves cwd, env, visibility, libs, model paths and KV format.
    # It also freezes expert placement / drafts; unlike production HTTP settings.
    require(64 <= prefill <= 4096 and prefill % 64 == 0,
            'prefill must be 64..4096 in multiples of 64')
    command, cwd, env, kv = streaming.settings(
        cfg, exe, context, resident, cache_mib, gpu, prefill=prefill,
        min_free_mib=0, vram_reserve_mib=vram_reserve_mib)
    # An existing --slots alias or draft lookup must not countermand the gate.
    for option in ('--slots', '--kv-reserve-ahead'):
        while option in command:
            index = command.index(option)
            require(index + 1 < len(command), f'missing value: {option}')
            del command[index:index + 2]
    if coherence_mtp_max > 1:
        require('--mtp' in command, 'MTP coherence audit requires the actual config --mtp weights')
        for option in ('--spec', '--mtp-max-t'):
            command[command.index(option) + 1] = str(coherence_mtp_max)
    # --suffix-draft 0 from streaming.settings disables INFO lookup. There is
    # no native --lookup option; do not invent one from the INFO field name.
    return command, cwd, env, kv


class Engine(common.NativeEngine):
    def start(self, command, cwd, env, context, resident, cache_mib, timeout, mtp_max=1):
        self.log = open(self.stderr_path, 'xb')
        self.p = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log,
                                  start_new_session=(os.name == 'posix'))
        self.evidence['pid'] = self.p.pid
        self.reader = threading.Thread(target=self._read_stdout, name='incremental-stdout', daemon=True)
        self.reader.start()
        deadline = time.monotonic() + timeout
        info = None
        while True:
            event = self.next_event(deadline)
            if event['kind'] == 'INFO':
                require(info is None, 'duplicate INFO')
                info = event['fields']
                self.evidence['info'] = info
            elif event['kind'] == 'READY':
                require(info is not None, 'READY without INFO')
                verify_info(info, context, resident, cache_mib, mtp_max)
                require(event['context'] == context, 'READY context differs')
                if resident:
                    self.evidence['allocation'] = streaming.verify_allocation(
                        streaming.diagnostics(Path(self.stderr_path).read_text(errors='replace')), context, resident)
                return
            else:
                require(event['kind'] == 'OTHER', f'unexpected startup protocol: {event}')


@dataclass
class Attempt(common.Request):
    temperature: float = 0.0
    seed: int = SEED

    def record(self):
        result = super().record()
        # Old helper calls this a full reservation; incremental KV does NOT.
        result['nominal_allowance_cells'] = result.pop('reservation_cells')
        result['nominal_allowance_pages'] = result.pop('reservation_pages')
        result['initial_pages_upper_bound'] = pages(len(self.prompt) + min(self.cap, AHEAD))
        result['seed'] = self.seed
        result['temperature'] = self.temperature
        result['rng_offset_added'] = False
        return result

    def command(self):
        head = f'GEN {self.cap}' if self.slot is None else f'BGEN {self.slot} {self.cap}'
        return head + f' temperature={self.temperature} seed={self.seed} ' + ','.join(map(str, self.prompt))


@dataclass
class LogicalRequest:
    name: str
    original: list[int]
    budget: int
    slot: int
    seed: int = SEED
    temperature: float = 0.0
    _sampling: tuple = field(init=False, repr=False)
    tokens: list[int] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    finish: str | None = None

    def __post_init__(self):
        require(0 < self.seed < 2**63 and math.isfinite(self.temperature) and self.temperature >= 0,
                'sampling requires a positive seed and finite nonnegative temperature')
        self._sampling = (self.seed, self.temperature)

    def attempt(self):
        require(self.finish is None and len(self.tokens) < self.budget, 'resubmitting completed request')
        require((self.seed, self.temperature) == self._sampling, 'sampling keys changed during replay')
        # ALL tokens: the last returned token has not yet been fed into the KV.
        req = Attempt(f'{self.name}-attempt-{len(self.attempts)}',
                      self.original + self.tokens, self.budget - len(self.tokens), self.slot,
                      temperature=self.temperature, seed=self.seed)
        self.attempts.append(req)
        return req

    def accept(self, req, reference):
        require(self.attempts and self.attempts[-1] is req, 'unknown replay attempt')
        require((req.seed, req.temperature) == self._sampling, 'attempt sampling keys differ from original request')
        require(req.prompt == self.original + self.tokens, 'replay lost the unfed token/history')
        require(req.cap == self.budget - len(self.tokens), 'replay reset the logical budget')
        require(req.completion is not None, 'attempt missing completion')
        finish = req.completion['finish']
        require(finish in ('pressure', 'length', 'stop', 'cancel'), f'unknown finish: {finish}')
        if finish == 'pressure':
            require(req.completion['kind'] == 'BDONE' and req.tokens and len(req.tokens) < req.cap,
                    'pressure must be a nonempty, nonfinal batch attempt')
        self.tokens.extend(req.tokens)
        require(len(self.tokens) <= self.budget, 'logical budget exceeded')
        require(self.tokens == reference.tokens[:len(self.tokens)], f'{self.name}: emitted token solo parity differs')
        if finish in ('length', 'stop'):
            require(self.tokens == reference.tokens, f'{self.name}: final solo parity/length differs')
            if finish == 'length':
                require(len(self.tokens) == self.budget, 'logical length finish before budget')
            self.finish = finish
        # cancel can be used as a deliberate suspend, or an abandoned owner.

    def record(self):
        return {'name': self.name, 'original_ids': self.original, 'logical_budget': self.budget,
                'slot': self.slot, 'seed': self.seed, 'temperature': self.temperature,
                'rng_offset_added': False, 'tokens': self.tokens, 'finish': self.finish,
                'attempts': [r.record() for r in self.attempts],
                'resume_position': len(self.original) - 1 + len(self.tokens)}


def overlap_proof(pair):
    require(all(r.badm and r.badm['continues'] and r.completion for r in pair), 'pair did not reach live batch rows')
    both_admitted = max(r.badm['seq'] for r in pair)
    first_end = min(r.completion['seq'] for r in pair)
    require(both_admitted < first_end, 'serialized admission / early EOS; no overlap')
    progress = [[e['seq'] for e in r.token_events if e['kind'] == 'BT' and both_admitted < e['seq'] < first_end]
                for r in pair]
    require(all(progress), 'both rows must actually emit BT before either finishes')
    return {'both_admitted_seq': both_admitted, 'first_completion_seq': first_end,
            'concurrent_BT_sequences': progress, 'both_rows_emitted_before_either_finished': True}


def pressure_diagnostics(text):
    records = []
    pattern = re.compile(r'pressure (parked|replay) target-only (\d+) tokens; parked=(\d+) bytes=(\d+)')
    for number, line in enumerate(text.splitlines(), 1):
        match = pattern.search(line)
        if match:
            result, tokens, count, size = match.groups()
            records.append({'kind': 'pressure_target_only', 'result': result, 'tokens': int(tokens),
                            'parked': int(count), 'bytes': int(size), 'line': number, 'raw': line})
        elif 'TARGET_ONLY restore: private MTP proposals suppressed until full replay' in line:
            records.append({'kind': 'target_only_restore', 'line': number, 'raw': line})
        elif 'TARGET_ONLY decode: T=1, MTP/suffix proposals disabled' in line:
            records.append({'kind': 'target_only_decode_t1', 'line': number, 'raw': line})
        elif 'pressure cache' in line and ('miss' in line or 'replay' in line):
            records.append({'kind': 'pressure_replay_miss', 'line': number, 'raw': line})
    return records


def verify_pressure_restore(logical, records, diagnostics, coherence_records=None):
    if coherence_records is not None:
        require(any(d['kind'] == 'target_only_restore' for d in coherence_records) and
                any(d['kind'] == 'target_only_decode_t1' for d in coherence_records),
                'MAIN GEN did not report TARGET_ONLY restore and decode T=1 suppression')
    proof = []
    for item in logical:
        for before, after in zip(item.attempts, item.attempts[1:]):
            prefix = len(before.prompt) + len(before.tokens) - 1
            parks = [d for d in records if d['kind'] == 'pressure_target_only' and
                     d['result'] == 'parked' and d['tokens'] == prefix and d['bytes'] > 0]
            reused = after.admission_done['reused']
            restores = [d for d in diagnostics if d['kind'] == 'restore' and d['tokens'] == reused]
            require(parks and reused is not None and reused >= prefix and restores,
                    'missing target-only pressure snapshot / matching positive canonical restore')
            proof.append({'owner': item.name, 'park': parks[-1], 'restore': restores[-1],
                          'last_unfed_token_replayed': after.prompt[-1] == before.tokens[-1],
                          'mtp_draft_coherence_exercised': False})
    require(proof, 'no target-only pressure restore exercised')
    return proof


def rolling_growth_evidence(req):
    common.normal(req)
    initial_end = len(req.prompt) + min(req.cap, AHEAD)
    consumed_end = len(req.prompt) + len(req.tokens) - 1  # last output is unfed
    new_pages = pages(consumed_end) - pages(initial_end)
    require(new_pages > 0, 'EOS/short target never wrote beyond initial 256-cell headroom')
    return {'initial_pages_upper_bound': pages(initial_end), 'consumed_history_pages': pages(consumed_end),
            'minimum_pages_beyond_initial_headroom': new_pages,
            'source': 'INFO bounded headroom contract + actual consumed output IDs; not a native allocation trace'}


def pressure_proof(pair, plan):
    proof = overlap_proof(pair)
    pressure = [r for r in pair if r.completion['finish'] == 'pressure']
    require(pressure, 'no actual pressure BDONE: natural EOS/length/serialization is NOT exhaustion coverage')
    require(all(len(r.tokens) > PAGE for r in pressure), 'no rolling growth before pressure')
    proof.update(plan=plan, pressure_slots=[r.slot for r in pressure],
                 pressure_sequences=[r.completion['seq'] for r in pressure],
                 evidence_source='explicit BDONE pressure after rolling BT growth; native releases backing before BDONE',
                 per_attempt_history_at_its_stop_not_simultaneous=[len(r.prompt) + len(r.tokens) - 1 for r in pair])
    return proof


def fixtures(tok, a_cells, b_cells):
    filler = tok.encode(' amber birch copper denim elm fern gold hazel')
    require(filler, 'empty filler')

    def padded(tag, cells):
        # Distinct BEGINNING tags, including raw yield/cancel fixtures, prevent
        # checkpoints from skipping the very stage we are trying to exercise.
        head = tok.encode(f'{tag}\n<|im_start|>user\nRead this inert data:\n', parse_special=True)
        tail = tok.encode('\nEnd data. Count from 1 through 100000, one number per line. '
                          'Keep counting; never summarize or stop early.\n<|im_end|>\n'
                          '<|im_start|>assistant\n<think>\n\n</think>\n\n', parse_special=True)
        require(cells >= len(head) + len(tail), 'prompt too short for long-count fixture')
        count = cells - len(head) - len(tail)
        return head + (filler * (count // len(filler) + 1))[:count] + tail

    a, b = padded('INCREMENTAL-A-animals', a_cells), padded('INCREMENTAL-B-mathematics', b_cells)
    require(len(a) != len(b) and a != b, 'unequal distinct prompts required')
    return padded, a, b


class Suite(common.Suite):
    def __init__(self, engine, evidence, output, timeout, context, temperature=0.0, seed=SEED):
        super().__init__(engine, evidence, output, timeout, context)
        self.temperature, self.seed = temperature, seed

    def make_attempt(self, *args):
        return Attempt(*args, temperature=self.temperature, seed=self.seed)

    def run(self, name, requests, cancel_prefix=None, deadline=None, preamble=(), stage_context=None):
        self.engine.stage = name
        start = Path(self.engine.stderr_path).stat().st_size
        stage = {'name': name, 'passed': False, 'requests': []}
        if preamble:
            stage['preamble_commands'] = list(preamble)
        if stage_context:
            stage.update(stage_context)
        self.evidence['stages'].append(stage)
        deadline = deadline if deadline is not None else time.monotonic() + self.timeout
        protocol = common.Protocol(requests)
        stopped = False
        try:
            # Controls and admissions are a SINGLE ordered stdin burst: an
            # idle late stop is immediately followed by the replacement BGEN.
            self.engine.send(*preamble, *(r.command() for r in requests), deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                protocol.consume(event)
                if cancel_prefix is not None and not stopped and len(protocol.active) == len(requests) and \
                        all(len(r.tokens) >= cancel_prefix for r in requests):
                    self.engine.send(*(f'BSTOP {r.slot}' for r in requests), deadline=deadline)
                    stage['bstop_after_seq'] = event['seq']
                    stopped = True
            if cancel_prefix is not None:
                require(stopped, 'EOS/pressure before bounded-prefix cancellation trigger')
                require(all(r.completion['finish'] == 'cancel' for r in requests), 'missing BSTOP cancel acknowledgement')
            return stage
        finally:
            stage['requests'] = [r.record() for r in requests]
            data = Path(self.engine.stderr_path).read_bytes()
            stage['stderr_byte_range'] = [start, len(data)]
            stage['diagnostics'] = streaming.diagnostics(data[start:].decode(errors='replace'))
            self.save()

    def solo(self, name, prompt, cap):
        req = self.make_attempt(name, prompt, cap)
        stage = self.run(name, [req])
        common.normal(req)
        stage['passed'] = True
        self.save()
        return req

    def check_prefix(self, req, ref):
        require(req.tokens and req.tokens == ref.tokens[:len(req.tokens)], f'{req.name}: prefix solo parity differs')
        require(len(req.tokens) <= len(ref.tokens), 'candidate outlived solo')

    def overlap(self, a, b, prefix):
        caps = [remaining_context(self.context, p) for p in (a, b)]
        plan = growth_plan([a, b], caps, self.context)
        refs = [self.solo(f'overlap-solo-{i}', p, min(c, prefix + 64)) for i, (p, c) in enumerate(zip((a, b), caps))]
        pair = [self.make_attempt(f'uncapped-allowance-{i}', p, c, i) for i, (p, c) in enumerate(zip((a, b), caps))]
        stage = self.run('remaining-context-allowances-overlap', pair, cancel_prefix=prefix)
        stage['capacity_plan'] = plan
        stage['overlap'] = overlap_proof(pair)
        for req, ref in zip(pair, refs):
            self.check_prefix(req, ref)
        stage['passed'] = True
        self.save()

    def late_stop(self, a, b, target):
        require(self.context <= 4096, 'late-stop pressure setup requires SMALL backing <=4096')
        plan = growth_plan([a, b], [target, target], self.context)
        refs = [self.solo(f'late-stop-healthy-solo-{i}', p, 16) for i, p in enumerate((a, b))]
        require(all(len(ref.tokens) == 16 and ref.completion['finish'] == 'length' for ref in refs),
                'healthy fixture ended early; cannot establish natural batch completion')
        pair = [self.make_attempt(f'late-stop-pressure-source-{i}', p, target, i) for i, p in enumerate((a, b))]
        source = self.run('late-stop-one-actual-pressure-pair', pair)
        source['pressure'] = pressure_proof(pair, plan)
        require(all(r.completion['kind'] == 'BDONE' and r.completion['finish'] in ('pressure', 'length', 'stop') for r in pair),
                'both original batch owners must acknowledge completion before late BSTOP')
        victim = next(r for r in pair if r.completion['finish'] == 'pressure')
        source['inactive_slots_after_completions'] = [r.slot for r in pair]
        source['full_pressure_output_parity_rechecked'] = False
        source['passed'] = True
        previous = victim.completion
        self.save()
        for i, (prompt, ref, reason) in enumerate(zip((a, b), refs, ('pressure-reset-inactive', 'natural-cached-inactive'))):
            require(previous['kind'] == 'BDONE' and previous['slot'] == victim.slot and
                    previous['finish'] == ('pressure' if i == 0 else 'length'), 'wrong late-stop predecessor')
            req = self.make_attempt(f'healthy-after-late-stop-{reason}', prompt, 16, victim.slot)
            start_seq = len(self.evidence['stdout'])
            stage = self.run(f'late-BSTOP-{reason}-immediate-same-slot-BGEN', [req],
                             preamble=(f'BSTOP {victim.slot}',),
                             stage_context={'deliberate_late_BSTOP': True, 'previous_owner_completion': previous,
                                            'ownership_contract': 'NoAckOwned: no active/partial/live-admit owner; late BSTOP owes no BDONE'})
            try:
                common.parity(req, ref)
                require(req.badm and req.badm['continues'] and req.completion['kind'] == 'BDONE' and
                        req.completion['finish'] == 'length' and len(req.tokens) == 16,
                        'replacement did not reach its own natural batch terminal')
                terminals = [e for e in self.evidence['stdout'][start_seq:]
                             if e.get('kind') == 'BDONE' and e.get('slot') == victim.slot]
                require(len(terminals) == 1 and terminals[0]['seq'] == req.completion['seq'],
                        'extra old BDONE / replacement terminal not exactly once')
                require(previous['seq'] < req.token_events[0]['seq'], 'replacement preceded previous owner completion')
                stage.update(terminal_count=1, no_extra_old_BDONE=True, passed=True)
                previous = req.completion
            finally:
                self.save()
        self.evidence['late_stop_no_ack_owned_verified'] = True
        self.save()

    def fifo_pressure(self, logical, references, order, plan):
        self.engine.stage = f'actual-small-backing-FIFO-pressure-order-{order[0]}'
        start = Path(self.engine.stderr_path).stat().st_size
        stage = {'name': self.engine.stage, 'passed': False, 'requeues': []}
        self.evidence['stages'].append(stage)
        initial = [logical[i].attempt() for i in order]
        protocol = common.Protocol(initial.copy())
        owner = {id(req): i for i, item in enumerate(logical) for req in item.attempts}
        deadline = time.monotonic() + self.timeout
        pressured = set()
        try:
            self.engine.send(*(r.command() for r in initial), deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                completed = None
                if event['kind'] == 'BDONE':
                    completed = protocol.active.get(event['slot'])
                elif event['kind'] == 'BADM' and not event['continues'] and protocol.pending:
                    completed = protocol.pending[0]
                protocol.consume(event)
                if completed is None:
                    continue
                i = owner[id(completed)]
                item = logical[i]
                item.accept(completed, references[i])
                if completed.completion['finish'] == 'pressure':
                    pressured.add(i)
                    require(len(item.attempts) <= item.budget, 'unbounded/no-progress pressure replay loop')
                    resumed = item.attempt()
                    # Append AFTER already submitted admissions. BT/BDONE from
                    # other rows can arrive while its T/DONE/BADM is pending.
                    protocol.requests.append(resumed)
                    protocol.pending.append(resumed)
                    owner[id(resumed)] = i
                    self.engine.send(resumed.command(), deadline=deadline)
                    stage['requeues'].append({'owner': i, 'pressure_seq': event['seq'],
                                              'history_cells': len(resumed.prompt), 'remaining': resumed.cap,
                                              'same_seed': self.seed, 'same_temperature': self.temperature,
                                              'pending_admissions': len(protocol.pending)})
                elif completed.completion['finish'] == 'cancel':
                    raise AssertionError('unexpected cancellation in pressure completion gate')
            stage['pressure'] = pressure_proof(initial, plan)
            require(pressured == {0, 1}, 'both owners must pressure-stop and FIFO-resume in this stage; increase finite --pressure-target')
            require(all(item.finish and item.tokens == references[i].tokens for i, item in enumerate(logical)),
                    'FIFO logical request final token parity/completion differs')
            fallback = []
            for item in logical:
                for before, after in zip(item.attempts, item.attempts[1:]):
                    require(before.completion['finish'] == 'pressure' and
                            before.completion['seq'] < after.token_events[0]['seq'],
                            'replay emitted before pressure release acknowledgement')
                    if self.evidence['expect_replay']:
                        reused = after.admission_done['reused']
                        require(reused is not None and reused < len(after.prompt), 'no positive replay work after pressure')
                        fallback.append({'name': after.name, 'reused_cells': reused,
                                         'replay_cells': len(after.prompt) - reused})
            stage['fallback_replays'] = fallback
            stage['passed'] = True
        finally:
            stage['logical_requests'] = [item.record() for item in logical]
            stage['requests'] = [req.record() for req in protocol.requests]
            raw = Path(self.engine.stderr_path).read_bytes()
            stage['stderr_byte_range'] = [start, len(raw)]
            text = raw[start:].decode(errors='replace')
            stage['diagnostics'] = streaming.diagnostics(text)
            stage['pressure_diagnostics'] = pressure_diagnostics(text)
            if self.evidence['expect_replay'] and any(d['kind'] == 'restore' for d in stage['diagnostics']):
                stage['passed'] = False
                self.save()
                raise AssertionError('disabled/tiny parking restored instead of exercising replay')
            self.save()

    def pressure(self, a, b, target, abandon=False):
        require(self.context <= 4096, 'actual pressure gate requires SMALL backing, not a 262K output marathon')
        plan = growth_plan([a, b], [target, target], self.context)
        refs = [self.solo(f'pressure-solo-{i}', p, target) for i, p in enumerate((a, b))]
        require(all(len(r.tokens) == target for r in refs),
                'long-count fixture ended naturally before finite target; change fixture, do not credit EOS')
        for order in ((0, 1), (1, 0)):
            logical = [LogicalRequest(f'pressure-{i}-order-{order[0]}', p, target, order.index(i),
                                      seed=self.seed, temperature=self.temperature)
                       for i, p in enumerate((a, b))]
            self.fifo_pressure(logical, refs, order, plan)
        self.evidence['pressure_requests_exercised'] = [0, 1]
        if self.evidence.get('require_pressure_restore'):
            self.positive_pressure_restore(a, b, target, refs, plan)
        if abandon:
            self.abandon_after_pressure(a, b, target, refs, plan)

    def positive_pressure_restore(self, a, b, target, refs, plan):
        # FIFO admissions may legitimately skip optional fresh-page restores
        # while another active row occupies backing. Do NOT demand every FIFO
        # resume restore. Establish the positive path in a separate controlled
        # pressure burst, then restore after capacity has actually been freed.
        owners = [LogicalRequest(f'canonical-pressure-{i}', p, target, i,
                                 seed=self.seed, temperature=self.temperature)
                  for i, p in enumerate((a, b))]
        pair = [item.attempt() for item in owners]
        parked_stage = self.run('canonical-target-only-pressure-source', pair)
        parked_stage['pressure'] = pressure_proof(pair, plan)
        for item, req, ref in zip(owners, pair, refs):
            item.accept(req, ref)
        victim = next(item for item in owners if item.finish is None)
        index = owners.index(victim)
        start, end = parked_stage['stderr_byte_range']
        raw = Path(self.engine.stderr_path).read_bytes()[start:end].decode(errors='replace')
        records = pressure_diagnostics(raw)
        parked_stage['pressure_diagnostics'] = records
        parked_stage['passed'] = True
        self.save()
        # A normal completed competitor may still own idle cached KV. A cold
        # 128-cell GEN forces its reclamation before the optional restore (which
        # intentionally doesn't evict other owners just to materialize a cache).
        cold = next((t for t in a + b if t not in (a[0], b[0])), None)
        require(cold is not None, 'no distinct sentinel token in count fixtures')
        sentinel = self.make_attempt('canonical-free-idle-backing-sentinel', [cold] * 128, 1)
        stage = self.run('canonical-free-idle-backing', [sentinel])
        require(sentinel.completion['finish'] in ('length', 'stop'), 'sentinel did not complete normally')
        stage['passed'] = True  # EOS is fine for this capacity-reclamation-only GEN.
        resumed = victim.attempt()
        resumed.slot = None  # MAIN GEN, so private draft coherence is exercised.
        restored_stage = self.run('canonical-target-only-pressure-restore-main-GEN', [resumed])
        try:
            victim.accept(resumed, refs[index])
            require(victim.finish is not None, 'canonical single-row resume did not finish')
            begin, end = restored_stage['stderr_byte_range']
            restored_text = Path(self.engine.stderr_path).read_bytes()[begin:end].decode(errors='replace')
            restored_stage['coherence_diagnostics'] = pressure_diagnostics(restored_text)
            restored_stage['target_only_pressure_restore'] = verify_pressure_restore(
                [victim], records, restored_stage['diagnostics'], restored_stage['coherence_diagnostics'])
            enabled_mtp = self.evidence.get('coherence_mtp_max', 1) > 1
            restored_stage['mtp_enabled_target_only_suppression_exercised'] = enabled_mtp
            for proof in restored_stage['target_only_pressure_restore']:
                proof['mtp_draft_coherence_exercised'] = enabled_mtp
            fields = resumed.admission_done['raw'].split()
            require(len(fields) > 7 and int(fields[7]) == 0, 'restored target-only MAIN GEN offered private drafts')
            restored_stage['restored_main_drafts_offered'] = int(fields[7])
            if enabled_mtp:
                offered = [int(ref.admission_done['raw'].split()[7]) for ref in refs]
                require(any(n > 0 for n in offered), 'enabled MTP audit never offered drafts in solo references')
                restored_stage['original_solo_drafts_offered'] = offered
            self.evidence['mtp_draft_coherence_exercised'] = enabled_mtp
            restored_stage['logical_request'] = victim.record()
            restored_stage['pressure_source_stage'] = parked_stage['name']
            restored_stage['passed'] = True
        finally:
            self.save()

    def abandon_after_pressure(self, a, b, target, refs, plan):
        # Native pressure already released backing. The external parked/waiting
        # owner is cancelled by NOT resubmitting, not by expecting another BDONE
        # on an inactive slot. A fresh BGEN in that slot proves no inherited BT.
        pair = [self.make_attempt(f'cancel-owner-{i}', p, target, i) for i, p in enumerate((a, b))]
        stage = self.run('cancel-owner-parked-after-pressure', pair)
        stage['pressure'] = pressure_proof(pair, plan)
        victim = next(r for r in pair if r.completion['finish'] == 'pressure')
        for req, ref in zip(pair, refs):
            self.check_prefix(req, ref)
        stage['cancelled_owner_slot'] = victim.slot
        stage['cancellation'] = 'discard replay continuation after pressure release; no second native acknowledgement required'
        stage['passed'] = True
        self.save()
        # Use unrelated IDs and a reference, not merely a nonempty answer.
        healthy = self.make_attempt('healthy-after-parked-cancel', b if victim.slot == 0 else a, 16, victim.slot)
        reference = self.solo('healthy-parked-reference', healthy.prompt, 16)
        next_stage = self.run('healthy-same-slot-no-inherited-replies', [healthy])
        common.parity(healthy, reference)
        require(victim.completion['seq'] < healthy.token_events[0]['seq'], 'healthy reply preceded pressure release')
        next_stage['passed'] = True
        self.save()

    def rolling(self, padded, tok):
        # >256 outputs is essential: a cap-32 branch fits entirely in initial
        # headroom and tests page addressing, NOT renewed backing growth.
        cap = AHEAD + 32
        for offset in (1, 2, 3):
            size = 160 + (offset - 7 - 160) % PAGE
            seed = padded(f'ROLLING-COW-{offset}', size)
            seed_ref = self.solo(f'rolling-seed-solo-{offset}', seed, 8)
            require(len(seed_ref.tokens) == 8, 'seed EOS prevents partial-page fixture')
            shared = seed + seed_ref.tokens[:-1]
            require(len(shared) % PAGE == offset, 'wrong COW tail offset')
            branches = [shared + tok.encode(text) for text in ('\nEven continuation:\n', '\nOdd continuation:\n')]
            refs = [self.solo(f'rolling-branch-solo-{offset}-{i}', p, cap) for i, p in enumerate(branches)]
            seeded = self.make_attempt(f'rolling-seed-slot-{offset}', seed, 8, 0)
            stage = self.run(f'rolling-seed-{offset}', [seeded])
            common.parity(seeded, seed_ref)
            stage['passed'] = True
            for i, (prompt, ref) in enumerate(zip(branches, refs)):
                req = self.make_attempt(f'rolling-branch-{offset}-{i}', prompt, cap, 1 - i)
                stage = self.run(f'rolling-page-COW-offset-{offset}-{i}', [req])
                stage['cached_prefix'] = streaming.verify_cached_branch(stage, req, ref, len(shared), 0)
                stage['renewed_backing_growth'] = rolling_growth_evidence(req)
                stage['consumed_output_page_crossings_lower_bound'] = (len(req.tokens) - 1) // PAGE
                stage['passed'] = True
                self.save()

    def parking(self, padded, tok):
        # Must run before BGEN so canonical restore cannot be an idle slot copy.
        a, b = padded('CANONICAL-A', 192), padded('CANONICAL-B', 288)
        seed = self.solo('canonical-seed-reference', a, 1)
        continuation = a + seed.tokens + tok.encode('\nContinue counting:\n')
        ref = self.solo('canonical-continuation-reference', continuation, 32)
        reset = self.solo('canonical-A-reset', a, 1)
        common.parity(reset, seed)
        self.solo('canonical-B-switch', b, 32)
        park = self.evidence['stages'][-1]
        restored = self.solo('canonical-A-return', continuation, 32)
        stage = self.evidence['stages'][-1]
        stage['passed'] = False
        try:
            if self.evidence['expect_replay']:
                common.parity(restored, ref)
                require(not any(d['kind'] == 'restore' for d in stage['diagnostics']), 'expected replay, got restore')
                require(restored.admission_done['reused'] == 0, 'A/B/A fallback did not replay cold')
                stage['fallback_replay_verified'] = True
            else:
                stage['parking'] = streaming.verify_park_restore(park, stage, restored, ref, len(a), self.evidence['cache_mib'])
            stage['passed'] = True
        finally:
            self.save()

    def cancel_waiting(self, padded):
        self.engine.stage = 'cancel-protected-paused-owner-during-capacity-wait'
        start = Path(self.engine.stderr_path).stat().st_size
        stage = {'name': self.engine.stage, 'passed': False}
        self.evidence['stages'].append(stage)
        deadline = time.monotonic() + self.timeout
        source = self.make_attempt('distinct-yield-source', padded('YIELD-FRESH-BEGINNING', self.context // 2 + 64), 16)
        protocol = common.Protocol([source])
        yielded = release = waiter = None
        try:
            self.engine.send(source.command(), 'BYIELD 0', deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                if event['kind'] == 'YIELDED':
                    require(yielded is None and event['slot'] == 0 and 0 < event['tokens'] < len(source.prompt), 'invalid YIELDED')
                    yielded = event
                else:
                    protocol.consume(event)
            require(yielded and not source.tokens and source.completion['finish'] == 'cancel', 'yield stage not exercised')
            require(source.completion['reused'] == 0, 'yield source reused a cache/checkpoint; not a fresh fixture')
            count = self.context - (yielded['tokens'] // PAGE * PAGE) + PAGE
            prompt = padded('WAITER-DISTINCT-BEGINNING', count)
            require(len(prompt) + 16 + GUARD <= self.context, 'yield too small to construct fitting waiting admission')
            require(pages(yielded['tokens']) + pages(len(prompt)) > self.context // PAGE, 'waiter actually fits alongside paused prefix')
            waiter = self.make_attempt('capacity-waiter', prompt, 16, 1)
            protocol = common.Protocol([waiter])
            self.engine.send(waiter.command(), 'BSTOP 0', deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                if event['kind'] == 'BDONE' and event['slot'] == 0:
                    require(release is None and event['count'] == 0 and event['finish'] == 'cancel', 'invalid paused release')
                    require(not waiter.tokens and waiter.admission_done is None, 'waiter emitted before capacity release')
                    release = event
                else:
                    if event['kind'] != 'OTHER':
                        require(release is not None, 'waiter admitted before protected prefix BSTOP acknowledgement')
                    protocol.consume(event)
            require(release is not None, 'missing paused release')
            common.normal(waiter)
            # Only now can an independent solo run avoid disturbing ownership.
            ref = self.solo('capacity-waiter-solo-reference', prompt, 16)
            common.parity(waiter, ref)
            stage.update(yielded=yielded, release=release, held_prefix_cells=yielded['tokens'],
                         waiter_prompt_cells=len(prompt), passed=True)
            healthy = self.make_attempt('healthy-after-wait-cancel', padded('HEALTHY-DISTINCT-BEGINNING', 160), 16, 0)
            ref = self.solo('healthy-wait-solo-reference', healthy.prompt, 16)
            healthy_stage = self.run('healthy-after-wait-cancel-no-inherited-replies', [healthy])
            common.parity(healthy, ref)
            healthy_stage['passed'] = True
        finally:
            stage['source'] = source.record()
            stage['waiter'] = waiter.record() if waiter else None
            stage['yielded'] = yielded
            stage['release'] = release
            raw = Path(self.engine.stderr_path).read_bytes()
            stage['stderr_byte_range'] = [start, len(raw)]
            stage['diagnostics'] = streaming.diagnostics(raw[start:].decode(errors='replace'))
            self.save()


def cleanup_check(evidence):
    cleanup = evidence['cleanup']
    require(cleanup.get('returncode') == 0 and not cleanup.get('quit_timeout'), 'native QUIT did not exit cleanly')
    require(cleanup.get('reader_stopped') and cleanup.get('writer_stopped'), 'native pipe thread did not stop')
    require(not cleanup.get('protocol_errors'), 'malformed shutdown protocol')
    require(not any(e.get('kind') == 'ERR' for e in evidence['stdout']), 'unexpected native ERR')
    require(all(e.get('kind', 'OTHER') == 'OTHER' for e in evidence['stdout'][cleanup['stdout_start_seq']:]),
            'inherited/trailing protocol after final completion')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exe', required=True, type=Path)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path, help='NEW JSON path; stderr appended as .stderr.log')
    ap.add_argument('--gate', choices=('all', 'overlap', 'pressure', 'rolling', 'parking', 'cancel', 'late-stop'), default='all')
    ap.add_argument('--context', type=int, default=1024)
    ap.add_argument('--resident', type=int, default=0)
    ap.add_argument('--cache-mib', type=int, default=0)
    ap.add_argument('--expect-replay', action='store_true', help='require optional parking miss and replay, including tiny cache')
    ap.add_argument('--prompt-a-cells', type=int, default=160)
    ap.add_argument('--prompt-b-cells', type=int, default=164)
    ap.add_argument('--pressure-target', type=int, help='finite budget; default context - larger prompt - 8; must exercise BOTH pressure owners')
    ap.add_argument('--prefix', type=int, default=16)
    ap.add_argument('--prefill', type=int, default=64,
                    help='fixed native chunk: 64..4096 in multiples of 64; default preserves small/yield fixtures')
    ap.add_argument('--temperature', type=float, default=0.0, help='0 greedy; e.g. 0.3 sampled pressure/replay parity')
    ap.add_argument('--seed', type=int, default=SEED, help='explicit POSITIVE seed retained on every attempt')
    ap.add_argument('--require-pressure-restore', action='store_true',
                    help='independent target-only pressure park + MAIN GEN restore and T=1 suppression diagnostics')
    ap.add_argument('--coherence-mtp-max', type=int, choices=(1, 2, 4), default=1,
                    help='explicit greedy MTP-enabled audit (2/4), only with --gate pressure --require-pressure-restore; default frozen 1')
    ap.add_argument('--gpu')
    ap.add_argument('--vram-reserve-mib', type=int, default=2048)
    ap.add_argument('--startup-timeout', type=float, default=900)
    ap.add_argument('--stage-timeout', type=float, default=900)
    ap.add_argument('--cleanup-timeout', type=float, default=15)
    a = ap.parse_args(argv)
    if not 512 <= a.context < 2**31 - GUARD or a.context % PAGE:
        ap.error('context must be >=512, <2^31-8 and four-aligned')
    if (a.resident != 0 and not streaming.MIN_RESIDENT <= a.resident < a.context) or a.resident % PAGE:
        ap.error('resident must be zero or four-aligned 20480 <= resident < context')
    if not 64 <= a.prefill <= 4096 or a.prefill % 64:
        ap.error('prefill must be 64..4096 in multiples of 64')
    if not math.isfinite(a.temperature) or a.temperature < 0 or not 0 < a.seed < 2**63:
        ap.error('temperature must be finite/nonnegative and seed must be positive <2^63')
    if a.require_pressure_restore and (a.cache_mib <= 0 or a.expect_replay or a.gate not in ('all', 'pressure', 'cancel')):
        ap.error('--require-pressure-restore needs enabled non-replay parking and a SMALL pressure gate')
    if a.coherence_mtp_max > 1 and (not a.require_pressure_restore or a.gate != 'pressure' or a.temperature != 0):
        ap.error('MTP-enabled coherence audit requires --gate pressure --require-pressure-restore --temperature 0')
    if a.cache_mib < 0 or a.vram_reserve_mib < 0 or min(a.prompt_a_cells, a.prompt_b_cells) < 96 or a.prefix < 8:
        ap.error('invalid cache/reserve/prompt/prefix sizes')
    if a.gate in ('all', 'pressure', 'cancel', 'late-stop') and a.context > 4096:
        ap.error('pressure/cancel/late-stop/all require SMALL backing <=4096; use --gate overlap/rolling/parking at production capacity')
    if a.gpu is not None and (not a.gpu.strip() or ',' in a.gpu):
        ap.error('select exactly one GPU')
    if any(not math.isfinite(t) or t <= 0 for t in (a.startup_timeout, a.stage_timeout, a.cleanup_timeout)):
        ap.error('timeouts must be positive and finite')
    target = a.pressure_target if a.pressure_target is not None else a.context - max(a.prompt_a_cells, a.prompt_b_cells) - GUARD
    output = a.output.expanduser().resolve()
    stderr = Path(str(output) + '.stderr.log')
    if output.exists() or stderr.exists():
        ap.error('existing artifact; select a fresh --output')
    with output.open('x', encoding='utf-8') as f:
        f.write('{}\n')
    evidence = {'schema': 1, 'harness': 'incremental-unified-kv', 'passed': False,
                'config': str(a.config.resolve()), 'context': a.context, 'resident': a.resident,
                'cache_mib': a.cache_mib, 'expect_replay': a.expect_replay or a.cache_mib == 0,
                'gate': a.gate, 'seed': a.seed, 'temperature': a.temperature, 'prefill': a.prefill,
                'sampling': {'temperature': a.temperature, 'seed': a.seed, 'rng_offset_added': False},
                'require_pressure_restore': a.require_pressure_restore,
                'coherence_mtp_max': a.coherence_mtp_max,
                'mtp_draft_coherence_exercised': False, 'pressure_target': target,
                'stderr_path': str(stderr), 'stdout': [], 'commands': [], 'stages': [],
                'timeouts': {'startup_s': a.startup_timeout, 'stage_s': a.stage_timeout, 'cleanup_s': a.cleanup_timeout}}
    engine = Engine(evidence, stderr, a.cleanup_timeout)
    suite = Suite(engine, evidence, output, a.stage_timeout, a.context, a.temperature, a.seed)
    success = False
    try:
        cfg = json.loads(a.config.read_text(encoding='utf-8-sig'))
        command, cwd, env, kv = settings(cfg, a.exe, a.context, a.resident, a.cache_mib, a.gpu,
                                        a.vram_reserve_mib, a.coherence_mtp_max, prefill=a.prefill)
        evidence.update(command=command, cwd=cwd, kv=kv,
                        device_env={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'STRATA_IQ_MT_MIN') if k in env})
        suite.save()
        tok = common.tokenizer(common.resolve_path(cfg['tokenizer'], cwd))
        padded, pa, pb = fixtures(tok, a.prompt_a_cells, a.prompt_b_cells)
        if a.gate in ('all', 'pressure', 'cancel', 'late-stop'):
            evidence['pressure_plan'] = growth_plan([pa, pb], [target, target], a.context)
        if a.gate in ('all', 'overlap'):
            growth_plan([pa, pb], [remaining_context(a.context, p) for p in (pa, pb)], a.context)
        engine.start(command, cwd, env, a.context, a.resident, a.cache_mib, a.startup_timeout, a.coherence_mtp_max)
        if a.gate in ('all', 'parking'):
            suite.parking(padded, tok)  # before ANY BGEN
        if a.gate in ('all', 'overlap'):
            suite.overlap(pa, pb, a.prefix)
        if a.gate in ('all', 'rolling'):
            suite.rolling(padded, tok)
        if a.gate in ('all', 'pressure', 'cancel'):
            suite.pressure(pa, pb, target, abandon=a.gate in ('all', 'cancel'))
        if a.gate in ('all', 'cancel'):
            suite.cancel_waiting(padded)
        if a.gate == 'late-stop':
            suite.late_stop(pa, pb, target)
        require(evidence['stages'] and all(s['passed'] for s in evidence['stages']), 'incomplete stages')
        success = True
    except (Exception, KeyboardInterrupt) as exc:
        evidence['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'stage': engine.stage,
                               'traceback': traceback.format_exc()}
    finally:
        try:
            engine.close(success)
            if success:
                cleanup_check(evidence)
        except (Exception, KeyboardInterrupt) as exc:
            success = False
            evidence['cleanup_failure'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        if stderr.exists():
            text = stderr.read_text(errors='replace')
            evidence['native_diagnostics'] = streaming.diagnostics(text)
            evidence['pressure_diagnostics'] = pressure_diagnostics(text)
        evidence['passed'] = success
        suite.save()
    print(f'{"PASS" if success else "FAIL"}: {output}; stderr: {stderr}', flush=True)
    if not success:
        print(json.dumps(evidence.get('failure') or evidence.get('cleanup_failure')), file=sys.stderr)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
