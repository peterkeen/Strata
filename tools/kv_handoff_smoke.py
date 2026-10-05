#!/usr/bin/env python3
"""Independent actual-model unified-KV handoff protocol gate.

Private native subprocess only; never points at a production HTTP service and never
builds, deploys, SSHes or commits.  Example operator runs:

  python3 tools/kv_handoff_smoke.py --exe /private/strata --config model.json \
      --output /tmp/kv-handoff-small.json --context 1024 --cache-mib 0

  python3 tools/kv_handoff_smoke.py --exe /private/strata --config model.json \
      --output /tmp/kv-handoff-prod.json --context 262144 --resident 32768 \
      --cache-mib 0 --prompt-a-cells 35000 --prompt-b-cells 35004 \
      --prefill 1024 --prefix 128 --gpu 0

Protocol audited here:
  * INFO must advertise kv_handoff=1 and disabled conversation parking
    (conversation_cache_mib=0).  Positive reuse must therefore come from live or
    idle KV/checkpoint handoff state, not canonical parked snapshots.
  * HANDOFF ends a main GEN with DONE ... handoff and retains a trimmed valid KV
    prefix.  The following BGEN of original+all-output must reuse at least
    prompt+outputs-1 cells because the last native output token was not fed.
  * BHANDOFF <slot> ends an active batch decoder with BDONE ... handoff and
    retains idle cached target-only KV/checkpoints.  Late BHANDOFF after a
    natural terminal owes no BDONE.
  * STOP/BSTOP remain real cancellation/release controls and recover to healthy
    requests.

The gate freezes greedy solo references, runs both BGEN admission orders over
repeated handoff cycles, proves overlap with BT from both rows before either
BDONE, verifies no full replay on every warm continuation, checks exact terminal
ack counts, and records raw stdout, commands, JSON evidence and stderr ranges.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incremental_kv_smoke as inc

common = inc.common
streaming = inc.streaming
require = common.require
PAGE = common.PAGE_CELLS
GUARD = inc.GUARD
SEED = inc.SEED
Attempt = inc.Attempt


@dataclass
class LogicalResponse:
    name: str
    original: list[int]
    reference: Attempt
    tokens: list[int] = field(default_factory=list)

    def prompt(self) -> list[int]:
        return self.original + self.tokens

    def remaining(self) -> int:
        return len(self.reference.tokens) - len(self.tokens)

    def accept(self, req: Attempt, finish: tuple[str, ...] = ('length', 'stop', 'handoff')):
        require(req.completion is not None, f'{req.name}: missing completion')
        require(req.completion['finish'] in finish,
                f'{req.name}: expected finish {finish}, got {req.completion["finish"]!r}')
        before = len(self.tokens)
        require(req.prompt == self.original + self.tokens,
                f'{req.name}: prompt is not the exact current history of {self.name}')
        expected = self.reference.tokens[before:before + len(req.tokens)]
        require(req.tokens == expected,
                f'{req.name}: logical response differs from solo at offset {before}')
        self.tokens.extend(req.tokens)
        require(len(self.tokens) <= len(self.reference.tokens), f'{self.name}: outlived frozen solo reference')
        return {'logical': self.name, 'accepted_tokens': len(req.tokens), 'offset_before': before,
                'finish': req.completion['finish']}


def expected_valid_prefix(prompt_len: int, emitted: int) -> int:
    """Native has not fed the final emitted output token back into KV."""
    require(emitted > 0, 'handoff/reuse proof requires at least one emitted token')
    return prompt_len + emitted - 1


def require_reuse(req: Attempt, cells: int, label: str):
    reused = req.admission_done and req.admission_done.get('reused')
    require(reused is not None, f'{label}: native DONE did not report reused cells')
    require(reused >= cells, f'{label}: full replay or short reuse, got {reused}, need >= {cells}')
    return {'label': label, 'required_reused_cells': cells, 'actual_reused_cells': reused,
            'no_full_replay': True}


def verify_info(info, context, resident, cache_mib, mtp_max=1):
    inc.verify_info(info, context, resident, cache_mib, mtp_max)
    require(info.get('kv_handoff') == '1', f'INFO requires kv_handoff=1, got {info.get("kv_handoff")!r}')
    require(cache_mib == 0 and info.get('conversation_cache_mib') == '0',
            'handoff smoke requires parking disabled (conversation_cache_mib=0)')


class Engine(inc.Engine):
    def start(self, command, cwd, env, context, resident, cache_mib, timeout, mtp_max=1):
        # Same bounded subprocess/reader as incremental_kv_smoke, with the extra
        # handoff INFO bit.  Keep this independent so old harnesses remain frozen.
        self.log = open(self.stderr_path, 'xb')
        import subprocess, threading
        self.p = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log,
                                  start_new_session=(os.name == 'posix'))
        self.evidence['pid'] = self.p.pid
        self.reader = threading.Thread(target=self._read_stdout, name='kv-handoff-stdout', daemon=True)
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


def settings(cfg, exe, context, resident, cache_mib, gpu=None, vram_reserve_mib=2048, prefill=64):
    require(cache_mib == 0, '--cache-mib must be 0 for the strong in-place handoff gate')
    return inc.settings(cfg, exe, context, resident, cache_mib, gpu,
                        vram_reserve_mib=vram_reserve_mib, coherence_mtp_max=1, prefill=prefill)


class Frontend:
    """Small protocol front-end: controls are explicit, separate from Suite policy."""
    def __init__(self, engine, deadline):
        self.engine = engine
        self.deadline = deadline

    def send(self, *lines):
        self.engine.send(*lines, deadline=self.deadline)

    def event(self):
        return self.engine.next_event(self.deadline)


class Suite(common.Suite):
    def __init__(self, engine, evidence, output, timeout, context, prefix, seed=SEED):
        super().__init__(engine, evidence, output, timeout, context)
        self.prefix = prefix
        self.seed = seed

    def make_attempt(self, name, prompt, cap, slot=None):
        return Attempt(name, prompt, cap, slot, temperature=0.0, seed=self.seed)

    def _stage(self, name, extra=None):
        self.engine.stage = name
        start = Path(self.engine.stderr_path).stat().st_size
        stage = {'name': name, 'passed': False, 'requests': []}
        if extra:
            stage.update(extra)
        self.evidence['stages'].append(stage)
        return stage, start

    def _finish_stage(self, stage, start, requests=()):
        stage['requests'] = [r.record() for r in requests]
        raw = Path(self.engine.stderr_path).read_bytes()
        stage['stderr_byte_range'] = [start, len(raw)]
        text = raw[start:].decode(errors='replace')
        stage['diagnostics'] = streaming.diagnostics(text)
        self.save()

    def solo(self, name, prompt, cap):
        req = self.make_attempt(name, prompt, cap)
        stage, start = self._stage(name)
        protocol = common.Protocol([req])
        deadline = time.monotonic() + self.timeout
        try:
            self.engine.send(req.command(), deadline=deadline)
            while not protocol.finished:
                protocol.consume(self.engine.next_event(deadline))
            common.normal(req)
            stage['passed'] = True
            return req
        finally:
            self._finish_stage(stage, start, [req])

    def main_handoff(self, name, prompt, cap, min_emit):
        require(cap > min_emit, 'handoff cap must leave room before natural length')
        req = self.make_attempt(name, prompt, cap)
        stage, start = self._stage(name, {'control': 'HANDOFF', 'min_emit': min_emit})
        protocol = common.Protocol([req])
        deadline = time.monotonic() + self.timeout
        fe = Frontend(self.engine, deadline)
        sent = False
        try:
            fe.send(req.command())
            while not protocol.finished:
                event = fe.event()
                protocol.consume(event)
                if not sent and req.completion is None and len(req.tokens) >= min_emit:
                    fe.send('HANDOFF')
                    stage['handoff_after_seq'] = event['seq']
                    stage['handoff_after_tokens'] = len(req.tokens)
                    sent = True
            require(sent, 'main decode reached terminal before HANDOFF trigger')
            require(req.completion['kind'] == 'DONE' and req.completion['finish'] == 'handoff',
                    'HANDOFF did not end main decode with DONE handoff')
            require(req.completion['count'] == len(req.tokens) and len(req.tokens) >= min_emit,
                    'HANDOFF DONE count/tokens mismatch')
            stage['terminal_ack_count'] = 1
            stage['passed'] = True
            return req
        finally:
            self._finish_stage(stage, start, [req])

    def batch_handoff_pair(self, name, requests, min_emit, required_reuse=None):
        require(len(requests) == 2 and all(r.slot is not None for r in requests), 'two BGEN requests required')
        required_reuse = required_reuse or {}
        stage, start = self._stage(name, {'control': 'BHANDOFF', 'min_emit': min_emit,
                                          'required_entry_reuse': dict(required_reuse)})
        protocol = common.Protocol(requests)
        deadline = time.monotonic() + self.timeout
        fe = Frontend(self.engine, deadline)
        sent = False
        try:
            fe.send(*(r.command() for r in requests))
            while not protocol.finished:
                event = fe.event()
                protocol.consume(event)
                if not sent and len(protocol.active) == 2 and all(len(r.tokens) >= min_emit for r in requests):
                    fe.send(*(f'BHANDOFF {r.slot}' for r in requests))
                    stage['bhandoff_after_seq'] = event['seq']
                    stage['bhandoff_after_tokens'] = {r.name: len(r.tokens) for r in requests}
                    sent = True
            require(sent, 'batch decoders reached terminal before BHANDOFF trigger')
            entry_reuse = []
            for req in requests:
                require(req.completion['kind'] == 'BDONE' and req.completion['finish'] == 'handoff',
                        f'{req.name}: BHANDOFF did not end with BDONE handoff')
                require(req.completion['count'] == len(req.tokens), f'{req.name}: BDONE handoff count mismatch')
                need = required_reuse.get(req.name, 0)
                if need > 0:
                    entry_reuse.append(require_reuse(req, need, req.name + ' entry'))
                else:
                    entry_reuse.append({'label': req.name + ' entry', 'required_reused_cells': 0,
                                        'actual_reused_cells': req.admission_done.get('reused') if req.admission_done else None,
                                        'cold_allowed': True})
            stage['entry_reuse'] = entry_reuse
            stage['overlap'] = inc.overlap_proof(requests)
            stage['terminal_ack_count'] = 2
            stage['passed'] = True
            return stage
        finally:
            self._finish_stage(stage, start, requests)

    def warm_bgen(self, name, logical, cap, slot, required_reuse, reference=None):
        req = self.make_attempt(name, logical.prompt() if isinstance(logical, LogicalResponse) else logical, cap, slot)
        stage, start = self._stage(name, {'required_reuse_cells': required_reuse})
        protocol = common.Protocol([req])
        deadline = time.monotonic() + self.timeout
        try:
            self.engine.send(req.command(), deadline=deadline)
            while not protocol.finished:
                protocol.consume(self.engine.next_event(deadline))
            common.normal(req)
            stage['reuse'] = require_reuse(req, required_reuse, name)
            if isinstance(logical, LogicalResponse):
                stage['accepted'] = logical.accept(req, ('length', 'stop'))
            elif reference is not None:
                common.parity(req, reference)
            stage['terminal_ack_count'] = 1
            stage['passed'] = True
            return req
        finally:
            self._finish_stage(stage, start, [req])

    def warm_main(self, name, logical: LogicalResponse, cap, required_reuse):
        req = self.make_attempt(name, logical.prompt(), cap)
        stage, start = self._stage(name, {'required_reuse_cells': required_reuse})
        protocol = common.Protocol([req])
        deadline = time.monotonic() + self.timeout
        try:
            self.engine.send(req.command(), deadline=deadline)
            while not protocol.finished:
                protocol.consume(self.engine.next_event(deadline))
            common.normal(req)
            stage['reuse'] = require_reuse(req, required_reuse, name)
            stage['accepted'] = logical.accept(req, ('length', 'stop'))
            stage['terminal_ack_count'] = 1
            stage['passed'] = True
            return req
        finally:
            self._finish_stage(stage, start, [req])

    def late_bhandoff_no_terminal(self, prompt, reference):
        done = self.make_attempt('late-bhandoff-natural-source', prompt, 4, 0)
        source = self.run_batch_natural('late-bhandoff-natural-source', [done])
        require(done.completion['finish'] in ('length', 'stop'), 'source did not naturally complete')
        req = self.make_attempt('healthy-after-late-BHANDOFF', prompt, min(4, len(reference.tokens)), 0)
        stage, start = self._stage('late-BHANDOFF-after-natural-terminal',
                                   {'previous_terminal_seq': done.completion['seq'],
                                    'ownership_contract': 'late BHANDOFF has no active decoder; no BDONE owed'})
        protocol = common.Protocol([req])
        deadline = time.monotonic() + self.timeout
        start_seq = len(self.evidence['stdout'])
        try:
            self.engine.send('BHANDOFF 0', req.command(), deadline=deadline)
            while not protocol.finished:
                protocol.consume(self.engine.next_event(deadline))
            require(req.tokens == reference.tokens[:len(req.tokens)], 'healthy request after late BHANDOFF differs')
            terminals = [e for e in self.evidence['stdout'][start_seq:]
                         if e.get('kind') == 'BDONE' and e.get('slot') == 0]
            require(len(terminals) == 1 and terminals[0]['seq'] == req.completion['seq'],
                    'late BHANDOFF produced an extra terminal or hid the replacement terminal')
            stage['terminal_count'] = 1
            stage['no_extra_old_BDONE'] = True
            stage['passed'] = True
        finally:
            self._finish_stage(stage, start, [req])
        source['late_bhandoff_followup_stage'] = stage['name']

    def run_batch_natural(self, name, requests):
        stage, start = self._stage(name)
        protocol = common.Protocol(requests)
        deadline = time.monotonic() + self.timeout
        try:
            self.engine.send(*(r.command() for r in requests), deadline=deadline)
            while not protocol.finished:
                protocol.consume(self.engine.next_event(deadline))
            for req in requests:
                common.normal(req)
            stage['passed'] = True
            return stage
        finally:
            self._finish_stage(stage, start, requests)

    def cancellation_recovery(self, a_prompt, b_prompt, a_ref, b_ref):
        # Main STOP remains cancellation, not handoff.
        main = self.make_attempt('real-STOP-main-source', a_prompt, max(self.prefix * 4, self.prefix + 8))
        stage, start = self._stage('real-STOP-main-cancel')
        protocol = common.Protocol([main])
        deadline = time.monotonic() + self.timeout
        stopped = False
        try:
            self.engine.send(main.command(), deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                protocol.consume(event)
                if not stopped and main.completion is None and len(main.tokens) >= self.prefix:
                    self.engine.send('STOP', deadline=deadline)
                    stopped = True
                    stage['stop_after_seq'] = event['seq']
            require(stopped and main.completion['finish'] == 'cancel', 'STOP did not cancel main decode')
            stage['passed'] = True
        finally:
            self._finish_stage(stage, start, [main])
        healthy = self.solo('healthy-main-after-STOP', a_prompt, min(8, len(a_ref.tokens)))
        require(healthy.tokens == a_ref.tokens[:len(healthy.tokens)], 'healthy main after STOP differs')

        # Batch BSTOP remains cancellation/release.
        bat = self.make_attempt('real-BSTOP-batch-source', b_prompt, max(self.prefix * 4, self.prefix + 8), 1)
        stage, start = self._stage('real-BSTOP-batch-cancel')
        protocol = common.Protocol([bat])
        deadline = time.monotonic() + self.timeout
        stopped = False
        try:
            self.engine.send(bat.command(), deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                protocol.consume(event)
                if not stopped and bat.badm and bat.completion is None and len(bat.tokens) >= self.prefix:
                    self.engine.send('BSTOP 1', deadline=deadline)
                    stopped = True
                    stage['bstop_after_seq'] = event['seq']
            require(stopped and bat.completion['finish'] == 'cancel', 'BSTOP did not cancel batch decode')
            stage['passed'] = True
        finally:
            self._finish_stage(stage, start, [bat])
        healthy = self.make_attempt('healthy-batch-after-BSTOP', b_prompt, min(8, len(b_ref.tokens)), 1)
        self.run_batch_natural('healthy-batch-after-BSTOP', [healthy])
        require(healthy.tokens == b_ref.tokens[:len(healthy.tokens)], 'healthy batch after BSTOP differs')

    def completed_cache_retention(self, padded, tok):
        for offset in (1, 2, 3):
            size = 160 + (offset - 7 - 160) % PAGE
            seed = padded(f'HANDOFF-COMPLETED-COW-{offset}', size)
            seed_ref = self.solo(f'completed-cache-seed-ref-{offset}', seed, 8)
            require(len(seed_ref.tokens) == 8, 'seed EOS prevents partial-page completed-cache fixture')
            shared = seed + seed_ref.tokens[:-1]
            require(len(shared) % PAGE == offset, 'wrong completed-cache partial-page offset')
            suffix = tok.encode(f'\nFresh completed-cache branch {offset}: continue with exact numbering.\n')
            prompt = shared + suffix
            ref = self.solo(f'completed-cache-independent-ref-{offset}', prompt, 16)
            seeded = self.make_attempt(f'completed-cache-populate-slot-{offset}', seed, 8, 0)
            self.run_batch_natural(f'completed-cache-populate-slot-{offset}', [seeded])
            require(seeded.tokens == seed_ref.tokens, 'completed seed parity failed')
            req = self.make_attempt(f'completed-cache-warm-branch-{offset}', prompt, 16, 1)
            stage, start = self._stage(f'completed-cache-warm-branch-{offset}',
                                       {'shared_prefix_cells': len(shared), 'partial_page_offset': offset})
            protocol = common.Protocol([req])
            deadline = time.monotonic() + self.timeout
            try:
                self.engine.send(req.command(), deadline=deadline)
                while not protocol.finished:
                    protocol.consume(self.engine.next_event(deadline))
                common.parity(req, ref)
                stage['reuse'] = require_reuse(req, len(shared), req.name)
                stage['passed'] = True
            finally:
                self._finish_stage(stage, start, [req])


def append_turn(tok, cycle, order):
    return tok.encode(f'\nUser asks for continuation marker cycle {cycle} order {order}: keep counting.\n')


def run_handoff_suite(suite: Suite, tok, padded, pa, pb, cycles, segment_cap, resume_tokens, append_tokens, ref_new,
                      gate='minimal'):
    ref_a = suite.solo('solo-reference-A', pa, ref_new)
    ref_b = suite.solo('solo-reference-B', pb, ref_new)
    needed = cycles * (suite.prefix + resume_tokens) + suite.prefix + resume_tokens + 8
    require(len(ref_a.tokens) >= needed and len(ref_b.tokens) >= needed,
            'solo references ended too early for repeated handoff cycles; increase fixture/caps')
    logical = [LogicalResponse('A', pa, ref_a), LogicalResponse('B', pb, ref_b)]

    # Prewarm B before starting A: a large cold B prefill can legitimately
    # consume A's entire finite allowance before both rows become active.
    suite.warm_bgen('seed-idle-B-before-overlap', logical[1], resume_tokens, 1, 0)
    first = suite.main_handoff('main-A-HANDOFF-source', pa, segment_cap, suite.prefix)
    logical[0].accept(first, ('handoff',))
    first_reuse = expected_valid_prefix(len(pa), len(first.tokens))
    suite.warm_bgen('resume-BGEN-A-after-main-HANDOFF', logical[0], resume_tokens, 0, first_reuse)
    # Both streams have valid idle prefixes. Every pair entry must reuse them.
    last_cached = [len(item.prompt()) - 1 for item in logical]

    orders = [(0, 1), (1, 0)]
    for cycle in range(cycles):
        order = orders[cycle % 2]
        reqs = []
        before_prompts = {}
        required_entry = {}
        for index in order:
            slot = index  # reverse admission ORDER, not cache ownership/placement
            item = logical[index]
            before_prompts[item.name] = len(item.prompt())
            req = suite.make_attempt(f'cycle-{cycle}-BGEN-{item.name}-slot-{slot}',
                                     item.prompt(), segment_cap, slot)
            reqs.append(req)
            required_entry[req.name] = last_cached[index]
        suite.batch_handoff_pair(f'cycle-{cycle}-BHANDOFF-order-{order[0]}{order[1]}', reqs,
                                 suite.prefix, required_entry)
        handoff_cached = {}
        for req, index in zip(reqs, order):
            logical[index].accept(req, ('handoff',))
            # The next admission for this logical response must be able to reuse
            # through all but the unfed last token emitted by this request.
            handoff_cached[index] = expected_valid_prefix(before_prompts[logical[index].name], len(req.tokens))
            last_cached[index] = handoff_cached[index]

        # Resume A on the main decoder regardless of admission order; its next
        # batch admission should then reuse this completed continuation, not the
        # older BHANDOFF point.
        suite.warm_main(f'cycle-{cycle}-resume-main-A', logical[0], resume_tokens, handoff_cached[0])
        # Materialize the resumed main prefix back into its own idle slot before
        # a reverse-order B admission overwrites main. No snapshots are available.
        suite.warm_bgen(f'cycle-{cycle}-resume-batch-A', logical[0], resume_tokens, 0,
                        len(logical[0].prompt()) - 1)
        last_cached[0] = len(logical[0].prompt()) - 1

    # Next-turn branch checks are after the repeated handoff loop so they cannot
    # overwrite B's idle target-only checkpoint before a later cycle.  Run all
    # warm admissions before independent references so they cannot mask reuse.
    # Reference requests may reuse checkpoints; only the original solo streams
    # are cold baselines. Do not claim these next-turn oracles perform full replay.
    append_checks = []
    for index, item in enumerate(logical):
        base = item.prompt()
        appended_prompt = base + append_turn(tok, cycles, item.name)
        warm = suite.make_attempt(f'post-cycles-warm-{item.name}-append-turn', appended_prompt,
                                  append_tokens, index)
        stage, start = suite._stage(f'post-cycles-warm-{item.name}-append-turn',
                                    {'base_response_prefix_cells': len(base),
                                     'required_reuse_cells': last_cached[index]})
        protocol = common.Protocol([warm])
        deadline = time.monotonic() + suite.timeout
        try:
            suite.engine.send(warm.command(), deadline=deadline)
            while not protocol.finished:
                protocol.consume(suite.engine.next_event(deadline))
            common.normal(warm)
            stage['reuse'] = require_reuse(warm, last_cached[index], warm.name)
            append_checks.append((index, stage, warm, appended_prompt))
        finally:
            suite._finish_stage(stage, start, [warm])
    for index, stage, warm, appended_prompt in append_checks:
        reference = suite.solo(f'post-cycles-independent-ref-{logical[index].name}-append', appended_prompt, append_tokens)
        require(warm.tokens == reference.tokens, f'{warm.name}: warm append differs from independent reference')
        stage['reference_stage'] = reference.name
        stage['reference_cache_reuse_permitted'] = True
        stage['passed'] = True
        suite.save()

    if gate == 'full':
        suite.completed_cache_retention(padded, tok)
        suite.late_bhandoff_no_terminal(pb, ref_b)
        suite.cancellation_recovery(pa, pb, ref_a, ref_b)


def cleanup_check(evidence):
    inc.cleanup_check(evidence)
    handoff_cmds = [c for c in evidence['commands'] if c['raw'] == 'HANDOFF']
    bhandoff_cmds = [c for c in evidence['commands'] if c['raw'].startswith('BHANDOFF') and
                     c.get('stage') != 'late-BHANDOFF-after-natural-terminal']
    done_handoff = [e for e in evidence['stdout'] if e.get('kind') == 'DONE' and e.get('finish') == 'handoff']
    bdone_handoff = [e for e in evidence['stdout'] if e.get('kind') == 'BDONE' and e.get('finish') == 'handoff']
    require(len(done_handoff) == len(handoff_cmds), 'HANDOFF command/DONE handoff acknowledgement count mismatch')
    require(len(bdone_handoff) == len(bhandoff_cmds), 'BHANDOFF command/BDONE handoff acknowledgement count mismatch')
    evidence['handoff_ack_counts'] = {'HANDOFF_commands': len(handoff_cmds), 'DONE_handoff': len(done_handoff),
                                      'BHANDOFF_commands': len(bhandoff_cmds), 'BDONE_handoff': len(bdone_handoff)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exe', required=True, type=Path)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path, help='NEW JSON path; stderr beside it as .stderr.log')
    ap.add_argument('--context', type=int, default=1024)
    ap.add_argument('--resident', type=int, default=0)
    ap.add_argument('--cache-mib', type=int, default=0, help='must be 0: disables parked snapshots')
    ap.add_argument('--prompt-a-cells', type=int, default=160)
    ap.add_argument('--prompt-b-cells', type=int, default=164)
    ap.add_argument('--cycles', type=int, default=2, help='>=2 exercises both BGEN admission orders')
    ap.add_argument('--gate', choices=('minimal', 'full'), default='minimal',
                    help='minimal = two-direction handoff + warm next-turn only (GPU-window default); full adds late/cancel/completed-cache probes')
    ap.add_argument('--prefix', type=int, default=16, help='tokens to emit before HANDOFF/BHANDOFF')
    ap.add_argument('--segment-cap', type=int, help='cap for handoff segments; default max(4*prefix, prefix+32)')
    ap.add_argument('--resume-tokens', type=int, default=4)
    ap.add_argument('--append-tokens', type=int, default=4)
    ap.add_argument('--reference-new', type=int, default=256)
    ap.add_argument('--prefill', type=int, default=64)
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
    if a.cache_mib != 0:
        ap.error('--cache-mib must be 0 for this strong in-place handoff gate')
    if a.cycles < 2 or a.prefix < 2 or min(a.resume_tokens, a.append_tokens) < 1:
        ap.error('cycles >=2, prefix >=2, and positive resume/append caps are required')
    if a.segment_cap is None:
        a.segment_cap = max(a.prefix * 4, a.prefix + 32)
    if not a.prefix < a.segment_cap <= a.reference_new or a.reference_new < 64:
        ap.error('need prefix < segment-cap <= reference-new and reference-new >=64')
    if not 64 <= a.prefill <= 4096 or a.prefill % 64:
        ap.error('prefill must be 64..4096 in multiples of 64')
    if min(a.prompt_a_cells, a.prompt_b_cells) < 96:
        ap.error('prompt sizes must be at least 96 tokens')
    if a.gpu is not None and (not a.gpu.strip() or ',' in a.gpu):
        ap.error('--gpu must select exactly one device')
    if any(not math.isfinite(t) or t <= 0 for t in (a.startup_timeout, a.stage_timeout, a.cleanup_timeout)):
        ap.error('timeouts must be positive and finite')
    output = a.output.expanduser().resolve()
    stderr = Path(str(output) + '.stderr.log')
    if output.exists() or stderr.exists():
        ap.error('existing artifact; select a fresh --output')
    with output.open('x', encoding='utf-8') as f:
        f.write('{}\n')
    evidence = {'schema': 1, 'harness': 'kv-handoff', 'passed': False,
                'config': str(a.config.resolve()), 'exe': str(a.exe.resolve()),
                'context': a.context, 'resident': a.resident, 'cache_mib': a.cache_mib,
                'parking_disabled_required': True, 'seed': SEED, 'temperature': 0.0,
                'gate': a.gate, 'cycles': a.cycles, 'prefix': a.prefix, 'segment_cap': a.segment_cap,
                'resume_tokens': a.resume_tokens, 'append_tokens': a.append_tokens,
                'reference_new': a.reference_new, 'prefill': a.prefill,
                'stderr_path': str(stderr), 'stdout': [], 'commands': [], 'stages': [],
                'timeouts': {'startup_s': a.startup_timeout, 'stage_s': a.stage_timeout,
                             'cleanup_s': a.cleanup_timeout}}
    engine = Engine(evidence, stderr, a.cleanup_timeout)
    suite = Suite(engine, evidence, output, a.stage_timeout, a.context, a.prefix)
    success = False
    try:
        cfg = json.loads(a.config.read_text(encoding='utf-8-sig'))
        command, cwd, env, kv = settings(cfg, a.exe, a.context, a.resident, a.cache_mib, a.gpu,
                                        a.vram_reserve_mib, a.prefill)
        evidence.update(command=command, cwd=cwd, kv=kv,
                        device_env={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                                                       'STRATA_IQ_MT_MIN') if k in env})
        suite.save()
        tok = common.tokenizer(common.resolve_path(cfg['tokenizer'], cwd))
        padded, pa, pb = inc.fixtures(tok, a.prompt_a_cells, a.prompt_b_cells)
        require(len(pa) + a.reference_new + GUARD <= a.context and len(pb) + a.reference_new + GUARD <= a.context,
                'reference caps do not fit logical context')
        engine.start(command, cwd, env, a.context, a.resident, a.cache_mib, a.startup_timeout)
        run_handoff_suite(suite, tok, padded, pa, pb, a.cycles, a.segment_cap,
                          a.resume_tokens, a.append_tokens, a.reference_new, a.gate)
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
            evidence['cleanup_failure'] = {'type': type(exc).__name__, 'message': str(exc),
                                           'traceback': traceback.format_exc()}
        if stderr.exists():
            text = stderr.read_text(errors='replace')
            evidence['native_diagnostics'] = streaming.diagnostics(text)
            evidence['parking_snapshot_diagnostics'] = [d for d in evidence['native_diagnostics']
                                                        if d.get('kind') in ('park', 'restore')]
            if success and evidence['parking_snapshot_diagnostics']:
                success = False
                evidence['cleanup_failure'] = {'type': 'AssertionError',
                                               'message': 'conversation parking snapshot used despite cache-mib=0'}
        evidence['passed'] = success
        suite.save()
    print(f'{"PASS" if success else "FAIL"}: {output}; stderr: {stderr}', flush=True)
    if not success:
        print(json.dumps(evidence.get('failure') or evidence.get('cleanup_failure')), file=sys.stderr)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
