#!/usr/bin/env python3
"""Pure CPU harness regressions: protocol, rolling capacity, replay and local HTTP.

  python3 tools/test_incremental_kv_smoke.py

Only tiny Python protocol stubs and localhost stdlib HTTP fixtures are launched;
NEVER invokes Strata, a tokenizer download, a model, CUDA, SSH or deployment.
"""
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incremental_kv_smoke as smoke
import incremental_kv_http_smoke as http
import incremental_kv_http_pressure_smoke as pressure_http


def event(raw, seq=0):
    return {**smoke.common.parse_line(raw), 'seq': seq, 'raw': raw, 'wall_s': seq / 100}


def feed(protocol, lines):
    for seq, raw in enumerate(lines):
        protocol.consume(event(raw, seq))


def info(context=1024, cache=0):
    return {'kv_incremental': '1', 'kv_reserve_ahead': '256', 'kv_unified': '1',
            'context': str(context), 'kv_capacity_cells': str(context), 'batch_slots': '2',
            'kv_resident': '0', 'conversation_cache_mib': str(cache), 'slot_cache': '1',
            'lookup': '0', 'mtp_max': '1', 'spec': '2', 'pcie_frac': '0.00'}


def data():
    return {'stdout': [], 'commands': [], 'stages': [], 'cache_mib': 0, 'expect_replay': True}


def complete(req, outputs, finish='length'):
    if req.slot is None:
        lines = [f'T {t}' for t in outputs] + [f'DONE {len(outputs)} {len(req.prompt)} 0 0 {finish} 0 0 0']
    else:
        lines = [f'T {outputs[0]}', f'DONE 1 {len(req.prompt)} 0 0 length 0 0 0', f'BADM {req.slot} 1']
        lines += [f'BT {req.slot} {t}' for t in outputs[1:]]
        lines += [f'BDONE {req.slot} {len(outputs)} {finish} 0']
    feed(smoke.common.Protocol([req]), lines)
    return req


class WordTokenizer:
    def __init__(self):
        self.words = {}

    def encode(self, text, parse_special=False):
        return [self.words.setdefault(w, len(self.words) + 1) for w in text.split()]


class ParserCapacityTests(unittest.TestCase):
    def test_pressure_parse_and_count_contract(self):
        e = event('BDONE 1 357 pressure 12.5')
        self.assertEqual((e['slot'], e['count'], e['finish']), (1, 357, 'pressure'))
        req = smoke.Attempt('pressure', [1, 2], 4, 1)
        complete(req, [8, 9], 'pressure')
        self.assertEqual(req.tokens, [8, 9])  # admission T INCLUDED
        with self.assertRaises(AssertionError):
            smoke.common.normal(req)  # internal stop is NOT logical success

    def test_bad_parser_records(self):
        for line in ('BDONE x 3 pressure 1', 'BDONE 1 3 pressure', 'BT 1 nope', 'BADM 0 2'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                event(line)

    def test_pressure_after_done_does_not_accept_inherited_bt(self):
        req = smoke.Attempt('a', [1], 4, 0)
        protocol = smoke.common.Protocol([req])
        feed(protocol, ['T 8', 'DONE 1 1 0 0 length', 'BADM 0 1', 'BT 0 9', 'BDONE 0 2 pressure 1'])
        self.assertTrue(protocol.finished)
        with self.assertRaisesRegex(AssertionError, 'orphan'):
            protocol.consume(event('BT 0 10', 6))

    def test_initial_headroom_not_full_nominal_reservation(self):
        plan = smoke.growth_plan([[1] * 160, [2] * 164], [512, 512], 1024)
        self.assertLessEqual(sum(plan['initial_pages_upper_bound']), 256)
        self.assertGreater(sum(plan['nominal_pages']), 256)
        self.assertEqual(plan['reserve_ahead_cells'], 256)
        self.assertTrue(plan['pressure_requires_native_BDONE_not_reservation_arithmetic'])

    def test_rounding_and_logical_remaining(self):
        self.assertEqual([smoke.pages(n) for n in range(9)], [0, 1, 1, 1, 1, 2, 2, 2, 2])
        self.assertEqual(smoke.remaining_context(1024, [1] * 160), 856)
        with self.assertRaises(AssertionError):
            smoke.pages(-1)

    def test_false_capacity_plans_rejected(self):
        for prompts, targets, context in (([[1] * 160, [2] * 164], [256, 256], 512),
                                           ([[1] * 160, [2] * 164], [16, 16], 1024),
                                           ([[1] * 160, [2] * 164], [1000, 512], 1024),
                                           ([[1] * 160, [2] * 164], [512, 512], 1025)):
            with self.subTest(context=context, targets=targets), self.assertRaises(AssertionError):
                smoke.growth_plan(prompts, targets, context)

    def test_rolling_growth_requires_actual_consumption_beyond_256_headroom(self):
        for offset in (1, 2, 3):
            prompt = [1] * (160 + offset)
            req = complete(smoke.Attempt('rolling', prompt, 288, 0), list(range(288)))
            proof = smoke.rolling_growth_evidence(req)
            self.assertGreater(proof['minimum_pages_beyond_initial_headroom'], 0)
        short = complete(smoke.Attempt('short', [1] * 160, 32, 0), list(range(32)))
        early = complete(smoke.Attempt('eos', [1] * 160, 288, 0), list(range(40)), 'stop')
        for req in (short, early):
            with self.assertRaisesRegex(AssertionError, 'initial 256-cell headroom'):
                smoke.rolling_growth_evidence(req)

    def test_info_requires_incremental_contract(self):
        smoke.verify_info(info(), 1024, 0, 0)
        for key, value in (('kv_incremental', '0'), ('kv_reserve_ahead', '0'), ('kv_capacity_cells', '2048'),
                            ('batch_slots', '1')):
            broken = {**info(), key: value}
            with self.subTest(key=key), self.assertRaises(AssertionError):
                smoke.verify_info(broken, 1024, 0, 0)
        broken = info()
        del broken['kv_incremental']
        with self.assertRaises(AssertionError):
            smoke.verify_info(broken, 1024, 0, 0)

    def test_fresh_beginning_tags_and_partial_pages(self):
        padded, a, b = smoke.fixtures(WordTokenizer(), 160, 164)
        self.assertEqual((len(a), len(b)), (160, 164))
        self.assertNotEqual(a[0], b[0])
        self.assertNotIn(padded('YIELD-BEGINNING', 324)[0], [a[0], b[0]])
        for offset in (1, 2, 3):
            size = 160 + (offset - 7 - 160) % 4
            self.assertEqual((len(padded(f'COW-{offset}', size)) + 7) % 4, offset)


class ReplayTests(unittest.TestCase):
    def test_replay_all_tokens_same_seed_remaining_budget_absolute_position(self):
        logical = smoke.LogicalRequest('a', [1, 2, 3], 6, 0)
        reference = complete(smoke.Attempt('solo', logical.original, 6), [10, 11, 12, 13, 14, 15])
        first = logical.attempt()
        complete(first, [10, 11], 'pressure')
        logical.accept(first, reference)
        self.assertIsNone(logical.finish)
        second = logical.attempt()
        self.assertEqual(second.prompt, [1, 2, 3, 10, 11])  # last 11 is UNFED, must be submitted
        self.assertEqual(second.cap, 4)
        self.assertIn('seed=12345', first.command())
        self.assertIn('seed=12345', second.command())
        self.assertNotIn('rng_offset', second.command())
        self.assertEqual(logical.record()['resume_position'], 4)
        complete(second, [12, 13, 14, 15])
        logical.accept(second, reference)
        self.assertEqual(logical.tokens, reference.tokens)
        self.assertEqual(logical.finish, 'length')
        with self.assertRaises(AssertionError):
            logical.attempt()

    def test_sampled_replay_retains_temperature_seed_no_offset(self):
        logical = smoke.LogicalRequest('sampled', [1, 2], 4, 0, seed=54321, temperature=.3)
        reference = complete(smoke.Attempt('ref', [1, 2], 4, temperature=.3, seed=54321), [10, 11, 12, 13])
        first = logical.attempt()
        complete(first, [10, 11], 'pressure')
        logical.accept(first, reference)
        resumed = logical.attempt()
        self.assertIn('temperature=0.3 seed=54321', resumed.command())
        self.assertNotIn('rng_offset', resumed.command())
        self.assertEqual(resumed.record()['temperature'], .3)
        logical.temperature = .4
        with self.assertRaisesRegex(AssertionError, 'sampling keys changed'):
            logical.attempt()

    def test_multiple_pressure_attempts_never_reset_budget(self):
        logical = smoke.LogicalRequest('a', [1, 2], 8, 0)
        reference = complete(smoke.Attempt('solo', logical.original, 8), list(range(10, 18)))
        for outputs in ([10, 11], [12, 13], [14, 15]):
            req = logical.attempt()
            complete(req, outputs, 'pressure')
            logical.accept(req, reference)
        self.assertEqual(logical.attempt().cap, 2)
        self.assertEqual(logical.attempts[-1].prompt[-1], 15)

    def test_bad_seed_lost_last_token_wrong_remaining_and_wrong_output_fail(self):
        for corruption in ('seed', 'last', 'cap', 'output'):
            logical = smoke.LogicalRequest('a', [1, 2], 4, 0)
            reference = complete(smoke.Attempt('solo', logical.original, 4), [10, 11, 12, 13])
            first = logical.attempt()
            complete(first, [10, 11], 'pressure')
            logical.accept(first, reference)
            if corruption == 'seed':
                logical.seed = 678
                with self.assertRaises(AssertionError):
                    logical.attempt()
                continue
            second = logical.attempt()
            if corruption == 'last':
                second.prompt = second.prompt[:-1]
            if corruption == 'cap':
                second.cap = 4
            complete(second, [99 if corruption == 'output' else 12, 13])
            with self.subTest(corruption=corruption), self.assertRaises(AssertionError):
                logical.accept(second, reference)

    def test_pressure_with_zero_progress_or_final_budget_fails(self):
        logical = smoke.LogicalRequest('a', [1], 2, 0)
        reference = complete(smoke.Attempt('solo', [1], 2), [8, 9])
        req = logical.attempt()
        complete(req, [8, 9], 'pressure')
        with self.assertRaisesRegex(AssertionError, 'nonfinal'):
            logical.accept(req, reference)

    def test_record_does_not_claim_nominal_allowance_is_reserved(self):
        record = smoke.Attempt('a', [1] * 160, 856, 0).record()
        self.assertNotIn('reservation_pages', record)
        self.assertEqual(record['nominal_allowance_cells'], 1016)
        self.assertEqual(record['initial_pages_upper_bound'], 104)


class ModelProtocolStub:
    """Deterministic CPU token schedule. Native model parity is NOT established here."""
    def __init__(self, evidence, log, eos=False):
        self.evidence, self.stderr_path, self.eos = evidence, log, eos
        self.queue = deque()
        self.stage = 'stub'
        log.write_text('')
        self.bursts = []
        self.waiting_survivor = None
        self.sampling_values = []

    def send(self, *lines, deadline=None):
        self.bursts.append(lines)
        self.evidence['commands'].extend({'raw': line, 'stage': self.stage} for line in lines)
        parsed = []
        for line in lines:
            parts = line.split()
            batch = parts[0] == 'BGEN'
            slot = int(parts[1]) if batch else None
            cap = int(parts[2] if batch else parts[1])
            ids = list(map(int, parts[-1].split(',')))
            original_len = 160 if ids[0] == 1 else 164
            produced = len(ids) - original_len
            temperature = float(next(x.split('=', 1)[1] for x in parts if x.startswith('temperature=')))
            seed = int(next(x.split('=', 1)[1] for x in parts if x.startswith('seed=')))
            self.sampling_values.append((temperature, seed))
            # CPU protocol fixture only, NOT actual Philox/model validation.
            # Sampled tokens depend on stable keys and absolute input position.
            if temperature:
                outputs = [ids[0] * 10000 + (seed + int(temperature * 1000) + original_len - 1 + produced + i) % 997
                           for i in range(cap)]
            else:
                outputs = [ids[0] * 10000 + produced + i for i in range(cap)]
            parsed.append((slot, ids, cap, outputs))
        if len(parsed) == 2:
            for slot, ids, _, outputs in parsed:
                self.queue.extend([f'T {outputs[0]}', f'DONE 1 {len(ids)} 0 0 length 0 0 0', f'BADM {slot} 1'])
            left, right = parsed
            nleft, nright = (2, 2) if self.eos else (400, 300)
            for i in range(1, max(nleft, nright)):
                for (slot, _, _, outputs), count in zip(parsed, (nleft, nright)):
                    if i < count:
                        self.queue.append(f'BT {slot} {outputs[i]}')
            self.queue.append(f'BDONE {left[0]} {nleft} {"stop" if self.eos else "pressure"} 1')
            for i in range(nright, right[2]):
                self.queue.append(f'BT {right[0]} {right[3][i]}')
            self.queue.append(f'BDONE {right[0]} {right[2]} length 1')
            self.waiting_survivor = right
        else:
            slot, ids, cap, outputs = parsed[0]
            if slot is not None and self.waiting_survivor and self.queue:
                survivor = self.waiting_survivor
                # Admission requeue waits on another active history. After a
                # bounded 256-token quantum native pressure-releases that row.
                if survivor[2] > 556:
                    self.queue.clear()
                    self.queue.extend(f'BT {survivor[0]} {survivor[3][i]}' for i in range(300, 556))
                    self.queue.append(f'BDONE {survivor[0]} 556 pressure 1')
                self.waiting_survivor = None
            if slot is None:
                self.queue.extend([f'T {t}' for t in outputs])
                self.queue.append(f'DONE {cap} {len(ids)} 0 0 length 0 0 0')
            else:
                self.queue.extend([f'T {outputs[0]}', f'DONE 1 {len(ids)} 0 0 length 0 0 0', f'BADM {slot} 1'])
                self.queue.extend([f'BT {slot} {t}' for t in outputs[1:]])
                self.queue.append(f'BDONE {slot} {cap} length 1')

    def next_event(self, deadline):
        if not self.queue:
            raise TimeoutError('CPU protocol stub ran out of events')
        parsed = event(self.queue.popleft(), len(self.evidence['stdout']))
        self.evidence['stdout'].append(parsed)
        return parsed


class LateStopStub(ModelProtocolStub):
    """CPU ownership/transport schedule; does NOT validate the native predicate."""
    def __init__(self, evidence, log, duplicate_when=None):
        super().__init__(evidence, log)
        self.owned = {0: False, 1: False}
        self.cached = {0: False, 1: False}
        self.duplicate_when = duplicate_when
        self.stop_observations = []
        self.control_bursts = []

    def next_event(self, deadline):
        e = super().next_event(deadline)
        if e['kind'] == 'BADM':
            self.owned[e['slot']] = e['continues']
        elif e['kind'] == 'BDONE':
            self.owned[e['slot']] = False
            self.cached[e['slot']] = e['finish'] == 'length'
        return e

    def send(self, *lines, deadline=None):
        if lines[0].startswith('BSTOP '):
            slot = int(lines[0].split()[1])
            phase = 'natural' if self.cached[slot] else 'pressure'
            if self.owned[slot] or self.queue:
                raise AssertionError('late-stop fixture still has an owned row / undrained replies')
            self.stop_observations.append({'owned': self.owned[slot], 'cached': self.cached[slot], 'phase': phase})
            self.control_bursts.append(lines)
            self.evidence['commands'].append({'raw': lines[0], 'stage': self.stage})
            super().send(*lines[1:], deadline=deadline)
            if self.duplicate_when == phase:
                self.queue.appendleft(f'BDONE {slot} 0 cancel 0')
        else:
            super().send(*lines, deadline=deadline)


class LateStopTests(unittest.TestCase):
    def test_pressure_reset_and_natural_cached_idle_late_stop_own_no_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = data()
            engine = LateStopStub(evidence, Path(tmp) / 'stderr')
            suite = smoke.Suite(engine, evidence, Path(tmp) / 'out.json', 2, 1024)
            suite.late_stop([1] * 160, [2] * 164, 852)
            self.assertTrue(evidence['late_stop_no_ack_owned_verified'])
            self.assertEqual(len(evidence['stages']), 5)  # 2 short solos, ONE pressure pair, 2 readmissions
            self.assertEqual(engine.stop_observations, [{'owned': False, 'cached': False, 'phase': 'pressure'},
                                                       {'owned': False, 'cached': True, 'phase': 'natural'}])
            self.assertEqual(len(engine.control_bursts), 2)
            for burst, stage in zip(engine.control_bursts, evidence['stages'][-2:]):
                self.assertEqual(burst[0], 'BSTOP 0')
                self.assertTrue(burst[1].startswith('BGEN 0 16 '))
                self.assertEqual(len(burst), 2)
                self.assertTrue(stage['deliberate_late_BSTOP'])
                self.assertIn('NoAckOwned', stage['ownership_contract'])
                self.assertEqual(stage['terminal_count'], 1)
                self.assertTrue(stage['no_extra_old_BDONE'])
                self.assertTrue(stage['passed'])
            self.assertEqual(json.loads(suite.output.read_text()), evidence)

    def test_old_duplicate_bdone_is_rejected_at_both_late_stop_boundaries(self):
        for phase in ('pressure', 'natural'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                evidence = data()
                engine = LateStopStub(evidence, Path(tmp) / 'stderr', duplicate_when=phase)
                suite = smoke.Suite(engine, evidence, Path(tmp) / 'out.json', 2, 1024)
                with self.assertRaisesRegex(AssertionError, 'orphan batch event'):
                    suite.late_stop([1] * 160, [2] * 164, 852)
                failed = evidence['stages'][-1]
                self.assertFalse(failed['passed'])
                self.assertTrue(failed['deliberate_late_BSTOP'])
                self.assertIn('NoAckOwned', failed['ownership_contract'])
                self.assertEqual(failed['preamble_commands'], ['BSTOP 0'])
                self.assertEqual(evidence['stdout'][-1]['raw'], 'BDONE 0 0 cancel 0')
                self.assertEqual(json.loads(suite.output.read_text())['stages'], evidence['stages'])

    def test_late_stop_cli_selects_only_focused_gate_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output = Path(tmp) / 'cfg.json', Path(tmp) / 'out.json'
            cfg.write_text(json.dumps({'args': ['--kv', 'int8'], 'tokenizer': 'unused'}))
            with patch.object(smoke.common, 'tokenizer', return_value=WordTokenizer()), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = smoke.main(['--exe', '/no/such/native', '--config', str(cfg), '--output', str(output),
                                     '--gate', 'late-stop', '--context', '1024', '--cache-mib', '0'])
            self.assertEqual(result, 1)  # missing executable deliberately prevents model execution
            saved = json.loads(output.read_text())
            self.assertEqual(saved['gate'], 'late-stop')
            self.assertEqual(saved['pressure_target'], 852)
            self.assertEqual(saved['pressure_plan']['initial_pages_upper_bound'], [104, 105])
            self.assertEqual(saved['failure']['type'], 'FileNotFoundError')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            smoke.main(['--exe', '/unused', '--config', '/unused', '--output', '/unused',
                        '--gate', 'late-stop', '--context', '262144'])


class SuiteTests(unittest.TestCase):
    def test_both_requests_pressure_and_resume_with_reversed_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = data()
            engine = ModelProtocolStub(evidence, Path(tmp) / 'stderr')
            suite = smoke.Suite(engine, evidence, Path(tmp) / 'out.json', 2, 1024)
            suite.pressure([1] * 160, [2] * 164, 852)
            self.assertEqual(evidence['pressure_requests_exercised'], [0, 1])
            self.assertTrue(all(s['passed'] for s in evidence['stages']))
            fifo = [s for s in evidence['stages'] if 'FIFO' in s['name']]
            self.assertEqual(len(fifo), 2)
            for stage in fifo:
                self.assertEqual(len(stage['fallback_replays']), 2)
                self.assertEqual(len(stage['requeues']), 2)
                self.assertEqual(stage['requeues'][1]['pending_admissions'], 2)
                for logical in stage['logical_requests']:
                    self.assertEqual(len(logical['tokens']), 852)
                    count = len(logical['attempts'][0]['tokens'])
                    self.assertIn(count, (400, 556))
                    self.assertEqual(logical['attempts'][1]['max_new'], 852 - count)
                    self.assertEqual(len(logical['attempts'][1]['prompt_ids']), len(logical['original_ids']) + count)
            self.assertEqual(json.loads(suite.output.read_text())['stages'], evidence['stages'])

    def test_natural_eos_is_not_exhaustion_and_failure_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = data()
            engine = ModelProtocolStub(evidence, Path(tmp) / 'stderr', eos=True)
            suite = smoke.Suite(engine, evidence, Path(tmp) / 'out.json', 2, 1024)
            with self.assertRaisesRegex(AssertionError, 'no actual pressure|final solo parity/length differs'):
                suite.pressure([1] * 160, [2] * 164, 512)
            failed = evidence['stages'][-1]
            self.assertFalse(failed['passed'])
            self.assertTrue(failed['requests'][0]['completion'])
            self.assertFalse(json.loads(suite.output.read_text())['stages'][-1]['passed'])

    def test_no_overlap_or_one_row_with_no_concurrent_bt_fails(self):
        a, b = smoke.Attempt('a', [1], 8, 0), smoke.Attempt('b', [2], 8, 1)
        for req in (a, b):
            complete(req, list(range(8)))
        # Both have identical sequences from separate streams, not valid overlap.
        a.completion['seq'] = b.badm['seq']
        with self.assertRaisesRegex(AssertionError, 'no overlap'):
            smoke.overlap_proof([a, b])
        a.completion['seq'] = b.completion['seq'] = 20
        a.token_events = [e for e in a.token_events if e['kind'] == 'T']
        with self.assertRaisesRegex(AssertionError, 'both rows'):
            smoke.overlap_proof([a, b])

    def test_sampled_fifo_solo_parity_and_every_command_same_sampling_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = data()
            engine = ModelProtocolStub(evidence, Path(tmp) / 'stderr')
            suite = smoke.Suite(engine, evidence, Path(tmp) / 'out.json', 2, 1024, temperature=.3, seed=12345)
            suite.pressure([1] * 160, [2] * 164, 852)
            self.assertEqual(set(engine.sampling_values), {(.3, 12345)})
            self.assertEqual(evidence['pressure_requests_exercised'], [0, 1])
            for stage in evidence['stages']:
                self.assertTrue(stage['passed'])
                for req in stage.get('requests', []):
                    self.assertEqual(req['temperature'], .3)
                    self.assertEqual(req['seed'], 12345)
                    self.assertFalse(req['rng_offset_added'])

    def test_target_only_pressure_restore_requires_matching_positive_diagnostics(self):
        logical = smoke.LogicalRequest('a', [1, 2], 4, 0)
        reference = complete(smoke.Attempt('ref', [1, 2], 4), [10, 11, 12, 13])
        before = logical.attempt()
        complete(before, [10, 11], 'pressure')
        logical.accept(before, reference)
        after = logical.attempt()
        complete(after, [12, 13])
        after.admission_done['reused'] = 3
        records = smoke.pressure_diagnostics('strata serve: pressure parked target-only 3 tokens; parked=1 bytes=2048\n')
        restore = smoke.streaming.diagnostics('conversation cache: restored 3 tokens (live) in 1.0 ms; parked=1 bytes=2048\n')
        proof = smoke.verify_pressure_restore([logical], records, restore)
        self.assertTrue(proof[0]['last_unfed_token_replayed'])
        self.assertFalse(proof[0]['mtp_draft_coherence_exercised'])
        with self.assertRaises(AssertionError):
            smoke.verify_pressure_restore([logical], records, [])
        with self.assertRaises(AssertionError):
            smoke.verify_pressure_restore([logical], [], restore)
        with self.assertRaisesRegex(AssertionError, 'MAIN GEN'):
            smoke.verify_pressure_restore([logical], records, restore, [])
        coherence = smoke.pressure_diagnostics(
            'TARGET_ONLY restore: private MTP proposals suppressed until full replay\n'
            'TARGET_ONLY decode: T=1, MTP/suffix proposals disabled\n')
        smoke.verify_pressure_restore([logical], records, restore, coherence)
        miss = smoke.pressure_diagnostics('strata serve: pressure cache budget/RAM miss; replay 3 tokens\n')
        self.assertEqual(miss[0]['kind'], 'pressure_replay_miss')

    def test_enabled_positive_pressure_restore_separately_reclaims_idle_capacity(self):
        class CanonicalStub(ModelProtocolStub):
            def send(self, *lines, deadline=None):
                super().send(*lines, deadline=deadline)
                if len(lines) == 2:
                    with self.stderr_path.open('a') as log:
                        log.write('strata serve: pressure parked target-only 559 tokens; parked=1 bytes=2048\n')
                else:
                    ids = lines[0].split()[-1].split(',')
                    if len(ids) == 560:
                        updated = deque()
                        for raw in self.queue:
                            if raw.startswith('DONE '):
                                fields = raw.split()
                                fields[8] = '559'
                                raw = ' '.join(fields)
                            updated.append(raw)
                        self.queue = updated
                        with self.stderr_path.open('a') as log:
                            log.write('conversation cache: restored 559 tokens (live) in 1.0 ms; parked=1 bytes=2048\n')
                            log.write('strata serve: TARGET_ONLY restore: private MTP proposals suppressed until full replay\n')
                            log.write('strata serve: TARGET_ONLY decode: T=1, MTP/suffix proposals disabled\n')
        with tempfile.TemporaryDirectory() as tmp:
            evidence = {**data(), 'expect_replay': False, 'cache_mib': 4096, 'require_pressure_restore': True}
            engine = CanonicalStub(evidence, Path(tmp) / 'stderr')
            suite = smoke.Suite(engine, evidence, Path(tmp) / 'out.json', 2, 1024)
            suite.pressure([1] + [3] * 159, [2] + [4] * 163, 852)
            restored = next(s for s in evidence['stages'] if s['name'] == 'canonical-target-only-pressure-restore-main-GEN')
            self.assertTrue(restored['passed'])
            self.assertTrue(restored['target_only_pressure_restore'])
            self.assertIsNone(restored['requests'][0]['slot'])  # MAIN GEN, not a BGEN T1 shortcut
            self.assertEqual(restored['restored_main_drafts_offered'], 0)
            self.assertTrue(all(s['passed'] for s in evidence['stages']))
            sentinel = next(s for s in evidence['stages'] if s['name'] == 'canonical-free-idle-backing')
            self.assertEqual(sentinel['requests'][0]['prompt_ids'], [3] * 128)

    def test_too_short_target_does_not_credit_only_one_pressure_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = data()
            engine = ModelProtocolStub(evidence, Path(tmp) / 'stderr')
            suite = smoke.Suite(engine, evidence, Path(tmp) / 'out.json', 2, 1024)
            with self.assertRaisesRegex(AssertionError, 'both owners must pressure-stop'):
                suite.pressure([1] * 160, [2] * 164, 512)
            self.assertFalse(evidence['stages'][-1]['passed'])
            self.assertTrue(evidence['stages'][-1]['requeues'])

    def test_abandoned_pressure_owner_then_same_slot_has_only_healthy_replies(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = data()
            engine = ModelProtocolStub(evidence, Path(tmp) / 'stderr')
            suite = smoke.Suite(engine, evidence, Path(tmp) / 'out.json', 2, 1024)
            suite.pressure([1] * 160, [2] * 164, 852, abandon=True)
            parked = next(s for s in evidence['stages'] if s['name'] == 'cancel-owner-parked-after-pressure')
            healthy = evidence['stages'][-1]
            self.assertTrue(parked['passed'])
            self.assertTrue(healthy['passed'])
            self.assertEqual(healthy['requests'][0]['slot'], parked['cancelled_owner_slot'])
            self.assertEqual(len(healthy['requests'][0]['tokens']), 16)

    def test_paused_owner_stop_while_admission_waits_and_fresh_recovery(self):
        class PausedStub(ModelProtocolStub):
            def send(self, *lines, deadline=None):
                self.bursts.append(lines)
                self.evidence['commands'].extend({'raw': line, 'stage': self.stage} for line in lines)
                first = lines[0].split()
                batch = first[0] == 'BGEN'
                slot = int(first[1]) if batch else None
                cap = int(first[2] if batch else first[1])
                prompt = first[-1].split(',')
                if len(lines) == 2 and lines[1] == 'BYIELD 0':
                    self.queue.extend(['YIELDED 0 64', f'DONE 0 {len(prompt)} 0 0 cancel 0 0 0'])
                    return
                if len(lines) == 2 and lines[1] == 'BSTOP 0':
                    self.queue.append('BDONE 0 0 cancel 0')
                outputs = [60000 + len(prompt) + i for i in range(cap)]
                if batch:
                    self.queue.extend([f'T {outputs[0]}', f'DONE 1 {len(prompt)} 0 0 length 0 0 0', f'BADM {slot} 1'])
                    self.queue.extend(f'BT {slot} {t}' for t in outputs[1:])
                    self.queue.append(f'BDONE {slot} {cap} length 0')
                else:
                    self.queue.extend(f'T {t}' for t in outputs)
                    self.queue.append(f'DONE {cap} {len(prompt)} 0 0 length 0 0 0')
        with tempfile.TemporaryDirectory() as tmp:
            evidence = data()
            engine = PausedStub(evidence, Path(tmp) / 'stderr')
            suite = smoke.Suite(engine, evidence, Path(tmp) / 'out.json', 2, 1024)
            padded, _, _ = smoke.fixtures(WordTokenizer(), 160, 164)
            suite.cancel_waiting(padded)
            stage = evidence['stages'][0]
            self.assertTrue(stage['passed'])
            self.assertEqual(stage['release']['finish'], 'cancel')
            self.assertEqual(stage['source']['tokens'], [])
            self.assertEqual(len(stage['waiter']['prompt_ids']), 964)
            self.assertLess(stage['release']['seq'], stage['waiter']['token_events'][0]['seq'])
            self.assertTrue(evidence['stages'][-1]['passed'])

    def test_positive_restore_diagnostics_cannot_be_slot_copy(self):
        ref = complete(smoke.Attempt('solo', [1] * 200, 2), [7, 8])
        restored = complete(smoke.Attempt('returned', [1] * 200, 2), [7, 8])
        restored.admission_done['reused'] = 192
        park = {'diagnostics': smoke.streaming.diagnostics(
            'conversation cache: parked 192 tokens in 1.0 ms; parked=1 bytes=2048 evictions=0 snapshot_bytes=2048\n')}
        positive = {'diagnostics': smoke.streaming.diagnostics(
            'conversation cache: restored 192 tokens (live) in 1.0 ms; parked=1 bytes=2048\n')}
        smoke.streaming.verify_park_restore(park, positive, restored, ref, 192, 4096)
        copy = {'diagnostics': smoke.streaming.diagnostics(
            'strata batch: slot 0 gave back 192 tokens of this conversation (all it holds) in 1.0 ms\n')}
        with self.assertRaises(AssertionError):
            smoke.streaming.verify_park_restore(park, copy, restored, ref, 192, 4096)


class ConfigLifecycleTests(unittest.TestCase):
    def test_reuses_actual_config_env_and_frozen_placement_with_no_writes(self):
        cfg = {'cwd': '/tmp', 'gpu': 3, 'env': {'TUNING': 'keep'}, 'lib_dirs': ['libs'],
               'args': ['--native', 'actual-model', '--kv', 'int8', '--vram-reserve-mib', '700',
                        '--slots', '8', '--suffix-draft', '32', '--adapt-swaps', '40']}
        command, cwd, env, _ = smoke.settings(cfg, '/tmp/engine', 1024, 0, 0)
        self.assertIn('actual-model', command)
        self.assertEqual(env['TUNING'], 'keep')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '3')
        self.assertTrue(env['LD_LIBRARY_PATH'].startswith('/tmp/libs'))
        for option, expected in (('--max-context', '1024'), ('--batch', '2'), ('--kv-resident', '0'),
                                 ('--suffix-draft', '0'), ('--adapt-swaps', '0'), ('--vram-reserve-mib', '2048'),
                                 ('--prefill', '64')):
            self.assertEqual(command.count(option), 1)
            self.assertEqual(command[command.index(option) + 1], expected)
        self.assertNotIn('--slots', command)
        self.assertNotIn('--lookup', command)
        self.assertEqual(cfg['args'][-1], '40')

    def test_prefill_override_preserves_fused_env_and_removes_config_duplicates(self):
        cfg = {'args': ['--kv', 'int8', '--prefill', 'auto', '--prefill', '512'],
               'env': {'STRATA_PF_FUSED': '1', 'STRATA_PREFILL_FUSED_TAIL': '0'}}
        for prefill in (64, 1024, 4096):
            with self.subTest(prefill=prefill):
                command, _, env, _ = smoke.settings(cfg, '/tmp/native', 262144, 32768, 4096, prefill=prefill)
                self.assertEqual(command.count('--prefill'), 1)
                self.assertEqual(command[command.index('--prefill') + 1], str(prefill))
                self.assertEqual(env['STRATA_PF_FUSED'], '1')
                self.assertEqual(env['STRATA_PREFILL_FUSED_TAIL'], '0')
                self.assertEqual(command[command.index('--mtp-max-t') + 1], '1')
                self.assertEqual(command[command.index('--adapt-swaps') + 1], '0')
        self.assertEqual(cfg['args'], ['--kv', 'int8', '--prefill', 'auto', '--prefill', '512'])
        for invalid in (-64, 0, 32, 65, 4097, 8192):
            with self.subTest(prefill=invalid), self.assertRaises(AssertionError):
                smoke.settings(cfg, '/tmp/native', 1024, 0, 0, prefill=invalid)
            with patch.object(smoke.Engine, 'start') as start, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                smoke.main(['--exe', '/unused', '--config', '/unused', '--output', '/unused', '--prefill', str(invalid)])
            start.assert_not_called()

    def test_large_overlap_prefill_cli_evidence_without_native_or_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output = Path(tmp) / 'cfg.json', Path(tmp) / 'out.json'
            cfg.write_text(json.dumps({'args': ['--kv', 'int8', '--prefill', 'auto'], 'tokenizer': 'unused',
                                       'env': {'STRATA_PF_FUSED': '1', 'STRATA_PREFILL_FUSED_TAIL': '0'}}))
            with patch.object(smoke.common, 'tokenizer', return_value=WordTokenizer()), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = smoke.main(['--exe', '/no/such/engine', '--config', str(cfg), '--output', str(output),
                                     '--gate', 'overlap', '--context', '262144', '--resident', '32768',
                                     '--cache-mib', '4096', '--prompt-a-cells', '35000',
                                     '--prompt-b-cells', '35004', '--prefill', '1024', '--prefix', '128'])
            self.assertEqual(result, 1)  # intentional missing executable; no model loaded
            saved = json.loads(output.read_text())
            self.assertEqual(saved['failure']['type'], 'FileNotFoundError')
            self.assertEqual(saved['prefill'], 1024)
            self.assertEqual(saved['command'].count('--prefill'), 1)
            self.assertEqual(saved['command'][saved['command'].index('--prefill') + 1], '1024')
            self.assertEqual((saved['context'], saved['resident'], saved['cache_mib']), (262144, 32768, 4096))

    def test_optional_mtp_audit_preserves_placement_and_checks_actual_info(self):
        cfg = {'args': ['--kv', 'int8', '--mtp', '/actual/draft']}
        command, _, env, _ = smoke.settings(cfg, '/tmp/native', 1024, 0, 4096, coherence_mtp_max=4)
        self.assertEqual(command[command.index('--spec') + 1], '4')
        self.assertEqual(command[command.index('--mtp-max-t') + 1], '4')
        self.assertEqual(command[command.index('--adapt-swaps') + 1], '0')
        self.assertEqual(env['STRATA_IQ_MT_MIN'], '1')
        smoke.verify_info({**info(cache=4096), 'mtp_max': '4', 'spec': '4'}, 1024, 0, 4096, 4)
        with self.assertRaises(AssertionError):
            smoke.verify_info(info(cache=4096), 1024, 0, 4096, 4)
        with self.assertRaises(AssertionError):
            smoke.settings({'args': ['--kv', 'int8']}, '/tmp/native', 1024, 0, 4096, coherence_mtp_max=4)
        for extra in (['--coherence-mtp-max', '2'],
                      ['--gate', 'pressure', '--cache-mib', '4096', '--require-pressure-restore',
                       '--coherence-mtp-max', '4', '--temperature', '0.3']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                smoke.main(['--exe', '/unused', '--config', '/unused', '--output', '/unused', *extra])

    def test_native_stub_start_and_clean_quit(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = data()
            engine = smoke.Engine(evidence, Path(tmp) / 'stderr', .3)
            header = 'INFO ' + ' '.join(f'{k}={v}' for k, v in info().items())
            script = f'import sys\nprint({header!r})\nprint("READY 1024 stop")\nfor line in sys.stdin:\n if line.strip()=="QUIT": break\n'
            try:
                engine.start([sys.executable, '-u', '-c', script], os.getcwd(), dict(os.environ), 1024, 0, 0, 1)
            finally:
                engine.close(True)
            smoke.cleanup_check(evidence)

    def test_timeout_kills_python_stub_no_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = smoke.Engine(data(), Path(tmp) / 'stderr', .2)
            try:
                with self.assertRaises(TimeoutError):
                    engine.start([sys.executable, '-u', '-c', 'import time; time.sleep(60)'],
                                 os.getcwd(), dict(os.environ), 1024, 0, 0, .05)
            finally:
                engine.close(False)
            self.assertIsNotNone(engine.p.poll())
            self.assertTrue(engine.evidence['cleanup']['reader_stopped'])

    def test_invalid_capacity_rejected_before_native_launch_and_existing_artifacts_refused(self):
        for args in (['--context', '262144'], ['--context', '1025'], ['--stage-timeout', 'inf'], ['--resident', '128']):
            with self.subTest(args=args), patch.object(smoke.Engine, 'start') as start, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                smoke.main(['--exe', '/unused', '--config', '/unused', '--output', '/unused', *args])
            start.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'old.json'
            output.write_text('KEEP')
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                smoke.main(['--exe', '/unused', '--config', '/unused', '--output', str(output)])
            self.assertEqual(output.read_text(), 'KEEP')

    def test_sample_cli_and_default_greedy_evidence_without_model_run(self):
        for extra, temperature in (([], 0.0), (['--temperature', '0.3', '--seed', '12345'], .3)):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                cfg, output = Path(tmp) / 'cfg.json', Path(tmp) / 'out.json'
                cfg.write_text(json.dumps({'args': ['--kv', 'int8'], 'tokenizer': 'unused'}))
                with patch.object(smoke.common, 'tokenizer', return_value=WordTokenizer()), \
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = smoke.main(['--exe', '/no/such/engine', '--config', str(cfg), '--output', str(output), *extra])
                self.assertEqual(result, 1)
                saved = json.loads(output.read_text())
                self.assertEqual(saved['sampling'], {'temperature': temperature, 'seed': 12345, 'rng_offset_added': False})
                self.assertEqual(saved['pressure_target'], 852)
                self.assertEqual(saved['prefill'], 64)
                self.assertEqual(saved['command'][saved['command'].index('--prefill') + 1], '64')
        for extra in (['--temperature', '-1'], ['--temperature', 'nan'], ['--seed', '0'], ['--seed', '-1']):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                smoke.main(['--exe', '/unused', '--config', '/unused', '--output', '/unused', *extra])

    def test_main_load_failure_keeps_json_without_model_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output = Path(tmp) / 'cfg.json', Path(tmp) / 'out.json'
            cfg.write_text(json.dumps({'args': ['--kv', 'int8'], 'tokenizer': 'unused'}))
            with patch.object(smoke.common, 'tokenizer', return_value=WordTokenizer()), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = smoke.main(['--exe', '/no/such/engine', '--config', str(cfg), '--output', str(output)])
            self.assertEqual(result, 1)
            evidence = json.loads(output.read_text())
            self.assertEqual(evidence['failure']['type'], 'FileNotFoundError')
            self.assertFalse(evidence['passed'])


def production_cfg():
    return {'parallel': 2, 'args': ['--kv-unified', '--vision', '--max-context', '262144',
                                  '--kv-resident', '32768', '--conversation-cache-mib', '4096',
                                  '--conversation-cache-slots', '4', '--vram-reserve-mib', '2048'],
            'vision': {'gpu': True, 'exe': 'vision', 'mmproj': 'model'}}


class HTTPPureTests(unittest.TestCase):
    def test_omitted_caps_not_zero_or_hidden_limit(self):
        body = http.payload('model', 'long prompt')
        http.assert_omitted(body)
        self.assertEqual(body['seed'], smoke.SEED)
        for key in http.LIMIT_KEYS:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                http.assert_omitted({**body, key: 0})

    def test_sse_parser_comments_done_deltas_errors(self):
        self.assertEqual(http.parse_sse_data(b': keep-alive\n')['kind'], 'ignore')
        self.assertEqual(http.parse_sse_data(b'data: [DONE]\n')['kind'], 'done')
        e = http.parse_sse_data('data: {"choices":[{"delta":{"content":"123"},"finish_reason":null}]}\n')
        self.assertEqual((e['kind'], e['text']), ('delta', '123'))
        for line in ('data: {"error":{"message":"pressure leaked"}}', 'data: nope', 'data: []'):
            with self.subTest(line=line), self.assertRaises((AssertionError, ValueError)):
                http.parse_sse_data(line)

    def test_production_settings_cannot_be_shrunk_or_vision_cpu(self):
        cfg = production_cfg()
        http.config_evidence(cfg)
        for key, value in (('--max-context', '32768'), ('--kv-resident', '20480'),
                           ('--conversation-cache-mib', '0'), ('--vram-reserve-mib', '700')):
            broken = production_cfg()
            broken['args'][broken['args'].index(key) + 1] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                http.config_evidence(broken)
        cfg['vision']['gpu'] = False
        with self.assertRaises(AssertionError):
            http.config_evidence(cfg)

    def test_matching_our_two_prompts_required_not_unrelated_clients(self):
        metrics = {'live': {'waiting': 3, 'slots': [{'state': 'decoding', 'generated': 12, 'prompt_tokens': 146001},
                                                  {'state': 'decoding', 'generated': 9, 'prompt_tokens': 22001}]}}
        slots = [{'n_ctx': 262144, 'is_processing': True}] * 2
        self.assertTrue(http.matching_decode_sample(metrics, slots, [146000, 22000]))
        self.assertFalse(http.matching_decode_sample(metrics, slots, [90000, 22000]))
        metrics['live']['slots'][0]['generated'] = 0
        self.assertFalse(http.matching_decode_sample(metrics, slots, [146000, 22000]))
        metrics['live']['slots'][0].update(generated=4, state='reading')
        self.assertFalse(http.matching_decode_sample(metrics, slots, [146000, 22000]))

    def test_dry_run_never_opens_network_or_config(self):
        with patch.object(http.Client, 'open') as opened, redirect_stdout(io.StringIO()):
            result = http.main(['--url', 'http://unused', '--config', '/missing', '--output', '/missing'])
        self.assertEqual(result, 0)
        opened.assert_not_called()

    def test_http_config_failure_writes_fresh_failure_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'out.json'
            with redirect_stdout(io.StringIO()):
                result = http.main(['--run', '--url', 'http://127.0.0.1:1', '--config', '/missing', '--output', str(output)])
            self.assertEqual(result, 1)
            saved = json.loads(output.read_text())
            self.assertFalse(saved['passed'])
            self.assertIn('Traceback', saved['failure']['traceback'])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                http.main(['--run', '--url', 'http://unused', '--config', '/missing', '--output', str(output)])


class HTTPGateTests(unittest.TestCase):
    class FakeClient:
        io_timeout = 1
        def __init__(self, corrupt_fresh=False):
            self.large = []
            self.corrupt_fresh = corrupt_fresh

        def count(self, text):
            return 100 + 6 * text.count('orange apple pear banana.\n')

        def json(self, path, body, deadline):
            if path == '/v1/messages/count_tokens':
                return {'input_tokens': self.count(body['messages'][0]['content'])}
            if path == '/v1/chat/completions':
                word = 'FRESH' if '-FRESH:' in body['messages'][0]['content'] else 'HEALTHY'
                if self.large and self.corrupt_fresh and word == 'FRESH':
                    word = 'HEALTHY'
                return {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': word}}]}
            if path == '/props':
                return {'total_slots': 2, 'kv_unified': True, 'kv_capacity_cells': 262144, 'kv_resident': 32768,
                        'kv_resident_capacity_cells': 32768, 'modalities': {'vision': True}, 'model_alias': 'm'}
            if path == '/slots':
                return [{'n_ctx': 262144, 'is_processing': len(self.large) == 2}] * 2
            if path == '/metrics':
                return {'engine': {**info(262144, 4096), 'images': True, 'conversation_cache_slots': 4, 'model': 'm'},
                        'live': {'waiting': 3, 'slots': [{'state': 'decoding', 'generated': 4,
                                                       'prompt_tokens': self.count(b['messages'][0]['content'])}
                                                      for b in self.large]}}
            raise AssertionError(path)

        def open(self, path, body, deadline):
            http.assert_omitted(body)
            self.large.append(body)
            class Response(io.BytesIO):
                def getheader(self, name):
                    return 'text/event-stream'
            line = b'data: {"choices":[{"delta":{"content":"1"},"finish_reason":null}]}\n\n'
            return object(), Response(line * 4)

        def release(self, conn, response=None):
            if response:
                response.close()

        def shutdown(self):
            pass

    def test_full_large_prompt_gate_omits_limits_proves_overlap_and_two_recoveries(self):
        client = self.FakeClient()
        evidence = {'samples': []}
        http.run(client, production_cfg(), evidence, 3, 2, .001, 1)
        self.assertEqual(len(client.large), 2)
        self.assertTrue(all(not http.LIMIT_KEYS.intersection(body) for body in client.large))
        self.assertTrue(all(140000 < count < 150000 if i == 0 else 21000 < count < 23000
                            for i, count in enumerate(r['actual'] for r in evidence['prompt_sizing'])))
        self.assertTrue(evidence['overlap_proof'])
        self.assertTrue(all(evidence['cleanup']['workers_stopped']))
        self.assertEqual(evidence['healthy_before']['message'], evidence['healthy_after']['message'])
        self.assertEqual(evidence['healthy_fresh_before']['message'], evidence['healthy_fresh']['message'])
        self.assertEqual(evidence['after']['waiting'], 3)  # production need NOT be globally idle

    def test_inherited_healthy_response_in_fresh_request_fails(self):
        client = self.FakeClient(corrupt_fresh=True)
        evidence = {'samples': []}
        with self.assertRaisesRegex(AssertionError, 'fresh post-close request differs'):
            http.run(client, production_cfg(), evidence, 3, 2, .001, 1)
        self.assertTrue(evidence['overlap_proof'])


class HTTPPressureTests(unittest.TestCase):
    def reference(self):
        return {'text': '1\n2\n3\n', 'finish': 'length', 'usage': {'completion_tokens': pressure_http.TARGET, 'prompt_tokens': 402}}

    def record(self):
        return {'text': '', 'chunks': []}

    def line(self, text='', finish=None, usage=None):
        value = {'choices': [{'delta': {'content': text}, 'finish_reason': finish}]}
        if usage is not None:
            value['usage'] = {'completion_tokens': usage, 'prompt_tokens': 402}
        return ('data: ' + json.dumps(value) + '\n').encode()

    def test_finite_sse_concat_final_finish_usage_and_done(self):
        record, ref = self.record(), self.reference()
        self.assertFalse(pressure_http.consume_sse(record, b': keep-alive\n', ref, 0))
        for text in ('1\n', '2\n', '3\n'):
            self.assertFalse(pressure_http.consume_sse(record, self.line(text), ref, 0))
        pressure_http.consume_sse(record, self.line(finish='length', usage=pressure_http.TARGET), ref, 0)
        self.assertTrue(pressure_http.consume_sse(record, b'data: [DONE]\n', ref, 0))
        self.assertEqual(record['text'], ref['text'])
        wrong_prompt_usage = json.loads(self.line(finish='length', usage=pressure_http.TARGET)[5:])
        wrong_prompt_usage['usage']['prompt_tokens'] = 400  # endpoint count is NOT measured actual count
        with self.assertRaisesRegex(AssertionError, 'actual prompt usage'):
            pressure_http.consume_sse(self.record(), ('data: ' + json.dumps(wrong_prompt_usage)).encode(), ref, 0)

    def test_duplicate_missing_segments_and_internal_pressure_finish_fail(self):
        for lines in ((self.line('1\n'), self.line('1\n')),
                      (self.line('1\n'), self.line('3\n')),
                      (self.line(finish='pressure'),),
                      (self.line(finish='length', usage=300),),
                      (b'data: [DONE]\n',)):
            with self.subTest(lines=lines), self.assertRaises(AssertionError):
                record = self.record()
                for line in lines:
                    pressure_http.consume_sse(record, line, self.reference(), 0)

    def test_exact_history_identity_and_actual_pressure_for_both(self):
        rows = [{'time': 12, 'prompt_total': n, 'output_tokens': pressure_http.TARGET,
                 'engine_generated': pressure_http.TARGET, 'finish': 'length', 'pressure_pauses': 2}
                for n in pressure_http.PROMPT_COUNTS]
        proof = pressure_http.history_proof({'requests': rows}, pressure_http.PROMPT_COUNTS, 10, 15, pressure_http.TARGET)
        self.assertEqual(len(proof), 2)
        for corruption in ('missing', 'duplicate', 'no-pressure', 'wrong-count', 'old'):
            broken = [dict(r) for r in rows]
            if corruption == 'missing':
                broken.pop()
            elif corruption == 'duplicate':
                broken.append(dict(broken[0]))
            elif corruption == 'no-pressure':
                broken[0]['pressure_pauses'] = 0
            elif corruption == 'wrong-count':
                broken[0]['engine_generated'] = 350
            else:
                broken[0]['time'] = 1
            with self.subTest(corruption=corruption), self.assertRaises(AssertionError):
                pressure_http.history_proof({'requests': broken}, pressure_http.PROMPT_COUNTS, 10, 15, pressure_http.TARGET)

    def test_small_prompt_sizing_retains_realistic_default_terse_template(self):
        class TemplateCounter:
            def __init__(self):
                self.calls = []
            def json(self, path, body, deadline):
                self.calls.append((path, body))
                kwargs = body.get('chat_template_kwargs') or {}
                # A ~300-token default system/template base breaks the old
                # 160-token fixture. Do not pretend terse:false is forwarded.
                base = 80 if kwargs.get('terse') is False else 300
                text = body['messages'][0]['content']
                count = base + text.count(' alpha')
                if path.endswith('count_tokens'):
                    return {'input_tokens': count}
                cap = body['max_tokens']
                return {'choices': [{'finish_reason': 'length' if cap == pressure_http.TARGET else 'stop',
                                     'message': {'role': 'assistant', 'content': '1\n2\n' if cap > 16 else 'HEALTHY'}}],
                        'usage': {'prompt_tokens': count + 2, 'completion_tokens': cap if cap > 16 else min(2, cap)}}
        client = TemplateCounter()
        for target in pressure_http.PROMPT_COUNTS:
            text, count, samples = pressure_http.sized_prompt(client, 'model', 'fresh', 'A', target, 1)
            self.assertEqual(count, target)
            self.assertTrue(samples)
            self.assertIn('1 through 1000 inclusive', text)
            self.assertNotIn('100000', text)
            self.assertIn('one integer per line', text)
            self.assertIn('program checks the full sequence', text)
            self.assertIn('all 1000 integers are mandatory', text)
            self.assertIn('No omissions, ellipses, ranges, code, markdown', text)
            self.assertIn('documentation, summaries or commentary', text)
            self.assertIn('Inert padding: alpha', text)
            self.assertLessEqual(count + pressure_http.TARGET + smoke.GUARD, pressure_http.CONTEXT)
            calibration = pressure_http.measure_prompt(client, 'model', text, 1)
            self.assertEqual(calibration['usage']['prompt_tokens'], count + 2)
            self.assertEqual(calibration['request_max_tokens'], 1)
            ref = pressure_http.solo(client, 'model', text, pressure_http.TARGET, 1)
            self.assertEqual(ref['usage']['prompt_tokens'], calibration['usage']['prompt_tokens'])
            body = http.payload('model', text, cap=pressure_http.TARGET)
            self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
        http.healthy(client, 'model', 'fresh-THIRD: HEALTHY', 1)
        http.healthy(client, 'model', 'fresh-RECOVERY: HEALTHY', 1)
        for path, body in client.calls:
            self.assertNotIn('terse', body.get('chat_template_kwargs', {}))
            if path.endswith('count_tokens'):
                self.assertEqual(body['thinking'], {'type': 'disabled'})
                self.assertNotIn('chat_template_kwargs', body)
            else:
                self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
        native_plan = smoke.growth_plan([[1] * 400, [2] * 404], [pressure_http.TARGET] * 2, pressure_http.CONTEXT)
        self.assertEqual(sum(native_plan['initial_pages_upper_bound']) * 4, 1316)
        self.assertEqual(sum(native_plan['nominal_pages']) * 4, 4068)
        self.assertEqual(pressure_http.TARGET, 1632)
        with self.assertRaises(AssertionError):
            smoke.growth_plan([[1] * 400, [2] * 404], [1640, 1640], 2048)
        with self.assertRaisesRegex(AssertionError, 'base=300, probe=332, target=160'):
            pressure_http.sized_prompt(client, 'model', 'fresh', 'A', 160, 1)

    def test_first_invalid_solo_full_response_retained_in_failure_artifact(self):
        class EarlyEOSClient:
            def __init__(self, finish, tokens):
                self.closed = False
                self.response = {'id': 'early-eos', 'choices': [{'finish_reason': finish,
                                 'message': {'role': 'assistant', 'content': '1\n2\n...\n1000\n' + 'full evidence\n' * 500}}],
                                 'usage': {'prompt_tokens': 402, 'completion_tokens': tokens}}
            def json(self, path, body, deadline):
                if path == '/props':
                    return {'total_slots': 2, 'kv_unified': True, 'kv_capacity_cells': 2048,
                            'kv_resident': 0, 'model_alias': 'fake'}
                if path.startswith('/metrics'):
                    return {'engine': info(context=2048, cache=1),
                            'live': {'running': 0, 'waiting': 0, 'slots': []}}
                if path == '/slots':
                    return [{'n_ctx': 2048}, {'n_ctx': 2048}]
                counted = 300 + body['messages'][0]['content'].count(' alpha')
                if path.endswith('count_tokens'):
                    return {'input_tokens': counted}
                if body['max_tokens'] == 1:
                    return {'choices': [{'finish_reason': 'length'}],
                            'usage': {'prompt_tokens': counted + 2, 'completion_tokens': 1}}
                return self.response
            def shutdown(self):
                self.closed = True
        for finish, tokens in (('stop', 292), ('stop', pressure_http.TARGET), ('length', 292)):
            with self.subTest(finish=finish, tokens=tokens), tempfile.TemporaryDirectory() as directory:
                client = EarlyEOSClient(finish, tokens)
                artifact = Path(directory) / 'failed-solo.json'
                with patch.object(http, 'Client', return_value=client), redirect_stdout(io.StringIO()):
                    status = pressure_http.main(['--run', '--url', 'http://unused', '--output', str(artifact)])
                evidence = json.loads(artifact.read_text())
                self.assertEqual(status, 1)
                self.assertFalse(evidence['passed'])
                self.assertTrue(client.closed)
                self.assertEqual(len(evidence['solo_references']), 1)
                attempt = evidence['solo_references'][0]
                self.assertEqual(attempt['response'], client.response)
                self.assertEqual(attempt['text'], client.response['choices'][0]['message']['content'])
                self.assertEqual(attempt['finish'], finish)
                self.assertEqual(attempt['usage']['completion_tokens'], tokens)
                self.assertEqual(attempt['request_max_tokens'], pressure_http.TARGET)
                self.assertIn('completion_tokens=' + str(tokens), evidence['failure']['message'])
                self.assertNotIn('streams', evidence)
                self.assertNotIn('pressure_history', evidence)

    def test_actual_prompt_identity_measured_delta_guard_and_exact_native_matching(self):
        actual = [402, 406]
        proof = pressure_http.verify_prompt_identity([400, 404], actual)
        self.assertEqual(proof['deltas'], [2, 2])
        self.assertEqual(proof['actual_guard_sums'], [2042, 2046])
        self.assertTrue(proof['association_requires_exact_calibrated_actual_count'])
        pressure_http.verify_prompt_identity([400, 404], [402, 408])  # final permitted guard cell
        for actual_bad, message in (([402, 409], 'cannot fit finite target'),
                                    ([380, 406], 'delta outside bound'),
                                    ([404, 404], 'counts equal'),
                                    ([402, None], 'invalid actual solo')):
            with self.subTest(actual=actual_bad), self.assertRaisesRegex(AssertionError, message):
                pressure_http.verify_prompt_identity([400, 404], actual_bad)
        metrics = {'live': {'slots': [{'state': 'decoding', 'generated': 1, 'prompt_tokens': n} for n in actual]}}
        slots = [{'is_processing': True}, {'is_processing': True}]
        self.assertTrue(pressure_http.live_overlap_identity(metrics, slots, actual))
        self.assertTrue(http.matching_decode_sample(metrics, slots, [400, 404]))  # production's bounded tolerance
        self.assertFalse(pressure_http.live_overlap_identity(metrics, slots, [400, 404]))
        rows = [{'time': 12, 'prompt_total': n, 'pressure_pauses': 1, 'output_tokens': pressure_http.TARGET,
                 'engine_generated': pressure_http.TARGET, 'finish': 'length'} for n in actual]
        pressure_http.history_proof({'requests': rows}, actual, 10, 15, pressure_http.TARGET)
        with self.assertRaisesRegex(AssertionError, 'identity'):
            pressure_http.history_proof({'requests': rows}, [400, 404], 10, 15, pressure_http.TARGET)

    def test_live_overlap_bootstrap_is_bounded_nonnegative_and_generated_backed(self):
        slots = [{'is_processing': True}, {'is_processing': True}]
        def sample(counts, generated):
            return {'live': {'slots': [{'state': 'decoding', 'prompt_tokens': n, 'generated': g}
                                      for n, g in zip(counts, generated)]}}
        proof = pressure_http.live_overlap_identity(sample([402, 404], [50, 1]), slots, [400, 404])
        self.assertEqual([p['bootstrap_delta'] for p in proof['pairs']], [2, 0])
        self.assertEqual([p['live_admitted_prompt_tokens'] for p in proof['pairs']], [402, 404])
        self.assertFalse(proof['immutable_original_identity'])
        self.assertFalse(proof['prompt_hash_identity'])
        self.assertTrue(pressure_http.live_overlap_identity(sample([404, 402], [1, 50]), slots, [400, 404]))
        self.assertTrue(pressure_http.live_overlap_identity(sample([416, 420], [16, 16]), slots, [400, 404]))
        for counts, generated in (([399, 404], [50, 1]), ([417, 421], [50, 50]),
                                  ([402, 404], [1, 1]), ([416, 420], [15, 16])):
            with self.subTest(counts=counts, generated=generated):
                self.assertIsNone(pressure_http.live_overlap_identity(sample(counts, generated), slots, [400, 404]))
        self.assertIsNone(pressure_http.live_overlap_identity(sample([402, 404], [50, 1]),
                                                            [{'is_processing': True}], [400, 404]))

    def test_scripted_live_bootstrap_submits_queues_third_and_keeps_exact_final_identities(self):
        class ScriptedClient:
            io_timeout = 3
            def __init__(self):
                self.metric_calls = self.healthy_calls = 0
                self.started = {name: threading.Event() for name in ('A', 'B')}
                self.third_started = threading.Event()
                self.release = threading.Event()
                self.history = {}
                self.closed = False
            def json(self, path, body, deadline):
                if path == '/props':
                    return {'total_slots': 2, 'kv_unified': True, 'kv_capacity_cells': 2048,
                            'kv_resident': 0, 'model_alias': 'fake'}
                if path.startswith('/metrics'):
                    self.metric_calls += 1
                    phase = self.metric_calls
                    rows = []
                    live = {'running': 0, 'waiting': 0, 'slots': rows}
                    if phase in (2, 3, 4):
                        assert all(event.wait(2) for event in self.started.values()), 'both stream workers must start'
                        rows.extend({'state': 'decoding', 'prompt_tokens': n, 'generated': g, 'pressure_pauses': 0}
                                    for n, g in zip((402, 404), (50, 1)))
                        live['running'] = 2
                        if phase == 3:
                            assert self.third_started.wait(2), 'bootstrap overlap must submit third'
                            live.update(running=3, waiting=1)
                        if phase == 4:
                            rows[0].update(prompt_tokens=1000, generated=600, pressure_pauses=1)
                            live['pressure_waiting'] = 1
                            self.release.set()  # only AFTER the preceding queue sample was consumed
                    return {'engine': info(context=2048, cache=1), 'live': live,
                            'requests': list(self.history.values())}
                if path == '/slots':
                    return [{'n_ctx': 2048, 'is_processing': self.metric_calls in (2, 3, 4)}] * 2
                counted = 300 + body['messages'][0]['content'].count(' alpha')
                if path.endswith('count_tokens'):
                    return {'input_tokens': counted}
                cap = body['max_tokens']
                if cap == 16:
                    self.healthy_calls += 1
                    if self.healthy_calls == 2:
                        self.third_started.set()
                        assert self.release.wait(2), 'third must remain waiting until queue proof captured'
                    return {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'HEALTHY'}}],
                            'usage': {'prompt_tokens': counted, 'completion_tokens': 2}}
                return {'choices': [{'finish_reason': 'length', 'message': {'role': 'assistant', 'content': '1\n2\n3\n'}}],
                        'usage': {'prompt_tokens': counted, 'completion_tokens': cap}}
            def shutdown(self):
                self.closed = True
                self.release.set()
        def scripted_stream(client, body, record, reference, deadline, origin):
            try:
                client.history[record['name']] = {'time': time.time(), 'prompt_total': reference['usage']['prompt_tokens'],
                                                 'pressure_pauses': 1, 'output_tokens': pressure_http.TARGET,
                                                 'engine_generated': pressure_http.TARGET, 'finish': 'length'}
                client.started[record['name']].set()
                assert client.release.wait(2), 'stream must stay active until third queues'
                for text in ('1\n', '2\n', '3\n'):
                    pressure_http.consume_sse(record, self.line(text), reference, 0)
                terminal = {'choices': [{'delta': {}, 'finish_reason': 'length'}], 'usage': reference['usage']}
                pressure_http.consume_sse(record, ('data: ' + json.dumps(terminal)).encode(), reference, 0)
                pressure_http.consume_sse(record, b'data: [DONE]\n', reference, 0)
            except Exception as exc:
                record['error'] = {'message': str(exc)}
            finally:
                record['worker_done'] = True
        client, evidence = ScriptedClient(), {'samples': []}
        with patch.object(pressure_http, 'stream_worker', side_effect=scripted_stream):
            pressure_http.run(client, evidence, 3, .001, 1)
        self.assertTrue(client.closed)
        self.assertEqual(client.healthy_calls, 3)  # solo reference, queued third, recovery
        self.assertEqual(evidence['count_identity']['actual_prompt_tokens'], [400, 404])
        self.assertEqual([p['bootstrap_delta'] for p in evidence['live_overlap_identity']['pairs']], [2, 0])
        self.assertEqual(evidence['third_submitted_after_overlap']['live']['running'], 2)
        self.assertEqual(evidence['third_queue_proof']['live']['waiting'], 1)
        self.assertFalse(evidence['third_queue_live_overlap_identity']['immutable_original_identity'])
        self.assertEqual([r['prompt_total'] for r in evidence['pressure_history']], [400, 404])
        self.assertEqual([r['usage']['prompt_tokens'] for r in evidence['streams']], [400, 404])
        self.assertTrue(all(r['done'] and not r.get('error') for r in evidence['streams']))
        self.assertEqual(evidence['third']['response']['message'], evidence['third_reference']['message'])
        self.assertEqual(evidence['recovery']['message'], evidence['third_reference']['message'])
        self.assertEqual(evidence['cleanup']['workers_stopped'], [True, True, True])

    def test_dry_run_never_issues_requests(self):
        with patch.object(http.Client, 'open') as opened, redirect_stdout(io.StringIO()):
            self.assertEqual(pressure_http.main(['--url', 'http://unused', '--output', '/unused']), 0)
        opened.assert_not_called()


class HTTPLifecycleTests(unittest.TestCase):
    def server(self, mode):
        stopped = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'  # REAL server detaches conn.sock at getresponse
            def log_message(self, *args):
                pass
            def do_GET(self):
                body = json.dumps({'path': self.path, 'value': 'x' * (70000 if mode == 'json_large' else 2)}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                if mode != 'json_eof':
                    self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                if mode == 'slow_headers':
                    try:
                        self.wfile.write(b'HTTP/1.0 200 OK\r\nX-Slow: ')
                        self.wfile.flush()
                        for _ in range(40):
                            self.wfile.write(b'x')
                            self.wfile.flush()
                            if stopped.wait(.02):
                                break
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                if mode == 'normal':
                    for i in range(4):
                        self.wfile.write(('data: ' + json.dumps({'choices': [{'delta': {'content': str(i)},
                                                                             'finish_reason': None}]}) + '\n\n').encode())
                    self.wfile.flush()
                elif mode == 'eos':
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                stopped.wait(3)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        runner = threading.Thread(target=server.serve_forever, daemon=True)
        runner.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(stopped.set)
        return http.Client(f'http://127.0.0.1:{server.server_port}', .3)

    def test_http_1_0_short_content_length_json_sequential_props_metrics_gets(self):
        client = self.server('json_length')
        for path in ('/props', '/metrics?requests=all', '/slots', '/props', '/metrics'):
            result = client.json(path, None, time.monotonic() + 1)
            self.assertEqual(result, {'path': path, 'value': 'xx'})
            with client.lock:
                self.assertFalse(client.connections)
                self.assertFalse(client.deadline_timers)

    def test_http_1_0_large_content_length_json_reads_multiple_chunks_and_closes(self):
        client = self.server('json_large')
        result = client.json('/metrics', None, time.monotonic() + 1)
        self.assertEqual(result['path'], '/metrics')
        self.assertEqual(len(result['value']), 70000)
        with client.lock:
            self.assertFalse(client.connections)
            self.assertFalse(client.deadline_timers)

    def test_http_1_0_json_without_content_length_reads_eof_and_closes(self):
        client = self.server('json_eof')
        self.assertEqual(client.json('/props', None, time.monotonic() + 1), {'path': '/props', 'value': 'xx'})
        with client.lock:
            self.assertFalse(client.connections)
            self.assertFalse(client.deadline_timers)

    def test_pressure_operator_failure_retains_traceback_after_actual_json_gets(self):
        client = self.server('json_length')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'failure.json'
            with patch.object(http, 'Client', return_value=client), redirect_stdout(io.StringIO()):
                result = pressure_http.main(['--run', '--url', 'http://unused', '--output', str(output)])
            self.assertEqual(result, 1)  # JSON fixture isn't an actual model endpoint
            saved = json.loads(output.read_text())
            self.assertEqual(saved['failure']['type'], 'AssertionError')
            self.assertIn('verify_live', saved['failure']['traceback'])
            self.assertIn('Traceback', saved['failure']['traceback'])
        with client.lock:
            self.assertFalse(client.connections)
            self.assertFalse(client.deadline_timers)

    def test_bounded_prefix_held_open_then_closed_http_1_0(self):
        client = self.server('normal')
        record = {'deltas': []}
        hold = threading.Event()
        worker = threading.Thread(target=http.stream_worker,
                                  args=(client, http.payload('m', 'p'), record, 2, time.monotonic() + 2,
                                        hold, time.monotonic()), daemon=True)
        worker.start()
        deadline = time.monotonic() + 1
        while not record.get('prefix_ready_s') and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(len(record['deltas']), 2)
        self.assertTrue(worker.is_alive())
        hold.set()
        worker.join(1)
        client.shutdown()
        self.assertFalse(worker.is_alive())
        self.assertNotIn('error', record)
        self.assertTrue(record['worker_done'])

    def test_controller_shutdown_wakes_detached_http_1_0_sse_reader(self):
        client = self.server('silent')
        record = {'deltas': []}
        hold = threading.Event()
        worker = threading.Thread(target=http.stream_worker,
                                  args=(client, http.payload('m', 'p'), record, 2, time.monotonic() + 20,
                                        hold, time.monotonic()), daemon=True)
        worker.start()
        deadline = time.monotonic() + 1
        while 'opened_s' not in record and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertIn('opened_s', record)
        with client.lock:
            self.assertTrue(all(conn.sock is None for conn in client.connections))
        client.shutdown()
        worker.join(1)
        self.assertFalse(worker.is_alive(), 'SSE reader hung after detached-socket shutdown')
        self.assertIn('error', record)

    def test_absolute_header_deadline_is_not_reset_by_dripping_bytes(self):
        client = self.server('slow_headers')
        begin = time.monotonic()
        with self.assertRaises((AssertionError, OSError, ValueError)):
            client.json('/stub', {}, begin + .12)
        self.assertLess(time.monotonic() - begin, .7)
        with client.lock:
            self.assertFalse(client.connections)
            self.assertFalse(client.deadline_timers)

    def test_natural_eos_before_prefix_is_failed_not_credit(self):
        client = self.server('eos')
        record = {'deltas': []}
        http.stream_worker(client, http.payload('m', 'p'), record, 2, time.monotonic() + 1,
                           threading.Event(), time.monotonic())
        self.assertIn('early_finish', record)
        self.assertIn('error', record)
        self.assertTrue(record['worker_done'])


if __name__ == '__main__':
    unittest.main()
