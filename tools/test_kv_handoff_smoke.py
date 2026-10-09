#!/usr/bin/env python3
"""CPU-only regressions for kv_handoff_smoke.py.

No model, GPU, SSH, deployment or native Strata process is launched.  The fake
engine scripts stdout protocol records only.
"""
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv_handoff_smoke as smoke


def event(raw, seq=0):
    return {**smoke.common.parse_line(raw), 'seq': seq, 'raw': raw, 'wall_s': seq / 100}


def info(context=1024):
    base = {'kv_incremental': '1', 'kv_reserve_ahead': '256', 'kv_unified': '1',
            'context': str(context), 'kv_capacity_cells': str(context), 'batch_slots': '2',
            'kv_resident': '0', 'conversation_cache_mib': '0', 'slot_cache': '1',
            'lookup': '0', 'mtp_max': '1', 'spec': '2', 'pcie_frac': '0.00',
            'kv_handoff': '1'}
    return base


class TinyTokenizer:
    def __init__(self):
        self.vocab = {}

    def encode(self, text, parse_special=False):
        return [self.vocab.setdefault(w, len(self.vocab) + 100) for w in text.split()]


def padded_factory():
    vocab = {}
    def padded(tag, cells):
        first = vocab.setdefault(tag, len(vocab) + 10)
        return [first] + [first + 1000] * (cells - 1)
    return padded


class HandoffStub:
    def __init__(self, evidence, stderr_path, duplicate_late=False, replay=False):
        self.evidence = evidence
        self.stderr_path = Path(stderr_path)
        self.stderr_path.write_text('')
        self.stage = 'stub'
        self.queue = deque()
        self.emitted = {'main': 0}
        self.prompts = {}
        self.caps = {}
        self.active = set()
        self.terminal = set()
        self.duplicate_late = duplicate_late
        self.replay = replay
        self.cached = {}  # valid consumed prefixes, by main/idle slot owner

    def original_len(self, ids):
        if ids and ids[0] == 1:
            return 160
        if ids and ids[0] == 2:
            return 164
        return 160

    def outputs(self, ids, cap):
        produced = max(0, len(ids) - self.original_len(ids))
        return [ids[0] * 10000 + produced + i for i in range(cap)]

    def reused(self, ids):
        if self.replay:
            return 0
        return max((len(pre) for pre in self.cached.values()
                    if pre and len(pre) < len(ids) and ids[:len(pre)] == pre), default=0)

    def add(self, raw, owner=None):
        self.queue.append((raw, owner))

    def schedule_one(self, slot, ids, cap):
        owner = 'main' if slot is None else slot
        self.prompts[owner] = ids
        self.caps[owner] = cap
        self.emitted[owner] = 0
        outs = self.outputs(ids, cap)
        r = self.reused(ids)
        self.cached.pop('main', None)  # the admission replaces main's old branch
        if slot is None:
            for t in outs:
                self.add(f'T {t}', 'main')
            self.add(f'DONE {cap} {len(ids)} 0 0 length 0 0 {r} 0', 'main')
        else:
            self.add(f'T {outs[0]}', slot)
            self.add(f'DONE 1 {len(ids)} 0 0 length 0 0 {r} 0', slot)
            self.add(f'BADM {slot} 1', slot)
            for t in outs[1:]:
                self.add(f'BT {slot} {t}', slot)
            self.add(f'BDONE {slot} {cap} length 0', slot)

    def schedule_pair(self, gens):
        parsed = []
        for slot, ids, cap in gens:
            self.prompts[slot] = ids
            self.caps[slot] = cap
            self.emitted[slot] = 0
            parsed.append((slot, ids, cap, self.outputs(ids, cap), self.reused(ids)))
            self.cached.pop('main', None)
        for slot, ids, _, outs, r in parsed:
            self.add(f'T {outs[0]}', slot)
            self.add(f'DONE 1 {len(ids)} 0 0 length 0 0 {r} 0', slot)
            self.add(f'BADM {slot} 1', slot)
        for i in range(1, max(cap for _, _, cap, _, _ in parsed)):
            for slot, _, cap, outs, _ in parsed:
                if i < cap:
                    self.add(f'BT {slot} {outs[i]}', slot)
        for slot, _, cap, _, _ in parsed:
            self.add(f'BDONE {slot} {cap} length 0', slot)

    def clear_owner(self, owner):
        self.queue = deque((raw, meta) for raw, meta in self.queue if meta != owner)

    def send(self, *lines, deadline=None):
        self.evidence['commands'].extend({'raw': line, 'stage': self.stage} for line in lines)
        gens = []
        for line in lines:
            parts = line.split()
            if parts[0] == 'HANDOFF':
                self.clear_owner('main')
                n = self.emitted.get('main', 0)
                ids = self.prompts['main']
                self.add(f'DONE {n} {len(ids)} 0 0 handoff 0 0 0 0', 'main-terminal')
            elif parts[0] == 'STOP':
                self.clear_owner('main')
                n = self.emitted.get('main', 0)
                ids = self.prompts['main']
                self.add(f'DONE {n} {len(ids)} 0 0 cancel 0 0 0 0', 'main-terminal')
            elif parts[0] == 'BHANDOFF':
                slot = int(parts[1])
                if slot in self.active:
                    self.clear_owner(slot)
                    self.add(f'BDONE {slot} {self.emitted.get(slot, 0)} handoff 0', slot)
                elif self.duplicate_late:
                    self.add(f'BDONE {slot} 0 handoff 0', slot)
            elif parts[0] == 'BSTOP':
                slot = int(parts[1])
                self.clear_owner(slot)
                self.add(f'BDONE {slot} {self.emitted.get(slot, 0)} cancel 0', slot)
            elif parts[0] == 'GEN':
                cap = int(parts[1]); ids = list(map(int, parts[-1].split(',')))
                self.schedule_one(None, ids, cap)
            elif parts[0] == 'BGEN':
                slot = int(parts[1]); cap = int(parts[2]); ids = list(map(int, parts[-1].split(',')))
                gens.append((slot, ids, cap))
        if len(gens) == 1:
            self.schedule_one(*gens[0])
        elif len(gens) == 2:
            self.schedule_pair(gens)
        elif len(gens) > 2:
            raise AssertionError('too many fake BGENs')

    def next_event(self, deadline):
        if not self.queue:
            raise TimeoutError('fake protocol queue exhausted')
        raw, owner = self.queue.popleft()
        parsed = event(raw, len(self.evidence['stdout']))
        parsed['stage'] = self.stage
        self.evidence['stdout'].append(parsed)
        if parsed['kind'] == 'T':
            self.emitted[owner] = self.emitted.get(owner, 0) + 1
        elif parsed['kind'] == 'BT':
            self.emitted[parsed['slot']] = self.emitted.get(parsed['slot'], 0) + 1
        elif parsed['kind'] == 'BADM' and parsed['continues']:
            self.active.add(parsed['slot'])
        elif parsed['kind'] == 'BDONE':
            self.active.discard(parsed['slot'])
            self.terminal.add(parsed['slot'])
        if parsed['kind'] in ('T', 'BT'):
            cache_owner = owner if parsed['kind'] == 'T' else parsed['slot']
            ids = self.prompts[cache_owner]
            count = self.emitted[cache_owner]
            self.cached[cache_owner] = ids + self.outputs(ids, count)[:-1]
        if parsed['kind'] in ('DONE', 'BDONE') and parsed.get('finish') == 'cancel':
            self.cached.pop('main' if parsed['kind'] == 'DONE' else parsed['slot'], None)
        return parsed


class InfoSettingsTests(unittest.TestCase):
    def test_info_requires_handoff_and_disabled_parking(self):
        smoke.verify_info(info(), 1024, 0, 0)
        broken = info(); broken['kv_handoff'] = '0'
        with self.assertRaises(AssertionError):
            smoke.verify_info(broken, 1024, 0, 0)
        with self.assertRaises(AssertionError):
            smoke.verify_info({**info(), 'conversation_cache_mib': '4'}, 1024, 0, 4)

    def test_settings_preserves_incremental_freeze_and_refuses_cache(self):
        cfg = {'args': ['--kv', 'int8', '--conversation-cache-mib', '4096', '--slots', '8'],
               'env': {'KEEP': '1'}}
        command, _, env, _ = smoke.settings(cfg, '/tmp/native', 1024, 0, 0)
        self.assertEqual(env['KEEP'], '1')
        self.assertEqual(command[command.index('--conversation-cache-mib') + 1], '0')
        self.assertNotIn('--slots', command)
        with self.assertRaises(AssertionError):
            smoke.settings(cfg, '/tmp/native', 1024, 0, 1)


class SuiteTests(unittest.TestCase):
    def run_suite(self, replay=False, duplicate_late=False, gate='minimal'):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        evidence = {'stdout': [], 'commands': [], 'stages': [], 'cache_mib': 0, 'expect_replay': True}
        engine = HandoffStub(evidence, Path(tmp.name) / 'stderr.log', duplicate_late=duplicate_late, replay=replay)
        suite = smoke.Suite(engine, evidence, Path(tmp.name) / 'out.json', 2, 1024, prefix=3)
        padded = padded_factory()
        tok = TinyTokenizer()
        smoke.run_handoff_suite(suite, tok, padded, [1] * 160, [2] * 164,
                                cycles=2, segment_cap=12, resume_tokens=2,
                                append_tokens=2, ref_new=96, gate=gate)
        return evidence, suite

    def test_repeated_two_direction_handoff_continuation_and_completed_cache(self):
        evidence, suite = self.run_suite()
        self.assertTrue(all(s['passed'] for s in evidence['stages']))
        self.assertIn('cycle-0-BHANDOFF-order-01', [s['name'] for s in evidence['stages']])
        self.assertIn('cycle-1-BHANDOFF-order-10', [s['name'] for s in evidence['stages']])
        warm = [s for s in evidence['stages'] if s['name'].startswith('post-cycles-warm-')]
        self.assertEqual(len(warm), 2)
        self.assertTrue(all(s['reuse']['no_full_replay'] and s['reference_stage'] for s in warm))
        pair_stages = [s for s in evidence['stages'] if 'BHANDOFF-order' in s['name']]
        self.assertTrue(all(all(r['required_reused_cells'] > 0 and r['no_full_replay']
                                for r in s['entry_reuse']) for s in pair_stages))
        for stage in pair_stages:
            self.assertEqual({r['name'].split('-')[3]: r['slot'] for r in stage['requests']}, {'A': 0, 'B': 1})
        self.assertFalse([s for s in evidence['stages'] if 'completed-cache-warm-branch' in s['name']])
        self.assertFalse([s for s in evidence['stages'] if s['name'].startswith('real-STOP')])
        # Active handoffs are acknowledged exactly.
        evidence['cleanup'] = {'returncode': 0, 'reader_stopped': True, 'writer_stopped': True,
                               'stdout_start_seq': len(evidence['stdout'])}
        smoke.cleanup_check(evidence)
        self.assertEqual(evidence['handoff_ack_counts']['HANDOFF_commands'], 1)
        self.assertEqual(evidence['handoff_ack_counts']['BHANDOFF_commands'], 4)
        self.assertEqual(json.loads(suite.output.read_text())['stages'], evidence['stages'])

    def test_full_replay_reuse_failure_is_saved(self):
        with self.assertRaisesRegex(AssertionError, 'full replay|short reuse'):
            self.run_suite(replay=True)

    def test_late_bhandoff_extra_terminal_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'orphan batch event|extra terminal'):
            self.run_suite(duplicate_late=True, gate='full')

    def test_real_stop_and_bstop_are_cancellation_not_handoff(self):
        evidence, _ = self.run_suite(gate='full')
        stop_stage = next(s for s in evidence['stages'] if s['name'] == 'real-STOP-main-cancel')
        bstop_stage = next(s for s in evidence['stages'] if s['name'] == 'real-BSTOP-batch-cancel')
        self.assertEqual(stop_stage['requests'][0]['completion']['finish'], 'cancel')
        self.assertEqual(bstop_stage['requests'][0]['completion']['finish'], 'cancel')
        self.assertTrue(next(s for s in evidence['stages'] if s['name'] == 'healthy-batch-after-BSTOP')['passed'])


class MainLifecycleTests(unittest.TestCase):
    def test_missing_native_writes_failure_without_model_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output = Path(tmp) / 'cfg.json', Path(tmp) / 'out.json'
            cfg.write_text(json.dumps({'args': ['--kv', 'int8'], 'tokenizer': 'unused'}))
            with patch.object(smoke.common, 'tokenizer', return_value=TinyTokenizer()), \
                    patch.object(smoke.inc, 'fixtures', return_value=(padded_factory(), [1] * 160, [2] * 164)), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = smoke.main(['--exe', '/no/such/engine', '--config', str(cfg), '--output', str(output),
                                 '--cycles', '2', '--prefix', '3', '--segment-cap', '12', '--reference-new', '96'])
            self.assertEqual(rc, 1)
            saved = json.loads(output.read_text())
            self.assertFalse(saved['passed'])
            self.assertEqual(saved['failure']['type'], 'FileNotFoundError')
            self.assertEqual(saved['cache_mib'], 0)
            self.assertEqual(saved['gate'], 'minimal')

    def test_cli_rejects_nonzero_cache_before_launch(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            smoke.main(['--exe', '/unused', '--config', '/unused', '--output', '/unused', '--cache-mib', '1'])


if __name__ == '__main__':
    unittest.main()
