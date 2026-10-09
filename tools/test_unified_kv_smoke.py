#!/usr/bin/env python3
"""CPU-only unified_kv_smoke protocol/config/lifecycle tests. No models or GPUs.

  python3 tools/test_unified_kv_smoke.py
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_kv_smoke as smoke


def event(raw, seq=0):
    return {**smoke.parse_line(raw), 'raw': raw, 'seq': seq, 'wall_s': seq / 100}


def feed(protocol, lines):
    for seq, line in enumerate(lines):
        protocol.consume(event(line, seq))


def info(context=512):
    return {'kv_unified': '1', 'kv_capacity_cells': str(context), 'context': str(context),
            'batch_slots': '2', 'kv_resident': '0', 'conversation_cache_mib': '0',
            'slot_cache': '1', 'lookup': '0', 'mtp_max': '1', 'pcie_frac': '0.00'}


def evidence():
    return {'stdout': [], 'commands': [], 'stages': []}


class SemanticTokenizer:
    """Small deterministic tokenizer that preserves special markers for fixture tests."""
    def __init__(self):
        self.ids = {}
        self.pieces = {}

    def encode(self, text, parse_special=False):
        parts = re.findall(r'<\|[^|]+\|>|\S+', text)
        ids = []
        for part in parts:
            if part not in self.ids:
                token_id = len(self.ids) + 1
                self.ids[part] = token_id
                self.pieces[token_id] = part
            ids.append(self.ids[part])
        return ids


class DivergentSuffixTests(unittest.TestCase):
    """Honest depth recording and coverage marking for the partial-tail branch continuations."""

    def test_tail_suffix_catalog_encodes_and_rejects_unknown_candidates(self):
        tok = SemanticTokenizer()
        self.assertIn(smoke.TAIL_SUFFIX_DEFAULT, smoke.TAIL_SUFFIX_CANDIDATES)
        for name, entry in smoke.TAIL_SUFFIX_CANDIDATES.items():
            with self.subTest(candidate=name):
                ids = smoke.tail_suffix_ids(tok, name)
                self.assertTrue(ids)
                self.assertTrue(entry['text'])
                self.assertTrue(entry['why'])
        self.assertEqual(smoke.tail_suffix_ids(tok), smoke.tail_suffix_ids(tok, smoke.TAIL_SUFFIX_DEFAULT))
        with self.assertRaisesRegex(AssertionError, 'unknown tail-suffix candidate'):
            smoke.tail_suffix_ids(tok, 'no-such-candidate')

    def test_partial_tail_branches_append_distinct_candidates_to_the_same_prefix(self):
        tok = SemanticTokenizer()
        shared = list(range(20))
        offset = len(shared) % smoke.PAGE_CELLS
        branches = smoke.partial_tail_branches(tok, shared, offset, 512)
        self.assertEqual(len(branches), len(smoke.TAIL_SUFFIX_BRANCH_PAIR))
        self.assertEqual(len(set(smoke.TAIL_SUFFIX_BRANCH_PAIR)), len(smoke.TAIL_SUFFIX_BRANCH_PAIR))
        for branch in branches:
            self.assertEqual(branch[:len(shared)], shared, 'shared partial-page prefix changed')
        self.assertNotEqual(branches[0][len(shared):], branches[1][len(shared):])
        with self.assertRaisesRegex(AssertionError, 'shared prefix ending at the requested offset'):
            smoke.partial_tail_branches(tok, shared, (offset + 1) % smoke.PAGE_CELLS, 512)
        with self.assertRaisesRegex(AssertionError, 'output guard exceeds context'):
            smoke.partial_tail_branches(tok, shared, offset, len(shared) + 16 + 7)

    def test_multi_token_coverage_marks_run_only_when_every_branch_grew(self):
        evidence = {}
        status = smoke.apply_multi_token_coverage(evidence, [2, 3])
        self.assertIn('RUN', status)
        self.assertIn('depths [2, 3]', status)
        self.assertIs(evidence['coverage']['multi_token_divergent_suffix_restore'], status)
        for depths in ([1, 5], [1], []):
            with self.subTest(depths=depths):
                evidence = {}
                status = smoke.apply_multi_token_coverage(evidence, depths)
                self.assertIn('UNTESTED', status)
                self.assertIn(f'depths {depths}', status)
                self.assertIn('<= 1 token', status)


class PartialTailSeedTests(unittest.TestCase):
    def test_user_background_padding_preserves_generation_header_and_exact_offsets(self):
        tok = SemanticTokenizer()
        for offset in (1, 2, 3):
            with self.subTest(offset=offset):
                seed = smoke.build_partial_tail_seed(tok, offset, context=512, output_cap=8, guard_cells=8)
                tokens = seed.token_ids
                self.assertEqual(len(tokens), seed.target_cells)
                fixed_cells = (len(seed.prefix_ids) + len(seed.user_close_ids) + len(seed.assistant_header_ids))
                target_floor = max(96, fixed_cells + 1)
                expected_target = target_floor + (offset - 7 - target_floor) % smoke.PAGE_CELLS
                self.assertEqual(seed.target_cells, expected_target)
                self.assertEqual((len(tokens) + 7) % smoke.PAGE_CELLS, offset)
                self.assertEqual(tokens[:seed.background_start], list(seed.prefix_ids))
                expected_background = (list(seed.background_cycle_ids) *
                                       ((seed.background_end - seed.background_start +
                                         len(seed.background_cycle_ids) - 1) //
                                        len(seed.background_cycle_ids)))[:seed.background_end - seed.background_start]
                self.assertEqual(tokens[seed.background_start:seed.background_end], expected_background)
                self.assertEqual(seed.background_end, seed.user_close_start)
                self.assertEqual(tokens[seed.user_close_start:seed.user_close_end], list(seed.user_close_ids))
                self.assertEqual(seed.user_close_end, seed.assistant_start)
                self.assertEqual(tokens[seed.assistant_start:], list(seed.assistant_header_ids))
                prefix_pieces = [tok.pieces[token_id] for token_id in seed.prefix_ids]
                close_pieces = [tok.pieces[token_id] for token_id in seed.user_close_ids]
                assistant_pieces = [tok.pieces[token_id] for token_id in seed.assistant_header_ids]
                self.assertEqual(prefix_pieces[:2], ['<|im_start|>', 'user'])
                self.assertIn('<|im_end|>', close_pieces)
                self.assertLess(close_pieces.index('Do'), close_pieces.index('<|im_end|>'))
                expected_header = tok.encode("\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                             parse_special=True)
                self.assertEqual(list(seed.assistant_header_ids), expected_header)
                self.assertEqual(assistant_pieces[:3], ['<|im_start|>', 'assistant', '<think>'])
                self.assertIn('</think>', assistant_pieces)
                self.assertLessEqual(len(tokens) + 8 + 8, 512)
                branch_len = len(tokens) + 7 + len(tok.encode('Continue with even numbers:'))
                self.assertLessEqual(branch_len + 16 + 8, 512)

    def test_rejects_invalid_offset_or_context_headroom(self):
        tok = SemanticTokenizer()
        with self.assertRaisesRegex(AssertionError, 'offset must be'):
            smoke.build_partial_tail_seed(tok, 0, context=512)
        with self.assertRaisesRegex(AssertionError, 'context guard'):
            smoke.build_partial_tail_seed(tok, 1, context=100)


class ParserTests(unittest.TestCase):
    def test_records(self):
        self.assertEqual(smoke.parse_line('T 81')['token'], 81)
        self.assertEqual(smoke.parse_line('BT 1 99')['slot'], 1)
        self.assertFalse(smoke.parse_line('BADM 0 0')['continues'])
        self.assertEqual(smoke.parse_line('BDONE 1 17 cancel 15.0')['finish'], 'cancel')
        done = smoke.parse_line('DONE 1 65 1.0 2.0 length 0 0 61 0')
        self.assertEqual(done['reused'], 61)
        self.assertEqual(smoke.parse_line('READY 512 stop')['context'], 512)
        self.assertEqual(smoke.parse_line('YIELDED 0 256'), {'kind': 'YIELDED', 'slot': 0, 'tokens': 256})
        self.assertEqual(smoke.parse_line('INFO kv_unified=1 engine=v0.1')['fields']['kv_unified'], '1')
        self.assertEqual(smoke.parse_line('ERR bad request')['message'], 'bad request')
        self.assertEqual(smoke.parse_line('diagnostic message')['kind'], 'OTHER')

    def test_bad_control_records_fail(self):
        for line in ('T', 'T 1 2', 'T x', 'BT 1', 'BT x 2', 'BADM 0 2', 'BDONE 0 1',
                     'DONE 1 4', 'DONE x 4 1 2 length', 'READY x', 'YIELDED', 'YIELDED 0',
                     'YIELDED x 256', 'YIELDED 0 x', 'YIELDED 0 0', 'YIELDED -1 256',
                     'YIELDED 0 -1', 'YIELDED 0 256 extra'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                smoke.parse_line(line)

    def test_required_info(self):
        smoke.verify_info(info(), 512)
        for key in ('kv_unified', 'kv_capacity_cells', 'context', 'batch_slots', 'kv_resident'):
            broken = info()
            broken.pop(key)
            with self.subTest(key=key), self.assertRaises(AssertionError):
                smoke.verify_info(broken, 512)
        with self.assertRaises(AssertionError):
            smoke.verify_info(info(1024), 512)


class ProtocolTests(unittest.TestCase):
    def test_solo_done(self):
        req = smoke.Request('solo', [1, 2], 2)
        protocol = smoke.Protocol([req])
        feed(protocol, ['T 71', 'T 72', 'DONE 2 2 1 1 length'])
        self.assertTrue(protocol.finished)
        smoke.normal(req)
        self.assertEqual(req.tokens, [71, 72])

    def test_interleaved_admission_tokens_and_completions(self):
        a = smoke.Request('a', [1, 2], 3, 0)
        b = smoke.Request('b', [3, 4, 5], 2, 1)
        protocol = smoke.Protocol([a, b])
        feed(protocol, ['T 10', 'DONE 1 2 0 0 length', 'BADM 0 1', 'BT 0 11',
                        'T 20', 'DONE 1 3 0 0 length', 'BT 0 12', 'BDONE 0 3 length 1',
                        'BADM 1 1', 'BT 1 21', 'BDONE 1 2 length 1'])
        self.assertTrue(protocol.finished)
        self.assertEqual(a.tokens, [10, 11, 12])
        self.assertEqual(b.tokens, [20, 21])
        self.assertEqual(a.completion['kind'], 'BDONE')
        self.assertEqual(b.admission_done['kind'], 'DONE')

    def test_badm_zero_completes_without_bdone(self):
        req = smoke.Request('eos', [2], 4, 1)
        protocol = smoke.Protocol([req])
        feed(protocol, ['T 88', 'DONE 1 1 0 0 stop', 'BADM 1 0'])
        self.assertTrue(protocol.finished)
        smoke.normal(req)
        self.assertEqual(req.completion['finish'], 'stop')

    def test_done_does_not_complete_batch_admission(self):
        req = smoke.Request('a', [1], 2, 0)
        protocol = smoke.Protocol([req])
        feed(protocol, ['T 10', 'DONE 1 1 0 0 length'])
        self.assertFalse(protocol.finished)
        self.assertIsNone(req.completion)

    def test_orphans_counts_and_err_rejected(self):
        for lines in (['BT 0 1'], ['BDONE 0 1 length 0'], ['BADM 0 1'], ['ERR wrong'],
                      ['T 1', 'DONE 2 1 0 0 length'], ['T 1', 'DONE 1 99 0 0 length'],
                      ['T 1', 'DONE 1 1 0 0 length', 'BADM 1 1'],
                      ['T 1', 'DONE 1 1 0 0 stop', 'BADM 0 1'],
                      ['T 1', 'DONE 1 1 0 0 length', 'BADM 0 1', 'BT 0 2', 'BDONE 0 1 length 0']):
            with self.subTest(lines=lines), self.assertRaises(AssertionError):
                feed(smoke.Protocol([smoke.Request('a', [1], 2, 0)]), lines)

    def test_parity_includes_first_admission_token(self):
        ref = smoke.Request('ref', [1], 2)
        candidate = smoke.Request('candidate', [1], 2, 0)
        feed(smoke.Protocol([ref]), ['T 10', 'T 11', 'DONE 2 1 0 0 length'])
        feed(smoke.Protocol([candidate]), ['T 9', 'DONE 1 1 0 0 length', 'BADM 0 1',
                                          'BT 0 11', 'BDONE 0 2 length 0'])
        with self.assertRaises(AssertionError):
            smoke.parity(candidate, ref)


class PressureTests(unittest.TestCase):
    def completed(self, cancelled=False):
        # Incremental admission trims optional headroom. Required extents are
        # ceil(321/4) + ceil(256/4) = 81 + 64 = 145 > the pool's 128 pages.
        # Each request still fits alone: 320+128+8 and 256+2+8 <= 512 cells.
        a = smoke.Request('active', [1] * 320, 128, 0)
        b = smoke.Request('waiter', [2] * 256, 2, 1)
        pages = lambda cells: (cells + smoke.PAGE_CELLS - 1) // smoke.PAGE_CELLS
        self.assertGreater(pages(len(a.prompt) + 1) + pages(len(b.prompt)), 512 // smoke.PAGE_CELLS)
        for req in (a, b):
            self.assertLessEqual(len(req.prompt) + req.cap + 8, 512)
        feed(smoke.Protocol([a, b]), ['T 10', 'DONE 1 320 0 0 length', 'BADM 0 1',
                                     'BT 0 11', 'BT 0 12',
                                     f'BDONE 0 3 {"cancel" if cancelled else "stop"} 1',
                                     'T 20', 'DONE 1 256 0 0 length', 'BADM 1 1',
                                     'BT 1 21', 'BDONE 1 2 length 1'])
        return a, b

    def test_output_reservation_pressure_does_not_require_long_actual_output(self):
        a, b = self.completed()
        result = smoke.verify_pressure(a, b, 512, False)
        self.assertEqual(len(a.tokens), 3)  # Natural EOS AFTER observed progress is fine.
        self.assertGreater(sum(result['reserved_pages']), result['pool_pages'])

    def test_cancel_frees_remaining_reservation(self):
        a, b = self.completed(True)
        self.assertTrue(smoke.verify_pressure(a, b, 512, True)['cancelled'])

    def test_early_eos_cannot_count_as_waiting(self):
        a, b = self.completed()
        a.token_events = [e for e in a.token_events if e['kind'] == 'T']
        with self.assertRaisesRegex(AssertionError, 'no active BT'):
            smoke.verify_pressure(a, b, 512, False)

    def test_premature_admission_and_wrong_cancel_fail(self):
        a, b = self.completed()
        b.token_events[0]['seq'] = a.completion['seq'] - 1
        with self.assertRaises(AssertionError):
            smoke.verify_pressure(a, b, 512, False)
        a, b = self.completed()
        with self.assertRaises(AssertionError):
            smoke.verify_pressure(a, b, 512, True)

    def test_required_extent_larger_than_pool_is_rejected(self):
        # A small output cap no longer rules out pressure: incremental admission
        # trims optional headroom, but cannot trim required prompt/decode cells.
        # Deliberately invalid: prompt 513 needs p+1 = 514 cells, or 129 pages
        # > the pool's 128, even alone. Its cap of 2 cannot rescue admission.
        a = smoke.Request('oversized-active', [1] * 513, 2, 0)
        b = smoke.Request('waiter', [2] * 256, 2, 1)
        self.assertGreater(len(a.prompt) + 1, 512)
        with self.assertRaisesRegex(AssertionError, 'request cannot fit alone'):
            smoke.verify_pressure(a, b, 512, False)


class PausedPrefillTests(unittest.TestCase):
    """Script only protocol replies; never starts an engine or loads a model."""
    class ScriptEngine:
        def __init__(self, data, replies):
            self.evidence, self.replies = data, iter(replies)
            self.queue = smoke.deque()
            self.stage = 'startup'
            self.bursts = []

        def send(self, *lines, deadline=None):
            if self.queue:
                raise AssertionError('new command burst before previous replies drained')
            self.bursts.append(lines)
            self.evidence['commands'].extend({'stage': self.stage, 'raw': line} for line in lines)
            self.queue.extend(next(self.replies))

        def next_event(self, deadline):
            if not self.queue:
                raise TimeoutError('script exhausted without stage completion')
            parsed = event(self.queue.popleft(), len(self.evidence['stdout']))
            parsed['stage'] = self.stage
            self.evidence['stdout'].append(parsed)
            if parsed['kind'] == 'ERR':
                raise RuntimeError(parsed['raw'])
            return parsed

    def execute(self, context=512, pause=None, waiting=None, expected_error=None, solo_eos=False):
        source = [1] * (3 * context // 4 - 60)
        ref = smoke.Request('solo-waiter', [2] * (context // 2), 2)
        solo = ['T 7', f'DONE 1 {len(ref.prompt)} 0 0 stop'] if solo_eos else \
            ['T 7', 'T 8', f'DONE 2 {len(ref.prompt)} 0 0 length']
        feed(smoke.Protocol([ref]), solo)
        pause = pause if pause is not None else ['PP 64', 'REUSED 0', 'YIELDED 0 64',
                                                 f'DONE 0 {len(source)} 1 0 cancel 0 0 0']
        waiting = waiting if waiting is not None else ['BDONE 0 0 cancel 0', 'REUSED 0', 'T 7',
                                                       f'DONE 1 {len(ref.prompt)} 0 0 length',
                                                       'BADM 1 1', 'BT 1 8', 'BDONE 1 2 length 0']
        data = evidence()
        engine = self.ScriptEngine(data, [pause, waiting])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'evidence.json'
            suite = smoke.Suite(engine, data, output, 1, context)
            if expected_error:
                with self.assertRaises(expected_error):
                    suite.paused_capacity_wait(source, ref)
            else:
                suite.paused_capacity_wait(source, ref)
            saved = json.loads(output.read_text())
            self.assertEqual(saved, data)  # Failure paths must retain stage evidence too.
        return data['stages'][0], engine

    def test_paused_stop_then_waiter_parity_at_both_contexts(self):
        for context in (512, 1024):
            with self.subTest(context=context):
                stage, engine = self.execute(context)
                self.assertTrue(stage['passed'])
                self.assertEqual(len(engine.bursts), 2)
                self.assertTrue(engine.bursts[0][0].startswith('GEN 96 '))
                self.assertEqual(engine.bursts[0][1], 'BYIELD 0')
                self.assertTrue(engine.bursts[1][0].startswith('BGEN 1 2 '))
                self.assertEqual(engine.bursts[1][1], 'BSTOP 0')
                self.assertEqual(stage['paused_release']['raw'], 'BDONE 0 0 cancel 0')
                pressure = stage['pressure']
                self.assertGreater(pressure['expected_held_reservation_pages'] + pressure['waiter_pages'],
                                   pressure['pool_pages'])
                self.assertLess(pressure['source_done_seq'], pressure['paused_bdone_seq'])
                self.assertLess(pressure['paused_bdone_seq'], pressure['waiting_first_t_seq'])
                self.assertEqual(stage['requests'][0]['tokens'], [])
                self.assertEqual(stage['requests'][1]['tokens'], [7, 8])

    def test_missing_yield_fails_instead_of_skipping(self):
        stage, engine = self.execute(pause=['DONE 0 324 0 0 cancel 0 0 0'], expected_error=AssertionError)
        self.assertFalse(stage['passed'])
        self.assertEqual(len(engine.bursts), 1)

    def test_yield_slot_prefix_and_done_are_required(self):
        for pause in (['YIELDED 1 64', 'DONE 0 324 0 0 cancel 0 0 0'],
                      ['YIELDED 0 324', 'DONE 0 324 0 0 cancel 0 0 0'],
                      ['YIELDED 0 64', 'YIELDED 0 64', 'DONE 0 324 0 0 cancel 0 0 0'],
                      ['YIELDED 0 64', 'T 7', 'DONE 1 324 0 0 stop 0 0 0'],
                      ['YIELDED 0 64', 'DONE 0 324 0 0 cancel 0 0 100']):
            with self.subTest(pause=pause):
                stage, _ = self.execute(pause=pause, expected_error=AssertionError)
                self.assertFalse(stage['passed'])

    def test_capacity_error_when_no_active_rows_is_a_regression(self):
        stage, _ = self.execute(waiting=['ERR unified KV admission: shared capacity exhausted or admission cancelled'],
                                expected_error=RuntimeError)
        self.assertFalse(stage['passed'])
        self.assertEqual(stage['yielded']['tokens'], 64)
        self.assertEqual(stage['requests'][0]['completion']['finish'], 'cancel')

    def test_waiter_before_paused_release_fails(self):
        for waiting in (['T 7', 'BDONE 0 0 cancel 0'],
                        ['DONE 0 256 0 0 cancel', 'BDONE 0 0 cancel 0'],
                        ['BT 0 7', 'BDONE 0 0 cancel 0']):
            with self.subTest(waiting=waiting):
                stage, _ = self.execute(waiting=waiting, expected_error=AssertionError)
                self.assertFalse(stage['passed'])

    def test_paused_release_is_zero_token_cancel_without_decode(self):
        for release in ('BDONE 0 1 cancel 0', 'BDONE 0 0 length 0', 'BDONE 0 0 cancel 1', 'BDONE 1 0 cancel 0'):
            with self.subTest(release=release):
                stage, _ = self.execute(waiting=[release], expected_error=AssertionError)
                self.assertFalse(stage['passed'])

    def test_waiter_must_match_solo_including_early_eos(self):
        for waiting in (['BDONE 0 0 cancel 0', 'T 7', 'DONE 1 256 0 0 stop', 'BADM 1 0'],
                        ['BDONE 0 0 cancel 0', 'T 9', 'DONE 1 256 0 0 length', 'BADM 1 1',
                         'BT 1 8', 'BDONE 1 2 length 0']):
            with self.subTest(waiting=waiting):
                stage, _ = self.execute(waiting=waiting, expected_error=AssertionError)
                self.assertFalse(stage['passed'])

    def test_waiter_eos_at_admission_is_valid_after_paused_release(self):
        for context in (512, 1024):
            with self.subTest(context=context):
                waiting = ['BDONE 0 0 cancel 0', 'T 7', f'DONE 1 {context // 2} 0 0 stop', 'BADM 1 0']
                stage, _ = self.execute(context, waiting=waiting, solo_eos=True)
                self.assertTrue(stage['passed'])
                self.assertFalse(stage['pressure']['waiter_continues'])
                self.assertEqual(stage['requests'][1]['tokens'], [7])
                self.assertEqual(stage['requests'][1]['completion']['finish'], 'stop')
                self.assertLess(stage['paused_release']['seq'], stage['pressure']['waiting_first_t_seq'])
                self.assertLess(stage['paused_release']['seq'], stage['pressure']['waiting_badm_seq'])

    def test_eos_waiter_before_paused_release_still_fails(self):
        stage, _ = self.execute(1024, waiting=['T 7', 'DONE 1 512 0 0 stop', 'BADM 1 0',
                                             'BDONE 0 0 cancel 0'],
                                solo_eos=True, expected_error=AssertionError)
        self.assertFalse(stage['passed'])

    def test_missing_paused_release_cannot_silently_hang(self):
        stage, _ = self.execute(waiting=[], expected_error=TimeoutError)
        self.assertFalse(stage['passed'])

    def test_raw_source_fixture_sizes_and_cold_prefix(self):
        class WordTokenizer:
            def __init__(self):
                self.vocab = {}

            def encode(self, text, parse_special=False):
                return [self.vocab.setdefault(word, len(self.vocab) + 1) for word in text.split()]

        for context, count in ((512, 324), (1024, 708)):
            with self.subTest(context=context):
                _, a, b, waiter, _, source = smoke.fixtures(WordTokenizer(), context)
                self.assertEqual(len(source), count)
                # Native checks b0-q > max(C, short_read), not merely b0>q.
                self.assertGreater(len(source) - 1 - smoke.PREFILL_CELLS, max(smoke.PREFILL_CELLS, 64))
                self.assertLessEqual(len(source) + 96 + 8, context)
                self.assertGreater(len(source) + 96 + len(waiter) + 16, context)
                self.assertTrue(all(source[0] != p[0] for p in (a, b, waiter)))


class ConfigTests(unittest.TestCase):
    def test_real_model_paths_environment_and_single_gpu(self):
        cfg = {'cwd': '/tmp', 'tokenizer': 'model/tokenizer', 'gpu': [2, 3],
               'env': {'MODEL_TUNING': 'retained', 'STRATA_IQ_MT_MIN': '4', 'CUDA_VISIBLE_DEVICES': '8,9'},
               'lib_dirs': ['cuda/lib'], 'args': ['--native', 'model/shard', '--preset', 'model/preset',
                   '--batch', '8', '--kv-resident', '20480', '--max-context', '8192',
                   '--layer-split', 'auto', '--trim-stage-weights', '--batch', '4',
                   '--adapt-swaps', '10', '--expert-profile-save', 'production.profile', '--prefill', '1024']}
        command, cwd, env = smoke.engine_settings(cfg, '/tmp/engine', 512)
        self.assertEqual(command[command.index('--native') + 1], 'model/shard')
        self.assertEqual(env['MODEL_TUNING'], 'retained')
        self.assertEqual(env['STRATA_IQ_MT_MIN'], '1')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')
        self.assertEqual(cwd, '/tmp')
        self.assertTrue(env['LD_LIBRARY_PATH'].startswith('/tmp/cuda/lib'))
        self.assertEqual(smoke.resolve_path(cfg['tokenizer'], cwd), '/tmp/model/tokenizer')
        self.assertEqual(command.count('--batch'), 1)
        self.assertEqual(command[command.index('--batch') + 1], '2')
        self.assertEqual(command[command.index('--kv-resident') + 1], '0')
        self.assertEqual(command.count('--prefill'), 1)
        self.assertEqual(command[command.index('--prefill') + 1], '64')
        self.assertNotIn('--layer-split', command)
        self.assertNotIn('--expert-profile-save', command)
        self.assertIn('--kv-unified', command)
        _, _, selected = smoke.engine_settings(cfg, '/tmp/engine', 1024, '7')
        self.assertEqual(selected['CUDA_VISIBLE_DEVICES'], '7')

    def test_hip_and_inherited_visibility(self):
        cfg = {'args': [], 'backend': 'hip', 'gpu': [2, 3], 'hip_ordinal': 4,
               'env': {'HIP_TUNING': 'on'}}
        _, _, env = smoke.engine_settings(cfg, '/tmp/engine', 512)
        self.assertEqual(env['HIP_VISIBLE_DEVICES'], '4')
        self.assertEqual(env['HIP_TUNING'], 'on')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '5,6'}):
            _, _, env = smoke.engine_settings({'args': []}, '/tmp/engine', 512)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '5')


class LifecycleTests(unittest.TestCase):
    # These launch tiny Python protocol stubs only, NEVER a native/model engine.
    def make_engine(self, tmp):
        return smoke.NativeEngine(evidence(), Path(tmp) / 'stderr.log', cleanup_timeout=0.5)

    def start(self, engine, script, timeout=1):
        engine.start([sys.executable, '-u', '-c', script], os.getcwd(), dict(os.environ), 512, timeout)

    def test_startup_eof_and_log_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            try:
                with self.assertRaisesRegex(RuntimeError, 'EOF'):
                    self.start(engine, 'import sys; print("load failed", file=sys.stderr)')
            finally:
                engine.close(False)
            self.assertIsNotNone(engine.p.poll())
            self.assertTrue(engine.evidence['cleanup']['reader_stopped'])
            self.assertIn('load failed', engine.stderr_path.read_text())

    def test_startup_absolute_timeout_kills_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            try:
                with self.assertRaises(TimeoutError):
                    self.start(engine, 'import time; time.sleep(60)', timeout=0.1)
            finally:
                engine.close(False)
            self.assertIsNotNone(engine.p.poll())
            self.assertTrue(engine.evidence['cleanup']['reader_stopped'])

    def test_missing_unified_info_fails_and_cleanup_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            try:
                with self.assertRaisesRegex(AssertionError, 'required INFO'):
                    self.start(engine, 'import time; print("READY 512 stop"); time.sleep(60)')
            finally:
                engine.close(False)
            self.assertIsNotNone(engine.p.poll())

    def test_raw_reader_and_normal_quit(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            script = ('import sys\n'
                      f'print({"INFO " + " ".join(k + "=" + v for k, v in info().items())!r})\n'
                      'print("READY 512 stop")\n'
                      'for line in sys.stdin:\n'
                      ' if line.strip() == "QUIT": break\n'
                      ' print("T 42")\n')
            try:
                self.start(engine, script)
                engine.stage = 'test'
                engine.send('GEN 1 1')
                self.assertEqual(engine.next_event(time.monotonic() + 1)['token'], 42)
            finally:
                engine.close(True)
            self.assertEqual(engine.evidence['cleanup']['returncode'], 0)
            self.assertTrue(engine.evidence['cleanup']['reader_stopped'])
            self.assertIn('T 42', [e['raw'] for e in engine.evidence['stdout']])

    def test_blocked_stdin_has_deadline_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            script = ('import time\n'
                      f'print({"INFO " + " ".join(k + "=" + v for k, v in info().items())!r})\n'
                      'print("READY 512 stop")\n'
                      'time.sleep(60)\n')
            try:
                self.start(engine, script)
                with self.assertRaisesRegex(TimeoutError, 'stdin write'):
                    engine.send('X' * (2 * 1024 * 1024), deadline=time.monotonic() + 0.05)
            finally:
                engine.close(False)
            self.assertIsNotNone(engine.p.poll())
            self.assertTrue(engine.evidence['cleanup']['writer_stopped'])

    def test_absolute_deadline_is_not_reset_by_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            script = ('import time\n'
                      f'print({"INFO " + " ".join(k + "=" + v for k, v in info().items())!r})\n'
                      'print("READY 512 stop")\n'
                      'while True:\n'
                      ' print("T 42")\n'
                      ' time.sleep(0.01)\n')
            try:
                self.start(engine, script)
                deadline = time.monotonic() + 0.06
                count = 0
                with self.assertRaises(TimeoutError):
                    while True:
                        engine.next_event(deadline)
                        count += 1
                self.assertGreater(count, 0)
            finally:
                engine.close(False)
            self.assertIsNotNone(engine.p.poll())

    def test_unexpected_err_is_never_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            script = ('import time\n'
                      f'print({"INFO " + " ".join(k + "=" + v for k, v in info().items())!r})\n'
                      'print("READY 512 stop")\n'
                      'print("ERR simulated failure")\n'
                      'time.sleep(60)\n')
            try:
                self.start(engine, script)
                with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
                    engine.next_event(time.monotonic() + 1)
            finally:
                engine.close(False)
            self.assertIsNotNone(engine.p.poll())

    @unittest.skipUnless(os.name == 'posix', 'POSIX process-group signals')
    def test_kill_escalation_when_term_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            script = ('import signal, time\n'
                      'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
                      f'print({"INFO " + " ".join(k + "=" + v for k, v in info().items())!r})\n'
                      'print("READY 512 stop")\n'
                      'time.sleep(60)\n')
            try:
                self.start(engine, script)
            finally:
                engine.close(False)
            self.assertTrue(engine.evidence['cleanup']['killed'])
            self.assertIsNotNone(engine.p.poll())

    def test_ignored_quit_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            script = ('import time\n'
                      f'print({"INFO " + " ".join(k + "=" + v for k, v in info().items())!r})\n'
                      'print("READY 512 stop")\n'
                      'time.sleep(60)\n')
            try:
                self.start(engine, script)
            finally:
                engine.close(True)
            self.assertTrue(engine.evidence['cleanup']['quit_timeout'])
            self.assertTrue(engine.evidence['cleanup']['terminated'])
            self.assertIsNotNone(engine.p.poll())

    @unittest.skipUnless(os.name == 'posix', 'POSIX process-group signals')
    def test_exited_wrapper_cannot_leave_stdout_reader_hanging(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            script = ('import subprocess, sys\n'
                      f'print({"INFO " + " ".join(k + "=" + v for k, v in info().items())!r})\n'
                      'print("READY 512 stop")\n'
                      'subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n')
            try:
                self.start(engine, script)
                engine.p.wait(timeout=1)
            finally:
                engine.close(False)
            self.assertTrue(engine.evidence['cleanup']['killed_remaining_group'])
            self.assertTrue(engine.evidence['cleanup']['reader_stopped'])

    def test_launch_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self.make_engine(tmp)
            try:
                with self.assertRaises(FileNotFoundError):
                    engine.start(['/no/such/engine'], os.getcwd(), dict(os.environ), 512, 1)
            finally:
                engine.close(False)
            self.assertTrue(engine.log.closed)

    def test_main_records_model_start_failure_without_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'config.json'
            output = Path(tmp) / 'evidence.json'
            cfg.write_text(json.dumps({'args': [], 'tokenizer': 'unused'}))
            with patch.object(smoke, 'tokenizer', return_value=object()), patch.object(smoke, 'fixtures'), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = smoke.main(['--exe', '/no/such/engine', '--config', str(cfg), '--output', str(output)])
            self.assertEqual(result, 1)
            data = json.loads(output.read_text())
            self.assertFalse(data['passed'])
            self.assertEqual(data['failure']['type'], 'FileNotFoundError')
            self.assertTrue(Path(str(output) + '.stderr.log').exists())


if __name__ == '__main__':
    unittest.main()
