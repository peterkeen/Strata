#!/usr/bin/env python3
"""CPU-only shared-streaming harness tests: protocol stubs, never native models.

  python3 tools/test_unified_kv_streaming_smoke.py
"""
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_kv_streaming_smoke as smoke


def info(context=262144, resident=32768, cache=4096):
    return {'context': str(context), 'kv_unified': '1', 'kv_capacity_cells': str(context),
            'kv_resident': str(resident), 'kv_resident_capacity_cells': str(resident), 'batch_slots': '2',
            'conversation_cache_mib': str(cache), 'conversation_cache_slots': '4', 'slot_cache': '1',
            'lookup': '0', 'mtp_max': '1', 'spec': '2', 'pcie_frac': '0.00', 'kv': 'int8'}


def allocation(context=262144, resident=32768):
    return (f'strata generate: KV streaming: {resident} of {context} cells per QSA layer in VRAM, '
            'the K/V in 3.00 GiB of pinned RAM\n'
            f'strata: unified KV: {context} total backing cells shared by admission and 2 slots, '
            f'{resident} GPU-resident cells per layer (host-backed)\n')


PARK = ('strata serve: conversation cache: parked 192 tokens in 3.5 ms; parked=2 bytes=2048 '
        'evictions=0 snapshot_bytes=1024 reused_kv_bytes=0\n')
RESTORE = 'strata serve: conversation cache: restored 192 tokens (live) in 4.5 ms; parked=1 bytes=1024\n'
COUNTERS = 'strata serve: KV streaming: 97.25% of 123456 block reads hit VRAM, 13.4 MiB read from RAM\n'


def event(raw, seq):
    return {**smoke.common.parse_line(raw), 'raw': raw, 'seq': seq, 'wall_s': seq / 100}


def completed(name, ids, outputs, slot=None, reused=0):
    req = smoke.Request(name, ids, len(outputs), slot)
    protocol = smoke.Protocol([req])
    lines = [f'T {t}' for t in outputs] if slot is None else [f'T {outputs[0]}']
    lines.append(f'DONE {len(outputs) if slot is None else 1} {len(ids)} 0 0 length 0 0 {reused}')
    if slot is not None:
        if len(outputs) > 1:
            lines += [f'BADM {slot} 1'] + [f'BT {slot} {t}' for t in outputs[1:]]
            lines += [f'BDONE {slot} {len(outputs)} length 0']
        else:
            lines += [f'BADM {slot} 0']
    for seq, raw in enumerate(lines):
        protocol.consume(event(raw, seq))
    return req


class WordTokenizer:
    def __init__(self):
        self.vocab = {}

    def encode(self, text, parse_special=False):
        return [self.vocab.setdefault(word, len(self.vocab) + 1) for word in text.split()]


class InfoTests(unittest.TestCase):
    def test_backing_and_gpu_capacity_are_aggregate_not_slot_multiplied(self):
        result = smoke.verify_info(info(), 262144, 32768, 4096, 'int8')
        self.assertEqual(result['aggregate_shared_backing_cells'], 262144)
        self.assertEqual(result['aggregate_shared_gpu_cells_per_layer'], 32768)
        self.assertEqual(result['batch_slots'], 2)
        smoke.verify_allocation(smoke.diagnostics(allocation()), 262144, 32768)

    def test_full_resident_and_wrong_actual_capacity_fail(self):
        for key, bad in (('kv_unified', '0'), ('kv_capacity_cells', '524288'),
                         ('kv_resident_capacity_cells', '65536'), ('kv_resident', '0'),
                         ('batch_slots', '1'), ('conversation_cache_mib', '0')):
            data = info()
            data[key] = bad
            with self.subTest(key=key), self.assertRaises(AssertionError):
                smoke.verify_info(data, 262144, 32768, 4096)
        for key in ('kv_unified', 'kv_capacity_cells', 'kv_resident_capacity_cells', 'kv_resident'):
            data = info()
            data.pop(key)
            with self.subTest(missing=key), self.assertRaises(AssertionError):
                smoke.verify_info(data, 262144, 32768, 4096)

    def test_allocation_requires_host_backing_and_pinned_memory(self):
        for text in ('', allocation().replace(' (host-backed)', ''), allocation() + allocation(),
                     allocation().replace('3.00 GiB', '0.00 GiB'),
                     allocation().replace('262144 total backing', '524288 total backing')):
            with self.subTest(text=text), self.assertRaises(AssertionError):
                smoke.verify_allocation(smoke.diagnostics(text), 262144, 32768)

    def test_smaller_logical_pressure_configuration(self):
        smoke.verify_info(info(24576, 20480), 24576, 20480, 4096)
        smoke.verify_allocation(smoke.diagnostics(allocation(24576, 20480)), 24576, 20480)


class ConfigTests(unittest.TestCase):
    def test_actual_config_model_paths_env_and_streaming_controls(self):
        cfg = {'cwd': '/tmp', 'gpu': 0, 'env': {'MODEL_TUNING': 'retained', 'STRATA_IQ_MT_MIN': 4},
               'lib_dirs': ['cuda/lib'], 'args': ['--native', 'model/shard', '--mtp', 'model/mtp', '--kv', 'int8',
                   '--kv-resident', '0', '--batch', '8', '--kv-resident', '20480', '--max-context', '8192',
                   '--conversation-cache-mib', '0', '--layer-split', 'auto', '--trim-stage-weights',
                   '--expert-profile-save', 'production.profile', '--prefill', 'auto', '--adapt-swaps', '12']}
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '5,6'}):
            command, cwd, env, kv = smoke.settings(cfg, '/tmp/native', 262144, 32768, 4096)
        self.assertEqual(command[command.index('--native') + 1], 'model/shard')
        self.assertEqual(command[command.index('--mtp') + 1], 'model/mtp')
        for key, value in (('--kv-resident', '32768'), ('--max-context', '262144'), ('--batch', '2'),
                           ('--conversation-cache-mib', '4096'), ('--adapt-swaps', '0'), ('--prefill', '512')):
            self.assertEqual(command.count(key), 1)
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn('--kv-unified', command)
        self.assertNotIn('--layer-split', command)
        self.assertNotIn('--expert-profile-save', command)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(env['MODEL_TUNING'], 'retained')
        self.assertEqual(env['STRATA_IQ_MT_MIN'], '1')
        self.assertTrue(env['LD_LIBRARY_PATH'].startswith('/tmp/cuda/lib'))
        self.assertEqual((cwd, kv), ('/tmp', 'int8'))

    def test_vram_capture_headroom_retained_or_explicitly_overridden(self):
        cfg = {'args': ['--kv', 'int8', '--vram-reserve-mib', '700']}
        command, _, _, _ = smoke.settings(cfg, '/tmp/native', 262144, 32768, 4096)
        self.assertEqual(command[command.index('--vram-reserve-mib') + 1], '700')
        cfg['args'] += ['--vram-reserve-mib', '1024']
        command, _, _, _ = smoke.settings(cfg, '/tmp/native', 262144, 32768, 4096, vram_reserve_mib=2048)
        self.assertEqual(command.count('--vram-reserve-mib'), 1)
        self.assertEqual(command[command.index('--vram-reserve-mib') + 1], '2048')
        self.assertEqual(command[command.index('--kv-resident') + 1], '32768')
        self.assertEqual(command[command.index('--max-context') + 1], '262144')
        self.assertEqual(command[command.index('--conversation-cache-mib') + 1], '4096')

    def test_unsupported_config_format_requires_explicit_override(self):
        cfg = {'args': ['--kv', 'k8v4']}
        with self.assertRaisesRegex(AssertionError, 'K8V4'):
            smoke.settings(cfg, '/tmp/native', 262144, 32768, 4096)
        command, _, _, kv = smoke.settings(cfg, '/tmp/native', 262144, 32768, 4096, kv='int8')
        self.assertEqual(command[command.index('--kv') + 1], 'int8')
        self.assertEqual(kv, 'int8')

    def test_q4_alias_and_ram_floor_retained_or_explicitly_selected(self):
        cfg = {'args': ['--kv', 'q4', '--conversation-cache-min-free-mib', '4096']}
        command, _, _, kv = smoke.settings(cfg, '/tmp/native', 24576, 20480, 4096)
        self.assertEqual(kv, 'q4_0')
        self.assertEqual(command[command.index('--conversation-cache-min-free-mib') + 1], '4096')
        command, _, _, _ = smoke.settings(cfg, '/tmp/native', 24576, 20480, 4096, min_free_mib=0)
        self.assertEqual(command.count('--conversation-cache-min-free-mib'), 1)
        self.assertEqual(command[command.index('--conversation-cache-min-free-mib') + 1], '0')

    def test_fixture_reservations_fit_and_prompts_are_unequal(self):
        for context in (24576, 262144):
            _, padded, _, a, b = smoke.make_fixtures(WordTokenizer(), context, 160, 416, 32)
            self.assertEqual((len(a), len(b)), (160, 416))
            self.assertEqual(len(padded('Parking A astronomy', 192)), 192)
        with self.assertRaises(AssertionError):
            smoke.make_fixtures(WordTokenizer(), 24576, 160, 160, 32)
        with self.assertRaises(AssertionError):
            smoke.make_fixtures(WordTokenizer(), 24576, 16000, 16001, 32)

    def test_cli_rejects_nonstreaming_and_below_minimum_before_launch(self):
        for args in (['--resident', '0'], ['--resident', '16384'], ['--context', '32768'],
                     ['--context', '262145'], ['--resident', '32769'], ['--vram-reserve-mib', '-1'],
                     ['--prompt-cells', '20000', '--max-new', '32'],
                     ['--prompt-cells', '20000', '--prompt-a-cells', '20000']):
            with self.subTest(args=args), patch.object(smoke.StreamingEngine, 'start') as start, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                smoke.main(['--exe', '/unused', '--config', '/unused', '--output', '/unused', *args])
            start.assert_not_called()


class LongPromptTests(unittest.TestCase):
    def test_two_long_prompts_exceed_gpu_but_fit_backing_with_cap_headroom(self):
        _, _, _, a, b = smoke.make_fixtures(WordTokenizer(), 262144, 20000, 20004, 256)
        plan = smoke.long_plan(a, b, 256, 512, 262144, 32768)
        self.assertEqual(plan['prompt_cells'], [20000, 20004])
        self.assertEqual(plan['second_prompt_chunks_estimate'], 40)
        self.assertGreater(plan['max_new'], plan['second_prompt_chunks_estimate'])
        self.assertGreater(plan['unique_live_prompt_cells_lower_bound'], 32768)
        self.assertEqual(plan['aggregate_reservation_cells'], 40516)
        self.assertLess(plan['aggregate_reservation_cells'], 262144)
        self.assertFalse(plan['clock_churn_verified'])
        self.assertFalse(plan['native_backing_id_trace_available'])

    def test_short_cap_identical_prefix_or_only_reserved_overage_cannot_pass(self):
        for a, b, cap, context in (([1] * 20000, [2] * 20004, 32, 262144),
                                    ([1] * 20000, [1] * 20000, 256, 262144),
                                    ([1] * 16300, [2] * 16300, 256, 262144),
                                    ([1] * 20000, [2] * 20004, 256, 32768)):
            with self.subTest(cap=cap, context=context), self.assertRaises(AssertionError):
                smoke.long_plan(a, b, cap, 512, context, 32768)

    def pair(self, overlap):
        a, b = smoke.Request('A', [1] * 20000, 3, 0), smoke.Request('B', [2] * 20004, 2, 1)
        lines = ['T 7', 'DONE 1 20000 0 0 length', 'BADM 0 1', 'BT 0 8']
        if not overlap:
            lines += ['BT 0 9', 'BDONE 0 3 length 0']
        lines += ['T 17', 'DONE 1 20004 0 0 length', 'BADM 1 1']
        if overlap:
            lines += ['BT 0 9', 'BDONE 0 3 length 0']
        lines += ['BT 1 18', 'BDONE 1 2 length 0']
        protocol = smoke.Protocol([a, b])
        for seq, raw in enumerate(lines):
            protocol.consume(event(raw, seq))
        refs = [completed('solo A', a.prompt, [7, 8, 9]), completed('solo B', b.prompt, [17, 18])]
        return [a, b], refs

    def test_real_protocol_overlap_not_forecast_required(self):
        pair, refs = self.pair(True)
        proof = smoke.verify_interleaving(pair, refs)
        self.assertTrue(proof['actual_protocol_overlap_confirmed'])
        self.assertLess(proof['second_badm_seq'], proof['first_bdone_seq'])
        self.assertGreater(proof['first_slot_tokens_after_second_admission'], 0)
        pair, refs = self.pair(False)
        with self.assertRaisesRegex(AssertionError, 'actual overlap'):
            smoke.verify_interleaving(pair, refs)

    def test_long_cli_default_cap_and_private_capture_override_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output = Path(tmp) / 'config.json', Path(tmp) / 'evidence.json'
            cfg.write_text(json.dumps({'args': ['--kv', 'int8', '--vram-reserve-mib', '700'], 'tokenizer': 'unused'}))
            # Missing executable deliberately tests preflight/evidence only.
            with patch.object(smoke.common, 'tokenizer', return_value=WordTokenizer()), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = smoke.main(['--exe', '/no/such/native', '--config', str(cfg), '--output', str(output),
                                     '--prompt-cells', '20000', '--vram-reserve-mib', '2048'])
            self.assertEqual(result, 1)
            data = json.loads(output.read_text())
            self.assertEqual(data['failure']['type'], 'FileNotFoundError')
            self.assertEqual(data['workload']['max_new'], 256)
            self.assertEqual(data['workload']['prefill'], 512)
            self.assertEqual(data['workload']['prompt_cells'], [20000, 20004])
            self.assertGreater(data['long_prompt_plan']['unique_live_prompt_cells_lower_bound'], 32768)
            self.assertEqual(data['command'][data['command'].index('--vram-reserve-mib') + 1], '2048')


class DiagnosticTests(unittest.TestCase):
    def test_park_restore_and_slot_copy_are_distinct(self):
        text = allocation() + PARK + RESTORE + COUNTERS + \
            'strata batch: slot 0 gave back 121 tokens of this conversation (all it holds) in 1.0 ms\n'
        records = smoke.diagnostics(text)
        self.assertEqual([r['kind'] for r in records], ['pinned', 'allocation', 'park', 'restore', 'stream_counters', 'slot_copy'])
        self.assertEqual(records[2]['snapshot_bytes'], 1024)
        self.assertEqual(records[3]['source'], 'live')
        self.assertEqual(records[4]['lookups'], 123456)

    def test_parking_requires_positive_matching_canonical_restore(self):
        ref = completed('reference', [1] * 200, [7, 8])
        restored = completed('restored', [1] * 200, [7, 8], reused=192)
        parked_stage = {'diagnostics': smoke.diagnostics(PARK)}
        restored_stage = {'diagnostics': smoke.diagnostics(RESTORE)}
        result = smoke.verify_park_restore(parked_stage, restored_stage, restored, ref, 192, 4096)
        self.assertEqual(result['reused_prefix_cells'], 192)
        for bad in ('', 'strata serve: conversation cache: skip restore\n',
                    RESTORE.replace('192 tokens', '128 tokens'),
                    'strata batch: slot 0 gave back 192 tokens of this conversation (all it holds) in 0.1 ms\n'):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                smoke.verify_park_restore(parked_stage, {'diagnostics': smoke.diagnostics(bad)}, restored, ref, 192, 4096)

    def test_parking_miss_or_wrong_output_cannot_pass(self):
        ref = completed('reference', [1] * 200, [7, 8])
        restored = completed('restored', [1] * 200, [7, 8], reused=192)
        for bad in ('', PARK.replace('parked 192', 'parked 100'), PARK.replace('bytes=2048', 'bytes=0'),
                    PARK.replace('snapshot_bytes=1024', 'snapshot_bytes=0')):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                smoke.verify_park_restore({'diagnostics': smoke.diagnostics(bad)}, {'diagnostics': smoke.diagnostics(RESTORE)},
                                          restored, ref, 192, 4096)
        wrong = completed('wrong', [1] * 200, [9, 8], reused=192)
        with self.assertRaises(AssertionError):
            smoke.verify_park_restore({'diagnostics': smoke.diagnostics(PARK)}, {'diagnostics': smoke.diagnostics(RESTORE)},
                                      wrong, ref, 192, 4096)

    def test_partial_tail_must_clone_real_idle_slot_not_parked_image(self):
        for prefix in (121, 122, 123):
            ref = completed('reference', [1] * 140, [7, 8])
            req = completed('branch', [1] * 140, [7, 8], slot=1, reused=prefix)
            diag = smoke.diagnostics(f'strata batch: slot 0 gave back {prefix} tokens of this conversation '
                                     '(all it holds) in 0.1 ms\n')
            result = smoke.verify_cached_branch({'diagnostics': diag}, req, ref, prefix, 0)
            self.assertEqual(result['partial_page_offset'], prefix % 4)
            with self.assertRaises(AssertionError):
                smoke.verify_cached_branch({'diagnostics': smoke.diagnostics(RESTORE)}, req, ref, prefix, 0)

    def test_misses_are_reported_but_never_promoted_to_clock_churn(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'stderr'
            log.write_text(COUNTERS)
            data = {}
            smoke.finalize_diagnostics(data, log, require_upload=True)
            self.assertTrue(data['streaming_observations']['host_reads_reported'])
            self.assertFalse(data['streaming_observations']['clock_churn_verified'])
            self.assertIn('COW', data['streaming_observations']['limitation'])

    def test_overflow_or_missing_counters_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'stderr'
            for text in ('', COUNTERS.rstrip() + ' - OVERFLOW (too few resident cells)\n'):
                log.write_text(text)
                with self.subTest(text=text), self.assertRaises(AssertionError):
                    smoke.finalize_diagnostics({}, log)
            log.write_text('strata serve: KV streaming: 100.00% of 100 block reads hit VRAM, 0.0 MiB read from RAM\n')
            data = {}
            smoke.finalize_diagnostics(data, log)
            self.assertFalse(data['streaming_observations']['host_reads_reported'])
            with self.assertRaises(AssertionError):
                smoke.finalize_diagnostics({}, log, require_upload=True)


class ScriptEngine:
    def __init__(self, data, log, scripts):
        self.evidence, self.stderr_path, self.scripts = data, log, iter(scripts)
        self.pending = deque()
        self.stage = 'startup'
        log.write_text('')

    def send(self, *lines, deadline=None):
        expected, stdout, stderr = next(self.scripts)
        if tuple(lines) != tuple(expected):
            raise AssertionError(f'wrong command burst: {lines!r} != {expected!r}')
        self.evidence['commands'].extend({'raw': line, 'stage': self.stage} for line in lines)
        self.pending.extend(stdout)
        with self.stderr_path.open('a') as f:
            f.write(stderr)

    def next_event(self, deadline):
        if not self.pending:
            raise TimeoutError('script ran out before protocol completion')
        parsed = event(self.pending.popleft(), len(self.evidence['stdout']))
        self.evidence['stdout'].append(parsed)
        if parsed['kind'] == 'ERR':
            raise RuntimeError(parsed['raw'])
        return parsed


class StageTests(unittest.TestCase):
    def test_stop_acknowledgement_then_healthy_same_slot_readmission(self):
        cancelled = smoke.Request('cancelled', [1] * 160, 512, 0)
        healthy = smoke.Request('healthy', [2] * 416, 2, 0)
        scripts = [([cancelled.command()], ['T 7', 'DONE 1 160 0 0 length', 'BADM 0 1'], ''),
                   (['BSTOP 0'], ['BT 0 8', 'BDONE 0 2 cancel 0'], COUNTERS),
                   ([healthy.command()], ['T 42', 'DONE 1 416 0 0 length', 'BADM 0 1', 'BT 0 43',
                                          'BDONE 0 2 length 0'], COUNTERS)]
        data = {'commands': [], 'stdout': [], 'stages': []}
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'stderr'
            engine = ScriptEngine(data, log, scripts)
            suite = smoke.StreamingSuite(engine, data, Path(tmp) / 'evidence.json', 1, 262144)
            stage = suite.run('cancel', [cancelled], cancel=True)
            self.assertLess(stage['bstop_after_seq'], cancelled.completion['seq'])
            next_stage = suite.run('healthy', [healthy])
            ref = completed('reference', healthy.prompt, [42, 43])
            smoke.parity(healthy, ref)
            self.assertLess(cancelled.completion['seq'], healthy.token_events[0]['seq'])
            self.assertTrue(next_stage['diagnostics'])
            self.assertEqual(json.loads(suite.output.read_text()), data)

    def test_native_error_during_stage_retains_evidence(self):
        req = smoke.Request('request', [1], 2)
        data = {'commands': [], 'stdout': [], 'stages': []}
        with tempfile.TemporaryDirectory() as tmp:
            engine = ScriptEngine(data, Path(tmp) / 'stderr', [([req.command()], ['ERR failure'], 'diagnostic\n')])
            suite = smoke.StreamingSuite(engine, data, Path(tmp) / 'evidence.json', 1, 262144)
            with self.assertRaises(RuntimeError):
                suite.run('failed', [req])
            self.assertFalse(data['stages'][0]['passed'])
            self.assertEqual(data['stdout'][0]['kind'], 'ERR')
            self.assertEqual(json.loads(suite.output.read_text()), data)

    def test_capacity_wait_stop_is_sent_inside_pending_admission(self):
        # Incremental admission trims optional headroom. Required extents are
        # ceil(15361/4) + ceil(12288/4) = 3841 + 3072 = 6913 > 6144 pages.
        # Each fits alone: 15360+6144+8 and 12288+2+8 <= 24576 cells.
        active = smoke.Request('active', [1] * 15360, 6144, 0)
        waiter = smoke.Request('waiter', [2] * 12288, 2, 1)
        pages = lambda cells: (cells + smoke.PAGE - 1) // smoke.PAGE
        self.assertGreater(pages(len(active.prompt) + 1) + pages(len(waiter.prompt)), 24576 // smoke.PAGE)
        for req in (active, waiter):
            self.assertLessEqual(len(req.prompt) + req.cap + 8, 24576)
        scripts = [([active.command(), waiter.command()], ['T 7', 'DONE 1 15360 0 0 length', 'BADM 0 1', 'BT 0 8'], ''),
                   (['BSTOP 0'], ['BT 0 9', 'BDONE 0 3 cancel 0', 'T 42', 'DONE 1 12288 0 0 length',
                                  'BADM 1 1', 'BT 1 43', 'BDONE 1 2 length 0'], COUNTERS)]
        data = {'commands': [], 'stdout': [], 'stages': []}
        with tempfile.TemporaryDirectory() as tmp:
            engine = ScriptEngine(data, Path(tmp) / 'stderr', scripts)
            suite = smoke.StreamingSuite(engine, data, Path(tmp) / 'evidence.json', 1, 24576)
            suite.run('pressure', [active, waiter], cancel=True, capacity_wait=True)
            proof = smoke.common.verify_pressure(active, waiter, 24576, True)
            self.assertGreater(sum(proof['reserved_pages']), proof['pool_pages'])
            self.assertLess(proof['active_bdone_seq'], proof['waiting_first_t_seq'])


class LifecycleTests(unittest.TestCase):
    # All subprocesses here are tiny Python stdout/stderr stubs, never strata.
    def start_stub(self, engine, script, timeout=1):
        engine.start([sys.executable, '-u', '-c', script], os.getcwd(), dict(os.environ),
                     262144, 32768, 4096, 'int8', timeout)

    def script(self, extra='', data=None):
        header = 'INFO ' + ' '.join(k + '=' + v for k, v in (data or info()).items())
        return ('import sys, time\n'
                f'print({allocation()!r}, file=sys.stderr, end="")\n'
                f'print({header!r})\n'
                'print("READY 262144 stop")\n' + extra)

    def test_startup_uses_streaming_info_not_resident_only_assertions(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = {'commands': [], 'stdout': [], 'stages': []}
            engine = smoke.StreamingEngine(data, Path(tmp) / 'stderr', 0.5)
            try:
                self.start_stub(engine, self.script('for line in sys.stdin:\n if line.strip() == "QUIT": break\n'))
            finally:
                engine.close(True)
            self.assertEqual(data['capacity_evidence']['aggregate_shared_gpu_cells_per_layer'], 32768)
            self.assertTrue(data['startup_allocation']['allocation']['host_backed'])
            self.assertEqual(data['cleanup']['returncode'], 0)

    def test_bad_info_or_eof_cleans_up_loaded_process_stub(self):
        for script in ('print("load failed")', self.script('time.sleep(60)\n', {**info(), 'kv_resident_capacity_cells': '65536'})):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                data = {'commands': [], 'stdout': [], 'stages': []}
                engine = smoke.StreamingEngine(data, Path(tmp) / 'stderr', 0.5)
                try:
                    with self.assertRaises((AssertionError, RuntimeError)):
                        self.start_stub(engine, script)
                finally:
                    engine.close(False)
                self.assertIsNotNone(engine.p.poll())
                self.assertTrue(data['cleanup']['reader_stopped'])

    def test_startup_timeout_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = {'commands': [], 'stdout': [], 'stages': []}
            engine = smoke.StreamingEngine(data, Path(tmp) / 'stderr', 0.5)
            try:
                with self.assertRaises(TimeoutError):
                    self.start_stub(engine, 'import time; time.sleep(60)', 0.05)
            finally:
                engine.close(False)
            self.assertIsNotNone(engine.p.poll())

    def test_main_start_failure_evidence_without_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, output = Path(tmp) / 'config.json', Path(tmp) / 'evidence.json'
            cfg.write_text(json.dumps({'args': ['--kv', 'int8'], 'tokenizer': 'unused'}))
            with patch.object(smoke.common, 'tokenizer', return_value=WordTokenizer()), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = smoke.main(['--exe', '/no/such/native', '--config', str(cfg), '--output', str(output)])
            self.assertEqual(result, 1)
            data = json.loads(output.read_text())
            self.assertFalse(data['passed'])
            self.assertEqual(data['failure']['type'], 'FileNotFoundError')
            self.assertEqual(data['command'][data['command'].index('--kv-resident') + 1], '32768')
            self.assertTrue(Path(str(output) + '.stderr.log').exists())


if __name__ == '__main__':
    unittest.main()
