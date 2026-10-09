"""Rolling unified-KV frontend contract over a deterministic, gated subprocess (no GPU)."""
import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, CTX_SLACK, MockEngine, Service, StrataEngine
from serve import test_unified_lifecycle


# Draws are a deterministic stand-in for Philox(seed, absolute input position), not an RNG restarted at
# generated=0. The fake rejects incomplete/duplicated replay and any BSTOP addressing released ownership.
FAKE_INCREMENTAL = r'''import hashlib, json, queue, sys, threading, time
from pathlib import Path
args = sys.argv[1:]
root = Path(args[args.index('--control-dir') + 1])
legacy = '--legacy' in args
eos = '--eos' in args
main_only = '--main-only' in args
ungated = '--ungated' in args
zero = '--zero' in args
commands, stop = queue.Queue(), threading.Event()
def reader():
    for line in sys.stdin:
        line = line.strip()
        if line == 'STOP':
            stop.set()
            with open(root / 'commands.jsonl', 'a') as log:
                log.write(json.dumps({'cmd': 'STOP'}) + '\n')
        else:
            commands.put(line)
    commands.put(None)
threading.Thread(target=reader, daemon=True).start()
def emit(line):
    print(line, flush=True)
def log(record):
    with open(root / 'commands.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')
def token(owner, original_len, k, seed):
    if eos and k == 5:
        return 248046
    return 65 + hashlib.sha256(f'{owner}:{original_len - 1 + k}:{seed}'.encode()).digest()[0] % 26
emit('INFO kv_unified=1 kv_capacity_cells=4096' + ('' if main_only else ' batch_slots=2') +
     ('' if legacy else ' kv_incremental=1 kv_reserve_ahead=256'))
emit('READY 262144 stop')
original, returned, admissions, active = {}, {}, {}, {}
def finish(slot, state, reason):
    del active[slot]                         # RELEASE before BDONE, never an owner after pressure
    log({'cmd': 'RELEASE', 'slot': slot, 'owner': state['owner'], 'reason': reason})
    emit(f"BDONE {slot} {state['produced']} {reason} 2")
while True:
    try:
        line = commands.get(timeout=0.005)
    except queue.Empty:
        line = ''
    if line is None or line == 'QUIT':
        break
    if line.startswith('BSTOP '):
        slot = int(line.split()[1])
        log({'cmd': 'BSTOP', 'slot': slot, 'owned': slot in active})
        if slot not in active:
            emit('ERR phantom BSTOP')
        else:
            finish(slot, active[slot], 'cancel')
    elif line.startswith(('GEN ', 'BGEN ')):
        f = line.split()
        slot = int(f[1]) if f[0] == 'BGEN' else None
        cap = int(f[2] if slot is not None else f[1])
        ids = [int(t) for t in f[-1].split(',')]
        keys = dict(t.split('=', 1) for t in f[2 if slot is None else 3:-1])
        owner = ids[0]
        if owner not in original:
            original[owner], returned[owner], admissions[owner] = ids, [], 0
        if ids != original[owner] + returned[owner]:
            emit('ERR replay dropped or duplicated tokens')
            continue
        admissions[owner] += 1
        epoch = admissions[owner]
        seed = int(keys.get('seed', '0'))
        log({'cmd': f[0], 'slot': slot, 'owner': owner, 'ids': ids, 'cap': cap,
             'keys': keys, 'epoch': epoch, 'active': len(active)})
        stop.clear()
        if zero and owner == 10 and epoch == 1:
            emit(f'DONE 0 {len(ids)} 0 0 pressure 0 0 0')
            if slot is not None:
                emit(f'BADM {slot} 0')
            continue
        start = len(returned[owner])
        if slot is None:
            emit(f'PP {len(ids)} {len(ids)} 1 100')
            while not ungated and not (root / f'go-{owner}-{epoch}').exists() and not stop.is_set():
                time.sleep(0.005)
            count = 0
            reason = 'cancel' if stop.is_set() else 'length'
            for k in range(start, start + min(cap, cap if legacy else 2)):
                if stop.is_set():
                    reason = 'cancel'
                    break
                t = token(owner, len(original[owner]), k, seed)
                returned[owner].append(t)
                emit(f'T {t}')
                count += 1
                if t == 248046:
                    reason = 'stop'
                    break
            else:
                if not legacy and count < cap:
                    reason = 'pressure'
            emit(f'DONE {count} {len(ids)} 1 {count} {reason} 0 0 {len(ids) - 1}')
        else:
            t = token(owner, len(original[owner]), start, seed)
            returned[owner].append(t)
            emit(f'T {t}')
            reason = 'stop' if t == 248046 else 'length'
            emit(f'DONE 1 {len(ids)} 1 1 {reason} 0 0 {len(ids) - 1}')
            cont = cap > 1 and reason != 'stop'
            if cont:
                active[slot] = {'owner': owner, 'cap': cap, 'produced': 1, 'seed': seed, 'epoch': epoch}
            emit(f'BADM {slot} {int(cont)}')
    for slot, state in list(active.items()):
        owner, epoch = state['owner'], state['epoch']
        if not ungated and not (root / f'go-{owner}-{epoch}').exists():
            continue
        k = len(returned[owner])
        t = token(owner, len(original[owner]), k, state['seed'])
        returned[owner].append(t)
        state['produced'] += 1
        emit(f'BT {slot} {t}')
        if t == 248046:
            finish(slot, state, 'stop')
        elif state['produced'] >= state['cap']:
            finish(slot, state, 'length')
        elif not legacy and state['produced'] >= 2:
            finish(slot, state, 'pressure')
'''


def draw(owner, plen, k, seed):
    return 65 + hashlib.sha256(f'{owner}:{plen - 1 + k}:{seed}'.encode()).digest()[0] % 26


class IncrementalKV(unittest.TestCase):
    def start(self, *, gated=False, eos=False, main_only=False, legacy=False, force_batch=False, zero=False):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        script = self.root / 'engine.py'
        script.write_text(FAKE_INCREMENTAL)
        args = ['--control-dir', str(self.root)]
        args += ['--batch', '2'] if not main_only else ['--main-only']
        args += [] if gated else ['--ungated']
        args += ['--eos'] if eos else []
        args += ['--legacy'] if legacy else []
        args += ['--zero'] if zero else []
        real = server.subprocess.Popen
        with mock.patch.object(server.subprocess, 'Popen',
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine('strata', args)
        if force_batch:
            self.engine.slot_busy[1] = True  # an occupied peer forces BGEN, without patching protocol readers
        self.workers, self.errors, self.results, self.stats = [], [], {}, {}
        drain = self.engine._drain_control
        self.engine._drain_control = lambda until: drain(until, timeout=2)

    def tearDown(self):
        if hasattr(self, 'engine'):
            for owner in range(10, 15):
                for epoch in range(1, 8):
                    self.release(owner, epoch)
            for worker, cancel in getattr(self, 'workers', []):
                cancel.set()
                worker.join(2)
            self.engine.unload()
        if hasattr(self, 'tmp'):
            self.tmp.cleanup()

    def wait(self, condition, message='scripted engine did not progress'):
        end = time.monotonic() + 4
        while not condition() and time.monotonic() < end:
            time.sleep(0.005)
        self.assertTrue(condition(), message)

    def log(self):
        path = self.root / 'commands.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def admissions(self, owner=None):
        return [r for r in self.log() if r['cmd'] in ('GEN', 'BGEN') and
                (owner is None or r['owner'] == owner)]

    def release(self, owner, epoch=1):
        (self.root / f'go-{owner}-{epoch}').touch()

    def launch(self, owner, cap=6, seed=77):
        cancel = threading.Event()
        def run():
            try:
                self.results[owner] = [t for t in self.engine.generate([owner, 3, 99], cap,
                                      {'temperature': 0.7, 'seed': seed}, cancel) if t is not None]
                self.stats[owner] = dict(self.engine.last)
            except BaseException as exc:
                self.errors.append(exc)
        worker = threading.Thread(target=run, daemon=True)
        self.workers.append((worker, cancel))
        worker.start()
        return worker, cancel

    def assert_replay(self, owner, cap, seed):
        rows = self.admissions(owner)
        for row in rows:
            k = len(row['ids']) - 3
            self.assertEqual(row['ids'], [owner, 3, 99] + [draw(owner, 3, i, seed) for i in range(k)])
            self.assertEqual(row['cap'], cap - k)
            self.assertEqual(row['keys']['seed'], str(seed))
            self.assertNotIn('rng_offset', row['keys'])

    def test_repeated_pressure_replay_exact_stream_budget_length_and_slot_accounting(self):
        self.start(force_batch=True)
        expected = [draw(10, 3, k, 77) for k in range(7)]
        self.assertEqual([t for t in self.engine.generate([10, 3, 99], 7,
                         {'temperature': 0.7, 'seed': 77}, threading.Event()) if t is not None], expected)
        self.assert_replay(10, 7, 77)
        self.assertEqual([r['cap'] for r in self.admissions()], [7, 5, 3, 1])
        self.assertEqual(self.engine.last['finish'], 'length')
        self.assertEqual(self.engine.last['generated'], 7)
        self.assertEqual(self.engine.last['prompt_tokens'], 3)
        self.assertEqual(self.engine.last['pressure_pauses'], 3)
        self.assertEqual(self.engine.slot_held[0], [])  # final cap=1 has no decode window/cache
        self.assertEqual(self.engine.pressure_waits, {})
        self.assertFalse(any(r['cmd'] == 'BSTOP' for r in self.log()))

    def test_repeated_pressure_final_eos_and_cache_last_unfed_token(self):
        self.start(eos=True, force_batch=True)
        out = [t for t in self.engine.generate([10, 3, 99], 100, {'seed': 77},
               threading.Event()) if t is not None]
        self.assertEqual(out, [draw(10, 3, k, 77) for k in range(5)] + [248046])
        self.assert_replay(10, 100, 77)
        self.assertEqual(self.engine.last['finish'], 'stop')
        self.assertEqual(self.engine.last['generated'], 6)
        # Pressure caches were cleared; the terminal slot holds this replay prompt and the admission's token,
        # never its terminal unfed EOS. No duplicate first token from gen0 accounting.
        self.assertEqual(self.engine.slot_held[0], [10, 3, 99] + out[:-1])

    def test_main_done_pressure_with_and_without_batch_support(self):
        for main_only in (False, True):
            with self.subTest(main_only=main_only):
                self.start(main_only=main_only)
                out = [t for t in self.engine.generate([10, 3, 99], 6, {'seed': 77},
                       threading.Event()) if t is not None]
                self.assertEqual(out, [draw(10, 3, k, 77) for k in range(6)])
                self.assert_replay(10, 6, 77)
                self.assertEqual(self.engine.last['finish'], 'length')
                self.assertEqual(self.engine.last['generated'], 6)
                self.assertEqual(self.engine.last['pressure_pauses'], 2)
                self.tearDown()
                del self.engine, self.tmp

    def test_unseeded_sampling_chooses_one_seed_and_greedy_stays_greedy(self):
        for sampling, seeded in (({'temperature': 0.7}, True),
                                 ({'temperature': 0.7, 'seed': 0}, True), ({}, False)):
            with self.subTest(sampling=sampling):
                self.start(force_batch=True)
                list(self.engine.generate([10, 3, 99], 6, sampling, threading.Event()))
                seeds = [r['keys'].get('seed') for r in self.admissions()]
                self.assertEqual(len(set(seeds)), 1)
                self.assertTrue(int(seeds[0]) > 0 if seeded else seeds[0] is None)
                self.tearDown()
                del self.engine, self.tmp

    def test_omitted_caps_overlap_with_original_logical_context(self):
        self.start(gated=True, eos=True, force_batch=True)
        tok = ByteTokenizer()
        svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        # Service checks logical limits, not physical backing or per-slot divisions.
        ids, _, max_new = svc.prepare([{'role': 'user', 'content': 'hi'}], None, {})
        self.assertEqual(max_new, 262144 - CTX_SLACK - len(ids))
        cap = 262144 - CTX_SLACK - 3
        a, _ = self.launch(10, cap)
        self.wait(lambda: len(self.admissions(10)) == 1)
        self.engine.slot_busy[1] = False
        b, _ = self.launch(11, cap)
        self.wait(lambda: len(self.admissions(11)) == 1)
        self.assertEqual(self.admissions(11)[0]['active'], 1)
        self.assertEqual(sum(r is not None for r in self.engine.slot_live), 2)
        for owner in (10, 11):
            for epoch in range(1, 4):
                self.release(owner, epoch)
        a.join(4); b.join(4)
        self.assertFalse(a.is_alive() or b.is_alive())
        self.assertEqual(self.errors, [])
        for owner in (10, 11):
            self.assertEqual(self.results[owner], [draw(owner, 3, k, 77) for k in range(5)] + [248046])
            self.assert_replay(owner, cap, 77)
            self.assertEqual(self.stats[owner]['finish'], 'stop')

    def test_close_before_pressure_stops_owned_slot_until_ack(self):
        self.start(gated=True, force_batch=True)
        gen = self.engine.generate([10, 3, 99], 6, {}, threading.Event())
        self.assertIsInstance(next(gen), int)
        gen.close()  # admission drain discovers BADM=1 then BSTOPs real ownership
        self.wait(lambda: not self.engine.slot_busy[0])
        stops = [r for r in self.log() if r['cmd'] == 'BSTOP']
        self.assertEqual(len(stops), 1)
        self.assertTrue(stops[0]['owned'])
        self.assertIsNone(self.engine._yielded)

    def test_close_after_pressure_has_no_phantom_stop_or_cache(self):
        self.start(force_batch=True)
        gen = self.engine.generate([10, 3, 99], 6, {}, threading.Event())
        self.assertIsInstance(next(gen), int)
        self.assertIsInstance(next(gen), int)
        self.assertIsNone(next(gen))  # consumed BDONE pressure, released, before readmission
        self.assertFalse(self.engine.slot_busy[0])
        self.assertEqual(self.engine.slot_held[0], [])
        self.assertEqual(next(iter(self.engine.pressure_waits.values()))['generated'], 2)
        gen.close()
        self.assertEqual(self.engine.pressure_waits, {})
        self.assertEqual(self.engine.last['finish'], 'cancel')
        self.assertFalse(any(r['cmd'] == 'BSTOP' for r in self.log()))
        self.assertIsNone(self.engine._yielded)
        list(self.engine.generate([11, 3, 99], 2, {}, threading.Event()))
        self.assertTrue(self.engine.lines.empty())

    def test_close_with_already_queued_pressure_ack_does_not_stop_phantom_slot(self):
        self.start(force_batch=True)
        gen = self.engine.generate([10, 3, 99], 6, {}, threading.Event())
        next(gen); next(gen)
        self.wait(lambda: not self.engine.slot_q[0].empty())
        gen.close()  # BDONE is discovered by the close drain, not the normal reader
        self.assertFalse(self.engine.slot_busy[0])
        self.assertEqual(self.engine.slot_held[0], [])
        self.assertFalse(any(r['cmd'] == 'BSTOP' for r in self.log()))

    def test_cancel_while_pressure_waiting_and_third_waiter_ordering(self):
        self.start(gated=True, force_batch=True)
        a, ca = self.launch(10)
        self.wait(lambda: len(self.admissions(10)) == 1)
        self.engine.slot_busy[1] = False
        b, _ = self.launch(11, cap=2)
        self.wait(lambda: len(self.admissions(11)) == 1)
        third, _ = self.launch(12, cap=2)
        self.wait(lambda: self.engine.waiting == 1)
        self.release(10)
        self.wait(lambda: len(self.admissions(12)) == 1)
        self.wait(lambda: bool(self.engine.pressure_waits) and self.engine.waiting == 1)
        self.assertEqual([r['owner'] for r in self.admissions()], [10, 11, 12])
        ca.set()
        a.join(2)
        self.assertFalse(a.is_alive(), 'released owner cancellation blocked on phantom slot ownership')
        self.assertEqual(self.results[10], [draw(10, 3, k, 77) for k in range(2)])
        self.assertEqual(self.stats[10]['finish'], 'cancel')
        self.assertFalse(any(r['cmd'] == 'BSTOP' for r in self.log()))
        self.assertEqual(self.engine.pressure_waits, {})
        self.release(11); self.release(12)
        b.join(4); third.join(4)
        self.assertFalse(b.is_alive() or third.is_alive())
        self.assertEqual(self.errors, [])

    def test_released_owner_rejoins_fifo_before_later_fourth_request(self):
        self.start(gated=True, force_batch=True)
        a, _ = self.launch(10, cap=4)
        self.wait(lambda: len(self.admissions(10)) == 1)
        self.engine.slot_busy[1] = False
        b, _ = self.launch(11, cap=2)
        self.wait(lambda: len(self.admissions(11)) == 1)
        third, _ = self.launch(12, cap=2)
        self.wait(lambda: self.engine.waiting == 1)
        self.release(10)
        self.wait(lambda: len(self.admissions(12)) == 1 and self.engine.waiting == 1)
        fourth, _ = self.launch(13, cap=2)
        self.wait(lambda: self.engine.waiting == 2)
        self.release(11)
        self.wait(lambda: len(self.admissions(10)) == 2)
        self.assertEqual([r['owner'] for r in self.admissions()], [10, 11, 12, 10])
        for owner in (10, 12, 13):
            self.release(owner, 1); self.release(owner, 2)
        for worker in (a, b, third, fourth):
            worker.join(4)
            self.assertFalse(worker.is_alive())
        self.assertEqual(self.errors, [])
        self.assert_replay(10, 4, 77)

    def test_zero_progress_pressure_defers_retry_without_ctl_or_phantom_ownership(self):
        self.start(force_batch=True, zero=True)
        a, cancel = self.launch(10, cap=4)
        self.wait(lambda: bool(self.engine.pressure_waits))
        self.assertFalse(self.engine.slot_busy[0])
        self.assertTrue(self.engine.ctl.acquire(blocking=False))
        self.engine.ctl.release()
        a.join(0.15)
        self.assertTrue(a.is_alive(), 'zero-progress owner did not remain queued')
        self.assertEqual(len(self.admissions(10)), 1, 'zero-output pressure spun repeated admissions')
        # A different request can take ctl and produce native progress, allowing exactly the deferred replay.
        other, _ = self.launch(11, cap=1)
        other.join(4); a.join(4)
        self.assertFalse(other.is_alive() or a.is_alive())
        self.assertEqual(self.errors, [])
        self.assertEqual(self.results[10], [draw(10, 3, k, 77) for k in range(4)])
        self.assert_replay(10, 4, 77)
        self.assertEqual([r['cap'] for r in self.admissions(10)], [4, 4, 2])
        self.assertEqual(self.engine.pressure_waits, {})

    def test_zero_progress_pressure_wait_can_cancel_without_retry_or_stop(self):
        self.start(force_batch=True, zero=True)
        worker, cancel = self.launch(10, cap=4)
        self.wait(lambda: bool(self.engine.pressure_waits))
        cancel.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.errors, [])
        self.assertEqual(self.results[10], [])
        self.assertEqual(self.stats[10]['finish'], 'cancel')
        self.assertEqual(len(self.admissions(10)), 1)
        self.assertFalse(any(r['cmd'] == 'BSTOP' for r in self.log()))
        self.assertEqual(self.engine.pressure_waits, {})

    def test_main_only_pressure_close_clears_metadata_without_stop(self):
        self.start(main_only=True)
        gen = self.engine.generate([10, 3, 99], 6, {'seed': 77}, threading.Event())
        self.assertIsNone(next(gen))  # prefill
        next(gen); next(gen)
        self.assertIsNone(next(gen))  # consumed main DONE pressure
        self.assertEqual(next(iter(self.engine.pressure_waits.values()))['generated'], 2)
        gen.close()
        self.assertEqual(self.engine.pressure_waits, {})
        self.assertEqual(self.engine.last['finish'], 'cancel')
        self.assertFalse(any(r['cmd'] == 'STOP' for r in self.log()))

    def test_service_pressure_progress_has_no_intermediate_completion_or_history(self):
        self.start(force_batch=True)
        tok = ByteTokenizer()
        svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        run = svc.run([10, 3, 99], False, False, 7, {'seed': 77}, threading.Event())
        seen = []
        try:
            while not self.engine.pressure_waits:
                seen.append(next(run))
            metrics = svc.metrics()
            self.assertEqual(metrics['live']['state'], 'queued')
            self.assertEqual(metrics['live']['generated'], 2)
            self.assertIsNone(metrics['live']['tok_s'])
            self.assertEqual(metrics['live']['pressure_waiting'][0]['remaining'], 5)
            self.assertEqual(metrics['live']['pressure_waiting'][0]['pressure_pauses'], 1)
            self.assertEqual(metrics['live']['running'], 1)
            self.assertEqual(metrics['requests'], [])
            self.assertEqual(metrics['totals']['requests'], 0)
            self.assertFalse(any(kind == 'done' for kind, _ in seen))
            seen.extend(run)
        finally:
            run.close()
        done = [value for kind, value in seen if kind == 'done']
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]['finish'], 'length')
        self.assertEqual(done[0]['completion_tokens'], 7)
        history = list(svc.history)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['output_tokens'], 7)
        self.assertEqual(history[0]['engine_generated'], 7)
        self.assertEqual(history[0]['prompt_tokens'], 3)
        self.assertEqual(history[0]['finish'], 'length')
        self.assertEqual(history[0]['pressure_pauses'], 3)
        self.assertEqual(history[0]['reused'], 2, 'replayed output must not inflate original prompt cache hits')
        self.assertEqual(history[0]['prompt_ms'], 4)
        self.assertEqual(history[0]['decode_ms'], 7)
        self.assertEqual(svc.metrics()['live']['pressure_waiting'], [])

    def test_legacy_engine_no_headroom_metadata_and_single_full_command(self):
        self.start(legacy=True, force_batch=True)
        out = [t for t in self.engine.generate([10, 3, 99], 6, {'seed': 77},
               threading.Event()) if t is not None]
        self.assertEqual(out, [draw(10, 3, k, 77) for k in range(6)])
        self.assertEqual(len(self.admissions()), 1)
        self.assertNotIn('kv_incremental', self.engine.info)
        self.assertEqual(self.engine._kv_reservation(1000, 2000), 3000)
        self.assertEqual(self.engine.last['finish'], 'length')


class IncrementalSampling(unittest.TestCase):
    def test_service_reasoning_wrap_uses_same_seed_across_generate_passes(self):
        class Sampled(MockEngine):
            info = {'kv_incremental': 1}
            _continuation_sampling = StrataEngine._continuation_sampling

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                seen.append(dict(sampling))
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        seen = []
        tok = ByteTokenizer()
        engine = Sampled(tok, ['thinking ' * 10, 'answer'])
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        with mock.patch('serve.server.os.urandom', return_value=b'\x01' * 8) as random:
            events = list(svc.run([10, 3, 99], True, False, 400,
                                 {'temperature': 0.7, 'reasoning_budget_tokens': 20}, threading.Event()))
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0]['seed'], seen[1]['seed'])
        self.assertGreater(seen[0]['seed'], 0)
        random.assert_called_once_with(8)
        self.assertEqual(events[-1][1]['finish'], 'stop')


class IncrementalAdmission(unittest.TestCase):
    def scheduler(self, incremental=True):
        engine = test_unified_lifecycle.UnifiedLifecycle.scheduler(self)
        if incremental:
            engine.info.update(kv_incremental=1, kv_reserve_ahead=256)
        return engine

    def test_incremental_headroom_allows_unlimited_preemption_and_paused_owner_accounting(self):
        engine = self.scheduler()
        engine.slot_busy = [True, False]
        engine.wait_lens = [[2, 262134]]
        self.assertTrue(engine._shorter_waiting(2000, 260136, slot=0))
        engine.slot_live[0] = {'prompt_tokens': 2000, 'max_new': 260136, 'paused': True}
        self.assertTrue(engine._admission_room(2, 262134, None))
        self.assertFalse(engine._admission_room(2000, 260136, None))
        self.assertEqual(engine._kv_reservation(17, 8), 28)
        self.assertEqual(engine._kv_reservation(17, 262000), 276)

    def test_legacy_unlimited_reservation_still_serializes_and_missing_headroom_fails_closed(self):
        for info in ({}, {'kv_incremental': 1}, {'kv_incremental': 1, 'kv_reserve_ahead': 0}):
            engine = self.scheduler(incremental=False)
            engine.info.update(info)
            engine.slot_busy = [True, False]
            engine.wait_lens = [[2, 4090]]
            self.assertFalse(engine._shorter_waiting(2000, 2000, slot=0))
            engine.slot_live[0] = {'prompt_tokens': 1000, 'max_new': 64, 'paused': True}
            self.assertFalse(engine._admission_room(2, 4090, None))


if __name__ == '__main__':
    unittest.main()
