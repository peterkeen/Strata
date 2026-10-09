"""Frontend HANDOFF/BHANDOFF protocol over a scripted unified-KV engine (no GPU)."""
import json
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.server import StrataEngine


FAKE_HANDOFF = r'''import json, queue, sys, threading, time
from pathlib import Path
args = sys.argv[1:]
root = Path(args[args.index('--control-dir') + 1])
handoff = '--handoff' in args
handoff_zero = '--handoff-zero' in args
gate_first_gen = '--gate-first-gen' in args
pause_active = '--pause-active' in args
eos_bt = '--eos-bt' in args
handoff_terminal = '--handoff-terminal' in args
delay_handoff_ack = '--delay-handoff-ack' in args
handoff_after_cap = '--handoff-after-cap' in args
handoff_release = '--handoff-release' in args
commands = queue.Queue()
def reader():
    for line in sys.stdin:
        commands.put(line.strip())
    commands.put(None)
threading.Thread(target=reader, daemon=True).start()
def emit(line):
    print(line, flush=True)
def log(record):
    with open(root / 'commands.jsonl', 'a') as f:
        f.write(json.dumps(record) + '\n')
def toks(ids, n):
    base = 1000 + len(ids)
    return [base + i for i in range(n)]
info = 'INFO batch_slots=2 kv_unified=1 kv_capacity_cells=4096 slot_cache=1'
if handoff_zero:
    info += ' kv_handoff=0'
elif handoff:
    info += ' kv_handoff=1'
emit(info)
emit('READY 4096 stop')
main_cache = None
slot_cache = {}
active = {}
gen_count = 0
def warm_for(ids):
    if main_cache == ids:
        return True
    return any(cache == ids for cache in slot_cache.values())
def finish_state(slot, state, reason):
    # A BHANDOFF the engine cannot honour (prompt cache off, or a picture slot) releases the
    # slot and owes the cancel terminal, not the handoff one.
    terminal = 'cancel' if reason == 'handoff_release' else reason
    if terminal == 'cancel':
        slot_cache.pop(slot, None)
    else:
        slot_cache[slot] = list(state['session'])
    if delay_handoff_ack and reason == 'handoff':
        deadline = time.monotonic() + 5
        held = []
        while not (root / 'ack-handoff').exists() and time.monotonic() < deadline:
            try:
                cmd = commands.get(timeout=0.005)
            except queue.Empty:
                continue
            if cmd.startswith('BSTOP '):
                log({'cmd': 'BSTOP', 'slot': int(cmd.split()[1]), 'owned': False})
            else:
                held.append(cmd)
        for h in held:
            commands.put(h)
    log({'cmd': 'BDONE', 'slot': slot, 'finish': terminal, 'session': list(slot_cache.get(slot, []))})
    emit(f"BDONE {slot} {state['produced']} {terminal} 1")
def finish(slot, reason):
    global active
    finish_state(slot, active.pop(slot), reason)
def handle_gen(line):
    global gen_count, main_cache
    f = line.split()
    cap = int(f[1])
    ids = [int(t) for t in f[-1].split(',') if t]
    gen_count += 1
    log({'cmd': 'GEN', 'ids': ids, 'cap': cap, 'warm': warm_for(ids)})
    if gate_first_gen and gen_count == 1:
        deadline = time.monotonic() + 5
        while not (root / 'go-main').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('first GEN was never released')
            time.sleep(0.005)
    session = list(ids)
    produced = 0
    for t in toks(ids, cap):
        session.append(t)
        produced += 1
        main_cache = list(session)
        emit(f'T {t}')
        if gate_first_gen and gen_count == 1 and produced == 1:
            deadline = time.monotonic() + 5
            held = []
            while time.monotonic() < deadline:
                try:
                    cmd = commands.get(timeout=0.005)
                except queue.Empty:
                    continue
                if cmd == 'HANDOFF':
                    log({'cmd': 'HANDOFF'})
                    if handoff_after_cap:
                        for t2 in toks(ids, cap)[produced:cap]:
                            session.append(t2)
                            produced += 1
                            main_cache = list(session)
                            emit(f'T {t2}')
                    emit(f'DONE {produced} {len(ids)} 1 {produced} handoff 0 0 {len(ids)}')
                    for h in held:
                        commands.put(h)
                    return
                if cmd == 'STOP':
                    main_cache = None
                    log({'cmd': 'STOP'})
                    emit(f'DONE {produced} {len(ids)} 1 {produced} cancel 0 0 0')
                    for h in held:
                        commands.put(h)
                    return
                held.append(cmd)
            for h in held:
                commands.put(h)
        time.sleep(0.001)
    emit(f'DONE {produced} {len(ids)} 1 {produced} length 0 0 {len(ids)}')
def handle_bgen(line):
    f = line.split()
    slot, cap = int(f[1]), int(f[2])
    ids = [int(t) for t in f[-1].split(',') if t]
    log({'cmd': 'BGEN', 'slot': slot, 'ids': ids, 'cap': cap, 'warm': warm_for(ids)})
    t = toks(ids, 1)[0]
    session = list(ids) + [t]
    slot_cache[slot] = list(session)
    emit(f'T {t}')
    emit(f'DONE 1 {len(ids)} 1 1 length 0 0 {len(ids)}')
    cont = cap > 1
    if cont:
        active[slot] = {'cap': cap, 'produced': 1, 'session': session, 'paused': False}
    emit(f'BADM {slot} {1 if cont else 0}')
def step_active():
    for slot, state in list(active.items()):
        if state.get('paused'):
            continue
        if eos_bt and state['produced'] == 1:
            t = 248046
        else:
            t = toks(state['session'], 1)[0]
        state['session'].append(t)
        state['produced'] += 1
        slot_cache[slot] = list(state['session'])
        emit(f'BT {slot} {t}')
        if t == 248046:
            finish(slot, 'handoff' if handoff_terminal else 'stop')
        elif state['produced'] >= state['cap']:
            finish(slot, 'handoff' if handoff_terminal else 'length')
        elif pause_active:
            state['paused'] = True
while True:
    try:
        line = commands.get(timeout=0.01 if active else None)
    except queue.Empty:
        step_active()
        continue
    if line is None or line == 'QUIT':
        break
    if line == 'STOP':
        main_cache = None
        log({'cmd': 'STOP'})
        continue
    if line == 'HANDOFF':
        log({'cmd': 'HANDOFF-late'})
        continue
    if line.startswith('BHANDOFF '):
        slot = int(line.split()[1])
        log({'cmd': 'BHANDOFF', 'slot': slot, 'owned': slot in active})
        if slot in active:
            active[slot]['paused'] = False
            finish(slot, 'handoff_release' if handoff_release else 'handoff')
        continue
    if line.startswith('BSTOP '):
        slot = int(line.split()[1])
        log({'cmd': 'BSTOP', 'slot': slot, 'owned': slot in active})
        if slot in active:
            active[slot]['paused'] = False
            finish(slot, 'cancel')
        continue
    if line.startswith('GEN '):
        handle_gen(line)
        continue
    if line.startswith('BGEN '):
        handle_bgen(line)
        continue
    if line.startswith('BYIELD '):
        continue
    log({'cmd': 'ERR', 'line': line})
    emit('ERR unknown ' + line[:20])
'''


class HandoffFrontend(unittest.TestCase):
    def start(self, *, handoff=True, handoff_zero=False, gate_first_gen=False, pause_active=False, eos_bt=False,
              handoff_terminal=False, delay_handoff_ack=False, handoff_after_cap=False, handoff_release=False):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        script = self.root / 'engine.py'
        script.write_text(FAKE_HANDOFF, encoding='utf-8')
        args = ['--batch', '2', '--control-dir', str(self.root)]
        args += ['--handoff'] if handoff else []
        args += ['--handoff-zero'] if handoff_zero else []
        args += ['--gate-first-gen'] if gate_first_gen else []
        args += ['--pause-active'] if pause_active else []
        args += ['--eos-bt'] if eos_bt else []
        args += ['--handoff-terminal'] if handoff_terminal else []
        args += ['--delay-handoff-ack'] if delay_handoff_ack else []
        args += ['--handoff-after-cap'] if handoff_after_cap else []
        args += ['--handoff-release'] if handoff_release else []
        popen = server.subprocess.Popen
        with mock.patch.object(server.subprocess, 'Popen',
                               lambda cmd, **kw: popen([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine('strata', args)

    def tearDown(self):
        if getattr(self, 'engine', None):
            self.engine.unload()
        if getattr(self, 'tmp', None):
            self.tmp.cleanup()

    def log(self):
        path = self.root / 'commands.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def wait_for(self, predicate, message):
        deadline = time.monotonic() + 4
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(predicate(), message)

    def test_slot_to_solo_uses_bhandoff_and_keeps_warm_replay(self):
        self.start(handoff=True, pause_active=True)
        self.engine.slot_busy[1] = True          # force the request into slot 0 first
        gen = self.engine.generate([10, 20], 40, {}, threading.Event())
        try:
            self.assertIsInstance(next(gen), int)  # BGEN admission token
            self.engine.slot_busy[1] = False       # now it is alone and may go back to solo
            self.assertIsInstance(next(gen), int)  # first BT; next resume sends BHANDOFF
            rest = [t for t in gen if t is not None]
            self.assertTrue(rest)
        finally:
            gen.close()
            self.engine.slot_busy[1] = False
        rows = self.log()
        self.assertIn('BHANDOFF', [r['cmd'] for r in rows], rows)
        self.assertNotIn('BSTOP', [r['cmd'] for r in rows], rows)
        solo_replays = [r for r in rows if r['cmd'] == 'GEN']
        self.assertTrue(any(r['warm'] for r in solo_replays), rows)

    def test_engine_declining_to_keep_a_handed_off_slot_advertises_no_prefix(self):
        # The protocol hole: "handoff" promises the backing is still there, so the frontend
        # advertises the slot's held prefix and steers the next turn at it. An engine that
        # released the slot instead ends it with BDONE cancel; nothing may be advertised and
        # the continuation must not be believed warm.
        self.start(handoff=True, pause_active=True, handoff_release=True)
        self.engine.slot_busy[1] = True
        gen = self.engine.generate([10, 20], 40, {}, threading.Event())
        try:
            self.assertIsInstance(next(gen), int)   # BGEN admission token
            self.engine.slot_busy[1] = False
            self.assertIsInstance(next(gen), int)   # first BT; the next resume sends BHANDOFF
            rest = [t for t in gen if t is not None]
            self.assertTrue(rest)
        finally:
            gen.close()
            self.engine.slot_busy[1] = False
        rows = self.log()
        self.assertIn('BHANDOFF', [r['cmd'] for r in rows], rows)
        self.assertNotIn('BSTOP', [r['cmd'] for r in rows], rows)
        self.assertTrue(any(r['cmd'] == 'BDONE' and r['finish'] == 'cancel' for r in rows), rows)
        self.assertEqual(self.engine.slot_held[0], [])
        solo_replays = [r for r in rows if r['cmd'] == 'GEN']
        self.assertTrue(solo_replays, rows)
        self.assertFalse(any(r['warm'] for r in solo_replays), rows)

    def test_missing_capability_falls_back_to_legacy_bstop_cancel(self):
        self.start(handoff=False, pause_active=True)
        self.engine.slot_busy[1] = True
        gen = self.engine.generate([10, 20], 40, {}, threading.Event())
        try:
            next(gen)
            self.engine.slot_busy[1] = False
            next(gen)
            list(gen)
        finally:
            gen.close()
            self.engine.slot_busy[1] = False
        rows = self.log()
        self.assertIn('BSTOP', [r['cmd'] for r in rows], rows)
        self.assertNotIn('BHANDOFF', [r['cmd'] for r in rows], rows)
        self.assertTrue(any(r['cmd'] == 'BDONE' and r['finish'] == 'cancel' for r in rows), rows)
        self.assertTrue(any(r['cmd'] == 'GEN' and not r['warm'] for r in rows[1:]), rows)

    def test_solo_to_batch_promotion_uses_handoff_and_warm_bgen(self):
        self.start(handoff=True, gate_first_gen=True)
        first_out, second_out, errors = [], [], []
        def first():
            try:
                first_out.extend(t for t in self.engine.generate([1, 2], 12, {}, threading.Event()) if t is not None)
            except BaseException as exc:
                errors.append(exc)
        a = threading.Thread(target=first, daemon=True)
        a.start()
        self.wait_for(lambda: any(r['cmd'] == 'GEN' for r in self.log()), 'first GEN was not admitted')
        def second():
            try:
                second_out.extend(t for t in self.engine.generate([9], 1, {}, threading.Event()) if t is not None)
            except BaseException as exc:
                errors.append(exc)
        b = threading.Thread(target=second, daemon=True)
        b.start()
        self.wait_for(lambda: self.engine.waiting >= 1, 'second request did not wait for control')
        (self.root / 'go-main').touch()
        a.join(4); b.join(4)
        self.assertFalse(a.is_alive() or b.is_alive())
        self.assertEqual(errors, [])
        rows = self.log()
        self.assertIn('HANDOFF', [r['cmd'] for r in rows], rows)
        self.assertNotIn('STOP', [r['cmd'] for r in rows], rows)
        self.assertTrue(any(r['cmd'] == 'BGEN' and r['warm'] for r in rows), rows)

    def test_real_unified_cancel_uses_bstop_and_does_not_advertise_held_tokens(self):
        self.start(handoff=True, pause_active=True)
        self.engine.slot_busy[1] = True
        cancel = threading.Event()
        gen = self.engine.generate([10, 20], 40, {}, cancel)
        try:
            next(gen)
            self.engine.slot_busy[1] = False
            next(gen)
            cancel.set()
            list(gen)
        finally:
            gen.close()
            self.engine.slot_busy[1] = False
        rows = self.log()
        self.assertIn('BSTOP', [r['cmd'] for r in rows], rows)
        self.assertNotIn('BHANDOFF', [r['cmd'] for r in rows], rows)
        self.assertEqual(self.engine.slot_held[0], [])
        self.assertEqual(self.engine.last.get('finish'), 'cancel')

    def test_eos_bt_does_not_send_late_bhandoff(self):
        self.start(handoff=True, pause_active=True, eos_bt=True)
        self.engine.slot_busy[1] = True
        out = []
        try:
            gen = self.engine.generate([10, 20], 40, {}, threading.Event())
            out.append(next(gen))
            self.engine.slot_busy[1] = False
            out.extend(t for t in gen if t is not None)
        finally:
            self.engine.slot_busy[1] = False
        self.assertIn(248046, out)
        rows = self.log()
        self.assertNotIn('BHANDOFF', [r['cmd'] for r in rows], rows)
        self.assertEqual(self.engine.last.get('finish'), 'stop')

    def test_kv_handoff_zero_is_not_capable(self):
        self.start(handoff=False, handoff_zero=True, pause_active=True)
        self.engine.slot_busy[1] = True
        gen = self.engine.generate([10, 20], 40, {}, threading.Event())
        try:
            next(gen)
            self.engine.slot_busy[1] = False
            next(gen)
            list(gen)
        finally:
            gen.close()
            self.engine.slot_busy[1] = False
        rows = self.log()
        self.assertIn('BSTOP', [r['cmd'] for r in rows], rows)
        self.assertNotIn('BHANDOFF', [r['cmd'] for r in rows], rows)

    def test_late_cancel_after_bhandoff_ack_race_is_logical_cancel(self):
        self.start(handoff=True, pause_active=True, delay_handoff_ack=True)
        self.engine.slot_busy[1] = True
        cancel = threading.Event()
        out, errors, stats = [], [], {}
        def run():
            try:
                out.extend(t for t in self.engine.generate([10, 20], 40, {}, cancel) if t is not None)
                stats.update(self.engine.last)
            except BaseException as exc:
                errors.append(exc)
        old_get = self.engine.slot_q[0].get
        def short_get(*args, **kwargs):
            if kwargs.get('timeout') == 10.0:
                kwargs = {**kwargs, 'timeout': 0.05}
            return old_get(*args, **kwargs)
        self.engine.slot_q[0].get = short_get
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        self.wait_for(lambda: any(r['cmd'] == 'BGEN' for r in self.log()), 'BGEN was not admitted')
        self.engine.slot_busy[1] = False
        self.wait_for(lambda: any(r['cmd'] == 'BHANDOFF' for r in self.log()), 'BHANDOFF was not sent')
        cancel.set()
        self.wait_for(lambda: any(r['cmd'] == 'BSTOP' and not r.get('owned') for r in self.log()),
                      'late BSTOP against handed-off idle slot was not observed')
        (self.root / 'ack-handoff').touch()
        worker.join(4)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(stats.get('finish'), 'cancel')
        self.assertEqual(self.engine.slot_held[0], [])

    def test_terminal_bdone_handoff_is_reported_as_length_or_stop_not_handoff(self):
        for eos, expected, cap in ((False, 'length', 2), (True, 'stop', 40)):
            with self.subTest(eos=eos):
                self.start(handoff=True, eos_bt=eos, handoff_terminal=True)
                self.engine.slot_busy[1] = True
                try:
                    out = [t for t in self.engine.generate([10, 20], cap, {}, threading.Event()) if t is not None]
                finally:
                    self.engine.slot_busy[1] = False
                self.assertNotEqual(self.engine.last.get('finish'), 'handoff')
                self.assertEqual(self.engine.last.get('finish'), expected)
                if eos:
                    self.assertIn(248046, out)
                self.tearDown()
                del self.engine, self.tmp

    def test_solo_handoff_at_budget_is_reported_as_length_not_handoff(self):
        self.start(handoff=True, gate_first_gen=True, handoff_after_cap=True)
        first_stats, errors = {}, []
        def first():
            try:
                list(self.engine.generate([1, 2], 2, {}, threading.Event()))
                first_stats.update(self.engine.last)
            except BaseException as exc:
                errors.append(exc)
        a = threading.Thread(target=first, daemon=True)
        a.start()
        self.wait_for(lambda: any(r['cmd'] == 'GEN' for r in self.log()), 'first GEN was not admitted')
        def second():
            try:
                list(self.engine.generate([9], 1, {}, threading.Event()))
            except BaseException as exc:
                errors.append(exc)
        b = threading.Thread(target=second, daemon=True)
        b.start()
        self.wait_for(lambda: self.engine.waiting >= 1, 'second request did not wait for control')
        (self.root / 'go-main').touch()
        a.join(4); b.join(4)
        self.assertFalse(a.is_alive() or b.is_alive())
        self.assertEqual(errors, [])
        self.assertIn('HANDOFF', [r['cmd'] for r in self.log()])
        self.assertEqual(first_stats.get('finish'), 'length')


class ControlHandoffRace(unittest.TestCase):
    def control(self):
        engine = object.__new__(StrataEngine)
        engine.info = {'kv_unified': 1, 'kv_handoff': 1}
        engine.lines = queue.Queue()
        for line in ('T 10', 'T 11', 'DONE 2 8 0 0 cancel 0 0 0'):
            engine.lines.put(line)
        engine._ctl_mode, engine._ctl_result = 'solo', None
        engine._send = mock.Mock()
        engine._parse_done = mock.Mock()
        cancel = threading.Event()
        return engine, cancel, engine._control(cancel, lambda t: None, stop_when=lambda: True)

    def test_real_cancel_upgrades_pending_solo_handoff_to_stop(self):
        engine, cancel, control = self.control()
        self.assertFalse(next(control))
        self.assertEqual(engine._send.call_args_list, [mock.call('HANDOFF')])
        cancel.set()
        self.assertFalse(next(control))
        self.assertEqual(engine._send.call_args_list, [mock.call('HANDOFF'), mock.call('STOP')])
        with self.assertRaises(StopIteration):
            next(control)

    def test_promotion_cannot_downgrade_real_solo_stop(self):
        engine, cancel, control = self.control()
        cancel.set()
        list(control)
        self.assertEqual(engine._send.call_args_list, [mock.call('STOP')])


if __name__ == '__main__':
    unittest.main()
