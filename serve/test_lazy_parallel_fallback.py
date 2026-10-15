"""Failed lazy batch activation degrades to serialized GEN without dropping the handoff stream."""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve import server
from serve.test_parallel import FAKE_BATCH


# Reuse the scripted parallel protocol, substituting only the native L5 refusal before admission.
FAKE_LAZY = FAKE_BATCH.replace(
    '        if slot is not None and (b"BUSYALL" in pbx',
    '''        if slot is not None and "--lazy-refuse" in args:
            if "--refuse-once" not in args or "lazy" not in fired:
                fired.add("lazy")
                if log:
                    log.write(f"REFUSED {slot} {time.monotonic()}\\n"); log.flush()
                print("ERR BGEN: lazy parallel activation unavailable", flush=True)
                continue
        if slot is not None and (b"BUSYALL" in pbx''')


class LazyParallelFallback(unittest.TestCase):
    def start(self, once=False):
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / 'engine.py'
        script.write_text(FAKE_LAZY)
        self.log = Path(self.tmp.name) / 'commands.log'
        args = ['--batch', '2', '--slotcache', '--lazy-refuse', '--log', str(self.log)]
        if once:
            args += ['--refuse-once']
        real = server.subprocess.Popen
        with mock.patch.object(server.subprocess, 'Popen',
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = server.StrataEngine('strata', args)
        self.engine.info['kv_unified'] = 1
        self.engine.info['kv_incremental'] = 1
        self.engine.info['kv_reserve_ahead'] = 256
        self.engine.info['kv_capacity_cells'] = 4096
        self.engine.SOLO_AGAIN_MIN_LEFT = 1000
        self.results, self.errors = {}, []

    def tearDown(self):
        if hasattr(self, 'engine'):
            self.engine.unload()
        if hasattr(self, 'tmp'):
            self.tmp.cleanup()

    def rows(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def launch(self, owner, long=False):
        cancel = threading.Event()
        # No template needed: direct engine calls preserve the exact token stream across replay.
        ids = list(f'{owner} {"LONGREPLY" if long else "short"}'.encode())
        def run():
            try:
                self.results[owner] = ([t for t in self.engine.generate(ids, 100, {'seed': 314}, cancel)
                                        if t is not None], dict(self.engine.last))
            except Exception as exc:
                self.errors.append(exc)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        return worker, cancel

    def test_failed_activation_serializes_both_requests_then_retries_after_cooldown(self):
        self.start()
        sent = []
        original_send = self.engine._send
        def record(line):
            sent.append(line)
            return original_send(line)
        self.engine._send = record
        with mock.patch.object(server, 'LAZY_PARALLEL_RETRY_S', 0.8):
            a, _ = self.launch('AAAA', long=True)
            time.sleep(0.15)
            b, _ = self.launch('BBBB')
            a.join(15); b.join(15)
            self.assertFalse(a.is_alive() or b.is_alive())
            self.assertEqual(self.errors, [])
            long_reply = b'ok, ' + b'la ' * 15 + b'done.'
            self.assertEqual(self.results['AAAA'][0], list(long_reply) + [257])
            self.assertEqual(self.results['BBBB'][0], list(b'ok, done.') + [257])
            for owner in ('AAAA', 'BBBB'):
                self.assertEqual(self.results[owner][1]['generated'], len(self.results[owner][0]))
            rows = self.rows()
            self.assertEqual(sum(r.startswith('REFUSED') for r in rows), 1, rows)
            cmds = [r for r in rows if r.startswith(('GEN ', 'BGEN '))]
            self.assertEqual([r.split()[0] for r in cmds], ['GEN', 'GEN', 'GEN'], rows)
            # The continued A owns the main session first; B cannot start until A's GEN ends.
            self.assertEqual([r.split()[-1] for r in cmds], ['A', 'A', 'B'], rows)
            self.assertEqual(self.engine.slot_claims, set())
            self.assertFalse(any(self.engine.slot_busy))
            self.assertTrue(all('seed=314' in cmd for cmd in sent if cmd.startswith(('GEN ', 'BGEN '))), sent)
            # The next overlap (after the timer) makes a new attempt, rather than permanently disabling slots.
            time.sleep(0.85)
            c, _ = self.launch('CCCC', long=True)
            time.sleep(0.15)
            d, _ = self.launch('DDDD')
            c.join(15); d.join(15)
            self.assertFalse(c.is_alive() or d.is_alive())
            self.assertEqual(self.errors, [])
            self.assertGreaterEqual(sum(r.startswith('REFUSED') for r in self.rows()), 2, self.rows())

    def test_cancelled_solo_waiter_does_not_enter_a_slot(self):
        self.start()
        with mock.patch.object(server, 'LAZY_PARALLEL_RETRY_S', 5.0):
            a, _ = self.launch('AAAA', long=True)
            time.sleep(0.15)
            b, _ = self.launch('BBBB')
            deadline = time.monotonic() + 4
            while not self.engine.lazy_parallel_retry_at and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertGreater(self.engine.lazy_parallel_retry_at, time.monotonic())
            c, cancel = self.launch('CCCC')
            cancel.set()
            c.join(5); a.join(15); b.join(15)
            self.assertFalse(c.is_alive() or a.is_alive() or b.is_alive())
            self.assertEqual(self.errors, [])
            self.assertEqual(self.results['CCCC'][0], [])
            self.assertEqual(sum(r.startswith('REFUSED') for r in self.rows()), 1, self.rows())
            self.assertEqual(self.engine.slot_claims, set())

    def test_one_refusal_then_successful_next_overlap_uses_slots(self):
        self.start(once=True)
        with mock.patch.object(server, 'LAZY_PARALLEL_RETRY_S', 0.4):
            a, _ = self.launch('AAAA', long=True)
            time.sleep(0.15)
            b, _ = self.launch('BBBB')
            a.join(15); b.join(15)
            self.assertEqual(self.errors, [])
            self.assertEqual(sum(r.startswith('REFUSED') for r in self.rows()), 1)
            time.sleep(0.45)
            c, _ = self.launch('CCCC', long=True)
            time.sleep(0.15)
            d, _ = self.launch('DDDD')
            c.join(15); d.join(15)
            self.assertFalse(c.is_alive() or d.is_alive())
            self.assertEqual(self.errors, [])
            self.assertIn('BGEN', [r.split()[0] for r in self.rows()])
            self.assertEqual(self.results['CCCC'][0], list(b'ok, ' + b'la ' * 15 + b'done.') + [257])
            self.assertEqual(self.results['DDDD'][0], list(b'ok, done.') + [257])
            self.assertFalse(any(self.engine.slot_busy))


if __name__ == '__main__':
    unittest.main()
