"""Unified KV cancellation/rejection lifecycle over a scripted engine (no GPU).

    python3 -m unittest serve.test_unified_lifecycle -v
"""
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.server import StrataEngine


FAKE_UNIFIED = r'''import queue, sys, threading, time
from pathlib import Path
root = Path(sys.argv[sys.argv.index("--control-dir") + 1])
scenario = sys.argv[sys.argv.index("--scenario") + 1]
commands, stop = queue.Queue(), threading.Event()
def reader():
    for line in sys.stdin:
        line = line.strip()
        if line == "STOP":
            stop.set()
        else:
            commands.put(line)
    commands.put(None)
threading.Thread(target=reader, daemon=True).start()
def wait(name):
    deadline = time.monotonic() + 8
    while not (root / name).exists():
        if time.monotonic() > deadline:
            raise RuntimeError("test did not release " + name)
        time.sleep(0.005)
def emit(line):
    print(line, flush=True)
print("INFO batch_slots=2 kv_unified=1 kv_capacity_cells=4096", flush=True)
print("READY 4096 stop", flush=True)
first, paused = True, None
while True:
    line = commands.get()
    if line is None or line == "QUIT":
        break
    if line.startswith("BSTOP "):
        slot = int(line.split()[1])
        (root / "bstop").write_text(str(slot))
        if slot == paused:
            wait("release_ack")
            paused = None
            emit(f"BDONE {slot} 0 cancel 0")
        continue
    if not line.startswith(("GEN ", "BGEN ")):
        continue
    fields = line.split()
    slot = int(fields[1]) if fields[0] == "BGEN" else None
    n = len(fields[-1].split(","))
    if first:
        first = False
        if scenario == "legacy":
            emit("ERR legacy validation error")  # ERR-only: no DONE/BADM follows
            continue
        emit(f"PP 1 {n} 1 1")
        if scenario == "reject":
            wait("release_rejection")
            emit("ERR unified KV admission: shared capacity exhausted or admission cancelled")
        else:
            command = commands.get()
            if not command.startswith("BYIELD "):
                raise RuntimeError("expected BYIELD, got " + str(command))
            paused = int(command.split()[1])
            emit(f"YIELDED {paused} 1")
        emit(f"DONE 0 {n} 0 0 cancel 0 0 0")
        if slot is not None:
            emit(f"BADM {slot} 0")
        continue
    if paused == slot and slot is not None:
        emit("ERR slot reused before paused reservation was stopped")
        continue
    stop.clear()
    emit("T 120")
    emit(f"DONE 1 {n} 0 1 stop 0 0 0")
    if slot is not None:
        emit(f"BADM {slot} 0")
'''


class UnifiedLifecycle(unittest.TestCase):
    def start(self, scenario):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        script = self.root / "engine.py"
        script.write_text(FAKE_UNIFIED, encoding="utf-8")
        popen = server.subprocess.Popen
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: popen([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", ["--batch", "2", "--scenario", scenario,
                                                   "--control-dir", str(self.root)])
        # A failed regression must not spend the production 300-second drain timeout in a test.
        drain = self.engine._drain_control
        self.engine._drain_control = lambda until: drain(until, timeout=2)

    def tearDown(self):
        if getattr(self, "root", None):
            (self.root / "release_ack").touch()
            (self.root / "release_rejection").touch()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def wait_for(self, predicate, message):
        deadline = time.monotonic() + 4
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(predicate(), message)

    def assert_next_request(self, max_new):
        self.assertEqual(list(self.engine.generate([8, 9], max_new, {}, threading.Event())), [120])
        self.assertTrue(self.engine.lines.empty(), "old control replies contaminated the next request")

    def rejection(self, solo, early_close, batch=True):
        self.start("reject")
        if not batch:
            self.engine.batch = 0  # exercise the legacy non-batched generate/drain loop too
        cap = 64 if solo else 1  # cap=1 forces BGEN even without another active request
        gen = self.engine.generate([1, 2, 3], cap, {}, threading.Event())
        self.assertIsNone(next(gen))  # close before ERR arrives, while the engine is still reading
        (self.root / "release_rejection").touch()
        if early_close:
            gen.close()
        else:
            with self.assertRaisesRegex(ValueError, "unified KV admission:"):
                list(gen)
        self.assert_next_request(cap)

    def test_solo_early_close_drains_rejection_then_next_request(self):
        self.rejection(solo=True, early_close=True)

    def test_admission_early_close_drains_rejection_then_next_request(self):
        self.rejection(solo=False, early_close=True)

    def test_solo_observed_rejection_then_next_request(self):
        self.rejection(solo=True, early_close=False)

    def test_admission_observed_rejection_then_next_request(self):
        self.rejection(solo=False, early_close=False)

    def test_non_batched_early_close_drains_rejection_then_next_request(self):
        self.rejection(solo=True, early_close=True, batch=False)

    def test_non_batched_observed_rejection_then_next_request(self):
        self.rejection(solo=True, early_close=False, batch=False)

    def yielded(self, solo, early_close):
        self.start("yield")
        engine = self.engine
        engine.slot_order = [0]  # force reuse of the paused slot, not an unrelated free one
        cancel, waiting = threading.Event(), threading.Event()
        cap = 64 if solo else 1
        gen = engine.generate([1, 2, 3, 4], cap, {}, cancel)
        take_control = engine._take_control

        def blocked_resume(cancel, plen, after_epoch=None):
            if after_epoch is None:
                return (yield from take_control(cancel, plen))
            # Another admission owns ctl while the yielded request waits to resume. Use the real wait/cancel path.
            engine.ctl.acquire()
            waiting.set()
            try:
                return (yield from take_control(cancel, plen, after_epoch=after_epoch))
            finally:
                engine.ctl.release()

        errors = []
        def consume():
            try:
                list(gen)
            except BaseException as e:
                errors.append(e)

        with mock.patch.object(engine, "_shorter_waiting", return_value=True), \
                mock.patch.object(engine, "_take_control", side_effect=blocked_resume):
            self.assertIsNone(next(gen))  # PP causes BYIELD; the reply has not been consumed yet
            if early_close:
                gen.close()  # YIELDED is seen only by the cleanup drain, not the normal reader
            else:
                worker = threading.Thread(target=consume, daemon=True)
                worker.start()
                self.assertTrue(waiting.wait(4), "yielded request never waited to resume")
                cancel.set()
                worker.join(4)
                self.assertFalse(worker.is_alive(), "cancelled resume waiter did not exit")
                self.assertEqual(errors, [])
        self.wait_for(lambda: (self.root / "bstop").exists() and (self.root / "bstop").read_text() == "0",
                      "paused reservation never received BSTOP")
        self.assertTrue(engine.slot_busy[0], "server freed a paused slot before BDONE")
        self.assertIsNone(engine.pick_slot([8, 9]), "paused slot became reusable before BDONE")
        (self.root / "release_ack").touch()
        self.wait_for(lambda: not engine.slot_busy[0], "BDONE did not release the server slot")
        self.assertTrue(engine.slot_q[0].empty(), "BDONE acknowledgement was not consumed")
        self.assertEqual(engine.slot_held[0], [])
        self.assertEqual(engine.pick_slot([8, 9]), 0)
        self.assert_next_request(1)  # immediate BGEN reuses exactly the acknowledged slot

    def test_solo_yield_cancelled_while_waiting_stops_before_reuse(self):
        self.yielded(solo=True, early_close=False)

    def test_admission_yield_cancelled_while_waiting_stops_before_reuse(self):
        self.yielded(solo=False, early_close=False)

    def test_solo_yield_during_close_drain_stops_before_reuse(self):
        self.yielded(solo=True, early_close=True)

    def test_admission_yield_during_close_drain_stops_before_reuse(self):
        self.yielded(solo=False, early_close=True)

    def legacy_error(self, solo, batch=True):
        self.start("legacy")
        if not batch:
            self.engine.batch = 0
        cap = 64 if solo else 1
        before = time.monotonic()
        with self.assertRaisesRegex(ValueError, "legacy validation"):
            list(self.engine.generate([1, 2], cap, {}, threading.Event()))
        self.assertLess(time.monotonic() - before, 1, "ERR-only cleanup waited for a nonexistent terminal")
        self.assert_next_request(cap)

    def test_solo_legacy_err_only_does_not_wait_for_done(self):
        self.legacy_error(solo=True)

    def test_admission_legacy_err_only_does_not_wait_for_badm(self):
        self.legacy_error(solo=False)

    def test_non_batched_legacy_err_only_does_not_wait_for_done(self):
        self.legacy_error(solo=True, batch=False)

    def test_stop_without_ack_never_frees_slot_on_timeout(self):
        engine = StrataEngine("missing", [], lazy=True)
        engine.proc = object()
        engine.slot_q = [queue.Queue()]
        engine.slot_busy = [True]
        engine.slot_held = [[]]
        engine.slot_cv = threading.Condition()
        with mock.patch.object(engine, "_send") as send, \
                mock.patch("serve.server.threading.Thread") as thread, \
                mock.patch("serve.server.time.monotonic", side_effect=[0, 601]):
            thread.return_value.start.side_effect = lambda: thread.call_args.kwargs["target"]()
            engine._release_slot_when_done(0)
        send.assert_called_once_with("BSTOP 0")
        self.assertTrue(engine.slot_busy[0], "a timeout is not a BDONE acknowledgement")

    def test_drain_legacy_err_only_returns_immediately(self):
        engine = StrataEngine("missing", [], lazy=True)
        engine.lines = queue.Queue()
        engine.lines.put("ERR legacy error\n")
        self.assertEqual(engine._drain_control("BADM", timeout=0.1), "ERR legacy error\n")


if __name__ == "__main__":
    unittest.main()
