#!/usr/bin/env python3
"""Offline Linux process/API fixtures. Disposable processes; loopback fake API only."""
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nibbler_lifecycle as life
import nibbler_perf as perf

HERE = str(Path(__file__).resolve())


def wait_for(check, timeout=10):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        answer = check()
        if answer:
            return answer
        time.sleep(.025)
    raise AssertionError("offline fixture deadline")


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def fixture(mode, argv):
    if mode == "supervise":
        # Same Supervisor CLI, but an offline kernel without CONFIG_PROC_CHILDREN.
        with patch.object(life, 'has_children', side_effect=FileNotFoundError('unsupported proc children')):
            life.supervise(Path(argv[0]), int(argv[1]), json.loads(argv[2]), json.loads(argv[3]),
                           float(argv[4]), float(argv[5]), json.loads(argv[6]))
    elif mode == "grandchild":
        root = Path(argv[0])
        child = subprocess.Popen([sys.executable, HERE, "sleeper"], start_new_session=True)
        life.durable(root / "descendants.json", [life.identity(os.getpid()), life.identity(child.pid)])
        time.sleep(60)
        child.wait()
    elif mode == "sleeper":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
    elif mode == "tree":
        root = Path(argv[0])
        journal = read_json(root / "lifecycle.json")
        # The target MUST NOT execute before durable, PID-start-scoped ownership.
        assert journal["phase"] == "private"
        assert life.identity(os.getpid()) in journal["children"]
        life.durable(root / "frontend.json", life.identity(os.getpid()))
        child = subprocess.Popen([sys.executable, HERE, "grandchild", str(root)], start_new_session=True)
        wait_for(lambda: (root / "descendants.json").exists())
        life.durable(root / "tree-ready.json", True)
        if argv[1:] == ["exit"]:
            # Simulates a frontend which exits, leaving detached engine/vision alive.
            os._exit(0)
        time.sleep(60)
        child.wait()
    elif mode in ("recover", "mutate"):
        path = Path(argv[0])
        state = life.recovery_guard(path, phase="restoring" if mode == "recover" else "unloading")
        base = state["baseline"]["base"]
        if mode == "recover":
            root = path.parent
            rows = (read_json(root / "descendants.json") or [])
            parent = read_json(root / "frontend.json")
            if parent:
                rows.append(parent)
            assert not any(life.same(who) for who in rows), "restoration before private tree empty"
            response = perf.http(base + "/restore", {}, json_response=False)
            life.durable(root / "fake-restoration.json", {"response": response, "private_empty": True})
        else:
            perf.http(base + "/unload", {}, json_response=False)
    elif mode == "coordinator":
        root, base, how = Path(argv[0]), argv[1], argv[2]
        journal = root / "lifecycle.json"
        owner = life.Supervisor(journal, [sys.executable, HERE, "recover", str(journal)],
                                1.5 if how == "stopped" else 30, recovery_timeout=5,
                                mutation=[sys.executable, HERE, "mutate", str(journal)])
        with perf.interruption_handlers():
            try:
                owner.arm({"base": base})
                owner.mutate()
                if how == "lost-spawn-ack":
                    life._send(owner.sock, {"op": "spawn", "argv": [sys.executable, HERE, "tree", str(root)],
                                          "cwd": str(root), "env": dict(os.environ), "log": str(root / "private.log")})
                    owner.sock.close()  # A spawn can execute even if its ACK cannot be delivered.
                else:
                    child = owner.spawn([sys.executable, HERE, "tree", str(root)] +
                                        (["exit"] if how == "rootexit" else []), root, dict(os.environ),
                                        root / "private.log")
                    wait_for(lambda: (root / "tree-ready.json").exists())
                    if how == "rootexit":
                        wait_for(lambda: child.poll() is not None)
                life.durable(root / "coordinator-ready.json", {"supervisor": owner.who})
                if how == "disconnect":
                    owner.sock.close()
                elif how == "broken-rpc":
                    owner.sock.send(b"{")
                elif how in ("kill", "graceful", "rootexit", "stopped"):
                    time.sleep(60)
            finally:
                owner.finish()
    else:
        raise ValueError(mode)


class FakeAPI:
    def __init__(self, body=b"OK", failures=0):
        self.events = []
        self.ready = True
        self.body = body
        self.failures = failures
        self.disconnect_unload = False
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                owner.events.append(self.path)
                if self.path == "/unload":
                    owner.ready = False
                    if owner.disconnect_unload:
                        self.close_connection = True  # Mutation took effect; successful response was lost.
                        return
                elif self.path == "/restore":
                    if owner.failures:
                        owner.failures -= 1
                        self.send_response(503)
                        self.end_headers()
                        return
                    owner.ready = True  # Idempotent: repeating restore does not unload.
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(owner.body)
            def log_message(self, *_):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.api = FakeAPI()
        self.owners = []
        self.children = []
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.api.close)
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        for owner in self.owners:
            owner.sock.close()
            owner.process.wait(timeout=15)

    def owner(self, seconds=30, recovery=None, recovery_timeout=5):
        journal = self.root / "lifecycle.json"
        result = life.Supervisor(journal, recovery or [sys.executable, HERE, "recover", str(journal)],
                                 seconds, recovery_timeout=recovery_timeout,
                                 mutation=[sys.executable, HERE, "mutate", str(journal)])
        self.owners.append(result)
        return result

    def start_tree(self, owner, exit_parent=False):
        child = owner.spawn([sys.executable, HERE, "tree", str(self.root)] +
                            (["exit"] if exit_parent else []), self.root, dict(os.environ),
                            self.root / "private.log")
        wait_for(lambda: (self.root / "tree-ready.json").exists())
        return child

    def assert_restored(self):
        state = wait_for(lambda: (s if (s := read_json(self.root / "lifecycle.json")) and
                                 s["phase"] in ("restored", "blocked") else None))
        self.assertEqual(state["phase"], "restored", state)
        self.assertTrue(self.api.ready)
        self.assertEqual(self.api.events[0], "/unload")
        self.assertIn("/restore", self.api.events)
        self.assertTrue(read_json(self.root / "fake-restoration.json")["private_empty"])
        for who in read_json(self.root / "descendants.json") or []:
            self.assertFalse(life.same(who))
        return state

    def coordinator(self, mode):
        proc = subprocess.Popen([sys.executable, HERE, "coordinator", str(self.root), self.api.base, mode],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(proc)
        wait_for(lambda: (self.root / "coordinator-ready.json").exists())
        return proc

    def test_readiness_no_mutation_and_unarmed_finish(self):
        owner = self.owner()
        state = read_json(self.root / "lifecycle.json")
        self.assertEqual(state["phase"], "ready")
        self.assertTrue(life.same(state["supervisor"]))
        self.assertEqual(owner.finish()["phase"], "unarmed")
        self.assertEqual(self.api.events, [])

    def test_missing_kernel_children_refuses_readiness_before_journal_or_mutation(self):
        with self.assertRaisesRegex(life.LifecycleError, 'kernel direct-child list unavailable'):
            life.Supervisor(self.root / 'lifecycle.json', [sys.executable, HERE, 'recover'],
                            30, module=HERE, mutation=[sys.executable, HERE, 'mutate'])
        self.assertFalse((self.root / 'lifecycle.json').exists())
        self.assertEqual(self.api.events, [])
        with patch.object(life, 'has_children', return_value=True):
            self.assertEqual(life.probe_children(), {'supported': True})  # Availability != empty tree.

    def test_empty_and_plaintext_controls_full_tree_before_restore(self):
        # Both successful control response contracts exercised through real local HTTP.
        for body in (b"", b"Model unloaded"):
            with self.subTest(body=body):
                self.api.body = body
                owner = self.owner()
                owner.arm({"base": self.api.base})
                owner.mutate()
                child = self.start_tree(owner)
                perf.close_process(child)
                self.assertIsNotNone(child.poll())
                self.assertEqual(owner.finish()["phase"], "restored")
                self.assert_restored()
                for path in self.root.iterdir():
                    path.unlink()

    def test_coordinator_term_int_hup_and_hard_kill(self):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGKILL):
            with self.subTest(sig=sig):
                proc = self.coordinator("kill" if sig == signal.SIGKILL else "graceful")
                os.kill(proc.pid, sig)
                proc.wait(timeout=15)
                self.assert_restored()
                for path in self.root.iterdir():
                    path.unlink()

    def test_disconnect_broken_rpc_lost_spawn_ack_and_exited_frontend(self):
        for mode in ("disconnect", "broken-rpc", "lost-spawn-ack", "rootexit"):
            with self.subTest(mode=mode):
                proc = self.coordinator(mode)
                if mode == "rootexit":
                    proc.kill()
                proc.wait(timeout=15)
                self.assert_restored()
                for path in self.root.iterdir():
                    path.unlink()

    def test_independent_deadline_rejects_late_coordinator_mutation(self):
        owner = self.owner(seconds=.8)
        owner.arm({"base": self.api.base})
        owner.mutate()
        self.start_tree(owner)
        owner.process.wait(timeout=15)
        self.assertIn("deadline", self.assert_restored()["trigger"])
        # The now-stale coordinator cannot issue a late mutation after restoration.
        with self.assertRaises(life.LifecycleError):
            owner.mutate()
        self.assertEqual(self.api.events.count("/unload"), 1)

    def test_sigstop_coordinator_cannot_hold_private_tree_past_deadline(self):
        proc = self.coordinator("stopped")
        os.kill(proc.pid, signal.SIGSTOP)
        try:
            state = self.assert_restored()
            self.assertIn("deadline", state["trigger"])
            self.assertIsNone(proc.poll())  # Still stopped; no coordinator finally ran.
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_spawn_errors_cleanup_restore_or_unarmed_refusal(self):
        owner = self.owner()
        owner.arm({"base": self.api.base})
        owner.mutate()
        with self.assertRaises((EOFError, OSError, life.LifecycleError)):
            owner.spawn(["/missing/executable"], self.root / "missing-cwd", {}, self.root / "log")
        owner.process.wait(timeout=15)
        self.assert_restored()

    def test_unload_takes_effect_but_api_connection_drops_still_restores(self):
        self.api.disconnect_unload = True
        owner = self.owner()
        owner.arm({'base': self.api.base})
        with self.assertRaises((EOFError, OSError, life.LifecycleError)):
            owner.mutate()
        owner.process.wait(timeout=15)
        state = self.assert_restored()
        self.assertIn('mutation callback failed', state['trigger'])
        self.assertFalse((self.root / 'private.log').exists())

    def test_exec_error_after_gate_still_recovers(self):
        owner = self.owner()
        owner.arm({"base": self.api.base})
        owner.mutate()
        child = owner.spawn(["/missing/executable"], self.root, {}, self.root / "log")
        wait_for(lambda: child.poll() is not None)
        owner.finish()
        self.assert_restored()

    def test_startup_error_never_arms_or_mutates(self):
        journal = self.root / "lifecycle.json"
        journal.write_text("already exists")
        with self.assertRaises((EOFError, OSError, life.LifecycleError)):
            self.owner()
        self.assertEqual(journal.read_text(), "already exists")
        self.assertEqual(self.api.events, [])

    def test_readiness_response_connection_loss_never_arms_or_mutates(self):
        with patch.object(life, '_receive', side_effect=EOFError('lost ready response')):
            with self.assertRaises(EOFError):
                self.owner()
        state = read_json(self.root / 'lifecycle.json')
        self.assertEqual(state['phase'], 'unarmed')
        self.assertFalse(state['armed'])
        self.assertEqual(self.api.events, [])

    def test_two_failed_restores_are_bounded_and_honestly_blocked(self):
        self.api.failures = 2
        owner = self.owner()
        owner.arm({'base': self.api.base})
        owner.mutate()
        with self.assertRaisesRegex(life.LifecycleError, 'bounded original-model restoration'):
            owner.finish()
        self.assertFalse(self.api.ready)
        state = read_json(self.root / 'lifecycle.json')
        self.assertEqual(state['phase'], 'blocked')
        self.assertEqual(state['restore_attempt'], 2)
        self.assertEqual(self.api.events, ['/unload', '/restore', '/restore'])

    def test_gate_eof_never_executes_target(self):
        read_fd, write_fd = os.pipe()
        marker = self.root / "should-not-exist"
        child = subprocess.Popen([sys.executable, str(Path(life.__file__).resolve()), "gate", str(read_fd),
                                  sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
                                 pass_fds=(read_fd,))
        os.close(read_fd)
        os.close(write_fd)
        self.assertEqual(child.wait(timeout=5), 1)
        self.assertFalse(marker.exists())

    def test_identity_reuse_and_foreign_process_protected(self):
        foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.children.append(foreign)
        who = life.identity(foreign.pid)
        altered = dict(who, start=str(int(who["start"]) + 1))
        self.assertFalse(life.signal_identity(altered, signal.SIGKILL))
        with patch.object(life, "same", side_effect=[True, False]), \
             patch.object(signal, "pidfd_send_signal") as send:
            self.assertFalse(life.signal_identity(who, signal.SIGKILL))
            send.assert_not_called()  # Reuse BETWEEN identity check and pidfd open.
        owner = self.owner()
        owner.arm({"base": self.api.base})
        owner.mutate()
        self.start_tree(owner)
        owner.finish()
        self.assert_restored()
        self.assertIsNone(foreign.poll())

    def test_guard_rejects_forged_stale_unarmed_and_writable_journal(self):
        journal = self.root / "guard.json"
        state = {"armed": True, "phase": "restoring", "supervisor": life.identity(os.getppid())}
        life.durable(journal, state)
        journal.chmod(0o600)
        self.assertEqual(life.recovery_guard(journal), state)
        for changes in ({"armed": False}, {"phase": "ready"},
                        {"supervisor": dict(state["supervisor"], start="wrong")}):
            life.durable(journal, dict(state, **changes))
            with self.assertRaises(life.LifecycleError):
                life.recovery_guard(journal)
        life.durable(journal, state)
        journal.chmod(0o666)
        with self.assertRaises(life.LifecycleError):
            life.recovery_guard(journal)
        link = self.root / "symlink"
        link.symlink_to(journal)
        with self.assertRaises(life.LifecycleError):
            life.recovery_guard(link)

    def test_bounded_retry_and_failed_restore_report(self):
        self.api.failures = 1
        owner = self.owner()
        owner.arm({"base": self.api.base})
        owner.mutate()
        owner.finish()
        self.assertEqual(self.assert_restored()["restore_attempt"], 2)
        self.assertEqual(self.api.events, ["/unload", "/restore", "/restore"])

    def test_restore_timeout_kills_recovery_descendants_and_reports_blocked(self):
        owner = self.owner(recovery=[sys.executable, HERE, "sleeper"], recovery_timeout=.2)
        owner.arm({"base": self.api.base})
        owner.mutate()
        with self.assertRaisesRegex(life.LifecycleError, "restoration deadline"):
            owner.finish()
        state = read_json(self.root / "lifecycle.json")
        self.assertEqual(state["phase"], "blocked")
        self.assertEqual(state["restore_attempt"], 1)
        self.assertFalse(self.api.ready)  # Do NOT claim successful recovery of an unavailable API.

    def test_operator_lock_retained_by_supervisor(self):
        lockpath = self.root / "operator.lock"
        with lockpath.open("a") as lock:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner = life.Supervisor(self.root / "lifecycle.json", [sys.executable, "-c", "pass"],
                                    30, lock_fd=lock.fileno())
            self.owners.append(owner)
        with lockpath.open("a") as other:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner.finish()
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_unreapable_tree_fails_closed(self):
        with patch.object(life, "descendants", return_value=[{"pid": 999999}]), \
             patch.object(life, "signal_identity", return_value=False), patch.object(life, "reap"), \
             patch.object(life, "has_children", return_value=True):
            with self.assertRaisesRegex(life.LifecycleError, "restoration prohibited"):
                life.clear_tree(timeout=.05)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("supervise", "grandchild", "sleeper", "tree", "recover", "mutate", "coordinator"):
        fixture(sys.argv[1], sys.argv[2:])
    else:
        unittest.main()
