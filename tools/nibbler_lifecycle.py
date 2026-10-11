#!/usr/bin/env python3
"""Linux-only private process owner. No services, GPU calls or model operations.

The separate single-threaded subreaper survives coordinator signals/session loss.
It owns every launch, including a gated pre-exec child. Recovery is attempted only
AFTER its entire descendant tree is empty. Supervisor death/OOM/reboot is outside
this contract; a journal is evidence, not permission to kill guessed PIDs.
"""
from __future__ import annotations

import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time


class LifecycleError(RuntimeError):
    pass


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start": fields[19],
                "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (OSError, IndexError):
        return None


def same(who):
    return who is not None and identity(who["pid"]) == who


def durable(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def signal_identity(who, sig):
    """pidfd pins the selected process; recheck start identity after opening it."""
    if not same(who):
        return False
    try:
        fd = os.pidfd_open(who["pid"])
    except ProcessLookupError:
        return False
    try:
        if not same(who):
            return False
        signal.pidfd_send_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def descendants():
    """All descendants of THIS subreaper, including adopted double-fork/setsid children."""
    rows = {}
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            pid = int(path.name)
            rows[pid] = (int(fields[1]), {"pid": pid, "start": fields[19], "boot": boot})
        except (OSError, ValueError, IndexError):
            continue
    owned = {os.getpid()}
    while True:
        more = {pid for pid, (parent, _) in rows.items() if parent in owned and
                (parent == os.getpid() or same(rows[parent][1]))} - owned
        if not more:
            break
        owned.update(more)
    # Use the identity from the SAME stat read as PPID, never reread a reused leaf PID.
    return [rows[pid][1] for pid in owned - {os.getpid()}]


def reap():
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


def has_children():
    return bool(Path(f"/proc/self/task/{os.getpid()}/children").read_text().strip())


def probe_children():
    """Check CONFIG_PROC_CHILDREN/procfs access; nonempty is fine at preflight."""
    try:
        has_children()
    except OSError as exc:
        raise LifecycleError("kernel direct-child list unavailable: /proc/self/task/<PID>/children") from exc
    return {"supported": True}


def clear_tree(timeout=10):
    """Repeated census closes fork/reparent races; never restore with survivors."""
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        rows = descendants()
        for who in rows:
            signal_identity(who, signal.SIGKILL)
        reap()
        # /proc enumeration alone can miss a child forked while its parent exits.
        # The kernel's direct-child list is the final witness: with an established
        # subreaper, no direct children means no descendant can still fork/reparent.
        if not has_children():
            return
        time.sleep(.025)
    raise LifecycleError("private descendants survive bounded reaper; restoration prohibited")


def _send(sock, value):
    raw = json.dumps(value).encode()
    if len(raw) > 65536:
        raise LifecycleError("IPC packet bound")
    sock.sendall(raw)


def _receive(sock):
    raw = sock.recv(65537)
    if not raw:
        raise EOFError("coordinator transport disconnected")
    if len(raw) > 65536:
        raise LifecycleError("IPC packet bound")
    return json.loads(raw)


class OwnedProcess:
    def __init__(self, owner, who):
        self.owner, self.who = owner, who
        self.pid = who["pid"]

    def poll(self):
        return None if same(self.who) else 1

    def terminate(self):
        # No PID/PGID guessing. Full-tree cleanup happens in the supervisor.
        signal_identity(self.who, signal.SIGTERM)

    def close(self):
        self.owner.call({"op": "stop"})


class Supervisor:
    def __init__(self, journal, recovery, seconds, recovery_timeout=720, lock_fd=None, module=None,
                 mutation=None):
        self.journal = Path(journal).resolve()
        self.sock, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.sock.settimeout(30)
        self.process = None
        self.finish_timeout = 2 * recovery_timeout + 30
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(module or __file__).resolve()), "supervise", str(self.journal),
                 str(child.fileno()), json.dumps(identity(os.getpid())), json.dumps(recovery),
                 str(seconds), str(recovery_timeout), json.dumps(mutation)],
                pass_fds=(child.fileno(),) + (() if lock_fd is None else (lock_fd,)), start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            child.close()
            answer = _receive(self.sock)
            if not answer.get("ok"):
                raise LifecycleError(answer.get("error", "supervisor failed readiness"))
            self.who = answer["supervisor"]
        except BaseException:
            child.close()
            self.sock.close()  # EOF causes cleanup even when readiness response was lost.
            if self.process is not None:
                self.process.wait(timeout=15)
            raise

    def call(self, value):
        if not same(self.who):
            raise LifecycleError("independent supervisor lost; do not restore or launch")
        _send(self.sock, value)
        answer = _receive(self.sock)
        if not answer.get("ok"):
            raise LifecycleError(answer.get("error", "supervisor refused"))
        return answer

    def arm(self, baseline):
        self.call({"op": "arm", "baseline": baseline})

    def mutate(self):
        self.call({"op": "mutate"})

    def spawn(self, argv, cwd, env, log):
        answer = self.call({"op": "spawn", "argv": argv, "cwd": str(cwd),
                            "env": env, "log": str(log)})
        return OwnedProcess(self, answer["child"])

    def finish(self):
        # EOF is the same recovery trigger as transport loss / coordinator death.
        self.sock.close()
        self.process.wait(timeout=self.finish_timeout)
        state = json.loads(self.journal.read_text())
        if state["phase"] not in ("restored", "unarmed"):
            raise LifecycleError(state.get("error", "independent restoration failed"))
        return state


def supervise(path, fd, coordinator, recovery, seconds, recovery_timeout, mutation=None):
    os.umask(0o077)
    # This ctypes call is in an independent, single-threaded process, NOT preexec_fn.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise LifecycleError("cannot establish Linux subreaper")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise LifecycleError("Linux pidfd support required")
    probe = os.pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(probe, 0)  # Check kernel support, not merely Python symbols.
    finally:
        os.close(probe)
    try:
        probe_children()  # Load-bearing empty-tree witness must exist BEFORE readiness.
    except LifecycleError as exc:
        with socket.socket(fileno=fd) as sock:
            _send(sock, {"ok": False, "error": str(exc)})
        return
    with socket.socket(fileno=fd) as sock, Path(str(path) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"version": 1, "supervisor": identity(os.getpid()), "coordinator": coordinator,
                 "phase": "ready", "armed": False, "recovery": recovery, "children": [],
                 "deadline_monotonic": time.monotonic() + seconds}
        if Path(path).exists():
            raise LifecycleError("journal already exists")
        durable(path, state)
        # SIGHUP/TERM/INT request cleanup, but never interrupt recovery mid-flight.
        stopping = False
        def stop(signum, _frame):
            nonlocal stopping
            stopping = True
            state["trigger"] = f"supervisor signal {signal.Signals(signum).name}"
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, stop)
        roots = []  # Keep Popen handles until poll accounts for waitpid/reaping.
        try:
            _send(sock, {"ok": True, "supervisor": state["supervisor"]})
            while not stopping:
                for child in roots:
                    child.poll()
                reap()
                if not same(coordinator):
                    raise LifecycleError("coordinator identity ended")
                if time.monotonic() >= state["deadline_monotonic"]:
                    raise LifecycleError("independent window deadline")
                if not select.select([sock], [], [], .1)[0]:
                    continue
                msg = _receive(sock)
                op = msg["op"]
                if op == "arm":
                    if state["armed"]:
                        raise LifecycleError("already armed")
                    state.update(armed=True, baseline=msg["baseline"], phase="armed")
                    durable(path, state)  # ACK only AFTER durable unload intent.
                    _send(sock, {"ok": True})
                elif op == "mutate":
                    if not mutation or not state["armed"] or state.get("mutation_started") or has_children():
                        raise LifecycleError("mutation requires armed, unused, empty private tree")
                    state.update(phase="unloading", mutation_started=True)
                    durable(path, state)
                    with Path(str(path) + ".mutation.log").open("xb") as log:
                        child = subprocess.Popen(mutation, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT)
                        roots.append(child)
                        child.wait(timeout=45)
                    clear_tree()
                    if child.returncode:
                        raise LifecycleError("model-only mutation callback failed")
                    state["phase"] = "armed"
                    durable(path, state)
                    _send(sock, {"ok": True})
                elif op == "spawn":
                    if not state["armed"] or has_children():
                        raise LifecycleError("spawn requires armed, empty private tree")
                    state.update(phase="spawn-intent")
                    durable(path, state)  # ownership recorded BEFORE any fork
                    read_fd, write_fd = os.pipe()
                    try:
                        with open(msg["log"], "xb") as log:
                            proc = subprocess.Popen(
                                [sys.executable, str(Path(__file__).resolve()), "gate", str(read_fd),
                                 *msg["argv"]], cwd=msg["cwd"], env=msg["env"],
                                pass_fds=(read_fd,), stdout=log, stderr=subprocess.STDOUT)
                        roots.append(proc)
                        who = identity(proc.pid)
                        if who is None:
                            raise LifecycleError("gated child vanished")
                        state.update(phase="private", children=[who])
                        durable(path, state)
                        os.write(write_fd, b"G")  # no exec before durable PID/start record
                        _send(sock, {"ok": True, "child": who})
                    finally:
                        os.close(read_fd)
                        os.close(write_fd)
                elif op == "stop":
                    clear_tree()
                    state.update(phase="armed", children=[])
                    durable(path, state)
                    _send(sock, {"ok": True})
                else:
                    raise LifecycleError("unknown operation")
        except BaseException as exc:
            state["trigger"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                state["phase"] = "reaping"
                durable(path, state)
                clear_tree()
                for child in roots:
                    child.poll()
                state["children"] = []
                if state["armed"]:
                    state["phase"] = "restoring"
                    durable(path, state)
                    # Recovery may be re-entered, but is bounded and serialized by this lock.
                    for attempt in range(2):
                        state["restore_attempt"] = attempt + 1
                        durable(path, state)
                        with Path(str(path) + ".recovery.log").open("ab") as log:
                            p = subprocess.Popen(recovery, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT)
                            try:
                                rc = p.wait(timeout=recovery_timeout)
                            except subprocess.TimeoutExpired:
                                clear_tree()
                                p.wait(timeout=1)
                                raise LifecycleError("restoration deadline; no unlimited retries")
                        clear_tree()
                        if rc == 0:
                            break
                    else:
                        raise LifecycleError("bounded original-model restoration attempts failed")
                    state["phase"] = "restored"
                else:
                    state["phase"] = "unarmed"
                durable(path, state)
            except BaseException as exc:
                state.update(phase="blocked", error=f"{type(exc).__name__}: {exc}")
                durable(path, state)


def recovery_guard(path, phase="restoring"):
    """Only the journal's living owner may invoke recovery. Never trust reused PIDs."""
    path = Path(path).absolute()
    if path.is_symlink() or path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o022:
        raise LifecycleError("untrusted recovery journal")
    state = json.loads(path.read_text())
    if not state["armed"] or state["phase"] != phase:
        raise LifecycleError("journal not in restoration phase")
    if state["supervisor"] != identity(os.getppid()):
        raise LifecycleError("recovery requires original living supervisor; manual recovery unsupported")
    return state


def main():
    if sys.argv[1] == "gate":
        fd = int(sys.argv[2])
        token = os.read(fd, 1)
        os.close(fd)
        if token != b"G":
            return 1
        os.execve(sys.argv[3], sys.argv[3:], dict(os.environ))
    elif sys.argv[1] == "supervise":
        supervise(Path(sys.argv[2]), int(sys.argv[3]), json.loads(sys.argv[4]),
                  json.loads(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), json.loads(sys.argv[8]))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
