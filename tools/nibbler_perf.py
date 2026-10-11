#!/usr/bin/env python3
"""Bounded, opt-in nibbler calibration. Run ON nibbler through approved transport.

Default is a printed plan: no HTTP, processes, mkdir, unload or GPU access.
Live runs never install a binary or edit production configuration. See
 docs/NIBBLER_CALIBRATION.md for authorization, staged sweeps and limitations.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import resource
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zlib
import base64

# Even a first-ever dry-run must not create a local __pycache__ directory.
sys.dont_write_bytecode = True
from nibbler_lifecycle import Supervisor, recovery_guard
import nibbler_lifecycle

GIB = 1024 ** 3
MODEL = "qwen3.8-flash-next-nvfp4-gptq"
ROOT = Path("/data/llm/Strata")
TRIAL = Path("/data/llm/Strata-tests/perf-audit-20261011")
RUNTIME = Path("/data/llm/Strata-run/nibbler")
ERROR_RE = re.compile(r"(?im)(?:^ERR\b|out of memory|cudaMalloc.*fail|allocation.*fail|CUDA error|HIP error|Traceback|segmentation fault|bad_alloc)")


class Stop(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(argv, timeout=10):
    return subprocess.check_output(argv, text=True, timeout=timeout).strip()


def http(url, body=None, timeout=15, json_response=True):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise Stop("HTTP response exceeds 8 MiB")
        # Control endpoints can return empty or plain-text successful responses.
        # Only the explicitly named model-unload call opts out of strict JSON.
        if not json_response:
            return {"status": r.status, "body": data.decode(errors="replace")[:1024]}
        return json.loads(data) if data.strip() else {}


def record(path, obj):
    data = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    if len(data.encode()) > 8 * 1024 * 1024:
        raise Stop("JSON artifact exceeds 8 MiB")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data)
    tmp.replace(path)


def memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "SwapFree", "SwapTotal", "Cached", "AnonPages", "MemFree"):
            values[k] = int(v.split()[0]) * 1024
    return values


def safety(sample):
    if sample["MemAvailable"] < 10 * GIB:
        raise Stop("MemAvailable < 10 GiB")
    if sample["SwapFree"] < GIB:
        raise Stop("SwapFree < 1 GiB")


def gpu():
    raw = command([resolved_tool("nvidia-smi"), "--id=0", "--query-gpu=memory.used,memory.free,utilization.gpu,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader,nounits"])
    v = [int(x.strip()) for x in raw.split(",")]
    return dict(zip(("used_mib", "free_mib", "util_percent", "link_gen", "link_width"), v))


def identities():
    result = {}
    for name, prefix in (("router", "/opt/native-inference/bin/llama-swap"),
                         ("audio", "/opt/native-inference/bin/audiocpp_server")):
        found = []
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                cmd = (p / "cmdline").read_bytes().split(b"\0")[0].decode()
                if cmd == prefix:
                    # starttime plus PID prevents PID reuse from looking unchanged.
                    found.append({"pid": int(p.name), "start_ticks": (p / "stat").read_text().rsplit(")", 1)[1].split()[19]})
            except (OSError, UnicodeError, IndexError):
                continue
        if len(found) != 1:
            raise Stop(f"expected exactly one {name}, found {len(found)}")
        result[name] = found[0]
    return result


def production_hashes():
    paths = [ROOT / "engine/strata-nibbler", ROOT / "engine/strata-vision", ROOT / "engine/BUILD.json",
             ROOT / "deploy/nibbler/config.json", RUNTIME / "config.json", RUNTIME / "config.shared-settings.json"]
    return {str(p): sha(p) for p in paths}


def model_identity(cfg):
    pack = Path(cfg["args"][cfg["args"].index("--pack") + 1])
    root = pack.parent
    tokenizer = Path(cfg["tokenizer"])
    paths = [root / "SHA256SUMS", root / "SIZES.txt", tokenizer / "chat_template.jinja",
             tokenizer / "vocab.json", tokenizer / "merges.txt", tokenizer / "token_type.json"]
    # Manifest identity is NOT a new whole-pack checksum verification. No 63GiB read to warm caches.
    return {"hashes": {str(path): sha(path) for path in paths},
            "scope": "published pack manifest + exact tokenizer/template; pack payload not rehashed",
            "pack": str(pack)}


def idle(router):
    slots = http(router + f"/upstream/{MODEL}/slots")
    metrics = http(router + f"/upstream/{MODEL}/metrics")
    if len(slots) != 2 or any(s.get("is_processing") is not False for s in slots):
        raise Stop("production backend not verifiably idle")
    live = metrics["live"]
    if live["state"] != "idle" or any(live.get(k, 0) for k in ("queued", "running", "waiting", "outside_slots")):
        raise Stop("production has running or queued work")
    return {"slots": slots, "live": live, "engine": metrics["engine"]}


def running_summary(router):
    # /running includes command lines; retain only nonsensitive model/state.
    return [{"model": m["model"], "state": m["state"]} for m in http(router + "/running")["running"]]


def require_ready(rows):
    if not all(any(r["model"] == name and r["state"] == "ready" for r in rows) for name in (MODEL, "pocket-tts")):
        raise Stop("Qwen and pocket-tts are not both ready")


def flag(args, name, value):
    args = list(args)
    # Remove duplicates too; flag arguments in this runner always take one value.
    while name in args:
        i = args.index(name)
        del args[i:i + 2]
    return args + [name, str(value)]


def private_config(cfg, work, setting, port):
    out = copy.deepcopy(cfg)
    out.update(exe=str(work / "strata"), cwd=str(work), log=str(work / "engine.log"),
               host="127.0.0.1", port=port, allowed_hosts=["127.0.0.1", "localhost"])
    out["vision"]["exe"] = str(work / "strata-vision")
    out["args"] = flag(out["args"], "--vram-reserve-mib", setting["reserve"])
    # Baseline deliberately keeps the native startup link probe and default split.
    if setting["frac"] is not None:
        out["args"] = flag(out["args"], "--pcie-frac", setting["frac"])
    out["args"] = flag(out["args"], "--pool-workers", setting["workers"])
    # No profile saving, inherited test hooks, prefetch/task changes or elastic VRAM.
    out["env"].update(STRATA_DECODE_TIMING="1", STRATA_SPLIT_TIMING="1")
    for key in ("api_key", "before_load", "idle_unload_s", "slot_save_path", "mcp", "mcp_config"):
        out.pop(key, None)
    return out


def validate_config(cfg):
    a = cfg["args"]
    required = {"--max-context": "262144", "--kv": "int8", "--kv-resident": "32768",
                "--spec": "4", "--spec-min-p": "0.5", "--prefill": "auto:8192",
                "--conversation-cache-mib": "8192", "--conversation-cache-slots": "4"}
    if cfg.get("parallel") != 2 or cfg.get("vision", {}).get("max_tokens") != 1024 or not cfg["vision"].get("gpu"):
        raise Stop("two-slot/GPU vision baseline mismatch")
    for k, v in required.items():
        if k not in a or a[a.index(k) + 1] != v:
            raise Stop(f"baseline invariant {k}={v} failed")
    for k in ("--batch-mtp", "--vision", "--kv-unified", "--no-kv-grow"):
        if k not in a:
            raise Stop(f"baseline invariant {k} missing")
    if "qwen-nvfp4-gptq" not in a[a.index("--pack") + 1]:
        raise Stop("not the approved GPTQ pack")
    if cfg.get("model_name") != MODEL:
        raise Stop("model name mismatch")


def percentile(values, p=0.95):
    # Nearest-rank, clearly labelled descriptive p95; not a statistical confidence bound.
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


def summarize(rows):
    return {"requests": len(rows), "ttft_p95_s": percentile([r["ttft_s"] for r in rows]),
            "wall_p95_s": percentile([r["wall_s"] for r in rows]),
            "decode_tps": [r.get("timings", {}).get("predicted_per_second") for r in rows],
            "context_tokens": [r["usage"]["prompt_tokens"] for r in rows]}


def png_fixture():
    # Synthetic red circle on blue, 1024px to exercise the configured image-token cap.
    n = 1024
    rows = []
    for y in range(n):
        row = bytearray([0])
        for x in range(n):
            row.extend((255, 0, 0) if (x - n // 2) ** 2 + (y - n // 2) ** 2 < (n // 3) ** 2 else (0, 0, 255))
        rows.append(row)
    def chunk(k, b):
        return struct.pack(">I", len(b)) + k + b + struct.pack(">I", zlib.crc32(k + b) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")
    return [{"type": "text", "text": "Name the foreground shape and color and the background color, in one sentence."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]


def request(base, content, cap=96, barrier=None, timeout=600):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": cap,
               "temperature": 0, "seed": 12345, "chat_template_kwargs": {"enable_thinking": False},
               "stream": True, "stream_options": {"include_usage": True}}
    if barrier:
        barrier.wait(timeout=10)
    start = time.monotonic()
    first = None
    text, usage, timings, finish = "", None, None, None
    chunk_times = []
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        count = 0
        while True:
            raw = r.readline(65537)
            if not raw:
                break
            count += len(raw)
            if count > 2 * 1024 * 1024 or len(raw) > 65536 or time.monotonic() - start > timeout:
                raise Stop("stream byte/time bound exceeded")
            if not raw.startswith(b"data: "):
                continue
            raw = raw[6:].strip()
            if raw == b"[DONE]":
                break
            item = json.loads(raw)
            if "error" in item:
                raise Stop(f"stream error: {item['error']}")
            usage = item.get("usage") or usage
            timings = item.get("timings") or timings
            for choice in item.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    now = time.monotonic() - start
                    first = now if first is None else first
                    chunk_times.append(now)
                    text += piece
                finish = choice.get("finish_reason") or finish
    if first is None or not usage or not timings or finish not in ("length", "stop"):
        raise Stop("incomplete stream, usage/timing missing, or unexpected finish")
    if usage["completion_tokens"] <= 0 or usage["completion_tokens"] > cap or not text.strip():
        raise Stop("invalid output count/empty output")
    return {"started_monotonic": start, "first_monotonic": start + first, "ended_monotonic": time.monotonic(),
            "ttft_s": first, "wall_s": time.monotonic() - start, "usage": usage, "timings": timings,
            "finish": finish, "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "output_preview": text[:1024], "chunk_times_s": chunk_times,
            "inter_chunk_p95_s": percentile([b - a for a, b in zip(chunk_times, chunk_times[1:])])}


class Monitor:
    def __init__(self, out, router, deadline):
        self.out, self.router, self.deadline = out, router, deadline
        self.done, self.failed = threading.Event(), threading.Event()
        self.failure = None
        self.process = None
        self.logs = []
        self.phase = "preflight"
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def check(self):
        if self.failed.is_set():
            raise Stop(self.failure)

    def abort(self, message):
        self.failure = str(message)
        self.failed.set()
        self.kill()

    def kill(self):
        if self.process is not None:
            self.process.terminate()

    def sample(self):
        s = {"time": time.time(), "phase": self.phase, **memory(), "gpu": gpu(),
             "load": list(os.getloadavg())}
        safety(s)
        return s

    def _loop(self):
        try:
            with (self.out / "samples.jsonl").open("x") as f:
                n = 0
                while not self.done.is_set():
                    s = self.sample()
                    f.write(json.dumps(s) + "\n"); f.flush()
                    if time.monotonic() > self.deadline:
                        raise Stop("window deadline reached")
                    for log in self.logs:
                        if log.exists():
                            if log.stat().st_size > 8 * 1024 * 1024:
                                raise Stop("log exceeds 8 MiB")
                            if ERROR_RE.search(log.read_text(errors="replace")):
                                raise Stop(f"allocation/native error in {log.name}")
                    if n % 5 == 0 and self.process is not None:
                        if any(r["model"] == MODEL for r in running_summary(self.router)):
                            raise Stop("production Qwen reappeared during private test; no parallel load allowed")
                    n += 1
                    self.done.wait(1)
        except Exception as e:
            self.abort(e)
            record(self.out / "sampler-stop.json", {"reason": str(e), "phase": self.phase, "time": time.time(), "memory": memory()})

    def stop(self):
        self.done.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise Stop("sampler did not stop")


def wait_vram(monitor=None, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if monitor:
            monitor.check()
        if gpu()["used_mib"] < 2048:
            return
        time.sleep(1)
    raise Stop("VRAM did not fall below 2 GiB")


def close_process(p):
    if p is not None:
        p.close()  # Independent subreaper clears ALL private descendants, not a guessed PGID.


def resolved_tool(name):
    found = shutil.which(name, path="/usr/sbin:/usr/bin:/sbin:/bin")
    if not found:
        raise Stop(f"missing launch dependency: {name}")
    return str(Path(found).resolve(strict=True))


@contextlib.contextmanager
def interruption_handlers():
    def interrupted(signum, _frame):
        raise Stop(f"coordinator interrupted by {signal.Signals(signum).name}")
    old = {sig: signal.signal(sig, interrupted) for sig in
           (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def port_available(port):
    """Probe with the frontend's own bind semantics (serve.server.Server sets
    SO_REUSEADDR on Linux). A live listener still refuses; TIME_WAIT left by the
    previous arm's closed connections does not, matching the real server bind."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


def launch_check(config_path, uid, gid, ldd, execute=False):
    """Run AFTER runuser/PAM, with exact launch env. Never load a model here.

    Actual frontend exec passes this same check again. Finite memlock is a
    refusal, even if the coordinator recorded infinity. ldd checks trusted frozen
    ELF dependencies without running engine/vision main or touching a GPU.
    """
    if (os.getuid(), os.getgid()) != (uid, gid):
        raise Stop("effective native-inference identity mismatch")
    limits = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if limits != (resource.RLIM_INFINITY, resource.RLIM_INFINITY):
        raise Stop(f"effective post-PAM memlock degraded: {limits}")
    kernel_children = nibbler_lifecycle.probe_children()
    cfg = json.loads(Path(config_path).read_text())
    validate_config(cfg)
    import serve.server as server
    import strata_tokenizer  # noqa: F401 - dependencies, no model loading
    from PIL import Image  # noqa: F401
    paths = [cfg["tokenizer"], *cfg.get("lib_dirs", []), cfg["vision"]["mmproj"], cfg["vision"]["model"]]
    for opt in ("--pack", "--native", "--native-dense-gguf", "--ple-gguf", "--embd-gguf", "--mtp", "--expert-profile"):
        if opt in cfg["args"]:
            paths.append(cfg["args"][cfg["args"].index(opt) + 1])
    for value in paths:
        path = Path(value)
        if not path.is_absolute() or not os.access(path, os.R_OK):
            raise Stop(f"unreadable/nonabsolute model/library path: {path}")
        if path.is_dir():
            if not os.access(path, os.X_OK):
                raise Stop(f"unsearchable directory: {path}")
            with os.scandir(path) as entries:
                for entry in entries:
                    if not os.access(entry.path, os.R_OK):
                        raise Stop(f"unreadable entry: {entry.path}")
    for name in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja"):
        with (Path(cfg["tokenizer"]) / name).open("rb"):
            pass
    server.ChatTemplate(Path(cfg["tokenizer"]) / "chat_template.jinja")
    for value in (cfg["cwd"], os.environ["HOME"], os.environ["TMPDIR"], str(Path(cfg["log"]).parent)):
        with tempfile.TemporaryFile(dir=value):
            pass
    deps = {}
    for binary in (cfg["exe"], cfg["vision"]["exe"]):
        if not Path(binary).is_absolute() or not os.access(binary, os.R_OK | os.X_OK):
            raise Stop(f"private binary not executable/readable: {binary}")
        output = subprocess.check_output([ldd, binary], text=True, timeout=15, env=server.child_env(cfg))
        if "not found" in output:
            raise Stop(f"private binary has missing ELF dependencies: {binary}")
        deps[binary] = output
    # Still TOCTOU. Actual frontend checks bind before model load, too.
    port_available(cfg["port"])
    proof = {"uid": os.getuid(), "gid": os.getgid(), "memlock": list(limits),
             "python": sys.executable, "dependencies": deps, "config_sha256": sha(config_path),
             "kernel_children": kernel_children}
    if not execute:
        print(json.dumps(proof))
        return 0
    record(Path(cfg["cwd"]) / "launch-readiness.json", proof)
    os.execv(sys.executable, [sys.executable, "-m", "serve.server", "--engine", "strata",
                             "--config", str(config_path), "--host", "127.0.0.1", "--port", str(cfg["port"])])


def prepare_launches(cfg, out, frozen, schedule, port, preflight_only=False):
    native = pwd.getpwnam("native-inference")
    runuser, ldd = resolved_tool("runuser"), resolved_tool("ldd")
    # Preserve venv pathname: resolving the symlink would bypass venv site-packages.
    python = ROOT / ".venv/bin/python"
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise Stop("missing executable venv Python")
    launches = []
    for idx, setting in enumerate(schedule["settings"]):
        work = out / f"run-{idx:02}"
        work.mkdir(mode=0o700)
        for src, dst in ((ROOT / "engine/strata-nibbler", "strata"), (ROOT / "engine/strata-vision", "strata-vision")):
            shutil.copy2(src, work / dst)
            os.chmod(work / dst, 0o755)
        config = private_config(cfg, work, setting, port)
        record(work / "config.json", config)
        for path in work.iterdir():
            os.chown(path, native.pw_uid, native.pw_gid)
        os.chown(work, native.pw_uid, native.pw_gid)
        env = {"PATH": "/usr/local/cuda-13.3/bin:/usr/bin:/bin", "PYTHONPATH": str(frozen),
               "HOME": str(work), "TMPDIR": str(work), "PYTHONDONTWRITEBYTECODE": "1"}
        argv = [runuser, "-u", "native-inference", "--", str(python),
                str(frozen / "tools/nibbler_perf.py"), "--launch-check", str(work / "config.json"),
                str(native.pw_uid), str(native.pw_gid), ldd]
        checked = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=45)
        if checked.returncode:
            if preflight_only:
                try:
                    refused = json.loads(checked.stdout)
                except ValueError:
                    refused = {}
                record(work / "preflight.json", {"mode": "preflight-only", "ready": False,
                       **{key: refused[key] for key in ("uid", "gid", "memlock", "python") if key in refused}})
            raise Stop(f"no-model launch preflight refused: {checked.stderr[-4096:]}")
        proof = json.loads(checked.stdout)
        if proof["memlock"] != [resource.RLIM_INFINITY, resource.RLIM_INFINITY]:
            raise Stop("child preflight did not prove unlimited memlock")
        if preflight_only:
            # Neutral evidence is allowlisted, not arbitrary child stdout/library diagnostics.
            proof = {key: proof[key] for key in
                     ("uid", "gid", "memlock", "python", "config_sha256", "kernel_children")}
            proof.update(mode="preflight-only", ready=True, dependencies_resolved=True)
        record(work / "preflight.json", proof)
        launches.append((work, argv + ["--exec"], env))
    return launches


def unload_original(journal):
    state = recovery_guard(journal, phase="unloading")
    baseline = state["baseline"]
    if identities() != baseline["identities"] or production_hashes() != baseline["hashes"]:
        raise Stop("pre-unload identity/hash mismatch")
    router = "http://127.0.0.1:8088"
    safety(memory())
    idle(router)
    response = http(router + "/api/models/unload/" + MODEL, {}, timeout=30, json_response=False)
    record(Path(journal).parent / "unload-control.json", response)
    return 0


def restore_original(journal):
    """Independent, repeatable restoration; only callable by the live journal owner."""
    state = recovery_guard(journal)
    out = Path(journal).parent
    baseline = state["baseline"]
    router = "http://127.0.0.1:8088"
    try:
        if identities() != baseline["identities"] or production_hashes() != baseline["hashes"]:
            raise Stop("restoration identity/hash mismatch; no request permitted")
        # Arm ACK can precede any unload, or a retry can find Qwen already restored.
        rows = running_summary(router)
        if not any(r["model"] == MODEL and r["state"] == "ready" for r in rows):
            wait_vram()
        safety(memory())
        restore_samples = Path(tempfile.mkdtemp(dir=out, prefix=f"restore-monitor-{state['restore_attempt']}-"))
        monitor = Monitor(restore_samples, router, time.monotonic() + 600)
        monitor.phase = "production-restore-load"
        monitor.thread.start()
        try:
            restored = request(router, "Reply with exactly: CALIBRATION READY", cap=48)
            monitor.check()
        finally:
            monitor.stop()
        restored["measurement_kind"] = "health only; router may stream loading banners"
        rows = running_summary(router)
        require_ready(rows)
        after_ids, after_hash = identities(), production_hashes()
        audio = http("http://127.0.0.1:5800/health")
        if after_ids != baseline["identities"] or after_hash != baseline["hashes"] or audio.get("status") != "ok":
            raise Stop("production artifacts/router/audio changed")
        record(out / "restoration.json", {"health_request": restored, "running": rows,
               "identities": after_ids, "hashes_unchanged": True, "audio": audio,
               "memory": memory(), "gpu": gpu(), "restored": True})
        return 0
    except BaseException as exc:
        record(out / "restoration-error.json", {"error": str(exc), "memory": memory()})
        return 1


def plan(args):
    workers = [int(x) for x in args.workers.split(",")]
    fractions = [None if x == "baseline" else float(x) for x in args.fractions.split(",")]
    reserves = [int(x) for x in args.reserves.split(",")]
    if not workers or any(w not in (5, 7, 9, 11) for w in workers):
        raise Stop("workers must be 5,7,9,11")
    if not fractions or any(f not in (None, .10, .25, .40, .55) for f in fractions):
        raise Stop("fractions must be baseline,.10,.25,.40,.55")
    if not reserves or any(r not in (2048, 1536, 1024) for r in reserves):
        raise Stop("reserve must be 2048,1536,1024")
    if sum(len(x) > 1 for x in (workers, fractions, reserves)) > 1:
        raise Stop("one sweep dimension at a time; no combinatorial sweeps")
    if reserves[0] != 2048 or any(b > a for a, b in zip(reserves, reserves[1:])):
        raise Stop("reserve sweep must start at 2048 and descend")
    settings = [{"frac": f, "workers": w, "reserve": r} for r in reserves for w in workers for f in fractions]
    if len(settings) > 8 or not 1 <= args.repeats <= 3 or not 16 <= args.max_tokens <= 256:
        raise Stop("settings/repeat/output bound exceeded")
    if not 1 <= args.window_minutes <= 60 or not 1024 <= args.long_words <= 125000:
        raise Stop("window/context bound exceeded")
    if not 1 <= args.port <= 65535 or args.port in (8088, 5800, 5801):
        raise Stop("invalid or production test port")
    if any(r < 2048 for r in reserves) and not args.reserve_gates:
        raise Stop("lower reserves require --reserve-gates after reviewing control peaks")
    return {"settings": settings, "repeats": args.repeats, "port": args.port,
            "short_words": 128, "long_words": args.long_words, "max_tokens": args.max_tokens,
            "sequence": "alternate solo/dual, fresh then exact warm repeat; actual API prompt tokens recorded",
            "reserve_gates": args.reserve_gates, "window_minutes": args.window_minutes,
            "run": args.run, "production_mutations": "model-only unload/reload; no install/config edits"}


def live(args, schedule):
    if os.geteuid() != 0 or socket.gethostname().split(".")[0] != "nibbler":
        raise Stop("live coordinator must run as root ON nibbler")
    cfg = json.loads((RUNTIME / "config.json").read_text())
    validate_config(cfg)
    out = TRIAL / args.tag
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.umask(0o077)
    router = "http://127.0.0.1:8088"
    base = f"http://127.0.0.1:{args.port}"
    neutral = args.preflight_only
    owner = None
    p = None
    deadline = time.monotonic() + args.window_minutes * 60
    monitor = None if neutral else Monitor(out, router, deadline)
    result = {"plan": schedule, "passed": False, "runs": []}
    stage = "host-readiness"
    if neutral:
        result.update(mode="preflight-only", ready=False, production_mutations="none",
                      measurement_kind="readiness only; no performance evidence")
    with (TRIAL / "operator.lock").open("a") as lock, interruption_handlers():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # Bind only as a collision check, then close before starting the loopback server.
            with socket.socket() as s:
                s.bind(("127.0.0.1", args.port))
            safety(memory())
            # Child preflight MUST independently prove PAM did not degrade this limit.
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            before_ids, before_hash = identities(), production_hashes()
            if neutral:
                # A router upstream GET can autoload an absent model: refuse first.
                require_ready(running_summary(router))
            before = {"identities": before_ids, "hashes": before_hash, "idle": idle(router),
                      "running": running_summary(router), "audio": http("http://127.0.0.1:5800/health"),
                      "memory": memory(), "gpu": gpu(),
                      "gpu_consumers": command([resolved_tool("nvidia-smi"), "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"]),
                      "topology": command([resolved_tool("lscpu"), "-e=CPU,CORE,SOCKET,NODE,CACHE,ONLINE"]),
                      "model_identity": model_identity(cfg), "coordinator_memlock": list(resource.getrlimit(resource.RLIMIT_MEMLOCK)),
                      "coordinator_cpu_affinity": sorted(os.sched_getaffinity(0))}
            require_ready(before["running"])
            if neutral:
                if before["audio"].get("status") != "ok":
                    raise Stop("audio not ready")
                # Keep readiness facts, never raw slots/metrics/audio payloads or command lines.
                before["idle"] = {"verified": True, "slots": 2, "state": "idle"}
                before["audio"] = {key: before["audio"].get(key) for key in ("status", "backend")}
                before.pop("gpu_consumers")
                before.pop("topology")
                stage = "kernel-children"
                before["kernel_children"] = nibbler_lifecycle.probe_children()
            record(out / "before.json", before)
            record(out / "plan.json", schedule)
            # Freeze server code and deployed binaries, not a git worktree, before stopping Qwen.
            frozen = out / "frontend"
            frozen.mkdir()
            for d in ("serve", "tools"):
                shutil.copytree(ROOT / d, frozen / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            # The installed tools tree may not contain these new review-only files.
            for source in (Path(__file__), Path(nibbler_lifecycle.__file__)):
                shutil.copy2(source, frozen / "tools" / source.name)
            os.chmod(TRIAL, 0o711); os.chmod(out, 0o711)
            for path in frozen.rglob("*"):
                if path.is_dir():
                    os.chmod(path, 0o755)
                else:
                    os.chmod(path, 0o644)
            os.chmod(frozen, 0o755)
            stage = "launch-checks"
            launches = prepare_launches(cfg, out, frozen, schedule, args.port, preflight_only=neutral)
            if neutral:
                stage = "final-readiness"
                rows = running_summary(router)
                require_ready(rows)
                idle(router)  # GET only: no health completion or model control.
                audio = http("http://127.0.0.1:5800/health")
                after_ids, after_hash = identities(), production_hashes()
                safety(memory())
                if (after_ids != before_ids or after_hash != before_hash or
                        audio.get("status") != "ok" or audio.get("backend") != before["audio"].get("backend")):
                    raise Stop("production readiness/identity/hash changed during neutral preflight")
                if time.monotonic() >= deadline:
                    raise Stop("neutral preflight deadline exceeded")
                result.update(ready=True, production_unchanged=True,
                              readiness={"running": rows, "identities": after_ids, "hashes_unchanged": True,
                                         "audio": {key: audio.get(key) for key in ("status", "backend")}},
                              launch_checks=[json.loads((work / "preflight.json").read_text())
                                             for work, _, _ in launches])
                print("PREFLIGHT READY: no supervisor, unload, model request or private frontend started", flush=True)
                return 0  # finally records passed:false, runs:[]; no restoration path exists here.
            journal = out / "lifecycle.json"
            remaining = monitor.deadline - time.monotonic()
            if remaining <= 0:
                raise Stop("window deadline exhausted by no-model preflight; no unload")
            owner = Supervisor(journal,
                [sys.executable, str(frozen / "tools/nibbler_perf.py"), "--restore-journal", str(journal)],
                remaining, lock_fd=lock.fileno(),
                module=frozen / "tools/nibbler_lifecycle.py",
                mutation=[sys.executable, str(frozen / "tools/nibbler_perf.py"), "--unload-journal", str(journal)])
            monitor.thread.start()
            idle(router)  # immediately before mutation
            owner.arm({"identities": before_ids, "hashes": before_hash})
            # Only the owner can send unload: a suspended coordinator cannot mutate after recovery.
            monitor.check()
            owner.mutate()
            result["unload"] = json.loads((out / "unload-control.json").read_text())
            wait_vram(monitor)
            record(out / "unloaded.json", {"running": running_summary(router), "memory": memory(), "gpu": gpu()})
            print("UNLOADED Qwen only; starting private control", flush=True)
            for idx, setting in enumerate(schedule["settings"]):
                monitor.check()
                wait_vram(monitor)
                if setting["reserve"] < 2048:
                    previous = result["runs"][-1]
                    required_margin = 768 + (previous["setting"]["reserve"] - setting["reserve"])
                    if not previous.get("gates_passed") or previous["min_free_vram_mib"] < required_margin:
                        raise Stop("previous reserve gates/peak margin do not permit lower reserve")
                work, argv, env = launches[idx]
                log = work / "server.log"
                p = owner.spawn(argv, work, env, log)
                monitor.process = p
                monitor.logs = [log, work / "engine.log"]
                monitor.phase = f"run-{idx:02}-load"
                started = time.monotonic()
                deadline = started + 600
                while True:
                    monitor.check()
                    if p.poll() is not None:
                        raise Stop("private frontend exited before ready")
                    try:
                        metrics = http(base + "/metrics", timeout=2)
                        break
                    except (OSError, ValueError):
                        if time.monotonic() > deadline:
                            raise Stop("private model load deadline")
                        time.sleep(1)
                eng = metrics["engine"]
                if int(eng["batch_slots"]) != 2 or int(eng["kv_capacity_cells"]) != 262144:
                    raise Stop("private effective configuration mismatch")
                startup_log = (work / "engine.log").read_text(errors="replace")
                if "cudaHostRegister PORTABLE ok" not in startup_log:
                    raise Stop("missing arena registration evidence in private startup log")
                run = {"setting": setting, "engine": eng, "load_s": time.monotonic() - started,
                       "artifact_hashes": {"binary": sha(work / "strata"), "vision_binary": sha(work / "strata-vision"),
                                           "config": sha(work / "config.json"), "model": model_identity(cfg)},
                       "registration_lines": [line for line in startup_log.splitlines() if re.search(r"pinned|registered|arena", line, re.I)],
                       "load_cache_state": "OS cache not flushed; first observed load / subsequent reload, not proven cold",
                       "requests": [], "groups": []}
                result["runs"].append(run)
                for rep in range(args.repeats):
                    for words in (128, args.long_words):
                        for streams in ((1, 2) if rep % 2 == 0 else (2, 1)):
                            # Distinct beginning avoids treating this as a shared-prefix dual test.
                            contents = [f"Calibration {idx}-{rep}-{words}-{streams}-{slot}. Background: " +
                                        ((" alpha beta gamma delta" if slot == 0 else " one two three four") * (words // 4)) +
                                        "\nList integers from 1 through 1000, separated by commas. No prose."
                                        for slot in range(streams)]
                            for state in ("fresh", "warm"):
                                monitor.phase = f"run-{idx:02}-r{rep}-{words}-{streams}-{state}"
                                barrier = threading.Barrier(streams)
                                with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
                                    rows = list(pool.map(lambda c: request(base, c, args.max_tokens, barrier), contents))
                                monitor.check()
                                for row in rows:
                                    if not re.search(r"\d", row["output_preview"]):
                                        raise Stop("integer fixture correctness failed")
                                overlap = max(0, min(r["ended_monotonic"] for r in rows) - max(r["first_monotonic"] for r in rows))
                                group = {"phase": monitor.phase, "streams": streams, "state": state,
                                         "overlap_decode_s": overlap,
                                         "aggregate_wall_tps": sum(r["usage"]["completion_tokens"] for r in rows) /
                                            (max(r["ended_monotonic"] for r in rows) - min(r["started_monotonic"] for r in rows)),
                                         "requests": rows}
                                if streams == 2 and overlap <= 0:
                                    raise Stop("dual fixture has no overlapping output intervals")
                                run["groups"].append(group)
                                run["requests"].extend(rows)
                                record(work / "measurements.json", run)
                if args.reserve_gates:
                    monitor.phase = f"run-{idx:02}-vision"
                    image = request(base, png_fixture(), cap=96)
                    if not all(k in image["output_preview"].lower() for k in ("red", "circle", "blue")):
                        raise Stop("vision correctness gate failed")
                    run["vision"] = image
                    # Full-context / park/handoff need native fixtures; never pretend HTTP warm repeats prove them.
                    raise Stop("full-context pressure/handoff gate not yet supplied; lower reserves prohibited")
                run["summary"] = summarize(run["requests"])
                metrics = http(base + "/metrics")
                run["native_request_metrics"] = metrics.get("requests", [])
                run["conversation_cache"] = metrics.get("conversation_cache")
                close_process(p); p = None; monitor.process = None
                wait_vram(monitor)
                # Preserve bounded native timing/hit/PCIe/draft lines, excluding any unrelated production logs.
                native_text = (work / "engine.log").read_text(errors="replace")
                if ERROR_RE.search(native_text):
                    raise Stop("native error after request completion")
                run["native_lines"] = [line for line in native_text.splitlines() if any(k in line for k in
                    ("timing", "draft", "expert cache hit", "PCIe", "pool", "pressure", "park", "restore", "loader", "arena"))]
                samples = [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]
                mine = [s for s in samples if s["phase"].startswith(f"run-{idx:02}")]
                run["min_free_vram_mib"] = min(s["gpu"]["free_mib"] for s in mine)
                run["peak_vram_mib"] = max(s["gpu"]["used_mib"] for s in mine)
                record(work / "measurements.json", run)
                record(out / "result.json", result)
                print(f"COMPLETE {setting}: load {run['load_s']:.1f}s, peak VRAM {run['peak_vram_mib']} MiB", flush=True)
            result["passed"] = True
        except BaseException as exc:
            # Neutral refusals do not export arbitrary stderr/exception payloads from imports.
            reason = f"neutral preflight refused at {stage}; exception detail not exported" if neutral else str(exc)
            result["stop"] = {"type": type(exc).__name__, "reason": reason, "memory": memory()}
            print("STOP:", reason, flush=True)
        finally:
            # Cleanup failures/signals cannot prevent EOF from triggering the independent owner.
            try:
                if monitor is not None:
                    monitor.process = None
                    if monitor.thread.ident is not None:
                        monitor.stop()
            finally:
                if owner is not None:
                    try:
                        state = owner.finish()
                        result["restored"] = state["phase"] == "restored"
                        if state.get("trigger") != "EOFError: coordinator transport disconnected":
                            result["passed"] = False
                            result["lifecycle_stop"] = state.get("trigger", "unexpected owner exit")
                        print("LIFECYCLE:", state["phase"], flush=True)
                    except BaseException as exc:
                        result["restoration_error"] = str(exc)
                        print("RESTORATION BLOCKED:", exc, flush=True)
                record(out / "result.json", result)
    return 0 if result["passed"] and result.get("restored") else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", help="explicit execution opt-in; default prints plan only")
    ap.add_argument("--preflight-only", action="store_true",
                    help="requires --run; prepare/check every arm, GET readiness only, never start a window")
    ap.add_argument("--tag", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ap.add_argument("--fractions", default="baseline,0.10,0.40,0.55,0.55,0.40,0.10,0.25")
    ap.add_argument("--workers", default="11")
    ap.add_argument("--reserves", default="2048")
    ap.add_argument("--reserve-gates", action="store_true")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--long-words", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--window-minutes", type=int, default=30)
    ap.add_argument("--port", type=int, default=18089)
    args = ap.parse_args(argv)
    if args.preflight_only and not args.run:
        ap.error("--preflight-only requires explicit --run; otherwise use the ordinary dry-run plan")
    if args.preflight_only and args.reserve_gates:
        ap.error("--preflight-only cannot run --reserve-gates or authorize lower reserves")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.tag):
        ap.error("unsafe tag")
    schedule = plan(args)
    if args.preflight_only:
        schedule.update(mode="preflight-only", production_mutations="none; preparation and GET readiness only",
                        sequence="prepare/check all arms; no model requests or measurements")
    print(json.dumps(schedule, indent=2))
    return live(args, schedule) if args.run else 0


if __name__ == "__main__":
    # Internal entry points do not imply live authorization. Restoration is owner-guarded;
    # launch-check is a no-model probe unless invoked with --exec by the private owner.
    if len(sys.argv) > 1 and sys.argv[1] == "--unload-journal":
        raise SystemExit(unload_original(sys.argv[2]))
    if len(sys.argv) > 1 and sys.argv[1] == "--restore-journal":
        raise SystemExit(restore_original(sys.argv[2]))
    if len(sys.argv) > 1 and sys.argv[1] == "--launch-check":
        execute = sys.argv[6:] == ["--exec"]
        try:
            code = launch_check(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], execute=execute)
        except Exception:
            if not execute:
                # Structured refusal facts for neutral evidence; never export env/config/stderr.
                print(json.dumps({"uid": os.getuid(), "gid": os.getgid(), "python": sys.executable,
                                  "memlock": list(resource.getrlimit(resource.RLIMIT_MEMLOCK))}))
            raise
        raise SystemExit(code)
    raise SystemExit(main())
