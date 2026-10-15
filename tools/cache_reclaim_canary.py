#!/usr/bin/env python3
"""str-s1w.14 GPU canary: opt-in expert-cache reclaim/regrow in resident-complement mode.

Private native subprocess only (never a running service). The config's model/residency args are kept and
--serve --vram-elastic is added; STRATA_LAZY_CACHE_RECLAIM=1 enables the commands. Steps:

  1. greedy references: N prompts, solo GEN, fixed max_new
  2. CACHE_RECLAIM <keep>          -> CACHE line: expert_slots == keep, generation advanced, no ERR
  3. the same prompts              -> token IDs identical to (1): displaced experts are served by RAM/file tiers
  4. VRAM                          -> ERR (resident mode refuses; a reclaim also owns the cache)
  5. CACHE_REGROW                  -> CACHE line: expert_slots == full, unrefilled=0, no degraded flag
  6. the same prompts              -> identical again
  7. a second reclaim/regrow cycle at a smaller keep, then the prompts once more

Greedy decode is exact against itself only when the expert placement does not change the arithmetic; the CPU, PCIe
and VRAM paths of an expert compute the same blob with different kernels, so a mismatch is reported with the first
divergent position and the run is recorded as a FINDING, not silently passed.

  cache_reclaim_canary.py --exe build/strata --config model.json --output /tmp/reclaim.json [--keep 1200 --keep2 900]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_kv_smoke as smoke  # noqa: E402  (NativeEngine, parse_line, tokenizer, resolve_path)

PROMPTS = [
    "Explain how a hash table handles collisions, in two short paragraphs.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List five differences between TCP and UDP.",
]


def engine_command(cfg, exe, context):
    cwd = str(Path(cfg.get("cwd") or os.getcwd()).expanduser().resolve())
    drop_valued = {"--max-new", "--batch", "--batch-groups", "--conversation-cache-mib", "--conversation-cache-slots",
                   "--conversation-cache-min-free-mib", "--max-context", "--kv-resident", "--adapt-every",
                   "--adapt-swaps", "--peer-adapt-swaps"}
    drop_flags = {"--serve", "--kv-unified", "--batch-mtp", "--vram-elastic"}   # --no-kv-grow stays: elastic KV refuses reclaim
    a, args, i = list(cfg["args"]), [], 0
    while i < len(a):
        if a[i] in drop_valued:
            i += 2
        elif a[i] in drop_flags:
            i += 1
        else:
            args.append(a[i])
            i += 1
    # solo engine: the reclaim command refuses while batch slots decode, and the canary is about the cache, not slots.
    # Adaptive swaps off so the residency the canary checks is the one it set (no rebalancing between steps).
    args += ["--serve", "--vram-elastic", "--max-context", str(context), "--max-new", "64",
             "--adapt-every", "1000000", "--adapt-swaps", "0", "--conversation-cache-mib", "0"]
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (cfg.get("env") or {}).items()})
    env["STRATA_LAZY_CACHE_RECLAIM"] = "1"
    # Route tail skip (on by default on CUDA) drops a missed low-rank expert, so the answer depends on which experts
    # are cached: a reclaim legitimately changes it.  Off here, so the canary's parity measures the tier substitution
    # (the displaced experts served from RAM/file instead of VRAM), not a policy that keys on the cache size.
    env["STRATA_ROUTE_TAIL_SKIP"] = "0"
    env.pop("STRATA_PREFILL_SLOT_SHARE", None)
    dirs = [smoke.resolve_path(d, cwd) for d in cfg.get("lib_dirs") or []]
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    return [str(Path(exe).resolve()), *args], cwd, env


class Canary:
    def __init__(self, engine, evidence, timeout):
        self.e, self.ev, self.timeout = engine, evidence, timeout

    def lines_until(self, stop, allow_error=False):
        deadline = time.monotonic() + self.timeout
        out = []
        while True:
            ev = self.e.next_event(deadline, allow_error=allow_error)
            out.append(ev)
            if stop(ev):
                return out

    def gen(self, ids, max_new):
        self.e.stage = f"gen-{len(self.ev['generations'])}"
        self.e.send(f"GEN {max_new} " + " ".join(map(str, ids)))
        evs = self.lines_until(lambda ev: ev["kind"] in ("DONE", "ERR"), allow_error=True)
        smoke.require(evs[-1]["kind"] == "DONE", f"GEN failed: {evs[-1].get('raw')}")
        toks = [ev["token"] for ev in evs if ev["kind"] == "T"]
        self.ev["generations"].append({"stage": self.e.stage, "n": len(toks), "done": evs[-1]["raw"]})
        return toks

    def command(self, line, expect_ok):
        self.e.stage = line
        self.e.send(line)
        evs = self.lines_until(lambda ev: ev["kind"] == "ERR" or ev["raw"].startswith(("CACHE ", "VRAM ")),
                               allow_error=True)
        last = evs[-1]["raw"]
        rec = {"command": line, "reply": last, "ok": not last.startswith("ERR")}
        if last.startswith("CACHE "):
            rec["fields"] = dict(kv.split("=", 1) for kv in last.split()[1:] if "=" in kv)
        self.ev["commands"].append(rec)
        smoke.require(rec["ok"] == expect_ok, f"{line}: expected {'success' if expect_ok else 'ERR'}, got {last!r}")
        return rec


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--keep", type=int, default=1200)
    ap.add_argument("--keep2", type=int, default=900)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--startup-timeout", type=float, default=900)
    ap.add_argument("--stage-timeout", type=float, default=300)
    a = ap.parse_args(argv)
    out = Path(a.output)
    smoke.require(not out.exists(), "output exists")
    cfg = json.loads(Path(a.config).read_text())
    tok = smoke.tokenizer(cfg["tokenizer"])
    cmd, cwd, env = engine_command(cfg, a.exe, a.context)
    ev = {"schema": "cache_reclaim_canary/1", "command": cmd, "env_added": {"STRATA_LAZY_CACHE_RECLAIM": "1"},
          "commands": [], "generations": [], "rounds": {}, "stdout": [], "passed": False}
    engine = smoke.NativeEngine(ev, str(out) + ".stderr.log", cleanup_timeout=30)
    ok = False
    try:
        # startup: INFO then READY (not the unified-KV INFO contract: a solo engine)
        engine.log = open(engine.stderr_path, "xb")
        import subprocess, threading
        engine.p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=engine.log, start_new_session=True)
        ev["pid"] = engine.p.pid
        engine.reader = threading.Thread(target=engine._read_stdout, daemon=True)
        engine.reader.start()
        c = Canary(engine, ev, a.stage_timeout)
        deadline = time.monotonic() + a.startup_timeout
        while True:
            e = engine.next_event(deadline)
            if e["kind"] == "INFO":
                ev.setdefault("info", {}).update(e["fields"])
            if e["kind"] == "READY":
                break
        info = ev.get("info", {})
        full = int(info.get("expert_slots", 0))
        smoke.require(full > a.keep > a.keep2 > 0, f"keep values must be below expert_slots={full}")
        ids = [tok.encode(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          parse_special=True) for p in PROMPTS]

        def round_(name):
            r = [c.gen(x, a.max_new) for x in ids]
            ev["rounds"][name] = r
            return r

        ref = round_("reference")
        steps = [("CACHE_RECLAIM %d" % a.keep, True), ("after-reclaim", None), ("VRAM", False),
                 ("CACHE_REGROW", True), ("after-regrow", None),
                 ("CACHE_RECLAIM %d" % a.keep2, True), ("after-reclaim2", None),
                 ("CACHE_REGROW", True), ("after-regrow2", None)]
        gen_before = None
        for name, expect in steps:
            if expect is None:
                got = round_(name)
                div = [first_divergence(x, y) for x, y in zip(ref, got)]
                ev.setdefault("parity", {})[name] = div
                continue
            rec = c.command(name, expect)
            f = rec.get("fields") or {}
            if name.startswith("CACHE_RECLAIM"):
                keep = int(name.split()[1])
                # the cache shrinks in whole VMM segments: it keeps at least `keep`, and less than one segment more
                got = int(f["expert_slots"])
                smoke.require(keep <= got < keep + 140, f"reclaim mapped {got}, wanted {keep} (+ < one 512 MiB segment, ~128 slots)")
                smoke.require(int(f["unrefilled"]) == full - got, f"reclaim lists {f['unrefilled']} displaced, want {full - got}")
                smoke.require("degraded" not in f, f"reclaim reported degraded: {rec['reply']}")
            if name == "CACHE_REGROW":
                # A regrow maps what the driver can give back: other allocations may have taken part of the VRAM the
                # reclaim released.  Then the command must SAY so (degraded=1) and account for every expert it could
                # not refill (unrefilled == full - mapped, all on the file/RAM tier).  A full regrow must be clean.
                got = int(f["expert_slots"])
                smoke.require(got > a.keep, f"regrow mapped {got}, not above the reclaimed {a.keep}")
                if got == full:
                    smoke.require(int(f["unrefilled"]) == 0 and "degraded" not in f, f"full regrow degraded: {rec['reply']}")
                else:
                    smoke.require("degraded" in f and int(f["unrefilled"]) == full - got,
                                  f"partial regrow must report degraded with unrefilled == {full - got}: {rec['reply']}")
                ev.setdefault("regrow", []).append({"mapped": got, "full": full, "unrefilled": int(f["unrefilled"])})
            if "generation" in f:
                g = int(f["generation"])
                smoke.require(gen_before is None or g > gen_before, "generation did not advance")
                gen_before = g
        ev["exact_parity"] = all(d is None for v in ev["parity"].values() for d in v)
        ok = True
    except Exception as exc:
        ev["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            engine.close(ok)
        except Exception as exc:
            ev.setdefault("cleanup_error", str(exc))
        ev["passed"] = ok
        out.write_text(json.dumps(ev, indent=1) + "\n")
    print(("PASS" if ok else "FAIL") + f": {out} exact_parity={ev.get('exact_parity')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
