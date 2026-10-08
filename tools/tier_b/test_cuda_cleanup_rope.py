"""sta-osh.20: compile the entire RoPE benchmark with fake CUDA; never load CUDA.

Reproduce: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tools.tier_b.test_cuda_cleanup_rope -v
These probes validate ownership/error paths, not real kernel arithmetic, parity or timing.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tools/tier_b/fixtures/rope_cleanup"
SOURCE = ROOT / "src/kernels/tier_b_rope_bench.cpp"


@unittest.skipUnless(shutil.which("c++"), "offline C++ compiler unavailable")
class RopeCleanupProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="sta_osh20_rope_cleanup_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.exe = Path(cls.temp.name) / "rope_cleanup_probe"
        cls.compile_command = [
            "c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
            "-I", str(FIXTURES / "sta_osh20_fake_cuda"), "-I", str(ROOT / "include"),
            f'-DROPE_BENCH_SOURCE="{SOURCE}"', str(FIXTURES / "sta_osh20_probe.cpp"),
            "-o", str(cls.exe),
        ]
        result = subprocess.run(cls.compile_command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(f"offline compile exit={result.returncode}: {result.stderr}")

    def probe(self, *, fail="", second="", mode=None, flags=(), options=(), expected=0):
        env = {k: v for k, v in os.environ.items() if not k.startswith("ROPE_PROBE_")}
        env.pop("STRATA_ROPE_TABLE", None)
        if mode is not None:
            env["STRATA_ROPE_TABLE"] = mode
        env.update(ROPE_PROBE_FAIL=fail, ROPE_PROBE_SECOND_FAIL=second)
        env.update({key: "1" for key in options})
        run = subprocess.run([str(self.exe), "10", "1", "2", "1", "1", *flags],
                             env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, expected, run.stdout + run.stderr)
        match = re.search(r"^PROBE (.*)$", run.stdout, re.M)
        self.assertIsNotNone(match, "wrapper did not regain control (exit/terminate?): " + run.stderr)
        report = {k: int(v) for k, v in (item.split("=") for item in match[1].split())}
        self.assertEqual(report["rc"], expected)
        for field in ("unattempted", "captures", "table", "violations"):
            self.assertEqual(report[field], 0, f"{field}: {run.stdout}\n{run.stderr}")
        self.assertEqual(report["env"], 1, "STRATA_ROPE_TABLE not restored")
        if fail:
            self.assertIn("INJECT " + fail, run.stderr)
            self.assertEqual(report["injected"], 1)
        if second:
            self.assertIn("INJECT " + second, run.stderr)
        calls = {op: int(n) for op, n in re.findall(r"^CALL (\w+) (\d+)$", run.stdout, re.M)}
        created = {kind: int(n) for kind, n in re.findall(r"^CREATED (\w+) (\d+)$", run.stdout, re.M)}
        return run, report, calls, created

    def test_success_graphs_and_no_graph_and_mode_restoration(self):
        for flags in ((), ("--no-graph",)):
            for mode in (None, "", "0", "original-value"):
                with self.subTest(flags=flags, mode=mode):
                    run, report, _, created = self.probe(flags=flags, mode=mode)
                    self.assertEqual(report["live"], 0)
                    self.assertIn("12 cases, 12 timed, 0 gate/replay failures", run.stdout)
                    self.assertEqual(created["allocation"], 7)
                    self.assertEqual(created["event"], 2)
                    self.assertEqual(created["stream"], 1)
                    self.assertEqual(created.get("graph", 0), 0 if flags else 24)
                    self.assertEqual(created.get("exec", 0), 0 if flags else 24)

    def test_injected_operations_unwind_all_partial_owners(self):
        _, _, calls, _ = self.probe()
        # Every acquisition/capture/instantiate occurrence (including both arms, later shapes,
        # and table mode); first/second/last for dataflow and event-window operations.
        exhaustive = {"malloc", "event_create", "stream_create", "begin", "end", "instantiate",
                      "capture_norm", "capture_rope", "capture_fused", "capture_pack"}
        releases = {"free", "event_destroy", "stream_destroy", "graph_destroy", "exec_destroy"}
        points = []
        for op, count in sorted(calls.items()):
            if op in releases:
                continue
            ordinals = range(1, count + 1) if op in exhaustive else sorted({1, min(2, count), count})
            for n in ordinals:
                point = f"{op}:{n}"
                with self.subTest(point=point):
                    _, report, _, _ = self.probe(fail=point, mode="0", expected=2)
                    self.assertEqual(report["live"], 0)
                points.append(point)
        print(f"RoPE offline fault sweep: {len(points)} injected operation points")

    def test_null_partial_outputs_and_nonstandard_host_exceptions(self):
        for op in ("end", "instantiate"):
            for n in (1, 2):
                with self.subTest(op=op, n=n):
                    _, report, _, _ = self.probe(fail=f"{op}:{n}", expected=2,
                                               options=("ROPE_PROBE_NULL_PARTIAL",))
                    self.assertEqual(report["live"], 0)
        for option in ("ROPE_PROBE_UNKNOWN", "ROPE_PROBE_BAD_ALLOC"):
            with self.subTest(option=option):
                run, report, _, _ = self.probe(fail="capture_norm:1", expected=2, options=(option,))
                self.assertEqual(report["live"], 0)
                self.assertIn("tier_b_rope_bench FAILED:", run.stderr)

    def test_cleanup_errors_attempt_every_remaining_release_and_fail_closed(self):
        _, _, calls, _ = self.probe()
        points = 0
        for op in ("free", "event_destroy", "stream_destroy", "graph_destroy", "exec_destroy"):
            for n in range(1, calls[op] + 1):
                with self.subTest(op=op, n=n):
                    _, report, _, _ = self.probe(fail=f"{op}:{n}", expected=2)
                    # A refused fake release stays live; all handles still get exactly one attempt.
                    self.assertEqual(report["live"], 1)
                points += 1
        # Failure while capturing the second graph, followed by a failing abort EndCapture.
        _, report, _, created = self.probe(fail="capture_norm:1", second="end:2", expected=2)
        self.assertEqual(report["live"], 0)
        self.assertEqual(created["graph"], 2)  # first graph plus aborted second, neither overwritten
        # A failed destructor release must not replace the original kernel error or stop outer cleanup.
        run, report, _, _ = self.probe(fail="capture_norm:1", second="graph_destroy:1", expected=2)
        self.assertEqual(report["live"], 1)
        self.assertIn("rope-fake kernel exception", run.stderr)
        print(f"RoPE offline cleanup sweep: {points} refused releases + 2 dual-failure probes")

    def test_gate_replay_selector_failures_and_early_disabled_exit(self):
        for option in ("ROPE_PROBE_BAD_GATE", "ROPE_PROBE_BAD_REPLAY"):
            with self.subTest(option=option):
                run, report, _, _ = self.probe(options=(option,), expected=1)
                self.assertEqual(report["live"], 0)
                self.assertIn("12 cases, 0 timed, 12 gate/replay failures", run.stdout)
        run, report, _, _ = self.probe(options=("ROPE_PROBE_BAD_TABLE",), expected=2)
        self.assertEqual(report["live"], 0)
        self.assertIn("table path did not select the registered table", run.stderr)
        _, report, _, _ = self.probe(options=("ROPE_PROBE_DISABLED",), mode="original-value", expected=1)
        self.assertEqual(report["handles"], 0)


if __name__ == "__main__":
    unittest.main()
