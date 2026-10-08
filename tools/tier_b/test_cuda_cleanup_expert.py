"""sta-osh.21 offline expert lifecycle probes; no CUDA library, GPU, or downloads.

Compiles the entire owned source with real IQ API declarations. The dedicated fake
CUDA tracks every resource, ends captures, and injects failures at each constructor,
graph, parity and timing API (including partial output handles). Fake IQ kernels
only write finite constants: these tests do NOT prove CUDA arithmetic or performance.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tools/tier_b/fixtures/expert_cleanup"


@unittest.skipUnless(shutil.which("g++"), "offline expert probes require g++")
class ExpertCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="expert_cleanup_")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.exe = Path(cls.tmp.name) / "expert_cleanup_probe"
        cls.compile_command = [
            "g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
            "-I", str(FIXTURES), "-I", str(ROOT / "include"), "-I", str(ROOT),
            str(FIXTURES / "expert_cleanup_probe.cpp"), "-o", str(cls.exe),
        ]
        result = subprocess.run(cls.compile_command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(f"compile exit={result.returncode}:\n{result.stderr}")

    def probe(self, phase):
        result = subprocess.run([str(self.exe), phase], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"PROBE {phase} ", result.stdout)
        self.assertIn("resources=0", result.stdout)
        # Preserve exact probe totals in verbose unittest evidence.
        print(next(line for line in result.stdout.splitlines() if line.startswith("PROBE ")))
        return result

    def test_constructor_every_allocation_memset_copy_and_partial_handle(self):
        self.probe("constructor")

    def test_graph_slots_both_arms_partial_handles_and_active_capture(self):
        self.probe("graphs")

    def test_capture_cost_partial_graph_and_exec(self):
        self.probe("capture")

    def test_parity_errors_restore_all_initial_selectors(self):
        self.probe("parity")

    def test_timing_errors_release_both_graph_arms(self):
        self.probe("timing")

    def test_graph_correctness_readback_errors_release_both_arms(self):
        self.probe("graph_check")

    def test_original_error_survives_abandoned_capture_cleanup_failure(self):
        self.probe("dual")

    def test_all_refused_releases_are_attempted_and_selectors_restore(self):
        result = self.probe("refused")
        self.assertIn("refused_resources=20", result.stdout)
        self.assertEqual(result.stderr.count("CUDA cleanup error"), 20)

    def test_blob_validation_exception_unwinds_partial_constructor(self):
        self.probe("blob")

    def test_top_level_failure_exit_and_unwind(self):
        result = self.probe("entry")
        self.assertIn("tier_b_expert_bench FAIL: CUDA error", result.stderr)
        self.assertIn("tier_b_expert_bench FAIL: unknown exception", result.stderr)
        self.assertIn("tier_b_expert_bench: 0 parity failures, 7 timed pairs", result.stdout)

    def test_cleanup_errors_are_noexcept_and_do_not_stop_other_releases(self):
        result = self.probe("cleanup")
        self.assertEqual(result.stderr.count("CUDA cleanup error"), 10)

    def test_move_ownership_releases_exactly_once(self):
        self.probe("move")


if __name__ == "__main__":
    unittest.main()
