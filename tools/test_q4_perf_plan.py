#!/usr/bin/env python3
"""Offline tests for the Q4_K_XL plan/report helper."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q4_perf_plan as q4

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = ROOT / "tools/testdata/q4_perf/sanitized_q4_config.json"


class DryRunTests(unittest.TestCase):
    def test_dry_run_is_offline_and_contains_matrix(self):
        forbidden = AssertionError("dry run attempted a side effect")
        with patch.object(q4, "subprocess", create=True) as sp, \
             patch.object(q4, "socket", create=True) as sock, \
             patch.object(q4, "os", wraps=os) as os_mod, \
             patch.object(urllib.request, "urlopen", side_effect=forbidden), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            sp.Popen.side_effect = forbidden
            sp.run.side_effect = forbidden
            sp.check_output.side_effect = forbidden
            sock.socket.side_effect = forbidden
            os_mod.system.side_effect = forbidden
            self.assertEqual(q4.main(["--config", str(FIXTURE_CONFIG), "--dry-run"]), 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan["mode"], "dry-run-offline-plan")
        self.assertIn("no engine launch", plan["safety"])
        self.assertIsNotNone(plan["identities"]["config"]["sha256"])
        self.assertIsNotNone(plan["identities"]["executable"]["sha256"])
        self.assertIn("semantic_sha256", plan["identities"])
        workloads = {row["id"]: row for row in plan["workloads"]}
        for n in (31, 64, 66, 256, 512, 1024, 2048):
            self.assertEqual(workloads[f"residual-{n}"]["new_prompt_tokens"], n)
        self.assertEqual(workloads["prefill-8192"]["decode_tokens"], 256)
        self.assertEqual(workloads["prefill-32768"]["decode_tokens"], 256)
        self.assertEqual(workloads["decode-256"]["decode_tokens"], 256)
        self.assertTrue(all("overlap-2" in row["slots"] for row in plan["workloads"]))
        self.assertIn("bounded", json.dumps(plan["workloads"]))
        self.assertIn("MemAvailable_floor_bytes", plan["stop_gates"]["memory"])

    def test_fresh_cli_dry_run_creates_no_files_or_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools"
            data = tools / "testdata/q4_perf"
            data.mkdir(parents=True)
            shutil.copy2(ROOT / "tools/q4_perf_plan.py", tools / "q4_perf_plan.py")
            shutil.copy2(FIXTURE_CONFIG, data / "sanitized_q4_config.json")
            shutil.copy2(ROOT / "tools/testdata/q4_perf/fake-strata", data / "fake-strata")
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            result = subprocess.run([sys.executable, str(tools / "q4_perf_plan.py"), "--config",
                                     str(data / "sanitized_q4_config.json"), "--dry-run"],
                                    cwd=root, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(str(path.relative_to(root)) for path in root.rglob("*")), before)
            self.assertFalse(json.loads(result.stdout)["matched_ab_result"]["speedup_claim"])

    def test_no_network_or_process_even_when_stdlib_is_patched(self):
        with patch.object(subprocess, "Popen", side_effect=AssertionError("Popen")), \
             patch.object(subprocess, "run", side_effect=AssertionError("run")), \
             patch.object(subprocess, "check_output", side_effect=AssertionError("check_output")), \
             patch.object(socket, "socket", side_effect=AssertionError("socket")), \
             patch.object(urllib.request, "urlopen", side_effect=AssertionError("HTTP")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(q4.main(["--config", str(FIXTURE_CONFIG), "--dry-run"]), 0)

    def test_arms_are_one_change_at_a_time_and_interleaved(self):
        plan = q4.dry_run_report(FIXTURE_CONFIG, repeats=2)
        for arm in plan["arms"]:
            self.assertLessEqual(len(arm["changes"]), 1, arm)
        order = plan["interleaved_order"]
        self.assertGreater(len(order), 0)
        pairs = list(zip(order[0::2], order[1::2]))
        self.assertTrue(all({a["arm"], b["arm"]} == {"baseline", b["arm"] if a["arm"] == "baseline" else a["arm"]}
                            for a, b in pairs[:20]))
        self.assertEqual(pairs[0][0]["arm"], "baseline")
        # Repeat 2 reverses the first A/B pair so time trends do not always favor the same side.
        first_repeat2 = next(row for row in order if row["repeat"] == 2)
        self.assertNotEqual(first_repeat2["arm"], "baseline")

    def test_historical_observations_are_not_matched_ab_or_speedup(self):
        plan = q4.dry_run_report(FIXTURE_CONFIG)
        self.assertTrue(plan["observed_historical_traffic"]["not_matched_ab"])
        self.assertEqual(plan["matched_ab_result"]["status"], "not_run")
        self.assertIsNone(plan["matched_ab_result"]["speedup_claim"])
        self.assertIn("never claims speedup", plan["matched_ab_result"]["note"])


class ParserTests(unittest.TestCase):
    def parse(self, text: str) -> dict:
        return q4.LogParser().parse_text(text)

    def test_parses_real_request_done_and_counters(self):
        text = """\
DONE 20 20166 790.0 640.0 stop 7 12 20135 60 100 3 4 128.5 31 25
strata serve: prompt 20166 tokens = 20135 reused + 31 read in 790 ms (39.2 tok/s), 20 generated in 640 ms (31.2 tok/s), drafts accepted 7 of 12, 2 checkpoints
strata serve: decode expert cache hit rate: 60.0% (60 hits / 100 lookups); 25 more read by the GPU over PCIe or from another GPU (20.0% of all 125 routed)
strata batch_mtp_stats slot=1 windows=3 offered=34 accepted=29 rejected=5 discarded=0 fallback_attempts=1 fallback_incoherent=0 fallback_not_ready=0 fallback_limits=0 fallback_capacity=0 fallback_reserve=0
"""
        report = self.parse(text)
        self.assertEqual(report["done"][0]["prompt_read"], 31)
        self.assertAlmostEqual(report["done"][0]["pcie_share"], 0.2)
        self.assertEqual(report["requests"][0]["prompt_read_tokens"], 31)
        self.assertEqual(report["batch_mtp"][0]["accepted"], 29)
        self.assertIsNone(report["speedup_claim"])

    def test_cumulative_expert_tiers_are_reported_as_deltas(self):
        text = """\
strata serve: expert tiers: GPU 10 hits this request; since the start RAM 100 blobs, files 50 blobs 1000.0 MB read
strata serve: expert tiers: GPU 12 hits this request; since the start RAM 130 blobs, files 55 blobs 1125.0 MB read
DONE 1 1 1.0 1.0 stop
"""
        report = self.parse(text)
        self.assertEqual(report["expert_tiers"][0]["delta_since_previous_line"]["file_blobs"], 50)
        self.assertEqual(report["expert_tiers"][1]["delta_since_previous_line"]["file_blobs"], 5)
        self.assertEqual(report["expert_tiers"][1]["delta_since_previous_line"]["logical_mb"], 125.0)
        self.assertTrue(report["expert_tiers"][1]["logical_blob_bytes_only"])

    def test_physical_read_bytes_are_separate_from_logical_blob_bytes(self):
        report = self.parse("""\
strata serve: file tier I/O this request: the OS read 12.5 MB from storage (2 major faults) for 900.0 MB of expert reads; page cache held 887.5 MB at the read, not 12.5 MB
DONE 1 1 1.0 1.0 stop
""")
        row = report["file_io"][0]
        self.assertEqual(row["physical_read_bytes"], 12_500_000)
        self.assertEqual(row["logical_blob_bytes"], 900_000_000)
        self.assertNotEqual(row["physical_read_bytes"], row["logical_blob_bytes"])

    def test_missing_and_truncated_logs_warn_without_inventing_values(self):
        missing = self.parse("strata serve: unrelated startup line\n")
        self.assertIn("no complete request/DONE lines found", missing["warnings"])
        truncated = self.parse("strata serve: prompt 66 tokens = 0 reused + 66 read in")
        self.assertIn("truncated partial line", " ".join(truncated["warnings"]))
        self.assertEqual(truncated["requests"], [])

    def test_overlapping_timings_are_not_summed_as_disjoint(self):
        line = ("strata batch: 6 windows, avg 3.00 rows, 74.65 ms/window = run 63.93 "
                "(CUDA0 GPU-reach wait 29.25 + CPU experts 63.93) + commit 1.00 + emit 0.50 + adapt wait 6.51; "
                "per stage (ms/window): CUDA0 wait 29.25 + pool 63.93; per layer-window: CPU experts 2.00, "
                "VRAM hits 3.00, PCIe 4.00; routed entries per window: VRAM 16.3 (16.3%), PCIe 40.0 (40.0%), "
                "CPU 43.7 (43.7%); 120.0 rows/s over 448 ms of wall time (admissions included)\n")
        report = self.parse(line + "DONE 1 1 1.0 1.0 stop\n")
        batch = report["batch_windows"][0]
        self.assertTrue(batch["components_overlap"])
        self.assertIn("gpu_wait_ms", batch["do_not_sum_as_disjoint"])
        self.assertGreater(batch["gpu_wait"], 0)
        self.assertGreater(batch["cpu_experts"], 0)
        self.assertIn("not added as disjoint", report["note"])

    def test_prefill_loan_refill_stage_memory_high_water(self):
        text = """\
strata prefill timing: 32768 tokens, GPU timeline 6500 ms, wall 7100 ms, host staging 400 ms: GEMM 3000 (46.2%) MoE 2500 (38.5%) route 1000 (15.4%)
strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) 55 ms, waiting for each chunk 120 ms, after each chunk (the draft layer, progress) 25 ms, PLE 40 ms
strata serve: stage 0: 12 windows; per window: wait for the GPU 3.000 ms, pool + plan 5.500 ms, host staging 1.250 ms, commit 0.750 ms
{"MemAvailable": 20000000000, "SwapFree": 3000000000, "VmHWM": 123456, "VmRSS": 100000}
{"MemAvailable": 19000000000, "SwapFree": 3500000000, "VmHWM": 234567, "VmRSS": 120000}
DONE 1 32768 7100.0 30.0 stop
"""
        report = self.parse(text)
        self.assertEqual(report["prefill"][0]["tokens"], 32768)
        self.assertEqual(report["prefill"][0]["host_breakdown"]["waiting"], 120)
        self.assertEqual(report["stage_windows"][0]["pool"], 5.5)
        self.assertEqual(report["memory_high_water"]["VmHWM"], 234567)
        self.assertEqual(report["memory_high_water"]["SwapFree"], 3500000000)

    def test_cli_can_emit_plan_plus_parsed_report(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("DONE 1 31 790.0 10.0 stop 0 0 0\n")
            log = f.name
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(q4.main(["--config", str(FIXTURE_CONFIG), "--parse-log", log]), 0)
            payload = json.loads(out.getvalue())
            self.assertIn("plan", payload)
            self.assertEqual(payload["parsed_report"]["done"][0]["prompt"], 31)
            self.assertFalse(payload["parsed_report"]["matched_ab"])
        finally:
            Path(log).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
