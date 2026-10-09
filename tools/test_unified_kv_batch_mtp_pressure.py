"""Offline protocol, allocation-geometry and evidence-contract tests for the GPU gate."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_kv_batch_mtp_pressure_smoke as gate
import unified_kv_smoke as common


class BatchMtpEvidenceTests(unittest.TestCase):
    def test_counter_invariants_and_per_attempt_scope(self):
        line = ("strata batch_mtp_stats slot=0 windows=5 offered=4 accepted=2 rejected=1 discarded=1 "
                "fallback_attempts=1 fallback_incoherent=0 fallback_not_ready=0 fallback_limits=0 "
                "fallback_capacity=0 fallback_reserve=1")
        one = gate.parse_batch_stats(line)
        self.assertEqual(one[0]["slot"], 0)
        self.assertEqual(one[0]["fallback_reserve"], 1)
        # Same slot in a later lifecycle is valid: parser scope is a stderr byte range,
        # not a global uniqueness constraint on slot numbers.
        later = gate.parse_batch_stats(line.replace("windows=5", "windows=2")
                                       .replace("offered=4", "offered=2")
                                       .replace("accepted=2", "accepted=1")
                                       .replace("rejected=1", "rejected=0")
                                       .replace("discarded=1", "discarded=1")
                                       .replace("fallback_attempts=1", "fallback_attempts=0")
                                       .replace("fallback_reserve=1", "fallback_reserve=0"))
        self.assertEqual(later[0]["slot"], 0)
        with self.assertRaises(AssertionError):
            gate.parse_batch_stats(line.replace("offered=4", "offered=5"))

    def test_pressure_requires_a_real_offer_from_each_private_slot(self):
        rows = [
            {"slot": 0, "windows": 1, "offered": 0, "accepted": 0, "rejected": 0, "discarded": 0,
             "fallback_attempts": 1, "fallback_incoherent": 0, "fallback_not_ready": 0,
             "fallback_limits": 0, "fallback_capacity": 0, "fallback_reserve": 1},
            {"slot": 1, "windows": 1, "offered": 1, "accepted": 1, "rejected": 0, "discarded": 0,
             "fallback_attempts": 0, "fallback_incoherent": 0, "fallback_not_ready": 0,
             "fallback_limits": 0, "fallback_capacity": 0, "fallback_reserve": 0},
        ]
        with self.assertRaises(AssertionError):
            gate.verify_stats(rows, offer=True, every_slot_offer=True)
        proof = gate.verify_stats(rows, offer=True)
        self.assertEqual(proof["offered"], 1)

    def test_optional_stop_guard_handles_slot_drift_but_pressure_terminal_fails(self):
        reqs = [common.Request("fast", [1], 320, 0), common.Request("slow", [2], 320, 1)]
        for req, count in zip(reqs, (267, 258)):
            req.badm = {"continues": True}
            req.tokens.extend(range(count))
        self.assertTrue(gate.optional_stop_ready(reqs))
        reqs[1].tokens.pop()
        self.assertFalse(gate.optional_stop_ready(reqs))
        reqs[1].tokens.append(257)
        reqs[0].completion = {"kind": "BDONE", "finish": "pressure"}
        self.assertFalse(gate.optional_stop_ready(reqs))

    def test_optional_result_requires_counter_and_no_pressure_terminal(self):
        reqs = [common.Request("a", [1], 320, 0), common.Request("b", [2], 320, 1)]
        for req in reqs:
            req.badm = {"continues": True}
            req.tokens.append(3)
            req.token_events.append({"kind": "BT", "seq": 10 + req.slot})
            req.completion = {"kind": "BDONE", "finish": "cancel"}
        stats = [
            {"slot": 0, "windows": 2, "offered": 1, "accepted": 1, "rejected": 0, "discarded": 0,
             "fallback_attempts": 1, "fallback_incoherent": 0, "fallback_not_ready": 0,
             "fallback_limits": 0, "fallback_capacity": 0, "fallback_reserve": 1},
            {"slot": 1, "windows": 2, "offered": 2, "accepted": 1, "rejected": 1, "discarded": 0,
             "fallback_attempts": 0, "fallback_incoherent": 0, "fallback_not_ready": 0,
             "fallback_limits": 0, "fallback_capacity": 0, "fallback_reserve": 0},
        ]
        diagnostics = gate.parse_diagnostics("")
        diagnostics["batch_mtp_stats"] = stats
        proof = gate.validate_optional_result(reqs, stats, diagnostics)
        self.assertEqual(proof["fallback_slots"], [0])
        reqs[0].completion["finish"] = "pressure"
        with self.assertRaises(AssertionError):
            gate.validate_optional_result(reqs, stats, diagnostics)

    def test_optional_row_fallback_is_not_mandatory_pressure(self):
        text = ("strata batch_mtp_stats slot=1 windows=3 offered=2 accepted=1 rejected=1 discarded=0 "
                "fallback_attempts=1 fallback_incoherent=0 fallback_not_ready=0 fallback_limits=0 "
                "fallback_capacity=0 fallback_reserve=1\n")
        diagnostics = gate.parse_diagnostics(text)
        proof = gate.verify_stats(diagnostics["batch_mtp_stats"], offer=True, fallback_reserve=True)
        self.assertEqual(proof["fallback_reserve"], 1)
        self.assertFalse(any(d["kind"] == "pressure_target_only" for d in diagnostics["pressure"]))
        # A mandatory shortage has a different protocol terminal and a positive park.
        with self.assertRaises(AssertionError):
            gate.require_pressure_terminal(type("Req", (), {
                "completion": {"kind": "BDONE", "finish": "length"}, "tokens": [7], "cap": 8})())

    def test_pressure_restore_retains_all_ids_but_leaves_last_unfed(self):
        original, returned = [10, 11, 12], [20, 21, 22]
        replay, consumed = gate.pressure_replay_ids(original, returned)
        self.assertEqual(replay, [10, 11, 12, 20, 21, 22])
        self.assertEqual(consumed, len(replay) - 1)
        self.assertEqual(replay[-1], returned[-1])
        serialized = json.loads(json.dumps({"prompt_ids": replay, "expected_reuse": consumed}))
        self.assertEqual(serialized["prompt_ids"], replay)
        self.assertEqual(serialized["expected_reuse"], len(replay) - 1)
        diagnostics = gate.parse_diagnostics(
            "strata serve: pressure parked target-only 5 tokens; parked=1 bytes=2048 evictions=0 snapshot_bytes=2048\n"
            "conversation cache: restored 5 tokens (live) in 1.0 ms; parked=1 bytes=2048\n"
            "strata serve: TARGET_ONLY restore: private MTP proposals suppressed until full replay\n"
            "strata serve: TARGET_ONLY decode: T=1, MTP/suffix proposals disabled\n")
        proof = gate.verify_target_only_resume("DONE 2 6 1.0 2.0 length 0 0 5", diagnostics, 5, 5)
        self.assertTrue(proof["positive_target_only_restore"])
        self.assertEqual(proof["draft_offers"], 0)
        with self.assertRaises(AssertionError):
            gate.verify_target_only_resume("DONE 2 6 1.0 2.0 length 0 0 5", diagnostics, 5, 4)
        with self.assertRaises(AssertionError):
            gate.pressure_replay_ids(original, [])

    def test_pressure_vs_eos_length_and_dual_slot_overlap_protocol(self):
        reqs = [common.Request("a", [1, 2, 3], 8, 0), common.Request("b", [4, 5, 6], 8, 1)]
        protocol = common.Protocol(reqs)
        seq = 0

        def feed(kind, **fields):
            nonlocal seq
            seq += 1
            event = {"kind": kind, "seq": seq, "wall_s": float(seq), **fields}
            protocol.consume(event)

        feed("T", token=50)
        feed("DONE", count=1, prompt_count=3, prompt_ms=1.0, decode_ms=1.0, finish="length")
        feed("BADM", slot=0, continues=True)
        feed("T", token=60)
        feed("DONE", count=1, prompt_count=3, prompt_ms=1.0, decode_ms=1.0, finish="length")
        feed("BADM", slot=1, continues=True)
        feed("BT", slot=0, token=51)
        feed("BT", slot=1, token=61)
        feed("BDONE", slot=0, count=2, finish="pressure", decode_ms=1.0)
        self.assertEqual(set(protocol.active), {1})
        feed("BDONE", slot=1, count=2, finish="cancel", decode_ms=1.0)
        self.assertTrue(protocol.finished)
        self.assertTrue(gate.pressure_overlap_proof(reqs)["actual_dual_slot_progress"])
        self.assertEqual(gate.require_pressure_terminal(reqs[0])["finish"], "pressure")
        # EOS/length is never interchangeable with real pressure.
        reqs[0].completion = {"kind": "BDONE", "finish": "length"}
        with self.assertRaises(AssertionError):
            gate.require_pressure_terminal(reqs[0])

    def test_capacity_fixture_exceeds_aggregate_but_individuals_fit(self):
        prompts = (list(range(1, 1537)), list(range(5000, 6540)))
        plan = gate.pressure_capacity_plan(prompts, [768, 768], 4096)
        self.assertTrue(plan["combined_individual_histories_exceed_backing"])
        self.assertLessEqual(plan["initial_pages_total"], plan["pool_pages"])
        self.assertGreater(plan["individual_final_pages_total"], plan["pool_pages"])
        with self.assertRaises(AssertionError):
            gate.pressure_capacity_plan((prompts[0], prompts[1]), [3000, 3000], 4096)

    def test_optional_boundary_fixture_fits_initial_headroom_with_one_page_left(self):
        prompts = (list(range(1, 1788)), list(range(5000, 6791)))
        plan = gate.pressure_capacity_plan(prompts, [320, 320], 4096)
        self.assertEqual([len(p) % gate.PAGE for p in prompts], [3, 3])
        self.assertEqual(plan["initial_pages_total"], 1023)
        self.assertEqual(plan["pool_pages"], 1024)
        self.assertGreater(plan["individual_final_pages_total"], 1024)

    def test_ring_helper_dimensions_at_bounded_and_full_context_boundaries(self):
        bounded = gate.ring_capacity_from_source(128, 2, 4096)
        self.assertEqual(bounded["requested_private_ring_cells"], 200)
        self.assertEqual(bounded["allocated_page_rounded_ring_cells"], 200)
        self.assertTrue(bounded["bounded_ring"])
        full = gate.ring_capacity_from_source(4096, 2, 4096)
        self.assertEqual(full["requested_private_ring_cells"], -1)
        self.assertFalse(full["bounded_ring"])
        above = gate.ring_capacity_from_source(4097, 2, 4096)
        self.assertFalse(above["bounded_ring"])

    def test_settings_explicitly_enable_batch_mtp_and_remove_test_hooks(self):
        cfg = {"args": ["--mtp", "/model/draft", "--mtp-window=4096", "--batch-mtp=true"],
               "cwd": "/tmp", "env": {"STRATA_BATCH_MTP_TEST_PROPOSALS": "1"},
               "backend": "cuda", "gpu": ["0"]}
        command, _, env, _, contract = gate.settings(cfg, Path("/tmp/strata"), 4096, 4096, "0", 128, 2048)
        self.assertEqual(command.count("--batch-mtp"), 1)
        self.assertEqual(command.count("--mtp-window"), 1)
        self.assertEqual(command[command.index("--mtp-window") + 1], "128")
        self.assertEqual(command.count("--mtp"), 1)
        self.assertEqual(env["STRATA_BATCH_MTP"], "1")
        self.assertEqual(env["STRATA_BATCH_DECODE_SHARE"], "0")
        self.assertNotIn("STRATA_BATCH_MTP_TEST_PROPOSALS", env)
        self.assertEqual(command[command.index("--conversation-cache-mib") + 1], "4096")
        self.assertEqual(command[command.index("--mtp-max-t") + 1], "2")
        self.assertEqual(command[command.index("--spec") + 1], "2")
        self.assertEqual(contract["main_max_drafts_per_window"], 1)
        self.assertEqual(gate.validate_main_mtp_contract(command)["main_max_drafts_per_window"], 1)
        broken = command.copy()
        broken[broken.index("--mtp-max-t") + 1] = "1"
        with self.assertRaises(AssertionError):
            gate.validate_main_mtp_contract(broken)

    def test_source_references_are_process_isolated_and_handoff_reference_is_late(self):
        reference = {"label": "reference", "finalized": True, "returncode": 0, "pid": 100,
                    "cleanup": {"reader_stopped": True, "writer_stopped": True}}
        isolation = gate.process_history_isolation("source", [reference])
        self.assertEqual(isolation["prior_reference_pid"], 100)
        self.assertIn("fresh subprocess", isolation["history_isolation"])
        with self.assertRaises(AssertionError):
            gate.process_history_isolation("source", [{**reference, "returncode": None}])
        handoff = gate.process_history_isolation("handoff", [])
        self.assertIn("after continuation", handoff["history_isolation"])
        with self.assertRaises(AssertionError):
            gate.process_history_isolation("handoff", [reference])

    def test_deadlines_are_finite_positive_and_bounded(self):
        gate.validate_deadlines(900, 1800, 15)
        for values in ((0, 2, 3), (1, float("inf"), 3), (1, 7201, 3)):
            with self.assertRaises(AssertionError):
                gate.validate_deadlines(*values)

    def test_owned_startup_failures_close_the_only_worker_and_all_pipes(self):
        class TrackingEngine(gate.BatchMtpEngine):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.launch_calls = 0
                self.close_calls = 0

            def start(self, *args, **kwargs):
                self.launch_calls += 1
                return super().start(*args, **kwargs)

            def close(self, success):
                self.close_calls += 1
                return super().close(success)

        class Suite:
            def __init__(self, fail=False):
                self.fail = fail

            def save(self):
                if self.fail:
                    raise RuntimeError("synthetic evidence write failure")

        info = ("INFO kv_unified=1 kv_incremental=1 kv_reserve_ahead=256 context=4096 "
                "kv_capacity_cells=4096 batch_slots=2 kv_resident=0 conversation_cache_mib=4096 "
                "slot_cache=1 lookup=0 mtp_max=2 spec=2 pcie_frac=0.00 batch_groups=1")
        with tempfile.TemporaryDirectory() as tmp:
            for fault in ("startup-error", "bad-ready-info", "ring-fallback", "save-error"):
                with self.subTest(fault=fault):
                    stderr = Path(tmp) / f"{fault}.stderr"
                    evidence = {"commands": [], "stdout": [],
                                "source_ring": {"allocated_page_rounded_ring_cells": 200}}
                    engine = TrackingEngine(evidence, stderr, cleanup_timeout=2)
                    script = (
                        "import sys,time\n"
                        f"fault={fault!r}\n"
                        "if fault == 'startup-error':\n"
                        " print('ERR synthetic startup failure', flush=True)\n"
                        "else:\n"
                        f" sys.stderr.write({(gate.RING_FALLBACK if fault == 'ring-fallback' else '')!r} + ('\\n' if fault == 'ring-fallback' else ''))\n"
                        " sys.stderr.flush()\n"
                        f" print({(info.replace('batch_groups=1','batch_groups=2') if fault == 'bad-ready-info' else info)!r}, flush=True)\n"
                        " print('READY 4096', flush=True)\n"
                        "time.sleep(60)\n")
                    command = [sys.executable, "-c", script, "--spec", "2", "--mtp-max-t", "2", "--batch-mtp"]
                    args = type("Args", (), {"context": 4096, "cache_mib": 4096,
                                              "startup_timeout": 3})()
                    process = {"label": "fake", "stderr_path": str(stderr), "command": [], "finalized": False}
                    with self.assertRaises((RuntimeError, AssertionError)):
                        gate.start_owned_engine(engine, Suite(fault == "save-error"), command,
                                                tmp, {"STRATA_BATCH_MTP": "1"}, args, evidence)
                    pid = engine.p.pid
                    gate.finish_process(engine, evidence, process, False)  # Production outer-finally path.
                    self.assertEqual(engine.launch_calls, 1, "failure must not start a replacement worker")
                    self.assertEqual(engine.close_calls, 1, "startup exception and outer finally close exactly once")
                    self.assertIsNotNone(engine.p.poll(), f"worker leaked after {fault}")
                    self.assertFalse(engine.reader.is_alive(), f"reader leaked after {fault}")
                    self.assertTrue(engine.writer is None or not engine.writer.is_alive(),
                                    f"stdin writer leaked after {fault}")
                    self.assertTrue(engine.p.stdin.closed, f"stdin pipe left open after {fault}")
                    self.assertTrue(engine.log.closed, f"stderr log left open after {fault}")
                    self.assertEqual(engine.p.pid, pid)
                    self.assertTrue(process["finalized"])
                    self.assertEqual(process["pid"], pid)
                    self.assertEqual(len(evidence["processes"]), 1)

    def test_popen_failure_after_log_open_is_finalized_without_worker(self):
        class TrackingEngine(gate.BatchMtpEngine):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.launch_calls = 0
                self.close_calls = 0

            def start(self, *args, **kwargs):
                self.launch_calls += 1
                return super().start(*args, **kwargs)

            def close(self, success):
                self.close_calls += 1
                return super().close(success)

        class Suite:
            def save(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            stderr = Path(tmp) / "popen-failure.stderr"
            evidence = {"commands": [], "stdout": [],
                        "source_ring": {"allocated_page_rounded_ring_cells": 200}}
            engine = TrackingEngine(evidence, stderr, cleanup_timeout=2)
            command = [sys.executable, "-c", "pass", "--spec", "2", "--mtp-max-t", "2", "--batch-mtp"]
            args = type("Args", (), {"context": 4096, "cache_mib": 4096, "startup_timeout": 3})()
            process = {"label": "popen-failure", "stderr_path": str(stderr), "command": [], "finalized": False}
            with mock.patch("subprocess.Popen", side_effect=OSError("synthetic Popen failure")) as popen:
                with self.assertRaises(OSError):
                    gate.start_owned_engine(engine, Suite(), command, tmp,
                                            {"STRATA_BATCH_MTP": "1"}, args, evidence)
                self.assertEqual(popen.call_count, 1)
            gate.finish_process(engine, evidence, process, False)
            self.assertEqual(engine.launch_calls, 1)
            self.assertEqual(engine.close_calls, 1)
            self.assertIsNone(engine.p)
            self.assertTrue(engine.log.closed)
            self.assertIsNone(engine.reader)
            self.assertTrue(process["finalized"])
            self.assertEqual(len(evidence["processes"]), 1)


if __name__ == "__main__":
    unittest.main()
