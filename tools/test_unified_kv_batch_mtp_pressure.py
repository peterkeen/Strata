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


PARK_LINE = "strata serve: pressure parked target-only 2048 tokens; parked=2 bytes=528687156"


def counter_row(slot=0, windows=129, offered=128, accepted=128, rejected=0, discarded=0, attempts=1,
                incoherent=0, not_ready=0, limits=0, capacity=0, reserve=1):
    return (f"strata batch_mtp_stats slot={slot} windows={windows} offered={offered} accepted={accepted} "
            f"rejected={rejected} discarded={discarded} fallback_attempts={attempts} "
            f"fallback_incoherent={incoherent} fallback_not_ready={not_ready} fallback_limits={limits} "
            f"fallback_capacity={capacity} fallback_reserve={reserve}")


class FakeWitnessEngine:
    """Native-protocol stand-in: admission rows, BT rows, then a terminal once BSTOP lands."""

    def __init__(self, stderr_path, prompt_len, cap, tokens, *, terminal="cancel", park=False,
                 rows=None, parks=(), stop_at=None):
        self.stderr_path = Path(stderr_path)
        self.stage = "startup"
        self.sent = []
        self.prompt_len = prompt_len
        self.cap = cap
        self.tokens = list(tokens)
        self.terminal = terminal
        self.stop_at = stop_at
        self.seq = 0
        self.produced = 0
        self._rows = list(rows) if rows is not None else [counter_row()]
        self._parks = list(parks)
        self._park = park
        self._admission = ["T", "DONE", "BADM"]
        self._done = False
        self._stderr_written = False

    def _write_stderr(self):
        if self._stderr_written:
            return
        lines = list(self._rows) + list(self._parks)
        if self._park and PARK_LINE not in lines:
            lines.append(PARK_LINE)
        with self.stderr_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        self._stderr_written = True

    def send(self, *lines, deadline=None):
        self.sent.extend(lines)

    def _bstop_sent(self):
        return any(line.startswith("BSTOP") for line in self.sent)

    def next_event(self, deadline=None):
        self.seq += 1
        base = {"seq": self.seq, "wall_s": float(self.seq)}
        if self._admission:
            kind = self._admission.pop(0)
            if kind == "T":
                self.produced = 1
                return {**base, "kind": "T", "token": self.tokens[0]}
            if kind == "DONE":
                return {**base, "kind": "DONE", "count": 1, "prompt_count": self.prompt_len,
                        "prompt_ms": 1.0, "decode_ms": 1.0, "finish": "length", "reused": None}
            return {**base, "kind": "BADM", "slot": 0, "continues": True}
        if self._done:
            raise AssertionError("fake native emitted an event after its terminal")
        if self._bstop_sent() or (self.stop_at is not None and self.produced >= self.stop_at):
            self._done = True
            self._write_stderr()
            return {**base, "kind": "BDONE", "slot": 0, "count": self.produced,
                    "finish": self.terminal, "decode_ms": 1.0}
        if self.produced >= len(self.tokens):
            raise AssertionError("fake native ran out of scripted tokens before any terminal")
        self.produced += 1
        return {**base, "kind": "BT", "slot": 0, "token": self.tokens[self.produced - 1]}


def run_witness_stage(tmp, *, cap=320, terminal="cancel", park=False, rows=None, parks=(), stop_at=None,
                      plan=None, prompt_cells=None, reference_tokens=None, stream_tokens=None):
    """Drive PressureSuite.run_optional_row_fallback (the production gate path) with a fake native."""
    plan = plan or gate.optional_witness_plan(4096)
    prompt_cells = prompt_cells or plan["prompt_cells"]
    prompt = list(range(1, prompt_cells + 1))
    reference_tokens = reference_tokens or list(range(10_000, 10_000 + cap))
    stream_tokens = stream_tokens or list(reference_tokens)
    stderr = Path(tmp) / "witness.stderr"
    stderr.write_text("", encoding="utf-8")
    evidence = {"stdout": [], "commands": [], "stages": [], "processes": []}
    engine = FakeWitnessEngine(stderr, len(prompt), cap, stream_tokens, terminal=terminal, park=park,
                               rows=rows, parks=parks, stop_at=stop_at)
    output = Path(tmp) / "witness-evidence.json"
    suite = gate.PressureSuite(engine, evidence, output, 30.0, plan["context_cells"], 0.0, 12345,
                               stderr_path=stderr)
    reference = common.Request("fake-optional-reference", prompt, cap)
    reference.tokens = list(reference_tokens)
    reference.completion = {"kind": "DONE", "finish": "length"}
    suite.run_optional_row_fallback(prompt, cap, reference, plan, 30.0)
    return suite, evidence, output


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

    def test_optional_witness_plan_fills_the_pool_at_a_page_boundary(self):
        plan = gate.optional_witness_plan(4096)
        self.assertEqual(plan["prompt_cells"], 1919)
        self.assertEqual(plan["prompt_page_offset"], gate.PAGE - 1)
        self.assertEqual(plan["prompt_pages"], 480)
        self.assertEqual(plan["witness_mapping_cells"], 2176)
        self.assertEqual(plan["witness_mapping_pages"], 544)
        self.assertEqual(plan["pages_in_use_at_boundary"], plan["pool_pages"])
        self.assertEqual(plan["free_pages_at_boundary"], 0)
        self.assertEqual(plan["spare_pages"], 0)
        self.assertEqual(plan["boundary_output_count"], 257)
        self.assertEqual(plan["bstop_output_count"], 258)
        self.assertEqual(plan["boundary_output_count"], plan["reserve_ahead_cells"] + 1)

    def test_optional_witness_plan_is_derived_per_context_and_rejects_unsplittable_pools(self):
        for context in (1024, 2048, 4096):
            plan = gate.optional_witness_plan(context)
            self.assertEqual(plan["pages_in_use_at_boundary"], plan["pool_pages"])
            self.assertEqual(plan["free_pages_at_boundary"], 0)
            self.assertEqual(plan["prompt_page_offset"], gate.PAGE - 1)
            self.assertEqual(plan["prompt_pages"] + plan["witness_mapping_pages"], plan["pool_pages"])
        # 1023 pool pages cannot split into two equal prompt/headroom halves.
        with self.assertRaises(AssertionError):
            gate.optional_witness_plan(4092)

    def test_optional_bstop_trigger_comes_from_the_boundary_arithmetic(self):
        plan = gate.optional_witness_plan(4096)
        self.assertFalse(gate.optional_bstop_ready(plan["boundary_output_count"] - 1, plan))
        self.assertFalse(gate.optional_bstop_ready(plan["boundary_output_count"], plan))
        self.assertTrue(gate.optional_bstop_ready(plan["bstop_output_count"], plan))
        self.assertTrue(gate.optional_bstop_ready(plan["bstop_output_count"] + 32, plan))

    def test_optional_witness_stage_passes_with_isolated_fallback_and_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite, evidence, _ = run_witness_stage(Path(tmp))
            stage = evidence["stages"][-1]
            self.assertTrue(stage["passed"])
            self.assertEqual(stage["fixture_isolation"], "isolated_optional_witness")
            self.assertEqual(stage["bt_counts_at_bstop"], {"0": 258})
            self.assertTrue(stage["pressure_park_absence"]["absent"])
            self.assertEqual(stage["per_slot_counter_rows"][0]["fallback_reserve"], 1)
            self.assertTrue(all(check["observed"] for check in stage["witness"]["checks"].values()))
            self.assertEqual(stage["fixture_arithmetic"]["free_pages_at_boundary"], 0)
            self.assertEqual([record["slot"] for record in stage["requests"]], [0])
            self.assertEqual(stage["requests"][0]["completion"]["finish"], "cancel")

    def test_optional_witness_stage_fails_closed_when_a_pressure_park_appears(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(AssertionError) as caught:
                run_witness_stage(tmp, terminal="pressure", park=True)
            message = str(caught.exception)
            self.assertIn("no_pressure_park", message)
            self.assertIn("fixture non-isolation", message)
            stage = json.loads((tmp / "witness-evidence.json").read_text())["stages"][-1]
            self.assertEqual(stage["fixture_isolation"], "fixture_non_isolation")
            self.assertFalse(stage["passed"])

    def test_optional_witness_stage_fails_closed_on_zero_offers_or_a_second_reason(self):
        cases = (([counter_row(offered=0, accepted=0, windows=1)], "positive_offers"),
                 ([counter_row(capacity=1, reserve=1, attempts=2, windows=130)], "no_other_fallback_reason"))
        for rows, missed in cases:
            with self.subTest(missed=missed):
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaises(AssertionError) as caught:
                        run_witness_stage(Path(tmp), rows=rows)
                    self.assertIn(missed, str(caught.exception))
                    self.assertIn("fixture non-isolation", str(caught.exception))

    def test_optional_witness_stage_fails_closed_when_the_boundary_window_never_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError) as caught:
                run_witness_stage(Path(tmp), terminal="stop", stop_at=100)
            message = str(caught.exception)
            for missed in ("boundary_output_count_reached", "bstop_before_mandatory_shortage",
                           "healthy_cancel_terminal"):
                self.assertIn(missed, message)
            self.assertIn("fixture non-isolation", message)

    def test_optional_witness_stage_rejects_parity_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = gate.optional_witness_plan(4096)
            clean = [10_000 + index for index in range(plan["bstop_output_count"] + 64)]
            drifted = list(clean)
            drifted[200] += 7  # the gate observes this row before its bounded stop point
            with self.assertRaises(AssertionError) as caught:
                run_witness_stage(Path(tmp), plan=plan, reference_tokens=clean, stream_tokens=drifted)
            self.assertIn("parity", str(caught.exception))
            stage = json.loads((Path(tmp) / "witness-evidence.json").read_text())["stages"][-1]
            self.assertEqual(stage["fixture_isolation"], "fixture_non_isolation")

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
