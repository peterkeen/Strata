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


class FakeDualEngine:
    """Minimal dual-slot native stand-in for the production optional-witness stage.

    Each slot admits (T/DONE/BADM), then they advance in rotation. A slot finishes with a scripted
    ``pressure``/``length`` terminal at a produced count, otherwise it keeps emitting BT rows until the stage
    sends its BSTOP and then terminates ``cancel``.
    """

    def __init__(self, stderr_path, prompts, cap, tokens_by_slot, *, rows, parks=(), pressure_at=None,
                 length_at=None, weights=None):
        self.stderr_path = Path(stderr_path)
        self.stage = "startup"
        self.sent = []
        self.seq = 0
        self.cap = cap
        self._rows = list(rows)
        self._parks = list(parks)
        self._stderr_written = False
        self._pressure_at = dict(pressure_at or {})
        self._length_at = dict(length_at or {})
        self._slots = {slot: {"prompt_len": len(prompt), "tokens": list(tokens_by_slot[slot]),
                              "produced": 0, "done": False}
                       for slot, prompt in prompts.items()}
        self._admission = []
        for slot in sorted(self._slots):
            self._admission.extend([("T", slot), ("DONE", slot), ("BADM", slot)])
        per_round = {slot: 1 for slot in self._slots}
        per_round.update(weights or {})
        self._rotation = []
        for slot in sorted(per_round):
            self._rotation.extend([slot] * per_round[slot])
        self._turn = 0

    def _write_stderr(self):
        if self._stderr_written:
            return
        with self.stderr_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(list(self._rows) + list(self._parks)) + "\n")
        self._stderr_written = True

    def send(self, *lines, deadline=None):
        self.sent.extend(lines)

    def _bstop(self, slot):
        return f"BSTOP {slot}" in self.sent

    def _terminal(self, slot, finish, base):
        self._slots[slot]["done"] = True
        self._write_stderr()
        return {**base, "kind": "BDONE", "slot": slot, "count": self._slots[slot]["produced"],
                "finish": finish, "decode_ms": 1.0}

    def next_event(self, deadline=None):
        self.seq += 1
        base = {"seq": self.seq, "wall_s": float(self.seq)}
        if self._admission:
            kind, slot = self._admission.pop(0)
            state = self._slots[slot]
            if kind == "T":
                state["produced"] = 1
                return {**base, "kind": "T", "token": state["tokens"][0]}
            if kind == "DONE":
                return {**base, "kind": "DONE", "count": 1, "prompt_count": state["prompt_len"],
                        "prompt_ms": 1.0, "decode_ms": 1.0, "finish": "length", "reused": None}
            return {**base, "kind": "BADM", "slot": slot, "continues": True}
        for _ in range(2 * len(self._rotation)):
            slot = self._rotation[self._turn % len(self._rotation)]
            self._turn += 1
            state = self._slots[slot]
            if state["done"]:
                continue
            if slot in self._pressure_at and state["produced"] >= self._pressure_at[slot]:
                return self._terminal(slot, "pressure", base)
            if slot in self._length_at and state["produced"] >= self._length_at[slot]:
                return self._terminal(slot, "length", base)
            if self._bstop(slot):
                return self._terminal(slot, "cancel", base)
            if state["produced"] >= len(state["tokens"]):
                raise AssertionError("fake dual native ran out of scripted tokens before a terminal")
            state["produced"] += 1
            return {**base, "kind": "BT", "slot": slot, "token": state["tokens"][state["produced"] - 1]}
        raise AssertionError("fake dual native has no live slot to advance")


def dual_optional_inputs(*, cap=320, prompt_cells=(1787, 1791), drifting_slot=None, drift_index=200):
    prompts = {slot: list(range(1, cells + 1)) for slot, cells in enumerate(prompt_cells)}
    stream = {slot: [10_000 + slot * 1_000 + index for index in range(cap)] for slot in prompts}
    reference = {slot: list(stream[slot]) for slot in prompts}
    if drifting_slot is not None:
        stream[drifting_slot][drift_index] += 7  # observed before the bounded stop point
    refs = []
    for slot in sorted(prompts):
        req = common.Request(f"fake-optional-reference-{slot}", prompts[slot], cap)
        req.tokens = reference[slot]
        req.completion = {"kind": "DONE", "finish": "length"}
        refs.append(req)
    return prompts, stream, refs


def run_dual_optional_stage(tmp, *, cap=320, prompt_cells=(1787, 1791), rows=None, parks=(), pressure_at=None,
                            length_at=None, drifting_slot=None, plans=None, weights=None):
    """Drive PressureSuite.run_optional_row_fallback (the production gate path) with a fake dual native."""
    plans = plans or [gate.optional_slot_boundary(cells) for cells in prompt_cells]
    prompts, stream, refs = dual_optional_inputs(cap=cap, prompt_cells=prompt_cells, drifting_slot=drifting_slot)
    if rows is None:
        rows = [counter_row(slot=0, windows=129, offered=129, accepted=129, attempts=0, reserve=0),
                counter_row(slot=1, windows=129, offered=128, accepted=128, attempts=1, reserve=1)]
    stderr = Path(tmp) / "witness.stderr"
    stderr.write_text("", encoding="utf-8")
    evidence = {"stdout": [], "commands": [], "stages": [], "processes": []}
    engine = FakeDualEngine(stderr, prompts, cap, stream, rows=rows, parks=parks,
                            pressure_at=pressure_at, length_at=length_at, weights=weights)
    output = Path(tmp) / "witness-evidence.json"
    suite = gate.PressureSuite(engine, evidence, output, 30.0, 4096, 0.0, 12345, stderr_path=stderr)
    suite.run_optional_row_fallback(tuple(prompts[slot] for slot in sorted(prompts)), cap, refs, plans, 30.0)
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

    def test_optional_slot_boundary_arithmetic_is_per_prompt(self):
        first = gate.optional_slot_boundary(1787)
        second = gate.optional_slot_boundary(1791)
        self.assertEqual(first["prompt_page_offset"], gate.PAGE - 1)
        self.assertEqual(first["slot_mapping_end_cells"], 2044)
        self.assertEqual(first["slot_mapping_pages"], 511)
        self.assertEqual(second["slot_mapping_end_cells"], 2048)
        self.assertEqual(second["slot_mapping_pages"], 512)
        for plan in (first, second):
            self.assertEqual(plan["boundary_output_count"], plan["reserve_ahead_cells"] + 1)
            self.assertEqual(plan["boundary_output_count"], 257)
            self.assertEqual(plan["bstop_output_count"], 258)
        with self.assertRaises(AssertionError):
            gate.optional_slot_boundary(1788)  # not the last cell of a page

    def test_optional_fixture_sizing_is_documented_as_hypothesis_only(self):
        prompts = (list(range(1, 1788)), list(range(5_000, 6_791)))
        sizing = gate.optional_fixture_sizing(prompts, 320, 4096)
        self.assertEqual(sizing["pool_pages"], 1024)
        self.assertEqual(sizing["slot_mapping_pages_total"], 1023)
        self.assertFalse(sizing["mapping_pages_exceed_pool"])
        self.assertTrue(sizing["sizing_is_hypothesis_not_evidence"])
        with self.assertRaises(AssertionError):
            gate.optional_fixture_sizing(prompts, 128, 4096)  # cap below the boundary stop point
        with self.assertRaises(AssertionError):
            gate.optional_fixture_sizing((prompts[0], prompts[0]), 320, 4096)  # identical prompts

    def test_optional_stop_decision_is_per_slot_and_park_first(self):
        plans = [gate.optional_slot_boundary(1787), gate.optional_slot_boundary(1791)]
        requests = [common.Request("a", [1], 320, 0), common.Request("b", [2], 320, 1)]
        for request in requests:
            request.badm = {"continues": True}
        requests[0].tokens.extend(range(plans[0]["boundary_output_count"]))
        requests[1].tokens.extend(range(plans[1]["bstop_output_count"]))
        self.assertIsNone(gate.optional_stop_decision(requests, plans))
        requests[0].tokens.append(9)
        self.assertEqual(gate.optional_stop_decision(requests, plans), "both_boundaries_crossed_no_park")
        requests[1].completion = {"kind": "BDONE", "finish": "pressure"}
        self.assertEqual(gate.optional_stop_decision(requests, plans), "mandatory_park_observed")

    def test_optional_park_attribution_is_derived_from_the_witness_prefix(self):
        witness = common.Request("witness", list(range(1, 1788)), 320, 1)
        witness.tokens = list(range(0, 258))
        park = gate.parse_diagnostics(PARK_LINE)["pressure"][0]
        attributed = gate.optional_park_attribution(witness, [park])
        self.assertFalse(attributed["diagnostic_has_slot_field"])
        self.assertFalse(attributed["ordering_observable"])
        self.assertEqual(attributed["witness_slot"], 1)
        self.assertEqual(attributed["witness_consumed_prefix_cells"], 1787 + 258 - 1)
        self.assertIn("8739-8748", attributed["source"]["batch_mandatory_retry"]["lines"])
        self.assertIn("no reclaim retry", attributed["source"]["batch_optional_probe"]["fact"])
        self.assertIn("not observable", attributed["source"]["ordering"]["fact"])
        self.assertFalse(attributed["attributed_by_token_count"])  # the synthetic park line says 2048 tokens
        self.assertTrue(gate.optional_park_attribution(witness, [{"tokens": 1787 + 258 - 1}])[
            "attributed_by_token_count"])

    def test_measured_task12_rows_and_terminals_are_accepted(self):
        """The exact counter rows and terminals of the task12 dual run must be an accepted shape."""
        rows = gate.parse_batch_stats(
            "strata batch_mtp_stats slot=1 windows=129 offered=128 accepted=128 rejected=0 discarded=0 "
            "fallback_attempts=1 fallback_incoherent=0 fallback_not_ready=0 fallback_limits=0 "
            "fallback_capacity=0 fallback_reserve=1\n"
            "strata batch_mtp_stats slot=0 windows=130 offered=130 accepted=130 rejected=0 discarded=0 "
            "fallback_attempts=0 fallback_incoherent=0 fallback_not_ready=0 fallback_limits=0 "
            "fallback_capacity=0 fallback_reserve=0\n")
        requests = [common.Request("slot0", list(range(1, 1788)), 320, 0),
                    common.Request("slot1", list(range(1, 1792)), 320, 1)]
        requests[0].tokens = list(range(0, 261))
        requests[1].tokens = list(range(0, 258))
        requests[0].completion = {"kind": "BDONE", "finish": "cancel"}
        requests[1].completion = {"kind": "BDONE", "finish": "pressure"}
        diagnostics = gate.parse_diagnostics(
            "strata serve: pressure parked target-only 2048 tokens; parked=2 bytes=528687156\n")
        diagnostics["batch_mtp_stats"] = rows
        result = gate.validate_optional_shapes(requests, diagnostics, "mandatory_park_observed", True, True)
        self.assertEqual(result["shape"], "mandatory_park_then_cancel")
        self.assertEqual(result["missed"], [])
        self.assertTrue(result["park_attribution"]["attributed_by_token_count"])
        self.assertEqual(result["park_attribution"]["witness_slot"], 1)

    def test_optional_witness_stage_accepts_the_measured_park_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, evidence, _ = run_dual_optional_stage(
                tmp, pressure_at={1: 258}, weights={1: 2},
                parks=["strata serve: pressure parked target-only 2048 tokens; parked=2 bytes=528687156"])
            stage = evidence["stages"][-1]
            self.assertTrue(stage["passed"])
            self.assertEqual(stage["observed_shape"], "mandatory_park_then_cancel")
            self.assertEqual(stage["fixture_isolation"], "fixture_isolation_confirmed")
            self.assertTrue(all(check["observed"] for check in stage["witness"]["checks"].values()))
            finishes = {record["slot"]: record["completion"]["finish"] for record in stage["requests"]}
            self.assertEqual(finishes, {0: "cancel", 1: "pressure"})
            self.assertEqual(stage["per_slot_counter_rows"][1]["fallback_reserve"], 1)
            self.assertEqual(stage["per_slot_counter_rows"][0]["fallback_reserve"], 0)
            self.assertEqual(stage["bstop_reason"], "mandatory_park_observed")
            self.assertEqual(stage["stopped_slots"], [0])
            self.assertEqual(stage["park_attribution"]["parks"], 1)
            self.assertTrue(stage["park_attribution"]["attributed_by_token_count"])
            self.assertFalse(stage["park_ordering_observable"])
            self.assertIn("source", stage["park_attribution"])

    def test_optional_witness_stage_accepts_the_no_park_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, evidence, _ = run_dual_optional_stage(Path(tmp))
            stage = evidence["stages"][-1]
            self.assertTrue(stage["passed"])
            self.assertEqual(stage["observed_shape"], "no_park_both_cancel")
            self.assertEqual(stage["bstop_reason"], "both_boundaries_crossed_no_park")
            self.assertTrue(stage["pressure_park_absence"]["absent"])
            finishes = {record["slot"]: record["completion"]["finish"] for record in stage["requests"]}
            self.assertEqual(finishes, {0: "cancel", 1: "cancel"})
            self.assertEqual(stage["bt_counts_at_bstop"], {"0": 258, "1": 258})

    def test_optional_witness_stage_rejects_shapes_other_than_the_two_measured_ones(self):
        clean = counter_row(slot=0, windows=128, offered=128, accepted=128, attempts=0, reserve=0)
        cases = (
            ({"rows": [clean, counter_row(slot=1, windows=128, offered=128, accepted=128, attempts=0, reserve=0)],
              "pressure_at": {1: 258}, "weights": {1: 2}, "parks": [PARK_LINE]},
             "exactly_one_witness_slot_fallback_reserve"),
            ({"rows": [clean,
                        counter_row(slot=1, capacity=1, reserve=1, attempts=2, windows=130)],
              "pressure_at": {1: 258}, "weights": {1: 2}, "parks": [PARK_LINE]},
             "witness_slot_no_other_fallback_reason"),
            ({"rows": [counter_row(slot=0, windows=0, offered=0, accepted=0, attempts=0, reserve=0),
                        counter_row(slot=1, attempts=1, reserve=1)], "pressure_at": {1: 258},
              "weights": {1: 2}, "parks": [PARK_LINE]},
             "positive_offers_per_slot"),
            ({"rows": [clean, counter_row(slot=1, attempts=1, reserve=1)], "pressure_at": {1: 258},
              "weights": {1: 2}, "parks": [PARK_LINE, PARK_LINE]},
             "at_most_one_park"),
            ({"rows": [clean, counter_row(slot=1, attempts=1, reserve=1)], "length_at": {1: 258},
              "weights": {1: 2}},
             "accepted_terminal_shape"),
        )
        for kwargs, missed in cases:
            with self.subTest(missed=missed):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    with self.assertRaises(AssertionError) as caught:
                        run_dual_optional_stage(tmp, **kwargs)
                    self.assertIn(missed, str(caught.exception))
                    self.assertIn("fixture non-isolation", str(caught.exception))
                    stage = json.loads((tmp / "witness-evidence.json").read_text())["stages"][-1]
                    self.assertEqual(stage["fixture_isolation"], "fixture_non_isolation")
                    self.assertFalse(stage["passed"])

    def test_optional_witness_stage_rejects_parity_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with self.assertRaises(AssertionError) as caught:
                run_dual_optional_stage(tmp, drifting_slot=0)
            self.assertIn("parity", str(caught.exception))
            stage = json.loads((tmp / "witness-evidence.json").read_text())["stages"][-1]
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
