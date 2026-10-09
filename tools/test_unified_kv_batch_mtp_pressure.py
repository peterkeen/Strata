"""Offline protocol, allocation-geometry and evidence-contract tests for the GPU gate."""
from __future__ import annotations

from collections import deque
import io
import json
import sys
import tempfile
from pathlib import Path
import types
import unittest
from contextlib import redirect_stderr
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_kv_batch_mtp_pressure_smoke as gate
import unified_kv_smoke as common


PARK_LINE = "strata serve: pressure parked target-only 2048 tokens; parked=2 bytes=528687156"
TARGET_ONLY_CLONE_LINE = "strata serve: TARGET_ONLY slot clone: private MTP proposals suppressed until full replay"


def counter_row(slot=0, windows=129, offered=128, accepted=128, rejected=0, discarded=0, attempts=1,
                incoherent=0, not_ready=0, limits=0, capacity=0, reserve=1):
    return (f"strata batch_mtp_stats slot={slot} windows={windows} offered={offered} accepted={accepted} "
            f"rejected={rejected} discarded={discarded} fallback_attempts={attempts} "
            f"fallback_incoherent={incoherent} fallback_not_ready={not_ready} fallback_limits={limits} "
            f"fallback_capacity={capacity} fallback_reserve={reserve}")


PRESSURE_PARK_BYTES = 524843760
PRESSURE_RESTORE_BYTES = 288764012


class FakePressureEngine:
    """Scripted native stdout/stderr for the production pressure gate: the pair, then the restore GEN.

    The pair is fully scripted, including the sibling's cancel that follows the pressure park (the
    gate sends its BSTOP as soon as it consumes the pressure terminal). The restore attempt answers
    the next ``GEN`` with the drained slot's remaining continuation and its canonical-restore and
    TARGET_ONLY diagnostics, so the production restore witnesses are exercised for real.
    """

    def __init__(self, stderr_path, prompts, cap, streams, *, rows, park_slot=1, park_at=8,
                 park_bytes=PRESSURE_PARK_BYTES):
        self.stderr_path = Path(stderr_path)
        self.stage = "startup"
        self.sent = []
        self.seq = 0
        self.cap = cap
        self.prompts = {slot: list(ids) for slot, ids in prompts.items()}
        self.streams = {slot: list(ids) for slot, ids in streams.items()}
        self.rows = [rows] if isinstance(rows, str) else list(rows)
        self.park_slot = park_slot
        self.park_at = park_at
        self.park_bytes = park_bytes
        self.produced = {}
        self._script = deque()
        self._pair_scripted = False

    def _write(self, *lines):
        with self.stderr_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _event(self, raw):
        self.seq += 1
        return common.parse_line(raw) | {"raw": raw, "seq": self.seq, "wall_s": self.seq / 100}

    def _script_pair(self):
        if self._pair_scripted:
            return
        self._pair_scripted = True
        produced = {slot: 1 for slot in sorted(self.prompts)}
        for slot in sorted(self.prompts):
            self._script.append(self._event(f"T {self.streams[slot][0]}"))
            self._script.append(self._event(f"DONE 1 {len(self.prompts[slot])} 1.0 1.0 length 0 0 0 0 0 0 0.0 0 0"))
            self._script.append(self._event(f"BADM {slot} 1"))
        sibling = next(slot for slot in sorted(self.prompts) if slot != self.park_slot)
        while produced[self.park_slot] < self.park_at:
            for slot in sorted(self.prompts):
                self._script.append(self._event(f"BT {slot} {self.streams[slot][produced[slot]]}"))
                produced[slot] += 1
        self._script.append(self._event(f"BT {sibling} {self.streams[sibling][produced[sibling]]}"))
        produced[sibling] += 1
        park_tokens = len(self.prompts[self.park_slot]) + produced[self.park_slot] - 1
        self._script.append(("write", list(self.rows) + [
            f"strata serve: pressure parked target-only {park_tokens} tokens; parked=1 bytes={self.park_bytes}"]))
        self._script.append(self._event(f"BDONE {self.park_slot} {produced[self.park_slot]} pressure 1.0"))
        self._script.append(self._event(f"BDONE {sibling} {produced[sibling]} cancel 1.0"))
        self.produced = produced

    def _script_restore(self, line):
        fields = line.split()
        cap = int(fields[1])
        ids = [int(token) for token in fields[-1].split(",")]
        start = self.produced[self.park_slot]
        tokens = self.streams[self.park_slot][start:start + cap]
        assert len(tokens) == cap, "fake restore script is shorter than the requested continuation"
        reused = len(ids) - 1
        self._script.append(("write", [
            f"strata serve: conversation cache: restored {reused} tokens (live) in 1.0 ms; parked=1 "
            f"bytes={PRESSURE_RESTORE_BYTES}",
            "strata serve: TARGET_ONLY restore: private MTP proposals suppressed until full replay",
            "strata serve: TARGET_ONLY decode: T=1, MTP/suffix proposals disabled"]))
        for token in tokens:
            self._script.append(self._event(f"T {token}"))
        self._script.append(self._event(f"DONE {len(tokens)} {len(ids)} 1.0 1.0 length 0 0 {reused} 0 0 0 0.0 0 0"))

    def send(self, *lines, deadline=None):
        for line in lines:
            self.sent.append(line)
            if line.startswith("BGEN "):
                self._script_pair()
            elif line.startswith("GEN "):
                self._script_restore(line)

    def next_event(self, deadline=None):
        while True:
            if not self._script:
                raise AssertionError("fake pressure native ran out of scripted events")
            item = self._script.popleft()
            if isinstance(item, tuple) and item[0] == "write":
                self._write(*item[1])
                continue
            return item


def pressure_inputs(*, cap=768, prompt_cells=(1536, 1540), park_at=8):
    # Distinct first tokens, as in the real fixture: the capacity plan requires that the two
    # histories cannot fit the pool even under hypothetical common-prefix sharing.
    prompts = {slot: list(range(1 + slot * 5_000, 1 + slot * 5_000 + cells))
               for slot, cells in enumerate(prompt_cells)}
    streams = {slot: [10_000 + slot * 1_000 + index for index in range(cap)] for slot in prompts}
    refs = []
    for slot in sorted(prompts):
        req = common.Request(f"fake-pressure-reference-{slot}", prompts[slot], cap)
        req.tokens = list(streams[slot])
        req.completion = {"kind": "DONE", "finish": "length"}
        refs.append(req)
    return prompts, streams, refs


def run_pressure_gate(tmp, *, rows, substitute=None, park_at=8):
    """Drive the production pressure path: run_source_pair, then run_pressure_restore."""
    prompts, streams, refs = pressure_inputs(park_at=park_at)
    stderr = Path(tmp) / "pressure.stderr"
    stderr.write_text("", encoding="utf-8")
    evidence = {"stdout": [], "commands": [], "stages": [], "processes": [], "coverage": {},
                "source_ring": {"allocated_page_rounded_ring_cells": 200}}
    engine = FakePressureEngine(stderr, prompts, 768, streams, rows=rows, park_at=park_at)
    output = Path(tmp) / "pressure-evidence.json"
    output.write_text("{}\n", encoding="utf-8")
    suite = gate.PressureSuite(engine, evidence, output, 30.0, 4096, 0.0, 12345, stderr_path=stderr)
    prompts_tuple = tuple(prompts[slot] for slot in sorted(prompts))
    try:
        owner, source = suite.run_source_pair(prompts_tuple, 768, refs, 30.0, rejection_substitute=substitute)
        source["capacity_plan"] = gate.pressure_capacity_plan(prompts_tuple, [768, 768], 4096)
        source["passed"] = True
        args = types.SimpleNamespace(pressure_cap=768, stage_timeout=30)
        gate.run_pressure_restore(args, evidence, suite, stderr, prompts_tuple, refs, owner, source)
    finally:
        # Mirrors the production gate's finally: the artifact is written even when the
        # rejection verdict fails closed, so the restore evidence is always recorded.
        output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence, source, stderr


def all_accept_rows():
    """The measured all-accept shape (str-7ze.16/17: identical counters and park bytes), as stderr lines."""
    return "\n".join((
        counter_row(slot=0, windows=255, offered=255, accepted=255, rejected=0, attempts=0, reserve=0),
        counter_row(slot=1, windows=254, offered=254, accepted=254, rejected=0, attempts=0, reserve=0)))


def natural_rejection_rows():
    return "\n".join((
        counter_row(slot=0, windows=255, offered=255, accepted=255, rejected=0, attempts=0, reserve=0),
        counter_row(slot=1, windows=254, offered=254, accepted=253, rejected=1, attempts=0, reserve=0)))


# Provenance of the substitute fixture (read-only inspection of the real artifacts, str-7ze.20):
# the correctness gate writes its stages under processes[i]["stages"] and leaves the TOP-LEVEL
# "stages" list EMPTY; the batch process's stage carries the native rows (slot 0 8/7/1,
# slot 1 9/6/3 -> rejected 4) and its startup_diagnostics carries the hook activation line, while
# the solo-reference process has two stages and no rows. Observed on
# /data/llm/Strata-tests/deploy-gates-20261009/gates-correctness-deployhook-context1024.json
# (sha256 3d1da86b43fe98dcbd130abfcf3ee3ad9582577c9c8a343e4db25d6ee2ccc6cd) and
# .../gates-correctness-context1024.json (top-level stages 0; process stages 2 + 1).
CORRECTNESS_HOOK_LINE = "strata batch_mtp_test_hook enabled: explicit proposal substitutions are test-only"
CORRECTNESS_BATCH_STAGE = "dual-overlap-forced-accept-reject-and-natural-catchup"
CORRECTNESS_COUNTER_ROWS = ({"slot": 0, "offered": 8, "accepted": 7, "rejected": 1, "discarded": 0},
                            {"slot": 1, "offered": 9, "accepted": 6, "rejected": 3, "discarded": 0})


def correctness_processes(*, rows=None, hook_lines=None):
    """The processes[*].stages[*] shape the correctness gate emits (see provenance above)."""
    rows = [dict(row) for row in (CORRECTNESS_COUNTER_ROWS if rows is None else rows)]
    hook_lines = [CORRECTNESS_HOOK_LINE] if hook_lines is None else list(hook_lines)
    return [
        {"stages": [{"name": "solo-target-MTP-reference-0", "passed": True,
                      "stderr_diagnostics": {"batch_mtp_stats": []}},
                     {"name": "solo-target-MTP-reference-1", "passed": True,
                      "stderr_diagnostics": {"batch_mtp_stats": []}}],
         "startup_diagnostics": {"hook_lines": []}},
        {"stages": [{"name": CORRECTNESS_BATCH_STAGE, "passed": True,
                      "stderr_diagnostics": {"batch_mtp_stats": rows}}],
         "startup_diagnostics": {"hook_lines": hook_lines}},
    ]


def correctness_artifact(**overrides):
    """Shape of a passed `--mode correctness` artifact (hook-forced rejection), as the gate writes it."""
    artifact = {
        "schema": 1,
        "harness": "unified-kv-batch-mtp",
        "mode": "correctness",
        "passed": True,
        "context": 1024,
        "exe": "/tmp/fake-strata",
        "config": "/tmp/fake-model.json",
        "coverage": {"correctness": "RUN", "dual_overlap": "RUN"},
        "deterministic_proposal_schedule": {
            "slot0": {"offer": 1, "outcome": "accepted"},
            "slot1": {"offer": 1, "outcome": "rejected"}},
        "stages": [],  # the real gate leaves the top-level list empty
        "processes": correctness_processes(),
    }
    artifact.update(overrides)
    return artifact


def write_correctness_artifact(tmp, **overrides):
    path = Path(tmp) / "correctness.json"
    path.write_text(json.dumps(correctness_artifact(**overrides)), encoding="utf-8")
    return path


def valid_substitute(tmp):
    return gate.validate_rejection_substitute(write_correctness_artifact(tmp), exe=Path("/tmp/fake-strata"),
                                              config=Path("/tmp/fake-model.json"))


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
            self.assertEqual(stage["bt_counts_at_stop"], {"0": 258, "1": 258})

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

    def test_optional_fixture_sizing_pins_the_gated_page_boundary_contract(self):
        """The gate depends on per-slot page-boundary sizing, cap margin and context fit.

        The task-14 "pool exactly full / one spare page" derivation was explicitly
        abandoned in task16, so it is deliberately not pinned here; optional_fixture_sizing
        labels its arithmetic as sizing, never as the witness.
        """
        prompts = (list(range(1, 1788)), list(range(5000, 6791)))
        cap, context = 320, 4096
        sizing = gate.optional_fixture_sizing(prompts, cap, context)
        self.assertTrue(sizing["sizing_is_hypothesis_not_evidence"])
        self.assertEqual(sizing["pool_pages"], context // gate.PAGE)
        self.assertEqual(sizing["output_cap"], cap)
        for prompt, plan in zip(prompts, sizing["slots"]):
            # run_optional_row_fallback gates len(prompt) == plan["prompt_cells"].
            self.assertEqual(len(prompt), plan["prompt_cells"])
            self.assertEqual(plan["prompt_page_offset"], gate.PAGE - 1)
            self.assertEqual(plan["boundary_output_count"], plan["reserve_ahead_cells"] + 1)
            self.assertLessEqual(len(prompt) + cap + gate.GUARD, context)
        self.assertGreaterEqual(cap, max(sizing["bstop_output_counts"]) + gate.OPTIONAL_WITNESS_MIN_MARGIN)
        # The abandoned full-pool / one-spare-page arithmetic is not part of this contract.
        self.assertNotIn("initial_pages_total", sizing)
        self.assertNotIn("one_spare_page", json.dumps(sizing))

    def test_ring_wrap_verdict_is_computed_from_preserved_evidence(self):
        evidence = {"source_ring": {"allocated_page_rounded_ring_cells": 200}}
        run = gate.ring_wrap_verdict(evidence, prompt_cells=1536, consumed_history_cells=1596, proposals=3)
        self.assertEqual(run["verdict"], "RUN")
        self.assertTrue(run["prompt_exceeds_ring"])
        self.assertTrue(run["consumed_history_exceeds_ring"])
        self.assertIn("byte-level private-ring internals are not measured", run["evidence_limit"])
        self.assertEqual(gate.ring_wrap_verdict(evidence, prompt_cells=200, proposals=3)["verdict"], "UNTESTED")
        self.assertEqual(gate.ring_wrap_verdict(evidence, prompt_cells=1536, consumed_history_cells=1596,
                                               proposals=0)["verdict"], "UNTESTED")
        # The verdict needs BOTH the prompt and the consumed history beyond the ring,
        # matching the pressure gate's own prefill/continue precondition.
        short = gate.ring_wrap_verdict(evidence, prompt_cells=100, consumed_history_cells=900, proposals=1)
        self.assertEqual(short["verdict"], "UNTESTED")
        self.assertFalse(short["prompt_exceeds_ring"])
        missing = gate.ring_wrap_verdict({}, prompt_cells=1536, proposals=3)
        self.assertEqual(missing["verdict"], "UNTESTED")
        self.assertIn("not preserved", missing["reason"])
        text = gate.ring_wrap_coverage(evidence, prompt_cells=1536, consumed_history_cells=1596, proposals=3)
        self.assertTrue(text.startswith("RUN ("))
        self.assertIn("evidence_limit", text)
        self.assertEqual(evidence["coverage_evidence"]
                         ["bounded_private_ring_wrap_with_actual_proposals"]["verdict"], "RUN")

    def test_natural_rejection_witness_is_present_without_a_substitute(self):
        witness = gate.rejection_witness(gate.parse_batch_stats(natural_rejection_rows()))
        self.assertTrue(witness["satisfied"])
        self.assertEqual(witness["mechanism"], "natural_native_verifier_rejection")
        self.assertEqual(witness["natural_rejection_witness"], "present")
        self.assertIsNone(witness["substitute"])
        self.assertEqual(witness["rejected_total"], 1)
        self.assertEqual(witness["per_slot"][1]["rejected"], 1)
        # The strict helper stays available for callers that demand a natural rejection outright.
        self.assertEqual(gate.verify_stats(gate.parse_batch_stats(natural_rejection_rows()),
                                          reject=True)["rejected"], 1)
        self.assertIs(gate.require_rejection_witness({"rejection_witness": witness}), witness)

    def test_all_accept_run_requires_an_explicit_native_substitute(self):
        witness = gate.rejection_witness(gate.parse_batch_stats(all_accept_rows()))
        self.assertFalse(witness["satisfied"])
        self.assertEqual(witness["mechanism"], "none")
        self.assertEqual(witness["natural_rejection_witness"], gate.NATURAL_REJECTION_ABSENT)
        self.assertEqual(witness["rejected_total"], 0)
        self.assertIn("deterministic 100% acceptance", witness["natural_rejection_witness"])
        self.assertIn("--rejection-evidence", witness["failure_message"])
        self.assertIn("not a pressure defect", witness["failure_message"])
        with self.assertRaises(AssertionError) as caught:
            gate.require_rejection_witness({"rejection_witness": witness})
        self.assertIn("--rejection-evidence", str(caught.exception))
        with self.assertRaises(AssertionError):
            gate.require_rejection_witness({})

    def test_all_accept_run_with_a_validated_substitute_is_satisfied(self):
        with tempfile.TemporaryDirectory() as tmp:
            substitute = valid_substitute(tmp)
            witness = gate.rejection_witness(gate.parse_batch_stats(all_accept_rows()), substitute=substitute)
            self.assertTrue(witness["satisfied"])
            self.assertEqual(witness["mechanism"], gate.REJECTION_SUBSTITUTE_MECHANISM)
            self.assertEqual(witness["natural_rejection_witness"], gate.NATURAL_REJECTION_ABSENT)
            self.assertIs(witness["substitute"], substitute)
            self.assertTrue(substitute["forced_rejection_is_native"])
            self.assertEqual(substitute["rejected_total"], 4)
            self.assertEqual(substitute["counter_source"],
                             "processes[*].stages[*].stderr_diagnostics.batch_mtp_stats")
            self.assertEqual([(row["slot"], row["rejected"]) for row in substitute["counter_rows"]],
                             [(0, 1), (1, 3)])
            self.assertEqual({row["stage"] for row in substitute["counter_rows"]}, {CORRECTNESS_BATCH_STAGE})
            self.assertIn("same --exe and --config paths", substitute["identity_scope"])
            self.assertEqual(len(substitute["sha256"]), 64)
            self.assertTrue(gate.require_rejection_witness({"rejection_witness": witness})["satisfied"])

    def test_rejection_substitute_reads_the_process_scoped_stage_counters(self):
        """Regression guard: the real gate writes stages under processes[i], not at the top level."""
        exe, config = Path("/tmp/fake-strata"), Path("/tmp/fake-model.json")
        with tempfile.TemporaryDirectory() as tmp:
            # The exact real shape: empty top-level stages, counters in the batch process's stage.
            real_shape = json.loads(write_correctness_artifact(tmp).read_text())
            self.assertEqual(real_shape["stages"], [])
            self.assertEqual(len(real_shape["processes"][1]["stages"]), 1)
            substitute = gate.validate_rejection_substitute(Path(tmp) / "correctness.json",
                                                           exe=exe, config=config)
            self.assertEqual(substitute["rejected_total"], 4)
            self.assertTrue(substitute["counter_source"].startswith("processes[*].stages[*]"))
            self.assertEqual(substitute["hook_activation_lines"], [CORRECTNESS_HOOK_LINE])
        # Removing the process-scoped rows must fail closed: the parser really depends on that scope.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "correctness.json"
            path.write_text(json.dumps(correctness_artifact(processes=correctness_processes(rows=[]))),
                            encoding="utf-8")
            with self.assertRaises(AssertionError) as caught:
                gate.validate_rejection_substitute(path, exe=exe, config=config)
            self.assertIn("no rejected proposal", str(caught.exception))
            self.assertIn("read 0 row(s)", str(caught.exception))
        # A top-level stage list is still tolerated if a future gate ever emits one (with the
        # hook diagnostic likewise allowed at the top level instead of per process).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "correctness.json"
            path.write_text(json.dumps(correctness_artifact(
                processes=[], startup_diagnostics={"hook_lines": [CORRECTNESS_HOOK_LINE]},
                stages=[{"name": "batch", "stderr_diagnostics": {
                    "batch_mtp_stats": [dict(row) for row in CORRECTNESS_COUNTER_ROWS]}}])),
                encoding="utf-8")
            substitute = gate.validate_rejection_substitute(path, exe=exe, config=config)
            self.assertEqual(substitute["rejected_total"], 4)
            self.assertIn("top-level fallback", substitute["counter_source"])
            self.assertEqual(substitute["hook_activation_lines"], [CORRECTNESS_HOOK_LINE])
        # The hook-activation diagnostic stays a per-process read as well.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "correctness.json"
            path.write_text(json.dumps(correctness_artifact(processes=correctness_processes(hook_lines=[]))),
                            encoding="utf-8")
            with self.assertRaises(AssertionError) as caught:
                gate.validate_rejection_substitute(path, exe=exe, config=config)
            self.assertIn("hook activation", str(caught.exception))

    def test_rejection_substitute_validator_fails_closed_on_an_inconsistent_artifact(self):
        exe, config = Path("/tmp/fake-strata"), Path("/tmp/fake-model.json")
        cases = (
            ({"passed": False}, "passed correctness artifact"),
            ({"mode": "tails"}, "--mode correctness"),
            ({"coverage": {"correctness": "UNTESTED"}}, "correctness = RUN"),
            ({"exe": "/tmp/other-strata"}, "different --exe"),
            ({"config": "/tmp/other-model.json"}, "different --config"),
            ({"deterministic_proposal_schedule": {"slot1": {"outcome": "accepted"}}},
             "deterministic forced rejection"),
            ({"processes": correctness_processes(hook_lines=[])}, "hook activation"),
            ({"processes": correctness_processes(rows=[{"slot": 0, "offered": 8, "accepted": 8,
                                                          "rejected": 0, "discarded": 0}])},
             "no rejected proposal"),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmp:
                    path = write_correctness_artifact(tmp, **overrides)
                    with self.assertRaises(AssertionError) as caught:
                        gate.validate_rejection_substitute(path, exe=exe, config=config)
                    self.assertIn(message, str(caught.exception))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError) as caught:
                gate.validate_rejection_substitute(Path(tmp) / "missing.json", exe=exe, config=config)
            self.assertIn("not a file", str(caught.exception))

    def test_rejection_evidence_flag_is_validated_before_any_model_work(self):
        stderr = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(stderr):
            gate.main(["--exe", "/tmp/fake-strata", "--config", "/tmp/fake-model.json",
                       "--output", "/tmp/never-task34.json", "--gate", "pressure",
                       "--rejection-evidence", "/tmp/does-not-exist-task34.json"])
        self.assertIn("not a file", stderr.getvalue())

    def test_pressure_gate_runs_restore_witnesses_on_the_all_accept_shape(self):
        """The measured 100 %-acceptance shape: the restore witnesses still run and hold."""
        with tempfile.TemporaryDirectory() as tmp:
            evidence, source, _ = run_pressure_gate(Path(tmp), rows=all_accept_rows(),
                                                   substitute=valid_substitute(tmp))
        self.assertEqual(source["rejection_witness"]["mechanism"], gate.REJECTION_SUBSTITUTE_MECHANISM)
        self.assertEqual(source["rejection_evidence_mechanism"], gate.REJECTION_SUBSTITUTE_MECHANISM)
        self.assertEqual(source["natural_rejection_witness"], gate.NATURAL_REJECTION_ABSENT)
        self.assertTrue(evidence["rejection_evidence"]["observed"])
        self.assertTrue(evidence["rejection_evidence"]["substitute_used"])
        self.assertTrue(evidence["rejection_evidence"]["restore_witnesses_evaluated_before_enforcement"])
        restore = next(stage for stage in evidence["stages"] if stage["name"].startswith("positive-pressure"))
        self.assertTrue(restore["passed"])
        self.assertTrue(restore["positive_canonical_restore"])
        self.assertEqual(restore["actual_reused_cells"], restore["expected_reused_cells"])
        self.assertEqual(restore["restored_main_drafts_offered"], 0)
        self.assertTrue(restore["last_returned_token_unfed"])
        self.assertTrue(restore["exact_full_continuation_ids"])
        for key in ("canonical_positive_restore_MAIN_exact_continuation", "last_returned_token_unfed",
                    "target_only_restore_suppression_no_stale_offers",
                    "mandatory_pressure_BDONE_and_positive_target_only_park"):
            self.assertEqual(evidence["coverage"][key], "RUN")
        self.assertTrue(evidence["coverage"]["bounded_private_ring_wrap_with_actual_proposals"]
                        .startswith("RUN ("))

    def test_pressure_gate_accepts_the_natural_rejection_shape_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence, source, _ = run_pressure_gate(Path(tmp), rows=natural_rejection_rows())
        self.assertEqual(source["rejection_witness"]["mechanism"], "natural_native_verifier_rejection")
        self.assertEqual(source["natural_rejection_witness"], "present")
        self.assertTrue(evidence["rejection_evidence"]["observed"])
        self.assertFalse(evidence["rejection_evidence"]["substitute_used"])
        self.assertIsNone(evidence["rejection_evidence"]["substitute"])
        restore = next(stage for stage in evidence["stages"] if stage["name"].startswith("positive-pressure"))
        self.assertTrue(restore["passed"])

    def test_pressure_gate_fails_closed_without_a_substitute_but_still_runs_the_restore(self):
        """A missing rejection witness must fail closed AND must not suppress the restore witnesses."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError) as caught:
                run_pressure_gate(Path(tmp), rows=all_accept_rows())
            self.assertIn("--rejection-evidence", str(caught.exception))
            self.assertIn("not a pressure defect", str(caught.exception))
            # The restore stage was still evaluated and recorded before the verdict fired.
            evidence = json.loads((Path(tmp) / "pressure-evidence.json").read_text())
            restore = next(stage for stage in evidence["stages"] if stage["name"].startswith("positive-pressure"))
            self.assertTrue(restore["passed"])
            self.assertTrue(restore["positive_canonical_restore"])
            self.assertEqual(restore["restored_main_drafts_offered"], 0)
            self.assertEqual(evidence["coverage"]["canonical_positive_restore_MAIN_exact_continuation"], "RUN")
            self.assertFalse(evidence["rejection_evidence"]["observed"])
            self.assertEqual(evidence["rejection_evidence"]["mechanism"], "none")

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


# --- production run_handoff scoping regression ------------------------------
# These tests drive the real unified_kv_batch_mtp_pressure_smoke.run_handoff
# path (not a copy) with a fake native process and a real stderr file, so the
# continuation-vs-reference byte-range scoping is exercised end to end.
HANDOFF_REFERENCE_IDS = list(range(1000, 1064))
HANDOFF_SOURCE_IDS = HANDOFF_REFERENCE_IDS[:3]
HANDOFF_CONTINUATION_IDS = HANDOFF_REFERENCE_IDS[3:]
HANDOFF_SOURCE_STATS = counter_row(slot=0, windows=4, offered=3, accepted=3, rejected=0,
                                   discarded=0, attempts=1, limits=1, reserve=0)
HANDOFF_SUPPRESSION_LINES = "\n".join([
    TARGET_ONLY_CLONE_LINE,
    "strata batch: slot 0 gave back 1529 tokens of this conversation (its turn checkpoint) in 19.7 ms",
    "strata serve: TARGET_ONLY decode: T=1, MTP/suffix proposals disabled",
])


class FakeHandoffEngine:
    """Scripted native stdout plus real stderr writes for the production gate."""

    def __init__(self, suite):
        self.suite = suite
        self.stage = None
        self.queue = deque()

    def _write(self, text):
        with self.suite.stderr.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    def send(self, *lines, deadline=None):
        for line in lines:
            if line.startswith("BGEN"):
                self._write(HANDOFF_SOURCE_STATS)
                self.queue.extend([
                    f"T {HANDOFF_SOURCE_IDS[0]}",
                    f"DONE 1 {len(self.suite.last_attempt.prompt)} 10 5 length 0 0 0 0 0 0 0.0 98 0",
                    "BADM 0 1",
                    f"BT 0 {HANDOFF_SOURCE_IDS[1]}",
                    f"BT 0 {HANDOFF_SOURCE_IDS[2]}",
                    "BDONE 0 3 handoff 91.9"])
            elif line.startswith("GEN"):
                prompt_len = len(self.suite.last_attempt.prompt)
                self._write("strata serve: prompt %d tokens = %d reused + 1 read in 37 ms, 61 generated, "
                            "drafts accepted 30 of 30, 0 checkpoints" % (prompt_len, prompt_len - 1))
                if self.suite.suppression_in_continuation:
                    self._write(HANDOFF_SUPPRESSION_LINES)
                self.queue.extend(f"T {token}" for token in HANDOFF_CONTINUATION_IDS)
                self.queue.append(f"DONE {len(HANDOFF_CONTINUATION_IDS)} {prompt_len} 36.9 1960.5 length "
                                  f"30 30 {prompt_len - 1} 4797 29280 0 0 0.0 {prompt_len} 0")

    def next_event(self, deadline=None):
        raw = self.queue.popleft()
        seq = self.suite.sequence
        self.suite.sequence += 1
        return common.parse_line(raw) | {"raw": raw, "seq": seq, "wall_s": seq / 100}


class FakeHandoffSuite:
    """Minimal suite surface used by production run_handoff."""

    def __init__(self, stderr, evidence, *, suppression_in_continuation=False):
        self.stderr = stderr
        self.evidence = evidence
        self.suppression_in_continuation = suppression_in_continuation
        self.last_attempt = None
        self.sequence = 0
        self.engine = FakeHandoffEngine(self)

    def make_attempt(self, name, prompt, cap, slot=None):
        self.last_attempt = common.Request(name, list(prompt), cap, slot)
        return self.last_attempt

    def run(self, name, requests):
        stage = {"name": name, "passed": False, "requests": []}
        self.evidence.setdefault("stages", []).append(stage)
        protocol = common.Protocol(requests)
        self.engine.send(*(req.command() for req in requests))
        while not protocol.finished:
            protocol.consume(self.engine.next_event())
        stage["requests"] = [req.record() for req in requests]
        return stage

    def solo(self, name, prompt, cap):
        # The same-binary reference legitimately reuses a turn checkpoint and
        # goes target-only. Its diagnostics must never be read as the
        # continuation's own downgrade.
        self.engine._write(HANDOFF_SUPPRESSION_LINES)
        req = common.Request(name, list(prompt), cap)
        req.tokens = list(HANDOFF_REFERENCE_IDS)
        return req

    def save(self):
        pass


class HandoffScopingTests(unittest.TestCase):
    def drive(self, *, suppression_in_continuation=False):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = Path(tmp) / "handoff.stderr.log"
            stderr.write_text("", encoding="utf-8")
            # Geometry is supplied so the handoff ring-wrap verdict is computed for real.
            evidence = {"stages": [], "coverage": {},
                        "source_ring": {"allocated_page_rounded_ring_cells": 50}}
            suite = FakeHandoffSuite(stderr, evidence,
                                     suppression_in_continuation=suppression_in_continuation)
            args = types.SimpleNamespace(handoff_cap=64, stage_timeout=30)
            gate.run_handoff(args, evidence, suite, stderr, [list(range(1, 101))])
            return evidence["stages"][0], evidence

    def test_reference_suppression_after_the_continuation_is_not_misattributed(self):
        stage, evidence = self.drive(suppression_in_continuation=False)
        self.assertTrue(stage["passed"])
        continuation = stage["continuation_byte_range"]
        reference = stage["reference_byte_range"]
        self.assertLess(continuation[0], continuation[1])
        self.assertEqual(continuation[1], reference[0])
        self.assertLess(reference[0], reference[1])
        self.assertTrue(stage["continuation_range_clean"])
        self.assertEqual([item["stage"] for item in stage["suppression_offsets"]], ["reference"])
        self.assertTrue(all(reference[0] <= item["offset"] < reference[1]
                            for item in stage["suppression_offsets"]))
        # L8: the diagnostic absence check is recorded as corroborating, with the
        # resumed-offers counter named as the primary anti-downgrade guard.
        guard = stage["continuation_downgrade_guard"]
        self.assertIn("resumed MAIN draft offers", guard["primary_guard"])
        self.assertEqual(guard["diagnostics_present_in_continuation_range"], [])
        self.assertIn("corroborating only", guard["scope"])
        # L4: the coverage value is a real computed verdict, not a conditional string.
        coverage = evidence["coverage"]["bounded_private_ring_wrap_with_actual_proposals"]
        self.assertTrue(coverage.startswith("RUN ("))
        self.assertIn("evidence_limit", coverage)
        self.assertNotIn("RUN if", coverage)
        self.assertEqual(evidence["coverage_evidence"]
                         ["bounded_private_ring_wrap_with_actual_proposals"]["verdict"], "RUN")

    def test_suppression_inside_the_continuation_range_fails_closed(self):
        with self.assertRaisesRegex(AssertionError, "incorrectly downgraded to target-only"):
            self.drive(suppression_in_continuation=True)


if __name__ == "__main__":
    unittest.main()
