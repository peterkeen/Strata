#!/usr/bin/env python3
"""Bounded, private model gates for unified-KV batch-MTP pressure and ring wrap.

No builds, downloads, remote actions, or service interaction. Uses the native
binary as a private subprocess and writes complete JSON/stdout/stderr evidence.

Pressure + bounded private MTP ring wrap (4096 cells, 1536/1540 prompt cells,
768 output tokens, positive parking budget):
  python3 tools/unified_kv_batch_mtp_pressure_smoke.py --exe /path/to/strata \
      --config model.json --output /tmp/mtp-pressure-UNIQUE.json --gate pressure

Optional speculative-row shortage separated from mandatory pressure (dual slot; the
measured shape is one slot reporting fallback_reserve on its optional page boundary
and then parking on its own mandatory row, or both slots recovering and cancelling):
  same command with --gate optional

Full coherent active BHANDOFF back to MAIN and exact continuation parity:
  same command with --gate handoff

All modes explicitly pass --batch-mtp and STRATA_BATCH_MTP=1, use
--spec 2/--mtp-max-t 2 (one main-path draft), fix target placement/sampling,
and bound prompts/outputs/timeouts. Pressure/optional references run in a
separate clean process from their source; handoff references run only after the
continuation. Each process gets a fresh stderr/evidence range. A model/GPU gate
that cannot demonstrate its condition fails; it is never credited from sizing
arithmetic or scripted protocol alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import incremental_kv_smoke as incremental  # noqa: E402
import unified_kv_smoke as common  # noqa: E402
import unified_kv_streaming_smoke as streaming  # noqa: E402

PAGE = common.PAGE_CELLS
require = common.require
Request = common.Request
Attempt = incremental.Attempt
GUARD = incremental.GUARD

STATS_RE = re.compile(
    r"^strata batch_mtp_stats slot=(\d+) windows=(\d+) offered=(\d+) accepted=(\d+) rejected=(\d+) "
    r"discarded=(\d+) fallback_attempts=(\d+) fallback_incoherent=(\d+) fallback_not_ready=(\d+) "
    r"fallback_limits=(\d+) fallback_capacity=(\d+) fallback_reserve=(\d+)$")
RING_FALLBACK = "strata mtp: no pinned RAM left for the draft layer's K/V copy; keeping it in VRAM"


def parse_batch_stats(text: str) -> list[dict]:
    """Parse one byte-range's counters; slot IDs may recur in later lifecycles."""
    records = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if "strata batch_mtp_stats" not in line:
            continue
        match = STATS_RE.fullmatch(line.strip())
        require(match is not None, f"malformed batch-MTP summary at stderr line {line_no}: {line}")
        names = ("slot", "windows", "offered", "accepted", "rejected", "discarded", "fallback_attempts",
                 "fallback_incoherent", "fallback_not_ready", "fallback_limits", "fallback_capacity",
                 "fallback_reserve")
        item = {key: int(value) for key, value in zip(names, match.groups())}
        require(all(value >= 0 for key, value in item.items() if key != "slot"), "negative native MTP counter")
        require(item["offered"] == item["accepted"] + item["rejected"] + item["discarded"],
                "offered != accepted + rejected + discarded")
        require(item["fallback_attempts"] == sum(item[key] for key in
                ("fallback_incoherent", "fallback_not_ready", "fallback_limits", "fallback_capacity",
                 "fallback_reserve")), "fallback reasons do not sum to fallback_attempts")
        require(item["windows"] == item["offered"] + item["fallback_attempts"],
                "windows != offered + fallback_attempts")
        item["raw"] = line
        records.append(item)
    return records


def parse_diagnostics(text: str) -> dict:
    records = incremental.pressure_diagnostics(text)
    stats = parse_batch_stats(text)
    pressure = [record for record in records if record["kind"] == "pressure_target_only"]
    require(all(record["bytes"] > 0 and record["tokens"] > 0 for record in pressure),
            "pressure target-only park must have positive token and snapshot-byte counts")
    cache_records = streaming.diagnostics(text)
    return {"batch_mtp_stats": stats, "pressure": records, "canonical_cache": cache_records,
            "target_only_restores": [r for r in records if r["kind"] == "target_only_restore"],
            "target_only_t1": [r for r in records if r["kind"] == "target_only_decode_t1"],
            "raw_stderr": text}


def verify_stats(stats: list[dict], *, offer=False, every_slot_offer=False, accept=False, reject=False,
                 fallback_reserve=False):
    require(stats, "missing native batch-MTP lifecycle counters")
    if offer:
        require(sum(s["offered"] for s in stats) > 0, "no verifier proposal rows were offered")
    if every_slot_offer:
        require(all(s["offered"] > 0 for s in stats),
                "every private slot must have actual proposal offers; aggregate-only offers are insufficient")
    if accept:
        require(sum(s["accepted"] for s in stats) > 0, "no proposal was accepted")
    if reject:
        require(sum(s["rejected"] for s in stats) > 0, "no proposal was rejected")
    if fallback_reserve:
        require(sum(s["fallback_reserve"] for s in stats) > 0,
                "no optional speculative-row reservation fallback was observed")
    return {"slots": stats, "windows": sum(s["windows"] for s in stats),
            "offered": sum(s["offered"] for s in stats), "accepted": sum(s["accepted"] for s in stats),
            "rejected": sum(s["rejected"] for s in stats),
            "fallback_reserve": sum(s["fallback_reserve"] for s in stats)}


def ring_capacity_from_source(window: int, spec: int, context: int) -> dict:
    """Evaluate the checked-in production helper; fail if its expression changes."""
    header = Path(__file__).resolve().parents[1] / "include/strata/core/mtp.hpp"
    text = header.read_text(encoding="utf-8")
    match = re.search(r"inline constexpr int64_t mtp_kv_ring_cells\(int64_t window, int64_t max_cells, int max_t\) \{\s*"
                      r"return \(window > 0 && window < max_cells\) \? window \+ 4 \* \(int64_t\) max_t \+ 64 : -1;\s*\}", text)
    require(match is not None, "production MTP ring allocation helper changed; update the evidence calculation")
    ring = window + 4 * spec + 64 if 0 < window < context else -1
    allocated = pages(ring) * PAGE if ring > 0 else context
    return {"source": str(header), "source_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
            "helper_expression": "window + 4 * max_t + 64 when 0 < window < max_cells, otherwise full context",
            "window_cells": window, "spec_max_t": spec, "target_slot_max_cells": context,
            "requested_private_ring_cells": ring, "allocated_page_rounded_ring_cells": allocated,
            "bounded_ring": ring > 0, "prompt_must_wrap_ring": ring > 0}


def pages(cells: int) -> int:
    require(cells >= 0, "negative cell extent")
    return (cells + PAGE - 1) // PAGE


def pressure_capacity_plan(prompts, caps, context):
    require(len(prompts) == len(caps) == 2 and context <= 4096 and context % PAGE == 0,
            "pressure gate requires two requests and <=4096 four-cell-aligned backing")
    pool_pages = context // PAGE
    initial = [pages(len(p) + min(cap, incremental.AHEAD)) for p, cap in zip(prompts, caps)]
    individual = [pages(len(p) + cap + 2) for p, cap in zip(prompts, caps)]
    require(all(len(p) + cap + GUARD <= context for p, cap in zip(prompts, caps)),
            "each prompt/output allowance must fit logical context with native guard")
    require(sum(initial) <= pool_pages, "initial prompt + headroom + speculative-row margins do not fit backing")
    require(sum(individual) > pool_pages, "combined individual histories cannot exhaust shared backing")
    common_tokens = 0
    for a, b in zip(*prompts):
        if a != b:
            break
        common_tokens += 1
    shared_upper = pages(common_tokens)
    require(sum(individual) - shared_upper > pool_pages,
            "fixture could fit even under hypothetical common-prefix sharing")
    return {"pool_cells": context, "pool_pages": pool_pages, "page_cells": PAGE,
            "prompt_cells": [len(p) for p in prompts], "output_caps": caps,
            "initial_pages_upper_bound_with_reserve_ahead": initial,
            "initial_pages_total": sum(initial), "individual_final_pages_upper_bound": individual,
            "individual_final_pages_total": sum(individual), "common_prefix_cells": common_tokens,
            "common_prefix_pages_upper_bound": shared_upper,
            "combined_individual_histories_exceed_backing": True,
            "arithmetic_is_fixture_sizing_only_not_pressure_evidence": True}


OPTIONAL_WITNESS_MIN_MARGIN = 16

# Source-derived park attribution. In this stage a pressure terminal is only reachable from the mandatory
# [p, p+1) reservation retry; the optional probe cannot reclaim and cannot park. Park ORDERING is not
# observable: the native prints the lifecycle counters and the park line only at the terminal, so nothing in
# the diagnostic contract can order a park against the optional window.
PARK_ATTRIBUTION_SOURCE = {
    "batch_mandatory_retry": {
        "file": "src/program/generate.cpp", "lines": "8739-8748 (batch_step)",
        "fact": "the mandatory [p, p+1) row is reserved first, and only a shortage that survives the "
                "reclaim_shared() retry reaches finish_shared_slot(b, \"pressure\") at line 8746"},
    "batch_optional_probe": {
        "file": "include/strata/core/batch_mtp_policy.hpp", "lines": "35-40",
        "fact": "batch_mtp_choose_window() calls reserve_optional(p + 2) once and maps a shortage straight to "
                "BatchMtpFallback::reservation: no reclaim retry, no pressure path"},
    "batch_optional_caller": {
        "file": "src/program/generate.cpp", "lines": "8758",
        "fact": "the caller only clears err for BatchMtpFallback::reservation and keeps that window target-only"},
    "park_site": {
        "file": "src/program/generate.cpp", "lines": "8503-8507",
        "fact": "finish_shared_slot(..., \"pressure\") runs park_slot_target(b) first, so the park line belongs to "
                "the slot whose terminal is BDONE <slot> ... pressure"},
    "other_pressure_site": {
        "file": "src/program/generate.cpp", "lines": "9095-9103",
        "fact": "a second pressure site parks a victim for the frontend prefill/admission path on seq 0; it is not "
                "reachable from a plain dual-slot decode stage with no prefill and no incoming request"},
    "ordering": {
        "file": "-", "lines": "-",
        "fact": "park ordering relative to the optional window is not observable (terminal-only diagnostics)"},
    "conclusion": "in this stage a pressure terminal is attributable to the mandatory reservation retry, and the "
                  "optional probe cannot escalate to reclaim or pressure. The witness proves the optional-shortage "
                  "fallback was taken and, by source inspection, does not escalate from the optional probe itself; "
                  "it does not prove park ordering.",
}

ACCEPTED_OPTIONAL_SHAPES = ("mandatory_park_then_cancel", "no_park_both_cancel")


def optional_slot_boundary(prompt_cells: int, ahead_cells: int = incremental.AHEAD) -> dict:
    """Per-slot page-boundary arithmetic behind the bounded stop rule (sizing, not evidence)."""
    require(prompt_cells > ahead_cells and prompt_cells % PAGE == PAGE - 1,
            "optional fixture prompts must end on the last cell of a page")
    require(ahead_cells > 0 and ahead_cells % PAGE == 0, "reserve-ahead must be a positive page multiple")
    mapping_end = pages(prompt_cells + ahead_cells) * PAGE
    boundary = mapping_end - prompt_cells
    require(boundary == ahead_cells + 1, "boundary output count must equal reserve-ahead + 1")
    return {"prompt_cells": prompt_cells, "prompt_page_offset": prompt_cells % PAGE,
            "reserve_ahead_cells": ahead_cells, "slot_mapping_end_cells": mapping_end,
            "slot_mapping_pages": mapping_end // PAGE,
            "boundary_output_count": boundary, "bstop_output_count": boundary + 1}


def optional_fixture_sizing(prompts, cap, context) -> dict:
    """Sizing only: the same dual fixture that produced the measured witness shape on hardware (task12).

    The task14 single-slot "pool exactly full" derivation did not reproduce (task15), so no spare/free-page
    claim is gated on here; the witness is the native fallback_reserve counter.
    """
    require(len(prompts) == 2 and prompts[0] != prompts[1], "optional fixture needs two distinct prompts")
    plans = [optional_slot_boundary(len(prompt)) for prompt in prompts]
    require(all(len(prompt) + cap + GUARD <= context for prompt in prompts),
            "each optional fixture prompt plus its cap must fit the logical context")
    require(cap >= max(plan["bstop_output_count"] for plan in plans) + OPTIONAL_WITNESS_MIN_MARGIN,
            "optional cap must clear the boundary stop point with margin")
    mapped = sum(plan["slot_mapping_pages"] for plan in plans)
    pool_pages = context // PAGE
    return {"context_cells": context, "pool_pages": pool_pages, "output_cap": cap, "slots": plans,
            "slot_mapping_pages_total": mapped, "mapping_pages_exceed_pool": mapped > pool_pages,
            "boundary_output_counts": [plan["boundary_output_count"] for plan in plans],
            "bstop_output_counts": [plan["bstop_output_count"] for plan in plans],
            "sizing_is_hypothesis_not_evidence": True,
            "measured_shape_reference": "task12 dual run: one slot fallback_reserve=1 then BDONE pressure, the "
                                        "other healthy cancel, exactly one park",
            "note": "no free-page/spare-page claim is gated on; the witness is the native fallback_reserve counter"}


def optional_stop_decision(requests, plans) -> str | None:
    """Bounded stop rule, arithmetic-justified per slot (p = prompt_cells + produced - 1).

    Stop as soon as a mandatory park appears (the witness slot's own terminal), otherwise once both slots have
    passed their own optional page boundary. Neither branch depends on equal acceptance between the slots.
    """
    if any(r.completion is not None and r.completion.get("finish") == "pressure" for r in requests):
        return "mandatory_park_observed"
    if any(r.completion is not None for r in requests):
        return "unexpected_terminal_observed"
    if (all(r.completion is None for r in requests) and
            all(len(r.tokens) >= plan["bstop_output_count"] for r, plan in zip(requests, plans))):
        return "both_boundaries_crossed_no_park"
    return None


def optional_park_attribution(witness, parks) -> dict:
    """Attribute the park line to the witness slot. The raw diagnostic has no slot field, so it is matched by
    the parked prefix (prompt + produced - 1) and by the pressure terminal's slot."""
    expected = None if witness is None else len(witness.prompt) + len(witness.tokens) - 1
    observed = sorted(record["tokens"] for record in parks)
    return {"diagnostic_has_slot_field": False,
            "witness_slot": None if witness is None else witness.slot,
            "witness_consumed_prefix_cells": expected,
            "park_tokens": observed,
            "attributed_by_token_count": expected is not None and expected in observed,
            "parks": len(parks),
            "ordering_observable": False,
            "source": PARK_ATTRIBUTION_SOURCE}


def optional_shape_checks(requests, diagnostics, stop_reason, overlap_ok, parity_ok) -> dict:
    """Named witnesses for the two accepted measured shapes; anything else fails closed."""
    stats = diagnostics["batch_mtp_stats"]
    parks = [record for record in diagnostics["pressure"] if record["kind"] == "pressure_target_only"]
    completions = {r.slot: (r.completion or {}) for r in requests}
    finishes = {slot: completion.get("finish") for slot, completion in completions.items()}
    other_reasons = ("fallback_incoherent", "fallback_not_ready", "fallback_limits", "fallback_capacity")
    witnesses = [row for row in stats if row["fallback_reserve"] >= 1]
    witness_row = witnesses[0] if len(witnesses) == 1 else None
    witness = next((r for r in requests if witness_row is not None and r.slot == witness_row["slot"]), None)
    others = [row for row in stats if witness_row is not None and row["slot"] != witness_row["slot"]]
    attribution = optional_park_attribution(witness, parks)
    park_shape = (witness_row is not None and len(parks) == 1 and
                  finishes.get(witness_row["slot"]) == "pressure" and len(others) == len(requests) - 1 and
                  all(finishes.get(row["slot"]) == "cancel" for row in others))
    no_park_shape = (witness_row is not None and not parks and len(others) == len(requests) - 1 and
                     all(completion.get("finish") == "cancel" for completion in completions.values()))
    checks = {
        "both_slots_admitted_and_overlapping": (bool(overlap_ok), {"overlap": overlap_ok}),
        "positive_offers_per_slot":
            (len(stats) == len(requests) and all(row["offered"] > 0 for row in stats), {"rows": stats}),
        "exactly_one_witness_slot_fallback_reserve":
            (witness_row is not None,
             {"fallback_reserve": {row["slot"]: row["fallback_reserve"] for row in stats}}),
        "witness_slot_no_other_fallback_reason":
            (witness_row is not None and sum(witness_row[key] for key in other_reasons) == 0,
             {"witness_row": witness_row,
              "other_fallback_reasons": None if witness_row is None
              else sum(witness_row[key] for key in other_reasons)}),
        "target_parity_both_slots": (bool(parity_ok), {"parity": parity_ok}),
        "bounded_stop_rule_fired": (stop_reason is not None, {"stop_reason": stop_reason}),
        "accepted_terminal_shape":
            (park_shape or no_park_shape,
             {"shape": "mandatory_park_then_cancel" if park_shape else
                       ("no_park_both_cancel" if no_park_shape else None),
              "finishes": finishes, "parks": len(parks)}),
        "at_most_one_park": (len(parks) <= 1, {"parks": len(parks)}),
        "park_attribution_consistent": (not parks or attribution["attributed_by_token_count"], attribution),
    }
    shape = ("mandatory_park_then_cancel" if park_shape else
             ("no_park_both_cancel" if no_park_shape else None))
    return {"shape": shape,
            "checks": {name: {"observed": ok, "detail": detail} for name, (ok, detail) in checks.items()},
            "missed": [name for name, (ok, _) in checks.items() if not ok],
            "rows": stats, "parks": parks, "park_attribution": attribution,
            "accepted_shapes": list(ACCEPTED_OPTIONAL_SHAPES)}


def validate_optional_shapes(requests, diagnostics, stop_reason, overlap_ok, parity_ok) -> dict:
    """Fail closed, naming every missed witness; the coverage is never credited from arithmetic."""
    result = optional_shape_checks(requests, diagnostics, stop_reason, overlap_ok, parity_ok)
    if result["missed"]:
        raise AssertionError(
            "optional witness shapes failed: missed " + ", ".join(result["missed"]) +
            " (fixture non-isolation: the optional reservation path was not isolated from mandatory pressure; "
            "coverage stays UNTESTED)")
    return result


def pressure_overlap_proof(requests):
    require(len(requests) == 2 and all(r.badm and r.badm["continues"] for r in requests),
            "pressure overlap requires both slots admitted")
    second_badm = max(r.badm["seq"] for r in requests)
    first_done = min(r.completion["seq"] for r in requests if r.completion is not None)
    require(first_done > second_badm, "a request completed before both admissions")
    progress = {str(r.slot): [e["seq"] for e in r.token_events
                              if e["kind"] == "BT" and second_badm < e["seq"] < first_done]
                for r in requests}
    require(all(progress.values()), "both slots must emit BT before first completion")
    return {"both_BADM_seq": [r.badm["seq"] for r in requests], "second_BADM_seq": second_badm,
            "first_BDONE_seq": first_done, "per_slot_interleaved_BT_seq": progress,
            "actual_dual_slot_progress": True}


def pressure_replay_ids(original: list[int], returned: list[int]) -> tuple[list[int], int]:
    require(original and returned, "pressure replay needs original prompt and at least one returned token")
    prompt = list(original) + list(returned)  # The final returned token is still unfed.
    return prompt, len(prompt) - 1


def require_pressure_terminal(req: Request) -> dict:
    require(req.completion is not None and req.completion["kind"] == "BDONE" and
            req.completion["finish"] == "pressure", "mandatory shortage must be BDONE pressure")
    require(req.tokens and len(req.tokens) < req.cap, "pressure must make positive, nonfinal progress")
    return req.completion


def verify_target_only_resume(done_raw: str, diagnostics: dict, expected_reused: int, actual_reused: int) -> dict:
    require(expected_reused > 0 and actual_reused == expected_reused,
            "positive restore must reuse the entire consumed prefix; replay is not a pass")
    require(any(d["kind"] == "restore" and d["tokens"] == expected_reused
                for d in diagnostics["canonical_cache"]),
            "missing positive canonical restore for exact prefix")
    require(diagnostics["target_only_restores"] and diagnostics["target_only_t1"],
            "target-only restore must report T=1 suppression")
    fields = done_raw.split()
    require(len(fields) >= 8 and int(fields[7]) == 0,
            "target-only MAIN continuation offered stale private drafts")
    return {"expected_reused_cells": expected_reused, "actual_reused_cells": actual_reused,
            "last_returned_token_unfed": True, "draft_offers": int(fields[7]),
            "positive_target_only_restore": True}


def validate_deadlines(startup: float, stage: float, cleanup: float) -> None:
    require(all(math.isfinite(x) and 0 < x <= 7200 for x in (startup, stage, cleanup)),
            "timeouts must be finite, positive and no greater than two hours")


def argument_value(command: list[str], option: str) -> str:
    values = []
    i = 0
    while i < len(command):
        if command[i] == option:
            require(i + 1 < len(command), f"{option} missing value")
            values.append(command[i + 1])
            i += 2
        elif command[i].startswith(option + "="):
            values.append(command[i].split("=", 1)[1])
            i += 1
        else:
            i += 1
    require(len(values) == 1, f"expected exactly one {option}, got {values}")
    return values[0]


def validate_main_mtp_contract(command: list[str], mtp_max: int = 2) -> dict:
    spec = int(argument_value(command, "--spec"))
    cap = int(argument_value(command, "--mtp-max-t"))
    require(spec == 2 and cap == mtp_max == 2,
            "main-path MTP offers require --spec 2 and --mtp-max-t 2 (one draft per window)")
    return {"spec": spec, "mtp_max_t": cap, "main_max_drafts_per_window": cap - 1,
            "expected_main_draft_offers": "positive on coherent solo/continued MAIN paths"}


def settings(cfg: dict, exe: Path, context: int, cache_mib: int, gpu, mtp_window: int,
             vram_reserve_mib: int):
    require(context == 4096, "this bounded fixture is calibrated for context/backing 4096")
    require(cache_mib > 0, "pressure and target-only restore require a positive parking budget")
    require(mtp_window > 0, "bounded-ring gate requires a positive --mtp-window")
    # incremental.settings owns the stable streaming/model-path/device setup.
    # Remove these controlled source options first to prevent duplicate flags.
    clean = dict(cfg)
    original = list(cfg.get("args") or [])
    filtered = []
    i = 0
    while i < len(original):
        arg = original[i]
        if arg == "--mtp-window":
            require(i + 1 < len(original), "config --mtp-window missing value")
            i += 2
        elif arg.startswith("--mtp-window=") or arg == "--batch-mtp" or arg.startswith("--batch-mtp="):
            i += 1
        else:
            filtered.append(arg)
            i += 1
    clean["args"] = filtered
    command, cwd, env, kv = incremental.settings(clean, exe, context, 0, cache_mib, gpu,
                                                  vram_reserve_mib=vram_reserve_mib,
                                                  coherence_mtp_max=2, prefill=64)
    require(command.count("--kv-unified") == 1 and command.count("--batch") == 1,
            "shared incremental helper did not produce a single unified batch command")
    command.extend(("--batch-mtp", "--mtp-window", str(mtp_window)))
    require(command.count("--batch-mtp") == 1 and command.count("--mtp-window") == 1,
            "native process must use one explicit batch-MTP switch and bounded MTP window")
    require("--mtp" in command and command.count("--mtp") == 1,
            "config must retain exactly one real MTP artifact")
    main_mtp_contract = validate_main_mtp_contract(command, 2)
    env["STRATA_BATCH_MTP"] = "1"
    env["STRATA_BATCH_DECODE_SHARE"] = "0"
    env["STRATA_IQ_MT_MIN"] = "1"
    env["STRATA_KV_GROW"] = "0"
    env.pop("STRATA_BATCH_MTP_TEST_PROPOSALS", None)
    return command, cwd, env, kv, main_mtp_contract


class BatchMtpEngine(incremental.Engine):
    def start(self, command, cwd, env, context, resident, cache_mib, timeout, mtp_max=2):
        super().start(command, cwd, env, context, resident, cache_mib, timeout, mtp_max)


class PressureSuite(incremental.Suite):
    def __init__(self, *args, stderr_path: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.stderr_path = stderr_path

    def run_source_pair(self, prompts, cap, references, deadline_s):
        requests = [self.make_attempt(f"pressure-source-slot-{i}", p, cap, i) for i, p in enumerate(prompts)]
        for req, ref in zip(requests, references):
            req.reference = ref
        self.engine.stage = "dual-slot-MTP-pressure-park-stop-sibling"
        stderr_start = self.stderr_path.stat().st_size
        stage = {"name": self.engine.stage, "passed": False, "requests": []}
        self.evidence["stages"].append(stage)
        protocol = common.Protocol(requests)
        deadline = time.monotonic() + deadline_s
        pressure_owner = None
        stop_sent = False
        try:
            self.engine.send(*(r.command() for r in requests), deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                protocol.consume(event)
                if event["kind"] == "BDONE" and event["finish"] == "pressure":
                    require(pressure_owner is None, "more than one owner pressure-parked before sibling stop")
                    pressure_owner = next((r for r in requests if r.slot == event["slot"]), None)
                    require(pressure_owner is not None, "pressure BDONE has no request owner")
                    siblings = [r for r in protocol.active.values() if r is not pressure_owner]
                    require(len(siblings) == 1, "first pressure must leave exactly one live private sibling to stop")
                    sibling = siblings[0]
                    # Demand real overlap before acting on first pressure: each slot already emitted BT.
                    require(all(any(e["kind"] == "BT" for e in r.token_events) for r in requests),
                            "pressure happened before both slots demonstrated BT overlap")
                    self.engine.send(f"BSTOP {sibling.slot}", deadline=deadline)
                    stage["stopped_sibling_slot"] = sibling.slot
                    stage["bstop_command_after_seq"] = event["seq"]
                    stop_sent = True
            require(pressure_owner is not None and stop_sent, "no actual pressure park and sibling BSTOP")
            sibling = next(r for r in requests if r is not pressure_owner)
            require(sibling.completion and sibling.completion["kind"] == "BDONE" and
                    sibling.completion["finish"] == "cancel",
                    "remaining private sibling was not safely cancelled after first pressure")
            require_pressure_terminal(pressure_owner)
            for req, ref in zip(requests, references):
                require(req.tokens and req.tokens == ref.tokens[:len(req.tokens)],
                        f"{req.name}: pressure/cancel prefix differs from same-binary solo reference")
            raw = self.stderr_path.read_bytes()
            text = raw[stderr_start:].decode("utf-8", errors="replace")
            diagnostics = parse_diagnostics(text)
            parks = [p for p in diagnostics["pressure"] if p["kind"] == "pressure_target_only" and
                     p["result"] == "parked" and p["bytes"] > 0]
            expected_park_tokens = len(pressure_owner.prompt) + len(pressure_owner.tokens) - 1
            require(any(p["tokens"] == expected_park_tokens for p in parks),
                    "first pressure lacked a positive target-only parked snapshot for the consumed prefix")
            stats = diagnostics["batch_mtp_stats"]
            proof = verify_stats(stats, offer=True, every_slot_offer=True, accept=True, reject=True)
            by_slot = {s["slot"]: s for s in stats}
            require(len(stats) == 2 and set(by_slot) == {0, 1},
                    "each active slot needs exactly one per-attempt MTP counter summary")
            stage.update(requests=[r.record() for r in requests], first_pressure_owner=pressure_owner.slot,
                         first_pressure_completion=pressure_owner.completion,
                         positive_pressure_park=next(p for p in parks if p["tokens"] == expected_park_tokens),
                         expected_pressure_park_tokens=expected_park_tokens,
                         sibling_cancel_completion=sibling.completion,
                         overlap=pressure_overlap_proof(requests),
                         batch_mtp_proof=proof,
                         natural_accept_reject_scope="aggregate across both slots; offers are independently required per slot",
                         per_slot_natural_accept_reject={str(k): {"accepted": v["accepted"],
                                                               "rejected": v["rejected"],
                                                               "offered": v["offered"]}
                                                        for k, v in by_slot.items()},
                         stderr_byte_range=[stderr_start, len(raw)], diagnostics=diagnostics)
            stage["passed"] = True
            self.save()
            return pressure_owner, stage
        finally:
            stage.setdefault("requests", [r.record() for r in requests])
            raw = self.stderr_path.read_bytes() if self.stderr_path.exists() else b""
            stage.setdefault("stderr_byte_range", [stderr_start, len(raw)])
            if not stage.get("diagnostics"):
                stage["diagnostics"] = parse_diagnostics(raw[stderr_start:].decode("utf-8", errors="replace"))
            self.save()

    def run_optional_row_fallback(self, prompts, cap, refs, plans, deadline_s):
        """Dual-slot optional-reservation witness.

        Measured runtime shape (task12): both slots grow until reclaim can free nothing, the slot that crosses
        its optional page boundary last reports fallback_reserve on that window, and its next mandatory row then
        parks (BDONE pressure) while the other slot stays healthy. A no-park variant (both slots recover by
        reclaim and cancel) is accepted too. Everything else fails closed with the missed witness named.
        """
        requests = [self.make_attempt(f"optional-row-slot-{i}", prompt, cap, i)
                    for i, prompt in enumerate(prompts)]
        for request, reference in zip(requests, refs):
            request.reference = reference
        self.engine.stage = "optional-row-fallback-witness"
        stderr_start = self.stderr_path.stat().st_size
        stage = {"name": self.engine.stage, "passed": False, "requests": [],
                 "stop_plan": {str(index): plan for index, plan in enumerate(plans)},
                 "output_cap": cap, "accepted_shapes": list(ACCEPTED_OPTIONAL_SHAPES),
                 "stop_rule": "stop on the first mandatory park, else once both slots passed their own optional "
                              "page boundary (p = prompt_cells + produced - 1); independent of acceptance"}
        self.evidence["stages"].append(stage)
        protocol = common.Protocol(requests)
        deadline = time.monotonic() + deadline_s
        stop_reason = None
        try:
            self.engine.send(*(r.command() for r in requests), deadline=deadline)
            while not protocol.finished:
                event = self.engine.next_event(deadline)
                protocol.consume(event)
                if stop_reason is None:
                    reason = optional_stop_decision(requests, plans)
                    if reason is not None:
                        survivors = list(protocol.active.values())
                        require(survivors, f"stop rule ({reason}) fired with no live slot to stop")
                        self.engine.send(*(f"BSTOP {r.slot}" for r in survivors), deadline=deadline)
                        stop_reason = reason
                        stage["bstop_after_seq"] = event["seq"]
                        stage["bstop_reason"] = reason
                        stage["stopped_slots"] = [r.slot for r in survivors]
            stage["bt_counts_at_bstop"] = {str(r.slot): len(r.tokens) for r in requests}
            drifted = [r.slot for r, ref in zip(requests, refs) if r.tokens != ref.tokens[:len(r.tokens)]]
            require(not drifted, f"optional witness: target prefix parity differs for slots {drifted}")
            overlap_ok, overlap_detail = True, None
            try:
                overlap_detail = pressure_overlap_proof(requests)
            except AssertionError as exc:
                overlap_ok, overlap_detail = False, str(exc)
            raw = self.stderr_path.read_bytes()
            text = raw[stderr_start:].decode("utf-8", errors="replace")
            diagnostics = parse_diagnostics(text)
            try:
                witness = validate_optional_shapes(requests, diagnostics, stop_reason, overlap_ok, not drifted)
            except AssertionError:
                stage["fixture_isolation"] = "fixture_non_isolation"
                stage["overlap_detail"] = overlap_detail
                raise
            stage.update(requests=[r.record() for r in requests], witness=witness,
                         observed_shape=witness["shape"], overlap=overlap_detail,
                         per_slot_counter_rows=witness["rows"],
                         park_attribution=witness["park_attribution"],
                         pressure_park_absence={"absent": not witness["parks"], "parks": witness["parks"]},
                         park_ordering_observable=False,
                         fixture_isolation="fixture_isolation_confirmed",
                         stderr_byte_range=[stderr_start, len(raw)], diagnostics=diagnostics)
            stage["passed"] = True
            self.save()
        finally:
            stage.setdefault("requests", [r.record() for r in requests])
            stage.setdefault("fixture_isolation", "fixture_non_isolation")
            stage.setdefault("park_ordering_observable", False)
            raw = self.stderr_path.read_bytes() if self.stderr_path.exists() else b""
            stage.setdefault("stderr_byte_range", [stderr_start, len(raw)])
            if not stage.get("diagnostics"):
                stage["diagnostics"] = parse_diagnostics(raw[stderr_start:].decode("utf-8", errors="replace"))
            self.save()


def process_history_isolation(label: str, processes: list[dict]) -> dict:
    if label == "source":
        require(processes and processes[-1]["label"] == "reference" and processes[-1].get("finalized") and
                processes[-1].get("returncode") == 0 and processes[-1].get("cleanup", {}).get("reader_stopped") and
                processes[-1].get("cleanup", {}).get("writer_stopped"),
                "source generation must start only after its separate reference process exited cleanly")
        return {"prior_reference_pid": processes[-1].get("pid"),
                "history_isolation": "fresh subprocess after reference exit; no conversation cache is shared"}
    if label == "handoff":
        require(not processes, "coherent handoff source must start before any same-prompt solo reference")
        return {"history_isolation": "fresh subprocess; exact-prompt solo reference is generated after continuation"}
    require(label == "reference" and not processes, "reference subprocess must be the first process")
    return {"history_isolation": "isolated reference-only subprocess; source starts after clean exit"}


def setup_engine(args, evidence, output, cfg, label):
    command, cwd, env, kv, mtp_contract = settings(cfg, args.exe.resolve(), args.context, args.cache_mib, args.gpu,
                                                    args.mtp_window, args.vram_reserve_mib)
    exe_hash = hashlib.sha256(args.exe.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    # Files cited by PARK_ATTRIBUTION_SOURCE must be present, so the file hashes are never silently skipped.
    attribution_sources = [root / "src/program/generate.cpp",
                           root / "include/strata/core/batch_mtp_policy.hpp",
                           root / "include/strata/core/shared_kv_reservation.hpp"]
    require(all(path.exists() for path in attribution_sources),
            "park-attribution source files are missing; cannot record the citation file hashes")
    sources = [Path(__file__).resolve(), Path(__file__).resolve().parent / "test_unified_kv_batch_mtp_pressure.py",
               Path(incremental.__file__).resolve(), Path(common.__file__).resolve(),
               Path(streaming.__file__).resolve(), root / "include/strata/core/mtp.hpp",
               *attribution_sources, root / "src/core/verify.cpp"]
    evidence.update(source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.exists()},
                    command=command, cwd=cwd, target_binary_sha256=exe_hash,
                    selected_gpu=env.get("CUDA_VISIBLE_DEVICES") or env.get("HIP_VISIBLE_DEVICES"),
                    kv_format=kv, explicit_opt_in={"argv_batch_mtp": True, "STRATA_BATCH_MTP": "1"},
                    main_mtp_contract=mtp_contract,
                    model_paths={"mtp": command[command.index("--mtp") + 1]},
                    source_ring=ring_capacity_from_source(args.mtp_window, 2, args.context))
    require(evidence["source_ring"]["bounded_ring"] and
            args.context - PAGE > evidence["source_ring"]["allocated_page_rounded_ring_cells"],
            "logical context must exceed a helper-derived bounded private ring")
    stderr = Path(f"{output}.{label}.stderr.log")
    require(not stderr.exists(), f"stderr path already exists: {stderr}")
    engine = BatchMtpEngine(evidence, stderr, args.cleanup_timeout)
    suite = PressureSuite(engine, evidence, output, args.stage_timeout, args.context, args.temperature,
                          args.seed, stderr_path=stderr)
    processes = evidence.setdefault("processes", [])
    process = {"label": label, "command": command, "cwd": cwd, "stderr_path": str(stderr),
               "stdout_start_seq": len(evidence.get("stdout", [])),
               "command_start_index": len(evidence.get("commands", [])),
               "started_wall_time": None, "finished_wall_time": None, "finalized": False}
    process.update(process_history_isolation(label, processes))
    return engine, suite, stderr, command, cwd, env, process


def close_owned_engine(engine, success):
    if engine is None or getattr(engine, "_pressure_closed", False):
        return
    engine.close(success)
    engine._pressure_closed = True


def start_owned_engine(engine, suite, command, cwd, env, args, evidence, process=None):
    """Start and validate only an already outer-owned engine; failures close it before propagating."""
    try:
        mtp_contract = validate_main_mtp_contract(command, 2)
        require("--batch-mtp" in command and env.get("STRATA_BATCH_MTP") == "1",
                "batch-MTP must be explicitly enabled in argv and environment")
        engine.start(command, cwd, env, args.context, 0, args.cache_mib, args.startup_timeout, mtp_max=2)
        if process is not None:
            process["pid"] = engine.p.pid
            if process.get("prior_reference_pid") is not None:
                require(process["pid"] != process["prior_reference_pid"],
                        "reference and source must be distinct native processes")
        info = evidence.get("info") or {}
        require(info.get("batch_groups") == "1", "INFO must confirm batch_groups=1")
        require(info.get("mtp_max") == "2" and info.get("spec") == "2",
                "INFO must confirm the two-token MTP window required for one main-path draft")
        if process is not None:
            process["info"] = info
            process["main_mtp_contract"] = mtp_contract
        text = Path(engine.stderr_path).read_text(encoding="utf-8", errors="replace")
        require(RING_FALLBACK not in text,
                "MTP draft K/V fell back to full resident mode; actual configured private ring was not allocated")
        evidence["startup"] = {"info": info, "stderr_sha256": hashlib.sha256(text.encode()).hexdigest(),
                               "ring_private_host_fallback_absent": True,
                               "main_mtp_contract": mtp_contract,
                               "raw_mtp_ring_capacity_source": evidence["source_ring"]}
        suite.save()
    except BaseException:
        try:
            close_owned_engine(engine, False)
        except BaseException as cleanup_exc:
            evidence["startup_cleanup_failure"] = {"type": type(cleanup_exc).__name__,
                                                    "message": str(cleanup_exc)}
        raise


def finish_process(engine, evidence, process, success):
    if process is None or process.get("finalized"):
        return
    failure = None
    try:
        close_owned_engine(engine, success)
        if success:
            incremental.cleanup_check(evidence)
    except BaseException as exc:
        failure = exc
        if not getattr(engine, "_pressure_closed", False):
            try:
                close_owned_engine(engine, False)
            except BaseException as retry_exc:
                evidence["cleanup_retry_failure"] = {"type": type(retry_exc).__name__,
                                                      "message": str(retry_exc)}
    finally:
        cleanup = evidence.pop("cleanup", {})
        stderr_path = Path(process["stderr_path"])
        process["pid"] = engine.p.pid if engine and engine.p is not None else None
        process["returncode"] = engine.p.returncode if engine and engine.p is not None else None
        process["cleanup"] = cleanup
        process["finished_wall_time"] = time.time()
        process["stdout_seq_range"] = [process.get("stdout_start_seq", 0), len(evidence.get("stdout", []))]
        process["command_index_range"] = [process.get("command_start_index", 0), len(evidence.get("commands", []))]
        process["argv_count"] = len(process["command"])
        if stderr_path.exists():
            raw = stderr_path.read_bytes()
            process["stderr_sha256"] = hashlib.sha256(raw).hexdigest()
            process["stderr_bytes"] = len(raw)
        evidence.setdefault("processes", []).append(process)
        process["finalized"] = True
    if failure is not None:
        raise failure


def make_workload(args, cfg):
    tok_path = common.resolve_path(cfg["tokenizer"], str(Path(cfg.get("cwd") or os.getcwd()).expanduser().resolve()))
    tok = common.tokenizer(tok_path)
    padded, a, b = incremental.fixtures(tok, args.prompt_a_cells, args.prompt_b_cells)
    return tok, padded, (a, b)


def solo_references(suite, evidence, prompts, cap):
    refs = []
    for slot, prompt in enumerate(prompts):
        ref = suite.solo(f"same-binary-target-MTP-reference-{slot}", prompt, cap)
        require(len(ref.tokens) == cap and ref.completion["finish"] == "length",
                f"solo target reference {slot} ended before requested cap (EOS is not pressure evidence)")
        fields = ref.completion["raw"].split()
        require(len(fields) >= 8 and int(fields[7]) > 0,
                f"solo reference {slot} did not exercise actual main-path MTP draft offers")
        refs.append(ref)
    evidence["same_binary_references"] = [r.record() for r in refs]
    evidence["same_binary_reference_draft_offers"] = [int(r.completion["raw"].split()[7]) for r in refs]
    return refs


def run_optional(args, evidence, suite, stderr, prompts, refs, plans):
    require([len(prompt) for prompt in prompts] == [plan["prompt_cells"] for plan in plans],
            "fixture builder did not produce the requested optional witness prompt lengths")
    sizing = optional_fixture_sizing(prompts, args.optional_cap, args.context)
    evidence["optional_fixture_sizing"] = sizing
    require(len(refs) == len(prompts) and all(ref.prompt == prompt and len(ref.tokens) == args.optional_cap and
                                              ref.completion["finish"] == "length"
                                              for ref, prompt in zip(refs, prompts)),
            "optional fixture requires complete same-binary solo references for the exact boundary prompts")
    suite.run_optional_row_fallback(prompts, args.optional_cap, refs, plans, args.stage_timeout)
    stage = evidence["stages"][-1]
    evidence["coverage"]["optional_reservation_fallback_without_pressure"] = (
        "RUN: exactly one slot reports fallback_reserve with no other fallback reason; observed shape "
        f"{stage['observed_shape']} ({'no park at all' if stage['observed_shape'] == 'no_park_both_cancel' else 'its mandatory row parks, attributed to that slot'})")
    evidence["coverage"]["optional_fallback_reserve_source_attribution"] = (
        "RUN (source-derived): a pressure terminal in this stage is reachable only from the mandatory reservation "
        "retry (generate.cpp 8739-8748) and the optional probe cannot reclaim or park "
        "(batch_mtp_policy.hpp 35-40). Park ordering is not observable.")
    evidence["coverage"]["bounded_private_ring_wrap_with_actual_proposals"] = "RUN (prompts exceed helper-derived ring capacity)"
    evidence["coverage"]["positive_full_coherent_BHANDOFF"] = "UNTESTED; run --gate handoff"
    evidence["coverage"]["older_slot_checkpoint_private_draft_restore"] = "UNTESTED; not borrowed"


def run_handoff(args, evidence, suite, stderr, prompts):
    prompt = prompts[0]
    # Exercise the fresh main->private-ring copy first; create the exact-prompt
    # same-binary solo reference after the continuation to avoid a cached solo
    # history supplying the handoff source's state.
    req = suite.make_attempt("coherent-active-slot-handoff-source", prompt, args.handoff_cap, 0)
    stage = {"name": "active-coherent-BHANDOFF-and-MAIN-continuation", "passed": False}
    evidence["stages"].append(stage)
    start = stderr.stat().st_size
    protocol = common.Protocol([req])
    deadline = time.monotonic() + args.stage_timeout
    handoff_sent = False
    try:
        suite.engine.stage = stage["name"]
        suite.engine.send(req.command(), deadline=deadline)
        while not protocol.finished:
            event = suite.engine.next_event(deadline)
            protocol.consume(event)
            if event["kind"] == "BT" and not handoff_sent:
                suite.engine.send("BHANDOFF 0", deadline=deadline)
                stage["handoff_command_after_seq"] = event["seq"]
                handoff_sent = True
        require(handoff_sent and req.completion["kind"] == "BDONE" and req.completion["finish"] == "handoff",
                "active slot did not acknowledge a positive BHANDOFF")
        require(len(req.tokens) < args.handoff_cap,
                "handoff source must preserve a nonterminal target prefix")
        raw = stderr.read_bytes()
        source_text = raw[start:].decode("utf-8", errors="replace")
        source_diags = parse_diagnostics(source_text)
        verify_stats(source_diags["batch_mtp_stats"], offer=True)
        continuation_prompt = prompt + req.tokens
        expected_reused = len(continuation_prompt) - 1
        continuation = suite.make_attempt("coherent-full-slot-to-main-continuation", continuation_prompt,
                                          args.handoff_cap - len(req.tokens))
        resume_start = stderr.stat().st_size
        main_stage = suite.run("coherent-handoff-main-GEN", [continuation])
        reference = suite.solo("coherent-handoff-same-binary-reference", prompt, args.handoff_cap)
        require(len(reference.tokens) == args.handoff_cap and req.tokens + continuation.tokens == reference.tokens,
                "coherent BHANDOFF continuation differs from same-binary solo IDs")
        require(continuation.admission_done["reused"] == expected_reused and expected_reused > 0,
                "coherent full-slot clone did not reuse the precise unfed-token prefix")
        require(continuation.completion["kind"] == "DONE" and continuation.completion["finish"] == "length",
                "coherent MAIN continuation did not finish normally")
        fields = continuation.completion["raw"].split()
        require(len(fields) >= 8 and int(fields[7]) > 0,
                "coherent full-slot private ring transfer failed to resume MTP proposals")
        main_text = stderr.read_bytes()[resume_start:].decode("utf-8", errors="replace")
        require("TARGET_ONLY slot clone" not in main_text,
                "coherent full-slot transfer was incorrectly downgraded to target-only")
        stage.update(source=req.record(), source_mtp=verify_stats(source_diags["batch_mtp_stats"], offer=True),
                     handoff_command_after_seq=stage["handoff_command_after_seq"],
                     main_continuation=continuation.record(), reused_cells=continuation.admission_done["reused"],
                     expected_reused_cells=expected_reused, continuation_main_draft_offers=int(fields[7]),
                     exact_solo_ids=True, stderr_byte_range=[start, len(raw)], passed=True)
        main_stage["passed"] = True
        evidence["coverage"]["positive_full_coherent_BHANDOFF"] = "RUN"
        evidence["coverage"]["bounded_private_ring_wrap_with_actual_proposals"] = "RUN if prompt exceeds helper-derived private ring"
        evidence["coverage"]["older_slot_checkpoint_private_draft_restore"] = "UNTESTED; older checkpoint remains target-only"
    finally:
        raw = stderr.read_bytes() if stderr.exists() else b""
        stage.setdefault("stderr_byte_range", [start, len(raw)])
        stage.setdefault("source", req.record())
        suite.save()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path, help="new evidence JSON; stderr saved beside it")
    ap.add_argument("--gate", choices=("pressure", "optional", "handoff"), default="pressure")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--cache-mib", type=int, default=4096)
    ap.add_argument("--prompt-a-cells", type=int, default=1536)
    ap.add_argument("--prompt-b-cells", type=int, default=1540)
    ap.add_argument("--pressure-cap", type=int, default=768)
    ap.add_argument("--optional-a-cells", type=int, default=1787,
                    help="optional witness slot 0 prompt cells (must end on a page boundary)")
    ap.add_argument("--optional-b-cells", type=int, default=1791,
                    help="optional witness slot 1 prompt cells (must end on a page boundary)")
    ap.add_argument("--optional-cap", type=int, default=320)
    ap.add_argument("--handoff-cap", type=int, default=64)
    ap.add_argument("--mtp-window", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--gpu")
    ap.add_argument("--vram-reserve-mib", type=int, default=2048)
    ap.add_argument("--startup-timeout", type=float, default=900)
    ap.add_argument("--stage-timeout", type=float, default=1800)
    ap.add_argument("--cleanup-timeout", type=float, default=15)
    args = ap.parse_args(argv)
    if args.context != 4096 or not 1 <= args.cache_mib <= 16384:
        ap.error("current bounded fixture requires --context 4096 and --cache-mib in 1..16384")
    if args.prompt_a_cells < 512 or args.prompt_b_cells < 512 or not 32 <= args.pressure_cap <= 2048:
        ap.error("prompt and pressure output limits are outside the bounded fixture range")
    if not 0 < args.seed < 2**63 or not math.isfinite(args.temperature) or args.temperature != 0:
        ap.error("these fixed-placement gates require greedy temperature=0 and a positive seed")
    if not 0 <= args.vram_reserve_mib <= 32768:
        ap.error("--vram-reserve-mib must be in 0..32768")
    if args.gpu is not None and (not args.gpu.strip() or "," in args.gpu):
        ap.error("--gpu must select exactly one device")
    try:
        validate_deadlines(args.startup_timeout, args.stage_timeout, args.cleanup_timeout)
    except AssertionError as exc:
        ap.error(str(exc))
    if args.gate == "optional":
        if args.optional_a_cells == args.optional_b_cells:
            ap.error("optional fixture needs two distinct prompts (--optional-a-cells must differ from "
                     "--optional-b-cells)")
        try:
            optional_fixture_sizing(([0] * args.optional_a_cells, [1] * args.optional_b_cells),
                                    args.optional_cap, args.context)
        except AssertionError as exc:
            ap.error(str(exc))
    output = args.output.expanduser().resolve()
    stderr_paths = [Path(f"{output}.{label}.stderr.log") for label in ("reference", "source", "handoff")]
    if output.exists() or any(path.exists() for path in stderr_paths):
        ap.error("evidence or process stderr path already exists; choose a fresh output name")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        f.write("{}\n")
    evidence = {"schema": 1, "harness": "unified-kv-batch-mtp-pressure", "gate": args.gate,
                "passed": False, "config": str(args.config.expanduser().resolve()),
                "exe": str(args.exe.expanduser().resolve()), "context": args.context,
                "aggregate_shared_backing_cells": args.context, "resident_cells": 0,
                "parking_budget_mib": args.cache_mib, "mtp_window_cells": args.mtp_window,
                "temperature": args.temperature, "seed": args.seed, "rng_offset_added": False,
                "commands": [], "stdout": [], "stages": [], "processes": [], "timeouts": {
                    "startup_s": args.startup_timeout, "stage_s": args.stage_timeout,
                    "cleanup_s": args.cleanup_timeout},
                "coverage": {"dual_slot_BT_and_batch_MTP_before_exhaustion": "UNTESTED",
                             "mandatory_pressure_BDONE_and_positive_target_only_park": "UNTESTED",
                             "stop_sibling_after_first_park": "UNTESTED",
                             "canonical_positive_restore_MAIN_exact_continuation": "UNTESTED",
                             "last_returned_token_unfed": "UNTESTED",
                             "target_only_restore_suppression_no_stale_offers": "UNTESTED",
                             "bounded_private_ring_wrap_and_natural_accept_reject": "UNTESTED",
                             "optional_reservation_fallback_without_pressure": "UNTESTED",
                             "optional_fallback_reserve_source_attribution": "UNTESTED",
                             "positive_full_coherent_BHANDOFF": "UNTESTED",
                             "older_slot_checkpoint_private_draft_restore": "UNTESTED (must remain target-only; no private-ring borrowing)"}}
    engine = suite = stderr = process = None
    success = False
    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
        cfg["config_path"] = str(args.config.resolve())
        tok, padded, prompts = make_workload(args, cfg)
        label = "handoff" if args.gate == "handoff" else "reference"
        engine, suite, stderr, command, cwd, env, process = setup_engine(args, evidence, output, cfg, label)
        process["started_wall_time"] = time.time()
        start_owned_engine(engine, suite, command, cwd, env, args, evidence, process)
        if args.gate == "optional":
            plans = [optional_slot_boundary(args.optional_a_cells),
                     optional_slot_boundary(args.optional_b_cells)]
            for plan in plans:
                require(plan["prompt_cells"] > evidence["source_ring"]["allocated_page_rounded_ring_cells"],
                        "optional witness prompts must also exceed actual bounded private ring capacity")
            witness_prompts = (padded("OPTIONAL-PAGE-BOUNDARY-A", args.optional_a_cells),
                               padded("OPTIONAL-PAGE-BOUNDARY-B", args.optional_b_cells))
            require([len(prompt) for prompt in witness_prompts] == [plan["prompt_cells"] for plan in plans],
                    "fixture builder did not produce the requested optional witness prompt lengths")
            refs = [suite.solo(f"optional-witness-same-binary-reference-{slot}", prompt, args.optional_cap)
                    for slot, prompt in enumerate(witness_prompts)]
            for slot, ref in enumerate(refs):
                require(len(ref.tokens) == args.optional_cap and ref.completion["finish"] == "length",
                        f"optional witness reference {slot} ended early (EOS); resize the cap, never credit it")
            finish_process(engine, evidence, process, True)
            engine = suite = stderr = process = None
            engine, suite, stderr, command, cwd, env, process = setup_engine(args, evidence, output, cfg, "source")
            process["started_wall_time"] = time.time()
            start_owned_engine(engine, suite, command, cwd, env, args, evidence, process)
            run_optional(args, evidence, suite, stderr, witness_prompts, refs, plans)
        elif args.gate == "handoff":
            require(len(prompts[0]) > evidence["source_ring"]["allocated_page_rounded_ring_cells"],
                    "coherent handoff fixture must exercise a prompt longer than the bounded ring")
            require(len(prompts[0]) + args.handoff_cap + GUARD <= args.context,
                    "handoff prompt/output exceeds logical context")
            run_handoff(args, evidence, suite, stderr, prompts)
        else:
            require(all(len(p) > evidence["source_ring"]["allocated_page_rounded_ring_cells"] for p in prompts),
                    "pressure fixture must wrap both actual bounded private MTP rings during prompt prefill")
            refs = solo_references(suite, evidence, prompts, args.pressure_cap)
            finish_process(engine, evidence, process, True)
            engine = suite = stderr = process = None
            engine, suite, stderr, command, cwd, env, process = setup_engine(args, evidence, output, cfg, "source")
            process["started_wall_time"] = time.time()
            start_owned_engine(engine, suite, command, cwd, env, args, evidence, process)
            owner, source = suite.run_source_pair(prompts, args.pressure_cap, refs, args.stage_timeout)
            source["capacity_plan"] = pressure_capacity_plan(prompts, [args.pressure_cap] * 2, args.context)
            source["passed"] = True
            # Complete a separate MAIN GEN with precisely original + ALL output
            # tokens. The last returned one is not in the parked KV and is replayed.
            remaining = args.pressure_cap - len(owner.tokens)
            require(remaining > 0, "pressure owner consumed its entire output allowance")
            replay, expected_reuse = pressure_replay_ids(owner.prompt, owner.tokens)
            require(replay == prompts[owner.slot] + owner.tokens and len(replay) > 0,
                    "pressure continuation history differs from original + every returned token")
            resumed = suite.make_attempt(f"pressure-restore-main-slot-{owner.slot}", replay, remaining)
            stderr_start = stderr.stat().st_size
            restore_stage = suite.run("positive-pressure-park-restore-target-only-MAIN", [resumed])
            require(owner.tokens + resumed.tokens == refs[owner.slot].tokens,
                    "pressure/replay token IDs differ from same-binary solo target reference")
            require(resumed.completion["kind"] == "DONE" and resumed.completion["finish"] == "length" and
                    len(resumed.tokens) == remaining,
                    "restored MAIN continuation did not complete the remaining allowance")
            raw = stderr.read_bytes()
            text = raw[stderr_start:].decode("utf-8", errors="replace")
            diagnostics = parse_diagnostics(text)
            require(any(d["kind"] == "restore" and d["tokens"] == expected_reuse
                        for d in diagnostics["canonical_cache"]),
                    "missing positive canonical-restore diagnostic for exact target prefix")
            target_only = verify_target_only_resume(resumed.completion["raw"], diagnostics,
                                                    expected_reuse, resumed.admission_done["reused"])
            done_fields = resumed.completion["raw"].split()
            restore_stage.update(passed=True, request=resumed.record(), pressure_owner_slot=owner.slot,
                                 replay_prompt_ids=replay, expected_reused_cells=expected_reuse,
                                 actual_reused_cells=resumed.admission_done["reused"],
                                 last_returned_token_unfed=True, positive_canonical_restore=True,
                                 restored_main_drafts_offered=int(done_fields[7]),
                                 target_only_diagnostics=diagnostics,
                                 exact_full_continuation_ids=True, stderr_byte_range=[stderr_start, len(raw)])
            evidence["coverage"].update({"dual_slot_BT_and_batch_MTP_before_exhaustion": "RUN",
                "mandatory_pressure_BDONE_and_positive_target_only_park": "RUN",
                "stop_sibling_after_first_park": "RUN",
                "canonical_positive_restore_MAIN_exact_continuation": "RUN",
                "last_returned_token_unfed": "RUN",
                "target_only_restore_suppression_no_stale_offers": "RUN",
                "bounded_private_ring_wrap_and_natural_accept_reject": "RUN",
                "optional_reservation_fallback_without_pressure": "UNTESTED; separate --gate optional",
                "positive_full_coherent_BHANDOFF": "UNTESTED; separate --gate handoff"})
            ring_cells = evidence["source_ring"]["allocated_page_rounded_ring_cells"]
            consumed_history = len(owner.prompt) + len(owner.tokens) - 1
            require(len(owner.prompt) > ring_cells and consumed_history > ring_cells,
                    "pressure fixture did not prefill/wrap and continue beyond the actual bounded ring")
            evidence["ring_wrap"] = {"ring_geometry": evidence["source_ring"],
                                      "pressure_prompt_cells": [len(p) for p in prompts],
                                      "consumed_target_history_cells": consumed_history,
                                      "prompt_longer_than_ring": all(len(p) > ring_cells for p in prompts),
                                      "consumed_history_exceeds_ring": consumed_history > ring_cells,
                                      "natural_slot_outcomes": source["per_slot_natural_accept_reject"],
                                      "natural_accept_reject_scope": source["natural_accept_reject_scope"],
                                      "ring_fallback_to_full_resident": False,
                                      "evidence_limit": "native per-slot proposals/acceptance/rejection and solo parity are model evidence; byte-level ring internals require CUDA parity tests"}
        suite.save()
        require(evidence["stages"] and all(s.get("passed") for s in evidence["stages"]),
                "one or more native stages did not pass")
        success = True
    except (Exception, KeyboardInterrupt) as exc:
        evidence["failure"] = {"type": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc()}
    finally:
        if engine is not None and process is not None:
            try:
                finish_process(engine, evidence, process, success)
            except (Exception, KeyboardInterrupt) as exc:
                success = False
                evidence["cleanup_failure"] = {"type": type(exc).__name__, "message": str(exc),
                                               "traceback": traceback.format_exc()}
        evidence["stderr_logs"] = [{"path": p["stderr_path"], "sha256": p.get("stderr_sha256"),
                                    "bytes": p.get("stderr_bytes")} for p in evidence.get("processes", [])]
        evidence["finished_wall_time"] = time.time()
        evidence["passed"] = success
        output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"{'PASS' if success else 'FAIL'}: {output}; stderr: {stderr}", flush=True)
    if not success:
        print(json.dumps(evidence.get("failure") or evidence.get("cleanup_failure")), file=sys.stderr)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
