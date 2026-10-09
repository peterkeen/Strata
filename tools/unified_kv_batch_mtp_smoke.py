#!/usr/bin/env python3
"""Bounded opt-in unified-KV / one-proposal-per-slot MTP model gate.

Private subprocess only; never points at a running service. Example (hook-enabled
candidate, after the separate GPU validator has approved the isolated device):

  python3 tools/unified_kv_batch_mtp_smoke.py --exe /path/to/strata --config model.json \
      --output /tmp/mtp-core.json --mode correctness --context 1024 --gpu 0

The correctness mode runs same-binary solo target references, closes that process,
then starts a fresh batch-MTP process with the compile-time-gated deterministic
proposal hook. It requires two admitted, actually overlapping slots, post-admission
BT progress from both, exact token-ID parity, and native positive offer/accept/reject
counters. The hook only changes the proposal row; target output/acceptance/commit are
still native. Never use this harness or its hook for performance measurements.

`tails`, `lifecycle`, and `limits` are focused follow-up modes. Coverage not run by a
mode is explicitly retained as UNTESTED in evidence. No downloads, builds, SSH, or
server/production interaction are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_kv_smoke as common  # noqa: E402

PAGE = common.PAGE_CELLS
STATS_RE = re.compile(
    r"^strata batch_mtp_stats slot=(\d+) windows=(\d+) offered=(\d+) accepted=(\d+) rejected=(\d+) "
    r"discarded=(\d+) fallback_attempts=(\d+) fallback_incoherent=(\d+) fallback_not_ready=(\d+) "
    r"fallback_limits=(\d+) fallback_capacity=(\d+) fallback_reserve=(\d+)$")
ALLOC_RE = re.compile(r"unified KV: (\d+) total backing cells shared by admission and (\d+) slots, "
                       r"(\d+) GPU-resident cells per layer( \(host-backed\))?")
SLOT_COPY_RE = re.compile(r"strata batch: slot (\d+) gave back (\d+) tokens of this conversation "
                          r"\((all it holds|its turn checkpoint)\) in ([\d.]+) ms")

require = common.require
Request = common.Request
Protocol = common.Protocol


def parse_mtp_diagnostics(text: str) -> dict:
    """Parse exact native batch-MTP summaries; malformed/duplicate lines fail closed."""
    stats, slot_copies, allocations = [], [], []
    hook_lines, target_only = [], []
    for line_no, line in enumerate(text.splitlines(), 1):
        if "strata batch_mtp_stats" in line:
            match = STATS_RE.fullmatch(line.strip())
            require(match is not None, f"malformed batch_mtp_stats diagnostic at stderr line {line_no}: {line}")
            keys = ("slot", "windows", "offered", "accepted", "rejected", "discarded", "fallback_attempts",
                    "fallback_incoherent", "fallback_not_ready", "fallback_limits", "fallback_capacity",
                    "fallback_reserve")
            stats.append({key: int(value) for key, value in zip(keys, match.groups())} | {"raw": line})
        if "unified KV:" in line:
            match = ALLOC_RE.search(line)
            if match:
                allocations.append({"cells": int(match.group(1)), "slots": int(match.group(2)),
                                    "resident_cells": int(match.group(3)), "host_backed": bool(match.group(4)),
                                    "raw": line})
        if "strata batch: slot " in line:
            match = SLOT_COPY_RE.search(line)
            if match:
                slot_copies.append({"slot": int(match.group(1)), "tokens": int(match.group(2)),
                                    "source": match.group(3), "ms": float(match.group(4)), "raw": line})
        if "strata batch_mtp_test_hook enabled:" in line:
            hook_lines.append(line)
        if "TARGET_ONLY slot clone: private MTP proposals suppressed until full replay" in line:
            target_only.append(line)
    seen = [item["slot"] for item in stats]
    require(len(seen) == len(set(seen)), "duplicate batch_mtp_stats summary for a slot lifecycle")
    for item in stats:
        validate_stats(item)
    return {"batch_mtp_stats": stats, "allocations": allocations, "slot_copies": slot_copies,
            "hook_lines": hook_lines, "target_only_suppression": target_only}


def validate_stats(s: dict) -> dict:
    nonnegative = ("windows", "offered", "accepted", "rejected", "discarded", "fallback_attempts",
                  "fallback_incoherent", "fallback_not_ready", "fallback_limits", "fallback_capacity",
                  "fallback_reserve")
    require(all(isinstance(s.get(k), int) and s[k] >= 0 for k in nonnegative),
            "batch MTP counters must be nonnegative integers")
    require(s["offered"] == s["accepted"] + s["rejected"] + s["discarded"],
            "offered != accepted + rejected + discarded")
    fallbacks = sum(s[k] for k in ("fallback_incoherent", "fallback_not_ready", "fallback_limits",
                                   "fallback_capacity", "fallback_reserve"))
    require(s["fallback_attempts"] == fallbacks, "fallback reason counters do not sum to fallback_attempts")
    require(s["windows"] == s["offered"] + s["fallback_attempts"],
            "windows != offered + fallback_attempts")
    return s


FALLBACK_REASONS = {"incoherent": "fallback_incoherent", "not_ready": "fallback_not_ready",
                     "limits": "fallback_limits", "capacity": "fallback_capacity", "reserve": "fallback_reserve"}


def require_outcomes(stats, *, offers=False, accepted=False, rejected=False, allowed_fallbacks=()):
    """Require measured outcomes; permit only explicitly understood fallback reasons.

    A max_new request can legitimately end with one target-only row when only
    one output token remains. That terminal `limits` fallback is allowed at most
    once per slot; other fallback reasons must be named by the caller.
    """
    require(stats, "missing per-slot batch_mtp_stats; no evidence of slot MTP activity")
    unknown = set(allowed_fallbacks) - set(FALLBACK_REASONS)
    require(not unknown, f"unknown allowed fallback reason(s): {sorted(unknown)}")
    if offers:
        require(sum(s["offered"] for s in stats) > 0, "no actual proposal row was offered")
    if accepted:
        require(sum(s["accepted"] for s in stats) > 0, "no accepted proposal was committed")
    if rejected:
        require(sum(s["rejected"] for s in stats) > 0, "no rejected proposal was observed")
    unexpected = {name: sum(s[counter] for s in stats) for name, counter in FALLBACK_REASONS.items()
                  if name not in allowed_fallbacks and sum(s[counter] for s in stats)}
    require(not unexpected, f"unexpected target-only fallback reason(s): {unexpected}")
    if "limits" in allowed_fallbacks:
        require(all(s["fallback_limits"] <= 1 for s in stats),
                "more than one terminal max_new-limit fallback in a slot")
    fallback_counts = {name: sum(s[counter] for s in stats) for name, counter in FALLBACK_REASONS.items()}
    return {"slots": stats, "offered": sum(s["offered"] for s in stats),
            "accepted": sum(s["accepted"] for s in stats), "rejected": sum(s["rejected"] for s in stats),
            "discarded": sum(s["discarded"] for s in stats),
            "fallback_attempts": sum(s["fallback_attempts"] for s in stats),
            "fallbacks_by_reason": fallback_counts, "allowed_fallbacks": sorted(allowed_fallbacks)}


def verify_target_only_tail_restore(diagnostics: dict) -> dict:
    """Prove completed-slot partial-tail restore suppresses stale private MTP state."""
    require(diagnostics.get("target_only_suppression"),
            "completed-slot clone did not report private-MTP target-only suppression")
    stats = diagnostics.get("batch_mtp_stats", [])
    proof = require_outcomes(stats, allowed_fallbacks=("incoherent", "not_ready", "limits"))
    require(proof["offered"] == 0,
            "terminal slot clone unexpectedly offered private MTP proposals before coherent replay")
    require(proof["fallback_attempts"] > 0 and sum(s["windows"] for s in stats) > 0,
            "target-only clone path lacks native fallback/window evidence")
    return proof


def verify_batch_startup(info: dict, context: int, stderr: str) -> dict:
    common.verify_info(info, context)
    require(info.get("batch_groups") == "1", "INFO must confirm batch_groups=1")
    records = parse_mtp_diagnostics(stderr)
    require(len(records["allocations"]) == 1, "missing/duplicate unified-KV allocation diagnostic")
    alloc = records["allocations"][0]
    require(alloc["cells"] == context and alloc["slots"] == 2,
            "native allocation does not confirm one aggregate backing pool and two slots")
    require(alloc["resident_cells"] == context and not alloc["host_backed"],
            "this small correctness fixture requires fully resident shared KV")
    return {"info": info, "allocation": alloc}


def verify_dual_overlap(requests: list[Request]) -> dict:
    require(len(requests) == 2, "dual overlap proof requires exactly two requests")
    a, b = requests
    require(a.badm and a.badm["continues"] and b.badm and b.badm["continues"],
            "both BGEN admissions must continue into active slots")
    second_badm = max(a.badm["seq"], b.badm["seq"])
    require(a.completion and b.completion and a.completion["kind"] == b.completion["kind"] == "BDONE",
            "both active slots must finish through BDONE")
    first_completion = min(a.completion["seq"], b.completion["seq"])
    require(first_completion > second_badm, "a slot completed before both admissions")
    overlapping_bt = {}
    for request in requests:
        progress = [e for e in request.token_events
                    if e["kind"] == "BT" and second_badm < e["seq"] < first_completion]
        require(progress, f"slot {request.slot} made no BT progress before the first slot completed")
        common.parity(request, request.reference)
        overlapping_bt[str(request.slot)] = [e["seq"] for e in progress]
    return {"both_badm_seq": [a.badm["seq"], b.badm["seq"]], "second_badm_seq": second_badm,
            "first_completion_seq": first_completion,
            "completion_seq": [a.completion["seq"], b.completion["seq"]],
            "bt_between_both_badm_and_first_completion": overlapping_bt,
            "actual_dual_slot_progress": True, "exact_target_token_ids": True}


def tokenizer(path):
    return common.tokenizer(path)


def engine_settings(cfg: dict, exe: Path | str, context: int, *, gpu=None, hook_schedule: str | None = None,
                    cache_mib: int = 0) -> tuple[list[str], str, dict, dict]:
    """Retain model/MTP inputs and config env; replace only controlled test knobs."""
    require(context >= 512 and context <= 4096 and context % PAGE == 0,
            "context must be 512..4096 and four-cell aligned")
    require(cache_mib >= 0, "cache-mib must be nonnegative")
    cwd = str(Path(cfg.get("cwd") or os.getcwd()).expanduser().resolve())
    valued = {"--batch", "--batch-groups", "--max-context", "--max-new", "--kv-resident",
              "--conversation-cache-mib", "--conversation-cache-slots", "--conversation-cache-min-free-mib",
              "--pcie-frac", "--adapt-every", "--adapt-swaps", "--peer-adapt-swaps", "--spec",
              "--mtp-max-t", "--spec-min-p", "--suffix-draft", "--lookup-chain", "--prompt-cache",
              "--prefill", "--short-read", "--layer-split", "--split-device", "--expert-profile-save",
              "--expert-profile-save-every"}
    flags = {"--serve", "--kv-unified", "--batch-mtp", "--no-prefill-borrow", "--trim-stage-weights",
             "--batch-groups-auto"}
    original = list(cfg.get("args") or [])
    args, i = [], 0
    while i < len(original):
        arg = original[i]
        if arg in valued:
            require(i + 1 < len(original), f"config option missing a value: {arg}")
            i += 2
        elif arg in flags:
            i += 1
        else:
            args.append(arg)
            i += 1
    mtp_values = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "--mtp"]
    require(len(mtp_values) == 1 and mtp_values[0], "config must retain exactly one actual --mtp path")
    require("--layer-split" not in args and "--split-device" not in args,
            "test command must not retain a multi-GPU layer split")
    values = ["--serve", "--batch", "2", "--batch-groups", "1", "--kv-unified", "--batch-mtp",
              "--max-context", str(context), "--max-new", "16", "--kv-resident", "0",
              "--conversation-cache-mib", str(cache_mib), "--conversation-cache-slots", "4",
              "--pcie-frac", "0", "--adapt-every", "1000000", "--adapt-swaps", "0",
              "--peer-adapt-swaps", "0", "--spec", "2", "--mtp-max-t", "1", "--spec-min-p", "0",
              "--suffix-draft", "0", "--lookup-chain", "0", "--prompt-cache", "6",
              "--prefill", "512", "--short-read", "64", "--no-prefill-borrow"]
    args.extend(values)
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (cfg.get("env") or {}).items()})
    env["STRATA_IQ_MT_MIN"] = "1"
    env["STRATA_BATCH_MTP"] = "1"  # explicit opt-in is also reflected in the environment
    env["STRATA_BATCH_DECODE_SHARE"] = "0"  # do not advance A while B's one-chunk admission is read
    env["STRATA_KV_GROW"] = "0"
    env.pop("STRATA_BATCH_MTP_TEST_PROPOSALS", None)
    if hook_schedule is not None:
        env["STRATA_BATCH_MTP_TEST_PROPOSALS"] = hook_schedule
    devices = cfg.get("gpu") or []
    if not isinstance(devices, list):
        devices = [devices]
    if gpu is None:
        gpu = cfg.get("hip_ordinal") if cfg.get("backend") == "hip" else None
        if gpu is None and devices:
            gpu = devices[0]
    visibility = "HIP_VISIBLE_DEVICES" if cfg.get("backend") == "hip" else "CUDA_VISIBLE_DEVICES"
    if gpu is not None:
        require("," not in str(gpu), "select exactly one physical GPU")
        env[visibility] = str(gpu)
    elif env.get(visibility):
        env[visibility] = env[visibility].split(",")[0]
    if visibility == "CUDA_VISIBLE_DEVICES":
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    libs = [common.resolve_path(d, cwd) for d in cfg.get("lib_dirs") or []]
    if libs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(libs + ([env[var]] if env.get(var) else []))
    command = [str(Path(exe).expanduser().resolve()), *args]
    return command, cwd, env, {"mtp_path": common.resolve_path(mtp_values[0], cwd),
                               "tokenizer_path": common.resolve_path(cfg["tokenizer"], cwd),
                               "selected_gpu": env.get(visibility), "backend": cfg.get("backend", "cuda"),
                               "lib_dirs": libs, "cache_mib": cache_mib}


def verify_assets(cfg: dict, exe: Path, metadata: dict) -> None:
    require(exe.is_file(), f"native executable does not exist: {exe}")
    require(Path(metadata["tokenizer_path"]).exists(), f"tokenizer path does not exist: {metadata['tokenizer_path']}")
    require(Path(metadata["mtp_path"]).is_dir(), f"MTP path does not exist: {metadata['mtp_path']}")
    require(Path(cfg["config_path"]).is_file(), "config file does not exist")


def fixtures(tok, context: int, cap: int = 16) -> tuple[list[int], list[int]]:
    def encode(question):
        return tok.encode(f"<|im_start|>user\n{question}<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
    a = encode("List numbered facts from 1 onward about astronomy. Give one concise sentence per item; continue for many items.")
    b_question = "List numbered facts from 1 onward about mathematics. Give one concise sentence per item; continue for many items."
    b = encode(b_question)
    # Ensure actual token-count inequality rather than relying on the tokenizer
    # assigning different BPE lengths to two same-shape questions.
    for extra_words in range(1, 65):
        if len(b) > len(a):
            break
        b = encode(b_question + " Add detail" * extra_words)
    require(a and b and a != b and len(a) != len(b),
            "fixture prompts must be nonempty, distinct, and have explicitly unequal token counts")
    require(all(len(p) + cap + 8 <= context for p in (a, b)),
            "each prompt + max_new + native eight-token safety margin must fit context")
    reserved = sum((len(p) + cap + PAGE - 1) // PAGE for p in (a, b))
    require(reserved <= context // PAGE, "two prompt/output page reservations must fit shared backing")
    return a, b


def tail_fixtures(tok, context: int, cap: int = 8):
    outputs = []
    for offset in (1, 2, 3):
        seed_fixture = common.build_partial_tail_seed(tok, offset, context, output_cap=cap, guard_cells=8)
        seed = seed_fixture.token_ids
        require(len(seed) == seed_fixture.target_cells and
                (len(seed) + cap - 1) % PAGE == offset,
                "batch-MTP tail seed helper returned inconsistent exact token/page sizing")
        outputs.append((offset, seed))
    return outputs


class BatchSuite(common.Suite):
    def __init__(self, engine, evidence, output, timeout, context, stderr_path, root_evidence):
        super().__init__(engine, evidence, output, timeout, context)
        self.stderr_path = stderr_path
        self.root_evidence = root_evidence

    def save(self):
        self.output.write_text(json.dumps(self.root_evidence, indent=2) + "\n", encoding="utf-8")

    def run(self, name, requests, cancel=False):
        before = self.stderr_path.stat().st_size if self.stderr_path.exists() else 0
        stage = None
        try:
            stage = super().run(name, requests, cancel)
            return stage
        finally:
            raw = self.stderr_path.read_text(encoding="utf-8", errors="replace") if self.stderr_path.exists() else ""
            encoded = raw.encode("utf-8", errors="replace")
            delta = encoded[before:].decode("utf-8", errors="replace")
            if stage is None and self.evidence["stages"]:
                stage = self.evidence["stages"][-1]
            if stage is not None:
                stage["stderr_byte_range"] = [before, len(encoded)]
                stage["stderr_diagnostics"] = parse_mtp_diagnostics(delta)
                self.save()


def open_engine(evidence: dict, output: Path, cfg: dict, exe: Path, context: int, gpu,
                hook_schedule: str | None, startup_timeout: float, cleanup_timeout: float):
    command, cwd, env, metadata = engine_settings(cfg, exe, context, gpu=gpu, hook_schedule=hook_schedule)
    verify_assets(cfg, exe, metadata)
    evidence.setdefault("processes", []).append({"command": command, "cwd": cwd, "paths": metadata,
                                                   "env_overrides": {k: env[k] for k in ("STRATA_IQ_MT_MIN",
                                                       "STRATA_BATCH_MTP", "STRATA_BATCH_DECODE_SHARE",
                                                       "STRATA_KV_GROW", "STRATA_BATCH_MTP_TEST_PROPOSALS",
                                                       "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES") if k in env},
                                                   "config_env": dict(cfg.get("env") or {}),
                                                   "started_wall_time": time.time(), "stdout": [], "commands": [],
                                                   "stages": []})
    process_data = evidence["processes"][-1]
    stderr_path = Path(str(output) + f".process{len(evidence['processes'])}.stderr.log")
    engine = common.NativeEngine(process_data, stderr_path, cleanup_timeout)
    suite = BatchSuite(engine, process_data, output, evidence["limits"]["stage_deadline_s"], context,
                       stderr_path, evidence)
    try:
        engine.start(command, cwd, env, context, startup_timeout)
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        startup = verify_batch_startup(process_data["info"], context, stderr)
        process_data["startup"] = startup
        process_data["startup_diagnostics"] = parse_mtp_diagnostics(stderr)
        process_data["startup_stderr_bytes"] = len(stderr.encode("utf-8"))
        return engine, suite, process_data, stderr_path
    except BaseException:
        # Startup timeout/INFO/allocation failures must not leave a private
        # model process alive or its stderr handle open.
        try:
            engine.close(False)
            process_data["finished_wall_time"] = time.time()
        except BaseException:
            pass
        raise


def close_engine(engine, process_data, success):
    engine.close(success)
    process_data["finished_wall_time"] = time.time()
    result = process_data.setdefault("cleanup", {})
    if success:
        require(result.get("returncode") == 0, "native process did not exit cleanly after QUIT")
        require(not result.get("quit_timeout"), "native QUIT exceeded cleanup deadline")
        require(result.get("reader_stopped") and result.get("writer_stopped"),
                "native stdout reader or stdin writer remained alive")
        require(not result.get("protocol_errors"), "malformed protocol output during shutdown")


def run_references(evidence, output, cfg, exe, context, gpu, prompts, cap, startup_timeout, stage_timeout,
                   cleanup_timeout):
    engine, suite, proc, _ = open_engine(evidence, output, cfg, exe, context, gpu, None,
                                         startup_timeout, cleanup_timeout)
    references = []
    success = False
    try:
        for i, prompt in enumerate(prompts):
            req = suite.solo(f"target-reference-{i}", prompt, cap)
            require(len(req.tokens) >= 2, f"reference {i} ended before the first speculative target position")
            references.append(req)
        success = True
    finally:
        close_engine(engine, proc, success)
    return references


def run_core(evidence, output, cfg, exe, context, gpu, startup_timeout, stage_timeout, cleanup_timeout):
    prompts = fixtures(tokenizer(common.resolve_path(cfg["tokenizer"],
                          str(Path(cfg.get("cwd") or os.getcwd()).expanduser().resolve()))), context)
    refs = run_references(evidence, output, cfg, exe, context, gpu, prompts, 16,
                          startup_timeout, stage_timeout, cleanup_timeout)
    require(refs[0].tokens[1] != refs[1].tokens[1] or len(set(refs[0].tokens + refs[1].tokens)) > 1,
            "need an observed valid vocabulary ID distinct from the mismatch target")
    mismatch = next((t for t in refs[0].tokens + refs[1].tokens + prompts[0] + prompts[1]
                     if t != refs[1].tokens[1]), None)
    require(mismatch is not None, "could not find a valid token ID to force a distinct proposal")
    schedule = f"0/1/{refs[0].tokens[1]},1/1/{mismatch}"
    evidence["deterministic_proposal_schedule"] = {
        "text": schedule, "slot0": {"offer": 1, "proposal_token": refs[0].tokens[1],
                                      "expected_target_token_at_offer": refs[0].tokens[1], "outcome": "accepted"},
        "slot1": {"offer": 1, "proposal_token": mismatch,
                  "expected_target_token_at_offer": refs[1].tokens[1], "outcome": "rejected"},
        "derivation": "same-binary solo GEN token IDs; schedule applies to first offered ordinal only",
        "hook_changes_proposal_row_only": True}
    engine, suite, proc, stderr_path = open_engine(evidence, output, cfg, exe, context, gpu, schedule,
                                                   startup_timeout, cleanup_timeout)
    reference_proc = evidence["processes"][-2]
    require(reference_proc["command"] == proc["command"],
            "solo references and batch run must use byte-identical native command/target policy")
    expected_env = dict(reference_proc["env_overrides"])
    candidate_env = dict(proc["env_overrides"])
    expected_env.pop("STRATA_BATCH_MTP_TEST_PROPOSALS", None)
    candidate_env.pop("STRATA_BATCH_MTP_TEST_PROPOSALS", None)
    require(expected_env == candidate_env,
            "solo references and batch run differ in target arithmetic/device environment beyond test hook")
    evidence["reference_batch_policy"] = {"native_command_identical": True,
                                          "target_environment_identical_except_hook": True,
                                          "reference_process": len(evidence["processes"]) - 1,
                                          "batch_process": len(evidence["processes"])}
    reqs = [Request("forced-accept-slot0", prompts[0], 16, 0),
            Request("forced-reject-slot1", prompts[1], 16, 1)]
    for request, reference in zip(reqs, refs):
        request.reference = reference
    success = False
    try:
        stage = suite.run("dual-overlap-forced-accept-reject-and-natural-catchup", reqs)
        overlap = verify_dual_overlap(reqs)
        # STRATA_BATCH_DECODE_SHARE=0 and both commands are burst together. The
        # offer-1 solo reference is aligned only if each request has exactly its
        # admission T, and no BT, before both BADM records. Check captured events
        # rather than the final token list (which is complete by this point).
        second_badm = overlap["second_badm_seq"]
        for request in reqs:
            before = [e for e in request.token_events if e["seq"] < second_badm]
            require(len(before) == 1 and before[0]["kind"] == "T" and before[0]["seq"] < request.badm["seq"],
                    f"slot {request.slot} did not have exactly one admission T and no BT before dual start")
        stats = stage["stderr_diagnostics"]["batch_mtp_stats"]
        require(len(stats) == 2 and {s["slot"] for s in stats} == {0, 1},
                "expected exactly one native slot summary for both active requests")
        proof = require_outcomes(stats, offers=True, accepted=True, rejected=True, allowed_fallbacks=("limits",))
        by_slot = {s["slot"]: s for s in stats}
        require(by_slot[0]["accepted"] > 0, "forced matching proposal was not accepted by target verification")
        require(by_slot[1]["rejected"] > 0, "forced distinct proposal was not rejected by target verification")
        require(by_slot[0]["offered"] >= 2 and by_slot[1]["offered"] >= 2,
                "need subsequent natural MTP offers after each forced first proposal")
        require(len(proc["startup_diagnostics"]["hook_lines"]) == 1,
                "missing explicit test-only proposal-hook activation diagnostic")
        stage["overlap_proof"] = overlap
        stage["proposal_proof"] = proof
        stage["natural_catchup_offers_after_forced_first"] = {"slot0": by_slot[0]["offered"] - 1,
                                                               "slot1": by_slot[1]["offered"] - 1}
        stage["terminal_limit_fallback_policy"] = {
            "allowed_reason": "limits", "maximum_per_slot": 1,
            "why": "the remaining one output token cannot form a two-row speculative window; any other fallback fails"}
        stage["passed"] = True
        success = True
    finally:
        close_engine(engine, proc, success)
    # Catch unexpected ERRs even if they arrived after the final protocol record.
    require(not any(e.get("kind") == "ERR" for e in proc["stdout"]), "unexpected native ERR in correctness run")


def run_tails(evidence, output, cfg, exe, context, gpu, startup_timeout, stage_timeout, cleanup_timeout):
    tok = tokenizer(common.resolve_path(cfg["tokenizer"], str(Path(cfg.get("cwd") or os.getcwd()).expanduser().resolve())))
    prepared = []
    ref_engine, ref_suite, ref_proc, _ = open_engine(evidence, output, cfg, exe, context, gpu, None,
                                                       startup_timeout, cleanup_timeout)
    ref_success = False
    try:
        for offset, seed in tail_fixtures(tok, context):
            seed_ref = ref_suite.solo(f"tail-{offset}-seed-reference", seed, 8)
            require(len(seed_ref.tokens) == 8, "tail seed must complete its bounded cap")
            shared = seed + seed_ref.tokens[:-1]
            require(len(shared) % PAGE == offset, "shared cached prefix does not end at requested partial-page offset")
            suffix = tok.encode("\nContinue with further numbered observations.\n")
            branch = shared + suffix
            require(len(branch) + 8 + 8 <= context, "partial-tail branch exceeds logical context plus safety margin")
            branch_ref = ref_suite.solo(f"tail-{offset}-branch-reference", branch, 8)
            prepared.append((offset, seed, seed_ref, shared, branch, branch_ref))
        ref_success = True
    finally:
        close_engine(ref_engine, ref_proc, ref_success)
    engine, suite, proc, _ = open_engine(evidence, output, cfg, exe, context, gpu, None,
                                          startup_timeout, cleanup_timeout)
    success = False
    try:
        for offset, seed, seed_ref, shared, branch, branch_ref in prepared:
            seed_req = Request(f"tail-{offset}-seed", seed, 8, 0)
            seed_req.reference = seed_ref
            seed_stage = suite.run(f"populate-tail-source-{offset}", [seed_req])
            common.parity(seed_req, seed_ref)
            require(seed_req.badm and seed_req.badm["continues"], "tail source did not populate a live slot")
            seed_stage["passed"] = True
            branch_req = Request(f"tail-{offset}-divergent-branch", branch, 8, 1)
            branch_req.reference = branch_ref
            branch_stage = suite.run(f"partial-tail-target-only-restore-{offset}", [branch_req])
            common.parity(branch_req, branch_ref)
            require(branch_req.admission_done["reused"] is not None and
                    branch_req.admission_done["reused"] >= len(shared),
                    "native admission did not reuse the exact shared prefix")
            copies = [d for d in branch_stage["stderr_diagnostics"]["slot_copies"]
                      if d["slot"] == 0 and d["tokens"] >= len(shared)]
            require(copies, "native slot-to-main prefix copy did not identify the source slot/prefix")
            require(branch[len(shared):], "branch must write a divergent suffix after the shared cached prefix")
            diagnostics = branch_stage["stderr_diagnostics"]
            proof = verify_target_only_tail_restore(diagnostics)
            branch_stage["target_only_restore_witness"] = {
                "offset_cells": offset, "shared_prefix_cells": len(shared),
                "reused_cells": branch_req.admission_done["reused"], "source_slot_copy": copies[-1],
                "divergent_suffix_ids": branch[len(shared):], "proposal_count": proof["offered"],
                "fallback_counts": proof["fallbacks_by_reason"],
                "proof_kind": "terminal source slot has invalidated MTP provenance; exact target-only suppression verified"}
            branch_stage["passed"] = True
        success = True
    finally:
        close_engine(engine, proc, success)


def run_limits(evidence, output, cfg, exe, context, gpu, startup_timeout, stage_timeout, cleanup_timeout):
    tok = tokenizer(common.resolve_path(cfg["tokenizer"], str(Path(cfg.get("cwd") or os.getcwd()).expanduser().resolve())))
    prompt = tok.encode("<|im_start|>user\nList numbered facts from 1 onward; continue.\n<|im_end|>\n"
                        "<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
    require(len(prompt) + 2 + 8 <= context, "limit fixture does not fit context")
    ref_engine, ref_suite, ref_proc, _ = open_engine(evidence, output, cfg, exe, context, gpu, None,
                                                      startup_timeout, cleanup_timeout)
    good_ref = None
    success = False
    try:
        good_ref = ref_suite.solo("max-new-two-reference", prompt, 2)
        require(len(good_ref.tokens) == 2, "max_new edge reference ended before requested length")
        success = True
    finally:
        close_engine(ref_engine, ref_proc, success)
    engine, suite, proc, _ = open_engine(evidence, output, cfg, exe, context, gpu, None,
                                          startup_timeout, cleanup_timeout)
    success = False
    try:
        req = Request("max-new-two-fallback", prompt, 2, 0)
        req.reference = good_ref
        stage = suite.run("proposal-suppressed-at-output-limit", [req])
        require(req.badm and req.badm["continues"], "limit request must enter a batch slot")
        common.parity(req, good_ref)
        require(req.completion["kind"] == "BDONE", "bounded request did not complete in slot")
        stats = stage["stderr_diagnostics"]["batch_mtp_stats"]
        require(stats and sum(s["fallback_limits"] for s in stats) > 0,
                "max_new edge did not demonstrate native target-only limit fallback")
        require(sum(s["offered"] for s in stats) == 0,
                "proposal unexpectedly offered at a two-token remaining-output edge")
        stage["passed"] = True
        success = True
    finally:
        close_engine(engine, proc, success)


def run_lifecycle(evidence, output, cfg, exe, context, gpu, startup_timeout, stage_timeout, cleanup_timeout):
    tok = tokenizer(common.resolve_path(cfg["tokenizer"], str(Path(cfg.get("cwd") or os.getcwd()).expanduser().resolve())))
    a, b = fixtures(tok, context, cap=8)
    refs = run_references(evidence, output, cfg, exe, context, gpu, [a, b], 8,
                          startup_timeout, stage_timeout, cleanup_timeout)
    engine, suite, proc, _ = open_engine(evidence, output, cfg, exe, context, gpu, None,
                                          startup_timeout, cleanup_timeout)
    active = Request("cancel-slot0", a, 16, 0)
    sibling = Request("healthy-slot1", b, 8, 1)
    active.reference = refs[0]
    sibling.reference = refs[1]
    success = False
    try:
        engine.stage = "cancel-and-healthy-reuse"
        stage = {"name": engine.stage, "passed": False}
        proc["stages"].append(stage)
        before = Path(engine.stderr_path).stat().st_size
        protocol = Protocol([active, sibling])
        deadline = time.monotonic() + stage_timeout
        engine.send(active.command(), sibling.command(), deadline=deadline)
        both_admitted = False
        stopped = False
        while not protocol.finished:
            event = engine.next_event(deadline)
            protocol.consume(event)
            if active.badm and sibling.badm:
                both_admitted = True
            if both_admitted and not stopped and event["kind"] == "BT" and event["slot"] == 0:
                engine.send("BSTOP 0", deadline=deadline)
                stage["bstop_after_seq"] = event["seq"]
                stopped = True
        require(stopped and active.completion["finish"] == "cancel", "slot0 was not actively cancelled")
        require(sibling.completion["finish"] in ("length", "stop"), "slot1 did not complete normally")
        common.parity(sibling, refs[1])
        require(active.completion["seq"] > stage["bstop_after_seq"], "BSTOP acknowledgement preceded command")
        raw_stderr = Path(engine.stderr_path).read_text(encoding="utf-8", errors="replace")
        end_byte = len(raw_stderr.encode("utf-8"))
        stage["stderr_byte_range"] = [before, end_byte]
        stage["stderr_diagnostics"] = parse_mtp_diagnostics(
            raw_stderr.encode("utf-8")[before:].decode("utf-8", errors="replace"))
        lifecycle_stats = stage["stderr_diagnostics"]["batch_mtp_stats"]
        require({item["slot"] for item in lifecycle_stats} == {0, 1},
                "cancel and sibling lifecycles must each report native batch-MTP counters")
        require_outcomes(lifecycle_stats, offers=True, allowed_fallbacks=("limits",))
        # Reuse slot 0 after its terminal BDONE and demand exact target parity.
        healthy = Request("healthy-slot0-reuse", a, 8, 0)
        healthy.reference = refs[0]
        next_stage = suite.run("healthy-same-slot-after-cancel", [healthy])
        common.parity(healthy, refs[0])
        require(active.completion["seq"] < healthy.token_events[0]["seq"], "same slot reused before cancel BDONE")
        require_outcomes(next_stage["stderr_diagnostics"]["batch_mtp_stats"], offers=True,
                         allowed_fallbacks=("limits",))
        next_stage["passed"] = True
        stage["cancelled_request"] = active.record()
        stage["sibling_request"] = sibling.record()
        stage["passed"] = True
        success = True
    finally:
        close_engine(engine, proc, success)


def coverage_for(mode: str) -> dict:
    base = {"correctness": "UNTESTED", "dual_overlap": "UNTESTED",
            "tail_partial_page_restore_suppression_offsets_1_2_3": "UNTESTED",
            "coherent_shared_tail_cow_speculation": "UNTESTED (requires a separate live/coherent-source fixture)",
            "optional_reservation_fallback_vs_mandatory_pressure": "UNTESTED", "max_new_edge": "UNTESTED",
            "logical_context_edge": "UNTESTED", "cancel_and_same_slot_reuse": "UNTESTED",
            "partial_yield": "UNTESTED", "handoff_full_and_checkpoint": "UNTESTED",
            "pressure_park_restore_target_only_suppression": "UNTESTED",
            "private_draft_ring_wrap": "UNTESTED (small bounded contexts do not wrap the configured MTP ring; require a separate low-level/native gate)"}
    if mode == "correctness":
        base.update(correctness="RUN", dual_overlap="RUN")
    elif mode == "tails":
        base["tail_partial_page_restore_suppression_offsets_1_2_3"] = (
            "RUN (terminal source slot; requires native target-only marker and zero proposals; not COW speculation)")
    elif mode == "limits":
        base["max_new_edge"] = "RUN"
    elif mode == "lifecycle":
        base["cancel_and_same_slot_reuse"] = "RUN"
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path, help="NEW evidence JSON path; never overwritten")
    ap.add_argument("--mode", choices=("correctness", "tails", "limits", "lifecycle"), default="correctness")
    ap.add_argument("--context", type=int, default=1024, help="small fully resident shared pool; 512..4096, page aligned")
    ap.add_argument("--gpu", help="one physical CUDA/HIP ordinal; default first configured device")
    ap.add_argument("--startup-timeout", type=float, default=900)
    ap.add_argument("--stage-timeout", type=float, default=300, help="absolute deadline per request/stage")
    ap.add_argument("--cleanup-timeout", type=float, default=15)
    args = ap.parse_args(argv)
    if args.context < 512 or args.context > 4096 or args.context % PAGE:
        ap.error("context must be 512..4096 and a multiple of four")
    if args.gpu is not None and (not args.gpu.strip() or "," in args.gpu):
        ap.error("--gpu must select exactly one device")
    if any(not 0 < value < float("inf") for value in
           (args.startup_timeout, args.stage_timeout, args.cleanup_timeout)):
        ap.error("timeouts must be positive and finite")
    output = args.output.expanduser().resolve()
    if output.exists() or any(output.parent.glob(output.name + ".process*.stderr.log")):
        ap.error("evidence/log already exists; choose a new --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        f.write("{}\n")
    cfg_path = args.config.expanduser().resolve()
    evidence = {"schema": 1, "harness": "unified-kv-batch-mtp", "mode": args.mode, "passed": False,
                "config": str(cfg_path), "exe": str(args.exe.expanduser().resolve()), "context": args.context,
                "page_cells": PAGE, "commands": [], "stdout": [], "stages": [], "processes": [],
                "coverage": coverage_for(args.mode),
                "limits": {"startup_deadline_s": args.startup_timeout, "stage_deadline_s": args.stage_timeout,
                           "cleanup_timeout_s": args.cleanup_timeout}}
    success = False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["config_path"] = str(cfg_path)
        # The established NativeEngine stores protocol evidence on process-local
        # dicts; this top-level index makes their complete logs discoverable.
        if args.mode == "correctness":
            run_core(evidence, output, cfg, args.exe.expanduser().resolve(), args.context, args.gpu,
                     args.startup_timeout, args.stage_timeout, args.cleanup_timeout)
        elif args.mode == "tails":
            run_tails(evidence, output, cfg, args.exe.expanduser().resolve(), args.context, args.gpu,
                      args.startup_timeout, args.stage_timeout, args.cleanup_timeout)
        elif args.mode == "limits":
            run_limits(evidence, output, cfg, args.exe.expanduser().resolve(), args.context, args.gpu,
                       args.startup_timeout, args.stage_timeout, args.cleanup_timeout)
        else:
            run_lifecycle(evidence, output, cfg, args.exe.expanduser().resolve(), args.context, args.gpu,
                          args.startup_timeout, args.stage_timeout, args.cleanup_timeout)
        success = True
    except (Exception, KeyboardInterrupt) as exc:
        evidence["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        evidence["passed"] = success
        evidence["finished_wall_time"] = time.time()
        output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"{'PASS' if success else 'FAIL'}: {output}", flush=True)
    if not success:
        print(json.dumps(evidence.get("failure")), file=sys.stderr)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
