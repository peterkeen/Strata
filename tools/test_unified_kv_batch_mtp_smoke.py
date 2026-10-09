#!/usr/bin/env python3
"""Offline parser/config/protocol tests for unified_kv_batch_mtp_smoke (no GPU/models)."""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unified_kv_batch_mtp_smoke as smoke
import unified_kv_smoke as common


def mtp_stats(slot=0, *, windows=2, offered=2, accepted=1, rejected=1, discarded=0,
              fallbacks=0, incoherent=0, not_ready=0, limits=0, capacity=0, reserve=0):
    return (f"strata batch_mtp_stats slot={slot} windows={windows} offered={offered} accepted={accepted} "
            f"rejected={rejected} discarded={discarded} fallback_attempts={fallbacks} "
            f"fallback_incoherent={incoherent} fallback_not_ready={not_ready} fallback_limits={limits} "
            f"fallback_capacity={capacity} fallback_reserve={reserve}")


def good_info(context=1024):
    return {"kv_unified": "1", "kv_capacity_cells": str(context), "context": str(context), "batch_slots": "2",
            "kv_resident": "0", "conversation_cache_mib": "0", "slot_cache": "1", "lookup": "0",
            "mtp_max": "1", "pcie_frac": "0.00", "batch_groups": "1"}


def completed(name, prompt, tokens, slot):
    req = common.Request(name, prompt, len(tokens), slot)
    if slot is None:
        events = [*(f"T {token}" for token in tokens), f"DONE {len(tokens)} {len(prompt)} 0 0 length"]
    else:
        events = [f"T {tokens[0]}", f"DONE 1 {len(prompt)} 0 0 length", f"BADM {slot} 1"]
        events.extend(f"BT {slot} {token}" for token in tokens[1:])
        events.append(f"BDONE {slot} {len(tokens)} length 0")
    protocol = common.Protocol([req])
    for seq, raw in enumerate(events):
        event = common.parse_line(raw) | {"raw": raw, "seq": seq, "wall_s": seq / 10}
        protocol.consume(event)
    return req


class WordTokenizer:
    def __init__(self):
        self.vocab = {}

    def encode(self, text, parse_special=False):
        return [self.vocab.setdefault(word, len(self.vocab) + 1) for word in text.split()]


class DiagnosticTests(unittest.TestCase):
    def test_stats_parse_and_invariants(self):
        parsed = smoke.parse_mtp_diagnostics(mtp_stats(0) + "\n" + mtp_stats(1, windows=3, offered=2,
              accepted=0, rejected=1, discarded=1, fallbacks=1, limits=1))
        self.assertEqual([s["slot"] for s in parsed["batch_mtp_stats"]], [0, 1])
        self.assertEqual(parsed["batch_mtp_stats"][1]["discarded"], 1)
        self.assertEqual(parsed["batch_mtp_stats"][1]["fallback_limits"], 1)
        proof = smoke.require_outcomes(parsed["batch_mtp_stats"], offers=True, accepted=True, rejected=True,
                                       allowed_fallbacks=("limits",))
        self.assertEqual((proof["offered"], proof["accepted"], proof["rejected"]), (4, 1, 2))

    def test_vacuous_or_single_outcome_proof_fails(self):
        cases = [mtp_stats(windows=0, offered=0, accepted=0, rejected=0),
                 mtp_stats(windows=1, offered=1, accepted=1, rejected=0),
                 mtp_stats(windows=1, offered=1, accepted=0, rejected=1)]
        for line in cases:
            stats = smoke.parse_mtp_diagnostics(line)["batch_mtp_stats"]
            for required in ((True, True, True),):
                with self.subTest(line=line), self.assertRaises(AssertionError):
                    smoke.require_outcomes(stats, offers=required[0], accepted=required[1], rejected=required[2])

    def test_terminal_limit_fallback_is_allowed_but_unexpected_or_repeated_fallback_fails(self):
        stats = smoke.parse_mtp_diagnostics(
            mtp_stats(0, windows=9, offered=8, accepted=8, rejected=0, fallbacks=1, limits=1) + "\n" +
            mtp_stats(1, windows=9, offered=8, accepted=7, rejected=1, fallbacks=1, limits=1)
        )["batch_mtp_stats"]
        proof = smoke.require_outcomes(stats, offers=True, accepted=True, rejected=True,
                                       allowed_fallbacks=("limits",))
        self.assertEqual(proof["fallbacks_by_reason"]["limits"], 2)
        self.assertEqual(proof["allowed_fallbacks"], ["limits"])
        incoherent = smoke.parse_mtp_diagnostics(
            mtp_stats(0, windows=9, offered=8, accepted=8, rejected=0, fallbacks=1, incoherent=1)
        )["batch_mtp_stats"]
        with self.assertRaisesRegex(AssertionError, "unexpected target-only fallback"):
            smoke.require_outcomes(incoherent, allowed_fallbacks=("limits",))
        repeated = smoke.parse_mtp_diagnostics(
            mtp_stats(0, windows=10, offered=8, accepted=8, rejected=0, fallbacks=2, limits=2)
        )["batch_mtp_stats"]
        with self.assertRaisesRegex(AssertionError, "more than one terminal"):
            smoke.require_outcomes(repeated, allowed_fallbacks=("limits",))

    def test_terminal_target_only_tail_restore_rejects_any_offer(self):
        diagnostics = {"target_only_suppression": ["suppressed until full replay"],
                       "batch_mtp_stats": smoke.parse_mtp_diagnostics(
                           mtp_stats(1, windows=3, offered=0, accepted=0, rejected=0,
                                     fallbacks=3, incoherent=2, limits=1)
                       )["batch_mtp_stats"]}
        proof = smoke.verify_target_only_tail_restore(diagnostics)
        self.assertEqual(proof["offered"], 0)
        self.assertEqual(proof["fallbacks_by_reason"]["incoherent"], 2)
        diagnostics["batch_mtp_stats"] = smoke.parse_mtp_diagnostics(
            mtp_stats(1, windows=3, offered=1, accepted=1, rejected=0, fallbacks=2, incoherent=2)
        )["batch_mtp_stats"]
        with self.assertRaisesRegex(AssertionError, "unexpectedly offered"):
            smoke.verify_target_only_tail_restore(diagnostics)

    def test_malformed_duplicate_and_inconsistent_counters_fail(self):
        with self.assertRaisesRegex(AssertionError, "malformed"):
            smoke.parse_mtp_diagnostics("strata batch_mtp_stats slot=x")
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            smoke.parse_mtp_diagnostics(mtp_stats(0) + "\n" + mtp_stats(0))
        with self.assertRaisesRegex(AssertionError, "offered !="):
            smoke.parse_mtp_diagnostics(mtp_stats(windows=2, offered=2, accepted=1, rejected=0))
        with self.assertRaisesRegex(AssertionError, "fallback reason"):
            smoke.parse_mtp_diagnostics(mtp_stats(windows=2, offered=1, accepted=1, rejected=0,
                                                    fallbacks=1, limits=0))

    def test_startup_requires_real_unified_allocation_and_two_slots(self):
        allocation = "strata generate: unified KV: 1024 total backing cells shared by admission and 2 slots, 1024 GPU-resident cells per layer\n"
        result = smoke.verify_batch_startup(good_info(), 1024, allocation)
        self.assertEqual(result["allocation"]["cells"], 1024)
        for bad in ("", allocation.replace("2 slots", "1 slots"), allocation.replace("1024 total", "2048 total"),
                    allocation.replace("1024 GPU-resident", "512 GPU-resident")):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                smoke.verify_batch_startup(good_info(), 1024, bad)

    def test_hook_activation_and_slot_copy_are_parsed_as_diagnostics_not_proof(self):
        text = ("strata batch_mtp_test_hook enabled: explicit proposal substitutions are test-only\n"
                "strata serve: TARGET_ONLY slot clone: private MTP proposals suppressed until full replay\n"
                "strata batch: slot 0 gave back 121 tokens of this conversation (all it holds) in 1.0 ms\n")
        parsed = smoke.parse_mtp_diagnostics(text)
        self.assertEqual(len(parsed["hook_lines"]), 1)
        self.assertEqual(len(parsed["target_only_suppression"]), 1)
        self.assertEqual(parsed["slot_copies"][0]["tokens"], 121)
        with self.assertRaises(AssertionError):
            smoke.require_outcomes(parsed["batch_mtp_stats"], offers=True)


class OverlapTests(unittest.TestCase):
    def pair(self, dual=True, serial=False):
        prompts = ([1, 2, 3], [4, 5, 6, 7])
        refs = [completed("ra", prompts[0], [10, 11, 12], None),
                completed("rb", prompts[1], [20, 21, 22], None)]
        a = common.Request("a", prompts[0], 3, 0)
        b = common.Request("b", prompts[1], 3, 1)
        a.reference, b.reference = refs
        lines = ["T 10", "DONE 1 3 0 0 length", "BADM 0 1",
                 "T 20", "DONE 1 4 0 0 length", "BADM 1 1"]
        if serial:
            lines.extend(["BT 0 11", "BT 0 12", "BDONE 0 3 length 0",
                          "BT 1 21", "BT 1 22", "BDONE 1 3 length 0"])
        elif dual:
            lines.extend(["BT 0 11", "BT 1 21", "BT 0 12", "BT 1 22",
                          "BDONE 0 3 length 0", "BDONE 1 3 length 0"])
        else:
            lines.extend(["BT 0 11", "BT 0 12", "BDONE 0 3 length 0", "BDONE 1 1 length 0"])
        protocol = common.Protocol([a, b])
        for seq, raw in enumerate(lines):
            protocol.consume(common.parse_line(raw) | {"raw": raw, "seq": seq, "wall_s": seq / 10})
        return [a, b]

    def test_requires_both_slots_to_progress_after_both_badm_and_target_parity(self):
        reqs = self.pair()
        proof = smoke.verify_dual_overlap(reqs)
        self.assertTrue(proof["actual_dual_slot_progress"])
        self.assertTrue(proof["exact_target_token_ids"])
        with self.assertRaisesRegex(AssertionError, "slot 1 made no BT progress"):
            smoke.verify_dual_overlap(self.pair(dual=False))

    def test_rejects_early_completion_before_second_admission(self):
        reqs = self.pair()
        reqs[0].completion["seq"] = reqs[1].badm["seq"] - 1
        with self.assertRaises(AssertionError):
            smoke.verify_dual_overlap(reqs)

    def test_rejects_serial_post_admission_progress_with_exact_parity(self):
        reqs = self.pair(serial=True)
        for request in reqs:
            common.parity(request, request.reference)
        self.assertLess(reqs[0].completion["seq"], reqs[1].token_events[1]["seq"])
        with self.assertRaisesRegex(AssertionError, "before the first slot completed"):
            smoke.verify_dual_overlap(reqs)


class FixtureTests(unittest.TestCase):
    def test_dual_fixture_token_counts_are_explicitly_unequal(self):
        a, b = smoke.fixtures(WordTokenizer(), context=1024)
        self.assertNotEqual(a, b)
        self.assertNotEqual(len(a), len(b))
        self.assertLessEqual(max(len(a), len(b)) + 16 + 8, 1024)

    def test_run_core_fake_completed_16_tokens_accept_reject_and_terminal_limit_rows(self):
        prompts = ([1, 2, 3], [4, 5, 6, 7])
        refs = [completed("solo-a", prompts[0], list(range(10, 26)), None),
                completed("solo-b", prompts[1], list(range(30, 46)), None)]

        class FakeSuite:
            def __init__(self, process):
                self.process = process

            def run(self, name, requests):
                protocol = common.Protocol(requests)
                raw_lines = []
                for req in requests:
                    raw_lines.extend([f"T {req.reference.tokens[0]}",
                                      f"DONE 1 {len(req.prompt)} 0 0 length", f"BADM {req.slot} 1"])
                for token_index in range(1, 16):
                    for req in requests:
                        raw_lines.append(f"BT {req.slot} {req.reference.tokens[token_index]}")
                raw_lines.extend(f"BDONE {req.slot} 16 length 0" for req in requests)
                for seq, raw in enumerate(raw_lines):
                    protocol.consume(common.parse_line(raw) | {"raw": raw, "seq": seq, "wall_s": seq / 100})
                stats = "\n".join([
                    mtp_stats(0, windows=9, offered=8, accepted=8, rejected=0, fallbacks=1, limits=1),
                    mtp_stats(1, windows=9, offered=8, accepted=7, rejected=1, fallbacks=1, limits=1),
                ])
                stage = {"name": name, "requests": [req.record() for req in requests],
                         "stderr_diagnostics": smoke.parse_mtp_diagnostics(stats)}
                self.process["stages"].append(stage)
                return stage

        evidence = {"processes": []}
        process_env = {"CUDA_VISIBLE_DEVICES": "0", "STRATA_BATCH_MTP": "1"}

        def fake_references(root, *_args, **_kwargs):
            root["processes"].append({"command": ["same-native-command"], "env_overrides": dict(process_env)})
            return refs

        def fake_open_engine(root, *_args, **_kwargs):
            process = {"command": ["same-native-command"],
                       "env_overrides": process_env | {"STRATA_BATCH_MTP_TEST_PROPOSALS": "0/1/11,1/1/10"},
                       "startup_diagnostics": {"hook_lines": ["explicit proposal substitutions are test-only"]},
                       "stdout": [], "stages": []}
            root["processes"].append(process)
            return object(), FakeSuite(process), process, Path("/tmp/fake-mtp-stderr")

        patchers = [patch.object(smoke, "tokenizer", return_value=WordTokenizer()),
                    patch.object(smoke, "fixtures", return_value=prompts),
                    patch.object(smoke, "run_references", side_effect=fake_references),
                    patch.object(smoke, "open_engine", side_effect=fake_open_engine),
                    patch.object(smoke, "close_engine")]
        with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4]:
            smoke.run_core(evidence, Path("/tmp/fake-mtp-evidence.json"),
                           {"tokenizer": "unused", "cwd": "/tmp"}, Path("/tmp/fake-engine"),
                           1024, "0", 1, 1, 1)

        core_stage = evidence["processes"][-1]["stages"][0]
        self.assertEqual([len(req["tokens"]) for req in core_stage["requests"]], [16, 16])
        self.assertEqual(core_stage["proposal_proof"]["fallbacks_by_reason"]["limits"], 2)
        self.assertEqual(core_stage["natural_catchup_offers_after_forced_first"], {"slot0": 7, "slot1": 7})
        self.assertTrue(core_stage["passed"])
        self.assertEqual(evidence["reference_batch_policy"]["native_command_identical"], True)
        self.assertEqual(evidence["deterministic_proposal_schedule"]["slot0"]["offer"], 1)
        self.assertEqual(evidence["deterministic_proposal_schedule"]["slot1"]["offer"], 1)
        self.assertEqual(len(refs[0].tokens), 16)
        self.assertEqual(len(refs[1].tokens), 16)


class SettingsTests(unittest.TestCase):
    def config(self):
        return {"cwd": "/tmp", "gpu": [2, 3], "tokenizer": "model/tokenizer", "lib_dirs": ["cuda/lib"],
                "env": {"MODEL_TUNING": "retained", "STRATA_IQ_MT_MIN": "8", "STRATA_BATCH_DECODE_SHARE": "0.5"},
                "args": ["--pack", "model/pack", "--mtp", "model/mtp", "--native", "model/weights.gguf",
                         "--max-context", "65536", "--batch", "8", "--batch-groups", "auto", "--layer-split", "auto",
                         "--kv-resident", "32768", "--spec", "4", "--mtp-max-t", "3", "--suffix-draft", "2",
                         "--adapt-swaps", "12", "--prefill", "auto:8192"]}

    def test_retains_actual_artifacts_and_builds_explicit_one_gpu_policy(self):
        command, cwd, env, meta = smoke.engine_settings(self.config(), "/tmp/engine", 1024,
                                                         hook_schedule="0/1/7,1/1/8")
        self.assertEqual(cwd, "/tmp")
        self.assertEqual(command[command.index("--mtp") + 1], "model/mtp")
        self.assertEqual(command[command.index("--pack") + 1], "model/pack")
        self.assertEqual(command[command.index("--native") + 1], "model/weights.gguf")
        for key, value in (("--batch", "2"), ("--batch-groups", "1"), ("--max-context", "1024"),
                           ("--kv-resident", "0"), ("--spec", "2"), ("--mtp-max-t", "1"),
                           ("--suffix-draft", "0"), ("--prefill", "512"), ("--pcie-frac", "0")):
            self.assertEqual(command.count(key), 1)
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn("--kv-unified", command)
        self.assertIn("--batch-mtp", command)
        self.assertIn("--no-prefill-borrow", command)
        self.assertNotIn("--layer-split", command)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(env["STRATA_IQ_MT_MIN"], "1")
        self.assertEqual(env["STRATA_BATCH_MTP"], "1")
        self.assertEqual(env["STRATA_BATCH_DECODE_SHARE"], "0")
        self.assertEqual(env["STRATA_BATCH_MTP_TEST_PROPOSALS"], "0/1/7,1/1/8")
        self.assertEqual(env["MODEL_TUNING"], "retained")
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/tmp/cuda/lib"))
        self.assertEqual(meta["mtp_path"], "/tmp/model/mtp")

    def test_explicit_gpu_and_no_hook_by_default(self):
        _, _, env, _ = smoke.engine_settings(self.config(), "/tmp/engine", 1024, gpu="7")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7")
        self.assertNotIn("STRATA_BATCH_MTP_TEST_PROPOSALS", env)

    def test_rejects_missing_mtp_bad_context_or_multi_gpu_selector(self):
        cfg = self.config()
        cfg["args"].remove("--mtp")
        cfg["args"].remove("model/mtp")
        with self.assertRaisesRegex(AssertionError, "exactly one actual --mtp"):
            smoke.engine_settings(cfg, "/tmp/engine", 1024)
        with self.assertRaisesRegex(AssertionError, "aligned"):
            smoke.engine_settings(self.config(), "/tmp/engine", 1025)
        with self.assertRaisesRegex(AssertionError, "one physical GPU"):
            smoke.engine_settings(self.config(), "/tmp/engine", 1024, gpu="0,1")

    def test_assets_require_executable_tokenizer_and_mtp_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root / "engine"
            exe.touch()
            mtp = root / "mtp"
            mtp.mkdir()
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            cfg = {"config_path": str(root / "config.json")}
            Path(cfg["config_path"]).touch()
            meta = {"mtp_path": str(mtp), "tokenizer_path": str(tokenizer)}
            smoke.verify_assets(cfg, exe, meta)
            with self.assertRaises(AssertionError):
                smoke.verify_assets(cfg, exe, {**meta, "mtp_path": str(root / "missing")})

    def test_existing_default_unified_smoke_remains_target_only(self):
        cfg = {"cwd": "/tmp", "gpu": 0, "args": ["--native", "weights"]}
        command, _, _ = common.engine_settings(cfg, "/tmp/engine", 512)
        self.assertIn("--kv-unified", command)
        self.assertNotIn("--batch-mtp", command)

    def test_modes_report_unimplemented_coverage_explicitly(self):
        coverage = smoke.coverage_for("correctness")
        self.assertEqual(coverage["dual_overlap"], "RUN")
        for key in ("tail_partial_page_restore_suppression_offsets_1_2_3",
                    "coherent_shared_tail_cow_speculation", "optional_reservation_fallback_vs_mandatory_pressure",
                    "logical_context_edge", "partial_yield", "handoff_full_and_checkpoint",
                    "pressure_park_restore_target_only_suppression"):
            self.assertIn("UNTESTED", coverage[key])
        tails = smoke.coverage_for("tails")
        self.assertIn("RUN", tails["tail_partial_page_restore_suppression_offsets_1_2_3"])
        self.assertIn("UNTESTED", tails["coherent_shared_tail_cow_speculation"])
        self.assertIn("UNTESTED", coverage["private_draft_ring_wrap"])


class DeadlineCleanupTests(unittest.TestCase):
    def test_native_reader_deadline_and_terminate_cleanup_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = {"stdout": [], "commands": []}
            engine = common.NativeEngine(evidence, Path(tmp) / "stderr.log", cleanup_timeout=0.2)
            info = "INFO " + " ".join(f"{k}={v}" for k, v in good_info().items())
            script = ("import time\n" + f"print({info!r}, flush=True)\n" + "print('READY 1024 stop', flush=True)\n"
                      + "time.sleep(60)\n")
            try:
                engine.start([sys.executable, "-u", "-c", script], os.getcwd(), dict(os.environ), 1024, 1)
                with self.assertRaises(TimeoutError):
                    engine.next_event(time.monotonic() - 1)
            finally:
                engine.close(False)
            self.assertIsNotNone(engine.p.poll())
            self.assertTrue(evidence["cleanup"]["reader_stopped"])
            self.assertTrue(evidence["cleanup"]["terminated"])


if __name__ == "__main__":
    unittest.main()
