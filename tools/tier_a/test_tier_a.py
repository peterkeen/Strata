"""tools/tier_a/test_tier_a.py - offline unit tests for the Tier-A harness.

All tests run without a GPU and without model files.  Token ids are synthesised.
Engine subprocesses are replaced by tiny fixture binaries.

DEPENDENCY NOTE
───────────────
Tests that exercise strata_tokenizer (TestTokenizePrompt) require `regex`.
Set PYTHONPATH before running if regex is not installed system-wide:

    PYTHONPATH=/home/pete/.cache/uv/archive-v0/wicKyd1x7CYO0Nj0/lib/python3.11/site-packages \
        python3 -m unittest discover -s tools/tier_a -p test_*.py

All other test classes (TestParseStdout, TestBuildEnv, TestRunArm*, TestMathFinite,
TestPrompts, TestHelpInvocation, TestRunMatrixDryRun) pass with bare Python 3.11.

Usage:
    python3 -m unittest discover -s tools/tier_a -p test_*.py
    python3 -m unittest tools.tier_a.test_tier_a
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from tier_a.engine_cli import (  # noqa: E402
    ArmResult, build_env, math_finite, parse_stdout, run_arm,
)
from tier_a.prompts import LONG_DOC_Q, SHORT_CODE, STORY, prompt_token_count_approx  # noqa: E402

# ── stdout fixture factory ────────────────────────────────────────────────────

def _make_stdout(
    prompt_ids=(1, 2, 3),
    output_ids=(10, 20, 30, 40),
    decode_ms=120.0,
    decode_tok_s=33.33,
    prefill_ms=85.0,
    prefill_tok_s=100.0,
    ttft_ms=90.0,
    spec_rounds=5,
    spec_n=4,
    draft_accepted=3,
    draft_offered=8,
    draft_rate=0.375,
    tok_per_round=1.6,
) -> str:
    pid = " ".join(str(i) for i in prompt_ids)
    oid = " ".join(str(i) for i in output_ids)
    n_out = len(output_ids)
    n_pre = len(prompt_ids) - 1
    return textwrap.dedent(f"""\
        prompt  : {pid}
        output  : {oid}
        decode                   {n_out} tokens in {decode_ms:.1f} ms  ->  {decode_tok_s:.2f} tok/s
        prefill                  {n_pre} tokens in {prefill_ms:.1f} ms  ->  {prefill_tok_s:.2f} tok/s  (time to first token {ttft_ms:.1f} ms)
        speculation              {spec_rounds} rounds of {spec_n}, drafts accepted {draft_accepted} of {draft_offered} ({draft_rate:.3f}), {tok_per_round:.2f} tokens per round
    """)


# ── fixture binary helpers ────────────────────────────────────────────────────

def _write_fixture(tmp_dir: str, stdout_content: str, exit_code: int = 0,
                   extra_stderr: str = "") -> str:
    script = Path(tmp_dir) / "fake_strata.sh"
    safe = stdout_content.replace("'", "'\\''")
    err_line = f"echo '{extra_stderr}' >&2\n" if extra_stderr else ""
    script.write_text(
        f"#!/bin/sh\n{err_line}printf '%s' '{safe}'\nexit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def _write_fixture_reads_tokens(tmp_dir: str, exit_code: int = 0) -> str:
    """Fixture that reads the --tokens-file path from argv, echoes its content
    on stdout as 'token_file_content: <ids>', plus a minimal stats block."""
    script = Path(tmp_dir) / "fake_strata_reads.sh"
    script.write_text(textwrap.dedent("""\
        #!/bin/sh
        # parse --tokens-file <path> from argv
        ids_file=""
        while [ $# -gt 0 ]; do
            if [ "$1" = "--tokens-file" ]; then
                ids_file="$2"
                shift 2
            else
                shift
            fi
        done
        if [ -n "$ids_file" ] && [ -f "$ids_file" ]; then
            ids=$(cat "$ids_file")
        else
            ids="MISSING"
        fi
        printf 'prompt  : 1 2 3\\n'
        printf 'output  : 10 20 30 40\\n'
        printf 'decode                   4 tokens in 120.0 ms  ->  33.33 tok/s\\n'
        printf 'prefill                  2 tokens in 85.0 ms  ->  100.00 tok/s  (time to first token 90.0 ms)\\n'
        printf 'speculation              5 rounds of 4, drafts accepted 3 of 8 (0.375), 1.60 tokens per round\\n'
        printf 'token_file_content: %s\\n' "$ids"
        exit 0
    """), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


# ── TestParseStdout ───────────────────────────────────────────────────────────

class TestParseStdout(unittest.TestCase):

    def test_normal_output(self):
        p = parse_stdout(_make_stdout())
        self.assertEqual(p["prompt_ids"], [1, 2, 3])
        self.assertEqual(p["output_ids"], [10, 20, 30, 40])
        self.assertEqual(p["n_output"], 4)
        self.assertAlmostEqual(p["decode_ms"], 120.0)
        self.assertAlmostEqual(p["decode_tok_s"], 33.33, places=1)
        self.assertAlmostEqual(p["prefill_ms"], 85.0)
        self.assertAlmostEqual(p["ttft_ms"], 90.0)
        self.assertEqual(p["draft_accepted"], 3)
        self.assertEqual(p["draft_offered"], 8)
        self.assertAlmostEqual(p["draft_rate"], 0.375)
        self.assertAlmostEqual(p["tokens_per_round"], 1.6, places=1)

    def test_empty_output_line(self):
        txt = "prompt  : 1 2\noutput  :\ndecode                   0 tokens in 0.0 ms  ->  0.00 tok/s\n"
        p = parse_stdout(txt)
        self.assertEqual(p["output_ids"], [])
        self.assertEqual(p["n_output"], 0)

    def test_no_prefill_line(self):
        txt = "\n".join(l for l in _make_stdout(prompt_ids=(1,)).splitlines()
                        if "prefill" not in l) + "\n"
        p = parse_stdout(txt)
        self.assertEqual(p["prefill_ms"], 0.0)
        self.assertEqual(p["ttft_ms"], 0.0)

    def test_hash_stability(self):
        ids = (100, 200, 300)
        p = parse_stdout(_make_stdout(output_ids=ids))
        expected = hashlib.sha256("100 200 300".encode()).hexdigest()[:16]
        self.assertEqual(p["output_hash"], expected)

    def test_different_ids_differ_hash(self):
        p1 = parse_stdout(_make_stdout(output_ids=(1, 2, 3)))
        p2 = parse_stdout(_make_stdout(output_ids=(1, 2, 4)))
        self.assertNotEqual(p1["output_hash"], p2["output_hash"])

    def test_completely_empty(self):
        p = parse_stdout("")
        self.assertEqual(p["output_ids"], [])
        self.assertEqual(p["decode_ms"], 0.0)


# ── TestBuildEnv ──────────────────────────────────────────────────────────────

class TestBuildEnv(unittest.TestCase):

    def test_controlled_vars_stripped(self):
        base = {"PATH": "/usr/bin", "STRATA_GR_DOWN_MAX4": "1", "HOME": "/home/x"}
        env = build_env(base, ("STRATA_GR_DOWN_MAX4",), {})
        self.assertNotIn("STRATA_GR_DOWN_MAX4", env)
        self.assertIn("PATH", env)

    def test_arm_override_applied(self):
        base = {"STRATA_PF_FUSED": "0"}
        env = build_env(base, ("STRATA_PF_FUSED",), {"STRATA_PF_FUSED": "1"})
        self.assertEqual(env["STRATA_PF_FUSED"], "1")

    def test_uncontrolled_pass_through(self):
        env = build_env({"MYVAR": "hello"}, ("STRATA_GR_DOWN_MAX4",), {})
        self.assertEqual(env["MYVAR"], "hello")


# ── TestMathFinite ────────────────────────────────────────────────────────────

class TestMathFinite(unittest.TestCase):
    def test_finite(self):    self.assertTrue(math_finite(1.0))
    def test_zero(self):      self.assertTrue(math_finite(0.0))
    def test_inf(self):       self.assertFalse(math_finite(float("inf")))
    def test_nan(self):       self.assertFalse(math_finite(float("nan")))
    def test_none(self):      self.assertFalse(math_finite(None))
    def test_str(self):       self.assertFalse(math_finite("abc"))


# ── TestRunArmDryRun ──────────────────────────────────────────────────────────

class TestRunArmDryRun(unittest.TestCase):

    def test_dry_run_sentinel(self):
        r = run_arm("/nonexistent/strata", [], [1, 2, 3], 8,
                    {}, (), 5.0, dry_run=True)
        self.assertEqual(r.error, "dry_run")
        self.assertEqual(r.exit_code, 0)
        self.assertEqual(r.output_hash, "dry_run"[:16])
        self.assertEqual(r.n_output, 0)

    def test_dry_run_argv_contains_tokens_file_flag(self):
        """argv must contain --tokens-file even in dry_run, so artifact logs
        can show the intended invocation."""
        r = run_arm("/nonexistent/strata", ["--pack", "/fake/pack"], [1, 2, 3], 8,
                    {}, (), 5.0, dry_run=True)
        self.assertIn("--tokens-file", r.argv)
        # The tokens-file path is the element after --tokens-file
        idx = r.argv.index("--tokens-file")
        tf_path = r.argv[idx + 1]
        self.assertTrue(tf_path.endswith(".ids"), f"expected .ids suffix: {tf_path}")

    def test_dry_run_argv_contains_pack_and_expert_cache(self):
        """Ensure caller-supplied flags (--pack, --expert-cache) appear in argv."""
        flags = ["--pack", "/data/pack", "--expert-cache", "2000", "--pcie-frac", "0"]
        r = run_arm("/fake/strata", flags, [1, 2, 3], 32, {}, (), 5.0, dry_run=True)
        self.assertIn("--pack", r.argv)
        self.assertIn("--expert-cache", r.argv)
        self.assertIn("--pcie-frac", r.argv)
        self.assertIn("2000", r.argv)
        self.assertIn("0", r.argv)


# ── TestRunArmTokensFile ──────────────────────────────────────────────────────

class TestRunArmTokensFile(unittest.TestCase):
    """Verify the --tokens-file contract: flag name, file created with ids,
    child can read the file, file is cleaned up after run."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="tier_a_tokfile_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_flag_is_tokens_file_not_tokens(self):
        """The CLI flag must be --tokens-file, not --tokens."""
        exe = _write_fixture(self._tmp, _make_stdout(output_ids=(1, 2, 3, 4)))
        r = run_arm(exe, [], [10, 20, 30], 4, {}, (), 10.0, tmp_dir=self._tmp)
        self.assertIn("--tokens-file", r.argv,
                      f"argv={r.argv}: expected --tokens-file, not --tokens")
        self.assertNotIn("--tokens", [a for a in r.argv if a == "--tokens"],
                         "argv must not contain bare --tokens flag")

    def test_ids_file_cleaned_up_after_success(self):
        """The --tokens-file temp file must be deleted after a clean run."""
        exe = _write_fixture(self._tmp, _make_stdout(output_ids=(1, 2, 3, 4)))
        r = run_arm(exe, [], [10, 20, 30], 4, {}, (), 10.0, tmp_dir=self._tmp)
        idx = r.argv.index("--tokens-file")
        tf_path = r.argv[idx + 1]
        self.assertFalse(Path(tf_path).exists(),
                         f"tokens-file not cleaned up: {tf_path}")

    def test_ids_file_cleaned_up_after_error(self):
        """Temp file must be cleaned up even when the child exits non-zero."""
        exe = _write_fixture(self._tmp, _make_stdout(output_ids=(1, 2, 3, 4)),
                             exit_code=1)
        r = run_arm(exe, [], [10, 20, 30], 4, {}, (), 10.0, tmp_dir=self._tmp)
        self.assertIsNotNone(r.error)
        idx = r.argv.index("--tokens-file")
        tf_path = r.argv[idx + 1]
        self.assertFalse(Path(tf_path).exists(),
                         f"tokens-file not cleaned up on error: {tf_path}")

    def test_ids_file_cleaned_up_after_timeout(self):
        """Temp file must be cleaned up even after a timeout kill."""
        script = Path(self._tmp) / "hang.sh"
        script.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        r = run_arm(str(script), [], [1, 2], 8, {}, (), 0.3, tmp_dir=self._tmp)
        self.assertIn("timeout", r.error)
        idx = r.argv.index("--tokens-file")
        tf_path = r.argv[idx + 1]
        self.assertFalse(Path(tf_path).exists(),
                         f"tokens-file not cleaned up after timeout: {tf_path}")

    def test_child_receives_ids_via_tokens_file(self):
        """The child process can read the actual token ids from the file."""
        exe = _write_fixture_reads_tokens(self._tmp)
        token_ids = [101, 202, 303, 404]
        r = run_arm(exe, [], token_ids, 4, {}, (), 10.0, tmp_dir=self._tmp)
        # The fixture echoes 'token_file_content: <ids>' on stdout
        expected_ids = " ".join(str(i) for i in token_ids)
        self.assertIn(expected_ids, r.stdout_raw,
                      f"ids not present in stdout; stdout={r.stdout_raw[:200]}")

    def test_argv_contains_pack_expert_cache_pcie_stats(self):
        """Caller-supplied --pack, --expert-cache, --pcie-frac, --stats flags
        must appear in argv (they are part of FIXED_FLAGS in production)."""
        flags = [
            "--pack", "/data/llm/Strata-data/packs/iq3_s",
            "--expert-cache", "2000",
            "--pcie-frac", "0",
            "--stats",
        ]
        exe = _write_fixture(self._tmp, _make_stdout(output_ids=(1, 2, 3, 4)))
        r = run_arm(exe, flags, [1, 2, 3], 4, {}, (), 10.0, tmp_dir=self._tmp)
        for flag in ("--pack", "--expert-cache", "--pcie-frac", "--stats"):
            self.assertIn(flag, r.argv, f"missing {flag} in argv")
        pack_idx = r.argv.index("--pack")
        self.assertIn("iq3_s", r.argv[pack_idx + 1])
        cache_idx = r.argv.index("--expert-cache")
        self.assertEqual(r.argv[cache_idx + 1], "2000")
        pcie_idx = r.argv.index("--pcie-frac")
        self.assertEqual(r.argv[pcie_idx + 1], "0")


# ── TestRunArmFixture ─────────────────────────────────────────────────────────

class TestRunArmFixture(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="tier_a_fixture_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _run(self, stdout: str, exit_code: int = 0, **kw):
        exe = _write_fixture(self._tmp, stdout, exit_code)
        return run_arm(exe, [], [1, 2, 3], 8,
                       {}, (), 10.0, dry_run=False, tmp_dir=self._tmp, **kw)

    def test_clean_run(self):
        r = self._run(_make_stdout(output_ids=(10, 20, 30, 40, 50, 60, 70, 80)))
        self.assertIsNone(r.error)
        self.assertEqual(r.n_output, 8)
        self.assertAlmostEqual(r.decode_ms, 120.0)
        self.assertEqual(r.exit_code, 0)

    def test_nonzero_exit_sets_error(self):
        r = self._run(_make_stdout(output_ids=(1, 2, 3, 4)), exit_code=1)
        self.assertIsNotNone(r.error)
        self.assertIn("exit_code", r.error)

    def test_empty_stdout_sets_error(self):
        r = self._run("")
        self.assertIsNotNone(r.error)
        self.assertIn("empty_stdout", r.error)

    def test_zero_output_tokens_sets_error(self):
        txt = "prompt  : 1 2 3\noutput  :\ndecode                   0 tokens in 0.0 ms  ->  0.00 tok/s\n"
        r = self._run(txt)
        self.assertIsNotNone(r.error)

    def test_stderr_captured(self):
        exe = _write_fixture(self._tmp, _make_stdout(output_ids=(1, 2, 3, 4)),
                             extra_stderr="strata: loaded")
        r = run_arm(exe, [], [1, 2, 3], 4, {}, (), 10.0, tmp_dir=self._tmp)
        self.assertIn("strata: loaded", r.stderr_raw)

    def test_argv_recorded_in_result(self):
        r = self._run(_make_stdout(output_ids=(1, 2, 3, 4)))
        self.assertIsInstance(r.argv, list)
        self.assertGreater(len(r.argv), 0)
        self.assertIn("--tokens-file", r.argv)
        self.assertIn("--max-new", r.argv)


# ── TestRunArmTimeout ─────────────────────────────────────────────────────────

class TestRunArmTimeout(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="tier_a_timeout_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_timeout_sets_error(self):
        script = Path(self._tmp) / "hang.sh"
        script.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        r = run_arm(str(script), [], [1, 2, 3], 8, {}, (), 0.3, tmp_dir=self._tmp)
        self.assertIsNotNone(r.error)
        self.assertIn("timeout", r.error)
        self.assertEqual(r.exit_code, -1)


# ── TestConfigFlags ───────────────────────────────────────────────────────────

class TestConfigFlags(unittest.TestCase):
    """Verify _config.py FIXED_FLAGS contains the required flags."""

    def setUp(self):
        from tier_a._config import FIXED_FLAGS
        self._flags = FIXED_FLAGS

    def test_pack_present(self):
        self.assertIn("--pack", self._flags,
                      "--pack missing from FIXED_FLAGS; engine would use wrong pack")

    def test_pack_path_contains_iq3_s(self):
        idx = self._flags.index("--pack")
        pack_val = self._flags[idx + 1]
        self.assertIn("iq3_s", pack_val,
                      f"--pack value '{pack_val}' does not reference iq3_s pack")

    def test_expert_cache_present(self):
        self.assertIn("--expert-cache", self._flags,
                      "--expert-cache missing from FIXED_FLAGS; defaults to 0 (no VRAM cache)")

    def test_expert_cache_value_2000(self):
        idx = self._flags.index("--expert-cache")
        self.assertEqual(self._flags[idx + 1], "2000",
                         f"expected --expert-cache 2000, got {self._flags[idx+1]}")

    def test_pcie_frac_present(self):
        self.assertIn("--pcie-frac", self._flags,
                      "--pcie-frac missing; adaptive PCIe scheduling varies between arms")

    def test_pcie_frac_value_0(self):
        idx = self._flags.index("--pcie-frac")
        self.assertEqual(self._flags[idx + 1], "0",
                         f"expected --pcie-frac 0, got {self._flags[idx+1]}")

    def test_stats_present(self):
        self.assertIn("--stats", self._flags,
                      "--stats missing; cache-hit rates absent from stderr artifacts")


# ── TestPrompts ───────────────────────────────────────────────────────────────

class TestPrompts(unittest.TestCase):

    def test_short_code_nonempty(self):
        self.assertGreater(len(SHORT_CODE), 50)

    def test_story_nonempty(self):
        self.assertGreater(len(STORY), 100)

    def test_long_doc_approx_exceeds_8192_chars_threshold(self):
        """Rough estimate (chars/3.5) must exceed 8192.  The real tokeniser
        will produce a higher count on technical prose; this guards against
        accidentally truncating the corpus."""
        count = prompt_token_count_approx(LONG_DOC_Q)
        self.assertGreater(count, 8192,
                           f"long_doc approx={count} tokens; need >8192.  "
                           "Expand prompts.py appendices.")

    def test_long_doc_contains_expected_keywords(self):
        for kw in ("Timsort", "Introsort", "Quicksort", "Heapsort", "Radix"):
            self.assertIn(kw, LONG_DOC_Q, f"missing keyword: {kw}")

    def test_all_prompts_have_chat_template(self):
        for name, txt in (("short", SHORT_CODE), ("story", STORY), ("long", LONG_DOC_Q)):
            self.assertIn("<|im_start|>", txt, f"{name} missing im_start")
            self.assertIn("<think>", txt, f"{name} missing think tag")


# ── TestCounterbalancing ──────────────────────────────────────────────────────

class TestCounterbalancing(unittest.TestCase):
    """Verify arm ordering is counterbalanced across reps, not BCBC fixed."""

    def setUp(self):
        from tier_a.run_matrix import build_matrix
        self._full = build_matrix(smoke=False)

    def _group_arms(self, group_prefix: str) -> list[dict]:
        return [a for a in self._full if a["label"].startswith(group_prefix)]

    def _order_for_rep(self, arms: list[dict], rep: int) -> list[str]:
        return [a["variant"] for a in arms if f"_rep{rep}_" in a["label"]]

    def test_g1_rep0_starts_baseline(self):
        arms = self._group_arms("g1_")
        order = self._order_for_rep(arms, 0)
        self.assertTrue(len(order) >= 2, f"too few arms: {order}")
        self.assertEqual(order[0], "baseline", f"rep0 should start baseline; got {order}")

    def test_g1_rep1_starts_candidate(self):
        """rep1 must be the opposite order from rep0 (counterbalanced)."""
        arms = self._group_arms("g1_")
        order0 = self._order_for_rep(arms, 0)
        order1 = self._order_for_rep(arms, 1)
        self.assertNotEqual(order0[0], order1[0],
                            f"rep0 and rep1 have same first arm '{order0[0]}'; "
                            f"not counterbalanced")

    def test_g1_both_variants_present_each_rep(self):
        arms = self._group_arms("g1_")
        for rep in range(3):
            order = self._order_for_rep(arms, rep)
            self.assertIn("baseline", order, f"rep{rep} missing baseline")
            self.assertIn("candidate", order, f"rep{rep} missing candidate")

    def test_g5_rotation_across_reps(self):
        """adapt-every variants should not always appear in the same order."""
        arms = [a for a in self._full if a["label"].startswith("g5_")]
        orders = []
        for rep in range(3):
            order = self._order_for_rep(arms, rep)
            orders.append(tuple(order))
        # At least two distinct orderings across reps
        self.assertGreater(len(set(orders)), 1,
                           f"g5 arms have identical ordering in all reps: {orders}")

    def test_g4_rotation_across_reps(self):
        arms = [a for a in self._full if a["label"].startswith("g4_")]
        orders = []
        for rep in range(3):
            order = self._order_for_rep(arms, rep)
            orders.append(tuple(order))
        self.assertGreater(len(set(orders)), 1,
                           f"g4 arms have identical ordering in all reps: {orders}")


# ── TestSummariseGroup ────────────────────────────────────────────────────────

class TestSummariseGroup(unittest.TestCase):
    """summarise_group must key on arm['variant'], not parse label strings."""

    def _rec(self, variant: str, tok_s: float, error=None) -> dict:
        return {"label": f"g1_rep0_{variant}", "variant": variant,
                "decode_tok_s": tok_s, "error": error, "n_output": 10}

    def test_groups_by_variant_not_label_suffix(self):
        from tier_a.run_matrix import summarise_group
        recs = [
            self._rec("baseline",  40.0),
            self._rec("candidate", 50.0),
            self._rec("baseline",  42.0),
            self._rec("candidate", 48.0),
        ]
        result = summarise_group(recs)
        self.assertIn("baseline", result)
        self.assertIn("candidate", result)
        # median for baseline = 41.0, candidate = 49.0
        self.assertIn("41.00", result)
        self.assertIn("49.00", result)

    def test_skips_error_records(self):
        from tier_a.run_matrix import summarise_group
        recs = [
            self._rec("baseline", 40.0),
            self._rec("baseline", 0.0, error="exit_code=1"),
        ]
        result = summarise_group(recs)
        # Only the clean record counts: median of [40.0] = 40.0; n must be 1
        self.assertIn("40.00", result)
        self.assertIn("n=1", result,
                      f"error record was included in the count; got: {result}")

    def test_no_eligible_results(self):
        from tier_a.run_matrix import summarise_group
        recs = [self._rec("baseline", 0.0, error="timeout")]
        self.assertIn("no eligible", summarise_group(recs))

    def test_performance_eligible_false_excluded(self):
        """Records with performance_eligible=False must not count in summaries."""
        from tier_a.run_matrix import summarise_group
        recs = [
            {"label": "g1_rep0_baseline", "variant": "baseline",
             "decode_tok_s": 40.0, "prefill_tok_s": 200.0,
             "error": None, "performance_eligible": True},
            {"label": "g1_rep0_baseline2", "variant": "baseline",
             "decode_tok_s": 35.0, "prefill_tok_s": 180.0,
             "error": None, "performance_eligible": False,
             "performance_ineligible_reason": "short_output: n_output=10 < 200"},
        ]
        result = summarise_group(recs)
        self.assertIn("n=1", result,
                      f"ineligible record included in count; got: {result}")

    def test_prefill_tok_s_reported_in_summary(self):
        """prefill_tok_s must appear in the summary string."""
        from tier_a.run_matrix import summarise_group
        recs = [
            {"label": "g2_rep0_baseline", "variant": "baseline",
             "decode_tok_s": 40.0, "prefill_tok_s": 250.0,
             "error": None, "performance_eligible": True},
            {"label": "g2_rep0_candidate", "variant": "candidate",
             "decode_tok_s": 44.0, "prefill_tok_s": 260.0,
             "error": None, "performance_eligible": True},
        ]
        result = summarise_group(recs)
        self.assertIn("prefill=", result,
                      f"prefill_tok_s not shown in summary; got: {result}")
        self.assertIn("250.00", result)
        self.assertIn("260.00", result)


# ── TestDecodeHeaderValidation ────────────────────────────────────────────────

class TestDecodeHeaderValidation(unittest.TestCase):
    """engine_cli must reject arms where the decode header count
    mismatches len(output_ids), or where the prompt echo is missing/wrong."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="tier_a_hdr_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_matching_header_count_ok(self):
        """decode header count == len(output_ids) -> no count-mismatch error."""
        exe = _write_fixture(self._tmp, _make_stdout(
            prompt_ids=(1, 2, 3), output_ids=(10, 20, 30, 40)))
        r = run_arm(exe, [], [1, 2, 3], 8, {}, (), 10.0, tmp_dir=self._tmp)
        if r.error:
            self.assertNotIn("decode_count_mismatch", r.error,
                             f"false decode mismatch raised: {r.error}")

    def test_mismatched_header_count_is_error(self):
        """decode header says 5 tokens but output line has 4 ids -> error."""
        stdout = textwrap.dedent("""\
            prompt  : 1 2 3
            output  : 10 20 30 40
            decode                   5 tokens in 120.0 ms  ->  33.33 tok/s
            prefill                  2 tokens in 85.0 ms  ->  100.00 tok/s  (time to first token 90.0 ms)
            speculation              5 rounds of 4, drafts accepted 3 of 8 (0.375), 1.60 tokens per round
        """)
        exe = _write_fixture(self._tmp, stdout)
        r = run_arm(exe, [], [1, 2, 3], 8, {}, (), 10.0, tmp_dir=self._tmp)
        self.assertIsNotNone(r.error)
        self.assertIn("decode_count_mismatch", r.error,
                      f"expected decode_count_mismatch, got: {r.error}")

    def test_prompt_echo_matches_supplied_ids(self):
        """prompt echo matching supplied ids -> no prompt_mismatch error."""
        exe = _write_fixture(self._tmp, _make_stdout(
            prompt_ids=(1, 2, 3), output_ids=(10, 20, 30, 40)))
        r = run_arm(exe, [], [1, 2, 3], 8, {}, (), 10.0, tmp_dir=self._tmp)
        if r.error:
            self.assertNotIn("prompt_mismatch", r.error)
            self.assertNotIn("missing_prompt_echo", r.error)

    def test_prompt_echo_mismatch_is_error(self):
        """Engine echoes different ids than we supplied -> prompt_mismatch."""
        exe = _write_fixture(self._tmp, _make_stdout(
            prompt_ids=(1, 2, 3), output_ids=(10, 20, 30, 40)))
        r = run_arm(exe, [], [10, 20, 30], 8, {}, (), 10.0, tmp_dir=self._tmp)
        self.assertIsNotNone(r.error)
        self.assertIn("prompt_mismatch", r.error,
                      f"expected prompt_mismatch, got: {r.error}")

    def test_missing_prompt_echo_is_error(self):
        """stdout with no 'prompt  :' line -> missing_prompt_echo."""
        stdout_no_prompt = textwrap.dedent("""\
            output  : 10 20 30 40
            decode                   4 tokens in 120.0 ms  ->  33.33 tok/s
            prefill                  2 tokens in 85.0 ms  ->  100.00 tok/s  (time to first token 90.0 ms)
            speculation              5 rounds of 4, drafts accepted 3 of 8 (0.375), 1.60 tokens per round
        """)
        exe = _write_fixture(self._tmp, stdout_no_prompt)
        r = run_arm(exe, [], [1, 2, 3], 8, {}, (), 10.0, tmp_dir=self._tmp)
        self.assertIsNotNone(r.error)
        self.assertIn("missing_prompt_echo", r.error,
                      f"expected missing_prompt_echo, got: {r.error}")

    def test_parse_stdout_stores_decode_header_n(self):
        """parse_stdout must expose decode_header_n as an integer."""
        p = parse_stdout(_make_stdout(output_ids=(10, 20, 30, 40)))
        self.assertIn("decode_header_n", p)
        self.assertEqual(p["decode_header_n"], 4)

    def test_parse_stdout_missing_decode_line_gives_minus_one(self):
        """No decode line -> decode_header_n == -1 (sentinel)."""
        p = parse_stdout("prompt  : 1 2\noutput  : 10 20\n")
        self.assertEqual(p["decode_header_n"], -1)


# ── TestPerformanceEligibility ────────────────────────────────────────────────

class TestPerformanceEligibility(unittest.TestCase):
    """Arms with full budget but < MIN_OUTPUT_FULL output must be flagged
    performance_eligible=False; errors also ineligible; both preserved in JSON."""

    def test_summarise_excludes_ineligible(self):
        from tier_a.run_matrix import summarise_group
        recs = [
            {"variant": "baseline", "decode_tok_s": 40.0, "prefill_tok_s": 200.0,
             "error": None, "performance_eligible": True},
            {"variant": "baseline", "decode_tok_s": 35.0, "prefill_tok_s": 180.0,
             "error": None, "performance_eligible": False,
             "performance_ineligible_reason": "short_output: n_output=50 < 200"},
            {"variant": "baseline", "decode_tok_s": 40.0, "prefill_tok_s": 200.0,
             "error": "exit_code=1", "performance_eligible": False,
             "performance_ineligible_reason": "error: exit_code=1"},
        ]
        result = summarise_group(recs)
        self.assertIn("n=1", result, f"ineligible records counted; got: {result}")

    def test_dry_run_records_have_eligibility_fields(self):
        """Every non-skipped dry-run record must have both eligibility fields."""
        from tier_a import run_matrix
        import io, contextlib, json
        tmp = tempfile.mkdtemp(prefix="tier_a_eligfield_")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_matrix.run(["--dry-run", "--smoke", "--out", tmp])
            payload = json.loads(
                (Path(tmp) / "tier_a_results.json").read_text())
            for rec in payload["results"]:
                if rec.get("skipped"):
                    continue
                self.assertIn("performance_eligible", rec,
                              f"missing performance_eligible in {rec['label']}")
                self.assertIn("performance_ineligible_reason", rec,
                              f"missing performance_ineligible_reason in {rec['label']}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── TestFailFast ──────────────────────────────────────────────────────────────

class TestFailFast(unittest.TestCase):
    """--fail-fast stops the matrix on the first real arm error."""

    def test_fail_fast_flag_accepted_dry_run(self):
        """--fail-fast must be accepted without error when no arms fail."""
        from tier_a import run_matrix
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run_matrix.run(["--dry-run", "--smoke", "--fail-fast"])
        self.assertEqual(code, 0, buf.getvalue()[-300:])

    def test_fail_fast_stops_on_first_error(self):
        """When run_arm returns an error result on the first arm, --fail-fast
        must return exit code 1 and not call run_arm a second time."""
        import io, contextlib, unittest.mock as mock
        from tier_a import run_matrix

        error_result = ArmResult(
            prompt_ids=[], output_ids=[], output_hash="err",
            n_output=0, decode_ms=0.0, decode_tok_s=0.0,
            prefill_ms=0.0, prefill_tok_s=0.0, ttft_ms=0.0,
            draft_accepted=0, draft_offered=0, draft_rate=0.0, tokens_per_round=0.0,
            stdout_raw="", stderr_raw="", wall_s=0.0,
            exit_code=1, error="exit_code=1", argv=["fake"],
        )
        call_count = {"n": 0}

        def _fake_run_arm(*args, **kwargs):
            call_count["n"] += 1
            return error_result

        buf = io.StringIO()
        with mock.patch("tier_a.run_matrix.run_arm", side_effect=_fake_run_arm):
            with contextlib.redirect_stdout(buf):
                code = run_matrix.run(["--dry-run", "--smoke", "--fail-fast"])

        self.assertEqual(code, 1, f"expected exit 1 on fail-fast, got {code}")
        self.assertIn("FAIL-FAST", buf.getvalue(), "fail-fast message not printed")
        self.assertEqual(call_count["n"], 1,
                         f"fail-fast did not stop; run_arm called {call_count['n']} times")

    def test_fail_fast_does_not_trigger_without_flag(self):
        """Without --fail-fast the matrix completes even when run_arm errors."""
        import io, contextlib, unittest.mock as mock
        from tier_a import run_matrix

        error_result = ArmResult(
            prompt_ids=[], output_ids=[], output_hash="err",
            n_output=0, decode_ms=0.0, decode_tok_s=0.0,
            prefill_ms=0.0, prefill_tok_s=0.0, ttft_ms=0.0,
            draft_accepted=0, draft_offered=0, draft_rate=0.0, tokens_per_round=0.0,
            stdout_raw="", stderr_raw="", wall_s=0.0,
            exit_code=1, error="exit_code=1", argv=["fake"],
        )
        call_count = {"n": 0}
        smoke_arm_count = len(
            [a for a in run_matrix.build_matrix(smoke=True)
             if a["variant"] != "not_applicable"])

        def _fake_run_arm(*args, **kwargs):
            call_count["n"] += 1
            return error_result

        buf = io.StringIO()
        with mock.patch("tier_a.run_matrix.run_arm", side_effect=_fake_run_arm):
            with contextlib.redirect_stdout(buf):
                code = run_matrix.run(["--dry-run", "--smoke"])

        # Without fail-fast all arms run
        self.assertEqual(call_count["n"], smoke_arm_count,
                         f"expected {smoke_arm_count} arms, got {call_count['n']}")

    def test_short_output_does_not_trigger_fail_fast(self):
        """Short-output (error=None) must NOT trigger fail-fast; only real
        errors (non-None error field) do."""
        import io, contextlib, unittest.mock as mock
        from tier_a import run_matrix

        # A result with n_output < MIN_OUTPUT_FULL but error=None
        short_result = ArmResult(
            prompt_ids=[1,2,3], output_ids=list(range(10, 60)),
            output_hash="abcd", n_output=50,
            decode_ms=100.0, decode_tok_s=40.0,
            prefill_ms=80.0, prefill_tok_s=200.0, ttft_ms=85.0,
            draft_accepted=3, draft_offered=8, draft_rate=0.375, tokens_per_round=1.6,
            stdout_raw="", stderr_raw="", wall_s=1.0,
            exit_code=0, error=None, argv=["fake"],
        )
        call_count = {"n": 0}
        smoke_arm_count = len(
            [a for a in run_matrix.build_matrix(smoke=True)
             if a["variant"] != "not_applicable"])

        def _fake_run_arm(*args, **kwargs):
            call_count["n"] += 1
            return short_result

        buf = io.StringIO()
        with mock.patch("tier_a.run_matrix.run_arm", side_effect=_fake_run_arm):
            with contextlib.redirect_stdout(buf):
                code = run_matrix.run(["--dry-run", "--smoke", "--fail-fast"])

        # All arms run; short-output is not a stop condition
        self.assertEqual(call_count["n"], smoke_arm_count,
                         f"fail-fast wrongly triggered on short-output after "
                         f"{call_count['n']}/{smoke_arm_count} arms")


# ── TestRunMatrixDryRun ───────────────────────────────────────────────────────

class TestRunMatrixDryRun(unittest.TestCase):

    def test_dry_run_exits_zero(self):
        from tier_a import run_matrix
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run_matrix.run(["--dry-run", "--smoke"])
        self.assertEqual(code, 0, buf.getvalue()[-500:])
        self.assertIn("DRY RUN", buf.getvalue())

    def test_dry_run_writes_json(self):
        from tier_a import run_matrix
        import io, contextlib
        tmp = tempfile.mkdtemp(prefix="tier_a_dryout_")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = run_matrix.run(["--dry-run", "--smoke", "--out", tmp])
            self.assertEqual(code, 0, buf.getvalue()[-500:])
            jp = Path(tmp) / "tier_a_results.json"
            self.assertTrue(jp.exists(), "tier_a_results.json not written")
            payload = json.loads(jp.read_text())
            self.assertIn("methodology", payload)
            self.assertIn("results", payload)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dry_run_per_arm_artifacts_written(self):
        """Each arm should produce its own artifact directory immediately."""
        from tier_a import run_matrix
        import io, contextlib
        tmp = tempfile.mkdtemp(prefix="tier_a_arttest_")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_matrix.run(["--dry-run", "--smoke", "--out", tmp])
            arms_dir = Path(tmp) / "arms"
            self.assertTrue(arms_dir.exists(), "arms/ directory not created")
            arm_dirs = list(arms_dir.iterdir())
            self.assertGreater(len(arm_dirs), 0, "no per-arm directories written")
            # Spot-check one arm directory
            for d in arm_dirs:
                if d.is_dir():
                    for fname in ("argv.txt", "record.json"):
                        self.assertTrue((d / fname).exists(),
                                        f"missing {fname} in {d}")
                    break
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dry_run_record_has_prompt_hash(self):
        from tier_a import run_matrix
        import io, contextlib
        tmp = tempfile.mkdtemp(prefix="tier_a_hash_")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_matrix.run(["--dry-run", "--smoke", "--out", tmp])
            jp = Path(tmp) / "tier_a_results.json"
            payload = json.loads(jp.read_text())
            for rec in payload["results"]:
                if not rec.get("skipped"):
                    self.assertIn("prompt_hash", rec,
                                  f"prompt_hash missing from record {rec['label']}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dry_run_methodology_caveat_conservative(self):
        """Caveat must NOT claim equal budgets imply equal graph-capture cost."""
        from tier_a.run_matrix import METHODOLOGY_CAVEAT
        self.assertNotIn("equal graph-capture cost", METHODOLOGY_CAVEAT.lower().replace(
            "do not guarantee equal graph-capture cost", ""))
        # Must mention cold-graph
        self.assertIn("cold-graph", METHODOLOGY_CAVEAT)
        # Must mention prompt hash
        self.assertIn("prompt_hash", METHODOLOGY_CAVEAT)

    def test_dry_run_records_effective_env(self):
        from tier_a import run_matrix
        import io, contextlib
        tmp = tempfile.mkdtemp(prefix="tier_a_env_")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_matrix.run(["--dry-run", "--smoke", "--out", tmp])
            jp = Path(tmp) / "tier_a_results.json"
            payload = json.loads(jp.read_text())
            for rec in payload["results"]:
                if not rec.get("skipped"):
                    self.assertIn("effective_env", rec)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── TestHelpInvocation ────────────────────────────────────────────────────────

class TestHelpInvocation(unittest.TestCase):

    def test_help_exits_zero(self):
        from tier_a import run_matrix
        with self.assertRaises(SystemExit) as cm:
            run_matrix.run(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_smoke_shorter_than_full(self):
        from tier_a.run_matrix import build_matrix
        self.assertLess(len(build_matrix(smoke=True)), len(build_matrix(smoke=False)))

    def test_all_groups_present(self):
        from tier_a.run_matrix import build_matrix
        labels = [a["label"] for a in build_matrix(smoke=False)]
        for g in ("g1_", "g2_", "g3_", "g4_", "g5_", "g6_", "g7_"):
            self.assertTrue(any(l.startswith(g) for l in labels), f"missing group {g}")

    def test_gr_down_max4_note_is_eligibility(self):
        from tier_a.run_matrix import build_matrix
        arms = [a for a in build_matrix(smoke=False) if "g6_" in a["label"]]
        self.assertGreaterEqual(len(arms), 2)
        for arm in arms:
            self.assertIn("likely_noop", arm["note"])

    def test_iq256_gather_is_not_applicable(self):
        from tier_a.run_matrix import build_matrix
        arms = [a for a in build_matrix(smoke=False) if "iq256" in a["label"]]
        self.assertEqual(len(arms), 1)
        self.assertIn("SKIPPED", arms[0]["note"])
        self.assertEqual(arms[0]["variant"], "not_applicable")


# ── TestTokenizePrompt (requires regex) ──────────────────────────────────────

@unittest.skipUnless(
    __import__("importlib.util", fromlist=[""]).find_spec("regex") is not None,
    "regex package not available; set PYTHONPATH to uv cache archive "
    "(see DEPENDENCY NOTE at top of file)",
)
class TestTokenizePrompt(unittest.TestCase):

    def _make_tokenizer(self):
        import strata_tokenizer as ST
        byte_to_uni = ST.bytes_to_unicode()
        toks: list[str] = [""] * 256
        for b, u in byte_to_uni.items():
            toks[b] = u
        specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
        ids_map = {t: i for i, t in enumerate(toks)}
        for sp in specials:
            ids_map[sp] = len(toks)
            toks.append(sp)
        types = [0] * len(toks)
        types[ids_map["<|im_start|>"]] = 3
        types[ids_map["<|im_end|>"]]   = 3
        types[ids_map["<think>"]]      = 4
        types[ids_map["</think>"]]     = 4
        return ST.Tokenizer(toks, [], types)

    def test_encode_nonempty(self):
        tok = self._make_tokenizer()
        ids = tok.encode("hello world", parse_special=False)
        self.assertGreater(len(ids), 0)

    def test_encode_with_special(self):
        tok = self._make_tokenizer()
        ids = tok.encode("<|im_start|>hi<|im_end|>", parse_special=True)
        vocab = {t: i for i, t in enumerate(tok.tokens)}
        self.assertIn(vocab["<|im_start|>"], ids)


if __name__ == "__main__":
    unittest.main()
