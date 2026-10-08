"""tools/tier_a/test_tier_a.py - offline unit tests for the Tier-A harness.

All tests run without a GPU and without model files.  Token ids are synthesised.
Engine subprocesses are replaced by tiny fixture binaries.

DEPENDENCY NOTE
───────────────
Tests that exercise strata_tokenizer (TestTokenizePrompt) require `regex`.
Set PYTHONPATH to an existing Python environment with `regex` installed if it
is not present system-wide (do not rely on an ephemeral package-cache path).

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
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from tier_a.engine_cli import (  # noqa: E402
    ArmResult, build_env, math_finite, parse_stdout, run_arm,
    _terminate_owned_pgroup, _TERM_GRACE_S,
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

    @unittest.skipUnless(os.name == "posix", "POSIX-only")
    def test_timeout_uses_owned_pgid_and_bounded_drain(self):
        """H6: an absent leader is not a reason to miss its surviving group."""
        import signal
        from unittest import mock

        proc = mock.Mock(pid=123456)
        first = subprocess.TimeoutExpired("dummy", 1, output=b"out", stderr=b"err")
        events = []

        def communicate(*, timeout):
            events.append(("communicate", timeout))
            if len(events) == 1:
                raise first
            return b"out tail\xff", b"err tail"

        proc.communicate.side_effect = communicate
        proc.wait.side_effect = lambda **kw: events.append(("wait", kw["timeout"]))
        proc.stdout.close.side_effect = lambda: events.append("stdout.close")
        proc.stderr.close.side_effect = lambda: events.append("stderr.close")
        with (mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc) as launch,
              mock.patch("tier_a.engine_cli.os.getpgid", side_effect=ProcessLookupError) as lookup,
              mock.patch("tier_a.engine_cli.os.killpg",
                         side_effect=lambda pgid, sig: events.append((pgid, sig))) as killpg,
              mock.patch("tier_a.engine_cli.time.sleep") as sleep,
              mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0.1)):
            r = run_arm("/not-launched", [], [1], 1, {}, (), 1, tmp_dir=self._tmp)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        lookup.assert_not_called()
        killpg.assert_called_once_with(proc.pid, signal.SIGKILL)
        proc.kill.assert_not_called()
        proc.poll.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(events, [("communicate", 1), (proc.pid, signal.SIGKILL),
                                  ("communicate", 0.1), ("wait", 0.1),
                                  "stdout.close", "stderr.close"])
        self.assertEqual(r.stdout_raw, "out tail\ufffd")
        self.assertEqual(r.stderr_raw, "err tail")
        self.assertEqual(r.error, "timeout after 1s")
        self.assertEqual(r.exit_code, -1)
        self.assertFalse(Path(r.argv[r.argv.index("--tokens-file") + 1]).exists())

    @unittest.skipUnless(os.name == "posix", "POSIX-only")
    def test_timeout_persistent_pipes_keep_latest_partial_output(self):
        """A second timeout is still a timeout, not an unbounded drain/reap."""
        from unittest import mock

        for out, err in ((b"out tail", b"err tail"), (None, b"err tail"),
                         (b"out tail", None), (None, None)):
            with self.subTest(out=out, err=err):
                proc = mock.Mock(pid=123456)
                proc.communicate.side_effect = [
                    subprocess.TimeoutExpired("dummy", 1, output=b"out", stderr=b"err"),
                    subprocess.TimeoutExpired("dummy", 0.1, output=out, stderr=err),
                ]
                proc.wait.side_effect = subprocess.TimeoutExpired("dummy", 0.1)
                with (mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc),
                      mock.patch("tier_a.engine_cli.os.killpg"),
                      mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0.1)):
                    r = run_arm("/not-launched", [], [1], 1, {}, (), 1, tmp_dir=self._tmp)
                self.assertEqual(proc.communicate.call_args_list,
                                 [mock.call(timeout=1), mock.call(timeout=0.1)])
                proc.wait.assert_called_once_with(timeout=0.1)
                proc.stdout.close.assert_called_once()
                proc.stderr.close.assert_called_once()
                self.assertEqual(r.stdout_raw, (out if out is not None else b"out").decode())
                self.assertEqual(r.stderr_raw, (err if err is not None else b"err").decode())
                self.assertEqual(r.error, "timeout after 1s")
                self.assertEqual(r.exit_code, -1)
                self.assertEqual(list(Path(self._tmp).glob("*.ids")), [])

    @unittest.skipUnless(os.name == "posix", "POSIX-only")
    def test_timeout_drain_exceptions_reap_close_and_preserve_original(self):
        """Adapted timeout_check.py: second communicate cannot bypass cleanup."""
        import signal
        from unittest import mock

        for original in (KeyboardInterrupt("second communicate"), SystemExit(7),
                         RuntimeError("drain failed")):
            with self.subTest(exception=type(original).__name__):
                proc = mock.Mock(pid=123456)
                proc.communicate.side_effect = [subprocess.TimeoutExpired("dummy", 1), original]
                # Secondary cleanup failures must not mask the drain exception
                # or keep the other pipe / tokens-file open.
                proc.wait.side_effect = OSError("secondary reap error")
                proc.stdout.close.side_effect = KeyboardInterrupt("secondary close error")
                with (mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc),
                      mock.patch("tier_a.engine_cli.os.getpgid", side_effect=ProcessLookupError) as lookup,
                      mock.patch("tier_a.engine_cli.os.killpg") as killpg,
                      mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0.1)):
                    with self.assertRaises(type(original)) as cm:
                        run_arm("/not-launched", [], [1], 1, {}, (), 1, tmp_dir=self._tmp)
                self.assertIs(cm.exception, original)
                lookup.assert_not_called()
                killpg.assert_called_once_with(proc.pid, signal.SIGKILL)
                self.assertEqual(proc.communicate.call_args_list,
                                 [mock.call(timeout=1), mock.call(timeout=0.1)])
                self.assertEqual(proc.wait.call_count, 2)
                for call in proc.wait.call_args_list:
                    self.assertGreaterEqual(call.kwargs["timeout"], 0)
                    self.assertLessEqual(call.kwargs["timeout"], 0.1)
                proc.stdout.close.assert_called_once()
                proc.stderr.close.assert_called_once()
                self.assertEqual(list(Path(self._tmp).glob("*.ids")), [])

    @unittest.skipUnless(os.name == "posix", "POSIX-only")
    def test_timeout_kill_interruption_retries_before_reap(self):
        import signal
        from unittest import mock

        original = KeyboardInterrupt("KILL interrupted")
        proc = mock.Mock(pid=123456)
        proc.communicate.side_effect = subprocess.TimeoutExpired("dummy", 1)
        events = []

        def kill(pgid, sig):
            events.append((pgid, sig))
            if len(events) == 1:
                raise original

        proc.wait.side_effect = lambda **kw: events.append("wait")
        with (mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc),
              mock.patch("tier_a.engine_cli.os.killpg", side_effect=kill),
              mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0.1)):
            with self.assertRaises(KeyboardInterrupt) as cm:
                run_arm("/not-launched", [], [1], 1, {}, (), 1, tmp_dir=self._tmp)
        self.assertIs(cm.exception, original)
        self.assertEqual(events, [(proc.pid, signal.SIGKILL),
                                  (proc.pid, signal.SIGKILL), "wait"])
        proc.communicate.assert_called_once_with(timeout=1)
        proc.wait.assert_called_once_with(timeout=0.1)
        proc.stdout.close.assert_called_once()
        proc.stderr.close.assert_called_once()
        self.assertEqual(list(Path(self._tmp).glob("*.ids")), [])

    @unittest.skipUnless(os.name == "posix", "POSIX-only")
    def test_timeout_reap_interruption_retries_and_preserves_original(self):
        from unittest import mock

        original = KeyboardInterrupt("wait interrupted")
        proc = mock.Mock(pid=123456)
        proc.communicate.side_effect = [subprocess.TimeoutExpired("dummy", 1), (b"out", b"err")]
        proc.wait.side_effect = [original, 0]
        with (mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc),
              mock.patch("tier_a.engine_cli.os.killpg"),
              mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0.1)):
            with self.assertRaises(KeyboardInterrupt) as cm:
                run_arm("/not-launched", [], [1], 1, {}, (), 1, tmp_dir=self._tmp)
        self.assertIs(cm.exception, original)
        self.assertEqual(proc.wait.call_count, 2)
        for call in proc.wait.call_args_list:
            self.assertGreaterEqual(call.kwargs["timeout"], 0)
            self.assertLessEqual(call.kwargs["timeout"], 0.1)
        proc.stdout.close.assert_called_once()
        proc.stderr.close.assert_called_once()
        self.assertEqual(list(Path(self._tmp).glob("*.ids")), [])

    @unittest.skipUnless(os.name == "posix", "POSIX-only")
    def test_timeout_absent_group_never_retries_stale_pgid(self):
        import signal
        from unittest import mock

        proc = mock.Mock(pid=123456)
        proc.communicate.side_effect = [subprocess.TimeoutExpired("dummy", 1), (b"out", b"err")]
        with (mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc),
              mock.patch("tier_a.engine_cli.os.killpg", side_effect=ProcessLookupError) as killpg,
              mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0.1)):
            r = run_arm("/not-launched", [], [1], 1, {}, (), 1, tmp_dir=self._tmp)
        killpg.assert_called_once_with(proc.pid, signal.SIGKILL)
        proc.kill.assert_not_called()
        proc.wait.assert_called_once_with(timeout=0.1)
        proc.stdout.close.assert_called_once()
        proc.stderr.close.assert_called_once()
        self.assertEqual(r.error, "timeout after 1s")
        self.assertEqual(list(Path(self._tmp).glob("*.ids")), [])

    def test_timeout_nonposix_direct_child_fallback_is_bounded(self):
        from unittest import mock
        from tier_a import engine_cli

        proc = mock.Mock(pid=123456)
        proc.communicate.return_value = (b"out tail", b"err tail")
        with (mock.patch.object(engine_cli, "os") as mock_os,
              mock.patch.object(engine_cli, "_TERM_GRACE_S", 0.1)):
            del mock_os.killpg
            captured = _terminate_owned_pgroup(
                proc, timeout_exc=subprocess.TimeoutExpired("dummy", 1,
                                                           output=b"out", stderr=b"err"))
        self.assertEqual(captured, (b"out tail", b"err tail"))
        proc.kill.assert_called_once()
        proc.terminate.assert_not_called()
        proc.communicate.assert_called_once_with(timeout=0.1)
        proc.wait.assert_called_once_with(timeout=0.1)
        proc.stdout.close.assert_called_once()
        proc.stderr.close.assert_called_once()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux fork/process-state fixture")
    def test_real_timeout_exited_leader_inherited_pipes(self):
        self._real_timeout_fixture()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux fork/process-state fixture")
    def test_real_timeout_escaped_pipe_holder_is_bounded_not_signalled(self):
        self._real_timeout_fixture(escape=True)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux fork/process-state fixture")
    def test_real_timeout_drain_interruption_closes_retained_pipes(self):
        self._real_timeout_fixture(interrupt=True)

    def _real_timeout_fixture(self, *, escape=False, interrupt=False):
        """Real dummy child forks a pipe-inheriting grandchild then exits.

        Reap the leader before the first real communicate to force getpgid's
        ESRCH case.  An optional escaped fixture keeps pipes open past group
        KILL; the runner must bound its drain without signalling that session.
        """
        import signal
        from unittest import mock

        ready = Path(self._tmp) / "grandchild.ready"
        script = Path(self._tmp) / "timeout_fixture.py"
        script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(f"""\
            import os, signal, time
            from pathlib import Path
            pid = os.fork()
            if pid == 0:
                if {escape!r}:
                    os.setsid()
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                os.write(1, b'grandchild stdout\\n')
                os.write(2, b'grandchild stderr\\n')
                ready = Path({str(ready)!r})
                pending = ready.with_suffix('.tmp')
                pending.write_text(str(os.getpid()))
                pending.rename(ready)
                time.sleep(60)
                os._exit(0)
            os.write(1, b'leader stdout\\n')
            os.write(2, b'leader stderr\\n')
            os._exit(0)
        """), encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        real_communicate = subprocess.Popen.communicate
        real_killpg = os.killpg
        children = []
        calls = []
        gc_pid = None
        original = KeyboardInterrupt("real timeout drain interrupted")
        retained = None
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                     start_new_session=True)

        def communicate(proc, *, timeout):
            calls.append(timeout)
            if len(calls) == 1:
                children.append(proc)
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists(), "grandchild fixture never became ready")
                self.assertEqual(proc.wait(timeout=2), 0)
                grandchild = int(ready.read_text())
                self.assertEqual(os.getpgid(grandchild), grandchild if escape else proc.pid)
                with self.assertRaises(ProcessLookupError):
                    os.getpgid(proc.pid)
            elif interrupt:
                raise original
            return real_communicate(proc, timeout=timeout)

        def fixture_running(pid):
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                return False
            return fields[0] != "Z"

        t0 = time.monotonic()
        try:
            with (mock.patch.object(subprocess.Popen, "communicate", communicate),
                  mock.patch("tier_a.engine_cli.os.killpg", wraps=real_killpg) as killpg,
                  mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0.15)):
                if interrupt:
                    try:
                        run_arm(str(script), [], [1], 1, {}, (), 0.15, tmp_dir=self._tmp)
                    except KeyboardInterrupt as exc:
                        retained = exc
                    self.assertIs(retained, original)
                    self.assertIsNotNone(retained.__traceback__)
                else:
                    r = run_arm(str(script), [], [1], 1, {}, (), 0.15, tmp_dir=self._tmp)
                    self.assertEqual(r.error, "timeout after 0s")
                    self.assertEqual(r.exit_code, -1)
                    for name in ("leader", "grandchild"):
                        self.assertIn(f"{name} stdout\n", r.stdout_raw)
                        self.assertIn(f"{name} stderr\n", r.stderr_raw)
                proc = children[0]
                killpg.assert_called_once_with(proc.pid, signal.SIGKILL)
            self.assertLess(time.monotonic() - t0, 3, "timeout cleanup was not bounded")
            self.assertEqual(calls, [0.15, 0.15])
            self.assertEqual(proc.returncode, 0, "exited leader not reaped")
            self.assertTrue(proc.stdout.closed, "retained stdout is open")
            self.assertTrue(proc.stderr.closed, "retained stderr is open")
            self.assertEqual(list(Path(self._tmp).glob("*.ids")), [])
            self.assertIsNone(unrelated.poll(), "runner killed an unrelated session")
            gc_pid = int(ready.read_text())
            if escape:
                self.assertTrue(fixture_running(gc_pid), "runner signalled an escaped session")
                self.assertEqual(os.getpgid(gc_pid), gc_pid)
            else:
                deadline = time.monotonic() + 2
                while fixture_running(gc_pid) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(fixture_running(gc_pid), "owned grandchild survived KILL")
        finally:
            # Safety cleanup targets only known test fixtures, never arbitrary
            # groups.  Do not reuse a reaped/closed leader's PGID on success.
            for proc in children:
                if not proc.stdout.closed or proc.returncode is None:
                    _terminate_owned_pgroup(proc, timeout_exc=subprocess.TimeoutExpired("fixture", 0))
            if gc_pid is None and ready.exists():
                gc_pid = int(ready.read_text())
            if escape and gc_pid is not None and fixture_running(gc_pid):
                if os.getpgid(gc_pid) == gc_pid:
                    os.kill(gc_pid, signal.SIGKILL)
            unrelated.kill()
            unrelated.wait(timeout=2)


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


# ── TestPrefillValidation ────────────────────────────────────────────────────

class TestPrefillValidation(unittest.TestCase):
    """sta-0a7: multi-token arms require a valid primary prefill diagnostic.

    generate.cpp reports only conditioning positions: len(prompt)-1 tokens.
    prefill_ms, prefill_tok_s and ttft_ms must be positive and finite.
    Missing / malformed / nonfinite / zero / negative / count-mismatch prefill
    sets ArmResult.error (so the arm cannot be performance eligible).
    """

    PROMPT = [1, 2, 3]   # len 3 -> prefill must report 2 tokens

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="tier_a_prefill_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _stdout_with_prefill(self, prefill_line: str) -> str:
        return textwrap.dedent(f"""\
            prompt  : 1 2 3
            output  : 10 20 30 40
            decode                   4 tokens in 120.0 ms  ->  33.33 tok/s
            {prefill_line}
            speculation              5 rounds of 4, drafts accepted 3 of 8 (0.375), 1.60 tokens per round
        """)

    def _run(self, stdout: str, token_ids=None) -> ArmResult:
        token_ids = self.PROMPT if token_ids is None else token_ids
        exe = _write_fixture(self._tmp, stdout)
        return run_arm(exe, [], token_ids, 4, {}, (), 10.0, tmp_dir=self._tmp)

    _VALID_LINE = ("prefill                  2 tokens in 85.0 ms  ->  100.00 tok/s"
                   "  (time to first token 90.0 ms)")

    def test_valid_prefill_multi_token_ok(self):
        r = self._run(self._stdout_with_prefill(self._VALID_LINE))
        self.assertIsNone(r.error, f"valid prefill rejected: {r.error}")
        self.assertAlmostEqual(r.prefill_ms, 85.0)
        self.assertAlmostEqual(r.prefill_tok_s, 100.0)
        self.assertAlmostEqual(r.ttft_ms, 90.0)

    def test_missing_prefill_multi_token_is_error(self):
        stdout = textwrap.dedent("""\
            prompt  : 1 2 3
            output  : 10 20 30 40
            decode                   4 tokens in 120.0 ms  ->  33.33 tok/s
            speculation              5 rounds of 4, drafts accepted 3 of 8 (0.375), 1.60 tokens per round
        """)
        r = self._run(stdout)
        self.assertIsNotNone(r.error)
        self.assertIn("prefill", r.error)

    def test_count_mismatch_prefill_is_error(self):
        line = self._VALID_LINE.replace("2 tokens", "5 tokens")
        r = self._run(self._stdout_with_prefill(line))
        self.assertIsNotNone(r.error)
        self.assertIn("prefill_count_mismatch", r.error)
        self.assertIn("header=5", r.error)
        self.assertIn("expected=2", r.error)

    def test_zero_prefill_ms_is_error(self):
        line = self._VALID_LINE.replace("85.0 ms", "0.0 ms")
        r = self._run(self._stdout_with_prefill(line))
        self.assertIsNotNone(r.error)
        self.assertIn("prefill_ms", r.error)

    def test_zero_prefill_tok_s_is_error(self):
        line = self._VALID_LINE.replace("100.00 tok/s", "0.00 tok/s")
        r = self._run(self._stdout_with_prefill(line))
        self.assertIsNotNone(r.error)
        self.assertIn("prefill_tok_s", r.error)

    def test_negative_prefill_ms_is_error(self):
        line = self._VALID_LINE.replace("85.0 ms", "-85.0 ms")
        r = self._run(self._stdout_with_prefill(line))
        self.assertIsNotNone(r.error)
        self.assertIn("prefill_ms", r.error)

    def test_nan_prefill_ms_is_error(self):
        line = self._VALID_LINE.replace("85.0 ms", "nan ms")
        r = self._run(self._stdout_with_prefill(line))
        self.assertIsNotNone(r.error)
        self.assertIn("prefill_ms", r.error)

    def test_inf_prefill_tok_s_is_error(self):
        line = self._VALID_LINE.replace("100.00 tok/s", "inf tok/s")
        r = self._run(self._stdout_with_prefill(line))
        self.assertIsNotNone(r.error)
        self.assertIn("prefill_tok_s", r.error)

    def test_malformed_prefill_header_is_error(self):
        line = self._VALID_LINE.replace("2 tokens", "abc tokens")
        r = self._run(self._stdout_with_prefill(line))
        self.assertIsNotNone(r.error)
        self.assertIn("prefill", r.error)

    def test_malformed_ttft_is_error(self):
        line = self._VALID_LINE.replace("90.0 ms", "?? ms")
        r = self._run(self._stdout_with_prefill(line))
        self.assertIsNotNone(r.error)
        self.assertIn("prefill", r.error)

    def test_malformed_prefill_does_not_escape_parser(self):
        """Malformed numeric text must not raise from parse_stdout alone."""
        for bad in ("??", "1.2.3", "0x10", "1e+"):
            with self.subTest(bad=bad):
                line = self._VALID_LINE.replace("85.0 ms", f"{bad} ms")
                stdout = self._stdout_with_prefill(line)
                p = parse_stdout(stdout)
                # Parser defaults apply; error is recorded, not raised.
                self.assertEqual(p["prefill_ms"], 0.0)
                self.assertIsNotNone(p["prefill_error"])

    def test_single_token_no_prefill_ok(self):
        """One-token prompts need no prefill diagnostic; unchanged behavior."""
        stdout = textwrap.dedent("""\
            prompt  : 7
            output  : 10 20 30 40
            decode                   4 tokens in 120.0 ms  ->  33.33 tok/s
            speculation              5 rounds of 4, drafts accepted 3 of 8 (0.375), 1.60 tokens per round
        """)
        r = self._run(stdout, token_ids=[7])
        self.assertIsNone(r.error, f"single-token run rejected: {r.error}")
        self.assertEqual(r.prefill_ms, 0.0)
        self.assertEqual(r.ttft_ms, 0.0)

    def test_error_precedence_preserved(self):
        """Existing decode/output errors win over prefill validation."""
        stdout = textwrap.dedent("""\
            prompt  : 1 2 3
            output  :
            decode                   0 tokens in 0.0 ms  ->  0.00 tok/s
        """)
        r = self._run(stdout)
        self.assertIsNotNone(r.error)
        self.assertIn("zero_output_tokens", r.error)
        self.assertNotIn("prefill", r.error)

    def test_prefill_error_not_performance_eligible(self):
        """A prefill-error ArmResult is flagged ineligible by run_matrix."""
        import io, contextlib, json as _json, unittest.mock as mock
        from tier_a import run_matrix

        err_result = ArmResult(
            prompt_ids=[1, 2, 3], output_ids=list(range(10, 30)),
            output_hash="e" * 16, n_output=20,
            decode_ms=100.0, decode_tok_s=40.0,
            prefill_ms=0.0, prefill_tok_s=0.0, ttft_ms=0.0,
            draft_accepted=0, draft_offered=0, draft_rate=0.0, tokens_per_round=0.0,
            stdout_raw="", stderr_raw="", wall_s=0.0, exit_code=0,
            error="prefill_count_mismatch: header=5 expected=2", argv=["fake"],
        )
        tmp = tempfile.mkdtemp(prefix="tier_a_prefill_elig_")
        try:
            with mock.patch("tier_a.run_matrix.run_arm", return_value=err_result):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    run_matrix.run(["--dry-run", "--smoke", "--out", tmp])
            payload = _json.loads((Path(tmp) / "tier_a_results.json").read_text())
            non_skipped = [rec for rec in payload["results"] if not rec.get("skipped")]
            self.assertTrue(non_skipped, "no non-skipped records written")
            for rec in non_skipped:
                self.assertFalse(rec["performance_eligible"],
                                 f"{rec['label']} eligible despite prefill error")
                self.assertIn("prefill_count_mismatch",
                              rec["performance_ineligible_reason"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


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
    "regex package not available; set PYTHONPATH to an environment with regex "
    "installed (see DEPENDENCY NOTE at top of file)",
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


# ── TestCancellationCleanup ──────────────────────────────────────────────────────────────────────

class TestCancellationCleanup(unittest.TestCase):
    """Regression tests for sta-d76: exceptional/cancellation exits of run_arm
    must not leave owned child / process-group processes running."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="tier_a_cancel_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Offline mock tests (no real child processes)                        #
    # ------------------------------------------------------------------ #

    def test_keyboard_interrupt_reraises(self):
        """KeyboardInterrupt must propagate out of run_arm unchanged."""
        import unittest.mock as mock

        proc = mock.Mock()
        proc.communicate.side_effect = KeyboardInterrupt
        with mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc):
            # _terminate_owned_pgroup is called; mock it out so we don't
            # need a real process group to kill.
            with mock.patch("tier_a.engine_cli._terminate_owned_pgroup"):
                with self.assertRaises(KeyboardInterrupt):
                    run_arm(
                        "/bin/true", [], [1, 2, 3], 8,
                        {}, (), 10.0, tmp_dir=self._tmp,
                    )

    def test_system_exit_reraises(self):
        """SystemExit must propagate out of run_arm unchanged."""
        import unittest.mock as mock

        proc = mock.Mock()
        proc.communicate.side_effect = SystemExit(2)
        with mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc):
            with mock.patch("tier_a.engine_cli._terminate_owned_pgroup"):
                with self.assertRaises(SystemExit) as cm:
                    run_arm(
                        "/bin/true", [], [1, 2, 3], 8,
                        {}, (), 10.0, tmp_dir=self._tmp,
                    )
                self.assertEqual(cm.exception.code, 2)

    def test_arbitrary_exception_reraises(self):
        """An arbitrary RuntimeError from communicate must propagate."""
        import unittest.mock as mock

        boom = RuntimeError("injected failure")
        proc = mock.Mock()
        proc.communicate.side_effect = boom
        with mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc):
            with mock.patch("tier_a.engine_cli._terminate_owned_pgroup"):
                with self.assertRaises(RuntimeError) as cm:
                    run_arm(
                        "/bin/true", [], [1, 2, 3], 8,
                        {}, (), 10.0, tmp_dir=self._tmp,
                    )
                self.assertIs(cm.exception, boom)

    def test_cancellation_calls_terminate_owned_pgroup(self):
        """_terminate_owned_pgroup must be called exactly once on KI."""
        import unittest.mock as mock

        calls = []
        def _fake_terminate(proc):
            calls.append(proc)

        proc = mock.Mock()
        proc.communicate.side_effect = KeyboardInterrupt
        with mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc):
            with mock.patch("tier_a.engine_cli._terminate_owned_pgroup",
                            side_effect=_fake_terminate):
                with self.assertRaises(KeyboardInterrupt):
                    run_arm(
                        "/bin/true", [], [1, 2, 3], 8,
                        {}, (), 10.0, tmp_dir=self._tmp,
                    )
        self.assertEqual(len(calls), 1, "_terminate_owned_pgroup not called exactly once")

    def test_cancellation_tokenfile_cleaned_up(self):
        """Even on KeyboardInterrupt the --tokens-file must be deleted.

        A mocked child raises KeyboardInterrupt from communicate(); the
        finally-block in run_arm must still unlink the .ids temp file.
        """
        import unittest.mock as mock

        proc = mock.Mock()
        proc.communicate.side_effect = KeyboardInterrupt
        with mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc):
            with mock.patch("tier_a.engine_cli._terminate_owned_pgroup"):
                with self.assertRaises(KeyboardInterrupt):
                    run_arm(
                        "/bin/true", [], [1, 2, 3], 8,
                        {}, (), 10.0, tmp_dir=self._tmp,
                    )

        # No .ids temp file should remain in self._tmp
        ids_files = list(Path(self._tmp).glob("*.ids"))
        self.assertEqual(ids_files, [],
                         f"tokens-file(s) not cleaned up after KI: {ids_files}")

    # ------------------------------------------------------------------ #
    # _terminate_owned_pgroup unit tests (real process, no GPU)           #
    # ------------------------------------------------------------------ #

    def test_terminate_owned_pgroup_already_exited(self):
        """Calling _terminate_owned_pgroup on an already-exited process must
        not raise."""
        proc = subprocess.Popen(
            ["/bin/true"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        proc.wait()  # let it exit naturally
        # Should not raise even though process is gone
        try:
            _terminate_owned_pgroup(proc)
        except Exception as e:  # noqa: BLE001
            self.fail(f"_terminate_owned_pgroup raised on dead proc: {e}")

    def test_terminate_owned_pgroup_kills_sleeping_process(self):
        """A sleeping child must be terminated (not survive) within the grace
        period."""
        proc = subprocess.Popen(
            ["/bin/sleep", "60"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            self.assertIsNone(proc.poll(), "process exited before test start")
            _terminate_owned_pgroup(proc)
            rc = proc.poll()
            self.assertIsNotNone(rc, "process still running after _terminate_owned_pgroup")
        finally:
            # Safety: kill if somehow still alive
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    # ------------------------------------------------------------------ #
    # Real child+grandchild orphan test (no GPU)                          #
    # ------------------------------------------------------------------ #

    @unittest.skipUnless(sys.platform != "win32", "POSIX-only process group test")
    def test_no_orphan_grandchild_after_cancellation(self):
        """Grandchild processes in the engine's session must not survive after
        cancellation.

        Scenario:
          - Fixture shell script forks a background grandchild (sleep 60)
            and writes the grandchild's PID to stdout.
          - run_arm's communicate() is interrupted by a mock KeyboardInterrupt
            after we can confirm the grandchild exists.
          - After the exception propagates, the grandchild must be dead
            (not just the direct child).

        Because the grandchild is in the same session/pgroup (it inherits
        start_new_session), killpg(SIGTERM/SIGKILL) covers it.
        """
        import unittest.mock as mock

        # Script: fork a grandchild (sleep 60), print its PID, then hang
        grandchild_pid_file = Path(self._tmp) / "gc_pid.txt"
        script = Path(self._tmp) / "parent_with_grandchild.sh"
        script.write_text(
            f"#!/bin/sh\n"
            f"sleep 60 &\n"
            f"echo $! > {grandchild_pid_file}\n"
            f"sleep 60\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        # Launch via Popen directly (not run_arm) so we control when KI fires
        proc = subprocess.Popen(
            [str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )

        # Wait for grandchild to start
        deadline = time.monotonic() + 5.0
        gc_pid = None
        while time.monotonic() < deadline:
            if grandchild_pid_file.exists():
                try:
                    gc_pid = int(grandchild_pid_file.read_text().strip())
                    break
                except ValueError:
                    pass
            time.sleep(0.05)

        self.assertIsNotNone(gc_pid,
                             "grandchild PID file not written in time")

        # Confirm grandchild is alive
        try:
            os.kill(gc_pid, 0)  # signal 0 = existence check
        except ProcessLookupError:
            self.skipTest(f"grandchild {gc_pid} already gone before test logic ran")

        # Simulate cancellation: terminate the owned pgroup
        _terminate_owned_pgroup(proc)

        # Grandchild must now be dead
        import time as _time
        _time.sleep(0.1)  # brief settle after reap
        try:
            os.kill(gc_pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = False  # process is a zombie or not ours; treat as gone

        self.assertFalse(alive,
                         f"grandchild {gc_pid} survived _terminate_owned_pgroup")

    @unittest.skipUnless(sys.platform != "win32", "POSIX-only process group test")
    def test_run_arm_ki_no_orphan_grandchild(self):
        """run_arm: KI raised during communicate() must leave no orphan
        grandchild in the engine's process group.

        Uses a fixture that forks a grandchild before hanging, combines it
        with a mock KeyboardInterrupt injected into communicate().
        """
        import unittest.mock as mock

        grandchild_pid_file = Path(self._tmp) / "gc2_pid.txt"
        script = Path(self._tmp) / "engine_with_grandchild.sh"
        script.write_text(
            f"#!/bin/sh\n"
            f"sleep 60 &\n"
            f"echo $! > {grandchild_pid_file}\n"
            f"sleep 60\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        # We need the real Popen to run the script but intercept communicate().
        # Use a wrapper that waits for the grandchild to appear, then raises KI.
        def _delayed_ki(self_proc, *args, **kwargs):
            # Wait until grandchild is up
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if grandchild_pid_file.exists():
                    try:
                        int(grandchild_pid_file.read_text().strip())
                        break
                    except ValueError:
                        pass
                time.sleep(0.05)
            raise KeyboardInterrupt

        with mock.patch.object(subprocess.Popen, "communicate", _delayed_ki):
            with self.assertRaises(KeyboardInterrupt):
                run_arm(
                    str(script), [], [1, 2, 3], 8,
                    {}, (), 30.0, tmp_dir=self._tmp,
                )

        # Read grandchild PID
        self.assertTrue(grandchild_pid_file.exists(),
                        "grandchild PID file not written")
        gc_pid = int(grandchild_pid_file.read_text().strip())

        import time as _time
        _time.sleep(0.1)  # brief settle
        try:
            os.kill(gc_pid, 0)
            alive = True
        except (ProcessLookupError, PermissionError):
            alive = False

        self.assertFalse(alive,
                         f"orphan grandchild {gc_pid} survived run_arm KI")

    # ------------------------------------------------------------------ #
    # Regression: success/timeout paths unaffected                        #
    # ------------------------------------------------------------------ #

    def test_successful_run_unaffected(self):
        """Normal successful run still completes and returns clean ArmResult."""
        exe = _write_fixture(self._tmp, _make_stdout(output_ids=(10, 20, 30, 40)))
        r = run_arm(exe, [], [1, 2, 3], 8, {}, (), 10.0, tmp_dir=self._tmp)
        self.assertIsNone(r.error)
        self.assertEqual(r.n_output, 4)

    def test_timeout_path_unaffected(self):
        """Timeout path still returns sentinel with 'timeout' in error."""
        script = Path(self._tmp) / "hang2.sh"
        script.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
        r = run_arm(str(script), [], [1, 2], 8, {}, (), 0.3, tmp_dir=self._tmp)
        self.assertIsNotNone(r.error)
        self.assertIn("timeout", r.error)
        self.assertEqual(r.exit_code, -1)


# ── TestTerminateOwnedPgroupUnit ────────────────────────────────────────────────────

class TestTerminateOwnedPgroupUnit(unittest.TestCase):
    """Unit tests for _terminate_owned_pgroup in isolation."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="tier_a_tpg_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_grace_constant_positive(self):
        """_TERM_GRACE_S must be a positive number."""
        self.assertIsInstance(_TERM_GRACE_S, (int, float))
        self.assertGreater(_TERM_GRACE_S, 0)

    def test_term_grace_does_not_exceed_timeout_budget(self):
        """_TERM_GRACE_S should stay small enough not to dominate test run
        times; budget is <= 10 seconds."""
        self.assertLessEqual(_TERM_GRACE_S, 10.0,
                             f"_TERM_GRACE_S={_TERM_GRACE_S} too large")

    @unittest.skipUnless(sys.platform != "win32", "POSIX-only")
    def test_does_not_kill_unrelated_process(self):
        """_terminate_owned_pgroup must ONLY kill the owned process group,
        not an unrelated process.  We start two independent processes;
        only the first has _terminate_owned_pgroup called on it."""
        # An unrelated background sleep
        unrelated = subprocess.Popen(
            ["/bin/sleep", "30"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        # Target process
        target = subprocess.Popen(
            ["/bin/sleep", "30"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            # Their pgids must differ (both use start_new_session=True)
            self.assertNotEqual(
                os.getpgid(target.pid), os.getpgid(unrelated.pid),
                "pgids must differ for this test to be meaningful",
            )
            _terminate_owned_pgroup(target)
            # target must be dead
            self.assertIsNotNone(target.poll(),
                                 "target survived _terminate_owned_pgroup")
            # unrelated must still be alive
            self.assertIsNone(unrelated.poll(),
                              "_terminate_owned_pgroup killed unrelated process")
        finally:
            for p in (unrelated, target):
                if p.poll() is None:
                    p.kill()
                    p.wait()
                p.stdout.close()
                p.stderr.close()

    @unittest.skipUnless(sys.platform != "win32", "POSIX-only")
    def test_term_then_kill_escalation(self):
        """A process that ignores SIGTERM must still be killed within
        _TERM_GRACE_S + _TERM_GRACE_S seconds by SIGKILL escalation."""
        # Script ignores SIGTERM and sleeps forever
        script = Path(self._tmp) / "sigterm_ignore.sh"
        script.write_text(
            "#!/bin/sh\ntrap '' TERM\nsleep 60\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        proc = subprocess.Popen(
            [str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        t0 = time.monotonic()
        try:
            _terminate_owned_pgroup(proc)
            elapsed = time.monotonic() - t0
            self.assertIsNotNone(proc.poll(),
                                 "SIGTERM-ignoring process survived _terminate_owned_pgroup")
            # Must complete within 2x grace window
            self.assertLess(
                elapsed, _TERM_GRACE_S * 2 + 1.0,
                f"escalation took {elapsed:.1f}s; expected < {_TERM_GRACE_S * 2 + 1.0:.1f}s",
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


# ── TestCancellationAcGaps ────────────────────────────────────────────────────────────────────────

class TestCancellationAcGaps(unittest.TestCase):
    """Regression tests that reproduce the three specific AC gaps identified
    in the sta-d76 review: these tests *would have failed* against the buggy
    implementation and must pass against the fixed one.

    Gap 1 – parent exits on TERM, grandchild ignores TERM:
      Old code: once proc.poll() non-None → exited=True → SIGKILL skipped.
      Grandchild in same pgroup survives.

    Gap 2 – leader already exited/reaped before cleanup runs:
      Old code: os.getpgid(proc.pid) on a reaped leader raises
      ProcessLookupError → falls to proc.terminate() on dead leader → no-op.
      Descendants remain alive.

    Gap 3 – child closes stdout/stderr but continues running:
      Old code used proc.communicate() to reap, which blocks until all
      readers are done with the pipe.  A grandchild keeping the write-end
      of stdout open can prevent reap from completing.

    Gap 4 (mock) – cleanup itself raising must not mask caller's exception.
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="tier_a_acgap_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    # Fast tests patch _TERM_GRACE_S to 0.5s; real-process tests tolerate
    # up to 2.0s worst-case.
    _FAST_GRACE = 0.5

    @unittest.skipUnless(sys.platform != "win32", "POSIX-only")
    def test_grandchild_ignores_term_parent_exits_on_term(self):
        """Gap 1: parent exits cleanly on SIGTERM, grandchild traps/ignores SIGTERM.
        Old bug: leader exit set exited=True → SIGKILL branch skipped →
        grandchild survived.  Fixed: SIGKILL is always sent unconditionally.
        """
        import unittest.mock as mock

        gc_pid_file = Path(self._tmp) / "gc_gap1.txt"
        ready_file  = Path(self._tmp) / "gc_gap1_ready.txt"
        script = Path(self._tmp) / "gap1.sh"
        # Parent: exits on SIGTERM (default).  Grandchild: ignores SIGTERM.
        script.write_text(
            f"#!/bin/sh\n"
            f"(trap '' TERM; sleep 60) &\n"
            f"echo $! > {gc_pid_file}\n"
            f"touch {ready_file}\n"
            f"sleep 60\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        proc = subprocess.Popen(
            [str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )

        # Wait for grandchild to be forked
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ready_file.exists():
                break
            time.sleep(0.05)
        self.assertTrue(ready_file.exists(), "grandchild not started in time")
        gc_pid = int(gc_pid_file.read_text().strip())

        try:
            os.kill(gc_pid, 0)
        except ProcessLookupError:
            self.skipTest(f"grandchild {gc_pid} vanished before test")

        with mock.patch("tier_a.engine_cli._TERM_GRACE_S", self._FAST_GRACE):
            _terminate_owned_pgroup(proc)

        time.sleep(0.1)
        try:
            os.kill(gc_pid, 0)
            alive = True
        except (ProcessLookupError, PermissionError):
            alive = False

        self.assertFalse(
            alive,
            f"Gap 1 regression: SIGTERM-ignoring grandchild {gc_pid} survived "
            f"after parent exited on TERM (SIGKILL was not sent unconditionally)",
        )

    @unittest.skipUnless(sys.platform != "win32", "POSIX-only")
    def test_leader_already_exited_descendant_still_alive(self):
        """Gap 2: process group leader has already exited and been reaped
        before _terminate_owned_pgroup is called; a descendant is still alive.
        Old bug: os.getpgid(proc.pid) raised ProcessLookupError on the reaped
        leader → fell back to proc.terminate() on dead leader → descendant
        survived.  Fixed: pgid captured as proc.pid before any signals.
        """
        import unittest.mock as mock

        gc_pid_file = Path(self._tmp) / "gc_gap2.txt"
        ready_file  = Path(self._tmp) / "gc_gap2_ready.txt"
        script = Path(self._tmp) / "gap2.sh"
        # Leader exits immediately after forking a grandchild that sleeps.
        script.write_text(
            f"#!/bin/sh\n"
            f"sleep 60 &\n"
            f"echo $! > {gc_pid_file}\n"
            f"touch {ready_file}\n"
            # Leader exits here; grandchild is re-parented to init but stays
            # in the same pgroup (same session, pgid = original leader pid).
            f"exit 0\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        proc = subprocess.Popen(
            [str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        pgid_at_launch = proc.pid   # stable: pgid == leader pid

        # Wait for the leader to exit naturally
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ready_file.exists() and proc.poll() is not None:
                break
            time.sleep(0.05)
        self.assertIsNotNone(proc.poll(),
                             "leader did not exit in time")
        # Reap it so getpgid would fail on the pid
        proc.wait()

        gc_pid = int(gc_pid_file.read_text().strip())
        try:
            os.kill(gc_pid, 0)
        except ProcessLookupError:
            self.skipTest(f"grandchild {gc_pid} already gone")

        with mock.patch("tier_a.engine_cli._TERM_GRACE_S", self._FAST_GRACE):
            _terminate_owned_pgroup(proc)

        time.sleep(0.1)
        try:
            os.kill(gc_pid, 0)
            alive = True
        except (ProcessLookupError, PermissionError):
            alive = False

        self.assertFalse(
            alive,
            f"Gap 2 regression: descendant {gc_pid} survived after leader was "
            f"already reaped (getpgid fallback didn't kill the group)",
        )

    @unittest.skipUnless(sys.platform != "win32", "POSIX-only")
    def test_child_closes_pipes_grandchild_continues(self):
        """Gap 3: the direct child closes its stdout/stderr but a grandchild
        stays alive in the same pgroup.  Old code used communicate() for reap,
        which can block waiting for the grandchild to close its inherited
        write-end of the pipe.  Fixed: reap uses proc.wait(), not communicate().
        """
        import unittest.mock as mock

        gc_pid_file = Path(self._tmp) / "gc_gap3.txt"
        ready_file  = Path(self._tmp) / "gc_gap3_ready.txt"
        script = Path(self._tmp) / "gap3.sh"
        # Child closes its own stdout/stderr and exits; grandchild keeps
        # inherited file-descriptors open and sleeps.
        script.write_text(
            f"#!/bin/sh\n"
            f"(sleep 60 >/dev/null 2>&1) &\n"
            f"echo $! > {gc_pid_file}\n"
            f"touch {ready_file}\n"
            f"exec 1>/dev/null 2>/dev/null\n"
            f"sleep 60\n",
            encoding="utf-8",
        )
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        proc = subprocess.Popen(
            [str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ready_file.exists():
                break
            time.sleep(0.05)
        self.assertTrue(ready_file.exists(), "setup script not ready in time")
        gc_pid = int(gc_pid_file.read_text().strip())

        try:
            os.kill(gc_pid, 0)
        except ProcessLookupError:
            self.skipTest(f"grandchild {gc_pid} already gone")

        # Key assertion: _terminate_owned_pgroup must complete in bounded time
        # (wait() based, not communicate() based).
        t0 = time.monotonic()
        with mock.patch("tier_a.engine_cli._TERM_GRACE_S", self._FAST_GRACE):
            _terminate_owned_pgroup(proc)
        elapsed = time.monotonic() - t0

        # Must complete well within 2 grace windows
        self.assertLess(
            elapsed, self._FAST_GRACE * 2 + 1.0,
            f"Gap 3 regression: _terminate_owned_pgroup blocked for {elapsed:.1f}s "
            f"(communicate() on open pipe suspected)",
        )

        time.sleep(0.1)
        try:
            os.kill(gc_pid, 0)
            alive = True
        except (ProcessLookupError, PermissionError):
            alive = False

        self.assertFalse(
            alive,
            f"Gap 3 regression: grandchild {gc_pid} still alive after "
            f"_terminate_owned_pgroup (pipe-blocking reap suspected)",
        )

    @unittest.skipUnless(os.name == "posix", "POSIX-only")
    def test_retained_cancellation_closes_owned_pipes(self):
        """Retaining the original traceback must not retain open pipe handles."""
        import unittest.mock as mock

        interrupt = KeyboardInterrupt("retained cancellation")
        children = []
        retained = None
        script = Path(self._tmp) / "retained_pipes.sh"
        script.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IEXEC)

        def cancel(proc, *args, **kwargs):
            children.append(proc)
            raise interrupt

        try:
            with mock.patch.object(subprocess.Popen, "communicate", cancel):
                with mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0.05):
                    try:
                        run_arm(str(script), [], [1], 1, {}, (),
                                tmp_dir=self._tmp)
                    except KeyboardInterrupt as exc:
                        retained = exc
            self.assertIs(retained, interrupt)
            self.assertIsNotNone(retained.__traceback__)
            proc = children[0]
            self.assertIsNotNone(proc.returncode)
            self.assertTrue(proc.stdout.closed, "retained stdout handle is open")
            self.assertTrue(proc.stderr.closed, "retained stderr handle is open")
            self.assertEqual(list(Path(self._tmp).glob("*.ids")), [])
        finally:
            for proc in children:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=2)
                proc.stdout.close()
                proc.stderr.close()

    @unittest.skipUnless(os.name == "posix", "POSIX-only")
    def test_group_signals_precede_reap_and_pipe_close(self):
        """Do not release the leader PID by polling/reaping before escalation."""
        import signal
        import unittest.mock as mock

        proc = mock.Mock(pid=12345)
        events = []
        proc.wait.side_effect = lambda **kw: events.append("wait")
        proc.stdout.close.side_effect = lambda: events.append("stdout.close")
        proc.stderr.close.side_effect = lambda: events.append("stderr.close")
        with mock.patch("tier_a.engine_cli.os.killpg",
                        side_effect=lambda pgid, sig: events.append(sig)):
            with mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0):
                _terminate_owned_pgroup(proc)
        self.assertEqual(events, [signal.SIGTERM, signal.SIGKILL, "wait",
                                  "stdout.close", "stderr.close"])
        proc.poll.assert_not_called()
        proc.communicate.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX-only")
    def test_interrupted_term_or_grace_still_escalates_reaps_and_closes(self):
        """Shared timeout hardening must not regress TERM cancellation cleanup."""
        import signal
        from unittest import mock

        for stage in ("TERM", "grace"):
            with self.subTest(stage=stage):
                original = KeyboardInterrupt(f"interrupted {stage}")
                proc = mock.Mock(pid=123456)
                events = []

                def kill(pgid, sig):
                    events.append(sig)
                    if stage == "TERM" and sig == signal.SIGTERM:
                        raise original

                proc.wait.side_effect = lambda **kw: events.append("wait")
                with (mock.patch("tier_a.engine_cli.os.killpg", side_effect=kill),
                      mock.patch("tier_a.engine_cli.time.sleep",
                                 side_effect=original if stage == "grace" else None),
                      mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0.1)):
                    with self.assertRaises(KeyboardInterrupt) as cm:
                        _terminate_owned_pgroup(proc)
                self.assertIs(cm.exception, original)
                self.assertEqual(events, [signal.SIGTERM, signal.SIGKILL, "wait"])
                proc.wait.assert_called_once_with(timeout=0.1)
                proc.communicate.assert_not_called()
                proc.stdout.close.assert_called_once()
                proc.stderr.close.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "POSIX-only")
    def test_absent_group_is_not_signalled_again(self):
        """ESRCH on TERM ends group ownership; never retry a stale PGID."""
        import signal
        import unittest.mock as mock

        proc = mock.Mock(pid=12345)
        with mock.patch("tier_a.engine_cli.os.killpg",
                        side_effect=ProcessLookupError) as killpg:
            with mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0):
                _terminate_owned_pgroup(proc)
        killpg.assert_called_once_with(proc.pid, signal.SIGTERM)
        proc.wait.assert_called_once()
        proc.stdout.close.assert_called_once()
        proc.stderr.close.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "POSIX-only")
    def test_pipe_close_is_best_effort_even_if_wait_fails(self):
        """Wait errors and one close error must not prevent the other close."""
        import unittest.mock as mock

        for failure in (OSError("wait failure"),
                        subprocess.TimeoutExpired("dummy", 0),
                        KeyboardInterrupt("cleanup interrupted")):
            with self.subTest(failure=type(failure).__name__):
                proc = mock.Mock(pid=12345)
                proc.wait.side_effect = failure
                proc.stdout.close.side_effect = OSError("close failure")
                with mock.patch("tier_a.engine_cli.os.killpg"):
                    with mock.patch("tier_a.engine_cli._TERM_GRACE_S", 0):
                        _terminate_owned_pgroup(proc)
                proc.stdout.close.assert_called_once()
                proc.stderr.close.assert_called_once()

    def test_cleanup_error_does_not_mask_original_exception(self):
        """Gap 4: if _terminate_owned_pgroup itself raises, the original
        caller exception (here: KeyboardInterrupt) must still propagate;
        the cleanup error must be silently discarded.
        """
        import unittest.mock as mock

        def _raising_cleanup(proc):
            raise RuntimeError("simulated cleanup failure")

        proc = mock.Mock()
        proc.communicate.side_effect = KeyboardInterrupt
        with mock.patch("tier_a.engine_cli.subprocess.Popen", return_value=proc):
            with mock.patch(
                "tier_a.engine_cli._terminate_owned_pgroup",
                side_effect=_raising_cleanup,
            ):
                with self.assertRaises(KeyboardInterrupt,
                                       msg="original KI must propagate even when "
                                           "_terminate_owned_pgroup raises"):
                    run_arm(
                        "/bin/true", [], [1, 2, 3], 8,
                        {}, (), 10.0, tmp_dir=self._tmp,
                    )


if __name__ == "__main__":
    unittest.main()
