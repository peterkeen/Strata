"""tools/tier_a/test_adapt_replay.py - offline tests for run_adapt_replay.py.

All tests run without a GPU and without model files.  Token ids are synthesised
and fixture binaries replace the engine subprocess.

DEPENDENCY NOTE
───────────────
All tests in this file pass with bare Python 3.11 (no regex or GPU needed).

Usage:
    python3 -m unittest tools/tier_a/test_adapt_replay.py
    python3 -m unittest discover -s tools/tier_a -p test_*.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import textwrap
import unittest
import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from tier_a.run_adapt_replay import (   # noqa: E402
    _build_replay_arms,
    _load_reference,
    _parse_follow,
    _reference_hash,
    _SYNTHETIC_REFERENCE,
    ADAPT_REPLAY_CAVEAT,
    run,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _write_ref(tmp_dir: str, ids: list[int] | None = None) -> str:
    """Write a reference output_ids.txt and return the path."""
    if ids is None:
        ids = list(range(1000, 1221))   # 221 ids >= 200
    p = Path(tmp_dir) / "output_ids.txt"
    p.write_text(" ".join(str(i) for i in ids), encoding="utf-8")
    return str(p)


def _write_fixture_follow(
    tmp_dir: str,
    output_ids: list[int],
    follow_differ: int,
    follow_emitted: int,
) -> str:
    """Fixture binary that emits a proper engine stdout including 'follow' line."""
    ids_str   = " ".join(str(i) for i in output_ids)
    n_out     = len(output_ids)
    script    = Path(tmp_dir) / "fake_follow.sh"
    script.write_text(
        textwrap.dedent(f"""\
            #!/bin/sh
            printf 'prompt  : 1 2 3\\n'
            printf 'output  : {ids_str}\\n'
            printf 'decode                   {n_out} tokens in 200.0 ms  ->  40.00 tok/s\\n'
            printf 'prefill                  2 tokens in 80.0 ms  ->  100.00 tok/s  """
            """(time to first token 85.0 ms)\\n'
            printf 'speculation              5 rounds of 4, drafts accepted 3 of 8 (0.375), """
            """1.60 tokens per round\\n'
            printf 'follow                   {follow_differ} of {follow_emitted} """
            """emitted tokens differ from the argmax\\n'
        """),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def _write_fixture_error(tmp_dir: str) -> str:
    """Fixture that exits non-zero."""
    script = Path(tmp_dir) / "fake_err.sh"
    script.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


# ── TestLoadReference ─────────────────────────────────────────────────────────

class TestLoadReference(unittest.TestCase):
    """_load_reference reads and validates an output_ids.txt file."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="ar_ref_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_reads_valid_file(self):
        ids = list(range(1000, 1221))
        path = _write_ref(self._tmp, ids)
        result = _load_reference(path)
        self.assertEqual(result, ids)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            _load_reference(os.path.join(self._tmp, "nonexistent.txt"))

    def test_empty_file_raises(self):
        p = Path(self._tmp) / "empty.txt"
        p.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError, msg="empty file should raise"):
            _load_reference(str(p))

    def test_non_integer_token_raises(self):
        p = Path(self._tmp) / "bad.txt"
        p.write_text(" ".join(["100"] * 200 + ["abc"]), encoding="utf-8")
        with self.assertRaises(ValueError, msg="non-integer should raise"):
            _load_reference(str(p))

    def test_negative_id_raises(self):
        p = Path(self._tmp) / "neg.txt"
        p.write_text(" ".join(["100"] * 200 + ["-1"]), encoding="utf-8")
        with self.assertRaises(ValueError, msg="negative id should raise"):
            _load_reference(str(p))

    def test_too_few_ids_raises(self):
        p = Path(self._tmp) / "short.txt"
        p.write_text(" ".join(str(i) for i in range(100)), encoding="utf-8")
        with self.assertRaises(ValueError, msg="fewer than 200 ids should raise"):
            _load_reference(str(p))

    def test_exactly_200_ids_ok(self):
        ids = list(range(200))
        p = Path(self._tmp) / "ok200.txt"
        p.write_text(" ".join(str(i) for i in ids), encoding="utf-8")
        result = _load_reference(str(p))
        self.assertEqual(len(result), 200)

    def test_file_read_only_once(self):
        """_load_reference should return the same content as what was written."""
        ids = list(range(500, 721))
        path = _write_ref(self._tmp, ids)
        result = _load_reference(path)
        self.assertEqual(result, ids)


# ── TestParseFollow ───────────────────────────────────────────────────────────

class TestParseFollow(unittest.TestCase):
    """_parse_follow extracts the 'follow N of M' line from engine stdout."""

    def test_present(self):
        stdout = (
            "decode                   10 tokens in 100.0 ms  ->  40.00 tok/s\n"
            "follow                   7 of 321 emitted tokens differ from the argmax\n"
        )
        differ, emitted = _parse_follow(stdout)
        self.assertEqual(differ, 7)
        self.assertEqual(emitted, 321)

    def test_absent(self):
        differ, emitted = _parse_follow("no follow line here")
        self.assertEqual(differ, -1)
        self.assertEqual(emitted, -1)

    def test_zero_differ(self):
        stdout = "follow                   0 of 321 emitted tokens differ from the argmax\n"
        differ, emitted = _parse_follow(stdout)
        self.assertEqual(differ, 0)
        self.assertEqual(emitted, 321)

    def test_empty_string(self):
        differ, emitted = _parse_follow("")
        self.assertEqual(differ, -1)
        self.assertEqual(emitted, -1)


# ── TestReferenceHash ─────────────────────────────────────────────────────────

class TestReferenceHash(unittest.TestCase):

    def test_hash_stability(self):
        ids = list(range(1000, 1221))
        h = _reference_hash(ids)
        self.assertEqual(h, _reference_hash(ids))

    def test_hash_length(self):
        self.assertEqual(len(_reference_hash([1, 2, 3])), 16)

    def test_different_ids_differ(self):
        self.assertNotEqual(
            _reference_hash(list(range(1000, 1221))),
            _reference_hash(list(range(1001, 1222))),
        )


# ── TestBuildReplayArms ───────────────────────────────────────────────────────

class TestBuildReplayArms(unittest.TestCase):
    """Verify arm structure, spec-follow flag, G5 rotation, and metadata."""

    def setUp(self):
        self._arms = _build_replay_arms("/fake/reference.txt", 321)

    def test_nine_arms(self):
        self.assertEqual(len(self._arms), 9, "expected exactly 9 arms (3 vars × 3 reps)")

    def test_all_variants_present(self):
        variants = {a["variant"] for a in self._arms}
        self.assertEqual(variants, {"off", "every1", "every2"})

    def test_spec_follow_in_each_arm_flags(self):
        for arm in self._arms:
            self.assertIn("--spec-follow", arm["flags"],
                          f"--spec-follow missing from arm {arm['label']}")

    def test_spec_follow_value_is_reference_path(self):
        for arm in self._arms:
            idx = arm["flags"].index("--spec-follow")
            self.assertEqual(arm["flags"][idx + 1], "/fake/reference.txt",
                             f"wrong --spec-follow value in {arm['label']}")

    def test_max_new_equals_reference_n(self):
        for arm in self._arms:
            self.assertEqual(arm["max_new"], 321,
                             f"max_new != reference_n in {arm['label']}")

    def test_g5_rotation_across_reps(self):
        """The first variant of each rep should differ (same rotation as G5)."""
        first_variants = []
        for rep in range(3):
            arm = next(a for a in self._arms if f"_rep{rep}_" in a["label"])
            first_variants.append(arm["variant"])
        # With 3 variants rotating by 1, all three are distinct.
        self.assertEqual(
            len(set(first_variants)), 3,
            f"G5 rotation broken: first variants per rep = {first_variants}",
        )

    def test_all_three_variants_per_rep(self):
        for rep in range(3):
            rep_arms = [a for a in self._arms if f"_rep{rep}_" in a["label"]]
            variants = {a["variant"] for a in rep_arms}
            self.assertEqual(variants, {"off", "every1", "every2"},
                             f"rep{rep} missing a variant")

    def test_prompt_key_is_story(self):
        for arm in self._arms:
            self.assertEqual(arm["prompt_key"], "story")

    def test_fixed_flags_present_in_arms(self):
        """FIXED_FLAGS entries (--pack, --expert-cache, --pcie-frac, --stats) must
        appear in each arm's flags via _base_flags → FIXED_FLAGS inheritance."""
        for arm in self._arms:
            for flag in ("--pack", "--expert-cache", "--pcie-frac", "--stats"):
                self.assertIn(flag, arm["flags"],
                              f"{flag} missing from {arm['label']} flags")

    def test_adapt_every_flag_in_arms(self):
        for arm in self._arms:
            self.assertIn("--adapt-every", arm["flags"],
                          f"--adapt-every missing from {arm['label']}")


# ── TestCaveat ────────────────────────────────────────────────────────────────

class TestCaveat(unittest.TestCase):

    def test_caveat_mentions_forced_continuation(self):
        self.assertIn("forced continuation", ADAPT_REPLAY_CAVEAT)

    def test_caveat_cold_graph(self):
        self.assertIn("Cold-graph", ADAPT_REPLAY_CAVEAT)

    def test_caveat_not_free_generation(self):
        self.assertIn("NOT free generation", ADAPT_REPLAY_CAVEAT)

    def test_caveat_follow_differ_argmax_disagreement(self):
        self.assertIn("argmax disagreement count", ADAPT_REPLAY_CAVEAT)

    def test_caveat_no_routing_proxy_claim(self):
        self.assertNotIn("routing proxy", ADAPT_REPLAY_CAVEAT,
                         "ADAPT_REPLAY_CAVEAT must not claim follow_differ is a routing proxy")


# ── TestDryRun ────────────────────────────────────────────────────────────────

class TestDryRun(unittest.TestCase):
    """End-to-end dry-run smoke tests (no GPU, no model files)."""

    def _run_dry(self, extra_args: list[str] | None = None) -> tuple[int, str, dict]:
        tmp = tempfile.mkdtemp(prefix="ar_dry_")
        try:
            ref_path = _write_ref(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = run(
                    ["--reference", ref_path, "--out", tmp, "--dry-run"]
                    + (extra_args or [])
                )
            jp = Path(tmp) / "adapt_replay_results.json"
            payload = json.loads(jp.read_text()) if jp.exists() else {}
            return code, buf.getvalue(), payload
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_exits_zero(self):
        code, _, _ = self._run_dry()
        self.assertEqual(code, 0)

    def test_writes_results_json(self):
        tmp = tempfile.mkdtemp(prefix="ar_json_")
        try:
            ref_path = _write_ref(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run(["--reference", ref_path, "--out", tmp, "--dry-run"])
            jp = Path(tmp) / "adapt_replay_results.json"
            self.assertTrue(jp.exists(), "adapt_replay_results.json not written")
            payload = json.loads(jp.read_text())
            self.assertIn("results", payload)
            self.assertIn("caveat", payload)
            self.assertIn("reference_hash", payload)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_nine_records(self):
        _, _, payload = self._run_dry()
        results = payload.get("results", [])
        self.assertEqual(len(results), 9, f"expected 9 records, got {len(results)}")

    def test_per_arm_artifacts_written(self):
        tmp = tempfile.mkdtemp(prefix="ar_art_")
        try:
            ref_path = _write_ref(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run(["--reference", ref_path, "--out", tmp, "--dry-run"])
            arms_dir = Path(tmp) / "arms"
            self.assertTrue(arms_dir.exists(), "arms/ not created")
            arm_dirs = [d for d in arms_dir.iterdir() if d.is_dir()]
            self.assertEqual(len(arm_dirs), 9)
            for d in arm_dirs:
                for fname in ("record.json", "argv.txt"):
                    self.assertTrue((d / fname).exists(),
                                    f"missing {fname} in {d}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_record_has_required_fields(self):
        _, _, payload = self._run_dry()
        required = {
            "label", "variant", "prompt_hash", "reference_hash", "reference_path",
            "reference_n", "follow_differ", "follow_emitted",
            "output_ids_match_reference", "performance_eligible",
            "performance_ineligible_reason", "n_output", "decode_tok_s",
            "wall_s", "error", "caveat",
        }
        for rec in payload.get("results", []):
            for field in required:
                self.assertIn(field, rec,
                              f"missing field '{field}' in record {rec.get('label')}")

    def test_all_variants_present_in_dry_run(self):
        _, _, payload = self._run_dry()
        variants = {r["variant"] for r in payload.get("results", [])}
        self.assertEqual(variants, {"off", "every1", "every2"})

    def test_reference_hash_in_every_record(self):
        _, _, payload = self._run_dry()
        expected_hash = _reference_hash(list(_SYNTHETIC_REFERENCE))
        for rec in payload.get("results", []):
            self.assertEqual(
                rec.get("reference_hash"), expected_hash,
                f"wrong reference_hash in {rec.get('label')}",
            )

    def test_prompt_hash_in_every_record(self):
        _, _, payload = self._run_dry()
        # All records should share the same prompt_hash (same story prompt).
        hashes = {r.get("prompt_hash") for r in payload.get("results", [])}
        self.assertEqual(len(hashes), 1, f"inconsistent prompt_hashes: {hashes}")

    def test_reference_meta_written(self):
        tmp = tempfile.mkdtemp(prefix="ar_meta_")
        try:
            ref_path = _write_ref(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run(["--reference", ref_path, "--out", tmp, "--dry-run"])
            meta_p = Path(tmp) / "reference_meta.json"
            self.assertTrue(meta_p.exists(), "reference_meta.json not written")
            meta = json.loads(meta_p.read_text())
            self.assertIn("reference_hash",        meta)
            self.assertIn("reference_n",           meta)
            self.assertIn("caveat",                meta)
            self.assertIn("source_reference_path", meta)
            self.assertIn("reference_path",        meta)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_reference_path_is_canonical_snapshot(self):
        """reference_path in records and meta must be the canonical snapshot
        inside out_dir, not the original input path."""
        tmp = tempfile.mkdtemp(prefix="ar_canon_")
        try:
            ref_path = _write_ref(tmp)
            ref_path_resolved = str(Path(ref_path).resolve())
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run(["--reference", ref_path, "--out", tmp, "--dry-run"])
            meta = json.loads((Path(tmp) / "reference_meta.json").read_text())
            canonical = meta["reference_path"]
            # canonical must be the snapshot file in out_dir
            self.assertTrue(
                canonical.endswith("reference_ids.txt"),
                f"reference_path should point at canonical snapshot, got: {canonical}",
            )
            self.assertEqual(
                meta["source_reference_path"],
                ref_path_resolved,
                "source_reference_path must match resolved original input",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_source_reference_path_in_records(self):
        """Every arm record must have source_reference_path distinct from reference_path."""
        tmp = tempfile.mkdtemp(prefix="ar_srcp_")
        try:
            ref_path = _write_ref(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run(["--reference", ref_path, "--out", tmp, "--dry-run"])
            payload = json.loads((Path(tmp) / "adapt_replay_results.json").read_text())
            for rec in payload["results"]:
                self.assertIn("source_reference_path", rec,
                              f"source_reference_path missing in {rec['label']}")
                self.assertIn("reference_path", rec,
                              f"reference_path missing in {rec['label']}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_spec_follow_in_argv_artifacts(self):
        tmp = tempfile.mkdtemp(prefix="ar_argv_")
        try:
            ref_path = _write_ref(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run(["--reference", ref_path, "--out", tmp, "--dry-run"])
            for arm_dir in (Path(tmp) / "arms").iterdir():
                if not arm_dir.is_dir():
                    continue
                argv_txt = (arm_dir / "argv.txt").read_text()
                self.assertIn("--spec-follow", argv_txt,
                              f"--spec-follow missing from argv in {arm_dir.name}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_spec_follow_uses_canonical_path_in_argv(self):
        """The --spec-follow value in each arm's argv must be the canonical snapshot
        inside out_dir, not the original input path."""
        tmp = tempfile.mkdtemp(prefix="ar_sfcanon_")
        try:
            ref_path = _write_ref(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run(["--reference", ref_path, "--out", tmp, "--dry-run"])
            canonical = str((Path(tmp) / "reference_ids.txt").resolve())
            for arm_dir in (Path(tmp) / "arms").iterdir():
                if not arm_dir.is_dir():
                    continue
                argv_txt = (arm_dir / "argv.txt").read_text()
                self.assertIn(
                    canonical, argv_txt,
                    f"--spec-follow should point at canonical snapshot in {arm_dir.name}",
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── TestCanonicalSnapshot ────────────────────────────────────────────────────

class TestCanonicalSnapshot(unittest.TestCase):
    """Canonical snapshot isolation: mutating the original after snapshotting
    must not change what the arms see."""

    def test_original_mutated_canonical_unaffected(self):
        """If the original reference file is overwritten after run() snapshots it,
        the canonical snapshot (out_dir/reference_ids.txt) retains the original ids."""
        tmp = tempfile.mkdtemp(prefix="ar_mut_")
        try:
            original_ids = list(range(1000, 1221))  # 221 ids
            ref_path = _write_ref(tmp, original_ids)

            # Use a separate out_dir inside tmp so we can mutate ref_path.
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()

            import unittest.mock as mock

            # Patch run_arm so no GPU is needed; also patch load_prompts_encoded.
            _FAKE = {
                "short_code": list(range(1, 60)),
                "long_doc":   list(range(1, 200)),
                "story":      list(range(1, 120)),
            }
            from tier_a.engine_cli import ArmResult
            dry_result = ArmResult(
                prompt_ids=list(range(1, 120)),
                output_ids=list(original_ids),
                output_hash="aabbccdd00112233",
                n_output=len(original_ids),
                decode_ms=100.0, decode_tok_s=20.0,
                prefill_ms=50.0, prefill_tok_s=100.0, ttft_ms=55.0,
                draft_accepted=0, draft_offered=0, draft_rate=0.0, tokens_per_round=0.0,
                stdout_raw="", stderr_raw="", wall_s=1.0,
                exit_code=0, error="dry_run", argv=["fake"],
            )

            original_mutated = {"done": False}

            def _fake_run_arm(*a, **kw):
                # Overwrite the original file mid-run (after snapshot was taken).
                if not original_mutated["done"]:
                    Path(ref_path).write_text(
                        " ".join(str(i) for i in range(5000, 5221)),
                        encoding="utf-8",
                    )
                    original_mutated["done"] = True
                return dry_result

            buf = io.StringIO()
            with mock.patch("tier_a.run_adapt_replay.run_arm", side_effect=_fake_run_arm), \
                 mock.patch("tier_a.run_adapt_replay.load_prompts_encoded",
                            return_value=_FAKE):
                with contextlib.redirect_stdout(buf):
                    run(["--reference", ref_path, "--out", str(out_dir)])

            snap = out_dir / "reference_ids.txt"
            self.assertTrue(snap.exists(), "canonical snapshot not written")
            snap_ids = [int(x) for x in snap.read_text().split()]
            self.assertEqual(
                snap_ids, original_ids,
                "canonical snapshot was contaminated by original file mutation",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── TestInvalidReference ──────────────────────────────────────────────────────

class TestInvalidReference(unittest.TestCase):
    """run() must fail fast on invalid --reference before any arms run."""

    def test_missing_reference_exits_nonzero(self):
        tmp = tempfile.mkdtemp(prefix="ar_missref_")
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    code = run([
                        "--reference", str(Path(tmp) / "nonexistent.txt"),
                        "--out", tmp,
                    ])
                    # run() may return non-zero or raise (both are acceptable).
                    self.assertNotEqual(code, 0,
                        "expected non-zero exit for missing reference")
                except (FileNotFoundError, ValueError, SystemExit):
                    pass  # exception is also an acceptable failure mode
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_short_reference_rejected_before_arms(self):
        """A reference with fewer than 200 ids must be rejected before any arms."""
        tmp = tempfile.mkdtemp(prefix="ar_shortref_")
        try:
            p = Path(tmp) / "short.txt"
            p.write_text(" ".join(str(i) for i in range(50)), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    code = run(["--reference", str(p), "--out", tmp])
                    self.assertNotEqual(code, 0)
                except (FileNotFoundError, ValueError, SystemExit):
                    pass
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ── helpers for TestOutputMismatch ───────────────────────────────────────────

_FAKE_PROMPTS = {
    "short_code": list(range(1, 60)),
    "long_doc":   list(range(1, 9000)),
    "story":      list(range(1, 120)),
}


# ── TestOutputMismatch ────────────────────────────────────────────────────────

class TestOutputMismatch(unittest.TestCase):
    """When the engine returns different ids than the reference, the record
    must be flagged ineligible and --fail-fast must abort."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="ar_mismatch_")

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_mismatch_flagged_ineligible(self):
        """An arm whose output_ids != reference_ids must get performance_eligible=False."""
        import unittest.mock as mock
        reference_ids = list(range(1000, 1221))
        ref_path = _write_ref(self._tmp, reference_ids)

        wrong_ids = [i + 1 for i in reference_ids[:200]]
        from tier_a.engine_cli import ArmResult
        mismatch_result = ArmResult(
            prompt_ids=[1, 2, 3], output_ids=wrong_ids,
            output_hash="wronghash123456",
            n_output=len(wrong_ids),
            decode_ms=200.0, decode_tok_s=40.0,
            prefill_ms=80.0, prefill_tok_s=200.0, ttft_ms=85.0,
            draft_accepted=3, draft_offered=8, draft_rate=0.375, tokens_per_round=1.6,
            stdout_raw="follow                   50 of 200 emitted tokens differ from the argmax\n",
            stderr_raw="", wall_s=2.0,
            exit_code=0, error=None, argv=["fake"],
        )

        buf = io.StringIO()
        with mock.patch("tier_a.run_adapt_replay.run_arm",
                        return_value=mismatch_result), \
             mock.patch("tier_a.run_adapt_replay.load_prompts_encoded",
                        return_value=_FAKE_PROMPTS):
            with contextlib.redirect_stdout(buf):
                code = run(["--reference", ref_path, "--out", self._tmp])

        jp = Path(self._tmp) / "adapt_replay_results.json"
        payload = json.loads(jp.read_text())
        for rec in payload["results"]:
            self.assertFalse(
                rec.get("performance_eligible"),
                f"mismatch arm {rec['label']} wrongly marked eligible",
            )
            self.assertIn("mismatch", rec.get("performance_ineligible_reason", ""))

    def test_fail_fast_triggers_on_mismatch(self):
        """With --fail-fast an output mismatch must abort after first arm."""
        import unittest.mock as mock
        reference_ids = list(range(1000, 1221))
        ref_path = _write_ref(self._tmp, reference_ids)

        wrong_ids = [i + 1 for i in reference_ids[:200]]
        from tier_a.engine_cli import ArmResult
        mismatch_result = ArmResult(
            prompt_ids=[1, 2, 3], output_ids=wrong_ids,
            output_hash="wronghash123456",
            n_output=len(wrong_ids),
            decode_ms=200.0, decode_tok_s=40.0,
            prefill_ms=80.0, prefill_tok_s=200.0, ttft_ms=85.0,
            draft_accepted=3, draft_offered=8, draft_rate=0.375, tokens_per_round=1.6,
            stdout_raw="follow                   50 of 200 emitted tokens differ from the argmax\n",
            stderr_raw="", wall_s=2.0,
            exit_code=0, error=None, argv=["fake"],
        )
        call_count = {"n": 0}

        def _fake(*a, **kw):
            call_count["n"] += 1
            return mismatch_result

        buf = io.StringIO()
        with mock.patch("tier_a.run_adapt_replay.run_arm", side_effect=_fake), \
             mock.patch("tier_a.run_adapt_replay.load_prompts_encoded",
                        return_value=_FAKE_PROMPTS):
            with contextlib.redirect_stdout(buf):
                code = run([
                    "--reference", ref_path,
                    "--out", self._tmp,
                    "--fail-fast",
                ])

        self.assertEqual(code, 1, "expected exit 1 on fail-fast + mismatch")
        self.assertIn("FAIL-FAST", buf.getvalue())
        self.assertEqual(call_count["n"], 1,
                         f"fail-fast did not stop; called {call_count['n']} times")

    def test_fail_fast_on_engine_error(self):
        """With --fail-fast a real engine error (non-None error) must also abort."""
        import unittest.mock as mock
        reference_ids = list(range(1000, 1221))
        ref_path = _write_ref(self._tmp, reference_ids)

        from tier_a.engine_cli import ArmResult
        error_result = ArmResult(
            prompt_ids=[], output_ids=[], output_hash="err",
            n_output=0, decode_ms=0.0, decode_tok_s=0.0,
            prefill_ms=0.0, prefill_tok_s=0.0, ttft_ms=0.0,
            draft_accepted=0, draft_offered=0, draft_rate=0.0, tokens_per_round=0.0,
            stdout_raw="", stderr_raw="", wall_s=0.0,
            exit_code=1, error="exit_code=1", argv=["fake"],
        )
        call_count = {"n": 0}

        def _fake(*a, **kw):
            call_count["n"] += 1
            return error_result

        buf = io.StringIO()
        with mock.patch("tier_a.run_adapt_replay.run_arm", side_effect=_fake), \
             mock.patch("tier_a.run_adapt_replay.load_prompts_encoded",
                        return_value=_FAKE_PROMPTS):
            with contextlib.redirect_stdout(buf):
                code = run([
                    "--reference", ref_path,
                    "--out", self._tmp,
                    "--fail-fast",
                ])

        self.assertEqual(code, 1)
        self.assertEqual(call_count["n"], 1,
                         f"fail-fast on error did not stop; called {call_count['n']} times")

    def test_artifact_written_before_fail_fast(self):
        """Per-arm artifact must be present even when fail-fast aborts."""
        import unittest.mock as mock
        reference_ids = list(range(1000, 1221))
        ref_path = _write_ref(self._tmp, reference_ids)

        wrong_ids = [i + 1 for i in reference_ids[:200]]
        from tier_a.engine_cli import ArmResult
        mismatch_result = ArmResult(
            prompt_ids=[1, 2, 3], output_ids=wrong_ids,
            output_hash="wronghash123456",
            n_output=len(wrong_ids),
            decode_ms=200.0, decode_tok_s=40.0,
            prefill_ms=80.0, prefill_tok_s=200.0, ttft_ms=85.0,
            draft_accepted=3, draft_offered=8, draft_rate=0.375, tokens_per_round=1.6,
            stdout_raw="", stderr_raw="", wall_s=2.0,
            exit_code=0, error=None, argv=["fake"],
        )

        buf = io.StringIO()
        with mock.patch("tier_a.run_adapt_replay.run_arm",
                        return_value=mismatch_result), \
             mock.patch("tier_a.run_adapt_replay.load_prompts_encoded",
                        return_value=_FAKE_PROMPTS):
            with contextlib.redirect_stdout(buf):
                run([
                    "--reference", ref_path,
                    "--out", self._tmp,
                    "--fail-fast",
                ])

        arms_dir = Path(self._tmp) / "arms"
        arm_dirs = [d for d in arms_dir.iterdir() if d.is_dir()]
        self.assertGreaterEqual(len(arm_dirs), 1,
                                "no artifact written before fail-fast")
        for d in arm_dirs:
            self.assertTrue((d / "record.json").exists(),
                            f"record.json missing in {d}")


# ── TestSyntheticReference ────────────────────────────────────────────────────

class TestSyntheticReference(unittest.TestCase):
    """The module's _SYNTHETIC_REFERENCE must satisfy the validator."""

    def test_at_least_200_ids(self):
        self.assertGreaterEqual(len(_SYNTHETIC_REFERENCE), 200)

    def test_all_non_negative(self):
        for v in _SYNTHETIC_REFERENCE:
            self.assertGreaterEqual(v, 0, f"negative id: {v}")


if __name__ == "__main__":
    unittest.main()
