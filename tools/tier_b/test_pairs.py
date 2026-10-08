"""Offline tests for tools/tier_b/run_pairs.py (sta-osh.9).

No GPU, no network, no model.  Every engine launch is monkeypatched; the
--dry-run path must launch ZERO engines.  Fixtures below reproduce the REAL
STRATA_TIER_B_TRACE record format (src/kernels/cuda/*.cu) and the sta-osh.5
manifest.txt + rows.bin (TBE1) export format byte-for-byte.

Run explicitly (the Tier-A fast suite does not discover this directory):

    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONPATH=/home/pete/.local/share/uv/tools/nanobot-ai/lib/python3.11/site-packages \\
    python3 -m unittest discover -s tools/tier_b -p 'test_*.py' -q
"""
from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from tier_a.engine_cli import ArmResult  # noqa: E402
import tier_b.run_pairs as rp  # noqa: E402


# ── fixture builders ─────────────────────────────────────────────────────────

def trace_line(**kw) -> str:
    # M9: capture defaults to 1 (stream capture active) and not -1, matching the
    # retained r2 traces - a record with an unknown capture state is now
    # inadmissible evidence, so it must be requested explicitly by a fixture.
    fields = dict(v="1", family="-", caller="-", path="-", gu="-1", d="-1",
                  n_embd="-1", n_ff="-1", rows="-1", head_dim="-1", n_rot="-1",
                  ncols="-1", cap_groups="-1", cap_entries="-1", grid_groups="-1",
                  table="-1", capture="1", trunc="0")
    fields.update({k: str(v) for k, v in kw.items()})
    order = ["v", "family", "caller", "path", "gu", "d", "n_embd", "n_ff", "rows",
             "head_dim", "n_rot", "ncols", "cap_groups", "cap_entries",
             "grid_groups", "table", "capture", "trunc"]
    return "TIER_B_TRACE " + " ".join(f"{k}={fields[k]}" for k in order)


def rope_fused_trace() -> str:
    return "\n".join([
        trace_line(family="rope_norm", caller="native_qsa_rms_norm_rope",
                   path="norm_rope_fused", rows=12, head_dim=128, n_rot=64, table=1),
        trace_line(family="rope", caller="native_rope_apply", path="rope128",
                   rows=12, head_dim=128, n_rot=64, table=1),
    ])


def rope_plain_trace() -> str:
    return "\n".join([
        trace_line(family="rope", caller="native_rope_apply", path="rope128",
                   rows=12, head_dim=128, n_rot=64, table=1),
    ])


# M9: the grouped-expert family reports capacity fields, and they must be
# positive for the record to be admissible.  These match the shape of the
# retained r2 traces (cap_groups=cap_entries=20, grid_groups=4).
_CAPS = dict(cap_groups=20, cap_entries=20, grid_groups=4)


def sub16_default_trace(**over) -> str:
    kw = dict(gu=20, d=20, n_embd=2560, n_ff=640, **_CAPS)
    kw.update(over)
    return "\n".join([
        trace_line(family="iq_grouped", caller="native_expert_grouped.gu",
                   path="split_sub16", **kw),
        trace_line(family="iq_grouped", caller="native_expert_grouped.down",
                   path="split_sub16", **kw),
    ])


def sub16_optout_trace(**over) -> str:
    kw = dict(gu=20, d=20, n_embd=2560, n_ff=640, **_CAPS)
    kw.update(over)
    return "\n".join([
        trace_line(family="iq_grouped", caller="native_expert_grouped.gu",
                   path="split", **kw),
        trace_line(family="iq_grouped", caller="native_expert_grouped.down",
                   path="split", **kw),
    ])


HDR = rp._ROWS_HEADER_BYTES


def build_rows(vocab: int, all_rows: list[list[float]]) -> bytes:
    blob = struct.pack("<4sIqqq", rp._ROWS_MAGIC, 1, vocab, vocab, 0)
    for row in all_rows:
        blob += struct.pack(f"<{vocab}f", *row)
    return blob


def export_files(vocab: int, windows: list[dict], *, truncation=False,
                 export_error: str | None = None, complete=True) -> tuple[str, bytes]:
    """Raw (manifest.txt, rows.bin) in the sta-osh.5 on-disk format."""
    lines = ["# tier-b native window export v=1", f"# vocab {vocab}",
             "# max_windows 16", "# max_row_bytes 67108864"]
    all_rows: list[list[float]] = []
    offset = HDR
    for i, w in enumerate(windows):
        T = w["T"]
        rows_bytes = T * vocab * 4
        status = w.get("status", "ok")
        off = w.get("rows_offset", offset if status == "ok" else -1)
        rb = w.get("rows_bytes", rows_bytes if status == "ok" else -1)
        picks = ",".join(str(p) for p in w["picks"])
        lines.append(
            f"W {i} {w['pos']} {T} {w.get('accepted', T - 1)} {w.get('emitted', T)} "
            f"{w.get('input_hash', 'aa' * 8)} {'bb' * 8} {off} {rb} {status} "
            f"natural picks={picks}"
        )
        if status == "ok":
            all_rows.extend(w["rows"])
            offset += rows_bytes
    if truncation:
        lines.append("# truncation 1")
    if export_error:
        lines.append(f"# export_error {export_error}")
    if complete:
        lines.append("# complete 1")
    return "\n".join(lines) + "\n", build_rows(vocab, all_rows)


def build_export(vocab: int, windows: list[dict], **kw):
    """Returns (ExportManifest, RowsFile) parsed through the real parsers."""
    text, blob = export_files(vocab, windows, **kw)
    return rp.parse_manifest(text), rp.parse_rows(blob)


def window_fixture(T=1, **kw) -> dict:
    """Valid toy-vocab metadata by default; explicit overrides exercise rejection."""
    w = {"pos": 100, "T": T, "accepted": T - 1, "emitted": T,
         "picks": [i % 4 for i in range(T)], "rows": [[1.0, 2.0, 3.0, 4.0] for _ in range(T)]}
    w.update(kw)
    return w


def ok_result(**kw) -> ArmResult:
    base = dict(prompt_ids=[1, 2, 3], output_ids=[3, 3], output_hash="h", n_output=2,
                decode_ms=10.0, decode_tok_s=50.0, prefill_ms=5.0, prefill_tok_s=200.0,
                ttft_ms=1.0, draft_accepted=0, draft_offered=0, draft_rate=0.0,
                tokens_per_round=1.0, stdout_raw="", stderr_raw="", wall_s=1.0,
                exit_code=0, error=None, argv=["strata"])
    base.update(kw)
    return ArmResult(**base)


# ── environment / single-override discipline ─────────────────────────────────

class TestEnvDiscipline(unittest.TestCase):
    def test_default_arm_env_has_only_common(self):
        env = rp.build_arm_env(None, "timing")
        self.assertEqual(env, dict(rp.COMMON_STRATA))

    def test_optout_arm_sets_exactly_one_family_var(self):
        env = rp.build_arm_env(rp.FAMILIES["sub16"], "timing")
        self.assertEqual(env["STRATA_NO_SUB16_GU"], "1")
        present = [v for v in rp.FAMILY_OVERRIDE_VARS if v in env]
        self.assertEqual(present, ["STRATA_NO_SUB16_GU"])

    def test_combined_all_off_refused(self):
        bad = {"STRATA_NO_NORM_ROPE": "1", "STRATA_NO_SUB16_GU": "1"}
        with self.assertRaises(ValueError):
            rp.assert_single_family_override(bad)

    def test_diagnostic_env_has_trace_only(self):
        env = rp.build_arm_env(None, "diagnostic")
        self.assertEqual(env["STRATA_TIER_B_TRACE"], "1")
        self.assertNotIn("STRATA_TIER_B_EXPORT", env)   # export dir via flag only

    def test_timing_env_has_no_diagnostics(self):
        env = rp.build_arm_env(rp.FAMILIES["normrope"], "timing")
        for v in rp.DIAGNOSTIC_VARS:
            self.assertNotIn(v, env)
        rp.assert_clean_timing(env, ["--stats", "--stop-eos"])

    def test_assert_clean_timing_rejects_diagnostic_flag(self):
        with self.assertRaises(ValueError):
            rp.assert_clean_timing({}, ["--tier-b-export", "x"])

    def test_assert_clean_timing_rejects_diagnostic_env(self):
        with self.assertRaises(ValueError):
            rp.assert_clean_timing({"STRATA_TIER_B_TRACE": "1"}, [])

    def test_effective_env_strips_ambient_strata(self):
        base = {"PATH": "/x", "STRATA_OLD": "9", "STRATA_PF_FUSED": "0"}
        eff = rp.effective_env(base, {"STRATA_PF_FUSED": "1"})
        self.assertNotIn("STRATA_OLD", eff)
        self.assertEqual(eff["STRATA_PF_FUSED"], "1")
        self.assertEqual(eff["PATH"], "/x")

    def test_ambient_strata_vars_collects_all(self):
        base = {"STRATA_A": "1", "STRATA_B": "2", "HOME": "/h"}
        self.assertEqual(rp.ambient_strata_vars(base), ("STRATA_A", "STRATA_B"))


# ── contract / bounds validation ─────────────────────────────────────────────

class TestContracts(unittest.TestCase):
    def test_sub16_is_bitwise(self):
        f = rp.FAMILIES["sub16"]
        f.validate()
        self.assertEqual(f.contract, "bitwise")
        self.assertEqual(f.rel_tol, 0.0)

    def test_normrope_is_bitwise_identity_arm(self):
        # Fused kernel is bit-identical to the split path (rope_parity check 6).
        f = rp.FAMILIES["normrope"]
        f.validate()
        self.assertEqual(f.contract, "bitwise")
        self.assertEqual(f.rel_tol, 0.0)

    def test_tolerance_family_validation_still_enforced(self):
        with self.assertRaises(ValueError):
            rp.Family("x", "STRATA_X", "tolerance", 0.0).validate()
        with self.assertRaises(ValueError):
            rp.Family("x", "STRATA_X", "bitwise", 1e-3).validate()

    def test_bounds_reject_over_ceiling(self):
        with self.assertRaises(ValueError):
            rp.Bounds(per_arm_timeout_s=601.0).validate()
        with self.assertRaises(ValueError):
            rp.Bounds(max_windows=17).validate()
        with self.assertRaises(ValueError):
            rp.Bounds(session_cap_s=3601.0).validate()
        with self.assertRaises(ValueError):
            rp.Bounds(max_export_bytes=65 * 1024 * 1024).validate()

    def test_bounds_defaults_valid(self):
        rp.Bounds().validate()

    def test_window_bound_matches_binary(self):
        # Regression: harness said 8; generate.cpp opens TierBExport with 16.
        src = (ROOT / "src/program/generate.cpp").read_text(encoding="utf-8")
        self.assertIn("tier_b_export.open(tier_b_dir, 16, (int64_t) 64 << 20", src)
        self.assertEqual(rp.Bounds().max_windows, 16)
        self.assertEqual(rp.Bounds().max_export_bytes, 64 << 20)


# ── trace parsing + validation ───────────────────────────────────────────────

class TestTrace(unittest.TestCase):
    def test_parse_ignores_partial_lines(self):
        txt = "TIER_B_TRACE v=1 family=rope path\nnoise\n" + rope_plain_trace()
        recs = rp.parse_trace(txt)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["path"], "rope128")

    def test_normrope_default_needs_fused(self):
        recs = rp.parse_trace(rope_fused_trace())
        v = rp.validate_normrope_trace(recs, is_default_arm=True)
        self.assertTrue(v.ok, v.reason)

    def test_normrope_default_without_fused_fails(self):
        recs = rp.parse_trace(rope_plain_trace())
        v = rp.validate_normrope_trace(recs, is_default_arm=True)
        self.assertFalse(v.ok)

    def test_normrope_optout_needs_plain_no_fused(self):
        recs = rp.parse_trace(rope_plain_trace())
        v = rp.validate_normrope_trace(recs, is_default_arm=False)
        self.assertTrue(v.ok, v.reason)

    def test_normrope_optout_with_fused_fails(self):
        recs = rp.parse_trace(rope_fused_trace())
        v = rp.validate_normrope_trace(recs, is_default_arm=False)
        self.assertFalse(v.ok)

    def test_sub16_default_needs_split_sub16_both(self):
        recs = rp.parse_trace(sub16_default_trace())
        v = rp.validate_sub16_trace(recs, is_default_arm=True)
        self.assertTrue(v.ok, v.reason)

    def test_sub16_default_missing_down_fails(self):
        only_gu = trace_line(family="iq_grouped", caller="native_expert_grouped.gu",
                             path="split_sub16", n_embd=2560, n_ff=640)
        v = rp.validate_sub16_trace(rp.parse_trace(only_gu), is_default_arm=True)
        self.assertFalse(v.ok)

    def test_sub16_optout_needs_reference(self):
        recs = rp.parse_trace(sub16_optout_trace())
        v = rp.validate_sub16_trace(recs, is_default_arm=False)
        self.assertTrue(v.ok, v.reason)

    def test_sub16_optout_with_sub16_fails(self):
        recs = rp.parse_trace(sub16_default_trace())
        v = rp.validate_sub16_trace(recs, is_default_arm=False)
        self.assertFalse(v.ok)

    def test_truncated_trace_prefix_evidence_ok_but_incomplete(self):
        # Regression: the recorder caps at 1024 records; a cold model run
        # truncates.  Prefix evidence counts, coverage is never claimed full.
        txt = sub16_default_trace() + "\n" + trace_line(
            family="recorder", caller="emit", path="truncated", trunc=1)
        v = rp.validate_sub16_trace(rp.parse_trace(txt), is_default_arm=True)
        self.assertTrue(v.ok, v.reason)
        self.assertFalse(v.complete)

    def test_truncated_record_itself_never_positive(self):
        txt = trace_line(family="recorder", caller="emit", path="truncated", trunc=1)
        for fn in (rp.validate_sub16_trace, rp.validate_normrope_trace):
            v = fn(rp.parse_trace(txt), is_default_arm=True)
            self.assertFalse(v.ok)
            self.assertIn("truncated", v.reason)

    def test_truncated_forbidden_prefix_still_fails(self):
        txt = sub16_default_trace() + "\n" + trace_line(
            family="recorder", caller="emit", path="truncated", trunc=1)
        v = rp.validate_sub16_trace(rp.parse_trace(txt), is_default_arm=False)
        self.assertFalse(v.ok)

    def test_normrope_optout_inplace32_means_optout_ignored(self):
        txt = trace_line(family="rope", caller="native_rope_apply", path="rope_inplace32",
                         rows=12, head_dim=128, n_rot=64, table=1)
        v = rp.validate_normrope_trace(rp.parse_trace(txt), is_default_arm=False)
        self.assertFalse(v.ok)

    def test_skipped_path_not_positive(self):
        txt = "\n".join([
            trace_line(family="iq_grouped", caller="native_expert_grouped.gu",
                       path="skipped", n_embd=2560, n_ff=640),
            trace_line(family="iq_grouped", caller="native_expert_grouped.down",
                       path="skipped", n_embd=2560, n_ff=640),
        ])
        v = rp.validate_sub16_trace(rp.parse_trace(txt), is_default_arm=True)
        self.assertFalse(v.ok)

    def test_wrong_geometry_not_counted(self):
        txt = "\n".join([
            trace_line(family="iq_grouped", caller="native_expert_grouped.gu",
                       path="split_sub16", n_embd=4096, n_ff=640),
            trace_line(family="iq_grouped", caller="native_expert_grouped.down",
                       path="split_sub16", n_embd=2560, n_ff=1024),
        ])
        v = rp.validate_sub16_trace(rp.parse_trace(txt), is_default_arm=True)
        self.assertFalse(v.ok)


# ── export parsing + validation ──────────────────────────────────────────────

class TestExport(unittest.TestCase):
    def test_valid_full_export(self):
        man, rows = build_export(4, [
            {"pos": 0, "T": 2, "picks": [3, 2], "rows": [[0.1, 0.2, 0.3, 0.4],
                                                          [1.0, 2.0, 3.0, 4.0]]},
        ])
        v = rp.validate_export(man, rows)
        self.assertTrue(v.ok and v.full_proof, v.reason)
        self.assertEqual(v.n_windows, 1)

    def test_metadata_only_is_not_full_proof(self):
        man, rows = build_export(4, [])
        v = rp.validate_export(man, rows)
        self.assertFalse(v.ok)
        self.assertFalse(v.full_proof)

    def test_truncation_blocks_full_proof(self):
        man, rows = build_export(4, [
            {"pos": 0, "T": 1, "picks": [3], "rows": [[0.1, 0.2, 0.3, 0.4]]},
        ], truncation=True, complete=True)
        v = rp.validate_export(man, rows)
        self.assertTrue(v.ok)
        self.assertFalse(v.full_proof)

    def test_export_error_rejects_all_proof(self):
        man, rows = build_export(4, [
            {"pos": 0, "T": 1, "picks": [3], "rows": [[0.1, 0.2, 0.3, 0.4]]},
        ], export_error="shape", complete=False)
        v = rp.validate_export(man, rows)
        self.assertFalse(v.ok)
        self.assertFalse(v.full_proof)
        self.assertIn("export_error=shape", v.reason)

    def test_nonfinite_logits_rejected(self):
        man, rows = build_export(4, [
            {"pos": 0, "T": 1, "picks": [3], "rows": [[0.1, float("inf"), 0.3, 0.4]]},
        ])
        v = rp.validate_export(man, rows)
        self.assertFalse(v.ok)

    def test_non_ok_status_rejected(self):
        man, rows = build_export(4, [
            {"pos": 0, "T": 1, "picks": [3], "rows": [], "status": "shape"},
        ], export_error="shape", complete=False)
        v = rp.validate_export(man, rows)
        self.assertFalse(v.ok)

    def test_stale_header_rejected(self):
        man = rp.parse_manifest("# wrong header\n# vocab 4\n")
        self.assertIsNotNone(man.parse_error)

    def test_rows_bad_magic_rejected(self):
        bad = b"XXXX" + bytes(rp._ROWS_HEADER_BYTES - 4)
        self.assertIsNotNone(rp.parse_rows(bad).error)

    def test_short_export_rows_detected(self):
        man, rows = build_export(4, [
            {"pos": 0, "T": 2, "picks": [3, 2], "rows": [[0.1, 0.2, 0.3, 0.4],
                                                          [1.0, 2.0, 3.0, 4.0]]},
        ])
        truncated_rows = rp.parse_rows(
            struct.pack("<4sIqqq", rp._ROWS_MAGIC, 1, 4, 4, 0)
            + struct.pack("<4f", 0.1, 0.2, 0.3, 0.4))  # only one row present
        v = rp.validate_export(man, truncated_rows)
        self.assertFalse(v.ok)


class TestExportFinalization(unittest.TestCase):
    @staticmethod
    def _with_suffix(suffix):
        text, blob = export_files(4, [window_fixture()], complete=False)
        return rp.parse_manifest(text + suffix), rp.parse_rows(blob)

    def test_missing_completion_rejects_even_truncated(self):
        for suffix in ("", "# truncation 1\n"):
            with self.subTest(suffix=suffix):
                man, rows = self._with_suffix(suffix)
                self.assertIsNone(man.parse_error)
                v = rp.validate_export(man, rows)
                self.assertFalse(v.ok or v.full_proof)
                self.assertIn("no complete footer", v.reason)

    def test_export_errors_reject_valid_nonempty_rows(self):
        # close_one() can use multi-word error IDs, not just a single token.
        for suffix in ("# export_error rows_io\n", "# export_error rows close\n",
                       "# truncation 1\n# export_error unknown\n"):
            with self.subTest(suffix=suffix):
                man, rows = self._with_suffix(suffix)
                self.assertIsNone(man.parse_error)
                self.assertIsNotNone(man.export_error)
                self.assertFalse(man.complete)
                self.assertEqual(rows.n_rows, 1)
                v = rp.validate_export(man, rows)
                self.assertFalse(v.ok or v.full_proof)
                self.assertIn(f"export_error={man.export_error}", v.reason)

    def test_export_error_cannot_be_overridden_by_complete(self):
        for suffix in ("# export_error rows_io\n# complete 1\n",
                       "# complete 1\n# export_error rows_io\n",
                       "# truncation 1\n# export_error rows close\n# complete 1\n"):
            with self.subTest(suffix=suffix):
                man, rows = self._with_suffix(suffix)
                self.assertIsNotNone(man.parse_error)
                self.assertFalse(rp.validate_export(man, rows).ok)
                if suffix.startswith("# export_error"):
                    self.assertEqual(man.export_error, "rows_io")
                    self.assertFalse(man.complete)

    def test_completion_footer_must_be_exact(self):
        for line in ("# complete", "# complete ", "# complete 0", "# complete 2",
                     "# complete -1", "# complete true", "# complete 1junk",
                     "# complete 10", "# complete 1 extra", "# complete 1 ",
                     " # complete 1", "#complete 1", "# complete\t1"):
            for later_success in (False, True):
                with self.subTest(line=line, later_success=later_success):
                    suffix = line + "\n" + ("# complete 1\n" if later_success else "")
                    man, rows = self._with_suffix(suffix)
                    self.assertIsNotNone(man.parse_error)
                    self.assertFalse(man.complete)
                    self.assertFalse(rp.validate_export(man, rows).ok)

    def test_duplicate_terminal_metadata_rejected(self):
        for suffix in ("# complete 1\n# complete 1\n",
                       "# complete 1\n# complete 0\n",
                       "# truncation 1\n# truncation 1\n# complete 1\n",
                       "# export_error rows_io\n# export_error unknown\n# complete 1\n"):
            with self.subTest(suffix=suffix):
                man, rows = self._with_suffix(suffix)
                self.assertIn("duplicate terminal metadata", man.parse_error)
                self.assertFalse(rp.validate_export(man, rows).ok)
                if suffix.startswith("# export_error"):
                    self.assertEqual(man.export_error, "rows_io")  # never overwritten

    def test_malformed_terminal_lines_never_hidden_by_success(self):
        for line in ("# truncation", "# truncation 0", "# truncation 1junk",
                     "# truncation 1 extra", "# export_error", "# export_error ",
                     "# export_error   ", "#export_error rows_io",
                     " # export_error rows_io", "# export_error rows_io "):
            with self.subTest(line=line):
                man, rows = self._with_suffix(line + "\n# complete 1\n")
                self.assertIsNotNone(man.parse_error)
                self.assertFalse(rp.validate_export(man, rows).ok)

    def test_terminal_metadata_must_follow_windows_and_success_is_last(self):
        text, _ = export_files(4, [window_fixture()], complete=False)
        window_line = next(line for line in text.splitlines() if line.startswith("W ")) + "\n"
        for suffix in ("# complete 1\n" + window_line,
                       "# truncation 1\n" + window_line + "# complete 1\n",
                       "# export_error rows_io\n" + window_line,
                       "# complete 1\n# vocab 4\n", "# complete 1\n# unknown\n",
                       "# complete 1\n# truncation 1\n",
                       "# export_error rows_io\n# truncation 1\n"):
            with self.subTest(suffix=suffix):
                man, rows = self._with_suffix(suffix)
                self.assertIsNotNone(man.parse_error)
                self.assertFalse(rp.validate_export(man, rows).ok)

    def test_successful_finalization_with_optional_budget_truncation(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated):
                suffix = ("# truncation 1\n" if truncated else "") + "# complete 1\n"
                man, rows = self._with_suffix(suffix)
                self.assertIsNone(man.parse_error)
                self.assertTrue(man.complete)
                self.assertIsNone(man.export_error)
                v = rp.validate_export(man, rows)
                self.assertTrue(v.ok, v.reason)
                self.assertEqual(v.full_proof, not truncated)
                self.assertEqual(v.n_windows, 1)


class TestWindowValidation(unittest.TestCase):
    def test_supported_T_cap_matches_verifier_sources(self):
        self.assertEqual(rp.VERIFY_MAX_T, 8)
        for path, symbol in (("include/strata/kernels/verify_kernels.hpp", "kVerifyMaxT"),
                             ("include/strata/kernels/cpu/expert.hpp", "MAXT")):
            self.assertRegex((ROOT / path).read_text(encoding="utf-8"),
                             rf"inline constexpr int {symbol} = {rp.VERIFY_MAX_T};")

    def test_T_boundaries(self):
        for T in (-1, 0, *range(1, 9), 9):
            with self.subTest(T=T):
                man, rows = build_export(4, [window_fixture(T, accepted=max(0, T - 1),
                                                          emitted=max(0, T))])
                v = rp.validate_export(man, rows)
                self.assertEqual(v.ok, 1 <= T <= 8, v.reason)
                self.assertEqual(v.full_proof, v.ok, v.reason)
                if T in (0, 9):
                    self.assertIn("unsupported T", v.reason)

    def test_accepted_emitted_valid_boundaries_and_clipping(self):
        for T in (1, 4, 8):
            for accepted in range(T):
                for emitted in range(1, accepted + 2):
                    with self.subTest(T=T, accepted=accepted, emitted=emitted):
                        man, rows = build_export(4, [window_fixture(
                            T, accepted=accepted, emitted=emitted)])
                        v = rp.validate_export(man, rows)
                        self.assertTrue(v.ok and v.full_proof, v.reason)
                        self.assertEqual(len(man.windows[0].picks), T)  # not emitted
                        self.assertEqual(man.windows[0].rows_bytes, T * 4 * 4)

    def test_invalid_accepted_emitted_counts(self):
        for T, accepted, emitted in ((1, -1, 1), (1, 1, 1), (1, 0, 0), (1, 0, 2),
                                     (4, 4, 1), (4, 0, 2), (4, 3, 5), (4, 3, -1)):
            with self.subTest(T=T, accepted=accepted, emitted=emitted):
                man, rows = build_export(4, [window_fixture(T, accepted=accepted, emitted=emitted)])
                v = rp.validate_export(man, rows)
                self.assertFalse(v.ok or v.full_proof, v.reason)
        text, blob = export_files(4, [window_fixture(2)])
        for counts in ("2 x 2", "2 1 1.5", "2 1 -1", "2 -1 2"):
            with self.subTest(counts=counts):
                man = rp.parse_manifest(text.replace("W 0 100 2 1 2 ", f"W 0 100 {counts} "))
                self.assertIsNotNone(man.parse_error)
                self.assertFalse(rp.validate_export(man, rp.parse_rows(blob)).ok)

    def test_clipped_prefix_can_only_end_the_run(self):
        for wins, expected in (([window_fixture(4, emitted=1), window_fixture(pos=104)], False),
                               ([window_fixture(), window_fixture(4, pos=101, emitted=1)], True)):
            man, rows = build_export(4, wins)
            for reverse in (False, True):
                with self.subTest(expected=expected, reverse=reverse):
                    if reverse:
                        man.windows.reverse()  # validity uses ordinal, not comparison/list order
                    v = rp.validate_export(man, rows)
                    self.assertEqual(v.ok, expected, v.reason)
                    if not expected:
                        self.assertIn("clipped emitted prefix before later window", v.reason)

    def test_picks_shape_and_token_range(self):
        for picks in ([], [0], [0, 1, 2], [-1, 0], [0, 4]):
            with self.subTest(picks=picks):
                man, rows = build_export(4, [window_fixture(2, picks=picks)])
                self.assertFalse(rp.validate_export(man, rows).ok)
        man, rows = build_export(4, [window_fixture(2, picks=[0, 3])])
        self.assertTrue(rp.validate_export(man, rows).ok)  # both vocabulary boundaries

    def test_malformed_picks_fail_parsing(self):
        text, blob = export_files(4, [window_fixture(2)])
        for picks in ("0,x", "0,1.5", "0,,1", "0,1,"):
            with self.subTest(picks=picks):
                man = rp.parse_manifest(text.replace("picks=0,1", f"picks={picks}"))
                self.assertIsNotNone(man.parse_error)
                self.assertFalse(rp.validate_export(man, rp.parse_rows(blob)).ok)

    def test_bad_row_span_lengths_and_file_boundaries(self):
        for changes in ({"rows_bytes": 0}, {"rows_bytes": -1}, {"rows_bytes": 12},
                        {"rows_bytes": 20}, {"rows_offset": -1}, {"rows_offset": HDR - 1},
                        {"rows_offset": HDR + 16}):
            with self.subTest(changes=changes):
                man, rows = build_export(4, [window_fixture(**changes)])
                v = rp.validate_export(man, rows)
                self.assertFalse(v.ok or v.full_proof, v.reason)

    def test_spans_must_be_whole_row_aligned(self):
        for shift in (1, 4, 8, 12):
            with self.subTest(shift=shift):
                man, rows = build_export(4, [window_fixture(rows_offset=HDR + shift)])
                # Extra finite rows ensure rejection is alignment, not EOF/NaN.
                rows = rp.parse_rows(build_rows(4, [[1.0, 2.0, 3.0, 4.0]] * 3))
                v = rp.validate_export(man, rows)
                self.assertFalse(v.ok)
                self.assertIn("not row-aligned", v.reason)

    def test_row_span_overlap_and_aliasing_rejected(self):
        for second_T, second_offset in ((2, HDR), (2, HDR + 16), (1, HDR + 16)):
            man, rows = build_export(4, [window_fixture(2), window_fixture(
                second_T, pos=102, rows_offset=second_offset)])
            for reverse in (False, True):
                with self.subTest(T=second_T, offset=second_offset, reverse=reverse):
                    if reverse:
                        man.windows.reverse()
                    v = rp.validate_export(man, rows)
                    self.assertFalse(v.ok)
                    self.assertIn("overlapping row span", v.reason)

    def test_adjacent_disjoint_spans_pass_in_either_order(self):
        man, rows = build_export(4, [window_fixture(2), window_fixture(pos=102)])
        for reverse in (False, True):
            if reverse:
                man.windows.reverse()
            v = rp.validate_export(man, rows)
            self.assertTrue(v.ok and v.full_proof, v.reason)


# ── pair comparison under predeclared contracts ──────────────────────────────

class TestPairCompare(unittest.TestCase):
    def _twin(self, rows_a, rows_b, vocab=4):
        man_a, ra = build_export(vocab, [
            {"pos": 0, "T": len(rows_a), "picks": [3] * len(rows_a), "rows": rows_a}])
        man_b, rb = build_export(vocab, [
            {"pos": 0, "T": len(rows_b), "picks": [3] * len(rows_b), "rows": rows_b}])
        return man_a, ra, man_b, rb

    def test_sub16_bitwise_equal_passes(self):
        r = [[0.1, 0.2, 0.3, 0.4]]
        man_a, ra, man_b, rb = self._twin(r, [list(r[0])])
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb)
        self.assertTrue(v.ok and v.full_proof, v.reason)

    def test_identity_signed_zero_mismatch_rejected(self):
        for family in rp.FAMILIES.values():
            for zero_a, zero_b in ((0.0, -0.0), (-0.0, 0.0)):
                with self.subTest(family=family.name, zero_a=repr(zero_a)):
                    a = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, zero_a]]
                    b = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, zero_b]]
                    man_a, ra, man_b, rb = self._twin(a, b)
                    self.assertNotEqual(ra.data, rb.data)
                    self.assertEqual(rp.window_floats(man_a, ra, man_a.windows[0]),
                                     rp.window_floats(man_b, rb, man_b.windows[0]))
                    v = rp.compare_pair(family, man_a, ra, man_b, rb)
                    self.assertFalse(v.ok)
                    self.assertFalse(v.full_proof)
                    self.assertIn("bitwise contract violated", v.reason)

    def test_identity_identical_finite_bytes_pass(self):
        # Both zero encodings, subnormals and the largest finite float32 are valid.
        a = [[0.0, -0.0, 2.0 ** -149, -(2.0 ** -149)],
             [struct.unpack("<f", bytes.fromhex("ffff7f7f"))[0], -1.0, 2.0, 3.0]]
        for family in rp.FAMILIES.values():
            with self.subTest(family=family.name):
                man_a, ra, man_b, rb = self._twin(a, [list(row) for row in a])
                self.assertEqual(ra.data, rb.data)
                v = rp.compare_pair(family, man_a, ra, man_b, rb)
                self.assertTrue(v.ok and v.full_proof, v.reason)

    def test_nonfinite_rejected_independently_of_contract(self):
        tolerance = rp.Family("future", "STRATA_FUTURE", "tolerance", 1e-3)
        tolerance.validate()
        for family in (*rp.FAMILIES.values(), tolerance):
            for value in (float("inf"), float("-inf"), float("nan")):
                for bad_arm in ("A", "B", "both"):
                    with self.subTest(family=family.name, value=value, bad_arm=bad_arm):
                        a = [[value if bad_arm != "B" else 0.0, 1.0, 2.0, 3.0]]
                        b = [[value if bad_arm != "A" else 0.0, 1.0, 2.0, 3.0]]
                        man_a, ra, man_b, rb = self._twin(a, b)
                        if bad_arm == "both":
                            self.assertEqual(ra.data, rb.data)
                        v = rp.compare_pair(family, man_a, ra, man_b, rb)
                        self.assertFalse(v.ok)
                        self.assertFalse(v.full_proof)
                        self.assertIn("non-finite logits", v.reason)
                        self.assertIn(f"arm {'B' if bad_arm == 'B' else 'A'} export invalid",
                                      v.reason)

    def test_tolerance_remains_numeric_not_bitwise(self):
        family = rp.Family("future", "STRATA_FUTURE", "tolerance", 1e-3)
        family.validate()
        a = [[0.0, 0.0, 100.0, -100.0]]
        for b, expected in (([[-0.0, 0.0005, 100.05, -100.05]], True),
                            ([[-0.0, 0.002, 100.0, -100.0]], False),
                            ([[-0.0, 0.0, 100.2, -100.0]], False)):
            with self.subTest(rows_b=b):
                man_a, ra, man_b, rb = self._twin(a, b)
                self.assertNotEqual(ra.data, rb.data)
                v = rp.compare_pair(family, man_a, ra, man_b, rb)
                self.assertEqual(v.ok, expected, v.reason)
                self.assertEqual(v.full_proof, expected, v.reason)
                self.assertIn("tolerance", v.reason)

    def test_sub16_any_diff_fails(self):
        man_a, ra, man_b, rb = self._twin([[0.1, 0.2, 0.3, 0.4]],
                                          [[0.1, 0.2, 0.3, 0.4000001]])
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb)
        self.assertFalse(v.ok)

    def test_normrope_bitwise_equal_passes(self):
        a = [[1.0, 2.0, 3.0, 4.0]]
        man_a, ra, man_b, rb = self._twin(a, [list(a[0])])
        v = rp.compare_pair(rp.FAMILIES["normrope"], man_a, ra, man_b, rb)
        self.assertTrue(v.ok and v.full_proof, v.reason)

    def test_normrope_one_ulp_fails(self):
        a = [[1.0, 2.0, 3.0, 4.0]]
        b = [[1.0000001, 2.0, 3.0, 4.0]]
        man_a, ra, man_b, rb = self._twin(a, b)
        v = rp.compare_pair(rp.FAMILIES["normrope"], man_a, ra, man_b, rb)
        self.assertFalse(v.ok)

    def test_declared_bounds_mismatch_rejected(self):
        man_a, ra, man_b, rb = self._twin([[0.1, 0.2, 0.3, 0.4]], [[0.1, 0.2, 0.3, 0.4]])
        man_a.max_windows = 8
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb, rp.Bounds())
        self.assertFalse(v.ok)

    def test_normrope_beyond_tol_fails(self):
        a = [[1.0, 2.0, 3.0, 4.0]]
        b = [[1.0 * (1 + 1e-1), 2.0, 3.0, 4.0]]
        man_a, ra, man_b, rb = self._twin(a, b)
        v = rp.compare_pair(rp.FAMILIES["normrope"], man_a, ra, man_b, rb)
        self.assertFalse(v.ok)

    def test_mismatched_picks_rejected(self):
        man_a, ra = build_export(4, [{"pos": 0, "T": 1, "picks": [3],
                                      "rows": [[0.1, 0.2, 0.3, 0.4]]}])
        man_b, rb = build_export(4, [{"pos": 0, "T": 1, "picks": [2],
                                      "rows": [[0.1, 0.2, 0.3, 0.4]]}])
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb)
        self.assertFalse(v.ok)

    def test_matched_window_accepted_emitted_metadata_must_match(self):
        for family in rp.FAMILIES.values():
            for counts_a, counts_b in (((1, 2), (2, 2)), ((3, 3), (3, 4))):
                with self.subTest(family=family.name, A=counts_a, B=counts_b):
                    man_a, ra = build_export(4, [window_fixture(
                        4, accepted=counts_a[0], emitted=counts_a[1])])
                    man_b, rb = build_export(4, [window_fixture(
                        4, accepted=counts_b[0], emitted=counts_b[1])])
                    self.assertTrue(rp.validate_export(man_a, ra).ok)
                    self.assertTrue(rp.validate_export(man_b, rb).ok)
                    v = rp.compare_pair(family, man_a, ra, man_b, rb)
                    self.assertFalse(v.ok or v.full_proof)
                    self.assertIn("accepted/emitted counts mismatch", v.reason)

    def test_matched_clipped_window_compares_all_T_picks_and_rows(self):
        man_a, ra = build_export(4, [window_fixture(4, accepted=3, emitted=1)])
        man_b, rb = build_export(4, [window_fixture(4, accepted=3, emitted=1)])
        for family in rp.FAMILIES.values():
            v = rp.compare_pair(family, man_a, ra, man_b, rb)
            self.assertTrue(v.ok and v.full_proof, v.reason)
        man_b.windows[0].picks[-1] = 2  # valid token, outside the emitted prefix
        self.assertEqual(man_a.windows[0].picks[:1], man_b.windows[0].picks[:1])
        v = rp.compare_pair(rp.FAMILIES["normrope"], man_a, ra, man_b, rb)
        self.assertFalse(v.ok)
        self.assertIn("verifier picks mismatch", v.reason)
        self.assertNotIn("emitted token IDs mismatch", v.reason)

    def test_nonemitted_logit_rows_still_obey_family_contract(self):
        wa = window_fixture(4, accepted=3, emitted=1)
        wb = window_fixture(4, accepted=3, emitted=1)
        wb["rows"][-1][-1] = 4.000001
        man_a, ra = build_export(4, [wa])
        man_b, rb = build_export(4, [wb])
        for family in rp.FAMILIES.values():
            v = rp.compare_pair(family, man_a, ra, man_b, rb)
            self.assertFalse(v.ok)
            self.assertIn("bitwise contract violated", v.reason)

    def test_mismatched_input_hash_rejected(self):
        man_a, ra = build_export(4, [{"pos": 0, "T": 1, "picks": [3],
                                      "input_hash": "11" * 8, "rows": [[0.1, 0.2, 0.3, 0.4]]}])
        man_b, rb = build_export(4, [{"pos": 0, "T": 1, "picks": [3],
                                      "input_hash": "22" * 8, "rows": [[0.1, 0.2, 0.3, 0.4]]}])
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb)
        self.assertFalse(v.ok)

    # Regression (host session 1, sub16/long_doc): identical configs drafted
    # pos 8919 as T=4 in one run and T=3 in the other.  Matching is by
    # (pos, T, input_hash); unmatched windows are never compared.

    @staticmethod
    def _wins(spec, bump=None):
        """spec: list of (pos, T); rows are a deterministic function of (pos,T)."""
        out = []
        for pos, T in spec:
            rows = [[pos * 0.5, T * 0.25, j * 1.0, 1.0] for j in range(T)]
            if bump == (pos, T):
                rows[0][3] = 1.0000001
            out.append({"pos": pos, "T": T, "picks": [(pos + j) % 4 for j in range(T)],
                        "input_hash": f"{pos:08x}{T:08x}", "rows": rows})
        return out

    def test_draft_drift_matched_by_key_passes_incomplete(self):
        common = [(p, 1) for p in range(12)]
        man_a, ra = build_export(4, self._wins(common + [(20, 4), (24, 4)]))
        man_b, rb = build_export(4, self._wins(common + [(20, 3), (23, 5)]))
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb, rp.Bounds())
        self.assertTrue(v.ok, v.reason)
        self.assertFalse(v.full_proof)
        self.assertEqual((v.n_matched, v.n_unmatched_a, v.n_unmatched_b), (12, 2, 2))

    def test_draft_drift_order_independent(self):
        spec = [(p, 1) for p in range(12)]
        man_a, ra = build_export(4, self._wins(spec))
        man_b, rb = build_export(4, self._wins(list(reversed(spec))))
        v = rp.compare_pair(rp.FAMILIES["normrope"], man_a, ra, man_b, rb, rp.Bounds())
        self.assertTrue(v.ok and v.full_proof, v.reason)

    def test_matched_window_one_ulp_still_fails(self):
        spec = [(p, 1) for p in range(12)] + [(20, 4)]
        man_a, ra = build_export(4, self._wins(spec))
        man_b, rb = build_export(4, self._wins(spec, bump=(20, 4)))
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb, rp.Bounds())
        self.assertFalse(v.ok)
        self.assertIn("pos=20 T=4", v.reason)

    def test_too_few_matched_windows_rejected(self):
        man_a, ra = build_export(4, self._wins([(p, 1) for p in range(11)] + [(20, 4)]))
        man_b, rb = build_export(4, self._wins([(p, 1) for p in range(11)] + [(20, 3)]))
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb, rp.Bounds())
        self.assertFalse(v.ok)
        self.assertIn("matched windows 11 < required 12", v.reason)

    def test_unmatched_divergent_rows_not_compared(self):
        # Different T => different computation; even wildly different rows on
        # unmatched keys are not evidence either way.
        common = [(p, 1) for p in range(12)]
        wa = self._wins(common + [(20, 4)])
        wb = self._wins(common + [(20, 3)])
        wb[-1]["rows"] = [[9e9, -9e9, 0.0, 1.0]] * 3
        man_a, ra = build_export(4, wa)
        man_b, rb = build_export(4, wb)
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb, rp.Bounds())
        self.assertTrue(v.ok, v.reason)

    def test_same_key_different_picks_rejected(self):
        spec = [(p, 1) for p in range(12)]
        wa, wb = self._wins(spec), self._wins(spec)
        wb[5]["picks"] = [(wb[5]["picks"][0] + 1) % 4]
        man_a, ra = build_export(4, wa)
        man_b, rb = build_export(4, wb)
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb, rp.Bounds())
        self.assertFalse(v.ok)
        self.assertIn("emitted token IDs", v.reason)

    def test_duplicate_key_rejected(self):
        spec = [(p, 1) for p in range(12)] + [(3, 1)]
        man_a, ra = build_export(4, self._wins(spec))
        man_b, rb = build_export(4, self._wins(spec))
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb, rp.Bounds())
        self.assertFalse(v.ok)
        self.assertIn("duplicate window key", v.reason)

    def test_min_matched_bound_validated(self):
        with self.assertRaises(ValueError):
            rp.Bounds(min_matched_windows=0).validate()
        with self.assertRaises(ValueError):
            rp.Bounds(min_matched_windows=17).validate()

    def test_incomplete_coverage_not_full_proof(self):
        a = [[0.1, 0.2, 0.3, 0.4]]
        man_a, ra = build_export(4, [{"pos": 0, "T": 1, "picks": [3], "rows": a}],
                                 truncation=True, complete=True)
        man_b, rb = build_export(4, [{"pos": 0, "T": 1, "picks": [3], "rows": [list(a[0])]}])
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb)
        self.assertTrue(v.ok)
        self.assertFalse(v.full_proof)


# ── file-backed diagnostic evaluation ─────────────────────────────────────────

class TestDiagnosticEvaluation(unittest.TestCase):
    def _finalization_verdict(self, family, suffix_a, suffix_b, n_windows=12):
        import tempfile
        traces = {"normrope": (rope_fused_trace(), rope_plain_trace()),
                  "sub16": (sub16_default_trace(), sub16_optout_trace())}
        wins = [window_fixture(pos=100 + i, input_hash=f"{i:016x}") for i in range(n_windows)]
        text, blob = export_files(4, wins, complete=False)
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp) / arm for arm in ("A", "B")]
            for d, suffix in zip(dirs, (suffix_a, suffix_b)):
                d.mkdir()
                (d / "manifest.txt").write_text(text + suffix, encoding="utf-8")
                (d / "rows.bin").write_bytes(blob)
            trace_a, trace_b = traces[family.name]
            v = rp.evaluate_diagnostic_pair(
                family, "short_code", ok_result(stderr_raw=trace_a), ok_result(stderr_raw=trace_b),
                str(dirs[0]), str(dirs[1]), rp.Bounds())
        self.assertTrue(v["arms_clean"] and v["whole_run_ids_match"])
        self.assertTrue(all(arm["ok"] for arm in v["trace"].values()))
        return v

    def test_failed_and_unfinished_exports_rejected_end_to_end(self):
        for family in rp.FAMILIES.values():
            for suffix in ("", "# truncation 1\n", "# export_error rows_io\n",
                           "# export_error rows close\n", "# truncation 1\n# export_error unknown\n"):
                for bad_arm in ("A", "B", "both"):
                    with self.subTest(family=family.name, suffix=suffix, bad_arm=bad_arm):
                        v = self._finalization_verdict(
                            family, suffix if bad_arm != "B" else "# complete 1\n",
                            suffix if bad_arm != "A" else "# complete 1\n")
                        self.assertFalse(v["proof"] or v["full_proof"] or v["pair"]["ok"])
                        for arm in ("A", "B"):
                            self.assertEqual(v["export"][arm]["ok"], bad_arm not in (arm, "both"))
                            if bad_arm in (arm, "both"):
                                self.assertFalse(v["export"][arm]["full_proof"])
                                expected = "export_error=" if "export_error" in suffix else "no complete footer"
                                self.assertIn(expected, v["export"][arm]["reason"])

    def test_malformed_and_conflicting_footers_rejected_end_to_end(self):
        for family in rp.FAMILIES.values():
            for suffix in ("# complete 1junk\n", "# complete 0\n", "# complete\n",
                           "# complete 1junk\n# complete 1\n", "# complete 0\n# complete 1\n",
                           "# complete 1\n# complete 1\n", "# complete 1\n# complete 0\n",
                           "# export_error rows_io\n# complete 1\n",
                           "# complete 1\n# export_error rows_io\n",
                           "# export_error\n# complete 1\n",
                           "# export_error rows_io\n# export_error unknown\n# complete 1\n",
                           "# truncation 1junk\n# complete 1\n",
                           "# truncation 1\n# truncation 1\n# complete 1\n",
                           "# complete 1\n# truncation 1\n", "# complete 1\n# unknown\n"):
                for bad_arm in ("A", "B", "both"):
                    with self.subTest(family=family.name, suffix=suffix, bad_arm=bad_arm):
                        v = self._finalization_verdict(
                            family, suffix if bad_arm != "B" else "# complete 1\n",
                            suffix if bad_arm != "A" else "# complete 1\n")
                        self.assertFalse(v["proof"] or v["full_proof"] or v["pair"]["ok"])
                        for arm in ("A", "B"):
                            self.assertEqual(v["export"][arm]["ok"], bad_arm not in (arm, "both"))
                            if bad_arm in (arm, "both"):
                                self.assertFalse(v["export"][arm]["full_proof"])
                                self.assertIn("manifest:", v["export"][arm]["reason"])

    def test_finalized_truncation_requires_twelve_nonempty_matches(self):
        for family in rp.FAMILIES.values():
            for truncated_arm in ("A", "B", "both"):
                for n_windows in (11, 12, 16):
                    with self.subTest(family=family.name, arm=truncated_arm, n=n_windows):
                        suffixes = ["# truncation 1\n# complete 1\n" if truncated_arm in (arm, "both")
                                    else "# complete 1\n" for arm in ("A", "B")]
                        v = self._finalization_verdict(family, *suffixes, n_windows=n_windows)
                        self.assertTrue(all(arm["ok"] for arm in v["export"].values()))
                        self.assertEqual(v["proof"], n_windows >= 12, v["reasons"])
                        self.assertEqual(v["pair"]["ok"], n_windows >= 12)
                        self.assertFalse(v["full_proof"] or v["pair"]["full_proof"])
                        self.assertEqual(v["pair"]["n_matched"], n_windows)
                        self.assertEqual(v["pair"]["n_unmatched"], {"A": 0, "B": 0})
                        if n_windows < 12:
                            self.assertIn("matched windows 11 < required 12", v["pair"]["reason"])

    def test_twelve_empty_windows_rejected_end_to_end(self):
        import tempfile
        wins = [{"pos": 100 + i, "T": 0, "accepted": 0, "emitted": 0,
                 "picks": [], "input_hash": f"{i:016x}", "rows": []} for i in range(12)]
        text, blob = export_files(4, wins)
        self.assertEqual(len(blob), HDR)  # the audit's header-only payload
        for name, traces in (("normrope", (rope_fused_trace(), rope_plain_trace())),
                             ("sub16", (sub16_default_trace(), sub16_optout_trace()))):
            with self.subTest(family=name), tempfile.TemporaryDirectory() as tmp:
                dirs = [Path(tmp) / arm for arm in ("A", "B")]
                for d in dirs:
                    d.mkdir()
                    (d / "manifest.txt").write_text(text, encoding="utf-8")
                    (d / "rows.bin").write_bytes(blob)
                v = rp.evaluate_diagnostic_pair(
                    rp.FAMILIES[name], "short_code", ok_result(stderr_raw=traces[0]),
                    ok_result(stderr_raw=traces[1]), str(dirs[0]), str(dirs[1]), rp.Bounds())
                self.assertTrue(v["arms_clean"] and v["whole_run_ids_match"])
                self.assertTrue(all(arm["ok"] for arm in v["trace"].values()))
                self.assertFalse(any(arm["ok"] for arm in v["export"].values()))
                self.assertFalse(v["pair"]["ok"] or v["proof"] or v["full_proof"])
                self.assertEqual(v["pair"]["n_matched"], 0)
                self.assertIn("unsupported T=0", v["pair"]["reason"])

    def test_signed_zero_end_to_end_evaluation(self):
        import tempfile
        traces = {"normrope": (rope_fused_trace(), rope_plain_trace()),
                  "sub16": (sub16_default_trace(), sub16_optout_trace())}
        for family in rp.FAMILIES.values():
            for zero_a, zero_b, expected in ((0.0, -0.0, False),
                                              (0.0, 0.0, True), (-0.0, -0.0, True)):
                with self.subTest(family=family.name, zeros=(repr(zero_a), repr(zero_b))):
                    with tempfile.TemporaryDirectory() as tmp:
                        dirs = [Path(tmp) / arm for arm in ("A", "B")]
                        for d, zero in zip(dirs, (zero_a, zero_b)):
                            d.mkdir()
                            # Reuse H1's 12-window shape; differ only in the last
                            # window's final logit, keeping all metadata/IDs valid.
                            wins = [{"pos": 100 + i, "T": 1, "picks": [3],
                                     "input_hash": f"{i:016x}",
                                     "rows": [[1.0, 2.0, 3.0, zero if i == 11 else 0.0]]}
                                    for i in range(12)]
                            text, blob = export_files(4, wins)
                            (d / "manifest.txt").write_text(text, encoding="utf-8")
                            (d / "rows.bin").write_bytes(blob)
                        trace_a, trace_b = traces[family.name]
                        v = rp.evaluate_diagnostic_pair(
                            family, "short_code", ok_result(stderr_raw=trace_a),
                            ok_result(stderr_raw=trace_b), str(dirs[0]), str(dirs[1]), rp.Bounds())
                    self.assertTrue(v["arms_clean"] and v["whole_run_ids_match"])
                    self.assertTrue(all(arm["ok"] for arm in v["trace"].values()))
                    self.assertTrue(all(arm["full_proof"] for arm in v["export"].values()))
                    self.assertEqual(v["pair"]["n_matched"], 12)
                    self.assertEqual(v["pair"]["n_unmatched"], {"A": 0, "B": 0})
                    self.assertEqual(v["pair"]["ok"], expected, v["pair"]["reason"])
                    self.assertEqual(v["proof"], expected, v["reasons"])
                    self.assertEqual(v["full_proof"], expected, v["reasons"])
                    if not expected:
                        self.assertIn("bitwise contract violated at window pos=111 T=1",
                                      v["pair"]["reason"])


# ── timing discipline ────────────────────────────────────────────────────────

class TestTiming(unittest.TestCase):
    def test_valid_sample(self):
        s = rp.timing_sample(ok_result(), n_prompt=3)
        self.assertTrue(s.ok, s.reason)
        self.assertEqual(s.decode_tok_s, 50.0)

    def test_error_arm_rejected(self):
        s = rp.timing_sample(ok_result(error="timeout"), n_prompt=3)
        self.assertFalse(s.ok)

    def test_zero_decode_rejected(self):
        s = rp.timing_sample(ok_result(n_output=0), n_prompt=3)
        self.assertFalse(s.ok)

    def test_missing_prefill_on_multitoken_rejected(self):
        s = rp.timing_sample(ok_result(prefill_tok_s=0.0), n_prompt=3)
        self.assertFalse(s.ok)

    def test_single_token_prompt_allows_no_prefill(self):
        s = rp.timing_sample(ok_result(prefill_tok_s=0.0), n_prompt=1)
        self.assertTrue(s.ok, s.reason)

    def test_summary_pairs_and_spread(self):
        out = rp.summarise_pairs([10.0, 12.0, 11.0], [20.0, 24.0, 22.0])
        self.assertEqual(out["n_pairs"], 3)
        self.assertEqual(out["paired_deltas"], [10.0, 12.0, 11.0])
        self.assertEqual(out["arm_a"]["median"], 11.0)
        self.assertEqual(out["arm_b"]["n"], 3)


# ── planning: counterbalanced, interleaved, isolated, excluded prep pair ─────

class TestPlanning(unittest.TestCase):
    def setUp(self):
        self.flags = rp.fixed_flags("pack", "native", "ple", "prof", "mtp")
        self.root = Path("/tmp/tier_b_test_root_unused")

    def test_diagnostic_plan_is_one_ab_pair_per_cell(self):
        pairs = rp.plan_diagnostic([rp.FAMILIES["sub16"]], ["short_code"], self.flags, self.root)
        self.assertEqual(len(pairs), 1)
        specs = pairs[0].launch_specs()
        self.assertEqual([s.arm for s in specs], ["A", "B"])
        for s in specs:
            self.assertIn("--tier-b-export", s.flags)
            d = s.flags[s.flags.index("--tier-b-export") + 1]
            self.assertTrue(Path(d).is_absolute())          # regression: was relative
            self.assertEqual(d, s.export_dir)
        self.assertNotEqual(specs[0].export_dir, specs[1].export_dir)

    def test_timing_plan_prep_then_interleaved_rounds(self):
        fams = [rp.FAMILIES["normrope"], rp.FAMILIES["sub16"]]
        wls = ["short_code", "long_doc"]
        pairs = rp.plan_timing(fams, wls, rp.Bounds(recorded_pairs=3), self.flags)
        prep = [p for p in pairs if not p.recorded]
        rec = [p for p in pairs if p.recorded]
        self.assertEqual(len(prep), 4)                   # one excluded pair per cell
        self.assertTrue(all(pairs.index(p) < 4 for p in prep))
        self.assertEqual(len(rec), 12)
        # Regression: rounds are interleaved across ALL cells (not cell-by-cell).
        self.assertEqual([p.round_idx for p in rec], [0] * 4 + [1] * 4 + [2] * 4)
        # Rotation: first cell differs between rounds.
        firsts = [(p.family.name, p.workload) for p in rec[::4]]
        self.assertEqual(len(set(firsts)), 3)
        # AB/BA/AB counterbalance.
        self.assertEqual([p.order for p in rec[::4]], [("A", "B"), ("B", "A"), ("A", "B")])

    def test_timing_arms_carry_no_diagnostics(self):
        pairs = rp.plan_timing([rp.FAMILIES["sub16"]], ["short_code"], rp.Bounds(), self.flags)
        for p in pairs:
            for s in p.launch_specs():
                for v in rp.DIAGNOSTIC_VARS:
                    self.assertNotIn(v, s.env_overrides)
                self.assertNotIn("--tier-b-export", s.flags)
                self.assertIsNone(s.export_dir)

    def test_default_arms_have_no_family_override(self):
        p = rp.plan_diagnostic([rp.FAMILIES["sub16"]], ["short_code"], self.flags, self.root)[0]
        self.assertNotIn("STRATA_NO_SUB16_GU", p.arm_a.env_overrides)
        self.assertEqual(p.arm_b.env_overrides["STRATA_NO_SUB16_GU"], "1")


# ── session clock: the cap bounds every arm, not only the gaps ───────────────

class TestSessionClock(unittest.TestCase):
    def test_arm_timeout_shrinks_to_remaining_budget(self):
        t = {"now": 0.0}
        c = rp.SessionClock(rp.Bounds(), now=lambda: t["now"])
        self.assertEqual(c.arm_timeout(), 600.0)
        t["now"] = 3300.0
        self.assertAlmostEqual(c.arm_timeout(), 3600.0 - 3300.0 - c.GRACE_S)
        t["now"] = 3600.0 - c.GRACE_S - rp.MIN_REMAINING_S + 1
        self.assertIsNone(c.arm_timeout())

    def test_arm_timeout_never_exceeds_cap(self):
        t = {"now": 0.0}
        c = rp.SessionClock(rp.Bounds(), now=lambda: t["now"])
        for now in range(0, 3600, 50):
            t["now"] = float(now)
            to = c.arm_timeout()
            if to is not None:
                self.assertLessEqual(now + to + c.GRACE_S, 3600.0)


# ── report formatting ────────────────────────────────────────────────────────

class TestReport(unittest.TestCase):
    def test_user_line_format(self):
        line = rp.format_variant_line("sub16.default", [40.0, 42.5, 41.0], [100.0, 99.0, 101.0])
        self.assertEqual(line, "sub16.default  decode=41.00tok/s(n=3)  prefill=100.00tok/s(n=3)")

    def test_variant_names(self):
        self.assertEqual(rp.variant_name(rp.FAMILIES["normrope"], "B"), "normrope.no_norm_rope")
        self.assertEqual(rp.variant_name(rp.FAMILIES["sub16"], "A"), "sub16.default")


# ── live driver with a fake engine (no GPU, no model) ────────────────────────

class FakeEngine:
    """Stands in for tier_a.engine_cli.run_arm.  For diagnostic arms it writes a
    real export directory (manifest.txt + rows.bin) exactly where the flag
    points, and emits the family's trace on stderr."""

    def __init__(self, *, diverge_b=False, b_trace_override=None, decode=None,
                 output_divergence_at=None):
        self.calls: list[dict] = []
        self.diverge_b = diverge_b
        self.b_trace_override = b_trace_override
        self.decode = decode or {"A": 50.0, "B": 49.0}
        # Audit H5: 0-based launch index at which this engine returns DIFFERENT
        # decoded output ids, simulating an A/B divergence inside a timing pair.
        self.output_divergence_at = output_divergence_at

    def __call__(self, *, exe, flags, token_ids, max_new, env_overrides, controlled_vars,
                 timeout_s, dry_run, tmp_dir):
        self.calls.append(dict(flags=list(flags), env=dict(env_overrides), timeout_s=timeout_s))
        is_b = any(v in env_overrides for v in rp.FAMILY_OVERRIDE_VARS)
        arm = "B" if is_b else "A"
        stderr = ""
        if "--tier-b-export" in flags:
            d = Path(flags[flags.index("--tier-b-export") + 1])
            assert d.is_absolute() and d.parent.is_dir() and not d.exists()
            d.mkdir()
            wins = []
            for i in range(16):
                row = [0.5, 0.25, 0.125, float(i)]
                if is_b and self.diverge_b and i == 9:
                    row = [0.5, 0.25, 0.125, 9.000001]
                wins.append({"pos": 100 + i, "T": 1, "picks": [3], "input_hash": f"{i:016x}",
                             "rows": [row]})
            # Budget-limited prefix, but generation and export finalized successfully.
            text, blob = export_files(4, wins, truncation=True, complete=True)
            (d / "manifest.txt").write_text(text)
            (d / "rows.bin").write_bytes(blob)
            nr = "STRATA_NO_NORM_ROPE" in env_overrides or not is_b
            if "STRATA_NO_NORM_ROPE" in env_overrides:
                stderr = rope_plain_trace()
            elif "STRATA_NO_SUB16_GU" in env_overrides:
                stderr = sub16_optout_trace()
            else:
                stderr = rope_fused_trace() + "\n" + sub16_default_trace()
            if is_b and self.b_trace_override is not None:
                stderr = self.b_trace_override
            del nr
        output_ids = [3, 2]
        if self.output_divergence_at is not None \
                and len(self.calls) - 1 == self.output_divergence_at:
            output_ids = [3, 7]
        return ok_result(prompt_ids=list(token_ids), output_ids=output_ids,
                         n_output=len(output_ids),
                         decode_tok_s=self.decode[arm], prefill_tok_s=300.0,
                         stderr_raw=stderr, argv=[str(exe), *flags])


class DriverHarness(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        for f in ("native.gguf", "ple.gguf", "prof.bin"):
            (self.tmp / f).write_bytes(b"x")
        for d in ("pack", "mtp"):
            (self.tmp / d).mkdir()
        exe = self.tmp / "strata"
        exe.write_text("#!/bin/sh\nexit 99\n")
        exe.chmod(0o755)
        self.common = ["--exe", str(exe), "--pack", str(self.tmp / "pack"),
                       "--native", str(self.tmp / "native.gguf"),
                       "--ple-gguf", str(self.tmp / "ple.gguf"),
                       "--expert-profile", str(self.tmp / "prof.bin"),
                       "--mtp", str(self.tmp / "mtp")]
        self._real_run_arm = rp.run_arm
        self._real_load = rp.load_prompts
        fake_prompts = {"short_code": [1, 2, 3], "long_doc": [4, 5, 6, 7]}
        rp.load_prompts = lambda dry_run: fake_prompts
        self._real_expect = (dict(rp.EXPECTED_PROMPT_IDS), dict(rp.EXPECTED_PROMPT_HASH))
        rp.EXPECTED_PROMPT_IDS.update({k: len(v) for k, v in fake_prompts.items()})
        rp.EXPECTED_PROMPT_HASH.update({k: rp.prompt_hash(v) for k, v in fake_prompts.items()})

    def tearDown(self):
        rp.run_arm = self._real_run_arm
        rp.load_prompts = self._real_load
        rp.EXPECTED_PROMPT_IDS.clear(); rp.EXPECTED_PROMPT_IDS.update(self._real_expect[0])
        rp.EXPECTED_PROMPT_HASH.clear(); rp.EXPECTED_PROMPT_HASH.update(self._real_expect[1])
        self._td.cleanup()

    def run_live(self, fake, *extra):
        rp.run_arm = fake
        return rp.run([*self.common, *extra])

    def summary(self, out):
        import json
        return json.loads((Path(out) / "summary.json").read_text())


class TestDriverDiagnostic(DriverHarness):
    def test_diagnostic_evaluates_every_pair_and_passes(self):
        # Regression: the driver used to persist artifacts and never validate them.
        out = self.tmp / "diag"
        rc = self.run_live(FakeEngine(), "--mode", "diagnostic", "--out", str(out))
        self.assertEqual(rc, 0)
        s = self.summary(out)
        self.assertTrue(s["complete"])
        self.assertEqual(len(s["pairs"]), 4)
        self.assertTrue(all(p["proof"] for p in s["pairs"]))
        # Fake export is truncated at the 16-window cap: proof, never full proof.
        self.assertFalse(any(p["full_proof"] for p in s["pairs"]))
        self.assertTrue(all(p["pair"]["n_matched"] == 16 for p in s["pairs"]))

    def test_failed_unfinished_and_malformed_exports_fail_driver(self):
        import contextlib
        import io
        for family in rp.FAMILIES.values():
            for suffix in ("", "# export_error rows_io\n", "# complete 1junk\n"):
                for bad_arm in ("A", "B"):
                    with self.subTest(family=family.name, suffix=suffix, bad_arm=bad_arm):
                        fake_engine = FakeEngine()

                        def fake(**kw):
                            result = fake_engine(**kw)  # synthetic, no subprocess/model launch
                            is_b = any(v in kw["env_overrides"] for v in rp.FAMILY_OVERRIDE_VARS)
                            if ("B" if is_b else "A") == bad_arm:
                                flags = kw["flags"]
                                path = Path(flags[flags.index("--tier-b-export") + 1]) / "manifest.txt"
                                text = path.read_text(encoding="utf-8")
                                path.write_text(text.replace("# complete 1\n", suffix), encoding="utf-8")
                            return result

                        out = self.tmp / f"diag-{family.name}-{bad_arm}-{len(suffix)}"
                        with contextlib.redirect_stdout(io.StringIO()):
                            rc = self.run_live(fake, "--mode", "diagnostic", "--family", family.name,
                                               "--workload", "short_code", "--out", str(out))
                        self.assertEqual(rc, 1)
                        s = self.summary(out)
                        self.assertFalse(s["complete"])
                        self.assertTrue(s["failures"])
                        self.assertFalse(s["pairs"][0]["proof"] or s["pairs"][0]["full_proof"])
                        self.assertFalse(s["pairs"][0]["export"][bad_arm]["ok"])

    def test_diagnostic_logit_divergence_fails_run(self):
        out = self.tmp / "diag"
        rc = self.run_live(FakeEngine(diverge_b=True), "--mode", "diagnostic",
                           "--family", "normrope", "--workload", "short_code", "--out", str(out))
        self.assertEqual(rc, 1)
        s = self.summary(out)
        self.assertFalse(s["complete"])
        self.assertIn("bitwise", s["pairs"][0]["pair"]["reason"])

    def test_diagnostic_wrong_dispatch_fails_run(self):
        out = self.tmp / "diag"
        rc = self.run_live(FakeEngine(b_trace_override=sub16_default_trace()),
                           "--mode", "diagnostic", "--family", "sub16",
                           "--workload", "short_code", "--out", str(out))
        self.assertEqual(rc, 1)
        self.assertFalse(self.summary(out)["pairs"][0]["trace"]["B"]["ok"])

    def test_nonempty_out_refused(self):
        out = self.tmp / "diag"
        out.mkdir()
        (out / "stale").write_text("x")
        fake = FakeEngine()
        self.assertEqual(self.run_live(fake, "--mode", "diagnostic", "--out", str(out)), 2)
        self.assertEqual(fake.calls, [])

    def test_empty_model_path_refused_before_launch(self):
        # Regression: --native / --ple-gguf defaulted to "".
        fake = FakeEngine()
        rc = self.run_live(fake, "--mode", "diagnostic", "--native", "",
                           "--out", str(self.tmp / "d"))
        self.assertEqual(rc, 2)
        self.assertEqual(fake.calls, [])

    def test_prompt_hash_mismatch_refused(self):
        rp.EXPECTED_PROMPT_HASH["short_code"] = "0" * 16
        fake = FakeEngine()
        with self.assertRaises(RuntimeError):
            self.run_live(fake, "--mode", "diagnostic", "--out", str(self.tmp / "d"))
        self.assertEqual(fake.calls, [])

    def test_session_cap_bounds_each_arm_timeout(self):
        fake = FakeEngine()
        self.run_live(fake, "--mode", "diagnostic", "--family", "sub16",
                      "--workload", "short_code", "--session-cap-s", "200",
                      "--out", str(self.tmp / "d"))
        self.assertTrue(fake.calls)
        for c in fake.calls:
            self.assertLessEqual(c["timeout_s"], 200.0 - rp.SessionClock.GRACE_S)


class TestDriverTiming(DriverHarness):
    def _proof(self):
        out = self.tmp / "diag"
        self.assertEqual(self.run_live(FakeEngine(), "--mode", "diagnostic", "--out", str(out)), 0)
        return str(out / "summary.json")

    def test_timing_without_proof_refused(self):
        fake = FakeEngine()
        rc = self.run_live(fake, "--mode", "timing", "--out", str(self.tmp / "t"))
        self.assertEqual(rc, 2)
        self.assertEqual(fake.calls, [])

    def test_timing_with_failed_proof_refused(self):
        out = self.tmp / "diag"
        self.run_live(FakeEngine(diverge_b=True), "--mode", "diagnostic", "--out", str(out))
        fake = FakeEngine()
        rc = self.run_live(fake, "--mode", "timing", "--proof", str(out / "summary.json"),
                           "--out", str(self.tmp / "t"))
        self.assertEqual(rc, 2)
        self.assertEqual(fake.calls, [])

    def test_timing_proof_for_other_exe_refused(self):
        proof = self._proof()
        exe2 = self.tmp / "strata2"
        exe2.write_text("#!/bin/sh\nexit 98\n")
        exe2.chmod(0o755)
        fake = FakeEngine()
        rc = self.run_live(fake, "--mode", "timing", "--proof", proof, "--exe", str(exe2),
                           "--out", str(self.tmp / "t"))
        self.assertEqual(rc, 2)
        self.assertEqual(fake.calls, [])

    def test_timing_runs_clean_and_reports_user_format(self):
        import contextlib
        import io
        proof = self._proof()
        fake = FakeEngine(decode={"A": 50.0, "B": 45.0})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.run_live(fake, "--mode", "timing", "--proof", proof,
                               "--out", str(self.tmp / "t"))
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertEqual(len(fake.calls), (4 + 12) * 2)
        for c in fake.calls:
            self.assertNotIn("--tier-b-export", c["flags"])
            self.assertNotIn("STRATA_TIER_B_TRACE", c["env"])
        txt = buf.getvalue()
        self.assertIn("sub16.default  decode=50.00tok/s(n=3)  prefill=300.00tok/s(n=3)", txt)
        self.assertIn("sub16.no_sub16_gu  decode=45.00tok/s(n=3)  prefill=300.00tok/s(n=3)", txt)
        s = self.summary(self.tmp / "t")
        st = s["paired_stats"]["normrope/long_doc"]["decode"]
        self.assertEqual(st["n_pairs"], 3)
        self.assertEqual(st["paired_deltas"], [-5.0, -5.0, -5.0])


# ── M9: recorder errors are not ordinary truncation ────────────────────────

FORMAT_ERROR_MARKER = trace_line(family="recorder", caller="emit", path="format_error", trunc=1)
TRUNCATED_MARKER = trace_line(family="recorder", caller="emit", path="truncated",
                              trunc=1, capture=-1)


class TestRecorderErrorEvidence(unittest.TestCase):
    """Audit M9 (sta-osh.24): a recorder format_error and an unknown capture
    state are recorder FAILURES that reject the trace as evidence, while
    ordinary record-budget truncation only restricts coverage.  Records must
    also carry complete, strictly positive geometry (and, for grouped experts,
    positive nominal capacities) to count."""

    # ---- ordinary truncation still only restricts coverage ----------------
    def test_budget_truncation_restricts_coverage_only(self):
        for name, trace, default in (
            ("normrope-A", rope_fused_trace(), True),
            ("normrope-B", rope_plain_trace(), False),
            ("sub16-A", sub16_default_trace(), True),
            ("sub16-B", sub16_optout_trace(), False),
        ):
            with self.subTest(case=name):
                fam = rp.FAMILIES["normrope" if name.startswith("normrope") else "sub16"]
                recs = rp.parse_trace(trace + "\n" + TRUNCATED_MARKER)
                v = rp.validate_trace(recs, fam, default)
                self.assertTrue(v.ok, v.reason)
                self.assertFalse(v.complete, "budget truncation must clear complete")
                self.assertIn("truncated", v.reason)
                self.assertTrue(rp.trace_truncated(recs))
                self.assertEqual(rp.recorder_errors(recs), [],
                                 "the budget marker is not a recorder error")

    # ---- format_error is a failure, not truncation -------------------------
    def test_format_error_rejects_trace(self):
        for name, trace, default in (
            ("normrope-A", rope_fused_trace(), True),
            ("normrope-B", rope_plain_trace(), False),
            ("sub16-A", sub16_default_trace(), True),
            ("sub16-B", sub16_optout_trace(), False),
        ):
            with self.subTest(case=name):
                fam = rp.FAMILIES["normrope" if name.startswith("normrope") else "sub16"]
                recs = rp.parse_trace(trace + "\n" + FORMAT_ERROR_MARKER)
                v = rp.validate_trace(recs, fam, default)
                self.assertFalse(v.ok, "format_error must reject, not downgrade")
                self.assertFalse(v.complete)
                self.assertIn("recorder evidence rejected", v.reason)
                self.assertIn("format_error", v.reason)
                self.assertIn("not ordinary truncation", v.reason)
                # It is explicitly NOT reported as ordinary coverage truncation.
                self.assertFalse(rp.trace_truncated(recs))
                self.assertTrue(rp.recorder_errors(recs))

    def test_format_error_rejects_even_with_budget_truncation_present(self):
        recs = rp.parse_trace("\n".join([sub16_default_trace(), TRUNCATED_MARKER,
                                         FORMAT_ERROR_MARKER]))
        v = rp.validate_trace(recs, rp.FAMILIES["sub16"], True)
        self.assertFalse(v.ok, v.reason)
        self.assertIn("format_error", v.reason)

    # ---- unknown capture state is a failure -------------------------------
    def test_capture_minus_one_rejects_trace(self):
        for name, trace, default in (
            ("normrope-A", rope_fused_trace(), True),
            ("sub16-A", sub16_default_trace(capture=-1), True),
            ("sub16-B", sub16_optout_trace(capture=-1), False),
        ):
            with self.subTest(case=name):
                fam = rp.FAMILIES["normrope" if name.startswith("normrope") else "sub16"]
                text = trace.replace("capture=1", "capture=-1")
                recs = rp.parse_trace(text)
                v = rp.validate_trace(recs, fam, default)
                self.assertFalse(v.ok, "capture=-1 must not be admissible evidence")
                self.assertIn("unknown capture state capture=-1", v.reason)
                self.assertFalse(rp.trace_truncated(recs),
                                 "an unknown capture state is not truncation")

    def test_out_of_range_and_missing_capture_rejected(self):
        for name, cap in (("two", "2"), ("negative", "-7"), ("word", "unknown")):
            with self.subTest(case=name):
                recs = rp.parse_trace(sub16_default_trace().replace("capture=1", f"capture={cap}"))
                v = rp.validate_trace(recs, rp.FAMILIES["sub16"], True)
                self.assertFalse(v.ok, v.reason)
                self.assertIn(f"unknown capture state capture={cap}", v.reason)

    def test_valid_capture_states_are_admissible(self):
        for cap in ("0", "1"):
            with self.subTest(capture=cap):
                recs = rp.parse_trace(sub16_default_trace().replace("capture=1", f"capture={cap}"))
                self.assertEqual(rp.recorder_errors(recs), [])
                v = rp.validate_trace(recs, rp.FAMILIES["sub16"], True)
                self.assertTrue(v.ok, v.reason)
                self.assertTrue(v.complete)

    # ---- geometry / capacity must be positive ------------------------------
    def test_invalid_sub16_capacities_rejected(self):
        for field in ("cap_groups", "cap_entries", "grid_groups"):
            for bad in (0, -1):
                with self.subTest(field=field, value=bad):
                    recs = rp.parse_trace(sub16_default_trace(**{field: bad}))
                    v = rp.validate_trace(recs, rp.FAMILIES["sub16"], True)
                    self.assertFalse(v.ok, f"{field}={bad} must not count as evidence")
                    self.assertIn("missing split_sub16", v.reason)

    def test_invalid_sub16_geometry_rejected(self):
        # gu/d are the quantization types; 0/-1 means unknown, not a dispatch.
        for field in ("gu", "d"):
            for bad in (0, -1):
                with self.subTest(field=field, value=bad):
                    recs = rp.parse_trace(sub16_default_trace(**{field: bad}))
                    v = rp.validate_trace(recs, rp.FAMILIES["sub16"], True)
                    self.assertFalse(v.ok, f"{field}={bad} must not count as evidence")

    def test_invalid_rope_geometry_rejected(self):
        for field in ("rows", "head_dim", "n_rot"):
            for bad in (0, -1):
                with self.subTest(field=field, value=bad):
                    base = dict(rows=12, head_dim=128, n_rot=64, table=1)
                    base[field] = bad
                    text = trace_line(family="rope_norm", caller="native_qsa_rms_norm_rope",
                                      path="norm_rope_fused", **base)
                    v = rp.validate_trace(rp.parse_trace(text), rp.FAMILIES["normrope"], True)
                    self.assertFalse(v.ok, f"{field}={bad} must not count as evidence")
                    self.assertIn("no norm_rope_fused record", v.reason)

    def test_non_numeric_geometry_rejected(self):
        recs = rp.parse_trace(sub16_default_trace(n_embd="many"))
        v = rp.validate_trace(recs, rp.FAMILIES["sub16"], True)
        self.assertFalse(v.ok, v.reason)

    def test_forbidden_path_still_rejected_despite_valid_geometry(self):
        """M9 narrows what counts as evidence; it must not weaken the existing
        forbidden-record checks."""
        recs = rp.parse_trace(sub16_default_trace())
        v = rp.validate_trace(recs, rp.FAMILIES["sub16"], False)
        self.assertFalse(v.ok)
        self.assertIn("still dispatched split_sub16", v.reason)

    # ---- retained real traces stay valid ----------------------------------
    def test_retained_r2_traces_still_accepted(self):
        """The real r2 measurement traces (capture=0/1, positive geometry and
        capacities) must remain admissible under the tightened rules."""
        root = Path("/tmp/strata-tier-b-artifacts/measure-20261008-r2")
        if not root.is_dir():
            self.skipTest("retained r2 artifacts not present on this host")
        texts = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            try:
                t = p.read_text(errors="replace")
            except OSError:
                continue
            if rp._TRACE_PREFIX in t:
                texts.append((p, t))
        self.assertTrue(texts, "no retained r2 trace files found")
        checked = 0
        for p, t in texts:
            recs = rp.parse_trace(t)
            if not recs:
                continue
            with self.subTest(path=str(p)):
                # Real traces contain no recorder errors, and every record is
                # admissible evidence: nothing was lost to the new rules.
                self.assertEqual(rp.recorder_errors(recs), [])
                usable = rp.admissible_records(recs)
                positives = [r for r in recs if r.get("path") not in rp._NONPOSITIVE_PATHS]
                self.assertEqual(len(usable), len(positives),
                                 "a real positive record was rejected by the new rules")
                for r in usable:
                    self.assertIn(r["capture"], ("0", "1"))
                checked += 1
        self.assertTrue(checked, "no retained r2 records parsed")


# ── H5: output-mismatched timing pair is rejected and stops the run ──────────

class TestTimingOutputMismatchRejected(DriverHarness):
    """Audit H5 (sta-osh.18): when the A and B arms of a timing pair decode
    different output ids, the two arms did not do the same work and their
    speeds are not comparable.  The driver must persist the rejected pair, mark
    the session INCOMPLETE, stop launching further arms, and never let the
    mismatched pair contribute a timing sample."""

    FAMILY = "sub16"
    WORKLOAD = "short_code"

    def _proof(self):
        import contextlib
        import io
        out = self.tmp / "diag"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = self.run_live(FakeEngine(), "--mode", "diagnostic",
                               "--family", self.FAMILY, "--workload", self.WORKLOAD,
                               "--out", str(out))
        self.assertEqual(rc, 0)
        return str(out / "summary.json")

    def _timing(self, fake, tag):
        import contextlib
        import io
        import json
        out = self.tmp / f"timing-{tag}"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.run_live(fake, "--mode", "timing", "--proof", self._proof(),
                               "--family", self.FAMILY, "--workload", self.WORKLOAD,
                               "--out", str(out))
        summary = json.loads((out / "summary.json").read_text())
        return rc, summary, buf.getvalue()

    def test_clean_timing_run_is_the_baseline(self):
        """With matching outputs: 1 prep pair + 3 recorded pairs all launch and
        all 3 recorded pairs are sampled per arm."""
        fake = FakeEngine()
        rc, summary, txt = self._timing(fake, "clean")
        self.assertEqual(rc, 0, txt)
        self.assertEqual(len(fake.calls), 8)            # (1 + 3) pairs x 2 arms
        self.assertEqual(summary["complete"], True)
        self.assertEqual(summary["failures"], [])
        self.assertEqual(len(summary["pairs"]), 4)
        self.assertTrue(all(p["outputs_match"] for p in summary["pairs"]))
        cell = summary["paired_stats"][f"{self.FAMILY}/{self.WORKLOAD}"]
        self.assertEqual(cell["decode"]["n_pairs"], 3)
        self.assertEqual(cell["decode"]["arm_a"]["n"], 3)

    def test_mismatch_in_preparatory_pair_stops_after_first_pair(self):
        """The EXCLUDED preparatory pair counts too: a mismatch there must stop
        the run after exactly that one pair (2 arms), with no recorded pair
        ever launched and no statistics collected."""
        fake = FakeEngine(output_divergence_at=1)       # B arm of the prep pair
        rc, summary, txt = self._timing(fake, "prep")
        self.assertEqual(rc, 1, txt)
        self.assertEqual(len(fake.calls), 2, "launched past the rejected prep pair")
        self.assertIn("FAIL-FAST", txt)
        self.assertIn("A/B output IDs differ in a timing pair", txt)
        self.assertIn("run INCOMPLETE", txt)
        self.assertEqual(summary["complete"], False)
        self.assertEqual(summary["launches"], 2)
        # The rejected pair IS persisted, flagged, and is the only pair present.
        self.assertEqual(len(summary["pairs"]), 1)
        rejected = summary["pairs"][0]
        self.assertIs(rejected["outputs_match"], False)
        self.assertEqual(rejected["recorded"], False)
        self.assertEqual(rejected["round_idx"], -1)
        self.assertEqual(summary["failures"],
                         [f"{self.FAMILY}/{self.WORKLOAD} r-1: "
                          "A/B output IDs differ in a timing pair"])
        # No invalid samples entered the statistics.
        cell = summary["paired_stats"][f"{self.FAMILY}/{self.WORKLOAD}"]
        self.assertEqual(cell["decode"]["n_pairs"], 0)
        self.assertEqual(cell["decode"]["raw_a"], [])
        self.assertEqual(cell["decode"]["raw_b"], [])
        self.assertEqual(cell["prefill"]["n_pairs"], 0)

    def test_mismatch_in_first_recorded_pair_discards_that_pair(self):
        """A mismatch in recorded round 0 stops the run with zero samples: the
        mismatched pair is persisted but contributes nothing."""
        fake = FakeEngine(output_divergence_at=2)       # A arm of recorded r0
        rc, summary, txt = self._timing(fake, "r0")
        self.assertEqual(rc, 1, txt)
        self.assertEqual(len(fake.calls), 4)            # prep pair + rejected pair
        self.assertEqual(len(summary["pairs"]), 2)
        self.assertIs(summary["pairs"][0]["outputs_match"], True)
        self.assertIs(summary["pairs"][1]["outputs_match"], False)
        self.assertEqual(summary["pairs"][1]["round_idx"], 0)
        self.assertEqual(summary["complete"], False)
        cell = summary["paired_stats"][f"{self.FAMILY}/{self.WORKLOAD}"]
        self.assertEqual(cell["decode"]["n_pairs"], 0)
        self.assertEqual(cell["decode"]["raw_a"], [])

    def test_mismatch_in_later_recorded_pair_keeps_only_earlier_samples(self):
        """A mismatch in recorded round 1 stops the run: round 0's valid sample
        is kept, the mismatched round-1 pair is NOT sampled, and round 2 never
        launches."""
        fake = FakeEngine(output_divergence_at=5)       # B arm of recorded r1
        rc, summary, txt = self._timing(fake, "r1")
        self.assertEqual(rc, 1, txt)
        self.assertEqual(len(fake.calls), 6, "round 2 must never launch")
        self.assertEqual(len(summary["pairs"]), 3)
        self.assertEqual([p["outputs_match"] for p in summary["pairs"]],
                         [True, True, False])
        self.assertEqual(summary["complete"], False)
        self.assertEqual(summary["failures"],
                         [f"{self.FAMILY}/{self.WORKLOAD} r1: "
                          "A/B output IDs differ in a timing pair"])
        # Exactly the one valid recorded pair was sampled - not the rejected one.
        cell = summary["paired_stats"][f"{self.FAMILY}/{self.WORKLOAD}"]
        self.assertEqual(cell["decode"]["n_pairs"], 1)
        self.assertEqual(len(cell["decode"]["raw_a"]), 1)
        self.assertEqual(len(cell["decode"]["raw_b"]), 1)


# ── H4: versioned timing-admission protocol ──────────────────────────────────

class TestProofAdmissionProtocol(DriverHarness):
    """Audit H4 (sta-osh.17): a diagnostic summary is admissible for timing only
    under the explicit versioned protocol, with every constituent verdict,
    declared contract/tolerance, recorded common setting, bound and cell-set
    requirement still matching the timing run.  Each fixture below edits exactly
    one thing in an otherwise-valid modern summary and must be REJECTED before a
    single timing engine is launched."""

    FAMILY = "sub16"
    WORKLOAD = "short_code"

    _seq = 0

    def _next_tag(self):
        type(self)._seq += 1
        return f"{type(self)._seq:03d}"

    def _make_proof(self):
        """Produce a genuine, versioned, passing diagnostic summary.json.

        Each call uses a fresh output directory so repeated use inside subTests
        never trips the driver's refusal to reuse an existing --out."""
        import contextlib
        import io
        out = self.tmp / f"diag-{self._next_tag()}"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = self.run_live(FakeEngine(), "--mode", "diagnostic",
                               "--family", self.FAMILY, "--workload", self.WORKLOAD,
                               "--out", str(out))
        self.assertEqual(rc, 0)
        return out / "summary.json"

    def _timing(self, proof_path, *extra):
        """Attempt a timing run against `proof_path`; returns (rc, fake, stdout)."""
        import contextlib
        import io
        fake = FakeEngine()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self.run_live(fake, "--mode", "timing", "--proof", str(proof_path),
                               "--family", self.FAMILY, "--workload", self.WORKLOAD,
                               "--out", str(self.tmp / f"timing-{self._next_tag()}"), *extra)
        return rc, fake, buf.getvalue()

    def _mutate(self, fn, name="mutated"):
        """Apply `fn` to a valid summary dict, save it, and try to admit it."""
        import json
        src = self._make_proof()
        s = json.loads(src.read_text())
        fn(s)
        dst = self.tmp / f"proof-{name}-{self._next_tag()}.json"
        dst.write_text(json.dumps(s, indent=2))
        return self._timing(dst)

    def _reject(self, fn, expected, name="mutated"):
        rc, fake, txt = self._mutate(fn, name)
        self.assertEqual(rc, 2, txt)
        self.assertEqual(fake.calls, [], "timing launched engines on a rejected proof")
        self.assertIn(expected, txt)
        return txt

    # (a) A genuinely valid modern summary still passes end-to-end.
    def test_valid_modern_summary_admitted(self):
        import json
        proof = self._make_proof()
        s = json.loads(proof.read_text())
        self.assertEqual(s["proof_admission_version"], rp.PROOF_ADMISSION_VERSION)
        self.assertEqual(s["provenance"]["proof_admission_version"],
                         rp.PROOF_ADMISSION_VERSION)
        cell = s["pairs"][0]
        self.assertEqual(cell["contract"], "bitwise")
        self.assertEqual(cell["rel_tol"], 0.0)
        self.assertEqual(cell["bounds"]["min_matched_windows"], 12)
        rc, fake, txt = self._timing(proof)
        self.assertEqual(rc, 0, txt)
        self.assertIn("proof gate: PASS", txt)
        self.assertEqual(len(fake.calls), (1 + 3) * 2)   # prep pair + 3 recorded pairs

    # (b) A legacy summary WITHOUT the version stamp requires revalidation.
    def test_legacy_unversioned_summary_requires_revalidation(self):
        def drop_version(s):
            s.pop("proof_admission_version", None)
            s["provenance"].pop("proof_admission_version", None)
        txt = self._reject(drop_version, "requires revalidation", "legacy")
        self.assertIn("no proof_admission_version", txt)
        self.assertIn(rp.PROOF_ADMISSION_VERSION, txt)
        # No silent backward compatibility: the legacy refusal is the ONLY
        # problem printed, so none of the weaker checks were consulted.
        self.assertEqual(txt.count("REFUSED (proof gate):"), 1)

    def test_partially_versioned_and_stale_versions_rejected(self):
        for name, fn in (
            ("summary-only", lambda s: s["provenance"].pop("proof_admission_version")),
            ("prov-only", lambda s: s.pop("proof_admission_version")),
            ("disagree", lambda s: s.__setitem__("proof_admission_version", "other/9")),
            ("stale-both", lambda s: (s.__setitem__("proof_admission_version", "old/0"),
                                      s["provenance"].__setitem__(
                                          "proof_admission_version", "old/0"))),
        ):
            with self.subTest(case=name):
                self._reject(fn, "requires revalidation", name)

    # (c) Tolerance edited 0.0 -> 0.5 no longer proves the identity arms.
    def test_edited_tolerance_rejected(self):
        def loosen(s):
            s["pairs"][0]["rel_tol"] = 0.5
        self._reject(loosen, "proof rel_tol for sub16/short_code is 0.5, declared 0.0",
                     "tol05")

    def test_edited_contract_name_rejected(self):
        def retitle(s):
            s["pairs"][0]["contract"] = "tolerance"
        self._reject(retitle, "proof contract for sub16/short_code is 'tolerance'",
                     "contract")

    # (d) min_matched_windows lowered to 1 in the proof summary.
    def test_lowered_min_matched_windows_rejected(self):
        def lower(s):
            s["provenance"]["bounds"]["min_matched_windows"] = 1
            s["pairs"][0]["bounds"]["min_matched_windows"] = 1
        txt = self._reject(lower, "proof bound min_matched_windows=1 differs from timing 12",
                           "minmatch1")
        self.assertIn("proof bound min_matched_windows for sub16/short_code is 1, declared 12",
                      txt)

    # (e) Only one matched window in the cell.
    def test_single_matched_window_rejected(self):
        def shrink(s):
            s["pairs"][0]["pair"]["n_matched"] = 1
            s["pairs"][0]["n_windows"] = 1
        self._reject(shrink, "proof matched windows for sub16/short_code is 1 < required 12",
                     "one-window")

    def test_window_count_disagreement_rejected(self):
        def desync(s):
            s["pairs"][0]["n_windows"] = 16
            s["pairs"][0]["pair"]["n_matched"] = 12
        self._reject(desync, "disagrees with pair.n_matched", "desync")

    # (f) A nested verdict flipped false while proof stays True.
    def test_nested_verdict_false_under_proof_true_rejected(self):
        for path in rp.ADMISSION_NESTED_VERDICTS:
            with self.subTest(verdict=".".join(path)):
                def flip(s, path=path):
                    node = s["pairs"][0]
                    for k in path[:-1]:
                        node = node[k]
                    node[path[-1]] = False
                    self.assertIs(s["pairs"][0]["proof"], True)
                self._reject(flip, f"nested verdict {'.'.join(path)} is False, required True",
                             "nested-" + "-".join(path))

    def test_arm_error_recorded_under_proof_true_rejected(self):
        def poison(s):
            s["pairs"][0]["arm_errors"]["B"] = "timeout"
        self._reject(poison, "arm_errors", "armerr")

    def test_empty_output_metadata_rejected(self):
        for name, fn in (
            ("hash", lambda s: s["pairs"][0]["output_hash"].__setitem__("A", "")),
            ("n_output", lambda s: s["pairs"][0]["n_output"].__setitem__("B", 0)),
            ("export-n", lambda s: s["pairs"][0]["export"]["A"].__setitem__("n_windows", 0)),
        ):
            with self.subTest(field=name):
                self._reject(fn, "proof cell sub16/short_code", "empty-" + name)

    # (g) Changed common flags / controlled settings.
    def test_changed_common_flags_rejected(self):
        def retune(s):
            s["provenance"]["common_strata"] = {"STRATA_PF_FUSED": "0",
                                                "STRATA_IQ_MT_MIN": "1"}
        self._reject(retune, "proof common_strata", "commonflags")

    def test_changed_controlled_vars_rejected(self):
        def retune(s):
            s["provenance"]["controlled_strata_vars"] = ["STRATA_SOMETHING_ELSE"]
        self._reject(retune, "proof controlled STRATA variable list differs", "ctrlvars")

    # (h) Changed export/session bounds.
    def test_changed_bounds_rejected(self):
        for key, value, expected in (
            ("max_windows", 8, "proof bound max_windows=8 differs from timing 16"),
            ("max_export_bytes", 1024, "proof bound max_export_bytes=1024 differs"),
            ("recorded_pairs", 1, "proof bound recorded_pairs=1 differs from timing 3"),
            ("max_new_short", 8, "proof bound max_new_short=8 differs from timing 256"),
            ("max_new_long", 8, "proof bound max_new_long=8 differs from timing 512"),
        ):
            with self.subTest(bound=key):
                def change(s, key=key, value=value):
                    s["provenance"]["bounds"][key] = value
                    if key in s["pairs"][0]["bounds"]:
                        s["pairs"][0]["bounds"][key] = value
                self._reject(change, expected, "bound-" + key)

    def test_timeout_exceeding_session_cap_rejected(self):
        def overrun(s):
            s["provenance"]["bounds"]["session_cap_s"] = 100.0
            s["provenance"]["bounds"]["per_arm_timeout_s"] = 600.0
        self._reject(overrun, "not within session_cap_s", "timeout-cap")

    def test_missing_cell_bounds_rejected(self):
        def strip(s):
            s["pairs"][0].pop("bounds")
        self._reject(strip, "records no applied bounds", "nobounds")

    # (i) Duplicate / conflicting / unknown / missing cells.
    def test_duplicated_conflicting_cell_rejected(self):
        import copy

        def dup(s):
            clone = copy.deepcopy(s["pairs"][0])
            clone["pair"]["n_matched"] = 1
            clone["n_windows"] = 1
            s["pairs"].append(clone)
        self._reject(dup, "duplicate conflicting cell for sub16/short_code", "dupconflict")

    def test_duplicated_identical_cell_rejected(self):
        import copy

        def dup(s):
            s["pairs"].append(copy.deepcopy(s["pairs"][0]))
        self._reject(dup, "duplicate cell for sub16/short_code", "dupsame")

    def test_unknown_extra_cell_and_wrong_family_names_rejected(self):
        import copy
        for name, mangle, expected in (
            ("unknown-family", lambda c: c.__setitem__("family", "not_a_family"),
             "unknown family 'not_a_family'"),
            ("unknown-workload", lambda c: c.__setitem__("workload", "not_a_workload"),
             "unknown workload 'not_a_workload'"),
            ("extra-cell", lambda c: c.__setitem__("workload", "long_doc"),
             "unknown extra cell sub16/long_doc"),
        ):
            with self.subTest(case=name):
                def add(s, mangle=mangle):
                    clone = copy.deepcopy(s["pairs"][0])
                    mangle(clone)
                    s["pairs"].append(clone)
                self._reject(add, expected, name)

    def test_mislabelled_family_inside_required_cell_rejected(self):
        def relabel(s):
            s["pairs"] = []
        self._reject(relabel, "no proof for sub16/short_code", "nocells")

    def test_unknown_declared_family_in_provenance_rejected(self):
        def inject(s):
            s["provenance"]["families"]["bogus"] = {"override_env": "STRATA_BOGUS",
                                                    "contract": "bitwise", "rel_tol": 0.0}
        self._reject(inject, "declares unknown families ['bogus']", "bogusfam")

    def test_changed_declared_family_contract_rejected(self):
        for key, value in (("override_env", "STRATA_OTHER"), ("contract", "tolerance"),
                           ("rel_tol", 0.5)):
            with self.subTest(field=key):
                def change(s, key=key, value=value):
                    s["provenance"]["families"][self.FAMILY][key] = value
                self._reject(change, f"proof family sub16 {key}", "fam-" + key)

    # (j) H1/H2/H3 behaviour is unchanged: a run that recorded failures, or one
    # that did not complete, is still inadmissible.
    def test_recorded_failures_still_rejected(self):
        def fail(s):
            s["failures"] = ["sub16/short_code: proof failed"]
        self._reject(fail, "proof run recorded failures", "failures")

    def test_incomplete_or_dry_run_summary_still_rejected(self):
        for name, fn, expected in (
            ("incomplete", lambda s: s.__setitem__("complete", False),
             "proof run did not complete"),
            ("dry", lambda s: s.__setitem__("dry_run", True),
             "not a live diagnostic run"),
            ("timing-mode", lambda s: s.__setitem__("mode", "timing"),
             "not a live diagnostic run"),
        ):
            with self.subTest(case=name):
                self._reject(fn, expected, name)

    def test_unreadable_and_non_object_summaries_rejected(self):
        for name, text, expected in (("missing", None, "cannot read proof summary"),
                                     ("garbage", "{not json", "cannot read proof summary"),
                                     ("list", "[]", "is not a JSON object")):
            with self.subTest(case=name):
                p = self.tmp / f"bad-{name}.json"
                if text is not None:
                    p.write_text(text)
                rc, fake, txt = self._timing(p)
                self.assertEqual(rc, 2)
                self.assertEqual(fake.calls, [])
                self.assertIn(expected, txt)


# ── dry-run launches ZERO engines ────────────────────────────────────────────

class TestDryRun(unittest.TestCase):
    def test_dry_run_launches_nothing(self):
        calls = {"n": 0}
        real = rp.run_arm

        def spy(*a, **kw):
            if not kw.get("dry_run"):
                calls["n"] += 1
            return real(*a, **kw)

        rp.run_arm = spy
        try:
            rc = rp.run(["--mode", "diagnostic", "--dry-run",
                         "--family", "sub16", "--workload", "short_code"])
        finally:
            rp.run_arm = real
        self.assertEqual(rc, 0)
        self.assertEqual(calls["n"], 0)

    def test_dry_run_timing_both_families(self):
        calls = {"n": 0}
        real = rp.run_arm

        def spy(*a, **kw):
            if not kw.get("dry_run"):
                calls["n"] += 1
            return real(*a, **kw)

        rp.run_arm = spy
        try:
            rc = rp.run(["--mode", "timing", "--dry-run"])
        finally:
            rp.run_arm = real
        self.assertEqual(rc, 0)
        self.assertEqual(calls["n"], 0)


# ── provenance manifest ──────────────────────────────────────────────────────

class TestProvenance(unittest.TestCase):
    def test_provenance_records_identity_bounds_prompts(self):
        import argparse
        import tempfile
        args = argparse.Namespace(
            mode="diagnostic", dry_run=True, exe="/no/such/strata",
            pack="/no/pack", native="", ple_gguf="", expert_profile="/no/prof",
            mtp="/no/mtp")
        prompts = {"short_code": [1, 2, 3], "long_doc": list(range(50))}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            prov = rp.write_provenance(
                out, args, rp.Bounds(), [rp.FAMILIES["sub16"]],
                ["short_code"], prompts, {"STRATA_X": "1"})
            self.assertTrue((out / "provenance.json").exists())
        self.assertIsNone(prov["exe_sha256"])            # absent exe => None, no raise
        self.assertEqual(prov["controlled_strata_vars"], ["STRATA_X"])
        self.assertEqual(prov["families"]["sub16"]["contract"], "bitwise")
        self.assertEqual(prov["bounds"]["max_windows"], 16)
        self.assertEqual(prov["prompts"]["short_code"]["n_ids"], 3)
        self.assertIn("hash", prov["prompts"]["long_doc"])


if __name__ == "__main__":
    unittest.main()
