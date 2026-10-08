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
    fields = dict(v="1", family="-", caller="-", path="-", gu="-1", d="-1",
                  n_embd="-1", n_ff="-1", rows="-1", head_dim="-1", n_rot="-1",
                  ncols="-1", cap_groups="-1", cap_entries="-1", grid_groups="-1",
                  table="-1", capture="-1", trunc="0")
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


def sub16_default_trace() -> str:
    return "\n".join([
        trace_line(family="iq_grouped", caller="native_expert_grouped.gu",
                   path="split_sub16", gu=20, d=20, n_embd=2560, n_ff=640),
        trace_line(family="iq_grouped", caller="native_expert_grouped.down",
                   path="split_sub16", gu=20, d=20, n_embd=2560, n_ff=640),
    ])


def sub16_optout_trace() -> str:
    return "\n".join([
        trace_line(family="iq_grouped", caller="native_expert_grouped.gu",
                   path="split", gu=20, d=20, n_embd=2560, n_ff=640),
        trace_line(family="iq_grouped", caller="native_expert_grouped.down",
                   path="split", gu=20, d=20, n_embd=2560, n_ff=640),
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
        off = offset if status == "ok" else -1
        rb = rows_bytes if status == "ok" else -1
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


def ok_result(**kw) -> ArmResult:
    base = dict(prompt_ids=[1, 2, 3], output_ids=[9, 9], output_hash="h", n_output=2,
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
            {"pos": 0, "T": 2, "picks": [3, 4], "rows": [[0.1, 0.2, 0.3, 0.4],
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
        ], truncation=True, complete=False)
        v = rp.validate_export(man, rows)
        self.assertTrue(v.ok)
        self.assertFalse(v.full_proof)

    def test_export_error_blocks_full_proof(self):
        man, rows = build_export(4, [
            {"pos": 0, "T": 1, "picks": [3], "rows": [[0.1, 0.2, 0.3, 0.4]]},
        ], export_error="shape", complete=False)
        v = rp.validate_export(man, rows)
        self.assertFalse(v.full_proof)

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
            {"pos": 0, "T": 2, "picks": [3, 4], "rows": [[0.1, 0.2, 0.3, 0.4],
                                                          [1.0, 2.0, 3.0, 4.0]]},
        ])
        truncated_rows = rp.parse_rows(
            struct.pack("<4sIqqq", rp._ROWS_MAGIC, 1, 4, 4, 0)
            + struct.pack("<4f", 0.1, 0.2, 0.3, 0.4))  # only one row present
        v = rp.validate_export(man, truncated_rows)
        self.assertFalse(v.ok)


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
        man_b, rb = build_export(4, [{"pos": 0, "T": 1, "picks": [5],
                                      "rows": [[0.1, 0.2, 0.3, 0.4]]}])
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb)
        self.assertFalse(v.ok)

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
            out.append({"pos": pos, "T": T, "picks": [pos + j for j in range(T)],
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
        wb[5]["picks"] = [999]
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
                                 truncation=True, complete=False)
        man_b, rb = build_export(4, [{"pos": 0, "T": 1, "picks": [3], "rows": [list(a[0])]}])
        v = rp.compare_pair(rp.FAMILIES["sub16"], man_a, ra, man_b, rb)
        self.assertTrue(v.ok)
        self.assertFalse(v.full_proof)


# ── file-backed diagnostic evaluation ─────────────────────────────────────────

class TestDiagnosticEvaluation(unittest.TestCase):
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
                            wins = [{"pos": 100 + i, "T": 1, "picks": [7],
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

    def __init__(self, *, diverge_b=False, b_trace_override=None, decode=None):
        self.calls: list[dict] = []
        self.diverge_b = diverge_b
        self.b_trace_override = b_trace_override
        self.decode = decode or {"A": 50.0, "B": 49.0}

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
                wins.append({"pos": 100 + i, "T": 1, "picks": [7], "input_hash": f"{i:016x}",
                             "rows": [row]})
            text, blob = export_files(4, wins, truncation=True, complete=False)
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
        return ok_result(prompt_ids=list(token_ids), output_ids=[7, 8], n_output=2,
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
