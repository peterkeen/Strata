"""tools/tier_b/test_diagnostics.py - offline regression tests for sta-osh.5.

Covers the native verification-window evidence exporter
(include/strata/program/tier_b_export.hpp) and its generate.cpp integration,
without a GPU or nvcc.  The tier_a fast suite does not discover this directory;
run it explicitly with unittest discover.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "include" / "strata" / "program" / "tier_b_export.hpp"
GENERATE = ROOT / "src" / "program" / "generate.cpp"


def _have_gxx() -> bool:
    return shutil.which("g++") is not None


class TestHeaderSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = HEADER.read_text(encoding="utf-8")

    def test_schema_fields(self):
        for token in ("# tier-b native window export v=1", "# vocab", "# max_windows",
                      "# max_row_bytes", "rows_offset", "rows_bytes", "picks=", "natural",
                      "# complete 1", "# truncation 1", "# export_error"):
            self.assertIn(token, self.text)

    def test_binary_magic_and_header(self):
        self.assertIn('"TBE1"', self.text)
        self.assertIn("kHeaderBytes = 32", self.text)
        self.assertIn("write_rows_header", self.text)

    def test_checked_byte_arithmetic_and_finite(self):
        for s in ("vocab byte overflow", "row byte overflow", "INT64_MAX", "std::isfinite"):
            self.assertIn(s, self.text)

    def test_fail_closed_footer_and_exclusive_fresh_dir(self):
        for s in ("complete && !failed_", "mkdir(dir.c_str(), 0700)", "O_EXCL", "O_NOFOLLOW",
                  "rows must be durable before success footer", "close_one"):
            self.assertIn(s, self.text)

    def test_whole_window_api_not_partial_rows_contract(self):
        for s in ("Pre-reserve one WHOLE window", "No partial success", "nbytes > max_row_bytes_ - row_bytes_"):
            self.assertIn(s, self.text)


class TestGenerateIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = GENERATE.read_text(encoding="utf-8")

    def test_flag_env_and_unsupported_modes(self):
        for s in ("--tier-b-export", "STRATA_TIER_B_EXPORT", "--tier-b-export unsupported",
                  "--spec-follow forced continuations", "native (IQ) pack"):
            self.assertIn(s, self.text)

    def test_readback_whole_window_after_commit_before_draft(self):
        self.assertIn("tier_b_export.can_write_window(T, err)", self.text)
        self.assertIn("ver.wait_commit(err)", self.text)
        self.assertIn("ver.copy_logits(t, logit_rows.data()", self.text)
        self.assertIn("tier_b_export.write_rows(logit_rows.data(), T", self.text)
        at = self.text.index("tier_b_export.write_rows(logit_rows.data(), T")
        self.assertLess(at, self.text.index("mtp.draft(T, outv.data()", at))

    def test_budget_exhaustion_skips_wait_hash_and_readback(self):
        start = self.text.index("if (tier_b_export.enabled() && !tier_b_export_failed)")
        precheck = self.text.index("tier_b_export.can_write_window(T, err)", start)
        prefix = self.text[start:precheck]
        self.assertNotIn("ver.wait_commit", prefix)
        self.assertNotIn("fnv1a(window.data()", prefix)
        body = self.text[precheck:self.text.index("} else if (tier_b_export.failed())", precheck)]
        self.assertLess(body.index("ver.wait_commit(err)"), body.index("fnv1a(window.data()"))
        self.assertLess(body.index("fnv1a(window.data()"), body.index("ver.copy_logits(t, logit_rows.data()"))

    def test_actual_emitted_and_picks_after_produced_loop(self):
        self.assertIn("produced_before", self.text)
        self.assertIn("emitted_actual", self.text)
        self.assertIn("outv.data(), tier_b_T, tier_b_status", self.text)
        self.assertLess(self.text.index("produced.push_back"), self.text.index("emitted_actual"))

    def test_failures_propagate_cli_nonzero_and_finish_checked(self):
        for s in ("tier_b_export_failed", "return 1;", "tier-b export failed", "!tier_b_export.finish(true"):
            self.assertIn(s, self.text)

    def test_finish_failure_is_returned_not_logged_and_ignored(self):
        at = self.text.index("if (tier_b_export.enabled())")
        block = self.text[at: self.text.index("std::printf", at)]
        self.assertIn("return 1;", block)

    def test_zero_window_lifecycle(self):
        self.assertIn("!(spec_pos > 0 && (int64_t) produced.size() < o.max_new && !ended)", self.text)
        self.assertIn("tier_b_export.finish(true, terr)", self.text)


_HARNESS = r'''
#define private public
#include "strata/program/tier_b_export.hpp"
#undef private
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <unistd.h>
using namespace strata::program;

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s\n", m); ++failures; } } while (0)

static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

int main() {
    char tpl[] = "/tmp/tbe_XXXXXX";
    std::string base = mkdtemp(tpl);

    // clean run: one T=2 window writes exactly two full rows and persists natural picks.
    {
        std::string dir = base + "/a", err;
        TierBExport e;
        CHECK(e.open(dir, 2, 24, 3, err), ("open: " + err).c_str());
        CHECK(e.can_write_window(2, err), "whole T=2 window should fit");
        float rows[6] = {1, 2, 3, 4, 5, 6};
        int64_t off = -1, bytes = -1;
        CHECK(e.write_rows(rows, 2, off, bytes, err), ("write_rows: " + err).c_str());
        CHECK(off == 32 && bytes == 24, "T=2 rows offset/bytes wrong");
        int32_t picks[2] = {11, 12};
        CHECK(e.write_window(0, 100, 2, 1, 1, 0xabc, 0xdef, off, bytes, picks, 2, "ok", err), "win0");
        CHECK(e.finish(true, err), ("finish: " + err).c_str());
        std::string m = slurp(dir + "/manifest.txt");
        CHECK(m.find("W 0 100 2 1 1") != std::string::npos, "manifest missing actual emitted count");
        CHECK(m.find("natural picks=11,12") != std::string::npos, "manifest missing natural picks");
        CHECK(m.find("# complete 1") != std::string::npos, "manifest missing complete");
        std::string r = slurp(dir + "/rows.bin");
        CHECK(r.size() == 32 + 24, "rows.bin must contain header plus both T rows");
        CHECK(std::memcmp(r.data(), "TBE1", 4) == 0, "rows.bin magic wrong");
    }

    // zero-window lifecycle: a supported run that produces no verify windows still has a complete artifact.
    {
        std::string dir = base + "/zero", err;
        TierBExport e;
        CHECK(e.open(dir, 4, 64, 3, err), "open zero");
        CHECK(e.finish(true, err), ("finish zero: " + err).c_str());
        CHECK(slurp(dir + "/manifest.txt").find("# complete 1") != std::string::npos, "zero-window missing complete");
        CHECK(slurp(dir + "/rows.bin").size() == 32, "zero-window rows should contain only header");
    }

    // budget precheck: if the whole T window does not fit, no rows are written and no partial W appears.
    {
        std::string dir = base + "/b", err;
        TierBExport e;
        CHECK(e.open(dir, 4, 12, 3, err), "open b");
        CHECK(!e.can_write_window(2, err) && e.truncated(), "T=2 should truncate before readback");
        float rows[6] = {1,2,3,4,5,6};
        int64_t off = 0, bytes = 0;
        CHECK(e.write_rows(rows, 2, off, bytes, err), "write_rows after truncation is no-op");
        CHECK(e.finish(true, err), "finish b");
        CHECK(slurp(dir + "/manifest.txt").find("W ") == std::string::npos, "truncated precheck wrote orphan W");
        CHECK(slurp(dir + "/rows.bin").size() == 32, "truncated precheck wrote orphan rows");
    }

    // non-finite full rows fail closed and suppress completion.
    {
        std::string dir = base + "/c", err;
        TierBExport e;
        CHECK(e.open(dir, 4, 64, 2, err), "open c");
        CHECK(e.can_write_window(1, err), "window c fits");
        float row[2] = {1.0f, 0.0f / 0.0f};
        int64_t off = 0, bytes = 0;
        CHECK(!e.write_rows(row, 1, off, bytes, err) && e.failed(), "NaN row should fail");
        int32_t pick[1] = {7};
        CHECK(e.write_window(0, 1, 1, 0, 0, 0, 0, -1, 0, pick, 1, "io_error", err), "record failed W");
        CHECK(!e.finish(true, err), "finish should report failed export");
        CHECK(slurp(dir + "/manifest.txt").find("# complete 1") == std::string::npos, "complete despite NaN");
    }

    // overflow is checked before multiplication/allocation.
    {
        std::string dir = base + "/d", err;
        TierBExport e;
        CHECK(e.open(dir, 4, 64, (INT64_MAX / 2), err), "open d");
        CHECK(!e.can_write_window(2, err) && e.failed(), "vocab*4 overflow should fail");
    }

    // path collision: existing artifact directory is refused and not truncated.
    {
        std::string dir = base + "/collision", err;
        mkdir(dir.c_str(), 0700);
        { std::ofstream f(dir + "/manifest.txt"); f << "keep"; }
        TierBExport e;
        CHECK(!e.open(dir, 4, 64, 4, err), "existing dir should be refused");
        CHECK(slurp(dir + "/manifest.txt") == "keep", "collision truncated prior file");
    }

    // footer/close failure: rows close must fail before a success manifest footer can be written.
    {
        std::string dir = base + "/closefail", err;
        TierBExport e;
        CHECK(e.open(dir, 4, 64, 2, err), "open closefail");
        CHECK(::close(::fileno(e.rows_)) == 0, "forced close rows fd");
        CHECK(!e.finish(true, err), "finish should fail when rows close fails");
        CHECK(slurp(dir + "/manifest.txt").find("# complete 1") == std::string::npos, "complete despite rows close failure");
    }

    std::fprintf(stderr, "OK failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
'''


@unittest.skipUnless(_have_gxx(), "g++ not available")
class TestCompileRunHarness(unittest.TestCase):
    def test_harness(self):
        tmp = tempfile.mkdtemp(prefix="tier_b_export_")
        try:
            (Path(tmp) / "main.cpp").write_text(_HARNESS, encoding="utf-8")
            exe = Path(tmp) / "proof"
            cmd = ["g++", "-std=c++17", "-I", str(ROOT / "include"),
                   "-Wall", "-Wextra", "-Werror", str(Path(tmp) / "main.cpp"), "-o", str(exe)]
            cp = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(cp.returncode, 0, f"compile failed:\n{cp.stderr}")
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, f"harness failed:\n{run.stdout}\n{run.stderr}")
            self.assertIn("OK failures=0", run.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
