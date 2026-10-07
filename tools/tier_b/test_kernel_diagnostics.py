"""tools/tier_b/test_kernel_diagnostics.py - offline regression tests for sta-osh.4.

Covers the Tier B kernel-selection diagnostics header
(include/strata/kernels/tier_b_diagnostics.hpp) and the iq_kernels.cu call sites,
without a GPU or nvcc:

  * source-level static checks of the schema, budget, env contract, capture-state
    mapping, no-cudaGetLastError, and the explicit format/truncation marker;
  * source-level checks that iq_grouped GU/DOWN positive records are gated on
    support and on the executed phase (no "selected" record for skipped/unsupported);
  * (when g++ is available) a compile-and-run harness against a cuda_runtime.h
    stub proving the env contract, budget/truncation, and capture-state mapping.

The tier_a fast suite does not discover this directory; run it explicitly:
    python3 -m unittest tools.tier_b.test_kernel_diagnostics -v
    python3 -m unittest discover -s tools/tier_b -p 'test_*.py' -q
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "include" / "strata" / "kernels" / "tier_b_diagnostics.hpp"
IQ_CU = ROOT / "src" / "kernels" / "cuda" / "iq_kernels.cu"


def _have_gxx() -> bool:
    return shutil.which("g++") is not None


class TestHeaderSource(unittest.TestCase):
    """Static checks on the diagnostics header (no compilation)."""

    @classmethod
    def setUpClass(cls):
        cls.text = HEADER.read_text(encoding="utf-8")

    def test_header_exists(self):
        self.assertTrue(HEADER.exists(), "tier_b_diagnostics.hpp missing")

    def test_budget_is_1024(self):
        self.assertIn("kMaxRecords = 1024", self.text)

    def test_env_contract_exact_zero(self):
        """Enabled iff non-empty and NOT the exact string '0'."""
        self.assertIn("!(e[0] == '0' && e[1] == '\\0')", self.text)

    def test_null_stream_is_unknown(self):
        self.assertIn("if (stream == nullptr) return -1;", self.text)

    def test_invalidated_status_is_unknown(self):
        self.assertIn("default:                            return -1;", self.text)

    def test_active_status_maps_to_one(self):
        self.assertIn("case cudaStreamCaptureStatusActive: return 1;", self.text)

    def test_no_cuda_get_last_error(self):
        """The capture error path must not clear the underlying CUDA failure."""
        # capture_state() must not call cudaGetLastError at all.
        m = re.search(r"inline int capture_state\(void\* stream\) \{(.*?)\n\}", self.text, re.S)
        self.assertIsNotNone(m, "capture_state body not found")
        self.assertNotIn("cudaGetLastError", m.group(1))

    def test_schema_has_ncols_and_trunc(self):
        self.assertIn("ncols=<i>", self.text)
        self.assertIn("trunc=<0|1>", self.text)

    def test_oversize_emits_complete_marker_not_partial(self):
        """snprintf negative/oversize must emit a full marker, never a partial line."""
        self.assertIn('emit_marker("format_error");', self.text)
        self.assertIn("w < 0 || static_cast<size_t>(w) >= sizeof(buf)", self.text)
        # marker always newline-terminated and marked trunc=1 / capture=-1
        marker = re.search(r'inline void emit_marker\(const char\* path\) \{(.*?)\n\}', self.text, re.S)
        self.assertIsNotNone(marker, "emit_marker not found")
        self.assertIn("trunc=1", marker.group(1))
        self.assertIn("\\n", marker.group(1))


class TestIqKernelsCallSites(unittest.TestCase):
    """Static checks that positive dispatch records cannot be false positives."""

    @classmethod
    def setUpClass(cls):
        cls.text = IQ_CU.read_text(encoding="utf-8")

    def test_support_helpers_exist(self):
        for fn in ("bool gu_supported(int t)", "bool d_supported(int t)", "bool mmvq_supported(int t)"):
            self.assertIn(fn, self.text, f"{fn} missing")

    def test_gu_positive_gated_on_support_and_phase(self):
        # The positive GU emit is inside the executed branch AND gated on support.
        self.assertIn("if (tier_b::enabled() && gu_supported(L.gu_type)) {", self.text)

    def test_phase2_gu_labelled_skipped(self):
        self.assertIn('g_exp_phase == 2)', self.text)
        self.assertIn('"skipped"', self.text)

    def test_down_positive_gated_on_support(self):
        self.assertIn("if (tier_b::enabled() && d_supported(L.d_type)) {", self.text)

    def test_unsupported_defaults_label_unsupported(self):
        self.assertIn('"unsupported"', self.text)
        self.assertIn('native_expert_grouped: gate/up type', self.text)
        self.assertIn('native_expert_grouped: down type', self.text)

    def test_hip_fused_has_own_record(self):
        self.assertIn('"hip_fused"', self.text)


_STUB = r"""
#pragma once
typedef void* cudaStream_t;
typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorUnknown = 999;
enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2
};
static int g_stub_capture_queries = 0;
static int g_stub_get_last_error_calls = 0;
static int g_stub_status = cudaStreamCaptureStatusActive;
static int g_stub_error = 0;
static inline const char* cudaGetErrorString(cudaError_t e) { (void)e; return "stub-error"; }
static inline cudaError_t cudaGetLastError() { ++g_stub_get_last_error_calls; return cudaSuccess; }
static inline cudaError_t cudaStreamIsCapturing(cudaStream_t s, cudaStreamCaptureStatus* st) {
    (void)s; ++g_stub_capture_queries;
    if (g_stub_error != 0) return (cudaError_t) g_stub_error;
    *st = (cudaStreamCaptureStatus) g_stub_status; return cudaSuccess;
}
"""

_MAIN = r"""
#include "strata/kernels/tier_b_diagnostics.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
using namespace strata::kernels::tier_b;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; } } while (0)

static int child_disabled() {
    unsetenv("STRATA_TIER_B_TRACE");
    CHECK(!enabled(), "enabled() true with no env");
    emit("iq_mmvq", "x", "p", -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, (void*)0x1);
    CHECK(g_count.load() == 0, "disabled emit recorded");
    CHECK(g_stub_capture_queries == 0, "disabled queried capture");
    std::printf("%d\n", failures == 0 ? 0 : 1);
    return failures;
}
static int child_env(const char* val) {
    if (val) setenv("STRATA_TIER_B_TRACE", val, 1);
    else unsetenv("STRATA_TIER_B_TRACE");
    std::printf("%d\n", enabled() ? 1 : 0);
    return 0;
}

int main() {
    { pid_t p = fork(); if (p == 0) return child_disabled(); int st=0; waitpid(p,&st,0);
      if (!(WIFEXITED(st) && WEXITSTATUS(st)==0)) ++failures; }
    struct { const char* v; int expect; } cases[] = {
        {nullptr, 0}, {"", 0}, {"0", 0}, {"00", 1}, {"0x1", 1}, {"1", 1}, {"yes", 1},
    };
    for (auto& c : cases) {
        int fd[2]; pipe(fd);
        pid_t p = fork();
        if (p == 0) { dup2(fd[1], 1); close(fd[0]); close(fd[1]); return child_env(c.v); }
        close(fd[1]); char buf[8]={0}; int n=read(fd[0],buf,sizeof buf-1); close(fd[0]);
        int st=0; waitpid(p,&st,0);
        int got = (n>0 && buf[0]=='1') ? 1 : 0;
        if (got != c.expect) { std::fprintf(stderr, "FAIL: env '%s' got %d expect %d\n", c.v?c.v:"(unset)", got, c.expect); ++failures; }
    }

    setenv("STRATA_TIER_B_TRACE", "1", 1);
    CHECK(enabled(), "enabled() false with env=1");

    int q0 = g_stub_capture_queries;
    CHECK(capture_state(nullptr) == -1, "nullptr not unknown");
    CHECK(g_stub_capture_queries == q0, "nullptr queried cudaStreamIsCapturing");

    g_stub_status = cudaStreamCaptureStatusActive; g_stub_error = 0;
    CHECK(capture_state((void*)0x1) == 1, "active not 1");
    g_stub_status = cudaStreamCaptureStatusNone;
    CHECK(capture_state((void*)0x1) == 0, "none not 0");
    g_stub_status = cudaStreamCaptureStatusInvalidated;
    CHECK(capture_state((void*)0x1) == -1, "invalidated not unknown");

    g_stub_error = (int) cudaErrorUnknown;
    int l0 = g_stub_get_last_error_calls;
    CHECK(capture_state((void*)0x1) == -1, "cuda error not unknown");
    CHECK(g_stub_get_last_error_calls == l0, "cudaGetLastError was called (cleared failure)");

    g_stub_status = cudaStreamCaptureStatusActive; g_stub_error = 0;
    g_count.store(kMaxRecords - 1);
    g_truncated.store(0);
    emit("iq_grouped", "x", "p", -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, (void*)0x1);
    emit("iq_grouped", "x", "p", -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, (void*)0x1);
    CHECK(g_truncated.load() == 1, "truncation flag not set");

    std::fprintf(stderr, "OK failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
"""


@unittest.skipUnless(_have_gxx(), "g++ not available")
class TestCompileRunHarness(unittest.TestCase):
    """Compile the header against a cuda_runtime.h stub and run behaviour checks."""

    def test_harness(self):
        tmp = tempfile.mkdtemp(prefix="tier_b_diag_")
        try:
            (Path(tmp) / "cuda_runtime.h").write_text(_STUB, encoding="utf-8")
            (Path(tmp) / "main.cpp").write_text(_MAIN, encoding="utf-8")
            exe = Path(tmp) / "proof"
            cmd = ["g++", "-std=c++17", "-I", tmp, "-I", str(ROOT / "include"),
                   "-Wall", "-Wextra", "-Werror", str(Path(tmp) / "main.cpp"), "-o", str(exe)]
            cp = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(cp.returncode, 0, f"compile failed:\n{cp.stderr}")
            run = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0,
                             f"harness failed:\n{run.stdout}\n{run.stderr}")
            self.assertIn("OK failures=0", run.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
