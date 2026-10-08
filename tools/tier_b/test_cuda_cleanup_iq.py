"""Offline sta-osh.22 fault injection against the COMPLETE iq_multi_parity.cpp.

Compiles with g++, real Strata API headers, and an in-memory fake CUDA runtime.
No CUDA SDK, driver, GPU, model, downloads, or source rewriting. The kernels below
are deliberately synthetic: this proves lifecycle/error handling, NOT GPU parity.
The submit stub requires the caller's checked-error policy before injecting a
kernel exception; actual IQ host policy/default exits are tested separately.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

CUDA = r'''
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <new>
#include <string>
#include <vector>
struct FakeStream { bool capture = false; std::vector<std::function<void()>> calls; };
struct FakeGraph { std::vector<std::function<void()>> calls; };
struct FakeExec { std::vector<std::function<void()>> calls; };
struct FakeEvent {};
using cudaStream_t = FakeStream*;
using cudaGraph_t = FakeGraph*;
using cudaGraphExec_t = FakeExec*;
using cudaEvent_t = FakeEvent*;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorUnknown = 999;
constexpr int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
constexpr int cudaStreamCaptureModeThreadLocal = 0;
namespace iq_fake {
inline std::string operation;
inline int occurrence = 1;
inline bool after_acquire = false, injected = false, cleanup_errors = false;
inline bool fail_new = false, host_failure = false;
inline std::map<std::string, int> counts;
inline std::map<void*, size_t> allocations;
inline int streams = 0, graphs = 0, execs = 0, events = 0, captures = 0;
inline bool hit(const char* name) {
    const int n = ++counts[name];
    if (!injected && operation == name && n == occurrence) { injected = true; return true; }
    return false;
}
inline int release_error(bool hit) { return hit || (cleanup_errors && injected) ? 999 : 0; }
inline void submit(void* stream, const std::function<void()>& call) {
    if (hit("kernel")) {
        if (!strata::kernels::iq_host_errors_throw()) {
            std::fprintf(stderr, "injected kernel error without caller opt-in\n"); std::exit(1);
        }
        throw std::runtime_error("injected kernel exception");
    }
    auto* s = static_cast<cudaStream_t>(stream);
    if (s->capture) s->calls.push_back(call); else call();
}
}
inline const char* cudaGetErrorString(int) { return "iq-fake-injected"; }
inline int cudaMalloc(void** p, size_t n) {
    bool bad = iq_fake::hit("malloc");
    if (bad && !iq_fake::after_acquire) return 999;
    *p = std::malloc(n);
    if (!*p) return 999;
    iq_fake::allocations[*p] = n;
    // Throw at the next HOST allocation, e.g. blobs.push_back after a device upload.
    if (iq_fake::hit("host_after_malloc")) iq_fake::fail_new = true;
    return bad ? 999 : 0;
}
template<class T> inline int cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
inline int cudaFree(void* p) {
    bool bad = iq_fake::hit("free");
    if (p && iq_fake::allocations.erase(p) != 1) std::abort();
    std::free(p);
    return iq_fake::release_error(bad);
}
inline int cudaMemset(void* p, int v, size_t n) {
    if (iq_fake::hit("memset")) return 999;
    std::memset(p, v, n); return 0;
}
inline int cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t) {
    if (iq_fake::hit("memset_async")) return 999;
    std::memset(p, v, n); return 0;
}
inline int cudaMemcpy(void* d, const void* s, size_t n, int kind) {
    if (iq_fake::hit(kind == cudaMemcpyDeviceToHost ? "readback" : "upload")) return 999;
    if (n) std::memcpy(d, s, n); return 0;
}
inline int cudaStreamCreate(cudaStream_t* s) {
    bool bad = iq_fake::hit("stream_create");
    if (bad && !iq_fake::after_acquire) return 999;
    *s = new FakeStream; ++iq_fake::streams; return bad ? 999 : 0;
}
inline int cudaStreamDestroy(cudaStream_t s) {
    bool bad = iq_fake::hit("stream_destroy");
    if (s->capture) std::abort();
    delete s; --iq_fake::streams; return iq_fake::release_error(bad);
}
inline int cudaStreamSynchronize(cudaStream_t) { return iq_fake::hit("sync") ? 999 : 0; }
inline int cudaStreamBeginCapture(cudaStream_t s, int) {
    if (iq_fake::hit("capture_begin")) return 999;
    s->capture = true; ++iq_fake::captures; return 0;
}
inline int cudaStreamEndCapture(cudaStream_t s, cudaGraph_t* graph) {
    bool bad = iq_fake::hit("capture_end");
    if (!s->capture) std::abort();
    s->capture = false; --iq_fake::captures;
    if (!bad || iq_fake::after_acquire) {
        *graph = new FakeGraph{std::move(s->calls)}; ++iq_fake::graphs;
    }
    s->calls.clear(); return bad ? 999 : 0;
}
inline int cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long) {
    bool bad = iq_fake::hit("instantiate");
    if (bad && !iq_fake::after_acquire) return 999;
    *exec = new FakeExec{graph->calls}; ++iq_fake::execs; return bad ? 999 : 0;
}
inline int cudaGraphDestroy(cudaGraph_t g) {
    bool bad = iq_fake::hit("graph_destroy");
    delete g; --iq_fake::graphs; return iq_fake::release_error(bad);
}
inline int cudaGraphExecDestroy(cudaGraphExec_t g) {
    bool bad = iq_fake::hit("exec_destroy");
    delete g; --iq_fake::execs; return iq_fake::release_error(bad);
}
inline int cudaGraphLaunch(cudaGraphExec_t g, cudaStream_t) {
    if (iq_fake::hit("replay")) return 999;
    for (const auto& f : g->calls) f(); return 0;
}
inline int cudaEventCreate(cudaEvent_t* e) {
    bool bad = iq_fake::hit("event_create");
    if (bad && !iq_fake::after_acquire) return 999;
    *e = new FakeEvent; ++iq_fake::events; return bad ? 999 : 0;
}
inline int cudaEventDestroy(cudaEvent_t e) {
    bool bad = iq_fake::hit("event_destroy");
    delete e; --iq_fake::events; return iq_fake::release_error(bad);
}
inline int cudaEventRecord(cudaEvent_t, cudaStream_t) { return iq_fake::hit("event_record") ? 999 : 0; }
inline int cudaEventSynchronize(cudaEvent_t) { return iq_fake::hit("event_sync") ? 999 : 0; }
inline int cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
    if (iq_fake::hit("elapsed")) return 999;
    *ms = 1.0f; return 0;
}
'''

PROBE = r'''
#include <stdexcept>
#define main iq_cleanup_program_main
#include "src/kernels/iq_multi_parity.cpp"
#undef main

// Fail host vector growth once, without replacing/editing any production source.
void* operator new(size_t n) {
    if (iq_fake::fail_new) {
        iq_fake::fail_new = false; iq_fake::host_failure = true; throw std::bad_alloc();
    }
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

namespace strata::kernels {
bool fake_layout = true, fake_old = true;
thread_local bool fake_throw = false;
bool iq_host_errors_throw() noexcept { return fake_throw; }
void iq_set_host_errors_throw(bool enabled) noexcept { fake_throw = enabled; }
bool native_grouped_no_sub16_gu() { return fake_layout; }
void native_grouped_set_no_sub16_gu(bool v) { fake_layout = v; }
bool iq_old_kernels() { return fake_old; }
void iq_set_old_kernels(bool v) { fake_old = v; }
// Compact valid synthetic storage (NOT the GGUF formats); enough bytes for the source's scale fields.
size_t iq_row_bytes(int t, int64_t n) noexcept {
    return t == 20 ? n / 32 * 2 : t == 42 ? n / 64 * 2 : n / 256 * 64;
}
NativeExpertLayout native_expert_layout(int gu, int dt, int64_t H, int64_t FF) {
    auto L = NativeExpertLayout{gu, dt, H, FF};
    L.gu_row = iq_row_bytes(gu, H); L.d_row = iq_row_bytes(dt, FF);
    L.up_off = FF * L.gu_row; L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + H * L.d_row;
    if (iq_fake::hit("blob_mismatch") && iq_fake::counts["blob_mismatch"] == 2) ++L.bytes;
    return L;
}
bool native_expert_supported(int, int, int64_t, int64_t) noexcept { return true; }
size_t native_expert_scratch_bytes(int64_t cap, int64_t FF) {
    auto align = [](size_t n) { return (n + 255) & ~size_t(255); };
    return 3 * align(cap * FF * 4) + align(cap * (FF / 32) * 36);
}
void iq_dequant_f32(int, const void*, int64_t n, float* dst, void* stream) {
    iq_fake::submit(stream, [=] { std::fill(dst, dst + n, 0.f); });
}
void iq_mmvq(int, const void*, const void*, float* dst, int, int out, int cols, void* stream) {
    iq_fake::submit(stream, [=] { std::fill(dst, dst + out * cols, 0.f); });
}
void quantize_q8_1_rows(const float* x, int64_t rows, int64_t n, void* dst, void* stream) {
    iq_fake::submit(stream, [=] {
        auto* q = static_cast<uint8_t*>(dst);
        for (int64_t b = 0; b < rows * n / 32; ++b) {
            float lane[32], amax = 0.f;
            for (int i = 0; i < 32; ++i) { lane[i] = x[b * 32 + i]; amax = std::max(amax, std::fabs(lane[i])); }
            for (int o = 16; o > 0; o >>= 1) {
                float next[32]; for (int i = 0; i < 32; ++i) next[i] = lane[i] + lane[i ^ o];
                std::memcpy(lane, next, sizeof lane);
            }
            float d = std::min(amax / 127.f, 65504.f), sum = std::clamp(lane[0], -65504.f, 65504.f);
            uint16_t dh = f16_from_f32(d), sh = f16_from_f32(sum);
            std::memcpy(q + b * 36, &dh, 2); std::memcpy(q + b * 36 + 2, &sh, 2);
            for (int i = 0; i < 32; ++i)
                q[b * 36 + 4 + i] = (uint8_t)(int8_t)(amax == 0 ? 0 : std::clamp(std::round(x[b * 32 + i] / d), -127.f, 127.f));
        }
    });
}
void native_quantize_q8_1(const float* x, void* q, int n, int rows, void* s) { quantize_q8_1_rows(x, rows, n, q, s); }
void native_swiglu_quantize_q8_1(const float*, const float* up, void* q, int n, int rows, void* s) {
    std::vector<float> x(n * rows); for (size_t i = 0; i < x.size(); ++i) x[i] = up[i] * 32.f;
    quantize_q8_1_rows(x.data(), rows, n, q, s);
}
void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long*, const int32_t* start,
                          const int32_t* ng, const int32_t* dst, const int32_t*, int64_t, int64_t cap,
                          const void* x, void* scratch, float* out, void* stream, int64_t) {
    iq_fake::submit(stream, [=] {
        const auto* q = static_cast<const uint8_t*>(x);
        const float v = 1.f + q[0]; // changed-input graph replay must read the CURRENT bytes
        const size_t fa = (cap * L.n_ff * 4 + 255) & ~size_t(255);
        auto* scr = static_cast<uint8_t*>(scratch);
        for (int e = 0; e < start[*ng]; ++e) {
            for (int64_t r = 0; r < L.n_ff; ++r) {
                std::memcpy(scr + (e * L.n_ff + r) * 4, &v, 4);
                std::memcpy(scr + fa + (e * L.n_ff + r) * 4, &v, 4);
            }
            std::memset(scr + 3 * fa + e * (L.n_ff / 32) * 36, q[0], (L.n_ff / 32) * 36);
            std::fill(out + dst[e] * L.n_embd, out + (dst[e] + 1) * L.n_embd, v);
        }
    });
}
}

int main(int argc, char** argv) {
    if (argc != 8) return 90;
    const std::string action = argv[1];
    iq_fake::operation = argv[2]; iq_fake::occurrence = std::atoi(argv[3]);
    iq_fake::after_acquire = std::atoi(argv[4]); iq_fake::cleanup_errors = std::atoi(argv[5]);
    k::fake_layout = std::atoi(argv[6]); k::fake_old = !k::fake_layout;
    k::iq_set_host_errors_throw(std::atoi(argv[7]));
    const bool prior_layout = k::fake_layout, prior_old = k::fake_old, prior_policy = k::iq_host_errors_throw();
    int result = 0;
    try {
        if (action == "main") {
            char arg[] = "iq_multi_parity"; char* args[] = {arg};
            result = iq_cleanup_program_main(1, args);
        } else {
            SelectorGuard mode;
            Stream stream("probe stream free"); ck(cudaStreamCreate(&stream.value), "stream");
            std::mt19937 rng(18);
            if (action == "layout") check_sub16_layout(stream);
            else if (action == "mmvq") check_mmvq(18, 256, 5, stream, rng);
            else if (action == "grouped") check_grouped(18, 20, 256, 64, stream, rng);
            else if (action == "q8") check_q8_1_finite(stream, rng);
            else if (action == "events") time_ms(stream, 2, [] {});
            else if (action == "bench") bench(stream, rng);
            else if (action == "construct") {
                Grouped first(18, 20, 256, 64, {1}, 2, rng);
                Grouped second(18, 20, 256, 64, {1}, 2, rng);
            } else return 91;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "probe caught: %s\n", e.what()); result = 2;
    }
    iq_fake::fail_new = false;
    if (g_cleanup_fail) result = 2;
    const bool clean = iq_fake::allocations.empty() && iq_fake::streams == 0 && iq_fake::graphs == 0 &&
        iq_fake::execs == 0 && iq_fake::events == 0 && iq_fake::captures == 0;
    const bool policy_restored = k::iq_host_errors_throw() == prior_policy;
    const bool restored = k::fake_layout == prior_layout && k::fake_old == prior_old && policy_restored;
    std::printf("IQ_CLEANUP result=%d injected=%d host_failure=%d allocations=%zu streams=%d graphs=%d execs=%d "
                "events=%d captures=%d restored=%d cleanup_errors=%d parity_failures=%d policy_restored=%d policy=%d\n",
                result, iq_fake::injected, iq_fake::host_failure, iq_fake::allocations.size(), iq_fake::streams,
                iq_fake::graphs, iq_fake::execs, iq_fake::events, iq_fake::captures, restored, g_cleanup_fail, g_fail,
                policy_restored, k::iq_host_errors_throw());
    if (!clean || !restored) return 92;
    if (iq_fake::operation != "none" && (!iq_fake::injected || result != 2)) return 93;
    if (iq_fake::operation == "none" && (result != 0 || g_fail != 0)) return 94;
    return 0;
}
'''


@unittest.skipUnless(shutil.which("g++"), "g++ required for offline IQ cleanup probe")
class TestIqCudaCleanup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="iq_cleanup_")
        cls.addClassCleanup(cls.temp.cleanup)
        tmp = Path(cls.temp.name)
        (tmp / "cuda_runtime.h").write_text(CUDA)
        (tmp / "iq_cleanup_probe.cpp").write_text(PROBE)
        cls.exe = tmp / "iq_cleanup_probe"
        cmd = ["g++", "-std=c++20", "-O1", "-I", str(tmp), "-I", str(ROOT / "include"),
               "-I", str(ROOT), str(tmp / "iq_cleanup_probe.cpp"), "-o", str(cls.exe)]
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if run.returncode:
            raise AssertionError(f"{cmd}\n{run.stdout}\n{run.stderr}")

    def probe(self, action, operation, occurrence=1, after=False, cleanup=False, prior=True, policy=False):
        env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
        run = subprocess.run([str(self.exe), action, operation, str(occurrence), str(int(after)),
                              str(int(cleanup)), str(int(prior)), str(int(policy))],
                             env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, f"{action}/{operation}/{occurrence}/{after}:\n{run.stdout}\n{run.stderr}")
        self.assertIn("allocations=0 streams=0 graphs=0 execs=0 events=0 captures=0 restored=1", run.stdout)
        self.assertIn(f"policy_restored=1 policy={int(policy)}", run.stdout)
        return run

    def test_layout_success_both_initial_selectors(self):
        for prior in (False, True):
            with self.subTest(prior=prior):
                run = self.probe("layout", "none", prior=prior)
                self.assertEqual(run.stdout.count("bitwise direct + A/B/empty graph replay"), 7)
                self.assertIn("parity_failures=0", run.stdout)

    def test_partial_grouped_construction(self):
        # Nine blobs + nine metadata/input/scratch/output allocations in the first native sub16 case.
        for n in range(1, 19):
            for op in ("malloc", "memset"):
                with self.subTest(op=op, n=n):
                    self.probe("layout", op, n)
        for n in (1, 5, 9, 10, 15, 18):
            with self.subTest(partial_handle=n):
                self.probe("layout", "malloc", n, after=True)

    def test_upload_and_direct_readback_sync(self):
        for op, n in (("upload", 1), ("upload", 10), ("upload", 17), ("readback", 1),
                      ("readback", 2), ("readback", 4), ("sync", 1), ("sync", 2),
                      ("sync", 3), ("memset_async", 1), ("memset_async", 3)):
            with self.subTest(op=op, n=n):
                self.probe("layout", op, n)

    def test_capture_and_partial_graph_handles(self):
        for op in ("capture_begin", "capture_end", "instantiate", "graph_destroy"):
            for n in (1, 2):  # second capture must also free the previously returned exec
                with self.subTest(op=op, n=n):
                    self.probe("layout", op, n)
        for op in ("capture_end", "instantiate"):
            for n in (1, 2):
                with self.subTest(op=op, n=n, after=True):
                    self.probe("layout", op, n, after=True)

    def test_kernel_exception_during_capture(self):
        # Quantizer + two direct launches + first capture, then the second capture.
        for n in (4, 5):
            with self.subTest(kernel=n):
                self.probe("layout", "kernel", n)

    def test_graph_replay_changed_input_and_empty_failures(self):
        for n in range(1, 7):  # both graphs: A, changed B, and empty
            with self.subTest(replay=n):
                self.probe("layout", "replay", n)
        for op, n in (("upload", 18), ("readback", 9), ("readback", 23), ("readback", 35),
                      ("sync", 4), ("sync", 9), ("sync", 13), ("exec_destroy", 1), ("exec_destroy", 2)):
            with self.subTest(op=op, n=n):
                self.probe("layout", op, n)

    def test_host_growth_failure_after_device_acquisition(self):
        for n in (1, 3):
            with self.subTest(n=n):
                run = self.probe("layout", "host_after_malloc", n)
                self.assertIn("host_failure=1", run.stdout)

    def test_blob_size_exception_with_prior_live_resources(self):
        self.probe("construct", "blob_mismatch", 2)

    def test_cleanup_errors_do_not_interrupt_unwind(self):
        for op in ("instantiate", "replay"):
            with self.subTest(op=op):
                run = self.probe("layout", op, 2, cleanup=True)
                self.assertIn("CUDA cleanup error", run.stderr)
                self.assertRegex(run.stdout, r"cleanup_errors=[1-9][0-9]*")
        self.probe("layout", "free", 1)  # cleanup-only failure must not become a successful exit

    def test_benchmark_events(self):
        for op, n in (("event_create", 1), ("event_create", 2), ("event_record", 1),
                      ("event_record", 2), ("event_sync", 1), ("elapsed", 1),
                      ("event_destroy", 1), ("event_destroy", 2)):
            with self.subTest(op=op, n=n):
                self.probe("events", op, n)
        self.probe("events", "event_create", 2, after=True)
        self.probe("bench", "event_create", 2)

    def test_inherited_checks_unwind(self):
        for action in ("mmvq", "grouped", "q8"):
            for op in ("malloc", "readback", "sync", "kernel"):
                with self.subTest(action=action, op=op):
                    self.probe(action, op, 2 if op == "malloc" else 1)

    def test_exceptions_restore_both_initial_selector_states(self):
        for prior in (False, True):
            for action, op, n in (("layout", "kernel", 4), ("layout", "instantiate", 2),
                                  ("layout", "replay", 3), ("mmvq", "kernel", 3),
                                  ("grouped", "kernel", 2), ("bench", "event_create", 2)):
                with self.subTest(prior=prior, action=action, op=op):
                    self.probe(action, op, n, prior=prior)

    def test_checked_policy_restored_for_both_prior_states(self):
        for policy in (False, True):
            for action, op, n in (("main", "none", 1), ("main", "kernel", 1), ("main", "stream_create", 1),
                                  ("layout", "kernel", 4), ("mmvq", "kernel", 1), ("grouped", "kernel", 2)):
                with self.subTest(policy=policy, action=action, op=op):
                    self.probe(action, op, n, policy=policy)

    def test_full_program_synthetic_success(self):
        run = self.probe("main", "none")
        self.assertIn("iq_multi_parity: 0 failures", run.stdout)
        self.assertIn("result=0", run.stdout)

    def test_top_level_failure_exit_after_unwind(self):
        for op in ("stream_create", "malloc", "sync", "instantiate", "replay"):
            with self.subTest(op=op):
                run = self.probe("main", op)
                self.assertIn("iq_multi_parity FAIL:", run.stderr)
                self.assertIn("result=2", run.stdout)
        self.probe("main", "stream_create", after=True)
        self.probe("events", "stream_destroy")


if __name__ == "__main__":
    unittest.main()
