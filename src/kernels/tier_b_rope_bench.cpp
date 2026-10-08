// src/kernels/tier_b_rope_bench.cpp - paired fused norm+RoPE vs the split reference, after rope_parity.
//
//     tier_b_rope_bench [iters] [warmup] [samples] [t_lo] [t_hi] [--no-graph]
//
// What it measures (per shape, all on identical synthetic F32 inputs):
//   * the BITWISE identity gate first - `native_qsa_rms_norm_rope` must equal the split reference - and a
//     nonzero exit when any shape differs. A differing shape is NOT timed.
//   * direct-launch time of the fused kernel and of the split sequence, as alternating AB/BA pairs so
//     drift cancels, timed with CUDA events around a synchronized batch of iterations after a fixed warmup.
//
// Split reference per layout (the non-fused path in verify.cpp/mtp.cpp, no historical extras):
//   contiguous -> norm d_src into d_ref, then rope d_ref in place          (no device copy)
//   in-place   -> norm + rope both in place in d_src                      (genuine in-place)
//   q|gate     -> the genuine strided pack (cudaMemcpy2DAsync) + norm + rope in place (pack cost included)
//
// Fairness contract: the canonical input is re-uploaded BEFORE each timed arm (outside the CUDA-event
// window), so the in-place fused and split arms consume identical initial data; because fused == split
// bitwise, their recurrent in-place inputs stay matched.
//   * the same pair for warmed CUDA-graph replay (cudaGraphLaunch), reported SEPARATELY from direct launch;
//     graph capture and instantiate cost are measured separately with the host clock.
//   * a changed-input replay gate before timing: a fresh input/position set is uploaded and both graphs must
//     reproduce the fresh direct reference bit for bit (frozen graphs would silently re-play the old data).
//
// This is an isolated-kernel benchmark. None of its numbers is a steady-state model throughput claim.
#include "strata/kernels/rope.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Destructors must not throw or exit: attempt every release, preserving an earlier exception.
// main checks this flag AFTER all owners have unwound, including on the successful path.
void cleanup(cudaError_t e, const char* what, bool& failed) noexcept {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        failed = true;
    }
}

struct RopeTableMode {
    bool& cleanup_failed;
    bool had_value;
    std::string previous;

    explicit RopeTableMode(bool& failed) : cleanup_failed(failed) {
        const char* value = std::getenv("STRATA_ROPE_TABLE");
        had_value = value != nullptr;
        if (value) previous = value;
#ifdef _WIN32
        const int e = _putenv_s("STRATA_ROPE_TABLE", "1");
#else
        const int e = setenv("STRATA_ROPE_TABLE", "1", 1);
#endif
        if (e != 0) throw std::runtime_error("setenv STRATA_ROPE_TABLE failed");
    }
    RopeTableMode(const RopeTableMode&) = delete;
    RopeTableMode& operator=(const RopeTableMode&) = delete;
    ~RopeTableMode() noexcept {
#ifdef _WIN32
        const int e = _putenv_s("STRATA_ROPE_TABLE", had_value ? previous.c_str() : "");
#else
        const int e = had_value ? setenv("STRATA_ROPE_TABLE", previous.c_str(), 1) : unsetenv("STRATA_ROPE_TABLE");
#endif
        if (e != 0) {
            std::fprintf(stderr, "cleanup restore STRATA_ROPE_TABLE failed\n");
            cleanup_failed = true;
        }
    }
};

struct CudaResources {
    bool& cleanup_failed;
    std::vector<void*> allocations;
    cudaStream_t stream = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    const float* registered_cos = nullptr;

    explicit CudaResources(bool& failed) : cleanup_failed(failed) {}
    CudaResources(const CudaResources&) = delete;
    CudaResources& operator=(const CudaResources&) = delete;

    template <class T> T* allocate(size_t n) {
        T* p = nullptr;
        const auto e = cudaMalloc(&p, n * sizeof(T));
        // Retain any returned handle before checking the result, as with stream/event/graph creation.
        if (p) {
            try { allocations.push_back(p); }
            catch (...) { cleanup(cudaFree(p), "cleanup untracked allocation", cleanup_failed); throw; }
        }
        ck(e, "alloc");
        return p;
    }
    ~CudaResources() noexcept {
        if (stream) cleanup(cudaStreamSynchronize(stream), "cleanup synchronize", cleanup_failed);
        // Forget the benchmark's table before freeing its storage; native enable/scaling selectors
        // are never changed by this standalone program.
        try { K::rope_table_release(registered_cos); }
        catch (...) {
            std::fprintf(stderr, "cleanup release rope table failed\n");
            cleanup_failed = true;
        }
        if (e0) cleanup(cudaEventDestroy(e0), "cleanup event0", cleanup_failed);
        if (e1) cleanup(cudaEventDestroy(e1), "cleanup event1", cleanup_failed);
        if (stream) cleanup(cudaStreamDestroy(stream), "cleanup stream", cleanup_failed);
        for (void* p : allocations) cleanup(cudaFree(p), "cleanup allocation", cleanup_failed);
    }
};

// One scoped pair per shape: handles are owned from the instant CUDA writes them, even when
// capture/instantiate only partially succeeds. This owner dies before the stream and buffers.
struct CudaGraphs {
    cudaStream_t stream;
    bool& cleanup_failed;
    cudaGraph_t gF = nullptr, gS = nullptr;
    cudaGraphExec_t exF = nullptr, exS = nullptr;
    bool capturing = false;

    CudaGraphs(cudaStream_t cs, bool& failed) : stream(cs), cleanup_failed(failed) {}
    CudaGraphs(const CudaGraphs&) = delete;
    CudaGraphs& operator=(const CudaGraphs&) = delete;
    ~CudaGraphs() noexcept {
        if (capturing) {
            // Do not overwrite an already owned graph while aborting an invalidated capture.
            cudaGraph_t abandoned = nullptr;
            cleanup(cudaStreamEndCapture(stream, &abandoned), "cleanup end capture", cleanup_failed);
            if (abandoned) cleanup(cudaGraphDestroy(abandoned), "cleanup abandoned graph", cleanup_failed);
        }
        if (gF || gS || exF || exS)
            cleanup(cudaStreamSynchronize(stream), "cleanup graph synchronize", cleanup_failed);
        if (exF) cleanup(cudaGraphExecDestroy(exF), "cleanup execF", cleanup_failed);
        if (exS) cleanup(cudaGraphExecDestroy(exS), "cleanup execS", cleanup_failed);
        if (gF) cleanup(cudaGraphDestroy(gF), "cleanup gF", cleanup_failed);
        if (gS) cleanup(cudaGraphDestroy(gS), "cleanup gS", cleanup_failed);
    }
};

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

struct Series {
    std::vector<double> us;
    double med() const { return median_of(us); }
    double mn() const { return us.empty() ? 0.0 : *std::min_element(us.begin(), us.end()); }
    double mx() const { return us.empty() ? 0.0 : *std::max_element(us.begin(), us.end()); }
};

std::string csv(const Series& s) {
    std::string out;
    for (double v : s.us) {
        if (!out.empty()) out += ",";
        char b[32];
        std::snprintf(b, sizeof b, "%.2f", v);
        out += b;
    }
    return out;
}

}  // namespace

int benchmark_main(int argc, char** argv, bool& cleanup_failed) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int iters = 200, warmup = 50, samples = 5, t_lo = 1, t_hi = 8;
    bool no_graph = false;
    std::vector<int> pos_args;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--no-graph") { no_graph = true; continue; }
        if (a == "--help" || a == "-h") {
            std::printf("usage: tier_b_rope_bench [iters=200] [warmup=50] [samples=5] [t_lo=1] [t_hi=8] [--no-graph]\n");
            return 0;
        }
        char* end = nullptr;
        const long v = std::strtol(argv[i], &end, 10);
        if (end == argv[i] || *end != '\0') {
            std::fprintf(stderr, "usage: tier_b_rope_bench [iters] [warmup] [samples] [t_lo] [t_hi] [--no-graph]\n");
            return 2;
        }
        pos_args.push_back((int) v);
    }
    if (pos_args.size() > 5) {
        std::fprintf(stderr, "usage: tier_b_rope_bench [iters] [warmup] [samples] [t_lo] [t_hi] [--no-graph]\n");
        return 2;
    }
    if (!pos_args.empty()) iters = pos_args[0];
    if (pos_args.size() > 1) warmup = pos_args[1];
    if (pos_args.size() > 2) samples = pos_args[2];
    if (pos_args.size() > 3) t_lo = pos_args[3];
    if (pos_args.size() > 4) t_hi = pos_args[4];
    if (iters < 10 || iters > 20000 || warmup < 1 || warmup > 2000 || samples < 1 || samples > 50 ||
        t_lo < 1 || t_lo > 8 || t_hi < t_lo || t_hi > 8) {
        std::fprintf(stderr, "bounds: iters 10..20000, warmup 1..2000, samples 1..50, 1 <= t_lo <= t_hi <= 8\n");
        return 2;
    }

    // The table opt-in is cached at the first native call; enable it up front so the analytic path stays
    // analytic (no table registered) and the table path can be switched per shape by register/release.
    RopeTableMode table_mode(cleanup_failed);

    const bool usable = K::native_norm_rope_usable(128, 64) && K::native_norm_rope_usable(256, 64);
    if (!usable) {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(__HIPCC__)
        std::printf("tier_b_rope_bench SKIPPED: fused norm/RoPE disabled on this build (STRATA_NORM_ROPE=1 to opt in)\n");
        return 77;
#else
        std::fprintf(stderr, "tier_b_rope_bench FAILED: fused norm/RoPE disabled on CUDA; unset STRATA_NO_NORM_ROPE\n");
        return 1;
#endif
    }

    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "device properties");
    // cudaDeviceProp::clockRate was removed in CUDA 13; query the attribute (kHz) instead.
    int clock_khz = 0;
    ck(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0), "device clock rate");
    std::printf("tier_b_rope_bench: isolated kernel timings, NOT steady-state model throughput\n");
    std::printf("device: %s | SMs %d | clock %d MHz | mem %zu MiB | iters %d warmup %d samples %d T %d..%d\n",
                prop.name, prop.multiProcessorCount, clock_khz / 1000, (size_t) prop.totalGlobalMem >> 20,
                iters, warmup, samples, t_lo, t_hi);

    CudaResources res(cleanup_failed);
    ck(cudaStreamCreateWithFlags(&res.stream, cudaStreamNonBlocking), "stream");
    ck(cudaEventCreate(&res.e0), "event0");
    ck(cudaEventCreate(&res.e1), "event1");
    const cudaStream_t cs = res.stream;

    const int n_rot = 64, half = 32, heads = 24, npos = 2048;
    const float eps = 1e-6f;
    const K::RopeScaling none;                    // default: type None, freq_base 1e7, analytic identity

    // One device table shared by every table-mode shape (contents independent of hd/T/layout).
    std::vector<float> h_cos((size_t) npos * half), h_sin(h_cos.size());
    K::build_rope_table(n_rot, none, npos, h_cos.data(), h_sin.data());
    float* d_cos = res.allocate<float>(h_cos.size());
    float* d_sin = res.allocate<float>(h_sin.size());
    res.registered_cos = d_cos;
    ck(cudaMemcpy(d_cos, h_cos.data(), h_cos.size() * sizeof(float), cudaMemcpyHostToDevice), "upload cos");
    ck(cudaMemcpy(d_sin, h_sin.data(), h_sin.size() * sizeof(float), cudaMemcpyHostToDevice), "upload sin");

    // Max-size reusable buffers: hd 256, T 8 -> rows 192, stride 512.
    const int max_rows = 8 * heads, max_hd = 256, max_stride = 2 * max_hd;
    float* d_src = res.allocate<float>((size_t) max_rows * max_stride);
    float* d_gamma = res.allocate<float>(max_hd);
    float* d_ref = res.allocate<float>((size_t) max_rows * max_hd);
    float* d_fused = res.allocate<float>((size_t) max_rows * max_hd);
    int* d_pos = res.allocate<int>(max_rows);

    std::mt19937 rng(20261);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    auto time_us = [&](const auto& launch) {
        for (int i = 0; i < warmup; ++i) launch();
        ck(cudaEventRecord(res.e0, cs), "record start");
        for (int i = 0; i < iters; ++i) launch();
        ck(cudaEventRecord(res.e1, cs), "record end");
        ck(cudaEventSynchronize(res.e1), "sync timing");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, res.e0, res.e1), "elapsed");
        return 1e3 * ms / iters;                 // microseconds per launch
    };

    int failures = 0, cases = 0, timed_cases = 0;
    for (int hd : {128, 256}) {
        for (int T = t_lo; T <= t_hi; ++T) {
            const int rows = T * heads;
            for (int layout = 0; layout < 3; ++layout) {           // 0 contiguous, 1 in-place, 2 q|gate
                const int stride = layout == 2 ? 2 * hd : hd;
                const bool inplace = layout == 1;
                const char* layout_name = layout == 2 ? "q|gate" : inplace ? "inplace" : "contiguous";
                for (bool table : {false, true}) {
                    ++cases;
                    const size_t count = (size_t) rows * hd;
                    // Output of each method.  In-place writes both into d_src; the non-in-place split
                    // writes d_ref (norm src->ref, rope ref in place) and the fused writes d_fused.
                    float* split_out = inplace ? d_src : d_ref;
                    float* fused_out = inplace ? d_src : d_fused;

                    // ---- host fixtures: two distinct input/position sets (A for the gate, B for changed replay)
                    std::vector<float> xA((size_t) rows * stride), xB((size_t) rows * stride);
                    std::vector<float> gamma((size_t) hd);
                    std::vector<int> posA((size_t) rows), posB((size_t) rows);
                    for (int d = 0; d < hd; ++d) gamma[d] = 0.5f + std::fabs(gauss(rng));
                    for (int r = 0; r < rows; ++r) {
                        const int tok = r / heads;
                        for (int d = 0; d < stride; ++d) {
                            xA[(size_t) r * stride + d] = d < hd ? gauss(rng) : 20.0f + std::fabs(gauss(rng));
                            xB[(size_t) r * stride + d] = d < hd ? gauss(rng) : 20.0f + std::fabs(gauss(rng));
                        }
                        posA[r] = (tok * 53 + 1) % npos;
                        posB[r] = (tok * 89 + 7) % npos;
                    }
                    auto upload = [&](const std::vector<float>& x, const std::vector<int>& pos) {
                        ck(cudaMemcpyAsync(d_src, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, cs), "upload src");
                        ck(cudaMemcpyAsync(d_gamma, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice, cs), "upload gamma");
                        ck(cudaMemcpyAsync(d_pos, pos.data(), pos.size() * sizeof(int), cudaMemcpyHostToDevice, cs), "upload pos");
                    };

                    K::rope_table_release(d_cos);
                    if (table) {
                        K::rope_table_set(d_cos, d_sin, npos, none);
                        if (K::rope_table_for(none).cos != d_cos) {
                            throw std::runtime_error("table path did not select the registered table");
                        }
                    }

                    // ---- split and fused as one launch each (shared by the gate, capture and timing).
                    // The split reference is the non-fused caller path: contiguous has no device copy,
                    // in-place is genuinely in-place in d_src, and q|gate keeps only its necessary pack.
                    auto launch_split = [&] {
                        if (inplace) {
                            K::native_qsa_rms_norm_weighted(d_src, d_gamma, d_src, hd, rows, eps, cs);
                            K::native_rope_apply(d_src, d_src, rows, hd, n_rot, none, d_pos, cs);
                        } else if (layout == 2) {
                            ck(cudaMemcpy2DAsync(d_ref, (size_t) hd * sizeof(float), d_src, (size_t) stride * sizeof(float),
                                                 (size_t) hd * sizeof(float), (size_t) rows, cudaMemcpyDeviceToDevice, cs), "split pack");
                            K::native_qsa_rms_norm_weighted(d_ref, d_gamma, d_ref, hd, rows, eps, cs);
                            K::native_rope_apply(d_ref, d_ref, rows, hd, n_rot, none, d_pos, cs);
                        } else {
                            K::native_qsa_rms_norm_weighted(d_src, d_gamma, d_ref, hd, rows, eps, cs);
                            K::native_rope_apply(d_ref, d_ref, rows, hd, n_rot, none, d_pos, cs);
                        }
                    };
                    auto launch_fused = [&] {
                        K::native_qsa_rms_norm_rope(d_src, stride, d_gamma, fused_out, rows, hd, n_rot, eps, none, d_pos, cs);
                    };

                    // ---- 1. bitwise identity gate: parity must hold before anything is timed.  Each method
                    // starts from the SAME canonical input; the in-place split mutates d_src, so d_src is reset
                    // (outside any timed window) between the two methods.
                    std::vector<float> h_ref(count), h_fus(count);
                    upload(xA, posA);
                    launch_split();
                    ck(cudaStreamSynchronize(cs), "gate split sync");
                    ck(cudaMemcpy(h_ref.data(), split_out, count * sizeof(float), cudaMemcpyDeviceToHost), "read split");
                    upload(xA, posA);
                    launch_fused();
                    ck(cudaStreamSynchronize(cs), "gate fused sync");
                    ck(cudaMemcpy(h_fus.data(), fused_out, count * sizeof(float), cudaMemcpyDeviceToHost), "read fused");
                    bool identical = std::memcmp(h_ref.data(), h_fus.data(), count * sizeof(float)) == 0;
                    for (size_t i = 0; i < count; ++i) identical = identical && std::isfinite(h_ref[i]) && std::isfinite(h_fus[i]);
                    if (!identical) {
                        ++failures;
                        std::printf("T=%d hd=%d rows=%d %s %-9s: DIFFERS (not timed)\n",
                                    T, hd, rows, layout_name, table ? "table" : "analytic");
                        continue;   // parity fails stop timing for this shape
                    }

                    // ---- 2. capture both graphs, measuring capture/instantiate on the host clock
                    double capF = 0, capS = 0, instF = 0, instS = 0;
                    CudaGraphs graphs(cs, cleanup_failed);
                    auto& gF = graphs.gF;
                    auto& gS = graphs.gS;
                    auto& exF = graphs.exF;
                    auto& exS = graphs.exS;
                    if (!no_graph) {
                        upload(xA, posA);
                        auto capture = [&](const auto& launch, cudaGraph_t& g, double& cap_ms) {
                            auto t0 = std::chrono::steady_clock::now();
                            ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin capture");
                            graphs.capturing = true;
                            launch();
                            const auto e = cudaStreamEndCapture(cs, &g);
                            // EndCapture terminates capture even when it reports invalidation; do not
                            // attempt a second EndCapture during unwinding. Any output is already owned.
                            graphs.capturing = false;
                            ck(e, "end capture");
                            auto t1 = std::chrono::steady_clock::now();
                            cap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                        };
                        auto instantiate = [&](cudaGraph_t g, cudaGraphExec_t& ex, double& inst_ms) {
                            auto t0 = std::chrono::steady_clock::now();
                            ck(cudaGraphInstantiate(&ex, g, nullptr, nullptr, 0), "instantiate");
                            auto t1 = std::chrono::steady_clock::now();
                            inst_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                        };
                        capture(launch_fused, gF, capF);
                        capture(launch_split, gS, capS);
                        instantiate(gF, exF, instF);
                        instantiate(gS, exS, instS);

                        // ---- 3. changed-input replay gate: a fresh set must reproduce the fresh direct reference.
                        // Reset d_src before EACH method so every method does equivalent work on identical input.
                        upload(xB, posB);
                        launch_split();                                   // direct reference on set B
                        ck(cudaStreamSynchronize(cs), "changed direct sync");
                        ck(cudaMemcpy(h_ref.data(), split_out, count * sizeof(float), cudaMemcpyDeviceToHost), "read refB");
                        upload(xB, posB);
                        ck(cudaGraphLaunch(exS, cs), "graph split launch");
                        ck(cudaStreamSynchronize(cs), "graph split sync");
                        std::vector<float> h_g(count);
                        ck(cudaMemcpy(h_g.data(), split_out, count * sizeof(float), cudaMemcpyDeviceToHost), "read graph split");
                        upload(xB, posB);
                        ck(cudaGraphLaunch(exF, cs), "graph fused launch");
                        ck(cudaStreamSynchronize(cs), "graph fused sync");
                        ck(cudaMemcpy(h_fus.data(), fused_out, count * sizeof(float), cudaMemcpyDeviceToHost), "read graph fused");
                        const bool replay_ok = std::memcmp(h_g.data(), h_ref.data(), count * sizeof(float)) == 0 &&
                                               std::memcmp(h_fus.data(), h_ref.data(), count * sizeof(float)) == 0;
                        if (!replay_ok) {
                            ++failures;
                            std::printf("T=%d hd=%d rows=%d %s %-9s: CHANGED-INPUT GRAPH REPLAY DIFFERS (not timed)\n",
                                        T, hd, rows, layout_name, table ? "table" : "analytic");
                            continue;  // the graph pair unwinds before the next shape
                        }
                    }

                    // ---- 4. paired AB/BA timing: direct fused/split and warmed graph replay.
                    // Fairness contract: the canonical input (xB) is re-uploaded BEFORE EACH arm, outside the
                    // CUDA-event window.  Both in-place arms mutate d_src, so each starts from identical data;
                    // because fused == split bitwise (gate 1), their recurrent in-place inputs stay matched.
                    Series dF, dS, grF, grS;
                    for (int s = 0; s < samples; ++s) {
                        const bool ab_first = (s % 2) == 0;
                        if (ab_first) {
                            upload(xB, posB); dF.us.push_back(time_us(launch_fused));
                            upload(xB, posB); dS.us.push_back(time_us(launch_split));
                        } else {
                            upload(xB, posB); dS.us.push_back(time_us(launch_split));
                            upload(xB, posB); dF.us.push_back(time_us(launch_fused));
                        }
                        if (!no_graph) {
                            if (ab_first) {
                                upload(xB, posB); grF.us.push_back(time_us([&] { ck(cudaGraphLaunch(exF, cs), "launch gF"); }));
                                upload(xB, posB); grS.us.push_back(time_us([&] { ck(cudaGraphLaunch(exS, cs), "launch gS"); }));
                            } else {
                                upload(xB, posB); grS.us.push_back(time_us([&] { ck(cudaGraphLaunch(exS, cs), "launch gS"); }));
                                upload(xB, posB); grF.us.push_back(time_us([&] { ck(cudaGraphLaunch(exF, cs), "launch gF"); }));
                            }
                        }
                    }
                    ++timed_cases;
                    std::printf("T=%d hd=%d rows=%d %-9s %-9s: IDENTICAL\n", T, hd, rows, layout_name, table ? "table" : "analytic");
                    std::printf("  direct  fused %8.2f us [%7.2f..%7.2f] | split %8.2f us [%7.2f..%7.2f]\n",
                                dF.med(), dF.mn(), dF.mx(), dS.med(), dS.mn(), dS.mx());
                    if (!no_graph)
                        std::printf("  graph   fused %8.2f us [%7.2f..%7.2f] | split %8.2f us [%7.2f..%7.2f] | "
                                    "capture %.3f/%.3f ms instantiate %.3f/%.3f ms\n",
                                    grF.med(), grF.mn(), grF.mx(), grS.med(), grS.mn(), grS.mx(), capF, capS, instF, instS);
                    std::printf("  raw directF:%s\n  raw directS:%s\n", csv(dF).c_str(), csv(dS).c_str());
                    if (!no_graph) std::printf("  raw graphF:%s\n  raw graphS:%s\n", csv(grF).c_str(), csv(grS).c_str());
                }
            }
        }
    }

    std::printf("tier_b_rope_bench: %d cases, %d timed, %d gate/replay failures\n", cases, timed_cases, failures);
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    bool cleanup_failed = false;
    try {
        const int result = benchmark_main(argc, argv, cleanup_failed);
        return cleanup_failed ? 2 : result;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tier_b_rope_bench FAILED: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "tier_b_rope_bench FAILED: unknown exception\n");
    }
    return 2;
}
