// src/kernels/iq_multi_parity.cpp - the IQ kernels that decode each weight part once for every
// column / entry, against the older kernels that decode it again per column.  Synthetic blocks, no model.
//
//     build/iq_multi_parity            the checks (a ctest; needs the GPU)
//     build/iq_multi_parity --bench    the checks, then old/new timings of both paths
//
// (1) iq_mmvq, every format it takes, ncols 1..8 and 11 (the > 8 split): the new output must be BITWISE equal to
//     the old one (iq_set_old_kernels), and within float rounding of a double reference built from the GPU's own
//     dequantizer (validated against gguf-py by iq_parity) and the dequantized q8_1 activations - a gross-error
//     guard, not the contract.
// (2) native_expert_grouped, every gate/up format with the down formats: groups of 0..11 entries (more than a pass
//     of GRP_NC), unused grid rows (cap_groups > groups), scattered destinations; bitwise, new vs old, the rows it
//     must not write included.
// (3) Sub16 layout parity: native_expert_grouped with STRATA_NO_SUB16_GU=0 (sub16-enabled, the default) against
//     STRATA_NO_SUB16_GU=1 (wider-warp fallback) for the IQ3_S-file's actual GU/down format pairs at 2560 x 640.
//     Selector set before capture and restored after; selector change gates both GU and down sub-layouts.
//     Covers group tails 1/2/3/4/5/8/11, empty groups, scattered dests; guard: untouched sentinel rows stay.
//     Direct and captured-graph changed-input replay agree bitwise (identity contract, not just tolerance).
//
// Random bytes are valid codes for every format here (all grid indices are in range); only the fp16 block scales
// are set, small enough that the grouped path's SwiGLU output keeps a finite fp16 q8_1 scale.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;
int g_cleanup_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + w + ": " + cudaGetErrorString(e));
}

// Cleanup must attempt every resource, even with a sticky CUDA error during unwind.  Do not throw from a
// destructor or hide cleanup errors on the success path: main checks this flag AFTER all owners are destroyed.
void cleanup(cudaError_t e, const char* w) noexcept {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup error in %s: %s\n", w, cudaGetErrorString(e));
        ++g_cleanup_fail;
    }
}

struct DeviceAllocations {
    std::vector<void*> owned;
    DeviceAllocations() = default;
    DeviceAllocations(const DeviceAllocations&) = delete;
    DeviceAllocations& operator=(const DeviceAllocations&) = delete;
    ~DeviceAllocations() noexcept {
        for (auto i = owned.rbegin(); i != owned.rend(); ++i)
            if (*i) cleanup(cudaFree(*i), "free allocation");
    }
    template<typename T>
    T* alloc(size_t n) {
        // Register storage BEFORE CUDA can acquire anything (or vector growth can throw).  The member owner
        // also unwinds if Grouped's constructor fails; its raw pointers below are only non-owning aliases.
        owned.push_back(nullptr);
        ck(cudaMalloc(&owned.back(), n * sizeof(T) + 256), "malloc");
        ck(cudaMemset(owned.back(), 0, n * sizeof(T) + 256), "memset");
        return static_cast<T*>(owned.back());
    }
};

template<typename T, auto Destroy>
struct CudaHandle {
    T value = nullptr;
    const char* what;
    explicit CudaHandle(const char* w) : what(w) {}
    CudaHandle(const CudaHandle&) = delete;
    CudaHandle& operator=(const CudaHandle&) = delete;
    CudaHandle(CudaHandle&& other) noexcept : value(std::exchange(other.value, nullptr)), what(other.what) {}
    CudaHandle& operator=(CudaHandle&& other) noexcept {
        if (this != &other) {
            if (value) cleanup(Destroy(value), what);
            value = std::exchange(other.value, nullptr);
            what = other.what;
        }
        return *this;
    }
    ~CudaHandle() noexcept { if (value) cleanup(Destroy(value), what); }
    operator T() const noexcept { return value; }
    void close() { if (value) ck(Destroy(std::exchange(value, nullptr)), what); }
};
using Stream = CudaHandle<cudaStream_t, cudaStreamDestroy>;
using Event = CudaHandle<cudaEvent_t, cudaEventDestroy>;
using Graph = CudaHandle<cudaGraph_t, cudaGraphDestroy>;
using GraphExec = CudaHandle<cudaGraphExec_t, cudaGraphExecDestroy>;

struct SelectorGuard {
    // This standalone caller owns/unwinds its CUDA resources; unaudited engine callers stay fail-closed.
    k::IqHostErrorScope errors;
    const bool layout = k::native_grouped_no_sub16_gu(), old = k::iq_old_kernels();
    ~SelectorGuard() noexcept {
        k::native_grouped_set_no_sub16_gu(layout);
        k::iq_set_old_kernels(old);
    }
};

struct CaptureGuard {
    cudaStream_t stream;
    bool active = false;
    explicit CaptureGuard(cudaStream_t s) : stream(s) {
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "layout capture begin");
        active = true;
    }
    ~CaptureGuard() noexcept {
        if (active) {
            Graph abandoned("abandoned layout graph free");
            cleanup(cudaStreamEndCapture(stream, &abandoned.value), "layout capture unwind");
        }
    }
    void end(Graph& graph) {
        const auto e = cudaStreamEndCapture(stream, &graph.value);
        active = false; // EndCapture also ends an invalidated capture; graph is owned even if it reports failure.
        ck(e, "layout capture end");
    }
};

const char* name_of(int t) {
    switch (t) {
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 29: return "IQ1_M";
        case 42: return "Q2_0";
        default: return "?";
    }
}
int block_values(int t) { return t == 20 ? 32 : t == 42 ? 64 : 256; }

// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 3);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        if (t == 29) {
            // IQ1_M: the fp16 scale is the top nibbles of the four scale words (bytes 48..55); nibble 3 carries the
            // sign and the exponent's top bits: 0x1 / 0x2 (or 0x9 / 0xA) keeps it in 2^-11 .. 2^-3
            const uint8_t nib = (uint8_t) ((sgn(rng) == 0 ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
            w[o + 55] = (uint8_t) ((w[o + 55] & 0x0F) | (nib << 4));
        } else {
            const uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));   // 2^-13 .. 2^-6
            std::memcpy(&w[o], &h, 2);
        }
    }
    return w;
}

std::vector<float> random_x(size_t n, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    return x;
}

// q8_1 blocks (fp16 scale, fp16 sum, 32 int8) back to floats
std::vector<double> dequant_q8_1(const std::vector<uint8_t>& q, size_t n) {
    std::vector<double> v(n);
    for (size_t b = 0; b < n / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, &q[b * 36], 2);
        const double d = k::f32_from_f16(dh);
        for (int i = 0; i < 32; ++i) v[b * 32 + i] = d * (int8_t) q[b * 36 + 4 + i];
    }
    return v;
}

// ------------------------------------------------------------------------------------------------ (1) iq_mmvq
void check_mmvq(int t, int n_in, int n_out, cudaStream_t s, std::mt19937& rng) {
    SelectorGuard mode;
    DeviceAllocations memory;
    const auto w = random_rows(t, n_out, n_in, rng);
    uint8_t* dw = memory.alloc<uint8_t>(w.size());
    ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    // the double reference's weights: the GPU dequantizer (iq_parity checks it against gguf-py)
    float* dwf = memory.alloc<float>((size_t) n_out * n_in);
    k::iq_dequant_f32(t, dw, (int64_t) n_out * n_in, dwf, s);
    std::vector<float> wf((size_t) n_out * n_in);
    ck(cudaMemcpy(wf.data(), dwf, wf.size() * 4, cudaMemcpyDeviceToHost), "wf");
    const int max_cols = 11;
    const auto x = random_x((size_t) max_cols * n_in, rng);
    float* dx = memory.alloc<float>(x.size());
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    const size_t xq_bytes = (size_t) max_cols * (n_in / 32) * 36;
    uint8_t* dxq = memory.alloc<uint8_t>(xq_bytes);
    k::quantize_q8_1_rows(dx, max_cols, n_in, dxq, s);
    std::vector<uint8_t> xq(xq_bytes);
    ck(cudaMemcpy(xq.data(), dxq, xq_bytes, cudaMemcpyDeviceToHost), "xq");
    const auto xd = dequant_q8_1(xq, (size_t) max_cols * n_in);
    float* dy_old = memory.alloc<float>((size_t) max_cols * n_out);
    float* dy_new = memory.alloc<float>((size_t) max_cols * n_out);
    std::vector<float> y_old((size_t) max_cols * n_out), y_new(y_old.size());
    const int cols[] = {1, 2, 3, 4, 5, 6, 7, 8, 11};
    int bad = 0;
    double worst = 0.0;
    for (int nc : cols) {
        ck(cudaMemset(dy_old, 0xFF, y_old.size() * 4), "memset");
        ck(cudaMemset(dy_new, 0xFF, y_new.size() * 4), "memset");
        k::iq_set_old_kernels(true);
        k::iq_mmvq(t, dw, dxq, dy_old, n_in, n_out, nc, s);
        k::iq_set_old_kernels(false);
        k::iq_mmvq(t, dw, dxq, dy_new, n_in, n_out, nc, s);
        ck(cudaStreamSynchronize(s), "sync");
        ck(cudaMemcpy(y_old.data(), dy_old, y_old.size() * 4, cudaMemcpyDeviceToHost), "y_old");
        ck(cudaMemcpy(y_new.data(), dy_new, y_new.size() * 4, cudaMemcpyDeviceToHost), "y_new");
        size_t diff = 0;
        for (size_t i = 0; i < y_old.size(); ++i) diff += std::memcmp(&y_old[i], &y_new[i], 4) != 0;
        double num = 0, den = 0;
        bool finite = true;
        for (int c = 0; c < nc; ++c)
            for (int r = 0; r < n_out; ++r) {
                double acc = 0;
                for (int i = 0; i < n_in; ++i) acc += (double) wf[(size_t) r * n_in + i] * xd[(size_t) c * n_in + i];
                const float g = y_new[(size_t) c * n_out + r];
                finite = finite && std::isfinite(g);
                num += std::fabs(g - acc);
                den += std::fabs(acc);
            }
        const double rel = num / (den + 1e-30);
        worst = std::max(worst, rel);
        if (diff != 0 || !finite || !(rel < 1e-2)) {
            std::printf("  %-8s %5d x %5d  ncols %2d: %zu outputs differ from the old kernel, ref rel %.2e%s  FAIL\n",
                        name_of(t), n_out, n_in, nc, diff, rel, finite ? "" : ", non-finite");
            ++bad;
        }
    }
    std::printf("%-8s type %2d  %5d x %5d  iq_mmvq ncols 1..8, 11: %s (ref rel <= %.2e)\n", name_of(t), t, n_out, n_in,
                bad ? "FAIL" : "bitwise equal to the old kernel", worst);
    g_fail += bad;
}

// ---------------------------------------------------------------------------------- (2) native_expert_grouped
struct Grouped {
    DeviceAllocations memory;
    k::NativeExpertLayout L;
    int n_groups = 0, n_ent = 0, cap_groups = 0, cap_ent = 0, n_tok = 0;
    std::vector<uint8_t*> blobs;
    unsigned long long* dptr = nullptr;
    int32_t *dstart = nullptr, *dn = nullptr, *ddst = nullptr, *dtok = nullptr;
    float* dx = nullptr;
    uint8_t *dxq = nullptr, *dscr = nullptr;
    float* dout = nullptr;
    size_t out_floats = 0;
    std::vector<bool> written_rows;

    Grouped(int gu, int dt, int64_t H, int64_t FF, const std::vector<int>& counts, int tokens, std::mt19937& rng,
            bool holes = false) {
        L = k::native_expert_layout(gu, dt, H, FF);
        n_groups = (int) counts.size();
        cap_groups = n_groups + 2;              // the extra grid rows must exit
        n_tok = tokens;
        std::vector<int32_t> start(1, 0), tok, dst;
        for (int c : counts) start.push_back(start.back() + c);
        n_ent = start.back();
        cap_ent = n_ent + 3;
        std::uniform_int_distribution<int> tk(0, tokens - 1);
        for (int e = 0; e < n_ent; ++e) tok.push_back(tk(rng));
        dst.resize(n_ent);
        std::iota(dst.begin(), dst.end(), 0);
        // Layout gate: leave an interior hole as well as a leading and trailing untouched row.
        if (holes) for (int e = 0; e < n_ent; ++e) dst[e] = 1 + e + (e >= n_ent / 2);
        std::shuffle(dst.begin(), dst.end(), rng);
        written_rows.assign(cap_ent, false);
        for (int d : dst) written_rows[d] = true;
        std::vector<unsigned long long> ptr;
        for (int g = 0; g < n_groups; ++g) {
            // one blob per group: gate rows | up rows (format gu, H values) | down rows (format dt, FF values)
            auto b = random_rows(gu, 2 * FF, H, rng);
            const auto d = random_rows(dt, H, FF, rng);
            b.insert(b.end(), d.begin(), d.end());
            if (b.size() != L.bytes)
                throw std::runtime_error("blob size " + std::to_string(b.size()) + " != " + std::to_string(L.bytes));
            uint8_t* db = memory.alloc<uint8_t>(b.size());
            ck(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice), "blob");
            blobs.push_back(db);
            ptr.push_back((unsigned long long) db);
        }
        dptr = memory.alloc<unsigned long long>(cap_groups);
        dstart = memory.alloc<int32_t>(cap_groups + 1);
        dn = memory.alloc<int32_t>(1);
        ddst = memory.alloc<int32_t>(cap_ent);
        dtok = memory.alloc<int32_t>(cap_ent);
        ck(cudaMemcpy(dptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice), "ptr");
        ck(cudaMemcpy(dstart, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "start");
        ck(cudaMemcpy(dn, &n_groups, 4, cudaMemcpyHostToDevice), "n");
        if (n_ent) {
            ck(cudaMemcpy(ddst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
            ck(cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "tok");
        }
        const auto x = random_x((size_t) tokens * H, rng);
        dx = memory.alloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        dxq = memory.alloc<uint8_t>((size_t) tokens * (H / 32) * 36);
        dscr = memory.alloc<uint8_t>(k::native_expert_scratch_bytes(cap_ent, FF));
        out_floats = (size_t) cap_ent * H;
        dout = memory.alloc<float>(out_floats);
    }
    void run(bool old, cudaStream_t s) {
        SelectorGuard mode;
        k::iq_set_old_kernels(old);
        k::quantize_q8_1_rows(dx, n_tok, L.n_embd, dxq, s);
        k::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, cap_groups, cap_ent, dxq, dscr, dout, s);
        k::iq_set_old_kernels(false);
    }
    std::vector<float> result(bool old, cudaStream_t s) {
        ck(cudaMemset(dout, 0xFF, out_floats * 4), "memset");      // NaN rows: must stay NaN where nothing writes
        ck(cudaMemset(dscr, 0, k::native_expert_scratch_bytes(cap_ent, L.n_ff)), "memset");
        run(old, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> o(out_floats);
        ck(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost), "out");
        return o;
    }
};

void check_grouped(int gu, int dt, int64_t H, int64_t FF, cudaStream_t s, std::mt19937& rng) {
    // 0..11 entries per group: one pass, a partial pass, two passes and three passes of GRP_NC = 4
    Grouped G(gu, dt, H, FF, {1, 3, 4, 5, 0, 8, 2, 11, 1}, 8, rng);
    const auto a = G.result(true, s), b = G.result(false, s);
    size_t diff = 0, written = 0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        diff += std::memcmp(&a[i], &b[i], 4) != 0;
        if (i < (size_t) G.n_ent * H) { ++written; finite = finite && std::isfinite(b[i]); }
    }
    const bool ok = diff == 0 && finite;
    std::printf("%-8s/%-7s %5lld x %4lld  native_expert_grouped, %d groups, %d entries: %s\n", name_of(gu),
                name_of(dt), (long long) H, (long long) FF, G.n_groups, G.n_ent,
                ok ? "bitwise equal to the old kernels" : "FAIL");
    if (!ok) {
        std::printf("  %zu of %zu floats differ%s\n", diff, a.size(), finite ? "" : ", non-finite outputs");
        ++g_fail;
    }
}

// ------------------------------------------------------------------------------------------------ --bench
float time_ms(cudaStream_t s, int it, const auto& fn) {
    Event e0("bench event0 free"), e1("bench event1 free");
    ck(cudaEventCreate(&e0.value), "bench event0");
    ck(cudaEventCreate(&e1.value), "bench event1");
    fn();
    ck(cudaEventRecord(e0, s), "bench record0");
    for (int i = 0; i < it; ++i) fn();
    ck(cudaEventRecord(e1, s), "bench record1");
    ck(cudaEventSynchronize(e1), "bench");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, e0, e1), "bench elapsed");
    e0.close();
    e1.close();
    return ms / it;
}

void bench(cudaStream_t s, std::mt19937& rng) {
    SelectorGuard mode;
    std::printf("\n--bench: microseconds per call, old (per-column decode) / new (decode once), idle GPU assumed\n");
    const int n_in = 2560, n_out = 8192, it = 200;
    for (int t : {16, 17, 18, 21, 22, 29}) {
        DeviceAllocations memory;
        const auto w = random_rows(t, n_out, n_in, rng);
        uint8_t* dw = memory.alloc<uint8_t>(w.size());
        ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
        const auto x = random_x((size_t) 8 * n_in, rng);
        float* dx = memory.alloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        uint8_t* dxq = memory.alloc<uint8_t>((size_t) 8 * (n_in / 32) * 36);
        k::quantize_q8_1_rows(dx, 8, n_in, dxq, s);
        float* dy = memory.alloc<float>((size_t) 8 * n_out);
        std::printf("%-8s iq_mmvq %d x %d:", name_of(t), n_out, n_in);
        for (int nc = 1; nc <= 8; ++nc) {
            float us[2];
            for (int old = 1; old >= 0; --old) {
                k::iq_set_old_kernels(old != 0);
                us[old] = 1e3f * time_ms(s, it, [&] { k::iq_mmvq(t, dw, dxq, dy, n_in, n_out, nc, s); });
            }
            std::printf("  %d: %.1f/%.1f", nc, us[1], us[0]);
        }
        std::printf("\n");
        k::iq_set_old_kernels(false);
    }
    // the verify window's shape: 2560 x 640 experts, 16 groups of m entries each
    for (int gu : {16, 17, 18, 21, 22, 29}) {
        std::printf("%-8s/Q2_0 grouped, 16 groups of m entries:", name_of(gu));
        for (int m : {1, 2, 3, 4, 8}) {
            Grouped G(gu, 42, 2560, 640, std::vector<int>(16, m), 8, rng);
            float us[2];
            for (int old = 1; old >= 0; --old)
                us[old] = 1e3f * time_ms(s, 50, [&] { G.run(old != 0, s); });
            std::printf("  m=%d: %.1f/%.1f", m, us[1], us[0]);
        }
        std::printf("\n");
    }
}

// #606: the q8_1 activation quantizers (quantize_q8_1_rows, native_quantize_q8_1 and the shared expert's fused
// native_swiglu_quantize_q8_1) against a host transcription of the 0.1.38 formula: every finite block bit for bit; a
// block whose sum (or scale) overflowed fp16 is now stored finite (the largest half, its sign), with its values in the
// int8 range.  The fused one gets gate = 32 and up = x / 32: silu(32) is 32 in float, so its activations are x exactly.
void check_q8_1_finite(cudaStream_t s, std::mt19937& rng) {
    DeviceAllocations memory;
    const int n = 2560, rows = 3, nb = n / 32;
    std::vector<float> x = random_x((size_t) rows * n, rng);
    for (int i = 0; i < 32; ++i) x[(size_t) 5 * 32 + i] = 3000.0f;                  // sum 96,000: overflowed
    for (int i = 0; i < 32; ++i) x[(size_t) 9 * 32 + i] = i == 3 ? -70000.0f : 0.5f;   // sum -69,984.5
    for (int i = 0; i < 32; ++i) x[(size_t) 13 * 32 + i] = 2000.0f + i;            // sum 64,496: finite, kept
    x[(size_t) n + 40] = 2.0e7f;                                                    // amax / 127 overflowed too
    float* dx = memory.alloc<float>(x.size());
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    std::vector<float> gate(x.size(), 32.0f), up(x.size());
    for (size_t i = 0; i < x.size(); ++i) up[i] = x[i] / 32.0f;
    float* dg = memory.alloc<float>(x.size());
    float* du = memory.alloc<float>(x.size());
    ck(cudaMemcpy(dg, gate.data(), x.size() * 4, cudaMemcpyHostToDevice), "gate");
    ck(cudaMemcpy(du, up.data(), x.size() * 4, cudaMemcpyHostToDevice), "up");
    uint8_t* dq = memory.alloc<uint8_t>((size_t) rows * nb * 36);
    for (int path = 0; path < 3; ++path) {
        if (path == 0) k::quantize_q8_1_rows(dx, rows, n, dq, s);
        else if (path == 1) k::native_quantize_q8_1(dx, dq, n, rows, s);
        else k::native_swiglu_quantize_q8_1(dg, du, dq, n, rows, s);
        ck(cudaStreamSynchronize(s), "quantize");
        std::vector<uint8_t> got((size_t) rows * nb * 36);
        ck(cudaMemcpy(got.data(), dq, got.size(), cudaMemcpyDeviceToHost), "q");
        int same = 0, clamped = 0, bad = 0;
        for (int b = 0; b < rows * nb; ++b) {
            const float* v = x.data() + (size_t) b * 32;
            float lane[32], amax = 0.0f;
            for (int i = 0; i < 32; ++i) { lane[i] = v[i]; amax = std::max(amax, std::fabs(v[i])); }
            for (int o = 16; o > 0; o >>= 1) {   // the warp's butterfly: every lane ends with the same sum
                float nx[32];
                for (int i = 0; i < 32; ++i) nx[i] = lane[i] + lane[i ^ o];
                std::memcpy(lane, nx, sizeof lane);
            }
            const float d = amax / 127.0f, sum = lane[0];
            const uint8_t* g = got.data() + (size_t) b * 36;
            uint16_t gd, gs;
            std::memcpy(&gd, g, 2);
            std::memcpy(&gs, g + 2, 2);
            const uint16_t od = k::f16_from_f32(d), os = k::f16_from_f32(sum);
            const bool finite_before = (od & 0x7c00) != 0x7c00 && (os & 0x7c00) != 0x7c00;
            if (finite_before) {
                uint8_t want[36];
                std::memcpy(want, &od, 2);
                std::memcpy(want + 2, &os, 2);
                for (int i = 0; i < 32; ++i)   // CUDA's roundf: half away from zero, as std::round
                    want[4 + i] = (uint8_t) (int8_t) (amax == 0.0f ? 0.0f : std::round(v[i] / d));
                // native_mmvq.cu is built with --use_fast_math (its divisions are approximate): there the sum must be
                // the same bits, the scale within one fp16 step and each value within one step
                bool near = path >= 1 && gs == os && std::abs((int) gd - (int) od) <= 1;
                for (int i = 0; near && i < 32; ++i) near = std::abs((int) (int8_t) g[4 + i] - (int) (int8_t) want[4 + i]) <= 1;
                if (std::memcmp(want, g, 36) == 0 || near) ++same;
                else { ++bad; std::printf("  q8_1 path %d block %d: changed although it was finite\n", path, b); }
            } else {
                const float fd = k::f32_from_f16(gd), fs = k::f32_from_f16(gs);
                bool ok = std::isfinite(fd) && std::isfinite(fs) && fd > 0.0f &&
                          ((os & 0x7c00) != 0x7c00 ? gs == os : std::fabs(fs) == 65504.0f && (fs < 0) == (sum < 0));
                for (int i = 0; i < 32; ++i) ok = ok && (int8_t) g[4 + i] >= -127 && (int8_t) g[4 + i] <= 127;
                if (ok) ++clamped;
                else { ++bad; std::printf("  q8_1 path %d block %d: d %g sum %g not finite and clamped\n", path, b, fd, fs); }
            }
        }
        std::printf("q8_1 finite (%s): %d finite blocks as before, %d overflowed blocks clamped finite%s\n",
                    path == 0 ? "quantize_q8_1_rows" : path == 1 ? "native_quantize_q8_1" : "native_swiglu_quantize_q8_1",
                    same, clamped, bad ? "  FAIL" : "");
        g_fail += bad + (clamped != 3);
    }
}

// -------------------------------- (3) paired layout family, NOT the entire PR-h2715a752 delta:
// fork 9921eec already had subwarp down.  No per-column/codebook/rows2 A/B here.
// Contract: zero tolerance (memcmp), including active GU floats, the intermediate Q8_1 bytes consumed by down,
// final output and untouched rows.  The subwarp helpers explicitly preserve the fallback's addition tree.
// Written floats must also be finite.  Input bytes are quantized ONCE per input, outside capture, and checked
// unchanged after every call.  Both graphs replay A, changed input B and a zero-group call at the same addresses.
// This is a synthetic layout gate, not a model-weight oracle.  CUDA failures exit nonzero, never skip/pass.
struct LayoutModeGuard {
    const bool prior = k::native_grouped_no_sub16_gu(), old = k::iq_old_kernels();
    LayoutModeGuard() { k::iq_set_old_kernels(false); } // fixed decode-once for the entire layout gate
    ~LayoutModeGuard() {
        k::native_grouped_set_no_sub16_gu(prior);
        k::iq_set_old_kernels(old);
    }
};

void layout_call(Grouped& G, cudaStream_t s) {
    k::native_expert_grouped(G.L, G.dptr, G.dstart, G.dn, G.ddst, G.dtok, G.cap_groups, G.cap_ent,
                             G.dxq, G.dscr, G.dout, s, 3); // also exercise the group stride
}

GraphExec layout_graph(Grouped& G, bool no_sub16, cudaStream_t s) {
    k::native_grouped_set_no_sub16_gu(no_sub16); // only before capture, never during it
    Graph graph("layout graph free");
    GraphExec exec("layout exec free");
    CaptureGuard capture(s);
    layout_call(G, s);
    capture.end(graph);
    ck(cudaGraphInstantiate(&exec.value, graph, 0), "layout instantiate");
    graph.close();
    return exec;
}

// Includes the allocations' 256-byte tail guards.  Scratch is deliberately stale before each call.
struct LayoutResult {
    std::vector<uint32_t> out;
    std::vector<uint8_t> scratch;
};
LayoutResult layout_result(Grouped& G, bool no_sub16, cudaGraphExec_t graph, bool empty,
                           const std::vector<uint8_t>& input, cudaStream_t s) {
    const size_t sb = k::native_expert_scratch_bytes(G.cap_ent, G.L.n_ff);
    ck(cudaMemsetAsync(G.dout, 0xFF, G.out_floats * 4 + 256, s), "layout output reset");
    ck(cudaMemsetAsync(G.dscr, 0xA5, sb + 256, s), "layout scratch reset");
    // Replay with the HOST selector opposite to capture: a graph must keep its captured kernels.
    k::native_grouped_set_no_sub16_gu(graph ? !no_sub16 : no_sub16);
    if (graph) ck(cudaGraphLaunch(graph, s), "layout replay");
    else layout_call(G, s);
    ck(cudaStreamSynchronize(s), "layout sync");
    LayoutResult r{std::vector<uint32_t>(G.out_floats + 64), std::vector<uint8_t>(sb + 256)};
    ck(cudaMemcpy(r.out.data(), G.dout, r.out.size() * 4, cudaMemcpyDeviceToHost), "layout output");
    ck(cudaMemcpy(r.scratch.data(), G.dscr, r.scratch.size(), cudaMemcpyDeviceToHost), "layout scratch");
    std::vector<uint8_t> got(input.size());
    ck(cudaMemcpy(got.data(), G.dxq, got.size(), cudaMemcpyDeviceToHost), "layout input");
    size_t guards = (got != input);
    for (size_t i = input.size() - 256; i < got.size(); ++i) guards += got[i] != 0xA5;
    const size_t fa = ((size_t) G.cap_ent * G.L.n_ff * 4 + 255) & ~(size_t) 255;
    // Check active GU floats as well as the final output: finite down output alone is not an oracle for GU.
    for (size_t i = 0; !empty && i < (size_t) G.n_ent * G.L.n_ff; ++i) {
        float gate, up;
        std::memcpy(&gate, r.scratch.data() + i * 4, 4);
        std::memcpy(&up, r.scratch.data() + fa + i * 4, 4);
        guards += !std::isfinite(gate) || !std::isfinite(up);
    }
    for (size_t i = 0; i < r.out.size(); ++i) {
        const bool written = i < G.out_floats && !empty && G.written_rows[i / G.L.n_embd];
        float v;
        std::memcpy(&v, &r.out[i], 4);
        guards += written ? !std::isfinite(v) : r.out[i] != 0xFFFFFFFFu;
    }
    for (size_t i = sb; i < r.scratch.size(); ++i) guards += r.scratch[i] != 0xA5;
    if (guards) {
        std::printf("  layout %s no_sub16=%d empty=%d: %zu finite/input/untouched/tail failures\n",
                    graph ? "graph" : "direct", no_sub16, empty, guards);
        ++g_fail;
    }
    return r;
}

void layout_equal(const Grouped& G, const LayoutResult& ref, const LayoutResult& got, bool empty) {
    const size_t fa = ((size_t) G.cap_ent * G.L.n_ff * 4 + 255) & ~(size_t) 255;
    const size_t active = empty ? 0 : (size_t) G.n_ent;
    const size_t gu = active * G.L.n_ff * 4, hq = active * (G.L.n_ff / 32) * 36;
    const bool same = ref.out == got.out &&
        std::memcmp(ref.scratch.data(), got.scratch.data(), gu) == 0 &&
        std::memcmp(ref.scratch.data() + fa, got.scratch.data() + fa, gu) == 0 &&
        std::memcmp(ref.scratch.data() + 3 * fa, got.scratch.data() + 3 * fa, hq) == 0;
    if (!same) { std::printf("  layout GU/Q8_1/down bitwise mismatch\n"); ++g_fail; }
}

void check_sub16_layout(cudaStream_t s) {
#if defined(STRATA_HIP_GFX906)
    std::printf("sub16 layout FAIL: requires the CUDA layout dispatch, not gfx906 alternate layouts\n");
    ++g_fail; return;
#endif
    std::mt19937 rng(2715); // keep the inherited checks' seed/input sequence unchanged
    // V2 bypasses the selector for IQ3_S/IQ4_NL.  Reject rather than silently testing the same kernels twice.
    const char* v2 = std::getenv("STRATA_EXPERT_V2");
    if (v2 && v2[0] == '1') {
        std::printf("sub16 layout FAIL: STRATA_EXPERT_V2 must be unset/0\n"); ++g_fail; return;
    }
    LayoutModeGuard mode;
    // EXACT unique pairs in tests/data/native_experts/iq3_s.txt; IQ4_XS/Q2_0 is NOT in that manifest.
    const std::pair<int, int> pairs[] = {{18,20}, {18,42}, {22,20}, {22,42}, {21,20}, {21,42}, {23,20}};
    for (const auto& [gu, dt] : pairs) {
        if (!k::native_expert_supported(gu, dt, 2560, 640)) {
            std::printf("sub16 layout FAIL: required pair %d/%d unsupported\n", gu, dt); ++g_fail; continue;
        }
        const int before = g_fail;
        Grouped G(gu, dt, 2560, 640, {1, 0, 2, 3, 0, 4, 5, 8, 11}, 16, rng, true);
        const size_t qb = (size_t) G.n_tok * (G.L.n_embd / 32) * 36;
        GraphExec graphs[2] = {GraphExec("layout exec0 free"), GraphExec("layout exec1 free")};
        std::vector<uint8_t> previous_input;
        std::vector<uint32_t> previous_output;
        for (int input_id = 0; input_id < 3; ++input_id) {
            const int input_before = g_fail;
            const bool empty = input_id == 2;
            if (input_id == 1) {
                const auto x = random_x((size_t) G.n_tok * G.L.n_embd, rng);
                ck(cudaMemcpy(G.dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "layout changed x");
            }
            if (!empty) { // one quantization, shared byte-for-byte by ALL four executions below
                ck(cudaMemsetAsync(G.dxq, 0xA5, qb + 256, s), "layout input guard");
                k::quantize_q8_1_rows(G.dx, G.n_tok, G.L.n_embd, G.dxq, s);
                ck(cudaStreamSynchronize(s), "layout quantize");
            }
            const int ng = empty ? 0 : G.n_groups;
            ck(cudaMemcpy(G.dn, &ng, sizeof ng, cudaMemcpyHostToDevice), "layout groups");
            std::vector<uint8_t> input(qb + 256);
            ck(cudaMemcpy(input.data(), G.dxq, input.size(), cudaMemcpyDeviceToHost), "layout shared Q8_1");
            const auto ref = layout_result(G, false, nullptr, empty, input, s);
            const auto fallback = layout_result(G, true, nullptr, empty, input, s);
            layout_equal(G, ref, fallback, empty);
            if (input_id == 0) { // warm direct calls also initialize static launch choices before capture
                graphs[0] = layout_graph(G, false, s);
                graphs[1] = layout_graph(G, true, s);
            }
            for (int no_sub16 = 0; no_sub16 < 2; ++no_sub16)
                layout_equal(G, ref, layout_result(G, no_sub16 != 0, graphs[no_sub16], empty, input, s), empty);
            if (input_id == 1 && (input == previous_input || ref.out == previous_output)) {
                std::printf("  layout changed-input replay checked no change FAIL\n"); ++g_fail;
            }
            previous_input = input; previous_output = ref.out;
            if (g_fail != input_before)
                std::printf("  layout pair %d/%d input=%s FAIL\n", gu, dt,
                            input_id == 0 ? "A" : empty ? "empty" : "changed B");
        }
        for (auto& graph : graphs) graph.close();
        std::printf("sub16 layout %-8s/%-7s 2560x640 tails 1/2/3/4/5/8/11, empty, scattered: %s\n",
                    name_of(gu), name_of(dt), g_fail == before ? "bitwise direct + A/B/empty graph replay" : "FAIL");
    }
}

}  // namespace

int parity_main(int argc, char** argv) {
    SelectorGuard mode;
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    Stream s("stream free");
    ck(cudaStreamCreate(&s.value), "stream");
    std::mt19937 rng(18);
    const bool prior_layout = k::native_grouped_no_sub16_gu(), prior_old = k::iq_old_kernels();
    check_sub16_layout(s); // isolated paired gate runs before the historical old/per-column checks
    if (k::native_grouped_no_sub16_gu() != prior_layout || k::iq_old_kernels() != prior_old) {
        std::printf("sub16 layout selector restoration FAIL\n"); ++g_fail;
    }
    for (int t : {16, 17, 18, 20, 21, 22, 23, 29, 42}) {
        check_mmvq(t, 2560, 67, s, rng);   // the model's n_embd; 67 rows: a partial block of 4 rows
        check_mmvq(t, 1024, 5, s, rng);
    }
    for (int gu : {16, 17, 18, 21, 22, 23, 29, 42}) {
        for (int dt : {20, 42}) check_grouped(gu, dt, 2560, 640, s, rng);   // the model's shape
        check_grouped(gu, 23, 1024, 512, s, rng);                           // IQ4_XS down needs n_ff % 256 == 0
    }
    check_q8_1_finite(s, rng);
    if (do_bench) bench(s, rng);
    std::printf("iq_multi_parity: %d failures\n", g_fail);
    s.close();
    return g_fail ? 1 : 0;
}

int main(int argc, char** argv) {
    try {
        const int result = parity_main(argc, argv);
        return g_cleanup_fail ? 2 : result;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "iq_multi_parity FAIL: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "iq_multi_parity FAIL: unknown exception\n");
    }
    return 2; // All scoped resources and selectors have unwound before the failure exit.
}
