// src/kernels/tier_b_expert_bench.cpp - paired sub16 vs reference native expert layout timing on the
// IQ3_S-file's real 2560 x 640 expert geometry, synthetic bytes (no model).  Uses the .2 selector
// native_grouped_set_no_sub16_gu; the reference is the wider-warp decode-once fallback.  Both layouts get
// the same weights, the same quantized Q8_1 activations and the same destinations; only the selector differs.
//
//     build/tier_b_expert_bench [--groups G] [--tokens T] [--slots S] [--warmup W] [--samples M]
//                               [--max-bytes B] [--max-slots MS] [--pair gu,dt] [--help]
//
// Correctness before timing: each pair's sub16 and reference outputs must be bitwise equal AND the run must
// actually write finite values; otherwise it is reported and the program exits nonzero.  Selector state is
// restored after every measurement.  Old per-column decode is pinned OFF; codebook staging and rows2 are left
// to the environment and printed, never A/B'd here (rows2 is native_mmvq only, unused by grouped experts).
//
// Timing: a bounded rotating working set of `slots` x `groups` blobs whose total exceeds the device L2
// (cudaDeviceProp.l2CacheSize).  Each paired sample runs BOTH arms on the SAME slot (identical weights and
// inputs); before each timed arm an untimed `prime` touches the other slots so both arms start from an
// equivalent cache state - best-effort conditioning, NOT a claim that every timed launch is DRAM-cold (the
// pool > L2 only bounds residency).  Per-call CUDA-event latency only (idle GPU; NOT a throughput figure and
// not a model throughput claim).  Fixed warmup, alternating AB/BA raw paired samples (always printed, with
// per-sample slot and order) reported as median and min..max.  Direct launches, warmed graph replay and the
// host-side capture+instantiate cost (both layouts, paired) are reported separately.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + w + ": " + cudaGetErrorString(e));
    }
}

// Cleanup must neither throw nor stop subsequent releases (including during constructor unwinding).
void cleanup(cudaError_t e, const char* w) noexcept {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup error in %s: %s\n", w, cudaGetErrorString(e));
        ++g_fail;
    }
}

template<typename T, auto Destroy>
class CudaOwner {
    T value_ = nullptr;
public:
    CudaOwner() = default;
    CudaOwner(const CudaOwner&) = delete;
    CudaOwner& operator=(const CudaOwner&) = delete;
    CudaOwner(CudaOwner&& other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
    CudaOwner& operator=(CudaOwner&& other) noexcept {
        if (this != &other) {
            reset();
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    ~CudaOwner() noexcept { reset(); }
    operator T() const noexcept { return value_; }
    // Only used on empty owners: CUDA writes directly into owned storage, even on a failed create.
    T* put() noexcept { return &value_; }
    void reset() noexcept {
        if (value_) cleanup(Destroy(std::exchange(value_, nullptr)), "resource release");
    }
    void close(const char* w) {
        if (value_) {
            ck(Destroy(value_), w);
            value_ = nullptr;  // keep ownership on failure so unwinding can try best-effort release
        }
    }
};

template<typename T> using Device = CudaOwner<T*, cudaFree>;
using Stream = CudaOwner<cudaStream_t, cudaStreamDestroy>;
using Event = CudaOwner<cudaEvent_t, cudaEventDestroy>;
using Graph = CudaOwner<cudaGraph_t, cudaGraphDestroy>;
using GraphExec = CudaOwner<cudaGraphExec_t, cudaGraphExecDestroy>;
using Graphs = std::vector<GraphExec>;

constexpr size_t kDeviceGuardBytes = 256;
constexpr int kInputTokens = 8;

size_t add_bytes(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::overflow_error("device allocation size addition overflow");
    return a + b;
}
size_t mul_bytes(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::overflow_error("device allocation size multiplication overflow");
    return a * b;
}
size_t align256(size_t n) { return add_bytes(n, 255) & ~(size_t) 255; }

template<typename T>
size_t buffer_bytes(size_t n) { return add_bytes(mul_bytes(n, sizeof(T)), kDeviceGuardBytes); }

template<typename T>
Device<T> dalloc(size_t n) {
    const size_t bytes = buffer_bytes<T>(n);
    Device<T> p;
    ck(cudaMalloc(p.put(), bytes), "malloc");
    ck(cudaMemset(p, 0, bytes), "memset");
    return p;
}

// End even an invalidated capture before freeing captured inputs or destroying its stream.
// CUDA ends capture on an EndCapture error too; any returned graph is always owned.
class Capture {
    cudaStream_t s_;
    bool active_ = false;
public:
    explicit Capture(cudaStream_t s, const char* w) : s_(s) {
        ck(cudaStreamBeginCapture(s_, cudaStreamCaptureModeThreadLocal), w);
        active_ = true;
    }
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;
    void end(Graph& g, const char* w) {
        const auto e = cudaStreamEndCapture(s_, g.put());
        active_ = false;
        ck(e, w);
    }
    ~Capture() noexcept {
        if (active_) {
            Graph abandoned;
            cleanup(cudaStreamEndCapture(s_, abandoned.put()), "abandon capture");
        }
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

// `rows` rows of `n` values of format t: random bytes, then a finite fp16 scale in every block (same recipe as
// iq_multi_parity's sub16 gate, so the inputs are identical to the parity gate's).
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = k::iq_row_bytes(t, n), bs = k::iq_row_bytes(t, block_values(t));
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 3);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        if (t == 29) {
            const uint8_t nib = (uint8_t) ((sgn(rng) == 0 ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
            w[o + 55] = (uint8_t) ((w[o + 55] & 0x0F) | (nib << 4));
        } else {
            const uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));  // 2^-13 .. 2^-6
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

// ---------------------------------------------------------------- command line
struct Cli {
    int groups = 8, tokens = 2, slots = 8, warmup = 25, samples = 30, max_slots = 64;
    size_t max_bytes = 512ull << 20;                 // 512 MiB cap on explicit device buffers, guards included
    bool help = false;
    std::vector<std::pair<int, int>> pairs;          // empty = all 7 manifest pairs
};

int usage() {
    std::printf(
        "usage: tier_b_expert_bench [options]\n"
        "  --groups G     groups per call (1..64, default 8)\n"
        "  --tokens T     entries per group (1..8, default 2)\n"
        "  --slots S      rotation slots (1..max-slots, default 8; auto-bumped until working set > L2)\n"
        "  --warmup W     fixed warmup calls per layout (1..1000, default 25)\n"
        "  --samples M    raw paired samples per layout (1..10000, default 30)\n"
        "  --max-bytes B  explicit device-buffer cap, guards included (1048576..1099511627776, default 536870912)\n"
        "  --max-slots MS slot cap (1..1024, default 64)\n"
        "  --pair gu,dt   only this format pair (repeatable; default all 7 manifest pairs)\n"
        "  --help\n");
    return 0;
}

long long num(const char* s, const char* what, long long lo, long long hi) {
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < lo || v > hi) {
        std::fprintf(stderr, "%s must be an integer in %lld..%lld (got \"%s\")\n", what, lo, hi, s);
        std::exit(2);
    }
    return v;
}

Cli parse(int argc, char** argv) {
    Cli c;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") c.help = true;
        else if (a == "--groups") c.groups = (int) num(need("--groups"), "--groups", 1, 64);
        else if (a == "--tokens") c.tokens = (int) num(need("--tokens"), "--tokens", 1, 8);
        else if (a == "--slots") c.slots = (int) num(need("--slots"), "--slots", 1, 1024);
        else if (a == "--warmup") c.warmup = (int) num(need("--warmup"), "--warmup", 1, 1000);
        else if (a == "--samples") c.samples = (int) num(need("--samples"), "--samples", 1, 10000);
        else if (a == "--max-slots") c.max_slots = (int) num(need("--max-slots"), "--max-slots", 1, 1024);
        else if (a == "--max-bytes") {
            const char* v = need("--max-bytes");
            errno = 0;
            char* end = nullptr;
            const unsigned long long b = std::strtoull(v, &end, 10);
            if (errno != 0 || end == v || *end != '\0' || b < (1ull << 20) || b > (1ull << 40)) {
                std::fprintf(stderr, "--max-bytes must be 1048576..1099511627776 (got \"%s\")\n", v);
                std::exit(2);
            }
            c.max_bytes = (size_t) b;
        } else if (a == "--pair") {
            const std::string v = need("--pair");
            const size_t comma = v.find(',');
            if (comma == std::string::npos) { std::fprintf(stderr, "--pair needs gu,dt\n"); std::exit(2); }
            // Each side is parsed on its own NUL-terminated text: `num` requires the whole string, so
            // handing it "gu,dt" rejected every valid pair.  A second comma stays rejected via `dt`.
            const std::string gu_text = v.substr(0, comma), dt_text = v.substr(comma + 1);
            const int gu = (int) num(gu_text.c_str(), "--pair gu", 1, 1024);
            const int dt = (int) num(dt_text.c_str(), "--pair dt", 1, 1024);
            c.pairs.push_back({gu, dt});
        } else {
            std::fprintf(stderr, "unknown argument: %s (see --help)\n", a.c_str());
            std::exit(2);
        }
    }
    return c;
}

// ---------------------------------------------------------------- one pair's complete explicit allocation plan
// Weight working set excludes the allocation's guard; total_bytes includes every cudaMalloc request.
// CUDA's opaque graph/exec/stream/event/driver allocations are outside this count: their size is not measured.
struct DevicePlan {
    k::NativeExpertLayout L;
    int groups, tokens, slots, n_ent;
    size_t blob_stride, slot_bytes, arena_bytes, ptr_count, start_count;
    size_t x_floats, xq_bytes, scratch_bytes, out_floats, total_bytes;
};

DevicePlan device_plan(const k::NativeExpertLayout& L, int groups, int tokens, int slots) {
    if (groups <= 0 || tokens <= 0 || slots <= 0 || L.n_embd <= 0 || L.n_ff <= 0 ||
        L.n_embd % 32 != 0 || L.n_ff % 32 != 0)
        throw std::invalid_argument("invalid expert allocation geometry");
    if ((uint64_t) L.n_embd > std::numeric_limits<size_t>::max() ||
        (uint64_t) L.n_ff > std::numeric_limits<size_t>::max())
        throw std::overflow_error("expert allocation dimension does not fit size_t");
    const size_t H = (size_t) L.n_embd, FF = (size_t) L.n_ff;
    DevicePlan p{};
    p.L = L; p.groups = groups; p.tokens = tokens; p.slots = slots;
    const size_t entries = mul_bytes((size_t) groups, (size_t) tokens);
    if (entries > (size_t) std::numeric_limits<int>::max())
        throw std::overflow_error("expert entry count does not fit int");
    p.n_ent = (int) entries;
    p.blob_stride = align256(L.bytes);
    p.slot_bytes = mul_bytes((size_t) groups, p.blob_stride);
    p.arena_bytes = mul_bytes((size_t) slots, p.slot_bytes);
    p.ptr_count = mul_bytes((size_t) slots, (size_t) groups);
    p.start_count = add_bytes((size_t) groups, 1);
    p.x_floats = mul_bytes((size_t) kInputTokens, H);
    p.xq_bytes = mul_bytes(mul_bytes((size_t) kInputTokens, H / 32), 36);
    p.out_floats = mul_bytes(entries, H);
    // Check every operation in the kernel API's scratch formula before calling its unchecked size helper.
    const size_t f = mul_bytes(mul_bytes(entries, FF), sizeof(float));
    const size_t hq = mul_bytes(mul_bytes(entries, FF / 32), 36);
    p.scratch_bytes = add_bytes(mul_bytes(3, align256(f)), align256(hq));
    if (p.scratch_bytes != k::native_expert_scratch_bytes(p.n_ent, L.n_ff))
        throw std::runtime_error("expert scratch allocation contract changed");
    for (size_t bytes : {buffer_bytes<uint8_t>(p.arena_bytes),
                         buffer_bytes<unsigned long long>(p.ptr_count),
                         buffer_bytes<int32_t>(p.start_count), buffer_bytes<int32_t>(1),
                         buffer_bytes<int32_t>(entries), buffer_bytes<int32_t>(entries),
                         buffer_bytes<float>(p.x_floats), buffer_bytes<uint8_t>(p.xq_bytes),
                         buffer_bytes<uint8_t>(p.scratch_bytes), buffer_bytes<float>(p.out_floats)})
        p.total_bytes = add_bytes(p.total_bytes, bytes);
    return p;
}

// ---------------------------------------------------------------- one pair's device state
struct Bench {
    k::NativeExpertLayout L;
    int G = 0, T = 0, NTOK = kInputTokens, slots = 0;
    int cap_groups = 0, n_ent = 0;
    size_t blob_stride = 0, slot_bytes = 0, arena_bytes = 0, out_floats = 0;
    Device<uint8_t> darena;
    Device<unsigned long long> dptr;
    Device<int32_t> dstart, dn, ddst, dtok;
    Device<float> dx;
    Device<uint8_t> dxq, dscr;
    Device<float> dout;

    Bench(int gu, int dt, int64_t H, int64_t FF, int g, int t, int sl, std::mt19937& rng, cudaStream_t s,
          size_t max_bytes = 512ull << 20)
        : Bench(device_plan(k::native_expert_layout(gu, dt, H, FF), g, t, sl), rng, s, max_bytes) {}

    Bench(const DevicePlan& p, std::mt19937& rng, cudaStream_t s, size_t max_bytes)
        : L(p.L), G(p.groups), T(p.tokens), slots(p.slots), cap_groups(p.groups), n_ent(p.n_ent),
          blob_stride(p.blob_stride), slot_bytes(p.slot_bytes), arena_bytes(p.arena_bytes), out_floats(p.out_floats) {
        // Also protect direct constructor callers; the CLI performs this admission before constructing Bench.
        if (p.total_bytes > max_bytes) throw std::length_error("explicit device buffers exceed allocation budget");
        const int gu = L.gu_type, dt = L.d_type;
        const int64_t H = L.n_embd, FF = L.n_ff;
        darena = dalloc<uint8_t>(arena_bytes);
        std::vector<unsigned long long> ptr(p.ptr_count);
        for (int j = 0; j < slots; ++j) {
            for (int g = 0; g < G; ++g) {
                auto b = random_rows(gu, 2 * FF, H, rng);   // gate rows | up rows
                const auto d = random_rows(dt, H, FF, rng); // down rows
                b.insert(b.end(), d.begin(), d.end());
                if (b.size() != L.bytes) {
                    throw std::runtime_error("blob " + std::to_string(b.size()) + " != " + std::to_string(L.bytes));
                }
                uint8_t* dst = darena + (size_t) j * slot_bytes + (size_t) g * blob_stride;
                ck(cudaMemcpy(dst, b.data(), b.size(), cudaMemcpyHostToDevice), "blob");
                ptr[(size_t) j * G + g] = (unsigned long long) dst;
            }
        }
        dptr = dalloc<unsigned long long>(ptr.size());
        ck(cudaMemcpy(dptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice), "ptr");
        std::vector<int32_t> start(p.start_count), dst_v(n_ent), tok_v(n_ent);
        for (int g = 0; g < G; ++g) {
            start[g] = g * T;
            for (int t = 0; t < T; ++t) {
                dst_v[g * T + t] = g * T + t;              // identical, deterministic destinations
                tok_v[g * T + t] = (g * 7 + t * 3) % NTOK; // distinct tokens of an 8-token window
            }
        }
        start[G] = n_ent;
        dstart = dalloc<int32_t>(p.start_count);
        dn = dalloc<int32_t>(1);
        ddst = dalloc<int32_t>(n_ent);
        dtok = dalloc<int32_t>(n_ent);
        ck(cudaMemcpy(dstart, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "start");
        ck(cudaMemcpy(dn, &G, 4, cudaMemcpyHostToDevice), "n");
        ck(cudaMemcpy(ddst, dst_v.data(), dst_v.size() * 4, cudaMemcpyHostToDevice), "dst");
        ck(cudaMemcpy(dtok, tok_v.data(), tok_v.size() * 4, cudaMemcpyHostToDevice), "tok");
        const auto x = random_x(p.x_floats, rng);
        dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        dxq = dalloc<uint8_t>(p.xq_bytes);
        k::quantize_q8_1_rows(dx, NTOK, H, dxq, s);       // one quantization, shared by both layouts
        dscr = dalloc<uint8_t>(p.scratch_bytes);
        dout = dalloc<float>(out_floats);
    }
    void call(int slot, cudaStream_t s) {
        k::native_expert_grouped(L, dptr + (size_t) slot * cap_groups, dstart, dn, ddst, dtok,
                                 cap_groups, n_ent, dxq, dscr, dout, s, 0);
    }
};

// ---------------------------------------------------------------- selector + error-policy guard (restore on scope exit)
// This standalone benchmark is an audited opt-in caller for the IQ host-error policy: launch/argument
// failures must unwind through the RAII owners above instead of calling exit(1) and leaking device state.
// `errors` is declared FIRST so the policy is enabled before any selector call, any CUDA acquisition and any
// kernel launch, and is destroyed LAST so restore() and all unwinding still run under it.  Its destructor
// puts back whatever policy this thread had (nestable); the default engine policy is untouched.
struct Guard {
    k::IqHostErrorScope errors;
    const bool ns = k::native_grouped_no_sub16_gu();
    const bool old = k::iq_old_kernels();
    Guard() { k::iq_set_old_kernels(false); }   // decode-once only, never the old per-column kernels
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    void restore() noexcept { k::native_grouped_set_no_sub16_gu(ns); k::iq_set_old_kernels(old); }
    ~Guard() noexcept { restore(); }
};

// ---------------------------------------------------------------- correctness (parity before timing)
int parity(Bench& B, cudaStream_t s) {
    std::vector<uint32_t> a(B.out_floats), b(B.out_floats);
    auto read = [&](bool ns, std::vector<uint32_t>& v) {
        k::native_grouped_set_no_sub16_gu(ns);
        ck(cudaMemsetAsync(B.dout, 0xFF, B.out_floats * 4, s), "parity out reset");
        ck(cudaMemsetAsync(B.dscr, 0, k::native_expert_scratch_bytes(B.n_ent, B.L.n_ff), s), "parity scr reset");
        B.call(0, s);
        ck(cudaStreamSynchronize(s), "parity sync");
        ck(cudaMemcpy(v.data(), B.dout, v.size() * 4, cudaMemcpyDeviceToHost), "parity read");
    };
    read(false, a);   // sub16
    read(true, b);    // reference (wider warp)
    size_t diff = 0;
    bool any = false, finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        diff += a[i] != b[i];
        any |= a[i] != 0xFFFFFFFFu;
        float v;
        std::memcpy(&v, &a[i], 4);
        finite = finite && std::isfinite(v);
    }
    int bad = 0;
    if (diff) { std::printf("  parity: %zu of %zu outputs differ\n", diff, a.size()); ++bad; }
    if (!any) { std::printf("  parity: nothing was written (checked nothing)\n"); ++bad; }
    if (!finite) { std::printf("  parity: non-finite output\n"); ++bad; }
    return bad;
}

// one graph per slot; captured with the given layout before any timing of it
Graphs graphs(Bench& B, bool ns, cudaStream_t s) {
    k::native_grouped_set_no_sub16_gu(ns);
    Graphs gs(B.slots);
    for (int j = 0; j < B.slots; ++j) {
        Graph g;
        GraphExec ex;
        Capture capture(s, "graph begin");
        B.call(j, s);
        capture.end(g, "graph end");
        ck(cudaGraphInstantiate(ex.put(), g, 0), "graph instantiate");
        g.close("graph destroy");
        gs[j] = std::move(ex);
    }
    return gs;
}

// replay slot 0 with the opposite host selector: the graph must keep its captured layout, bit for bit
int graph_check(Bench& B, const Graphs& gs, bool captured_ns, cudaStream_t s) {
    std::vector<uint32_t> direct(B.out_floats), replay(B.out_floats);
    k::native_grouped_set_no_sub16_gu(captured_ns);
    ck(cudaMemsetAsync(B.dout, 0xFF, B.out_floats * 4, s), "gc reset");
    B.call(0, s);
    ck(cudaStreamSynchronize(s), "gc sync");
    ck(cudaMemcpy(direct.data(), B.dout, direct.size() * 4, cudaMemcpyDeviceToHost), "gc direct");
    k::native_grouped_set_no_sub16_gu(!captured_ns);
    ck(cudaMemsetAsync(B.dout, 0xFF, B.out_floats * 4, s), "gc reset2");
    ck(cudaGraphLaunch(gs[0], s), "gc launch");
    ck(cudaStreamSynchronize(s), "gc sync2");
    ck(cudaMemcpy(replay.data(), B.dout, replay.size() * 4, cudaMemcpyDeviceToHost), "gc replay");
    size_t diff = 0;
    for (size_t i = 0; i < direct.size(); ++i) diff += direct[i] != replay[i];
    if (diff) std::printf("  graph replay (slot 0, host selector flipped) differs from direct: %zu\n", diff);
    return diff ? 1 : 0;
}

// ---------------------------------------------------------------- timing helpers
template<typename F>
float us(cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1, F&& fn) {
    ck(cudaEventRecord(e0, s), "record0");
    fn();
    ck(cudaEventRecord(e1, s), "record1");
    ck(cudaEventSynchronize(e1), "sync");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    return 1e3f * ms;
}

// Untimed cache conditioning for the paired protocol: one launch per OTHER slot in round-robin from the
// target, so both arms of a paired sample start from an equivalent cache state.  Best-effort-equivalent,
// not a proof that every timed launch is DRAM-cold; the pool > L2 only bounds residency.
void prime(Bench& B, int skip, cudaStream_t s) {
    for (int j = 1; j < B.slots; ++j) B.call((skip + j) % B.slots, s);
}
void warm_direct(Bench& B, int n, cudaStream_t s) {
    int slot = 0;
    for (int i = 0; i < n; ++i) { k::native_grouped_set_no_sub16_gu(i & 1); B.call(slot, s); slot = (slot + 1) % B.slots; }
}
void warm_graph(Bench& B, Graphs& gs, int n, cudaStream_t s) {
    int slot = 0;
    for (int i = 0; i < n; ++i) { ck(cudaGraphLaunch(gs[slot], s), "warm launch"); slot = (slot + 1) % B.slots; }
}
float direct_arm(Bench& B, int slot, bool ns, cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1) {
    k::native_grouped_set_no_sub16_gu(ns);
    return us(s, e0, e1, [&] { B.call(slot, s); });
}
float graph_arm(int slot, const Graphs& gs, cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1) {
    return us(s, e0, e1, [&] { ck(cudaGraphLaunch(gs[slot], s), "graph launch"); });
}

struct Stats { float med, lo, hi; };
Stats stats(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front(), v.back()};
}

// host-side capture + instantiate of one slot, then destroyed
double capture_us(Bench& B, bool ns, cudaStream_t s) {
    k::native_grouped_set_no_sub16_gu(ns);
    const auto t0 = std::chrono::steady_clock::now();
    Graph g;
    GraphExec ex;
    Capture capture(s, "cap begin");
    B.call(0, s);
    capture.end(g, "cap end");
    ck(cudaGraphInstantiate(ex.put(), g, 0), "cap instantiate");
    const auto t1 = std::chrono::steady_clock::now();
    g.close("cap graph destroy");
    ex.close("cap exec destroy");
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

struct PairRun { std::vector<float> a, b; };   // a = sub16, b = reference (same slot per sample)

// alternating AB/BA on the SAME slot: prime -> arm A -> prime -> arm B.  Each sample prints its slot,
// order and the two paired latencies (raw samples always available).
PairRun paired_direct(Bench& B, int samples, cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1) {
    PairRun r;
    int slot = 0;
    for (int i = 0; i < samples; ++i) {
        const int sl = slot; slot = (slot + 1) % B.slots;
        const bool af = (i & 1) == 0;                     // A = sub16 first on even samples
        prime(B, sl, s);
        const float x = direct_arm(B, sl, af ? false : true, s, e0, e1);
        prime(B, sl, s);
        const float y = direct_arm(B, sl, af ? true : false, s, e0, e1);
        r.a.push_back(af ? x : y);
        r.b.push_back(af ? y : x);
        std::printf("    #%3d slot %2d %s: sub16 %8.2f us  ref %8.2f us\n", i, sl, af ? "AB" : "BA", r.a.back(), r.b.back());
    }
    return r;
}
PairRun paired_graph(Bench& B, Graphs& gs_sub16, Graphs& gs_ref,
                     int samples, cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1) {
    PairRun r;
    int slot = 0;
    for (int i = 0; i < samples; ++i) {
        const int sl = slot; slot = (slot + 1) % B.slots;
        const bool af = (i & 1) == 0;
        prime(B, sl, s);
        const float x = graph_arm(sl, af ? gs_sub16 : gs_ref, s, e0, e1);
        prime(B, sl, s);
        const float y = graph_arm(sl, af ? gs_ref : gs_sub16, s, e0, e1);
        r.a.push_back(af ? x : y);
        r.b.push_back(af ? y : x);
        std::printf("    #%3d slot %2d %s: sub16 %8.2f us  ref %8.2f us\n", i, sl, af ? "AB" : "BA", r.a.back(), r.b.back());
    }
    return r;
}

}  // namespace

int run_benchmark(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const Cli c = parse(argc, argv);
    if (c.help) return usage();

    // Before the first CUDA call: scoped throwing IQ host-error policy plus selector restoration.
    Guard guard;
    if (!k::iq_host_errors_throw()) {
        std::printf("tier_b_expert_bench FAIL: scoped IQ host-error policy did not take effect\n");
        return 1;
    }

    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "device props");
    const size_t l2 = (size_t) prop.l2CacheSize;
    std::printf("device %s, L2 %zu bytes, explicit device-buffer budget %zu bytes\n", prop.name, l2, c.max_bytes);
    std::printf("budget counts all cudaMalloc buffers and guards; opaque CUDA graph/exec/stream/event/driver\n"
                "allocations are excluded and their sizes are not measured (not a total VRAM-use cap).\n");

#if defined(STRATA_HIP_GFX906)
    std::printf("tier_b_expert_bench FAIL: needs the CUDA sub16/fallback dispatch, not gfx906 alternate layouts\n");
    return 1;
#endif
    const char* v2 = std::getenv("STRATA_EXPERT_V2");
    if (v2 && v2[0] == '1') { std::printf("tier_b_expert_bench FAIL: STRATA_EXPERT_V2 bypasses the selector\n"); return 1; }
    std::printf("fixed: per-column decode OFF; codebook stage = %s; rows2 = %s (native_mmvq only, unused here)\n",
                std::getenv("STRATA_IQ_STAGE_GRID") ? "env-set" : "default",
                std::getenv("STRATA_NO_MMVQ_ROWS2") ? "env-set" : "default");
    std::printf("protocol: both arms of a paired sample run on the SAME slot (identical weights/inputs); before each\n"
                "          timed arm an untimed prime touches the other slots for equivalent cache state; AB/BA\n"
                "          order alternates. Pool > L2 bounds residency only - it does NOT guarantee per-call cold.\n");

    const std::pair<int, int> kPairs[] = {{18, 20}, {18, 42}, {22, 20}, {22, 42}, {21, 20}, {21, 42}, {23, 20}};
    std::vector<std::pair<int, int>> pairs = c.pairs;
    if (pairs.empty()) for (const auto& p : kPairs) pairs.push_back(p);

    std::mt19937 rng(2715);
    Stream s;
    ck(cudaStreamCreate(s.put()), "stream");
    Event e0, e1;
    ck(cudaEventCreate(e0.put()), "e0");
    ck(cudaEventCreate(e1.put()), "e1");

    int parity_fail = 0, timed = 0;
    for (const auto& [gu, dt] : pairs) {
        if (!k::native_expert_supported(gu, dt, 2560, 640)) {
            std::printf("%-8s/%-7s 2560x640: not supported, skip\n", name_of(gu), name_of(dt));
            ++parity_fail;
            continue;
        }
        const auto L = k::native_expert_layout(gu, dt, 2560, 640);
        const size_t blob_stride = align256(L.bytes);
        const size_t slot_bytes = mul_bytes((size_t) c.groups, blob_stride);
        int slots = std::min(c.slots, c.max_slots);
        while (mul_bytes((size_t) slots, slot_bytes) <= l2 && slots < c.max_slots) ++slots;   // exceed L2, bounded
        const auto plan = device_plan(L, c.groups, c.tokens, slots);
        const size_t arena = plan.arena_bytes;
        if (arena <= l2) {
            std::printf("%-8s/%-7s: working set %zu B <= L2 %zu B even at max slots %d\n",
                        name_of(gu), name_of(dt), arena, l2, slots);
            ++parity_fail;
            continue;
        }
        if (plan.total_bytes > c.max_bytes) {
            std::printf("%-8s/%-7s: explicit device buffers %zu B (weight working set %zu B, auxiliary+guards %zu B)\n"
                        "  exceeds budget %zu B; rejected before device-buffer allocation\n",
                        name_of(gu), name_of(dt), plan.total_bytes, arena, plan.total_bytes - arena, c.max_bytes);
            ++parity_fail;
            continue;
        }
        Bench B(plan, rng, s, c.max_bytes);
        std::printf("%-8s/%-7s G %d T %d | work %zu B = %d slots x %d groups x %zu B/blob, rotation %d, L2 %zu B\n",
                    name_of(gu), name_of(dt), c.groups, c.tokens, arena, slots, c.groups, blob_stride, slots, l2);
        std::printf("  allocation: weight working set %zu B; explicit device buffers %zu B; auxiliary+guards %zu B; budget %zu B\n",
                    arena, plan.total_bytes, plan.total_bytes - arena, c.max_bytes);

        // parity first: fail nonzero, never time a pair that is not bitwise equal
        if (parity(B, s)) { ++parity_fail; continue; }

        // warmed graph replay (correctness cross-check against direct)
        auto gs_sub16 = graphs(B, false, s);
        auto gs_ref = graphs(B, true, s);
        int gc = graph_check(B, gs_sub16, false, s);
        gc += graph_check(B, gs_ref, true, s);
        if (gc) { ++parity_fail; }
        else {
            std::printf("  direct raw (sample slot order: sub16/ref us):\n");
            warm_direct(B, c.warmup, s);
            const PairRun d = paired_direct(B, c.samples, s, e0, e1);
            std::printf("  graph raw (sample slot order: sub16/ref us):\n");
            warm_graph(B, gs_sub16, c.warmup, s);
            warm_graph(B, gs_ref, c.warmup, s);
            const PairRun g = paired_graph(B, gs_sub16, gs_ref, c.samples, s, e0, e1);
            const Stats sa = stats(d.a), sb = stats(d.b), sga = stats(g.a), sgb = stats(g.b);
            std::printf("  direct  sub16 %8.2f us [%7.2f..%7.2f]   ref %8.2f us [%7.2f..%7.2f]   sub16/ref %.3f\n",
                        sa.med, sa.lo, sa.hi, sb.med, sb.lo, sb.hi, sa.med / sb.med);
            std::printf("  graph   sub16 %8.2f us [%7.2f..%7.2f]   ref %8.2f us [%7.2f..%7.2f]   sub16/ref %.3f\n",
                        sga.med, sga.lo, sga.hi, sgb.med, sgb.lo, sgb.hi, sga.med / sgb.med);
            std::printf("  capture+instantiate (host, 1 slot): sub16 %8.0f us  ref %8.0f us  (per-call latency only, NOT throughput)\n",
                        capture_us(B, false, s), capture_us(B, true, s));
            ++timed;
        }
        for (auto& g : gs_sub16) g.close("exec free");
        for (auto& g : gs_ref) g.close("exec free");
    }

    guard.restore();
    if (k::native_grouped_no_sub16_gu() != guard.ns || k::iq_old_kernels() != guard.old) {
        std::printf("selector restoration FAIL\n");
        ++g_fail;
    }
    e0.close("e0 free");
    e1.close("e1 free");
    s.close("stream free");
    std::printf("tier_b_expert_bench: %d parity failures, %d timed pairs\n", parity_fail, timed);
    return (g_fail || parity_fail) ? 1 : 0;
}

int main(int argc, char** argv) {
    try {
        const int result = run_benchmark(argc, argv);
        // Locals (including selectors and noexcept CUDA owners) have unwound before this check.
        return result ? result : (g_fail ? 2 : 0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tier_b_expert_bench FAIL: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "tier_b_expert_bench FAIL: unknown exception\n");
    }
    return 2;
}
