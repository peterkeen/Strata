// Compile the complete, unmodified benchmark translation unit against an offline fake runtime.
// The kernel stand-ins exercise layout/dataflow and ownership, NOT CUDA arithmetic or performance.
#define main rope_benchmark_entry
#include ROPE_BENCH_SOURCE
#undef main

namespace strata::kernels {
void build_rope_table(int n_rot, const RopeScaling&, int npos, float* c, float* s) {
    std::fill(c, c + size_t(npos) * (n_rot / 2), 1.0f);
    std::fill(s, s + size_t(npos) * (n_rot / 2), 0.0f);
}
bool native_norm_rope_usable(int, int) { return !std::getenv("ROPE_PROBE_DISABLED"); }
void rope_table_set(const float* c, const float*, int, const RopeScaling&) { rope_fake::registered_cos = c; }
void rope_table_release(const float* c) {
    if (c && rope_fake::registered_cos == c) rope_fake::registered_cos = nullptr;
}
RopeTab rope_table_for(const RopeScaling&) {
    return std::getenv("ROPE_PROBE_BAD_TABLE") ? RopeTab{} : RopeTab{rope_fake::registered_cos, nullptr, 2048};
}
void probe_launch(cudaStream_t s, const char* direct, const char* capture) {
    if (rope_fake::fail(s->capturing ? capture : direct)) {
        if (std::getenv("ROPE_PROBE_UNKNOWN")) throw 42;
        if (std::getenv("ROPE_PROBE_BAD_ALLOC")) throw std::bad_alloc();
        throw std::runtime_error("rope-fake kernel exception");
    }
}
// Simple matched operations, deliberately position-dependent so frozen replay is detectable.
void probe_norm(const float* x, int stride, const float* gamma, float* out, int width, int rows) {
    for (int r = 0; r < rows; ++r)
        for (int d = 0; d < width; ++d) out[r * width + d] = (x[r * stride + d] * 0.5f) * gamma[d];
}
void probe_rope(float* out, int rows, int width, const int* pos) {
    for (int r = 0; r < rows; ++r)
        for (int d = 0; d < 64; ++d) out[r * width + d] += pos[r] * 0.001f;
}
void native_qsa_rms_norm_weighted(const float* x, const float* gamma, float* out,
                                  int width, int rows, float, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    probe_launch(s, "norm", "capture_norm");
    rope_fake::submit(s, [=] { probe_norm(x, width, gamma, out, width, rows); });
}
void native_rope_apply(const float*, float* out, int rows, int width, int,
                       const RopeScaling&, const int* pos, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    probe_launch(s, "rope", "capture_rope");
    rope_fake::submit(s, [=] { probe_rope(out, rows, width, pos); });
}
void native_qsa_rms_norm_rope(const float* x, int stride, const float* gamma, float* out,
                              int rows, int width, int, float, const RopeScaling&, const int* pos, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    probe_launch(s, "fused", "capture_fused");
    rope_fake::submit(s, [=] {
        probe_norm(x, stride, gamma, out, width, rows);
        probe_rope(out, rows, width, pos);
        if (std::getenv("ROPE_PROBE_BAD_GATE")) out[0] += 1.0f;
    });
}
}

int main(int argc, char** argv) {
    const char* value = std::getenv("STRATA_ROPE_TABLE");
    const bool had_value = value != nullptr;
    const std::string original = value ? value : "";
    const char* fail = std::getenv("ROPE_PROBE_FAIL");
    const char* second = std::getenv("ROPE_PROBE_SECOND_FAIL");
    rope_fake::primary_fail = fail ? fail : "";
    rope_fake::secondary_fail = second ? second : "";
    const int result = rope_benchmark_entry(argc, argv);
    value = std::getenv("STRATA_ROPE_TABLE");
    const bool env_restored = had_value == (value != nullptr) && (!value || original == value);
    int unattempted = 0, live = 0;
    std::map<std::string, int> created;
    for (const auto& h : rope_fake::handles) {
        unattempted += h.attempts == 0;
        live += h.live;
        ++created[h.kind];
    }
    std::printf("PROBE rc=%d handles=%zu unattempted=%d live=%d captures=%d table=%d env=%d violations=%d injected=%d\n",
                result, rope_fake::handles.size(), unattempted, live, rope_fake::active_captures,
                rope_fake::registered_cos != nullptr, env_restored, rope_fake::violations, rope_fake::injected);
    for (const auto& item : created) std::printf("CREATED %s %d\n", item.first.c_str(), item.second);
    for (const auto& item : rope_fake::calls) std::printf("CALL %s %d\n", item.first.c_str(), item.second);
    // This wrapper must run after the benchmark's failure handler; std::exit/terminate cannot pass.
    return result;
}
