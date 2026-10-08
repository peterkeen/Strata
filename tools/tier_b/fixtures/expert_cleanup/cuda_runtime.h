// Expert-only offline fake CUDA. Never linked to a CUDA runtime or used by a GPU build.
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0, cudaErrorUnknown = 999;
struct ExpertFakeStream { bool capturing = false; float* out = nullptr; size_t n = 0; };
struct ExpertFakeEvent {};
struct ExpertFakeGraph { float* out; size_t n; };
struct ExpertFakeExec { float* out; size_t n; };
using cudaStream_t = ExpertFakeStream*;
using cudaEvent_t = ExpertFakeEvent*;
using cudaGraph_t = ExpertFakeGraph*;
using cudaGraphExec_t = ExpertFakeExec*;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
enum cudaStreamCaptureMode { cudaStreamCaptureModeThreadLocal };
struct cudaDeviceProp { char name[64]; int l2CacheSize; };

namespace expert_fake {
inline std::map<void*, std::string> live;
inline std::vector<std::string> calls;
inline int fail_at = -1, cleanup_attempts = 0, captures_ended = 0;
inline bool partial_create = false, all_cleanup_fail = false, refuse_cleanup = false;
inline bool end_error = false, bad_layout = false;
inline int kernel_throw = 0;
inline bool ns = true, old = true;
inline cudaError_t step(const char* name) {
    calls.emplace_back(name);
    return static_cast<int>(calls.size()) == fail_at ? cudaErrorUnknown : cudaSuccess;
}
inline void track(void* p, const char* kind) { live.emplace(p, kind); }
inline void untrack(void* p) {
    if (live.erase(p) != 1) throw std::runtime_error("double/foreign release");
}
inline bool empty() { return live.empty(); }
inline void reset() {
    if (!empty()) throw std::runtime_error("resources remain between probes");
    calls.clear(); fail_at = -1; cleanup_attempts = captures_ended = 0;
    partial_create = all_cleanup_fail = refuse_cleanup = end_error = bad_layout = false; kernel_throw = 0;
}
inline void kernel_step(const char* name) {
    if (step(name) != cudaSuccess) {
        if (kernel_throw == 1) throw std::bad_alloc();
        if (kernel_throw == 2) throw 71;
        throw std::runtime_error("injected kernel exception");
    }
}
inline void output(float* p, size_t n) { std::fill(p, p + n, 1.0f); }
}
inline const char* cudaGetErrorString(cudaError_t) { return "expert-fake-injected"; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    auto e = expert_fake::step("props");
    std::strcpy(p->name, "OFFLINE expert fake CUDA"); p->l2CacheSize = 1; return e;
}
template<typename T> cudaError_t cudaMalloc(T** p, size_t n) {
    auto e = expert_fake::step("malloc");
    if (e && !expert_fake::partial_create) return e;
    *p = static_cast<T*>(std::malloc(n));
    if (!*p) throw std::bad_alloc();
    expert_fake::track(*p, "allocation"); return e;
}
inline cudaError_t cudaFree(void* p) {
    ++expert_fake::cleanup_attempts;
    auto e = expert_fake::step("free");
    if (expert_fake::refuse_cleanup) return cudaErrorUnknown;
    expert_fake::untrack(p); std::free(p);
    // Model a release that reports an earlier async error; later releases still must run.
    return expert_fake::all_cleanup_fail ? cudaErrorUnknown : e;
}
inline cudaError_t cudaMemset(void* p, int v, size_t n) {
    auto e = expert_fake::step("memset"); if (!e) std::memset(p, v, n); return e;
}
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t) {
    auto e = expert_fake::step("memsetAsync"); if (!e) std::memset(p, v, n); return e;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    auto e = expert_fake::step("copy"); if (!e) std::memcpy(dst, src, n); return e;
}
inline cudaError_t cudaStreamCreate(cudaStream_t* s) {
    auto e = expert_fake::step("streamCreate");
    if (e && !expert_fake::partial_create) return e;
    *s = new ExpertFakeStream; expert_fake::track(*s, "stream"); return e;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    ++expert_fake::cleanup_attempts;
    auto e = expert_fake::step("streamDestroy"); if (e || expert_fake::refuse_cleanup) return cudaErrorUnknown;
    if (s->capturing) throw std::runtime_error("destroyed active capture");
    expert_fake::untrack(s); delete s; return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return expert_fake::step("streamSync"); }
inline cudaError_t cudaEventCreate(cudaEvent_t* ev) {
    auto e = expert_fake::step("eventCreate");
    if (e && !expert_fake::partial_create) return e;
    *ev = new ExpertFakeEvent; expert_fake::track(*ev, "event"); return e;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t ev) {
    ++expert_fake::cleanup_attempts;
    auto e = expert_fake::step("eventDestroy"); if (e || expert_fake::refuse_cleanup) return cudaErrorUnknown;
    expert_fake::untrack(ev); delete ev; return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return expert_fake::step("record"); }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return expert_fake::step("eventSync"); }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
    auto e = expert_fake::step("elapsed"); *ms = 0.001f; return e;
}
inline cudaError_t cudaStreamBeginCapture(cudaStream_t s, cudaStreamCaptureMode) {
    auto e = expert_fake::step("begin"); if (e) return e;
    if (s->capturing) throw std::runtime_error("nested capture");
    s->capturing = true; s->out = nullptr; s->n = 0; return cudaSuccess;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t* g) {
    auto e = expert_fake::step("end");
    if (expert_fake::end_error) e = cudaErrorUnknown;
    if (!s->capturing) throw std::runtime_error("end without capture");
    s->capturing = false; ++expert_fake::captures_ended;
    if (e && !expert_fake::partial_create) { *g = nullptr; return e; }
    *g = new ExpertFakeGraph{s->out, s->n}; expert_fake::track(*g, "graph"); return e;
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* ex, cudaGraph_t g, unsigned long long) {
    auto e = expert_fake::step("instantiate");
    if (e && !expert_fake::partial_create) return e;
    *ex = new ExpertFakeExec{g->out, g->n}; expert_fake::track(*ex, "exec"); return e;
}
inline cudaError_t cudaGraphDestroy(cudaGraph_t g) {
    ++expert_fake::cleanup_attempts;
    auto e = expert_fake::step("graphDestroy"); if (e || expert_fake::refuse_cleanup) return cudaErrorUnknown;
    expert_fake::untrack(g); delete g; return cudaSuccess;
}
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t ex) {
    ++expert_fake::cleanup_attempts;
    auto e = expert_fake::step("execDestroy"); if (e || expert_fake::refuse_cleanup) return cudaErrorUnknown;
    expert_fake::untrack(ex); delete ex; return cudaSuccess;
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t ex, cudaStream_t) {
    auto e = expert_fake::step("launch"); if (!e) expert_fake::output(ex->out, ex->n); return e;
}
