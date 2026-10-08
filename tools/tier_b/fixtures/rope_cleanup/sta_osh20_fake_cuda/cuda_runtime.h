// Offline-only CUDA-shaped runtime for the sta-osh.20 ownership probe. No CUDA linkage.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct RopeFakeStream { bool capturing = false; std::vector<std::function<void()>> ops; };
struct RopeFakeEvent { int ordinal; };
struct RopeFakeGraph { std::vector<std::function<void()>> ops; };
struct RopeFakeExec { std::vector<std::function<void()>> ops; };
using cudaStream_t = RopeFakeStream*;
using cudaEvent_t = RopeFakeEvent*;
using cudaGraph_t = RopeFakeGraph*;
using cudaGraphExec_t = RopeFakeExec*;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorUnknown = 999;
constexpr int cudaStreamNonBlocking = 1, cudaStreamCaptureModeThreadLocal = 1;
constexpr int cudaDevAttrClockRate = 1;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
struct cudaDeviceProp { char name[32]; int multiProcessorCount; size_t totalGlobalMem; };

namespace rope_fake {
struct Handle { void* ptr; std::string kind; bool live = true; int attempts = 0; };
inline std::vector<Handle> handles;
inline std::map<std::string, int> calls;
inline int violations = 0, event_count = 0, active_captures = 0;
inline bool in_timing = false, injected = false;
inline const float* registered_cos = nullptr;
inline std::string primary_fail, secondary_fail;
inline bool fail(const char* op) {
    const std::string point = std::string(op) + ":" + std::to_string(++calls[op]);
    if (point == primary_fail || point == secondary_fail) {
        injected = true;
        std::fprintf(stderr, "INJECT %s\n", point.c_str());
        return true;
    }
    return false;
}
inline void own(void* ptr, const char* kind) { handles.push_back({ptr, kind}); }
inline bool release(void* ptr, const char* kind, const char* op) {
    auto it = handles.rbegin();
    while (it != handles.rend() && (it->ptr != ptr || !it->live)) ++it;
    if (it == handles.rend() || it->kind != kind) { ++violations; return false; }
    if (++it->attempts != 1) ++violations;
    if (std::strcmp(kind, "stream") == 0 || std::strcmp(kind, "allocation") == 0) {
        for (const auto& h : handles)
            if ((h.kind == "graph" || h.kind == "exec") && h.live && h.attempts == 0) ++violations;
    }
    if (std::strcmp(kind, "allocation") == 0 && ptr == registered_cos) ++violations;
    if (fail(op)) return false; // Failed release leaves storage live, but must not stop later attempts.
    it->live = false;
    return true;
}
inline void submit(cudaStream_t stream, std::function<void()> op) {
    if (stream->capturing) stream->ops.push_back(std::move(op));
    else op();
}
inline cudaError_t result(bool error) { return error ? cudaErrorUnknown : cudaSuccess; }
}
inline const char* cudaGetErrorString(cudaError_t) { return "rope-fake injected CUDA error"; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    if (rope_fake::fail("properties")) return cudaErrorUnknown;
    std::strcpy(p->name, "OFFLINE FAKE CUDA"); p->multiProcessorCount = 1; p->totalGlobalMem = 1 << 20;
    return cudaSuccess;
}
inline cudaError_t cudaDeviceGetAttribute(int* value, int, int) {
    *value = 1000; return rope_fake::result(rope_fake::fail("attribute"));
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, int) {
    if (rope_fake::fail("stream_create")) return cudaErrorUnknown;
    *s = new RopeFakeStream; rope_fake::own(*s, "stream"); return cudaSuccess;
}
inline cudaError_t cudaEventCreate(cudaEvent_t* e) {
    if (rope_fake::fail("event_create")) return cudaErrorUnknown;
    *e = new RopeFakeEvent{rope_fake::event_count++}; rope_fake::own(*e, "event"); return cudaSuccess;
}
template<class T> cudaError_t cudaMalloc(T** p, size_t n) {
    if (rope_fake::fail("malloc")) return cudaErrorUnknown;
    *p = static_cast<T*>(std::malloc(n));
    if (!*p) throw std::bad_alloc();
    rope_fake::own(*p, "allocation"); return cudaSuccess;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (rope_fake::fail("copy")) return cudaErrorUnknown;
    std::memcpy(dst, src, n); return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t s) {
    if (rope_fake::in_timing) ++rope_fake::violations; // Canonical uploads must stay outside event windows.
    if (rope_fake::fail("upload")) return cudaErrorUnknown;
    rope_fake::submit(s, [=] { std::memcpy(dst, src, n); }); return cudaSuccess;
}
inline cudaError_t cudaMemcpy2DAsync(void* dst, size_t dp, const void* src, size_t sp,
                                    size_t width, size_t height, cudaMemcpyKind, cudaStream_t s) {
    if (rope_fake::fail(s->capturing ? "capture_pack" : "pack")) return cudaErrorUnknown;
    rope_fake::submit(s, [=] {
        for (size_t r = 0; r < height; ++r)
            std::memcpy(static_cast<char*>(dst) + r * dp, static_cast<const char*>(src) + r * sp, width);
    }); return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) {
    return rope_fake::result(rope_fake::fail("stream_sync"));
}
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) {
    if (rope_fake::fail("event_record")) return cudaErrorUnknown;
    rope_fake::in_timing = e->ordinal == 0; return cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t) {
    return rope_fake::result(rope_fake::fail("event_sync"));
}
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
    *ms = 1; return rope_fake::result(rope_fake::fail("elapsed"));
}
inline cudaError_t cudaStreamBeginCapture(cudaStream_t s, int) {
    if (rope_fake::fail("begin")) return cudaErrorUnknown;
    if (s->capturing) ++rope_fake::violations;
    s->capturing = true; ++rope_fake::active_captures; s->ops.clear(); return cudaSuccess;
}
inline cudaError_t cudaStreamEndCapture(cudaStream_t s, cudaGraph_t* g) {
    if (!s->capturing) { ++rope_fake::violations; return cudaErrorUnknown; }
    s->capturing = false; --rope_fake::active_captures;
    const bool error = rope_fake::fail("end");
    if (error && std::getenv("ROPE_PROBE_NULL_PARTIAL")) { *g = nullptr; s->ops.clear(); return cudaErrorUnknown; }
    *g = new RopeFakeGraph{std::move(s->ops)}; rope_fake::own(*g, "graph");
    return rope_fake::result(error); // Deliberately return an owned output even on injected error.
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* ex, cudaGraph_t g, void*, void*, unsigned long long) {
    const bool error = rope_fake::fail("instantiate");
    if (error && std::getenv("ROPE_PROBE_NULL_PARTIAL")) { *ex = nullptr; return cudaErrorUnknown; }
    *ex = new RopeFakeExec{g->ops}; rope_fake::own(*ex, "exec"); return rope_fake::result(error);
}
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t ex, cudaStream_t) {
    if (rope_fake::fail("graph_launch")) return cudaErrorUnknown;
    if (!std::getenv("ROPE_PROBE_BAD_REPLAY")) for (const auto& op : ex->ops) op();
    return cudaSuccess;
}
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t ex) {
    if (!rope_fake::release(ex, "exec", "exec_destroy")) return cudaErrorUnknown;
    delete ex; return cudaSuccess;
}
inline cudaError_t cudaGraphDestroy(cudaGraph_t g) {
    if (!rope_fake::release(g, "graph", "graph_destroy")) return cudaErrorUnknown;
    delete g; return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t e) {
    if (!rope_fake::release(e, "event", "event_destroy")) return cudaErrorUnknown;
    delete e; return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
    if (s->capturing) ++rope_fake::violations;
    if (!rope_fake::release(s, "stream", "stream_destroy")) return cudaErrorUnknown;
    delete s; return cudaSuccess;
}
inline cudaError_t cudaFree(void* p) {
    if (!rope_fake::release(p, "allocation", "free")) return cudaErrorUnknown;
    std::free(p); return cudaSuccess;
}
