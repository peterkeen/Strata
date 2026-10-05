// Host-only snapshot transfers, following conversation_validation_test.cpp's
// GNU/ELF wrapping backend. No CUDA context, page allocator or COW implementation.
// Link with --wrap=cudaMemcpy --wrap=cudaGetLastError --wrap=cudaGetErrorString.
// For a standalone build, compile this file and src/core/conversation_snapshot.cpp
// with -DSHARED_KV_SNAPSHOT_STANDALONE and a minimal cuda_runtime.h shim on the
// include path (cudaError_t, cudaMemcpyKind, cudaGraphExec_t, cudaEvent_t and the
// three wrapped function declarations suffice). No CMake changes are required.
/* Example from the repo root (HOST_SHIM_DIR contains that header):
   g++ -std=c++20 -O2 -Wall -Wextra -Werror -DSHARED_KV_SNAPSHOT_STANDALONE \
     -I"$HOST_SHIM_DIR" -Iinclude src/core/conversation_snapshot.cpp \
     tests/core/shared_kv_snapshot_test.cpp \
     -Wl,--wrap=cudaMemcpy,--wrap=cudaGetLastError,--wrap=cudaGetErrorString \
     -o /tmp/shared_kv_snapshot_test && /tmp/shared_kv_snapshot_test
*/
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>

namespace {
size_t copy_calls = 0, copied_bytes = 0, fail_copy = 0;
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (++copy_calls == fail_copy) return cudaErrorInvalidValue;
    copied_bytes += n;
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetLastError() { return cudaSuccess; }
extern "C" const char* __wrap_cudaGetErrorString(cudaError_t) { return "injected host transfer failure"; }

#if defined(SHARED_KV_SNAPSHOT_STANDALONE)
// Resident restores must not call block movers, even for hybrid K8V4.
namespace strata::core {
strata::kernels::QsaAttnPools qsa_attn_pools(const QsaState&) { std::abort(); }
}
namespace strata::kernels {
void kv_stream_reset(const KvStreamMap&, void*) { std::abort(); }
void kv_ring_restore(const QsaAttnPools&, const KvHostPools&, int, int64_t, int64_t,
                     int64_t, const QsaShapes&, void*) { std::abort(); }
}
#endif

using namespace strata::core;
namespace {
using Buffers = std::array<std::vector<uint8_t>, 5>;
std::array<ConversationBuffer*, 5> payloads(ConversationKv& image) {
    return {&image.k, &image.v, &image.k_scale, &image.v_scale, &image.pooled};
}
bool equal(const ConversationKv& a, const ConversationKv& b) {
    return a.format == b.format && a.cells == b.cells && a.page_size == b.page_size &&
           a.k == b.k && a.v == b.v && a.k_scale == b.k_scale &&
           a.v_scale == b.v_scale && a.pooled == b.pooled;
}
struct Fixture {
    ModelGeometry g;
    QsaState st;
    Buffers data;
    std::array<size_t, 4> page_bytes{};
    Fixture(int format, int64_t slots = 7, int64_t pages = 32) {
        g.n_head_kv = 3; g.head_dim = 128; g.idx_key_dim = 7;
        const auto s = strata::kernels::qsa_real_shapes();
        st.kv_int8 = format == 1; st.kv_q4 = format == 2; st.kv_hybrid = format == 3;
        st.shared_kv = true; st.n_slots = slots; st.n_pages = pages;
        st.max_cells = pages * s.page_size; st.idx_pooled_rows = st.max_cells / s.idx_block + 2;
        const size_t rows = size_t(s.page_size * g.n_head_kv);
        const size_t q4 = strata::kernels::kv_q4_bytes_per_head(int(g.head_dim));
        page_bytes[0] = rows * (format == 2 ? q4 : size_t(g.head_dim) * (format == 0 ? 2 : 1));
        page_bytes[1] = format == 3 ? rows * q4 : page_bytes[0];
        page_bytes[2] = (format == 1 || format == 3) ? rows * size_t(g.head_dim / 64) * 2 : 0;
        page_bytes[3] = format == 3 ? 0 : page_bytes[2];
        for (size_t i = 0; i < 4; ++i) data[i].resize(size_t(slots) * page_bytes[i], 0xa5);
        data[4].resize(size_t(st.idx_pooled_rows * g.idx_key_dim) * sizeof(float), 0xa5);
        st.k_pool = reinterpret_cast<uint16_t*>(data[0].data());
        st.v_pool = reinterpret_cast<uint16_t*>(data[1].data());
        st.k_q = reinterpret_cast<int8_t*>(data[0].data()); st.v_q = reinterpret_cast<int8_t*>(data[1].data());
        st.k_q4 = data[0].data(); st.v_q4 = data[1].data();
        st.k_scale = reinterpret_cast<uint16_t*>(data[2].data());
        st.v_scale = reinterpret_cast<uint16_t*>(data[3].data());
        st.idx_pooled = reinterpret_cast<float*>(data[4].data());
        st.shared_page_table = {5, 1, 6, 2, -1, std::numeric_limits<int32_t>::max()};
    }
    void fill() {
        for (size_t i = 0; i < data.size(); ++i) for (size_t j = 0; j < data[i].size(); ++j)
            data[i][j] = uint8_t(11 + i * 29 + j * 7 + j / 251 + (i < 4 && page_bytes[i] ? j / page_bytes[i] * 19 : 0));
    }
    ConversationKv save(int64_t upto, bool index = true) const {
        ConversationKv image;
        std::string error;
        check(conversation_kv_save(image, st, g, upto, index, error), "capture host fixture");
        return image;
    }
    // Independent oracle: explicitly gather into an ordinary identity-layout
    // state and use the pre-existing non-shared snapshot path as the reference.
    void canonical_into(Fixture& canonical, size_t pages) const {
        canonical.st.shared_kv = false;
        for (size_t i = 0; i < 4; ++i) if (page_bytes[i]) for (size_t p = 0; p < pages; ++p)
            std::memcpy(canonical.data[i].data() + p * page_bytes[i],
                        data[i].data() + size_t(st.shared_page_table[p]) * page_bytes[i], page_bytes[i]);
        canonical.data[4] = data[4];
        canonical.st.idx_pooled = reinterpret_cast<float*>(canonical.data[4].data());
    }
};
void segment_inside_pages(ConversationKv& image) {
    for (auto* buffer : payloads(image)) {
        if (buffer->size() <= 17) continue;
        std::vector<uint8_t> flat(buffer->size());
        check(buffer->read(flat.data(), 0, flat.size()), "flatten reference payload");
        ConversationBuffer split;
        split.resize(17); // Forces an initial segment ending inside a K/V page.
        split.resize(flat.size()); // The next segment crosses many pages.
        check(split.visit(0, split.size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, flat.data() + at, n); return true;
        }), "construct deliberately unaligned segments");
        *buffer = std::move(split);
    }
}
void check_scattered(ConversationKv& image, const Fixture& dst) {
    const auto buffers = payloads(image);
    const size_t pages = size_t(image.cells / image.page_size);
    for (size_t i = 0; i < 4; ++i) {
        if (!dst.page_bytes[i]) continue;
        std::vector<uint8_t> logical(dst.page_bytes[i]);
        std::vector<bool> used(size_t(dst.st.n_slots));
        for (size_t p = 0; p < pages; ++p) {
            check(buffers[i]->read(logical.data(), p * logical.size(), logical.size()), "read canonical page");
            const size_t slot = size_t(dst.st.shared_page_table[p]);
            used[slot] = true;
            check(!std::memcmp(dst.data[i].data() + slot * logical.size(), logical.data(), logical.size()),
                  "destination physical page equals canonical logical page");
        }
        for (size_t slot = 0; slot < used.size(); ++slot) if (!used[slot])
            check(std::all_of(dst.data[i].begin() + slot * logical.size(),
                              dst.data[i].begin() + (slot + 1) * logical.size(),
                              [](uint8_t b) { return b == 0xa5; }), "unmapped pool page untouched");
    }
    std::vector<uint8_t> pooled(image.pooled.size());
    check(image.pooled.read(pooled.data(), 0, pooled.size()), "read canonical pooled rows");
    check(std::equal(pooled.begin(), pooled.end(), dst.data[4].begin()), "pooled indexer stays contiguous");
    check(std::all_of(dst.data[4].begin() + pooled.size(), dst.data[4].end(),
                      [](uint8_t b) { return b == 0xa5; }), "unused pooled rows untouched");
}
void roundtrip(int format, int64_t upto, bool index) {
    Fixture source(format), identity(format, 32, 32), destination(format);
    source.fill();
    const size_t pages = size_t((upto + 3) / 4);
    source.canonical_into(identity, pages);
    auto expected = identity.save(upto, index);
    auto image = source.save(upto, index);
    check(equal(image, expected), "shared gather matches identity canonical serialization");
    check(image.bytes() == conversation_kv_bytes(source.st, source.g, upto, index), "canonical byte admission");
    segment_inside_pages(image);
    destination.st.shared_page_table = {3, 6, 0, 4};
    const auto mapping = destination.st.shared_page_table;
    std::string error;
    const size_t calls = copy_calls;
    check(conversation_kv_validate(image, destination.st, destination.g, upto, index, error), "validate reserved destination");
    check(copy_calls == calls, "validation makes no CUDA copies");
    check(conversation_kv_restore(image, destination.st, destination.g, upto, index, error), "scatter into different reserved mapping");
    check(destination.st.shared_page_table == mapping, "restore never allocates or changes mappings");
    check_scattered(image, destination);
    uint64_t logical_hash = 0, shared_hash = 0;
    check(conversation_kv_verify(expected, identity.st, identity.g, upto, index, logical_hash, error), "identity verification");
    check(conversation_kv_verify(image, destination.st, destination.g, upto, index, shared_hash, error), "shared verification");
    check(logical_hash == shared_hash, "fingerprint depends on logical bytes, not placement or segmentation");
    if (pages) {
        const size_t offset = size_t(mapping[pages - 1]) * destination.page_bytes[0] + destination.page_bytes[0] - 1;
        destination.data[0][offset] ^= 1;
        check(!conversation_kv_verify(image, destination.st, destination.g, upto, index, shared_hash, error), "verify detects mapped final-page corruption");
        destination.data[0][offset] ^= 1;
        fail_copy = copy_calls + 1;
        check(!conversation_kv_restore(image, destination.st, destination.g, upto, index, error), "transfer error reported");
        fail_copy = 0;
    }
}
void invalid_mappings(int format) {
    Fixture f(format);
    f.fill();
    auto image = f.save(13);
    auto reject = [&](const char* label) {
        const auto before = f.data;
        const auto saved = image;
        const size_t calls = copy_calls;
        std::string error;
        size_t bytes = 0;
        uint64_t hash = 123;
        check(!conversation_kv_capture_bytes(image, f.st, f.g, 13, true, bytes, error), label);
        check(!conversation_kv_save(image, f.st, f.g, 13, true, error, 13), "invalid mapping rejected even for reusable logical pages");
        check(!conversation_kv_validate(saved, f.st, f.g, 13, true, error), "invalid mapping prevalidation");
        check(!conversation_kv_restore(saved, f.st, f.g, 13, true, error), "invalid mapping restore rejected before writes");
        check(!conversation_kv_verify(saved, f.st, f.g, 13, true, hash, error), "invalid mapping verify rejected before reads");
        check(copy_calls == calls && f.data == before && equal(image, saved) && hash == 123,
              "invalid mapping leaves source, destination, image and fingerprint untouched");
    };
    f.st.shared_page_table.resize(3); reject("short mapping prefix rejected");
    f.st.shared_page_table = {5, 1, 6, -1}; reject("late unmapped page rejected");
    f.st.shared_page_table.back() = int32_t(f.st.n_slots); reject("late physical ID at pool limit rejected");
    f.st.shared_page_table.back() = std::numeric_limits<int32_t>::max(); reject("huge physical ID rejected");
    f.st.shared_page_table = {5, 1, 6, 2};
    f.st.kv_mode = 1; reject("shared streaming rejected");
    f.st.kv_mode = 2; reject("shared ring rejected");
    f.st.kv_mode = 0;
    {
        // The logical one-page payload fits size_t, but physical slot 5's
        // offset does not. Reject before allocation, not merely during a copy.
        auto huge_geometry = f.g;
        huge_geometry.n_head_kv = std::numeric_limits<int64_t>::max() / 512;
        const size_t calls = copy_calls;
        std::string error;
        size_t bytes = 0;
        check(!conversation_kv_capture_bytes({}, f.st, huge_geometry, 1, true, bytes, error) &&
              error.find("physical byte offset overflow") != std::string::npos,
              "physical offset overflow rejected independently of logical payload size");
        ConversationKv unpublished;
        check(!conversation_kv_save(unpublished, f.st, huge_geometry, 1, true, error) &&
              unpublished.k.empty() && copy_calls == calls, "physical overflow rejected before allocation or transfers");
    }
    f.st.shared_page_table.clear();
    auto empty = f.save(0);
    std::string error;
    check(conversation_kv_restore(empty, f.st, f.g, 0, true, error), "zero extent needs no mapped pages");
    f.st.shared_kv = false;
    check(conversation_kv_bytes(f.st, f.g, 13, true) == 0, "non-shared resident still requires full identity pool capacity");
}
void incremental(int format) {
    Fixture f(format);
    f.fill();
    auto retained = f.save(9);
    segment_inside_pages(retained);
    auto resident = f.save(16);
    f.st.shared_page_table = {3, 6, 0, 4};
    std::string error;
    check(conversation_kv_restore(resident, f.st, f.g, 16, true, error), "remap live pool before logical prefix reuse");
    // Dirty token 7 lies in logical page 1, regardless of its physical slot.
    for (size_t i = 0; i < 4; ++i) if (f.page_bytes[i]) for (size_t p = 1; p < 4; ++p)
        std::memset(f.data[i].data() + size_t(f.st.shared_page_table[p]) * f.page_bytes[i], 91, f.page_bytes[i]);
    const size_t row_bytes = size_t(f.g.idx_key_dim) * sizeof(float);
    std::fill(f.data[4].begin() + row_bytes, f.data[4].end(), 91);
    auto full = f.save(13);
    copied_bytes = 0;
    size_t reused = 0;
    check(conversation_kv_save(retained, f.st, f.g, 13, true, error, 7, &reused), "incremental logical growth capture");
    check(equal(full, retained), "incremental gather equals full capture with partial-page rewrite");
    const size_t expected_reused = std::accumulate(f.page_bytes.begin(), f.page_bytes.end(), row_bytes);
    size_t total = 0;
    for (const auto* buffer : payloads(full)) total += buffer->size();
    check(reused == expected_reused && copied_bytes + reused == total, "only complete logical pages and pooled rows skip copies");
    auto short_full = f.save(5);
    check(conversation_kv_save(retained, f.st, f.g, 5, true, error, 3), "logical rewind capture");
    check(equal(retained, short_full), "rewind recopies partial logical page");
}
void large_segment_crossing() {
    // Hybrid K pages are 1536 B: the fixed 16 MiB segment boundary splits a
    // physical page. V pages are 864 B, and scales 48 B, so translation must
    // independently use all three strides as well as split large visits.
    constexpr int64_t slots = 11003, pages = 10925;
    Fixture source(3, slots, slots + 20), destination(3, slots, slots + 20);
    source.st.shared_page_table.resize(pages);
    destination.st.shared_page_table.resize(pages);
    for (int64_t p = 0; p < pages; ++p) {
        source.st.shared_page_table[size_t(p)] = int32_t((p * 37) % slots);
        destination.st.shared_page_table[size_t(p)] = int32_t((p * 41 + 7) % slots);
    }
    source.fill();
    const int64_t upto = pages * 4 - 3; // Preserve padded bytes of the last page.
    auto image = source.save(upto);
    check(image.k.size() > ConversationBuffer::segment_bytes &&
          ConversationBuffer::segment_bytes % source.page_bytes[0] != 0, "large fixture genuinely splits a K page at segment boundary");
    std::string error;
    check(conversation_kv_restore(image, destination.st, destination.g, upto, true, error), "scatter across fixed 16 MiB boundary");
    check_scattered(image, destination);
    auto restored = destination.save(upto);
    check(equal(image, restored), "large segmented shared A/B/A exactness");
    uint64_t a = 0, b = 0;
    check(conversation_kv_verify(image, source.st, source.g, upto, true, a, error), "large source verification");
    check(conversation_kv_verify(image, destination.st, destination.g, upto, true, b, error) && a == b,
          "large remapped fingerprint invariant");
}
}
int main() {
    for (int format : {0, 1, 2, 3}) {
        for (int64_t upto : {0, 1, 3, 4, 5, 13, 16}) for (bool index : {false, true}) roundtrip(format, upto, index);
        invalid_mappings(format);
        incremental(format);
    }
    large_segment_crossing();
    std::printf("shared_kv_snapshot_test: %d host-only checks passed\n", checks);
}
