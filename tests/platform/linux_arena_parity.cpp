// Native CUDA/HIP integration gate: compare the unchanged buffered loader and Linux opt-in loader
// in ONE registered arena, sequentially. Large-file mode is for an operator-authorized quiet window.
#include "strata/core/pinned.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

int main(int argc, char** argv) {
    std::string path, mode = "parity";
    uint64_t bytes = 131209, layers = 3;
    int threads = 3;
    bool fixture = argc == 1;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 == argc) { std::fprintf(stderr, "missing value for %s\n", arg.c_str()); return 2; }
        const char* val = argv[++i];
        if (arg == "--file") path = val;
        else if (arg == "--mode") mode = val;
        else if (arg == "--bytes") bytes = std::strtoull(val, nullptr, 10);
        else if (arg == "--layers") layers = std::strtoull(val, nullptr, 10);
        else if (arg == "--threads") threads = std::atoi(val);
        else { std::fprintf(stderr, "unknown option %s\n", arg.c_str()); return 2; }
    }
    struct Cleanup { std::string path; ~Cleanup() { if (!path.empty()) unlink(path.c_str()); } } cleanup;
    if (fixture) {
        char p[] = "/tmp/strata-native-arena-XXXXXX";
        const int fd = mkstemp(p);
        if (fd < 0) return 1;
        close(fd);
        path = cleanup.path = p;
        std::ofstream f(path, std::ios::binary);
        for (uint64_t i = 0; i < bytes; ++i) f.put(char((i * 97 + i / 257) % 251));
        if (!f) return 1;
    }
    if (path.empty() || bytes == 0 || layers == 0 || layers > bytes || layers > 100000 || bytes > SIZE_MAX - 4096 ||
        (mode != "parity" && mode != "buffered" && mode != "direct")) {
        std::fprintf(stderr, "usage: linux_arena_parity [--file experts.bin --bytes N --layers N --threads N --mode parity|buffered|direct]\n");
        return 2;
    }
    strata::core::PinnedArena arena(bytes + 4096);
    if (!arena.valid()) { std::fprintf(stderr, "arena allocation refused: %s\n", arena.note.c_str()); return 1; }
    std::fprintf(stderr, "backing: %s\n", arena.note.c_str());
    if (arena.registered_bytes != arena.capacity) {
        std::fprintf(stderr, "native parity gate requires a fully registered arena (got %llu/%llu B)\n",
                     (unsigned long long) arena.registered_bytes, (unsigned long long) arena.capacity);
        return 1;
    }
    std::vector<uint64_t> offsets, sizes;
    for (uint64_t l = 0; l < layers; ++l) {
        const uint64_t a = (bytes / layers) * l, b = l + 1 == layers ? bytes : (bytes / layers) * (l + 1);
        offsets.push_back(a); sizes.push_back(b - a);
    }
    if (mode != "parity") {
        // Separate-process micro A/B; no pre-touch of the payload beyond what registration already does.
        std::fill_n(arena.data() + bytes, size_t(4096), uint8_t(0xa5));
        const auto st = mode == "direct"
            ? strata::core::load_experts_direct(path, arena.data(), offsets, sizes, threads, 8u << 20)
            : strata::core::load_experts_ranges(path, arena.data(), offsets, sizes, threads, 8u << 20);
        if (!st.ok) { std::fprintf(stderr, "load refused: %s\n", st.error.c_str()); return 1; }
        for (uint64_t i = bytes; i < arena.capacity; ++i) if (arena.data()[i] != 0xa5) return 1;
        std::printf("mode=%s path=%s (%s) bytes=%llu seconds=%.6f read_s=%.6f copy_hash_s=%.6f direct_payload=%llu buffered_payload=%llu\n",
                    mode.c_str(), mode == "buffered" ? "buffered" : st.io_path.c_str(), st.io_note.c_str(),
                    (unsigned long long) st.bytes, st.seconds, st.read_seconds, st.copy_seconds,
                    (unsigned long long) st.direct_bytes, (unsigned long long) st.buffered_bytes);
        for (size_t l = 0; l < st.layer_checksums.size(); ++l)
            std::printf("layer %zu %016llx\n", l, (unsigned long long) st.layer_checksums[l]);
        return st.bytes == bytes && (mode == "buffered" || st.io_path == "direct") ? 0 : 1;
    }
    // A single arena avoids two copies of a production-size pack. Reset bytes before the direct pass so
    // a skipped write cannot accidentally match what the buffered pass left behind.
    std::fill_n(arena.data(), size_t(arena.capacity), uint8_t(0xa5));
    const auto cached = strata::core::load_experts_ranges(path, arena.data(), offsets, sizes, threads, 8u << 20);
    if (!cached.ok) { std::fprintf(stderr, "buffered refused: %s\n", cached.error.c_str()); return 1; }
    const uint64_t cached_hash = strata::core::fnv1a64(arena.data(), bytes);
    std::fill_n(arena.data(), size_t(bytes), uint8_t(0x5a));
    const auto direct = strata::core::load_experts_direct(path, arena.data(), offsets, sizes, threads, 8u << 20);
    if (!direct.ok) { std::fprintf(stderr, "direct refused: %s\n", direct.error.c_str()); return 1; }
    const uint64_t direct_hash = strata::core::fnv1a64(arena.data(), bytes);
    for (uint64_t i = bytes; i < arena.capacity; ++i) if (arena.data()[i] != 0xa5) return 1;
    if (cached.bytes != bytes || direct.bytes != bytes || cached.layer_checksums != direct.layer_checksums ||
        cached_hash != direct_hash) { std::fprintf(stderr, "arena byte/hash parity FAILED\n"); return 1; }
    std::printf("PASS arena parity: %llu B, %llu per-layer checksums, whole FNV %016llx; path=%s (%s)\n",
                (unsigned long long) bytes, (unsigned long long) layers, (unsigned long long) direct_hash,
                direct.io_path.c_str(), direct.io_note.c_str());
    std::printf("buffered %.3f s read %.3f s copy+hash %.3f s; candidate %.3f s read %.3f s copy+hash %.3f s\n",
                cached.seconds, cached.read_seconds, cached.copy_seconds, direct.seconds, direct.read_seconds, direct.copy_seconds);
    // Fallback parity is useful, but it is NOT proof of direct-I/O parity on this host/filesystem.
    if (direct.io_path != "direct") { std::fprintf(stderr, "native direct-I/O gate NOT passed: candidate fell back\n"); return 1; }
    return 0;
}
