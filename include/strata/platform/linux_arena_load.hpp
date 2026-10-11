// Opt-in Linux startup I/O only; independent of CUDA/HIP and the serving file tier.
#pragma once
#include "strata/core/pinned.hpp"

#if defined(__linux__)
#include <cstddef>
#include <sys/types.h>

namespace strata::platform {

// A narrow syscall seam for deterministic failure fixtures, not a runtime tuning API.
struct ArenaFileOps {
    int (*open)(const char*, int);
    ssize_t (*pread)(int, void*, size_t, off_t);
};

// One bounded aligned scratch buffer, plus a cached descriptor for reliable fallback.
// Never evicts cached pages. Requested file ranges must stay immutable during loading.
class LinuxArenaFile {
public:
    static constexpr size_t alignment = 4096;
    static constexpr size_t max_chunk = 8u << 20;
    explicit LinuxArenaFile(size_t chunk, const ArenaFileOps* ops = nullptr);
    ~LinuxArenaFile();
    LinuxArenaFile(const LinuxArenaFile&) = delete;
    LinuxArenaFile& operator=(const LinuxArenaFile&) = delete;
    bool open(const std::string& path, std::string& error);
    // Returns exactly the requested bytes in data (valid until next read), or refuses the load.
    bool read(uint64_t offset, size_t bytes, const uint8_t*& data, std::string& error);
    size_t chunk() const { return chunk_; }
    uint64_t direct_bytes() const { return direct_bytes_; }
    uint64_t buffered_bytes() const { return buffered_bytes_; }
    const std::string& fallback_reason() const { return fallback_reason_; }
private:
    void fallback(const std::string& reason);
    const ArenaFileOps* ops_;
    size_t chunk_;
    uint8_t* buffer_ = nullptr;
    int cached_ = -1, direct_ = -1;
    uint64_t size_ = 0, direct_bytes_ = 0, buffered_bytes_ = 0;
    std::string fallback_reason_;
};

// Same range/scatter layout and FNV seeds as load_experts_ranges; no writes outside each range.
// Scratch is <= 8 MiB + 8 KiB per worker, at most 16 workers (also bounded by layer count).
strata::core::LoadStats load_linux_arena_ranges(
    const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
    const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
    const std::atomic<int>* ready = nullptr, const ArenaFileOps* ops = nullptr);

} // namespace strata::platform
#endif
