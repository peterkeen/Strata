#include "strata/platform/linux_arena_load.hpp"
#if defined(__linux__)
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <linux/magic.h>
#include <thread>
#include <unistd.h>

namespace strata::platform {
namespace {
int real_open(const char* path, int flags) { return ::open(path, flags); }
const ArenaFileOps real_ops{real_open, ::pread};
bool unsupported(int e) {
    return e == EINVAL || e == EOPNOTSUPP || e == ENOSYS || e == ENOTTY || e == EPERM;
}
uint64_t hash_bytes(const uint8_t* p, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
uint64_t elapsed_ns(std::chrono::steady_clock::time_point start) {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
}
}

LinuxArenaFile::LinuxArenaFile(size_t chunk, const ArenaFileOps* ops)
    : ops_(ops ? ops : &real_ops), chunk_(std::min(chunk, max_chunk)) {
    if (chunk_ != 0) {
        void* p = nullptr;
        if (posix_memalign(&p, alignment, chunk_ + 2 * alignment) == 0) buffer_ = static_cast<uint8_t*>(p);
    }
}
LinuxArenaFile::~LinuxArenaFile() {
    if (direct_ >= 0) ::close(direct_);
    if (cached_ >= 0) ::close(cached_);
    std::free(buffer_);
}
void LinuxArenaFile::fallback(const std::string& reason) {
    if (fallback_reason_.empty()) fallback_reason_ = reason;
    if (direct_ >= 0) ::close(direct_);
    direct_ = -1;
}
bool LinuxArenaFile::open(const std::string& path, std::string& error) {
    if (cached_ >= 0) { error = "arena reader already open"; return false; }
    if (!buffer_) { error = "cannot allocate bounded arena read buffer (or zero chunk)"; return false; }
    do { cached_ = ops_->open(path.c_str(), O_RDONLY | O_CLOEXEC); } while (cached_ < 0 && errno == EINTR);
    if (cached_ < 0) { error = "cannot open " + path + ": " + std::strerror(errno); return false; }
    struct stat cached_stat{};
    int rc;
    do { rc = fstat(cached_, &cached_stat); } while (rc != 0 && errno == EINTR);
    if (rc != 0 || cached_stat.st_size < 0 || !S_ISREG(cached_stat.st_mode)) {
        error = "arena input is not a stat-able regular file: " + path;
        return false;
    }
    size_ = uint64_t(cached_stat.st_size);
    // tmpfs accepts O_DIRECT but ignores it: do not claim cache bypass there.
    if (ops_ == &real_ops) {
        struct statfs fs{};
        do { rc = fstatfs(cached_, &fs); } while (rc != 0 && errno == EINTR);
        if (rc != 0 || fs.f_type == TMPFS_MAGIC) {
            fallback(rc != 0 ? "cannot verify filesystem for O_DIRECT: " + std::string(std::strerror(errno))
                             : "tmpfs ignores O_DIRECT");
            return true;
        }
    }
    do { direct_ = ops_->open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT); } while (direct_ < 0 && errno == EINTR);
    if (direct_ < 0) {
        fallback("O_DIRECT open failed: " + std::string(std::strerror(errno)));
        return true;
    }
    struct stat direct_stat{};
    do { rc = fstat(direct_, &direct_stat); } while (rc != 0 && errno == EINTR);
    if (rc != 0 || direct_stat.st_dev != cached_stat.st_dev ||
        direct_stat.st_ino != cached_stat.st_ino || direct_stat.st_size != cached_stat.st_size) {
        error = "arena input changed while opening direct and buffered descriptors: " + path;
        return false;
    }
    return true;
}
bool LinuxArenaFile::read(uint64_t offset, size_t bytes, const uint8_t*& data, std::string& error) {
    data = nullptr;
    if (cached_ < 0 || bytes == 0 || bytes > chunk_) { error = "invalid bounded arena read"; return false; }
    if (offset > size_ || bytes > size_ - offset) {
        error = "short arena read at offset " + std::to_string(offset) + ": requested " +
                std::to_string(bytes) + " B beyond file size " + std::to_string(size_);
        return false;
    }
    if (direct_ >= 0) {
        const uint64_t start = offset / alignment * alignment;
        const size_t skip = size_t(offset - start), need = skip + bytes;
        const size_t window = (need + alignment - 1) / alignment * alignment;
        size_t done = 0;
        while (done < need) {
            // A direct short read can be continued only with aligned buffer, offset AND length.
            const ssize_t got = ops_->pread(direct_, buffer_ + done, window - done, off_t(start + done));
            if (got < 0) {
                const int e = errno;
                if (e == EINTR) continue;
                if (unsupported(e)) { fallback("O_DIRECT read unsupported/alignment refused: " + std::string(std::strerror(e))); break; }
                error = "O_DIRECT read error at offset " + std::to_string(start + done) + ": " + std::strerror(e);
                return false; // EIO and other real failures must not be hidden by a second read.
            }
            if (got == 0) { error = "short O_DIRECT read at offset " + std::to_string(start + done); return false; }
            if (size_t(got) > window - done) { error = "invalid O_DIRECT read count"; return false; }
            done += size_t(got);
            if (done < need && done % alignment != 0) { fallback("unaligned partial O_DIRECT read; retry requested bytes buffered"); break; }
        }
        if (direct_ >= 0) {
            data = buffer_ + skip;
            direct_bytes_ += bytes;
            return true; // rounded window may end past EOF; only the requested bytes are required.
        }
    }
    size_t done = 0;
    while (done < bytes) {
        const ssize_t got = ops_->pread(cached_, buffer_ + done, bytes - done, off_t(offset + done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            error = std::string(got == 0 ? "short buffered arena read" : "buffered arena read error") +
                    " at offset " + std::to_string(offset + done) + (got < 0 ? ": " + std::string(std::strerror(errno)) : "");
            return false;
        }
        if (size_t(got) > bytes - done) { error = "invalid buffered read count"; return false; }
        done += size_t(got);
    }
    data = buffer_;
    buffered_bytes_ += bytes;
    return true;
}

strata::core::LoadStats load_linux_arena_ranges(
    const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
    const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
    const std::atomic<int>* ready, const ArenaFileOps* ops) {
    strata::core::LoadStats st;
    st.ok = false;
    st.io_path = "failed-before-payload";
    st.layers = layer_off.size();
    auto fail_input = [&](const char* error) { st.seconds = -1; st.error = error; return st; };
    if (!dst || chunk == 0 || layer_off.empty() || layer_off.size() != layer_bytes.size() ||
        layer_off.size() >= size_t(std::numeric_limits<int>::max())) return fail_input("invalid arena ranges/chunk/destination");
    uint64_t end = 0;
    for (size_t i = 0; i < layer_off.size(); ++i) {
        const uint64_t off = layer_off[i], n = layer_bytes[i];
        if (off < end || off > uint64_t(std::numeric_limits<off_t>::max()) ||
            n > uint64_t(std::numeric_limits<off_t>::max()) - off ||
            n > uint64_t(SIZE_MAX) - off || n > UINT64_MAX - st.bytes) return fail_input("overlapping or overflowing arena ranges");
        end = off + n;
        st.bytes += n;
    }
    const size_t bounded_chunk = size_t(std::min<uint64_t>(chunk, LinuxArenaFile::max_chunk));
    threads = std::max(1, std::min({threads, 16, int(layer_off.size())}));
    st.layer_checksums.resize(layer_off.size(), 1469598103934665603ull);
    const auto start = std::chrono::steady_clock::now();
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::atomic<uint64_t> read_ns{0}, copy_ns{0};
    std::mutex mu;
    auto worker = [&]() {
        LinuxArenaFile file(bounded_chunk, ops);
        std::string error;
        auto fail = [&]() { std::lock_guard<std::mutex> lock(mu); if (st.error.empty()) st.error = error; bad = true; };
        if (!file.open(path, error)) { fail(); return; }
        for (;;) {
            const size_t layer = next.fetch_add(1);
            if (layer >= layer_off.size() || bad) break;
            // Preserve the neighboring-slice fence: no arena page touched during its registration.
            if (ready) while (ready->load(std::memory_order_acquire) <= int(layer) + 1 && !bad) std::this_thread::yield();
            if (bad) break;
            uint64_t h = 1469598103934665603ull;
            for (uint64_t pos = 0; pos < layer_bytes[layer] && !bad;) {
                const size_t n = size_t(std::min<uint64_t>(bounded_chunk, layer_bytes[layer] - pos));
                const uint8_t* data = nullptr;
                auto t = std::chrono::steady_clock::now();
                const bool ok = file.read(layer_off[layer] + pos, n, data, error);
                read_ns.fetch_add(elapsed_ns(t), std::memory_order_relaxed);
                if (!ok) { error = "layer " + std::to_string(layer) + ": " + error; fail(); break; }
                t = std::chrono::steady_clock::now();
                std::memcpy(dst + layer_off[layer] + pos, data, n);
                h = hash_bytes(data, n, h);
                copy_ns.fetch_add(elapsed_ns(t), std::memory_order_relaxed);
                pos += n;
            }
            if (!bad) st.layer_checksums[layer] = h;
        }
        std::lock_guard<std::mutex> lock(mu);
        st.direct_bytes += file.direct_bytes();
        st.buffered_bytes += file.buffered_bytes();
        if (st.io_note.empty()) st.io_note = file.fallback_reason();
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& thread : pool) thread.join();
    st.io_path = st.direct_bytes ? (st.buffered_bytes ? "mixed-direct-buffered-fallback" : "direct")
                                : (bad && !st.buffered_bytes ? "failed-before-payload" : "buffered-fallback");
    st.read_seconds = double(read_ns.load()) / 1e9;
    st.copy_seconds = double(copy_ns.load()) / 1e9;
    st.ok = !bad;
    st.seconds = st.ok ? double(elapsed_ns(start)) / 1e9 : -1.0;
    if (!st.ok) st.layer_checksums.clear(); // Never publish checksums for a partly written arena.
    return st;
}
} // namespace strata::platform
#endif
