// Pure C++ startup-loader fixtures: no GPU headers/runtime/model required.
#include "strata/platform/linux_arena_load.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using strata::platform::ArenaFileOps;
using strata::platform::LinuxArenaFile;
using strata::platform::load_linux_arena_ranges;
namespace {
void check(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
uint64_t hash(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
struct Fixture {
    std::string path;
    std::vector<uint8_t> bytes;
    Fixture() : bytes(65536 + 137) {
        char p[] = "/tmp/strata-arena-fixture-XXXXXX";
        const int fd = mkstemp(p);
        check(fd >= 0, "mkstemp");
        path = p;
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t((i * 97 + i / 257) % 251);
        const ssize_t n = write(fd, bytes.data(), bytes.size());
        close(fd);
        check(n == ssize_t(bytes.size()), "write fixture");
    }
    ~Fixture() { unlink(path.c_str()); }
};
// Fault tests run with exactly one worker. Fake direct opens strip O_DIRECT, but verify every
// subsequent direct request's buffer/offset/length alignment. Real O_DIRECT is tested separately.
enum class Fault { None, OpenUnsupported, OpenInterrupted, ReadInterrupted, AlignedPartial,
                   UnalignedPartial, ReadUnsupported, LateUnsupported, BufferedPartial, Eof, IoError, BufferedEof };
Fault fault;
int direct_fd = -1, direct_calls = 0, cached_calls = 0, open_calls = 0;
void reset(Fault f) { fault = f; direct_fd = -1; direct_calls = cached_calls = open_calls = 0; }
int injected_open(const char* path, int flags) {
    ++open_calls;
    if (fault == Fault::OpenInterrupted && (open_calls == 1 || open_calls == 3)) { errno = EINTR; return -1; }
    if (flags & O_DIRECT) {
        if (fault == Fault::OpenUnsupported || fault == Fault::BufferedPartial || fault == Fault::BufferedEof) {
            errno = EOPNOTSUPP; return -1;
        }
        direct_fd = ::open(path, flags & ~O_DIRECT);
        return direct_fd;
    }
    return ::open(path, flags);
}
ssize_t injected_pread(int fd, void* buf, size_t n, off_t off) {
    if (fd == direct_fd) {
        check(uintptr_t(buf) % 4096 == 0 && n % 4096 == 0 && off % 4096 == 0, "direct request is aligned");
        ++direct_calls;
        if (fault == Fault::ReadInterrupted && direct_calls == 1) { errno = EINTR; return -1; }
        if (fault == Fault::Eof) return 0;
        if (fault == Fault::IoError) { errno = EIO; return -1; }
        if (fault == Fault::ReadUnsupported || (fault == Fault::LateUnsupported && direct_calls > 1)) {
            errno = EINVAL; return -1;
        }
        if (fault == Fault::AlignedPartial && direct_calls == 1) n = std::min<size_t>(4096, n);
        if (fault == Fault::UnalignedPartial && direct_calls == 1) n = 31;
    } else {
        ++cached_calls;
        if (fault == Fault::BufferedEof) return 0;
        if (fault == Fault::BufferedPartial) {
            if (cached_calls == 1) { errno = EINTR; return -1; }
            n = std::min<size_t>(127, n);
        }
    }
    return ::pread(fd, buf, n, off);
}
const ArenaFileOps injected_ops{injected_open, injected_pread};
void parity(Fixture& fixture, const ArenaFileOps* ops, uint64_t chunk, int threads,
            const std::vector<uint64_t>& offsets, const std::vector<uint64_t>& sizes,
            const std::string& expected_path = "") {
    std::vector<uint8_t> dst(fixture.bytes.size() + 34, 0xa5), expected = dst;
    // Intentionally unaligned arena destination, with head/tail and gap canaries.
    auto st = load_linux_arena_ranges(fixture.path, dst.data() + 1, offsets, sizes, threads, chunk, nullptr, ops);
    check(st.ok, "loader succeeded: " + st.error);
    uint64_t total = 0;
    for (size_t i = 0; i < offsets.size(); ++i) {
        const auto off = size_t(offsets[i]), n = size_t(sizes[i]);
        std::memcpy(expected.data() + 1 + off, fixture.bytes.data() + off, n);
        check(st.layer_checksums[i] == hash(fixture.bytes.data() + off, n), "byte-identical per-layer FNV");
        total += n;
    }
    check(dst == expected, "byte-identical arena; gaps and head/tail guards untouched");
    check(st.bytes == total && st.direct_bytes + st.buffered_bytes == total, "payload counts cover exactly the ranges");
    check(st.seconds >= 0 && st.read_seconds >= 0 && st.copy_seconds >= 0, "load timing");
    if (!expected_path.empty()) check(st.io_path == expected_path, "effective path " + st.io_path + ", expected " + expected_path);
    std::cout << "parity: " << st.io_path << " chunk=" << chunk << " threads=" << threads << " " << st.io_note << '\n';
}
void failure(Fixture& fixture, Fault f, const std::string& needle) {
    reset(f);
    std::vector<uint8_t> dst(fixture.bytes.size(), 0xa5);
    auto st = load_linux_arena_ranges(fixture.path, dst.data(), {0}, {8192}, 1, 8192, nullptr, &injected_ops);
    check(!st.ok && st.seconds == -1 && st.error.find(needle) != std::string::npos, "fatal load failure: " + st.error);
    check(st.layer_checksums.empty(), "no checksum published for refused arena");
    check(cached_calls == (f == Fault::BufferedEof ? 1 : 0), "EOF/EIO does not hide failure by retrying buffered");
}
void readiness(Fixture& fixture) {
    std::vector<uint8_t> dst(4096, 0xa5);
    std::atomic<int> ready{1};
    std::promise<void> started;
    auto future = std::async(std::launch::async, [&] {
        started.set_value();
        return load_linux_arena_ranges(fixture.path, dst.data(), {0}, {4096}, 1, 4096, &ready);
    });
    started.get_future().wait();
    const bool waited = future.wait_for(std::chrono::milliseconds(30)) == std::future_status::timeout;
    ready.store(2, std::memory_order_release);
    auto st = future.get();
    check(waited, "wait for neighboring slice, not just own slice");
    check(st.ok && std::equal(dst.begin(), dst.end(), fixture.bytes.begin()), "release fence allows exact load");
}
}
int main() {
    try {
        Fixture fixture;
        // Real filesystem direct I/O, or truthful fallback where unsupported. EOF is not sector aligned.
        parity(fixture, nullptr, 8192, 3, {0, 12289, 65521}, {8192, 17003, 152});
        parity(fixture, nullptr, 7777, 32, {3, 20000, 65530}, {16001, 30000, 143});
        parity(fixture, nullptr, 1, 1, {65536}, {137});
        // Deterministic direct/fallback paths; full and partial interrupted reads.
        for (Fault f : {Fault::None, Fault::OpenInterrupted, Fault::ReadInterrupted, Fault::AlignedPartial}) {
            reset(f);
            parity(fixture, &injected_ops, 16384, 1, {7, 65521}, {20000, 152}, "direct");
            check(cached_calls == 0, "direct never used cached pread");
        }
        for (Fault f : {Fault::OpenUnsupported, Fault::ReadUnsupported, Fault::UnalignedPartial, Fault::BufferedPartial}) {
            reset(f);
            parity(fixture, &injected_ops, 16384, 1, {7, 65521}, {20000, 152}, "buffered-fallback");
            check(cached_calls > 0, "fallback actually read cached fd");
        }
        reset(Fault::LateUnsupported);
        parity(fixture, &injected_ops, 4096, 1, {0}, {16384}, "mixed-direct-buffered-fallback");
        failure(fixture, Fault::Eof, "short O_DIRECT");
        failure(fixture, Fault::IoError, "Input/output error");
        failure(fixture, Fault::BufferedEof, "short buffered");
        // Input validation and real truncation refuse the load.
        std::vector<uint8_t> dst(fixture.bytes.size(), 0xa5);
        for (auto st : {
            load_linux_arena_ranges(fixture.path, dst.data(), {0}, {1, 2}, 1, 4096),
            load_linux_arena_ranges(fixture.path, dst.data(), {0}, {1}, 1, 0),
            load_linux_arena_ranges(fixture.path, dst.data(), {0, 4}, {8, 8}, 1, 4096),
            load_linux_arena_ranges(fixture.path, dst.data(), {UINT64_MAX}, {8}, 1, 4096),
            load_linux_arena_ranges(fixture.path, dst.data(), {0}, {uint64_t(fixture.bytes.size()) + 1}, 1, 4096),
            load_linux_arena_ranges(fixture.path + "-missing", dst.data(), {0}, {4096}, 1, 4096)}) {
            check(!st.ok && st.seconds == -1 && !st.error.empty(), "invalid/truncated input refused");
        }
        // A file can truncate after open: captured size isn't a substitute for checking every pread.
        {
            LinuxArenaFile file(4096);
            std::string error;
            check(file.open(fixture.path, error), "open before truncation");
            check(truncate(fixture.path.c_str(), 4096) == 0, "truncate");
            const uint8_t* data = nullptr;
            check(!file.read(8192, 4096, data, error) && error.find("short") != std::string::npos, "post-open truncation refused");
        }
        // Restore for ordering and resource-bound checks.
        const int fd = ::open(fixture.path.c_str(), O_WRONLY | O_TRUNC);
        check(fd >= 0 && write(fd, fixture.bytes.data(), fixture.bytes.size()) == ssize_t(fixture.bytes.size()), "restore");
        close(fd);
        readiness(fixture);
        LinuxArenaFile bounded(std::numeric_limits<size_t>::max());
        check(bounded.chunk() == LinuxArenaFile::max_chunk, "arbitrarily large chunk capped at 8 MiB");
        parity(fixture, nullptr, UINT64_MAX, 1000, {1}, {12345});
        std::cout << "PASS linux_arena_load: aligned/unaligned bytes, FNV parity, guards, EINTR/partials, unsupported I/O, mixed fallback, EOF/EIO, registration fence, bounded scratch\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL linux_arena_load: " << e.what() << '\n';
        return 1;
    }
}
