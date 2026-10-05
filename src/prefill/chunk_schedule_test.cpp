#include "strata/prefill/chunk_schedule.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

int main() {
    int checks = 0;
    auto check = [&](bool ok) { ++checks; if (!ok) { std::fprintf(stderr, "chunk schedule check %d failed\n", checks); std::exit(1); } };
    using strata::prefill::next_chunk_tokens;
    auto schedule = [&](int64_t n, int64_t cap, int64_t floor) {
        std::vector<int64_t> sizes;
        for (int64_t offset = 0; offset < n;) {
            const int64_t t = next_chunk_tokens(n - offset, cap, floor);
            check(t > 0 && t <= cap && t <= n - offset);
            sizes.push_back(t);
            offset += t;
        }
        return sizes;
    };
    check(schedule(9171, 8192, 1024) == std::vector<int64_t>({8147, 1024}));
    check(schedule(8193, 8192, 1024) == std::vector<int64_t>({7169, 1024}));
    check(schedule(16385, 8192, 1024) == std::vector<int64_t>({8192, 7169, 1024}));
    check(schedule(18171, 8192, 1024) == std::vector<int64_t>({8192, 8192, 1787}));
    check(schedule(9171, 8192, 0) == std::vector<int64_t>({8192, 979}));
    check(schedule(1025, 1024, 1024) == std::vector<int64_t>({1024, 1}));
    for (int64_t cap : {1, 16, 1024, 2048, 4096, 8192}) {
        for (int64_t n = 1; n <= 3 * cap + 2; ++n) {
            const auto s = schedule(n, cap, 1024);
            check(static_cast<int64_t>(s.size()) == 1 + (n - 1) / cap);
            int64_t total = 0;
            for (int64_t t : s) total += t;
            check(total == n);
            if (cap >= 2048 && n > cap) {
                for (int64_t t : s) check(t >= 1024);
            }
        }
    }
    check(next_chunk_tokens(0, 8192, 1024) == 0);
    check(next_chunk_tokens(10, 0, 1024) == 0);
    check(next_chunk_tokens(10, -1, 1024) == 0);
    const auto max = std::numeric_limits<int64_t>::max();
    check(next_chunk_tokens(max, max - 1, 1024) == max - 1024);
    check(next_chunk_tokens(max, max - 1, max - 1) == max - 1);
    check(next_chunk_tokens(max, 8192, 1024) == 8192);
    std::printf("chunk_schedule_test: %d checks passed\n", checks);
}
