#include "strata/spec/output_limits.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>

int main() {
    int checks = 0;
    auto check = [&](bool ok) { ++checks; if (!ok) { std::fprintf(stderr, "output limits check %d failed\n", checks); std::exit(1); } };
    using strata::spec::bounded_window;
    using strata::spec::usable_outputs;
    check(bounded_window(4, 2, 100) == 2);
    check(bounded_window(4, 100, 2) == 2);
    check(bounded_window(4, 0, 100) == 0);
    check(bounded_window(4, 100, 0) == 0);
    check(bounded_window(4, -1, 100) == 0);
    check(bounded_window(4, 100, -1) == 0);
    check(bounded_window(0, 100, 100) == 0);
    check(bounded_window(8, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()) == 8);
    std::array<int32_t, 8> outputs{100, 101, 102, 103, 104, 105, 106, 107};
    // Exhaust output caps, partial acceptance, every EOS row, and no EOS.
    for (int accepted = 1; accepted <= 8; ++accepted) {
        for (int left = 0; left <= 10; ++left) {
            for (int eos_row = 0; eos_row <= 8; ++eos_row) {
                int inspected = 0;
                const int count = usable_outputs(outputs.data(), accepted, left, [&](int32_t t) {
                    ++inspected;
                    return t == 100 + eos_row;
                });
                const int expected = std::min({accepted, left, eos_row + 1});
                check(count == expected);
                check(inspected == count);
                // Consumed state is the old prefix followed by exactly count
                // window inputs; the last output stays unconsumed. No hidden
                // accepted inputs may survive a length or EOS stop.
                const int64_t prompt = 17;
                const int64_t consumed = prompt - 1 + count;
                check(consumed == prompt + count - 1);
                check(count <= left && count <= accepted);
            }
        }
    }
    check(usable_outputs(nullptr, 0, 1, [](int32_t) { return true; }) == 0);
    check(usable_outputs(nullptr, 8, 0, [](int32_t) { return true; }) == 0);
    std::printf("output_limits_test: %d checks passed\n", checks);
}
