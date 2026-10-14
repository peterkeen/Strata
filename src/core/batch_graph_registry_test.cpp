#include "strata/core/batch_graph_registry.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string_view>
#include <vector>

using strata::core::BatchGraphRegistry;

// Not <cassert>: the default Release build defines NDEBUG and would compile every check away.
static int checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

int main() {
    BatchGraphRegistry registry;
    CHECK(!registry.attached());
    CHECK(registry.attach());
    CHECK(!registry.attach());
    const uint64_t first_generation = registry.generation();
    CHECK(registry.attached());

    const auto first_key = registry.graph_key({0, 1, 3});
    CHECK(first_key.size() == 7);
    CHECK(first_key[0] == 0 && first_key[1] == 1 && first_key[2] == 3);

    int invalidations = 0;
    bool allocation_live = true;
    bool invalidation_succeeds = false;
    std::string_view error;
    CHECK(!registry.detach([&](uint64_t generation) {
        ++invalidations;
        CHECK(generation == first_generation);
        if (!invalidation_succeeds) return false;
        allocation_live = false;
        return true;
    }, error));
    CHECK(invalidations == 1);
    CHECK(!error.empty());
    CHECK(registry.attached());
    CHECK(registry.generation() == first_generation);
    CHECK(registry.graph_key({0, 1, 3}) == first_key);
    // A failed invalidation keeps the attached allocations owned; free is not reached.
    CHECK(allocation_live);

    invalidation_succeeds = true;
    CHECK(registry.detach([&](uint64_t generation) {
        ++invalidations;
        CHECK(generation == first_generation);
        if (!invalidation_succeeds) return false;
        allocation_live = false;
        return true;
    }, error));
    CHECK(error.empty());
    CHECK(!allocation_live);
    CHECK(!registry.attached());
    CHECK(registry.generation() == first_generation + 1);
    CHECK(!registry.detach({}, error));

    CHECK(registry.attach());
    CHECK(registry.generation() == first_generation + 2);
    const auto reattached_key = registry.graph_key({0, 1, 3});
    CHECK(reattached_key != first_key);
    CHECK(registry.detach({}, error));
    std::printf("batch_graph_registry_test: %d checks passed\n", checks);
    return 0;
}
