#include "strata/core/batch_graph_registry.hpp"

#include <cassert>
#include <cstdint>
#include <string_view>
#include <vector>

using strata::core::BatchGraphRegistry;

int main() {
    BatchGraphRegistry registry;
    assert(!registry.attached());
    assert(registry.attach());
    assert(!registry.attach());
    const uint64_t first_generation = registry.generation();
    assert(registry.attached());

    const auto first_key = registry.graph_key({0, 1, 3});
    assert(first_key.size() == 7);
    assert(first_key[0] == 0 && first_key[1] == 1 && first_key[2] == 3);

    int invalidations = 0;
    bool allocation_live = true;
    bool invalidation_succeeds = false;
    std::string_view error;
    assert(!registry.detach([&](uint64_t generation) {
        ++invalidations;
        assert(generation == first_generation);
        if (!invalidation_succeeds) return false;
        allocation_live = false;
        return true;
    }, error));
    assert(invalidations == 1);
    assert(!error.empty());
    assert(registry.attached());
    assert(registry.generation() == first_generation);
    assert(registry.graph_key({0, 1, 3}) == first_key);
    // A failed invalidation keeps the attached allocations owned; free is not reached.
    assert(allocation_live);

    invalidation_succeeds = true;
    assert(registry.detach([&](uint64_t generation) {
        ++invalidations;
        assert(generation == first_generation);
        if (!invalidation_succeeds) return false;
        allocation_live = false;
        return true;
    }, error));
    assert(error.empty());
    assert(!allocation_live);
    assert(!registry.attached());
    assert(registry.generation() == first_generation + 1);
    assert(!registry.detach({}, error));

    assert(registry.attach());
    assert(registry.generation() == first_generation + 2);
    const auto reattached_key = registry.graph_key({0, 1, 3});
    assert(reattached_key != first_key);
    assert(registry.detach({}, error));
    return 0;
}
