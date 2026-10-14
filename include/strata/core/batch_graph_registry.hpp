#pragma once

#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

namespace strata::core {

// CPU-testable ownership contract for batch graphs that retain slot addresses.
// A generation changes on each successful attach/detach, and graph keys always
// include the generation of the attached slot allocations. Detach is committed
// only after the caller has successfully invalidated all graphs for that gen.
class BatchGraphRegistry {
public:
    bool attached() const { return attached_; }
    uint64_t generation() const { return generation_; }

    bool attach() {
        if (attached_ || generation_ == UINT64_MAX) return false;
        ++generation_;
        attached_ = true;
        return true;
    }

    bool detach(const std::function<bool(uint64_t)>& invalidate, std::string_view& error) {
        if (!attached_) {
            error = "batch slots are not attached";
            return false;
        }
        if (generation_ == UINT64_MAX) {
            error = "batch graph generation exhausted";
            return false;
        }
        if (invalidate && !invalidate(generation_)) {
            error = "batch graph invalidation failed";
            return false;
        }
        attached_ = false;
        ++generation_;
        error = {};
        return true;
    }

    std::vector<int> graph_key(std::vector<int> layout) const {
        for (unsigned shift = 0; shift < 64; shift += 16)
            layout.push_back(static_cast<int>((generation_ >> shift) & 0xffffU));
        return layout;
    }

private:
    uint64_t generation_ = 0;
    bool attached_ = false;
};

} // namespace strata::core
