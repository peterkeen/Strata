#define STRATA_SHARED_KV_REGISTRY_ONLY
#include "strata/core/shared_kv_runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
int checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}
struct Session {
    int sequence = -1;
    bool initialized = false;
};
}

int main() {
    using strata::core::SharedKvLogicalSlots;
    SharedKvLogicalSlots<Session> slots(3);
    Session main_session{0, true};
    Session slot_a{1, false}, slot_b{2, false};
    std::string error;

    check(slots.size() == 3 && slots.get(0) == nullptr && slots.get(2) == nullptr,
          "logical IDs exist before any physical session");
    check(slots.attach(1, &slot_a, [](Session& s, std::string& e) {
        s.initialized = true;
        e.clear();
        return true;
    }, error), "attach initialized session");
    check(slots.get(1) == &slot_a && slots.generation(1) == 1,
          "successful attach publishes binding and advances generation");
    check(!slots.attach(2, &slot_b, [](Session&, std::string& e) {
        e = "injected initialization failure";
        return false;
    }, error), "failed initialization is rejected");
    check(slots.get(2) == nullptr && slots.generation(2) == 0,
          "failed attach does not publish partial binding");
    check(!slots.detach(1, [](Session&, std::string& e) {
        e = "injected detach failure";
        return false;
    }, error), "failed detach is rejected");
    check(slots.get(1) == &slot_a && slots.generation(1) == 1,
          "failed detach preserves published binding");
    check(slots.detach(1, [&main_session](Session& s, std::string& e) {
        s.initialized = false;
        // The owner is deliberately not passed to or modified by detach.
        e.clear();
        return main_session.initialized;
    }, error), "detach binding");
    check(slots.get(1) == nullptr && slots.generation(1) == 2 && main_session.initialized,
          "detach removes only slot binding and preserves owner state");
    slot_b.initialized = true;
    check(slots.attach(1, &slot_b, [](Session& s, std::string& e) {
        if (!s.initialized) { e = "session is not initialized"; return false; }
        e.clear();
        return true;
    }, error), "reattach to the same stable logical ID");
    check(slots.get(1) == &slot_b && slots.generation(1) == 3,
          "reattach advances generation without changing logical ID");
    check(!slots.attach(1, &slot_a, [](Session&, std::string&) { return true; }, error),
          "occupied logical ID rejects duplicate attach");
    check(slots.get(1) == &slot_b && slots.size() == 3,
          "duplicate attach leaves published mapping unchanged");

    std::printf("shared_kv_runtime_lazy: %d checks passed\n", checks);
    return 0;
}
