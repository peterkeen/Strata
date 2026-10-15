#include "strata/core/exchange_storage.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using strata::core::detail::ExchangeStorage;
constexpr uint64_t absent = ~uint64_t{0};
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }

// UD-Q4_K_XL expert blob sizes. See docs/Q4_NIBBLER_PERFORMANCE_AUDIT.md for the derivation.
constexpr size_t q4k_q5_1_blob = 3072000;
constexpr size_t q4k_q8_0_blob = 3584000;
constexpr size_t q5k_q8_0_blob = 3993600;

std::vector<uint8_t> pattern(size_t bytes, uint32_t seed) {
    std::vector<uint8_t> v(bytes);
    std::mt19937 rng(seed);
    for (auto& b : v) b = (uint8_t) rng();
    return v;
}

void exercise_uniform(size_t bytes, size_t rounds) {
    constexpr size_t n = 24, resident = 16, spare_count = 4, guard = 19;
    std::vector<uint8_t> arena(guard + resident * bytes + guard, 0xcd);
    std::vector<uint8_t> spare(guard + spare_count * bytes + guard, 0xef);
    // Distinct alias arrays catch accidentally retaining the old CUDA address.
    std::vector<uint8_t> aliases(resident * bytes), spare_aliases(spare_count * bytes);
    std::vector<std::vector<uint8_t>> reference(n, std::vector<uint8_t>(bytes));
    std::mt19937 random(601);
    for (auto& row : reference) for (auto& v : row) v = (uint8_t)random();
    std::vector<uint64_t> offsets(n, absent);
    for (size_t i = 0; i < resident; ++i) {
        offsets[i] = i * bytes;
        std::memcpy(arena.data() + guard + offsets[i], reference[i].data(), bytes);
    }
    ExchangeStorage storage;
    std::string error;
    require(storage.initialize(offsets, arena.data() + guard, aliases.data(), resident * bytes,
        spare.data() + guard, spare_aliases.data(), spare_count, bytes, error), error.c_str());
    require(storage.spare_bytes() == spare_count * bytes, "uniform spare bytes misreported");
    std::set<uint8_t*> original;
    for (size_t i = 0; i < n; ++i) if (auto p = storage.resident(i).host) original.insert(p);
    for (size_t q = 0; q < spare_count; ++q) original.insert(storage.spare(q).host);
    // A lookahead thread may inspect residency while ownership is committed.
    // It only observes immutable address pairs, never reads mutable expert bytes.
    std::atomic<bool> stop{false}, reader_failed{false};
    std::atomic<uint64_t> observations{0};
    std::vector<ExchangeStorage::View> valid;
    for (size_t i = 0; i < n; ++i) if (storage.resident(i).host) valid.push_back(storage.resident(i));
    for (size_t q = 0; q < spare_count; ++q) valid.push_back(storage.spare(q));
    std::thread reader([&] {
        while (!stop.load()) {
            for (size_t i = 0; i < n; ++i) {
                const auto view = storage.resident(i);
                if (view.host && std::none_of(valid.begin(), valid.end(), [&](auto v) {
                        return v.host == view.host && v.device == view.device; })) reader_failed.store(true);
            }
            ++observations;
        }
    });
    struct Join {
        std::atomic<bool>& stop; std::thread& reader;
        void finish() { stop.store(true); if (reader.joinable()) reader.join(); }
        ~Join() { finish(); }
    } join{stop, reader};
    while (!observations.load()) std::this_thread::yield();
    uint64_t exchanges = 0;
    for (size_t round = 0; round < rounds; ++round) {
        std::vector<size_t> ins, outs;
        for (size_t i = 0; i < n; ++i) (storage.resident(i).host ? ins : outs).push_back(i);
        std::shuffle(ins.begin(), ins.end(), random);
        std::shuffle(outs.begin(), outs.end(), random);
        ExchangeStorage::View before[spare_count], staged[spare_count];
        for (size_t q = 0; q < spare_count; ++q) {
            before[q] = storage.resident(ins[q]); staged[q] = storage.spare(q);
            // Simulated completed D2H eviction. The old RAM input stays readable
            // until simulated H2D and all readers finish, before commit.
            std::memcpy(staged[q].host, reference[outs[q]].data(), bytes);
            require(!std::memcmp(before[q].host, reference[ins[q]].data(), bytes), "incoming expert overwritten early");
        }
        for (size_t q = 0; q < spare_count; ++q) {
            require(!storage.commit(ins[q], ins[q], q, staged[q].host, bytes), "self swap accepted");
            require(!storage.commit(ins[q], outs[q], q, staged[q].host, bytes + 1), "wrong size accepted");
            require(!storage.commit(ins[q], outs[q], q, nullptr, bytes), "missing staged buffer accepted");
            require(!storage.commit(ins[q], outs[q], spare_count, staged[q].host, bytes), "bad spare accepted");
            require(!storage.commit(n, outs[q], q, staged[q].host, bytes), "bad expert accepted");
            require(storage.commit(ins[q], outs[q], q, staged[q].host, bytes), "valid swap rejected");
            require(!storage.commit(ins[q], outs[q], q, staged[q].host, bytes), "duplicate commit accepted");
            require(storage.resident(outs[q]).host == staged[q].host &&
                    storage.resident(outs[q]).device == staged[q].device, "evicted buffer was not adopted with its alias");
            require(storage.spare(q).host == before[q].host && storage.spare(q).device == before[q].device,
                    "old RAM input was not recycled with its alias");
            // Even the old incoming bytes have not changed: commit copies metadata only.
            require(!std::memcmp(storage.spare(q).host, reference[ins[q]].data(), bytes), "commit changed spare bytes");
            ++exchanges;
        }
        std::set<uint8_t*> live;
        for (size_t i = 0; i < n; ++i) if (auto p = storage.resident(i).host) {
            require(live.insert(p).second, "two experts share storage");
            require(!std::memcmp(p, reference[i].data(), bytes), "resident bytes corrupted");
        }
        for (size_t q = 0; q < spare_count; ++q)
            require(live.insert(storage.spare(q).host).second, "spare aliases live expert");
        require(live == original, "storage leaked or duplicated");
    }
    require(storage.exchanges() == exchanges && storage.avoided_bytes() == exchanges * bytes, "wrong accounting");
    require(std::all_of(arena.begin(), arena.begin() + guard, [](auto x) { return x == 0xcd; }) &&
            std::all_of(arena.end() - guard, arena.end(), [](auto x) { return x == 0xcd; }) &&
            std::all_of(spare.begin(), spare.begin() + guard, [](auto x) { return x == 0xef; }) &&
            std::all_of(spare.end() - guard, spare.end(), [](auto x) { return x == 0xef; }), "guard overwritten");
    require(!storage.initialize(offsets, arena.data() + guard, aliases.data(), resident * bytes,
        spare.data() + guard, spare_aliases.data(), spare_count, bytes, error), "live storage reinitialized");
    join.finish();
    require(!reader_failed.load() && observations.load(), "concurrent residency reader observed invalid alias");
    storage.clear();
    require(!storage.active() && !storage.resident(0).host && !storage.spare(0).host &&
            !storage.exchanges() && !storage.avoided_bytes(), "clear retained state");
}

void invalid_uniform_geometry() {
    uint8_t h[64]{}, d[64]{}, x[64]{}, xd[64]{};
    ExchangeStorage s; std::string e;
    require(!s.initialize({0, 3}, h, d, 8, x, xd, 2, 4, e), "unaligned slot accepted");
    require(!s.initialize({0, 0}, h, d, 8, x, xd, 2, 4, e), "duplicate offset accepted");
    require(!s.initialize({0, 8}, h, d, 8, x, xd, 2, 4, e), "out of bounds accepted");
    require(!s.initialize({0, absent}, h, d, 8, x, xd, 2, 4, e), "lost slot accepted");
    require(!s.initialize({0, 4}, h, nullptr, 8, x, xd, 2, 4, e), "unmapped resident accepted");
    require(!s.initialize({0, 4}, h, d, 8, x, nullptr, 2, 4, e), "unmapped spare accepted");
    require(!s.initialize({0, 4}, h, d, 8, x, xd, 2, 0, e), "zero blob accepted");
    require(!s.active(), "failed initialization left active state");
}

void exercise_size_classes() {
    constexpr size_t n_layers = 3, n_expert = 4, resident_per_layer = 2, spare_per_class = 1, align = 16, guard = 64;
    const std::array<size_t, n_layers> layer_bytes = {q4k_q5_1_blob, q4k_q8_0_blob, q5k_q8_0_blob};
    std::vector<size_t> expert_bytes(n_layers * n_expert);
    for (size_t l = 0; l < n_layers; ++l)
        for (size_t e = 0; e < n_expert; ++e) expert_bytes[l * n_expert + e] = layer_bytes[l];

    std::vector<uint64_t> offsets(expert_bytes.size(), absent);
    uint64_t resident_bytes = 0;
    for (size_t l = 0; l < n_layers; ++l) {
        for (size_t e = 0; e < resident_per_layer; ++e) {
            offsets[l * n_expert + e] = resident_bytes;
            resident_bytes += layer_bytes[l];
        }
    }
    const size_t max_blob = *std::max_element(layer_bytes.begin(), layer_bytes.end());
    require(resident_bytes < n_layers * resident_per_layer * max_blob, "test accidentally padded compact complement");

    std::vector<uint8_t> arena(guard + (size_t) resident_bytes + guard, 0xcd);
    std::vector<uint8_t> aliases(guard + (size_t) resident_bytes + guard, 0xac);
    std::array<std::vector<uint8_t>, n_layers> spares = {
        std::vector<uint8_t>(guard + q4k_q5_1_blob + guard, 0xe1),
        std::vector<uint8_t>(guard + q4k_q8_0_blob + guard, 0xe2),
        std::vector<uint8_t>(guard + q5k_q8_0_blob + guard, 0xe3)};
    std::array<std::vector<uint8_t>, n_layers> spare_aliases = {
        std::vector<uint8_t>(guard + q4k_q5_1_blob + guard, 0xa1),
        std::vector<uint8_t>(guard + q4k_q8_0_blob + guard, 0xa2),
        std::vector<uint8_t>(guard + q5k_q8_0_blob + guard, 0xa3)};
    std::vector<std::vector<uint8_t>> reference;
    reference.reserve(expert_bytes.size());
    for (size_t i = 0; i < expert_bytes.size(); ++i) reference.push_back(pattern(expert_bytes[i], (uint32_t) (7000 + i)));
    for (size_t i = 0; i < offsets.size(); ++i)
        if (offsets[i] != absent) std::memcpy(arena.data() + guard + offsets[i], reference[i].data(), expert_bytes[i]);

    std::vector<ExchangeStorage::SpareClass> classes;
    for (size_t l = 0; l < n_layers; ++l)
        classes.push_back({layer_bytes[l], spares[l].data() + guard, spare_aliases[l].data() + guard, spare_per_class});

    ExchangeStorage storage;
    std::string error;
    require(storage.initialize_size_classes(offsets, expert_bytes, arena.data() + guard, aliases.data() + guard,
        resident_bytes, classes, align, error), error.c_str());
    require(storage.spare_capacity(q4k_q5_1_blob) == 1 && storage.spare_capacity(q4k_q8_0_blob) == 1 &&
            storage.spare_capacity(q5k_q8_0_blob) == 1 && storage.spare_capacity(1234) == 0, "wrong size-class capacities");
    const uint64_t spare_bytes = q4k_q5_1_blob + q4k_q8_0_blob + q5k_q8_0_blob;
    require(storage.spare_bytes() == spare_bytes && storage.spare_bytes() < 3 * max_blob,
            "spares were padded to max blob size");

    uint64_t exchanges = 0;
    for (size_t round = 0; round < 4; ++round) {
        for (size_t l = 0; l < n_layers; ++l) {
            const size_t bytes = layer_bytes[l];
            const size_t in = l * n_expert + (round % 2 == 0 ? 0 : 2);
            const size_t out = l * n_expert + (round % 2 == 0 ? 2 : 0);
            const auto before_in = storage.resident(in);
            const auto staged = storage.spare_for_bytes(bytes, 0);
            require(before_in.host && staged.host, "missing classed in/spare view");
            std::memcpy(staged.host, reference[out].data(), bytes);

            // Rollback: wrong size, wrong class, exhausted class queue and wrong staged pointer leave aliases unchanged.
            require(!storage.commit(in, out, 0, staged.host, bytes + align), "classed wrong size accepted");
            require(!storage.commit(in, out, 1, staged.host, bytes), "classed capacity exhaustion accepted");
            require(!storage.commit(in, out, 0, before_in.host, bytes), "classed non-spare staged pointer accepted");
            if (l + 1 < n_layers) {
                const size_t other_out = (l + 1) * n_expert + 2;
                require(!storage.commit(in, other_out, 0, staged.host, bytes), "cross-size exchange accepted");
            }
            require(storage.resident(in).host == before_in.host && storage.resident(out).host == nullptr &&
                    !storage.can_commit(in, out, 1, staged.host, bytes),
                    "failed classed commit changed residency");

            require(storage.commit(in, out, 0, staged.host, bytes), "classed valid swap rejected");
            ++exchanges;
            require(storage.resident(out).host == staged.host && storage.resident(out).device == staged.device,
                    "classed out did not adopt staged alias");
            require(storage.spare_for_bytes(bytes, 0).host == before_in.host &&
                    storage.spare_for_bytes(bytes, 0).device == before_in.device,
                    "classed in alias was not recycled as same-size spare");
            require(!std::memcmp(storage.resident(out).host, reference[out].data(), bytes), "classed out bytes corrupt");
            require(!std::memcmp(storage.spare_for_bytes(bytes, 0).host, reference[in].data(), bytes),
                    "classed commit copied or corrupted old in bytes");
        }
    }
    require(storage.exchanges() == exchanges && storage.avoided_bytes() ==
            4 * (q4k_q5_1_blob + q4k_q8_0_blob + q5k_q8_0_blob), "classed accounting wrong");
    // Compare every resolved resident against a copy-based oracle after random
    // exchanges. Include a rejected rotation followed by the engine's copy fallback.
    std::mt19937 rng(4019);
    for (size_t round = 0; round < 90; ++round) {
        const size_t l = rng() % n_layers, bytes = layer_bytes[l];
        std::vector<size_t> ins, outs;
        for (size_t e = 0; e < n_expert; ++e)
            (storage.resident(l * n_expert + e).host ? ins : outs).push_back(l * n_expert + e);
        const size_t in = ins[rng() % ins.size()], out = outs[rng() % outs.size()];
        const auto spare = storage.spare_for_bytes(bytes, 0);
        std::memcpy(spare.host, reference[out].data(), bytes);
        if (round % 7 == 0 || round % 11 == 0) {
            const size_t q = round % 11 == 0 ? 1 : 0;
            const uint8_t* staged = round % 11 == 0 ? spare.host : storage.resident(in).host;
            require(!storage.can_commit(in, out, q, staged, bytes),
                    "wrong staged pointer or exhausted class accepted");
            std::memcpy(storage.resident(in).host, spare.host, bytes);
            require(storage.commit_copy(in, out, bytes), "copy fallback rejected");
        } else {
            require(storage.can_commit(in, out, 0, spare.host, bytes) &&
                    storage.commit(in, out, 0, spare.host, bytes), "random rotation rejected");
            ++exchanges;
        }
        for (size_t i = 0; i < offsets.size(); ++i)
            if (auto p = storage.resident(i).host)
                require(!std::memcmp(p, reference[i].data(), expert_bytes[i]), "random rotation differs from copy oracle");
    }
    require(storage.exchanges() == exchanges, "copy fallback counted as rotation");
    for (size_t i = 0; i < offsets.size(); ++i)
        if (storage.resident(i).host) require(!std::memcmp(storage.resident(i).host, reference[i].data(), expert_bytes[i]),
                                      "classed resident bytes corrupt after repeated swaps");

    require(!storage.initialize_size_classes(offsets, expert_bytes, arena.data() + guard, aliases.data() + guard,
        resident_bytes, classes, align, error), "live classed storage reinitialized");
    storage.clear();
}

void invalid_size_class_geometry() {
    constexpr size_t align = 16;
    std::vector<uint64_t> offsets = {0, q4k_q5_1_blob, absent};
    std::vector<size_t> bytes = {q4k_q5_1_blob, q4k_q8_0_blob, q5k_q8_0_blob};
    const uint64_t resident_bytes = q4k_q5_1_blob + q4k_q8_0_blob;
    std::vector<uint8_t> arena((size_t) resident_bytes + align, 0), aliases((size_t) resident_bytes + align, 0);
    std::vector<uint8_t> a(q4k_q5_1_blob + align, 0), b(q4k_q8_0_blob + align, 0), c(q5k_q8_0_blob + align, 0);
    std::vector<ExchangeStorage::SpareClass> missing = {{q4k_q5_1_blob, a.data(), a.data(), 1},
                                                        {q4k_q8_0_blob, b.data(), b.data(), 1}};
    ExchangeStorage s; std::string e;
    require(!s.initialize_size_classes(offsets, bytes, arena.data(), aliases.data(), resident_bytes, missing, align, e),
            "missing spare class accepted");
    auto classes = missing;
    classes.push_back({q5k_q8_0_blob, c.data(), c.data(), 1});
    require(!s.initialize_size_classes(offsets, bytes, arena.data() + 1, aliases.data(), resident_bytes, classes, align, e),
            "misaligned resident host accepted");
    require(!s.initialize_size_classes(offsets, bytes, arena.data(), aliases.data(), resident_bytes, classes, 24, e),
            "non-power-of-two alignment accepted");
    classes[2].bytes = q4k_q8_0_blob;
    require(!s.initialize_size_classes(offsets, bytes, arena.data(), aliases.data(), resident_bytes, classes, align, e),
            "duplicate class accepted");
    classes[2] = {q5k_q8_0_blob, c.data() + 1, c.data(), 1};
    require(!s.initialize_size_classes(offsets, bytes, arena.data(), aliases.data(), resident_bytes, classes, align, e),
            "misaligned spare accepted");
    classes[2] = {q5k_q8_0_blob, c.data(), c.data(), 0};
    require(!s.initialize_size_classes(offsets, bytes, arena.data(), aliases.data(), resident_bytes, classes, align, e),
            "zero spare count accepted");
    bytes[1] = q4k_q8_0_blob + 8;
    classes[2] = {q5k_q8_0_blob, c.data(), c.data(), 1};
    require(!s.initialize_size_classes(offsets, bytes, arena.data(), aliases.data(), resident_bytes, classes, align, e),
            "unaligned expert bytes accepted");
    bytes[1] = q4k_q8_0_blob;
    offsets[1] = q4k_q5_1_blob - align;
    require(!s.initialize_size_classes(offsets, bytes, arena.data(), aliases.data(), resident_bytes, classes, align, e),
            "overlapping resident ranges accepted");
    require(!s.active(), "failed classed initialization left active state");
}

int main() {
    try {
        invalid_uniform_geometry();
        invalid_size_class_geometry();
        for (size_t bytes : {1u, 33u, 4097u}) exercise_uniform(bytes, 1024);
        exercise_uniform(5222400, 4); // existing uniform-size large path coverage
        exercise_size_classes();
        std::cout << "exchange_storage_test: PASS (uniform + Q4 size-class alias/ownership checks)\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
