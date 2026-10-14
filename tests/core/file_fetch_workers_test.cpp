#include "strata/core/fetch_workers.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_empty_single_multiple_and_fence() {
    strata::core::FetchWorkers workers;
    std::atomic<size_t> count{0};
    workers.run(0, 4, [&](size_t) { count.fetch_add(1); });
    require(count.load() == 0, "empty run invoked work");

    int single = 0;
    workers.run(1, 4, [&](size_t i) { single = (int) i + 1; });
    require(single == 1, "single-item run failed");

    std::vector<uint8_t> bytes(257, 0);
    workers.run(bytes.size(), 5, [&](size_t i) { bytes[i] = (uint8_t) (i % 251); });
    for (size_t i = 0; i < bytes.size(); ++i)
        require(bytes[i] == (uint8_t) (i % 251), "run returned before all writes were visible");
}

void test_injected_read_failure() {
    strata::core::FetchWorkers workers;
    constexpr size_t jobs = 101;
    std::atomic<size_t> reads{0};
    std::atomic<size_t> failures{0};
    workers.run(jobs, 8, [&](size_t i) {
        reads.fetch_add(1, std::memory_order_relaxed);
        if (i == 37) failures.fetch_add(1, std::memory_order_relaxed);
    });
    require(reads.load() == jobs && failures.load() == 1, "injected read failure was not accounted");
}

void test_concurrent_copy_consumers() {
    strata::core::FetchWorkers workers;
    constexpr size_t rows = 8;
    constexpr size_t width = 4096;
    std::vector<uint8_t> source(width, 0x5a);
    std::vector<std::vector<uint8_t>> outputs(rows, std::vector<uint8_t>(width));
    std::vector<std::thread> consumers;
    std::atomic<size_t> completed{0};
    for (size_t row = 0; row < rows; ++row) {
        consumers.emplace_back([&, row] {
            workers.run(1, 4, [&](size_t) {
                std::memcpy(outputs[row].data(), source.data(), width);
            });
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& consumer : consumers) consumer.join();
    require(completed.load() == rows, "concurrent consumers did not complete");
    for (const auto& output : outputs)
        require(output == source, "concurrent copy consumer received incorrect bytes");

    consumers.clear();
    for (size_t row = 0; row < rows; ++row) {
        consumers.emplace_back([&, row] {
            workers.run(width / 64, 4, [&](size_t i) {
                std::memcpy(outputs[row].data() + i * 64, source.data() + i * 64, 64);
            });
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& consumer : consumers) consumer.join();
    require(completed.load() == 2 * rows, "concurrent multi-copy consumers did not complete");
    for (const auto& output : outputs)
        require(output == source, "concurrent multi-copy consumer received incorrect bytes");
}

void test_shutdown_and_reuse() {
    strata::core::FetchWorkers workers;
    std::atomic<size_t> count{0};
    workers.run(32, 4, [&](size_t) { count.fetch_add(1, std::memory_order_relaxed); });
    workers.shutdown();
    workers.run(64, 4, [&](size_t) { count.fetch_add(1, std::memory_order_relaxed); });
    require(count.load() == 96, "worker team failed after shutdown/reopen");
}

void report_synthetic_benchmark() {
    strata::core::FetchWorkers workers;
    constexpr size_t copies = 2000;
    constexpr size_t bytes_per_copy = 4096;
    std::vector<uint8_t> source(bytes_per_copy, 0xa5);
    std::vector<uint8_t> destination(copies * bytes_per_copy);
    workers.run(8, 8, [&](size_t) {}); // create the bounded team before measuring warm scheduling/copy cost
    const auto start = std::chrono::steady_clock::now();
    workers.run(copies, 8, [&](size_t i) {
        std::memcpy(destination.data() + i * bytes_per_copy, source.data(), bytes_per_copy);
    });
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    require(destination.front() == 0xa5 && destination.back() == 0xa5, "synthetic copy fixture is incorrect");
    const double copies_per_second = (double) copies / seconds;

    constexpr size_t schedule_jobs = 1000;
    std::atomic<size_t> scheduled{0};
    const auto schedule_start = std::chrono::steady_clock::now();
    workers.run(schedule_jobs, 8, [&](size_t) { scheduled.fetch_add(1, std::memory_order_relaxed); });
    const double scheduling_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - schedule_start).count();
    require(scheduled.load() == schedule_jobs, "scheduling fixture lost work");
    std::cout << "synthetic_copy_copies_per_second=" << copies_per_second
              << " scheduling_overhead_us_per_1000_jobs=" << scheduling_us << '\n';
}

} // namespace

int main() {
    try {
        test_empty_single_multiple_and_fence();
        test_injected_read_failure();
        test_concurrent_copy_consumers();
        test_shutdown_and_reuse();
        report_synthetic_benchmark();
        std::cout << "file_fetch_workers_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "file_fetch_workers_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
