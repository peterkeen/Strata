#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace strata::core {

// A bounded synchronous worker team. run() is a completion fence: it does not
// return until every worker has finished the submitted range. Concurrent callers
// serialize, so callers cannot create nested worker teams or oversubscribe one
// another. The submitting thread participates in the work.
class FetchWorkers {
public:
    FetchWorkers() = default;
    ~FetchWorkers() { shutdown(); }
    FetchWorkers(const FetchWorkers&) = delete;
    FetchWorkers& operator=(const FetchWorkers&) = delete;

    template <class Function>
    void run(size_t count, size_t worker_count, Function&& function) {
        if (count == 0) return;
        std::unique_lock<std::mutex> run_lock(run_mu_);
        if (count == 1 || worker_count <= 1) {
            function(0);
            return;
        }
        const size_t backgrounds = worker_count - 1;
        resize(backgrounds);
        const size_t active = std::min(count - 1, workers_.size());

        auto invoke = [](void* context, size_t i) {
            (*static_cast<typename std::remove_reference<Function>::type*>(context))(i);
        };
        {
            std::lock_guard<std::mutex> lock(mu_);
            context_ = &function;
            invoke_ = invoke;
            count_ = count;
            next_.store(0, std::memory_order_relaxed);
            active_workers_ = active;
            remaining_ = active;
            ++generation_;
        }
        cv_.notify_all();
        work();
        {
            std::unique_lock<std::mutex> lock(mu_);
            done_cv_.wait(lock, [&] { return remaining_ == 0; });
            context_ = nullptr;
            invoke_ = nullptr;
        }
    }

    void shutdown() {
        std::unique_lock<std::mutex> run_lock(run_mu_);
        resize(0);
    }

private:
    using Invoke = void (*)(void*, size_t);

    void resize(size_t count) {
        if (workers_.size() == count) return;
        if (!workers_.empty()) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                stopping_ = true;
                ++generation_;
            }
            cv_.notify_all();
            for (std::thread& worker : workers_) if (worker.joinable()) worker.join();
            workers_.clear();
            {
                std::lock_guard<std::mutex> lock(mu_);
                stopping_ = false;
                ready_ = 0;
            }
        }
        workers_.reserve(count);
        for (size_t i = 0; i < count; ++i) workers_.emplace_back([this, i] { worker_loop(i); });
        if (count != 0) {
            std::unique_lock<std::mutex> lock(mu_);
            ready_cv_.wait(lock, [&] { return ready_ == count; });
        }
    }

    void worker_loop(size_t id) {
        uint64_t seen;
        {
            std::lock_guard<std::mutex> lock(mu_);
            seen = generation_;
            ++ready_;
            ready_cv_.notify_one();
        }
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                if (stopping_) return;
                seen = generation_;
                if (id >= active_workers_) continue;
            }
            work();
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (--remaining_ == 0) done_cv_.notify_one();
            }
        }
    }

    void work() {
        for (size_t i = next_.fetch_add(1, std::memory_order_relaxed); i < count_;
             i = next_.fetch_add(1, std::memory_order_relaxed)) {
            invoke_(context_, i);
        }
    }

    std::mutex run_mu_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::condition_variable ready_cv_;
    std::vector<std::thread> workers_;
    std::atomic<size_t> next_{0};
    void* context_ = nullptr;
    Invoke invoke_ = nullptr;
    size_t count_ = 0;
    size_t remaining_ = 0;
    size_t active_workers_ = 0;
    size_t ready_ = 0;
    uint64_t generation_ = 0;
    bool stopping_ = false;
};

} // namespace strata::core
