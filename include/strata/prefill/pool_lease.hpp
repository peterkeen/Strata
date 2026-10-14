// include/strata/prefill/pool_lease.hpp - who may run work on the CPU expert pool right now (str-s1w.3).
//
// The decode CPU pool is idle while a prompt chunk is read, so the prompt path can hand it the experts few of the
// chunk's tokens route to (STRATA_PREFILL_CPU_SHARE).  With batch slots that was refused outright (share_pool =
// o.batch <= 0), because a pool driven by two owners at once would mix their jobs.  The lease makes the ownership
// explicit instead: one owner at a time, tagged, so a prompt chunk takes the pool only when no decode window has it
// and a decode window can prove no prompt chunk does.
//
// On the serve loop the two never overlap in time - a prompt is read chunk by chunk on the main thread, and batch
// windows run between chunks (STRATA_BATCH_DECODE_SHARE), not beside them - so the prompt's try_acquire always
// succeeds and the decode window's acquire never waits.  The lease is the check that this stays true: a prompt chunk
// that finds the pool taken falls back to the GPU alone (the existing behaviour), and a decode window that cannot get
// it within a bound fails loudly instead of deadlocking or racing.
//
// Header-only and GPU-free: tests/prefill/prefill_pool_lease_test.cpp exercises it with threads.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace strata::prefill {

enum class PoolOwner : int { none = 0, prompt = 1, decode = 2 };

inline const char* pool_owner_name(PoolOwner o) {
    switch (o) {
        case PoolOwner::prompt: return "prompt";
        case PoolOwner::decode: return "decode";
        default: return "none";
    }
}

class PoolLease {
public:
    /// Take the pool for `who` if nobody has it.  Never blocks.
    bool try_acquire(PoolOwner who) noexcept {
        int expect = int(PoolOwner::none);
        const bool got = owner_.compare_exchange_strong(expect, int(who), std::memory_order_acq_rel);
        if (who == PoolOwner::prompt) (got ? prompt_grants_ : prompt_busy_).fetch_add(1, std::memory_order_relaxed);
        return got;
    }
    /// Spin (then yield) until the pool is free or `timeout` passes.  Bounded: a caller that already holds the pool on
    /// this thread under another tag times out instead of deadlocking.
    bool acquire_for(PoolOwner who, std::chrono::microseconds timeout) noexcept {
        if (try_acquire_quiet(who)) return true;
        waits_.fetch_add(1, std::memory_order_relaxed);
        const auto end = std::chrono::steady_clock::now() + timeout;
        for (int i = 0;; ++i) {
            if (try_acquire_quiet(who)) return true;
            if (std::chrono::steady_clock::now() >= end) {
                timeouts_.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            if (i > 64) std::this_thread::yield();
        }
    }
    /// Give the pool back.  Only the holder's own tag releases it; a mismatched release is counted and ignored (it
    /// would otherwise hand the pool to a third party while its holder still runs).
    bool release(PoolOwner who) noexcept {
        int expect = int(who);
        const bool ok = owner_.compare_exchange_strong(expect, int(PoolOwner::none), std::memory_order_acq_rel);
        if (!ok) bad_releases_.fetch_add(1, std::memory_order_relaxed);
        return ok;
    }
    PoolOwner owner() const noexcept { return PoolOwner(owner_.load(std::memory_order_acquire)); }

    uint64_t prompt_grants() const noexcept { return prompt_grants_.load(std::memory_order_relaxed); }
    uint64_t prompt_busy() const noexcept { return prompt_busy_.load(std::memory_order_relaxed); }
    uint64_t waits() const noexcept { return waits_.load(std::memory_order_relaxed); }
    uint64_t timeouts() const noexcept { return timeouts_.load(std::memory_order_relaxed); }
    uint64_t bad_releases() const noexcept { return bad_releases_.load(std::memory_order_relaxed); }

    /// RAII: released on every exit (an error return, a cancelled chunk, an exception).
    class Hold {
    public:
        Hold() = default;
        Hold(PoolLease& lease, PoolOwner who) : lease_(&lease), who_(who), held_(lease.try_acquire(who)) {}
        Hold(PoolLease& lease, PoolOwner who, std::chrono::microseconds timeout)
            : lease_(&lease), who_(who), held_(lease.acquire_for(who, timeout)) {}
        Hold(const Hold&) = delete;
        Hold& operator=(const Hold&) = delete;
        Hold(Hold&& o) noexcept : lease_(o.lease_), who_(o.who_), held_(o.held_) { o.held_ = false; }
        ~Hold() { reset(); }
        bool held() const noexcept { return held_; }
        void reset() noexcept {
            if (held_ && lease_ != nullptr) lease_->release(who_);
            held_ = false;
        }

    private:
        PoolLease* lease_ = nullptr;
        PoolOwner who_ = PoolOwner::none;
        bool held_ = false;
    };

private:
    bool try_acquire_quiet(PoolOwner who) noexcept {
        int expect = int(PoolOwner::none);
        return owner_.compare_exchange_strong(expect, int(who), std::memory_order_acq_rel);
    }
    std::atomic<int> owner_{int(PoolOwner::none)};
    std::atomic<uint64_t> prompt_grants_{0}, prompt_busy_{0}, waits_{0}, timeouts_{0}, bad_releases_{0};
};

// ---- when the prompt path may be given the pool while batch slots exist (STRATA_PREFILL_SLOT_SHARE=1) ----

struct SlotShareInputs {
    bool opt_in = false;          ///< STRATA_PREFILL_SLOT_SHARE=1
    int batch = 0;                ///< --batch slots (0: solo engine - the existing share rule applies, not this one)
    int batch_groups = 1;         ///< --batch-groups: >1 decodes groups on other threads (pipelined windows)
    bool batch_groups_auto = false;
    bool layer_split = false;     ///< stages present
    bool multi_gpu = false;
    bool no_pool = false;         ///< --no-pool
    bool hip = false;             ///< HIP builds keep the share off by default (not measured there)
};

struct SlotShareVerdict {
    bool pool_to_prompt = false;  ///< set_cpu_pool on the prompt path, with the lease guarding both owners
    const char* why = "";         ///< one line for the start-up log
};

constexpr SlotShareVerdict slot_share_verdict(const SlotShareInputs& in) {
    if (in.batch <= 0) return {false, "no batch slots: the solo share rule applies"};
    if (!in.opt_in) return {false, "off (STRATA_PREFILL_SLOT_SHARE=1 turns it on)"};
    if (in.no_pool) return {false, "--no-pool: there is no CPU pool"};
    if (in.batch_groups != 1 || in.batch_groups_auto)
        return {false, "batch groups decode on their own threads: the pool is never quiescent between chunks"};
    if (in.layer_split || in.multi_gpu) return {false, "layer split / several GPUs: not measured"};
    if (in.hip) return {false, "HIP build: not measured"};
    return {true, "on: prompt chunks take the CPU pool between batch windows (owner-tagged lease)"};
}

}  // namespace strata::prefill
