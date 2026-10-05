// Host-owned logical mappings for one full-resident or host-backed shared KV pool.
#pragma once

#include "strata/core/session.hpp"
#include "strata/core/shared_kv_pages.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <array>
#include <memory>
#include <stdexcept>

namespace strata::core {

// Mutations run on the serving thread, outside graphs, with an explicit device
// fence before touching authoritative bytes or residency. Session zero never
// resets these shared pools. The first session owns their allocations.
//
// Resident mode: logical -> physical GPU page.
// Streamed mode: logical -> authoritative host backing page; each layer also
// owns one shared backing -> GPU-slot cache. Its per-sequence GPU reader table
// is materialized by resolve inside each graph, not published by this class.
class SharedKvRuntime {
public:
    SharedKvRuntime(const ModelGeometry& g, std::vector<SessionState*> sessions)
        : g_(g), sessions_(std::move(sessions)),
          pages_(capacity(sessions_), sessions_.size()),
          streamed_(sessions_[0]->qsa_states[sessions_[0]->qsa_primary()].kv_mode == 1) {
        for (auto* s : sessions_) {
            if (!s || s->qsa_alloc != sessions_[0]->qsa_alloc || s->max_cells != sessions_[0]->max_cells)
                throw std::invalid_argument("unified KV: incompatible session");
            for (int64_t j = 0; j < s->qsa_alloc; ++j) {
                auto& q = s->qsa_states[s->qsa_ord0 + j];
                const auto& owner = sessions_[0]->qsa_states[sessions_[0]->qsa_ord0 + j];
                if (q.kv_mode != (streamed_ ? 1 : 0) ||
                    (streamed_ ? q.n_pages : q.n_slots) != int64_t(pages_.capacity()) ||
                    q.n_slots != owner.n_slots || q.kv_int8 != owner.kv_int8 ||
                    q.kv_q4 != owner.kv_q4 || q.kv_hybrid != owner.kv_hybrid)
                    throw std::invalid_argument("unified KV: incompatible physical pool");
                if (streamed_ && (!q.shared_kv || !q.host.present() || !q.host.logical_pages ||
                    q.host.resident_pages != owner.map.page_table || q.map.page_table != owner.map.page_table ||
                    q.page_table == q.map.page_table || q.map.n_blocks != int64_t(pages_.capacity())))
                    throw std::invalid_argument("unified KV: missing shared streaming bindings");
                q.shared_kv = true;
                q.shared_page_table.assign(size_t(q.n_pages), -1);
            }
        }
        std::string error;
        if (streamed_) {
            int32_t* p = nullptr;
            if (!checked(cudaMalloc(reinterpret_cast<void**>(&p), pages_.capacity() * sizeof(int32_t)), error))
                throw std::runtime_error(error);
            invalidation_ids_.reset(p);
        }
        for (size_t i = 0; i < sessions_.size(); ++i)
            if (!publish(i, error)) throw std::runtime_error(error);
    }

    const SharedKvPages& pages() const { return pages_; }
    bool streamed() const { return streamed_; }
    int64_t capacity_cells() const { return int64_t(pages_.capacity()) * kernels::qsa_real_shapes().page_size; }
    int64_t resident_cells() const {
        return sessions_[0]->qsa_states[sessions_[0]->qsa_primary()].n_slots * kernels::qsa_real_shapes().page_size;
    }
    size_t page_count(int64_t cells) const {
        const int64_t p = kernels::qsa_real_shapes().page_size;
        return size_t(cells / p + (cells % p != 0));
    }

    bool ensure(size_t seq, int64_t begin, int64_t end, std::string& error) {
        if (seq >= sessions_.size() || begin < 0 || end < begin || end > sessions_[seq]->max_cells) {
            error = "unified KV: invalid write extent";
            return false;
        }
        const size_t old_size = pages_.mapping(seq).size();
        std::vector<SharedKvPages::Copy> copies;
        std::vector<int32_t> fresh;
        if (streamed_) fresh.reserve(page_count(end)); // allocate before ownership mutation
        if (!pages_.ensure(seq, size_t(begin / kernels::qsa_real_shapes().page_size), page_count(end), copies, error))
            return false;
        // Ownership preflight is atomic on shortage. After it succeeds, CUDA
        // failures are fatal: callers must never execute another uncertain graph.
        if (!sync(error)) return false;
        if (streamed_) {
            for (const auto& c : copies) fresh.push_back(c.to);
            const auto& map = pages_.mapping(seq);
            for (size_t i = old_size; i < map.size(); ++i) fresh.push_back(map[i]);
            // New/recycled IDs must not inherit the previous owner's GPU cache.
            if (!invalidate(fresh, error)) return false;
        }
        for (const auto& c : copies)
            for (int64_t j = 0; j < sessions_[0]->qsa_alloc; ++j)
                if (!copy_page(sessions_[0]->qsa_states[sessions_[0]->qsa_ord0 + j], c.from, c.to, error))
                    return false;
        return publish(seq, error);
    }

    bool truncate(size_t seq, int64_t cells, std::string& error) {
        if (cells < 0 || seq >= sessions_.size() || cells > sessions_[seq]->max_cells) {
            error = "unified KV: invalid truncate extent";
            return false;
        }
        std::vector<int32_t> freed;
        if (!sync(error) || !pages_.truncate(seq, page_count(cells), error, streamed_ ? &freed : nullptr)) return false;
        return invalidate(freed, error) && publish(seq, error);
    }

    bool clone_prefix(size_t from, size_t to, int64_t cells, std::string& error) {
        if (from >= sessions_.size() || to >= sessions_.size() || cells < 0 || cells > sessions_[from]->max_cells) {
            error = "unified KV: invalid clone extent";
            return false;
        }
        std::vector<int32_t> freed;
        if (!sync(error) || !pages_.clone_prefix(from, to, page_count(cells), error, streamed_ ? &freed : nullptr)) return false;
        return invalidate(freed, error) && publish(to, error);
    }

    // Transfer the COMPLETE reservation, including unwritten output capacity.
    bool move(size_t from, size_t to, std::string& error) {
        if (from >= sessions_.size() || to >= sessions_.size()) {
            error = "unified KV: invalid move sequence";
            return false;
        }
        if (from == to) return true;
        std::vector<int32_t> freed;
        if (!sync(error) || !pages_.clone_prefix(from, to, pages_.mapping(from).size(), error,
                                               streamed_ ? &freed : nullptr)) return false;
        pages_.release(from, streamed_ ? &freed : nullptr);
        return invalidate(freed, error) && publish(to, error) && publish(from, error);
    }

    bool release(size_t seq, std::string& error) {
        if (seq >= sessions_.size()) { error = "unified KV: invalid release sequence"; return false; }
        std::vector<int32_t> freed;
        if (!sync(error)) return false;
        pages_.release(seq, streamed_ ? &freed : nullptr);
        return invalidate(freed, error) && publish(seq, error);
    }

private:
    struct DeviceFree {
        void operator()(int32_t* p) const noexcept { if (p) cudaFree(p); }
    };
    const ModelGeometry& g_;
    std::vector<SessionState*> sessions_;
    SharedKvPages pages_;
    bool streamed_ = false;
    std::unique_ptr<int32_t, DeviceFree> invalidation_ids_;
    bool published_ = false;

    static size_t capacity(const std::vector<SessionState*>& sessions) {
        if (sessions.empty() || !sessions[0] || !sessions[0]->qsa_states || sessions[0]->qsa_alloc <= 0)
            throw std::invalid_argument("unified KV: missing owner session");
        const auto& q = sessions[0]->qsa_states[sessions[0]->qsa_primary()];
        return size_t(q.kv_mode == 1 ? q.n_pages : q.n_slots);
    }
    static bool checked(cudaError_t status, std::string& error) {
        if (status == cudaSuccess) return true;
        error = std::string("unified KV transfer: ") + cudaGetErrorString(status);
        return false;
    }
    static bool sync(std::string& error) { return checked(cudaDeviceSynchronize(), error); }

    bool invalidate(const std::vector<int32_t>& ids, std::string& error) {
        if (!streamed_ || ids.empty()) return true;
        if (!checked(cudaMemcpy(invalidation_ids_.get(), ids.data(), ids.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice), error)) return false;
        for (int64_t j = 0; j < sessions_[0]->qsa_alloc; ++j)
            kernels::kv_stream_invalidate_pages(sessions_[0]->qsa_states[sessions_[0]->qsa_ord0 + j].map,
                                               invalidation_ids_.get(), int64_t(ids.size()), nullptr);
        return sync(error);
    }

    bool publish(size_t seq, std::string& error) {
        const auto& map = pages_.mapping(seq);
        auto& s = *sessions_.at(seq);
        for (int64_t j = 0; j < s.qsa_alloc; ++j) {
            auto& q = s.qsa_states[s.qsa_ord0 + j];
            bool changed = false;
            for (size_t i = 0; i < q.shared_page_table.size(); ++i) {
                const int32_t id = i < map.size() ? map[i] : -1;
                if (q.shared_page_table[i] != id) { q.shared_page_table[i] = id; changed = true; }
            }
            if (changed || !published_) {
                int32_t* table = streamed_ ? const_cast<int32_t*>(q.host.logical_pages) : q.page_table;
                if (!sync(error) || !checked(cudaMemcpy(table, q.shared_page_table.data(),
                        q.shared_page_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice), error)) return false;
            }
        }
        if (seq + 1 == sessions_.size()) published_ = true;
        return true;
    }

    bool copy_page(const QsaState& q, int32_t from, int32_t to, std::string& error) {
        const size_t rows = size_t(kernels::qsa_real_shapes().page_size) * size_t(g_.n_head_kv);
        const size_t q4 = rows * kernels::kv_q4_bytes_per_head(int(g_.head_dim));
        const size_t code = rows * size_t(g_.head_dim);
        const size_t scales = rows * size_t(g_.head_dim / kernels::KV_Q8_GROUP) * sizeof(uint16_t);
        std::array<std::pair<void*, size_t>, 4> pools{};
        if (q.kv_hybrid) pools = {{{q.k_q, code}, {q.v_q4, q4}, {q.k_scale, scales}, {nullptr, 0}}};
        else if (q.kv_q4) pools = {{{streamed_ ? q.host.k_q4 : q.k_q4, q4},
                                  {streamed_ ? q.host.v_q4 : q.v_q4, q4}, {nullptr, 0}, {nullptr, 0}}};
        else if (q.kv_int8) pools = {{{streamed_ ? q.host.k_q : q.k_q, code},
                                    {streamed_ ? q.host.v_q : q.v_q, code},
                                    {streamed_ ? q.host.k_scale : q.k_scale, scales},
                                    {streamed_ ? q.host.v_scale : q.v_scale, scales}}};
        else pools = {{{streamed_ ? q.host.k_pool : q.k_pool, code * 2},
                       {streamed_ ? q.host.v_pool : q.v_pool, code * 2}, {nullptr, 0}, {nullptr, 0}}};
        for (auto [base, bytes] : pools)
            if (bytes && !checked(cudaMemcpy(static_cast<uint8_t*>(base) + size_t(to) * bytes,
                    static_cast<uint8_t*>(base) + size_t(from) * bytes, bytes, cudaMemcpyDefault), error)) return false;
        return true;
    }
};

} // namespace strata::core
