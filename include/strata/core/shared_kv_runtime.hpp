// Host-owned mappings for one resident physical KV pool shared by serving slots.
#pragma once

#include "strata/core/session.hpp"
#include "strata/core/shared_kv_pages.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <array>
#include <stdexcept>

namespace strata::core {

// All mutations occur on the serving thread, outside captured graphs. The first
// session owns the allocation; all other sessions borrow its physical pools.
class SharedKvRuntime {
public:
    SharedKvRuntime(const ModelGeometry& g, std::vector<SessionState*> sessions)
        : g_(g), sessions_(std::move(sessions)),
          pages_(capacity(sessions_), sessions_.size()) {
        for (auto* s : sessions_) {
            if (!s || s->qsa_alloc != sessions_[0]->qsa_alloc || s->max_cells != sessions_[0]->max_cells)
                throw std::invalid_argument("unified KV: incompatible session");
            for (int64_t j = 0; j < s->qsa_alloc; ++j) {
                auto& q = s->qsa_states[s->qsa_ord0 + j];
                if (q.kv_mode != 0 || q.n_slots != int64_t(pages_.capacity()))
                    throw std::invalid_argument("unified KV requires fully resident pools");
                q.shared_kv = true;
                q.shared_page_table.assign(size_t(q.n_pages), -1);
            }
        }
        std::string error;
        for (size_t i = 0; i < sessions_.size(); ++i)
            if (!publish(i, error)) throw std::runtime_error(error);
    }

    const SharedKvPages& pages() const { return pages_; }
    size_t page_count(int64_t cells) const {
        const int64_t p = kernels::qsa_real_shapes().page_size;
        return size_t(cells / p + (cells % p != 0));
    }

    bool ensure(size_t seq, int64_t begin, int64_t end, std::string& error) {
        if (seq >= sessions_.size() || begin < 0 || end < begin || end > sessions_[seq]->max_cells) {
            error = "unified KV: invalid write extent";
            return false;
        }
        std::vector<SharedKvPages::Copy> copies;
        if (!pages_.ensure(seq, size_t(begin / kernels::qsa_real_shapes().page_size), page_count(end), copies, error))
            return false;
        // Successful bookkeeping must be followed by copies before the new map
        // is visible to a graph. A CUDA failure is fatal to the calling engine.
        if (!copies.empty()) {
            if (!sync(error)) return false;
            for (const auto& c : copies)
                for (int64_t j = 0; j < sessions_[0]->qsa_alloc; ++j)
                    if (!copy_page(sessions_[0]->qsa_states[sessions_[0]->qsa_ord0 + j], c.from, c.to, error))
                        return false;
        }
        return publish(seq, error);
    }

    bool truncate(size_t seq, int64_t cells, std::string& error) {
        if (cells < 0 || seq >= sessions_.size() || cells > sessions_[seq]->max_cells) {
            error = "unified KV: invalid truncate extent";
            return false;
        }
        if (!sync(error) || !pages_.truncate(seq, page_count(cells), error)) return false;
        return publish(seq, error);
    }

    bool clone_prefix(size_t from, size_t to, int64_t cells, std::string& error) {
        if (cells < 0 || !sync(error)) return false;
        if (!pages_.clone_prefix(from, to, page_count(cells), error)) return false;
        return publish(to, error);
    }

    // Transfer the complete reservation, including unwritten output capacity.
    // This avoids admission/slot copies and reserves enough space for completion.
    bool move(size_t from, size_t to, std::string& error) {
        if (from == to) return true;
        if (!sync(error) || !pages_.clone_prefix(from, to, pages_.mapping(from).size(), error)) return false;
        pages_.release(from);
        return publish(to, error) && publish(from, error);
    }

    bool release(size_t seq, std::string& error) {
        if (!sync(error)) return false;
        pages_.release(seq);
        return publish(seq, error);
    }

private:
    const ModelGeometry& g_;
    std::vector<SessionState*> sessions_;
    SharedKvPages pages_;

    static size_t capacity(const std::vector<SessionState*>& sessions) {
        if (sessions.empty() || !sessions[0] || !sessions[0]->qsa_states || sessions[0]->qsa_alloc <= 0)
            throw std::invalid_argument("unified KV: missing owner session");
        return size_t(sessions[0]->qsa_states[sessions[0]->qsa_primary()].n_slots);
    }
    static bool checked(cudaError_t status, std::string& error) {
        if (status == cudaSuccess) return true;
        error = std::string("unified KV transfer: ") + cudaGetErrorString(status);
        return false;
    }
    static bool sync(std::string& error) { return checked(cudaDeviceSynchronize(), error); }

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
            // Borrowed tables are already initialized to -1. The owner starts
            // with identity addressing, so its first publication is mandatory.
            if (changed || !published_) {
                if (!sync(error) || !checked(cudaMemcpy(q.page_table, q.shared_page_table.data(),
                        q.shared_page_table.size() * sizeof(int32_t), cudaMemcpyHostToDevice), error)) return false;
            }
        }
        if (seq + 1 == sessions_.size()) published_ = true;
        return true;
    }
    bool published_ = false;

    bool copy_page(const QsaState& q, int32_t from, int32_t to, std::string& error) {
        const size_t rows = size_t(kernels::qsa_real_shapes().page_size) * size_t(g_.n_head_kv);
        const size_t q4 = rows * kernels::kv_q4_bytes_per_head(int(g_.head_dim));
        const size_t code = rows * size_t(g_.head_dim);
        const size_t scales = rows * size_t(g_.head_dim / kernels::KV_Q8_GROUP) * sizeof(uint16_t);
        std::array<std::pair<void*, size_t>, 4> pools{};
        if (q.kv_hybrid) pools = {{{q.k_q, code}, {q.v_q4, q4}, {q.k_scale, scales}, {nullptr, 0}}};
        else if (q.kv_q4) pools = {{{q.k_q4, q4}, {q.v_q4, q4}, {nullptr, 0}, {nullptr, 0}}};
        else if (q.kv_int8) pools = {{{q.k_q, code}, {q.v_q, code}, {q.k_scale, scales}, {q.v_scale, scales}}};
        else pools = {{{q.k_pool, code * 2}, {q.v_pool, code * 2}, {nullptr, 0}, {nullptr, 0}}};
        for (auto [base, bytes] : pools)
            if (bytes && !checked(cudaMemcpy(static_cast<uint8_t*>(base) + size_t(to) * bytes,
                    static_cast<uint8_t*>(base) + size_t(from) * bytes, bytes, cudaMemcpyDeviceToDevice), error)) return false;
        return true;
    }
};

} // namespace strata::core
