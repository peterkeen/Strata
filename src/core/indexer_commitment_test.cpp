#include "strata/core/indexer_commitment.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

int checks = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "indexer_commitment_test: failed: %s\n", what);
        std::exit(1);
    }
}

void check_code(IndexerCommitmentCode got, IndexerCommitmentCode want, const char* what) {
    ++checks;
    if (got != want) {
        std::fprintf(stderr, "indexer_commitment_test: failed: %s (got %.*s want %.*s)\n", what,
                     static_cast<int>(indexer_commitment_code_name(got).size()),
                     indexer_commitment_code_name(got).data(),
                     static_cast<int>(indexer_commitment_code_name(want).size()),
                     indexer_commitment_code_name(want).data());
        std::exit(1);
    }
}

IndexerExtentGeometry geom(int64_t cells = 32) {
    return {cells, 4, 128, 12, 1};
}

PrivateIndexerCommitmentLedger make_ledger(int64_t cells = 32, int64_t chunk_rows = 3) {
    PrivateIndexerCommitmentLedger ledger;
    check_code(PrivateIndexerCommitmentLedger::create(geom(cells), chunk_rows, ledger).code,
               IndexerCommitmentCode::ok, "valid ledger geometry creates");
    return ledger;
}

enum class Phase { none, acquire, map, access, initialize, publish, unmap };

struct Event {
    std::string phase;
    int64_t chunk = -1;
    IndexerChunkSource source = IndexerChunkSource::fresh;
    int64_t begin = 0;
    int64_t end = 0;
};

struct HookLog {
    Phase fail_phase = Phase::none;
    int64_t fail_chunk = -1;
    IndexerCommitmentCode fail_code = IndexerCommitmentCode::ok;
    std::vector<int64_t> borrowed_chunks;
    std::vector<Event> events;

    bool borrowed(int64_t chunk) const {
        return std::find(borrowed_chunks.begin(), borrowed_chunks.end(), chunk) != borrowed_chunks.end();
    }

    bool fail(Phase phase, int64_t chunk) const {
        return fail_phase == phase && (fail_chunk < 0 || fail_chunk == chunk);
    }

    IndexerCommitmentCallbacks callbacks() {
        IndexerCommitmentCallbacks cb;
        cb.acquire = [this](int64_t chunk, IndexerChunkSource& source) {
            events.push_back({"acquire", chunk, source, 0, 0});
            if (fail(Phase::acquire, chunk)) return fail_code;
            source = borrowed(chunk) ? IndexerChunkSource::borrowed : IndexerChunkSource::fresh;
            events.back().source = source;
            return IndexerCommitmentCode::ok;
        };
        cb.map = [this](int64_t chunk, IndexerChunkSource source) {
            events.push_back({"map", chunk, source, 0, 0});
            if (fail(Phase::map, chunk)) return fail_code;
            return IndexerCommitmentCode::ok;
        };
        cb.set_access = [this](int64_t chunk, IndexerChunkSource source) {
            events.push_back({"access", chunk, source, 0, 0});
            if (fail(Phase::access, chunk)) return fail_code;
            return IndexerCommitmentCode::ok;
        };
        cb.initialize = [this](int64_t chunk, int64_t begin, int64_t end) {
            events.push_back({"initialize", chunk, IndexerChunkSource::fresh, begin, end});
            if (fail(Phase::initialize, chunk)) return fail_code;
            return IndexerCommitmentCode::ok;
        };
        cb.publish = [this](int64_t old_physical, int64_t new_physical, int64_t old_initialized,
                            int64_t new_initialized) {
            events.push_back({"publish", -1, IndexerChunkSource::fresh, old_physical + old_initialized,
                              new_physical + new_initialized});
            if (fail(Phase::publish, -1)) return fail_code;
            return IndexerCommitmentCode::ok;
        };
        cb.unmap = [this](int64_t chunk, IndexerChunkSource source) {
            events.push_back({"unmap", chunk, source, 0, 0});
            if (fail(Phase::unmap, chunk)) return fail_code;
            return IndexerCommitmentCode::ok;
        };
        cb.release = [this](int64_t chunk, IndexerChunkSource source) {
            events.push_back({"release", chunk, source, 0, 0});
        };
        return cb;
    }

    int count(const char* phase) const {
        int n = 0;
        for (const Event& e : events) {
            if (e.phase == phase) ++n;
        }
        return n;
    }

    int count(const char* phase, int64_t chunk) const {
        int n = 0;
        for (const Event& e : events) {
            if (e.phase == phase && e.chunk == chunk) ++n;
        }
        return n;
    }

    bool saw(const char* phase, int64_t chunk, IndexerChunkSource source) const {
        for (const Event& e : events) {
            if (e.phase == phase && e.chunk == chunk && e.source == source) return true;
        }
        return false;
    }
};

void require_state(const PrivateIndexerCommitmentLedger& ledger, int64_t physical, int64_t initialized,
                   int64_t prepared, int64_t valid, int64_t completed, const char* what) {
    check(ledger.physical_committed_rows() == physical, what);
    check(ledger.initialized_rows() == initialized, what);
    check(ledger.prepared_writer_rows() == prepared, what);
    check(ledger.valid_history_rows() == valid, what);
    check(ledger.valid_completed_rows() == completed, what);
    check(ledger.valid_history_rows() <= ledger.initialized_rows(), "valid history never exceeds initialized rows");
    check(ledger.valid_history_rows() <= ledger.physical_committed_rows(), "valid history never exceeds physical rows");
}

void test_create_validation_is_atomic() {
    PrivateIndexerCommitmentLedger ledger = make_ledger();
    check(ledger.ceiling_rows() == 10, "ceiling rows come from I1 extent helper");
    check(ledger.chunk_rows() == 3, "chunk rows are recorded separately from logical ceiling");

    PrivateIndexerCommitmentLedger original = ledger;
    check_code(PrivateIndexerCommitmentLedger::create({16, 0, 128, 12, 1}, 3, ledger).code,
               IndexerCommitmentCode::invalid_geometry, "invalid I1 geometry rejected");
    check(ledger.ceiling_rows() == original.ceiling_rows(), "invalid geometry does not replace ledger");
    check_code(PrivateIndexerCommitmentLedger::create(geom(), 0, ledger).code,
               IndexerCommitmentCode::invalid_geometry, "invalid chunk geometry rejected");
    check(ledger.chunk_rows() == original.chunk_rows(), "invalid chunk geometry does not replace ledger");
    check_code(PrivateIndexerCommitmentLedger::create(
                   {std::numeric_limits<int64_t>::max(), 1, 128, 12, 1}, 3, ledger).code,
               IndexerCommitmentCode::overflow, "ceiling row overflow rejected");
    check(ledger.ceiling_rows() == original.ceiling_rows(), "overflow create does not replace ledger");
}

void test_successful_growth_separates_prepared_and_valid() {
    PrivateIndexerCommitmentLedger ledger = make_ledger();
    HookLog log;
    log.borrowed_chunks = {1};
    IndexerCommitmentCallbacks cb = log.callbacks();

    check_code(ledger.prepare_append(0, 9, cb).code, IndexerCommitmentCode::ok,
               "first writer extent commits one physical chunk");
    require_state(ledger, 3, 3, 3, 0, 0, "prepared rows publish without accepted history validity");
    check(log.count("acquire") == 1 && log.count("map") == 1 && log.count("access") == 1,
          "first growth acquires/maps/accesses exactly one chunk");
    check(log.count("initialize") == 1 && log.events[3].begin == 0 && log.events[3].end == 3,
          "first growth initializes only requested rows");
    check(log.count("publish") == 1, "first growth publishes initialized extent");

    check_code(ledger.publish_history(8).code, IndexerCommitmentCode::ok, "accepted history can publish within extent");
    require_state(ledger, 3, 3, 3, 3, 2, "history validity is separate from prepared writer extent");

    check_code(ledger.prepare_append(9, 4, cb).code, IndexerCommitmentCode::ok,
               "second writer extent can borrow a new chunk");
    require_state(ledger, 6, 4, 4, 3, 2, "chunk commitment can exceed initialized and valid rows");
    check(ledger.chunk_source(1) == IndexerChunkSource::borrowed, "borrowed chunk source is retained after commit");
    bool threw_bounds = false;
    try {
        (void) ledger.chunk_source(2);
    } catch (const std::out_of_range&) {
        threw_bounds = true;
    }
    check(threw_bounds, "chunk_source bounds-checks chunk indices");
    check(log.saw("acquire", 1, IndexerChunkSource::borrowed), "acquire records borrowed disposition");
    check(log.events.back().phase == "publish", "successful growth publishes after initialization");

    check_code(ledger.publish_history(12).code, IndexerCommitmentCode::ok, "valid history advances after writer success");
    require_state(ledger, 6, 4, 4, 4, 3, "valid rows can advance only inside initialized commitment");

    IndexerHistoryExtent too_large{};
    check_code(indexer_history_extent(geom(), 16, too_large) == IndexerExtentError::ok ?
                   ledger.publish_history_extent(too_large).code : IndexerCommitmentCode::overflow,
               IndexerCommitmentCode::ceiling_exceeded,
               "validity publication above initialized rows is rejected");
    require_state(ledger, 6, 4, 4, 4, 3, "rejected validity publication leaves prior history intact");
}

void test_chunk_rows_price_full_physical_commitment() {
    PrivateIndexerCommitmentLedger ledger = make_ledger(26, 3); // ceiling rows = floor(26 / 4) + 2 = 8
    HookLog log;
    IndexerCommitmentCallbacks cb = log.callbacks();
    check_code(ledger.prepare_append(0, 26, cb).code, IndexerCommitmentCode::ok,
               "non-divisible ceiling can commit final partial usable chunk");
    check(ledger.ceiling_rows() == 8, "test geometry ceiling is not divisible by chunk rows");
    check(ledger.physical_committed_rows() == 8, "usable committed rows are capped at logical ceiling");
    int64_t chunk_rows = -1;
    check_code(ledger.committed_chunk_rows(chunk_rows), IndexerCommitmentCode::ok,
               "committed chunk rows are available for physical pricing");
    check(chunk_rows == 9, "physical commitment reports whole chunks beyond logical ceiling");
    IndexerByteExtent bytes{};
    check_code(indexer_commitment_from_extent_error(indexer_byte_extent_for_rows(ledger.geometry(), chunk_rows, bytes)),
               IndexerCommitmentCode::ok, "whole committed chunk rows can be priced with I1 byte helper");
    check(bytes.rows == 9 && bytes.bytes_all_layers > 8 * 128 * 4, "byte extent uses whole physical chunks");
}

void test_noop_and_invalid_requests_do_not_call_adapter() {
    PrivateIndexerCommitmentLedger ledger = make_ledger();
    HookLog log;
    IndexerCommitmentCallbacks cb = log.callbacks();
    check_code(ledger.prepare_append(0, 9, cb).code, IndexerCommitmentCode::ok, "setup growth succeeds");
    const int events_after_setup = static_cast<int>(log.events.size());

    check_code(ledger.prepare_append(1, 1, cb).code, IndexerCommitmentCode::ok,
               "writer extent inside initialized rows is a no-op");
    require_state(ledger, 3, 3, 1, 0, 0, "no-op writer updates prepared extent but not physical rows");
    check(static_cast<int>(log.events.size()) == events_after_setup, "no-op writer does not call adapter phases");

    check_code(ledger.prepare_append(0, 0, cb).code, IndexerCommitmentCode::ok,
               "zero-length append prepares an empty extent");
    require_state(ledger, 3, 3, 0, 0, 0, "zero-length append clears prepared writer extent to zero");
    check(static_cast<int>(log.events.size()) == events_after_setup, "zero-length append does not call adapter phases");

    check_code(ledger.prepare_append(-1, 1, cb).code, IndexerCommitmentCode::negative_extent,
               "negative append start rejected");
    check_code(ledger.prepare_append(0, -1, cb).code, IndexerCommitmentCode::negative_extent,
               "negative append length rejected");
    check_code(ledger.prepare_append(31, 2, cb).code, IndexerCommitmentCode::ceiling_exceeded,
               "append past logical ceiling rejected");
    check_code(ledger.ensure_rows(-1, cb).code, IndexerCommitmentCode::negative_extent,
               "negative direct row request rejected");
    check_code(ledger.ensure_rows(11, cb).code, IndexerCommitmentCode::ceiling_exceeded,
               "direct row request beyond ceiling rejected");
    check_code(ledger.publish_history(-1).code, IndexerCommitmentCode::negative_extent,
               "negative history rejected");
    check_code(ledger.publish_history(33).code, IndexerCommitmentCode::ceiling_exceeded,
               "history beyond logical ceiling rejected");
    check(static_cast<int>(log.events.size()) == events_after_setup, "invalid requests do not call adapter phases");
    require_state(ledger, 3, 3, 0, 0, 0, "invalid requests leave ledger state unchanged");
}

PrivateIndexerCommitmentLedger one_chunk_setup() {
    PrivateIndexerCommitmentLedger ledger = make_ledger();
    HookLog setup;
    IndexerCommitmentCallbacks cb = setup.callbacks();
    check_code(ledger.prepare_append(0, 9, cb).code, IndexerCommitmentCode::ok, "one-chunk setup succeeds");
    ledger.clear_prepared_writer_rows();
    return ledger;
}

void test_acquire_failure_rolls_back_new_resources_once() {
    PrivateIndexerCommitmentLedger ledger = one_chunk_setup();
    HookLog log;
    log.fail_phase = Phase::acquire;
    log.fail_chunk = 2;
    log.fail_code = IndexerCommitmentCode::acquisition_failed;
    IndexerCommitmentCallbacks cb = log.callbacks();

    check_code(ledger.prepare_append(9, 15, cb).code, IndexerCommitmentCode::acquisition_failed,
               "acquire failure is recoverable");
    require_state(ledger, 3, 3, 0, 0, 0, "acquire failure preserves prior published state");
    check(!ledger.poisoned(), "recoverable acquire failure does not poison ledger");
    check(log.count("acquire", 1) == 1 && log.count("acquire", 2) == 1,
          "acquire failure attempts chunks up to failing chunk");
    check(log.count("map") == 0 && log.count("unmap") == 0, "acquire failure maps no chunks");
    check(log.count("release", 1) == 1 && log.count("release", 2) == 0,
          "only successfully acquired transaction chunks are released once");
}

void test_map_failure_rolls_back_mapped_and_unmapped_owned_chunks() {
    PrivateIndexerCommitmentLedger ledger = one_chunk_setup();
    HookLog log;
    log.borrowed_chunks = {2};
    log.fail_phase = Phase::map;
    log.fail_chunk = 2;
    log.fail_code = IndexerCommitmentCode::map_failed;
    IndexerCommitmentCallbacks cb = log.callbacks();

    check_code(ledger.prepare_append(9, 15, cb).code, IndexerCommitmentCode::map_failed,
               "map failure is recoverable after proven rollback");
    require_state(ledger, 3, 3, 0, 0, 0, "map failure preserves prior published state");
    check(!ledger.poisoned(), "recoverable map failure does not poison ledger");
    check(log.count("release", 1) == 1 && log.count("release", 2) == 1,
          "map failure releases both acquired chunks exactly once");
    check(log.count("unmap", 1) == 1 && log.count("unmap", 2) == 0,
          "map failure unmaps only chunks whose map phase succeeded");
    check(log.saw("release", 2, IndexerChunkSource::borrowed), "borrowed failed-map chunk returns as borrowed");
}

void test_initialization_failure_rolls_back_after_access() {
    PrivateIndexerCommitmentLedger ledger = one_chunk_setup();
    HookLog log;
    log.fail_phase = Phase::initialize;
    log.fail_chunk = 2;
    log.fail_code = IndexerCommitmentCode::initialization_failed;
    IndexerCommitmentCallbacks cb = log.callbacks();

    check_code(ledger.prepare_append(9, 15, cb).code, IndexerCommitmentCode::initialization_failed,
               "initialization failure is recoverable");
    require_state(ledger, 3, 3, 0, 0, 0, "initialization failure leaves old initialized rows published");
    check(log.count("access", 1) == 1 && log.count("access", 2) == 1,
          "initialization failure happens after access is set on all new chunks");
    check(log.count("initialize", 1) == 1 && log.count("initialize", 2) == 1,
          "partial initialization before a later chunk failure stays unpublished");
    check(log.count("unmap", 1) == 1 && log.count("unmap", 2) == 1,
          "initialization failure unmaps every transaction-owned mapping");
    check(log.count("release", 1) == 1 && log.count("release", 2) == 1,
          "initialization failure releases every transaction-owned chunk once");
    check(!ledger.poisoned(), "recoverable initialization failure does not poison ledger");
}

void test_existing_chunk_initialization_failure_keeps_unpublished_rows_private() {
    PrivateIndexerCommitmentLedger ledger = make_ledger();
    HookLog setup;
    IndexerCommitmentCallbacks setup_cb = setup.callbacks();
    check_code(ledger.prepare_append(0, 13, setup_cb).code, IndexerCommitmentCode::ok,
               "setup commits two chunks but initializes four rows");
    require_state(ledger, 6, 4, 4, 0, 0, "setup has existing uninitialized capacity in a committed chunk");
    ledger.clear_prepared_writer_rows();

    HookLog fail;
    fail.fail_phase = Phase::initialize;
    fail.fail_chunk = 1;
    fail.fail_code = IndexerCommitmentCode::initialization_failed;
    IndexerCommitmentCallbacks cb = fail.callbacks();
    check_code(ledger.ensure_rows(5, cb).code, IndexerCommitmentCode::initialization_failed,
               "recoverable initialization failure in an existing chunk leaves unpublished bytes private");
    require_state(ledger, 6, 4, 0, 0, 0, "existing-chunk init failure does not advance initialized rows");
    check(fail.count("acquire") == 0 && fail.count("map") == 0 && fail.count("release") == 0,
          "existing-chunk init failure has no transaction-owned chunks to return");
    check(!ledger.poisoned(), "recoverable existing-chunk initialization failure does not poison ledger");
}

void test_recoverable_publication_failure_rolls_back() {
    PrivateIndexerCommitmentLedger ledger = one_chunk_setup();
    HookLog log;
    log.fail_phase = Phase::publish;
    log.fail_code = IndexerCommitmentCode::publication_failed;
    IndexerCommitmentCallbacks cb = log.callbacks();

    check_code(ledger.prepare_append(9, 15, cb).code, IndexerCommitmentCode::publication_failed,
               "recoverable publication failure returns publication_failed");
    require_state(ledger, 3, 3, 0, 0, 0, "recoverable publication failure preserves prior state");
    check(log.count("publish") == 1, "publication phase is injected");
    check(log.count("unmap", 1) == 1 && log.count("unmap", 2) == 1,
          "recoverable publication failure unmaps transaction chunks");
    check(log.count("release", 1) == 1 && log.count("release", 2) == 1,
          "recoverable publication failure releases transaction chunks once");
    check(!ledger.poisoned(), "recoverable publication failure does not poison ledger");
}

void test_callback_exceptions_poison_conservatively() {
    {
        PrivateIndexerCommitmentLedger ledger = one_chunk_setup();
        bool release_called = false;
        IndexerCommitmentCallbacks cb;
        cb.acquire = [](int64_t, IndexerChunkSource&) -> IndexerCommitmentCode {
            throw std::runtime_error("acquire throw");
        };
        cb.release = [&](int64_t, IndexerChunkSource) { release_called = true; };
        check_code(ledger.prepare_append(9, 4, cb).code, IndexerCommitmentCode::poisoned,
                   "acquire exception poisons because resource ownership is unknown");
        check(ledger.poisoned(), "acquire exception leaves ledger poisoned");
        check(!release_called, "acquire exception does not fabricate a resource release");
    }
    {
        PrivateIndexerCommitmentLedger ledger = one_chunk_setup();
        HookLog log;
        IndexerCommitmentCallbacks cb = log.callbacks();
        cb.publish = [](int64_t, int64_t, int64_t, int64_t) -> IndexerCommitmentCode {
            throw std::runtime_error("publish throw");
        };
        check_code(ledger.prepare_append(9, 15, cb).code, IndexerCommitmentCode::publication_uncertain,
                   "publish exception becomes publication uncertainty");
        check(ledger.poisoned(), "publish exception poisons ledger");
        check(log.count("unmap") == 0 && log.count("release") == 0,
              "publish exception does not fake rollback or release transaction resources");
    }
    {
        PrivateIndexerCommitmentLedger ledger = make_ledger();
        HookLog grow;
        IndexerCommitmentCallbacks grow_cb = grow.callbacks();
        check_code(ledger.prepare_append(0, 24, grow_cb).code, IndexerCommitmentCode::ok,
                   "setup for unmap exception succeeds");
        ledger.clear_prepared_writer_rows();
        bool release_called = false;
        IndexerCommitmentCallbacks cb;
        cb.unmap = [](int64_t, IndexerChunkSource) -> IndexerCommitmentCode {
            throw std::runtime_error("unmap throw");
        };
        cb.release = [&](int64_t, IndexerChunkSource) { release_called = true; };
        check_code(ledger.shrink_to_rows(3, cb).code, IndexerCommitmentCode::unmap_uncertain,
                   "unmap exception becomes unmap uncertainty");
        check(ledger.poisoned(), "unmap exception poisons ledger");
        check(!release_called, "unmap exception does not release an uncertain chunk");
    }
}

void test_access_and_publication_uncertainty_poison_without_fake_rollback() {
    {
        PrivateIndexerCommitmentLedger ledger = one_chunk_setup();
        HookLog log;
        log.fail_phase = Phase::access;
        log.fail_chunk = 1;
        log.fail_code = IndexerCommitmentCode::access_uncertain;
        IndexerCommitmentCallbacks cb = log.callbacks();
        check_code(ledger.prepare_append(9, 15, cb).code, IndexerCommitmentCode::access_uncertain,
                   "access uncertainty is fatal");
        check(ledger.poisoned(), "access uncertainty poisons ledger");
        check(log.count("unmap") == 0 && log.count("release") == 0,
              "access uncertainty does not pretend rollback is guaranteed");
        check_code(ledger.prepare_append(0, 0, cb).code, IndexerCommitmentCode::poisoned,
                   "poisoned ledger rejects future writers");
    }
    {
        PrivateIndexerCommitmentLedger ledger = one_chunk_setup();
        HookLog log;
        log.fail_phase = Phase::access;
        log.fail_chunk = 2;
        log.fail_code = IndexerCommitmentCode::access_uncertain;
        IndexerCommitmentCallbacks cb = log.callbacks();
        check_code(ledger.prepare_append(9, 15, cb).code, IndexerCommitmentCode::access_uncertain,
                   "later chunk access uncertainty is also fatal");
        check(ledger.poisoned(), "later chunk access uncertainty poisons ledger");
        check(log.count("access", 1) == 1 && log.count("access", 2) == 1,
              "later chunk access failure is injected after earlier chunk access succeeds");
        check(log.count("unmap") == 0 && log.count("release") == 0,
              "later chunk access uncertainty does not fake rollback");
    }
    {
        PrivateIndexerCommitmentLedger ledger = one_chunk_setup();
        HookLog log;
        log.fail_phase = Phase::publish;
        log.fail_code = IndexerCommitmentCode::publication_uncertain;
        IndexerCommitmentCallbacks cb = log.callbacks();
        check_code(ledger.prepare_append(9, 15, cb).code, IndexerCommitmentCode::publication_uncertain,
                   "publication uncertainty is fatal");
        check(ledger.poisoned(), "publication uncertainty poisons ledger");
        check(log.count("unmap") == 0 && log.count("release") == 0,
              "publication uncertainty does not fake rollback or return resources");
        check_code(ledger.shrink_to_rows(0, cb).code, IndexerCommitmentCode::poisoned,
                   "poisoned ledger rejects shrink");
    }
}

void test_rollback_unmap_uncertainty_poisons() {
    PrivateIndexerCommitmentLedger ledger = one_chunk_setup();
    HookLog log;
    log.fail_phase = Phase::map;
    log.fail_chunk = 2;
    log.fail_code = IndexerCommitmentCode::map_failed;
    IndexerCommitmentCallbacks cb = log.callbacks();
    HookLog poison_unmap;
    poison_unmap.fail_phase = Phase::unmap;
    poison_unmap.fail_chunk = 1;
    poison_unmap.fail_code = IndexerCommitmentCode::unmap_uncertain;

    IndexerCommitmentCallbacks poison_cb = poison_unmap.callbacks();
    poison_cb.acquire = cb.acquire;
    poison_cb.map = cb.map;
    poison_cb.set_access = cb.set_access;
    poison_cb.initialize = cb.initialize;
    poison_cb.publish = cb.publish;
    poison_cb.release = cb.release;

    check_code(ledger.prepare_append(9, 15, poison_cb).code, IndexerCommitmentCode::unmap_uncertain,
               "uncertain unmap during rollback is fatal");
    check(ledger.poisoned(), "rollback unmap uncertainty poisons ledger");
    check(poison_unmap.count("unmap", 1) == 1, "rollback reports the uncertain unmap chunk");
}

void test_shrink_retention_and_borrowed_release() {
    PrivateIndexerCommitmentLedger ledger = make_ledger();
    HookLog grow;
    grow.borrowed_chunks = {1};
    IndexerCommitmentCallbacks grow_cb = grow.callbacks();
    check_code(ledger.prepare_append(0, 24, grow_cb).code, IndexerCommitmentCode::ok,
               "setup grows through three chunks");
    check_code(ledger.publish_history(16).code, IndexerCommitmentCode::ok, "setup publishes retained history");
    require_state(ledger, 9, 7, 7, 5, 4, "setup has chunk-rounded physical rows and exact initialized rows");

    check_code(ledger.set_reader_retention_rows(6).code, IndexerCommitmentCode::ok,
               "reader retention can hold initialized rows");
    check_code(ledger.set_checkpoint_retention_rows(5).code, IndexerCommitmentCode::ok,
               "checkpoint retention can hold initialized rows");
    HookLog shrink_log;
    IndexerCommitmentCallbacks shrink_cb = shrink_log.callbacks();
    check_code(ledger.shrink_to_rows(6, shrink_cb).code, IndexerCommitmentCode::retained_extent,
               "prepared writer retention refuses shrink");
    check(ledger.shrink_floor_rows() == 7, "prepared writer extent is separate highest shrink floor");
    ledger.clear_prepared_writer_rows();
    check_code(ledger.shrink_to_rows(5, shrink_cb).code, IndexerCommitmentCode::retained_extent,
               "reader retention refuses shrink");
    check(ledger.shrink_floor_rows() == 6, "reader retention remains after clearing prepared writer");
    ledger.clear_reader_retention_rows();
    check_code(ledger.shrink_to_rows(4, shrink_cb).code, IndexerCommitmentCode::retained_extent,
               "checkpoint retention refuses shrink");
    check(ledger.shrink_floor_rows() == 5, "checkpoint retention remains separate from reader retention");
    ledger.clear_checkpoint_retention_rows();
    check_code(ledger.shrink_to_rows(4, shrink_cb).code, IndexerCommitmentCode::retained_extent,
               "valid history refuses shrink below accepted rows");

    check_code(ledger.publish_history(12).code, IndexerCommitmentCode::ok, "history can shrink valid rows before trim");
    check_code(ledger.shrink_to_rows(4, shrink_cb).code, IndexerCommitmentCode::ok,
               "retained tail shrink succeeds at chunk granularity");
    require_state(ledger, 6, 6, 0, 4, 3, "shrink keeps the chunk containing retained tail rows");
    check(shrink_log.count("unmap", 2) == 1 && shrink_log.count("release", 2) == 1,
          "shrink releases only fully discarded chunk once");
    check(shrink_log.count("unmap", 1) == 0 && shrink_log.count("release", 1) == 0,
          "shrink retains borrowed chunk that contains retained tail rows");

    check_code(ledger.publish_history(8).code, IndexerCommitmentCode::ok, "history can shrink again");
    check_code(ledger.shrink_to_rows(3, shrink_cb).code, IndexerCommitmentCode::ok,
               "second shrink releases borrowed chunk after retention clears");
    require_state(ledger, 3, 3, 0, 3, 2, "second shrink reaches one physical chunk");
    check(shrink_log.saw("release", 1, IndexerChunkSource::borrowed),
          "borrowed committed chunk is returned with borrowed disposition");
    check(shrink_log.count("release", 1) == 1 && shrink_log.count("release", 2) == 1,
          "each shrunk committed chunk is released once");
}

void test_shrink_invalid_and_uncertain_unmap() {
    PrivateIndexerCommitmentLedger ledger = make_ledger();
    HookLog grow;
    IndexerCommitmentCallbacks grow_cb = grow.callbacks();
    check_code(ledger.prepare_append(0, 24, grow_cb).code, IndexerCommitmentCode::ok,
               "setup for shrink invalid tests succeeds");
    check_code(ledger.set_reader_retention_rows(-1).code, IndexerCommitmentCode::negative_extent,
               "negative reader retention rejected");
    check_code(ledger.set_checkpoint_retention_rows(8).code, IndexerCommitmentCode::ceiling_exceeded,
               "checkpoint retention above initialized rows rejected");
    check(ledger.reader_retained_rows() == 0 && ledger.checkpoint_retained_rows() == 0,
          "invalid retention requests leave watermarks unchanged");

    HookLog bad_shrink;
    IndexerCommitmentCallbacks cb = bad_shrink.callbacks();
    check_code(ledger.shrink_to_rows(-1, cb).code, IndexerCommitmentCode::negative_extent,
               "negative shrink rejected");
    check_code(ledger.shrink_to_rows(11, cb).code, IndexerCommitmentCode::ceiling_exceeded,
               "shrink beyond ceiling rejected");
    require_state(ledger, 9, 7, 7, 0, 0, "invalid shrink requests leave state unchanged");

    ledger.clear_prepared_writer_rows();
    bad_shrink.fail_phase = Phase::unmap;
    bad_shrink.fail_chunk = 2;
    bad_shrink.fail_code = IndexerCommitmentCode::unmap_uncertain;
    cb = bad_shrink.callbacks();
    check_code(ledger.shrink_to_rows(3, cb).code, IndexerCommitmentCode::unmap_uncertain,
               "uncertain unmap during shrink is fatal");
    check(ledger.poisoned(), "uncertain shrink unmap poisons ledger");
    check(bad_shrink.count("release", 2) == 0, "uncertain shrink unmap does not release the uncertain chunk");
}

} // namespace

int main() {
    test_create_validation_is_atomic();
    test_successful_growth_separates_prepared_and_valid();
    test_chunk_rows_price_full_physical_commitment();
    test_noop_and_invalid_requests_do_not_call_adapter();
    test_acquire_failure_rolls_back_new_resources_once();
    test_map_failure_rolls_back_mapped_and_unmapped_owned_chunks();
    test_initialization_failure_rolls_back_after_access();
    test_existing_chunk_initialization_failure_keeps_unpublished_rows_private();
    test_recoverable_publication_failure_rolls_back();
    test_callback_exceptions_poison_conservatively();
    test_access_and_publication_uncertainty_poison_without_fake_rollback();
    test_rollback_unmap_uncertainty_poisons();
    test_shrink_retention_and_borrowed_release();
    test_shrink_invalid_and_uncertain_unmap();
    std::printf("indexer_commitment_test: %d checks passed\n", checks);
    return 0;
}
