// include/strata/program/tier_b_export.hpp - opt-in bounded export of completed native
// verification-window evidence and full logit rows (Tier B, sta-osh.5).
//
// This is COMPLETED-WINDOW correctness evidence, distinct from sta-osh.4's dispatch/capture records:
// every window line here is written only after Verifier::run + Verifier::commit have completed (the
// caller waits commit only when export is enabled), using the native window's own head logits via
// Verifier::copy_logits (NOT the ordinary --dump-logits path).  It is a bounded snapshot for paired
// correctness comparisons, never a claim of full coverage and never a performance measurement.
//
// Opt-in via --tier-b-export DIR (or STRATA_TIER_B_EXPORT). Off = no files, no I/O, no allocation, no
// readback beyond one cached branch in the caller. On = a new private DIR (must not already exist) with:
//
//   manifest.txt  (text)  versioned header, one `W` line per completed window, then at most one
//                         `# truncation 1` and one `# complete 1` footer line.
//   rows.bin      (binary) a 32-byte header (magic "TBE1", version, vocab, row_floats, reserved)
//                         followed by raw float32 logit rows (row-major, n_rows x vocab).
//
// Manifest schema (tokens are space-separated; hashes are 16-hex):
//   # tier-b native window export v=1
//   # vocab <n>
//   # max_windows <n>
//   # max_row_bytes <n>
//   W <ordinal> <pos> <T> <accepted> <emitted> <input_hash> <window_hash> <rows_offset> <rows_bytes> <status> natural picks=<id0,id1,...>
//   # truncation 1        (only when a budget was exhausted)
//   # export_error <id>   (only when an I/O/shape/arithmetic/finite check failed)
//   # complete 1          (only after a fully successful generation with no export failure)
//
// Bounds: hard max_windows and max_row_bytes (float data bytes, excluding the 32-byte rows.bin header).
// The caller pre-reserves the WHOLE window before any residual/logit readback; exceeding a budget sets the
// truncation marker and stops further export work (the engine keeps running).  No partial window rows are
// accepted.  Fail closed: a readback/shape/I/O failure marks the export failed, finish() refuses `# complete 1`,
// and the caller turns that into a non-zero CLI exit.
#pragma once

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <direct.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strata::program {

class TierBExport {
  public:
    bool open(const std::string& dir, int64_t max_windows, int64_t max_row_bytes, int64_t vocab, std::string& err);
    bool enabled() const { return manifest_ != nullptr; }

    /// Pre-reserve one WHOLE window. False with truncated()==true means budget exhausted (not an error); false with
    /// failed()==true means an arithmetic/shape error. The caller must not read back logits when this is false.
    bool can_write_window(int T, std::string& err);

    /// Append exactly n_rows full logit rows. No partial success: all bytes are written or this fails/truncates.
    bool write_rows(const float* rows, int64_t n_rows, int64_t& offset, int64_t& bytes, std::string& err);

    /// One completed natural window. `status` is "ok" or a short error id; any non-ok status fails closed.
    bool write_window(int64_t ordinal, int64_t pos, int T, int accepted, int emitted,
                      uint64_t input_hash, uint64_t window_hash, int64_t rows_offset, int64_t rows_bytes,
                      const int32_t* picks, int n_picks, const char* status, std::string& err);

    /// Completion footer ONLY after a fully successful generation. Rows are flushed and closed before a successful
    /// manifest footer can be written; then the manifest is flushed and closed and every close error is checked.
    bool finish(bool complete, std::string& err);

    bool truncated() const { return truncated_; }
    bool failed() const { return failed_; }
    int64_t windows_written() const { return windows_; }
    ~TierBExport() { close_quiet(); }

  private:
    static constexpr int64_t kHeaderBytes = 32;

    bool fail(const char* msg, std::string& err) { failed_ = true; err = msg; return false; }
    bool row_bytes_for(int64_t n_rows, int64_t& nbytes, std::string& err);
    bool write_rows_header(std::string& err);
    bool close_one(std::FILE*& f, const char* which, bool flush, std::string& err);
    void close_quiet() {
        if (manifest_ != nullptr) { std::fclose(manifest_); manifest_ = nullptr; }
        if (rows_ != nullptr) { std::fclose(rows_); rows_ = nullptr; }
    }

    std::FILE* manifest_ = nullptr;
    std::FILE* rows_ = nullptr;
    int64_t vocab_ = 0;
    int64_t max_windows_ = 0;
    int64_t max_row_bytes_ = 0;
    int64_t windows_ = 0;
    int64_t row_bytes_ = 0;
    bool truncated_ = false;
    bool failed_ = false;
    std::string error_id_;
};

inline std::FILE* tier_b_open_exclusive(const std::string& path) {
#ifdef _WIN32
    return std::fopen(path.c_str(), "wbx");
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL
#ifdef O_NOFOLLOW
                          | O_NOFOLLOW
#endif
                          , 0600);
    if (fd < 0) return nullptr;
    std::FILE* f = ::fdopen(fd, "wb");
    if (f == nullptr) { ::close(fd); return nullptr; }
    return f;
#endif
}

inline bool TierBExport::open(const std::string& dir, int64_t max_windows, int64_t max_row_bytes,
                              int64_t vocab, std::string& err) {
    if (enabled()) return fail("tier-b export: already open", err);
    if (dir.empty()) { err = "tier-b export: empty dir"; return false; }
    if (max_windows <= 0 || max_row_bytes <= 0 || vocab <= 0) {
        err = "tier-b export: non-positive budget/vocab"; return false;
    }
#ifdef _WIN32
    if (_mkdir(dir.c_str()) != 0) { err = "tier-b export: mkdir (directory must be new): " + dir; return false; }
#else
    if (::mkdir(dir.c_str(), 0700) != 0) { err = "tier-b export: mkdir (directory must be new): " + dir; return false; }
#endif
    const std::string mpath = dir + "/manifest.txt", rpath = dir + "/rows.bin";
    manifest_ = tier_b_open_exclusive(mpath);
    rows_ = tier_b_open_exclusive(rpath);
    if (manifest_ == nullptr || rows_ == nullptr) {
        err = "tier-b export: cannot create " + (manifest_ == nullptr ? mpath : rpath);
        close_quiet(); return false;
    }
    vocab_ = vocab;
    max_windows_ = max_windows;
    max_row_bytes_ = max_row_bytes;
    if (std::fprintf(manifest_, "# tier-b native window export v=1\n# vocab %lld\n# max_windows %lld\n# max_row_bytes %lld\n",
                     (long long) vocab_, (long long) max_windows_, (long long) max_row_bytes_) < 0) {
        return fail("tier-b export: manifest header io", err);
    }
    if (!write_rows_header(err)) return false;
    return true;
}

inline bool TierBExport::row_bytes_for(int64_t n_rows, int64_t& nbytes, std::string& err) {
    nbytes = 0;
    if (vocab_ <= 0 || n_rows <= 0) return fail("tier-b export: non-positive row shape", err);
    if (vocab_ > INT64_MAX / 4) return fail("tier-b export: vocab byte overflow", err);
    const int64_t vbytes = vocab_ * 4;
    if (n_rows > INT64_MAX / vbytes) return fail("tier-b export: row byte overflow", err);
    nbytes = n_rows * vbytes;
    return true;
}

inline bool TierBExport::write_rows_header(std::string& err) {
    char hdr[kHeaderBytes] = {};
    std::memcpy(hdr, "TBE1", 4);
    const uint32_t ver = 1;
    std::memcpy(hdr + 4, &ver, 4);
    int64_t v = vocab_, rf = vocab_, rsv = 0;
    std::memcpy(hdr + 8, &v, 8);
    std::memcpy(hdr + 16, &rf, 8);
    std::memcpy(hdr + 24, &rsv, 8);
    if (std::fwrite(hdr, 1, sizeof hdr, rows_) != sizeof hdr) return fail("tier-b export: rows header io", err);
    return true;
}

inline bool TierBExport::can_write_window(int T, std::string& err) {
    if (!enabled() || truncated_ || failed_) return false;
    int64_t nbytes = 0;
    if (!row_bytes_for(T, nbytes, err)) { error_id_ = "shape"; return false; }
    if (windows_ >= max_windows_ || nbytes > max_row_bytes_ - row_bytes_) { truncated_ = true; return false; }
    return true;
}

inline bool TierBExport::write_rows(const float* rows, int64_t n_rows, int64_t& offset, int64_t& bytes,
                                    std::string& err) {
    offset = -1; bytes = 0;
    if (!enabled() || truncated_ || failed_) return true;
    int64_t nbytes = 0;
    if (rows == nullptr) return fail("tier-b export: write_rows null", err);
    if (!row_bytes_for(n_rows, nbytes, err)) { error_id_ = "shape"; return false; }
    if (nbytes > max_row_bytes_ - row_bytes_) { truncated_ = true; return true; }
    const int64_t nf = n_rows * vocab_;
    for (int64_t i = 0; i < nf; ++i)
        if (!std::isfinite(rows[i])) { error_id_ = "nonfinite"; return fail("tier-b export: non-finite logits", err); }
    offset = kHeaderBytes + row_bytes_;
    if (std::fwrite(rows, 1, (size_t) nbytes, rows_) != (size_t) nbytes) {
        error_id_ = "rows_io"; return fail("tier-b export: rows io", err);
    }
    row_bytes_ += nbytes;
    bytes = nbytes;
    return true;
}

inline bool TierBExport::write_window(int64_t ordinal, int64_t pos, int T, int accepted, int emitted,
                                      uint64_t input_hash, uint64_t window_hash,
                                      int64_t rows_offset, int64_t rows_bytes, const int32_t* picks, int n_picks,
                                      const char* status, std::string& err) {
    if (!enabled() || truncated_) return true;
    if (status == nullptr) status = "ok";
    int64_t want = 0;
    if (!row_bytes_for(T, want, err)) { error_id_ = "shape"; return false; }
    if (accepted < 0 || accepted >= T || emitted < 0 || emitted > accepted + 1 || n_picks != T || picks == nullptr)
        return fail("tier-b export: bad window shape", err);
    if (std::strcmp(status, "ok") == 0) {
        if (rows_offset < kHeaderBytes || rows_bytes != want) return fail("tier-b export: bad row span", err);
    } else {
        failed_ = true;
        if (error_id_.empty()) error_id_ = status;
    }
    if (windows_ >= max_windows_) { truncated_ = true; return true; }
    if (std::fprintf(manifest_, "W %lld %lld %d %d %d %016llx %016llx %lld %lld %s natural picks=",
                     (long long) ordinal, (long long) pos, T, accepted, emitted,
                     (unsigned long long) input_hash, (unsigned long long) window_hash,
                     (long long) rows_offset, (long long) rows_bytes, status) < 0)
        return fail("tier-b export: manifest io", err);
    for (int i = 0; i < n_picks; ++i) {
        if (std::fprintf(manifest_, "%s%d", i == 0 ? "" : ",", (int) picks[i]) < 0)
            return fail("tier-b export: manifest io", err);
    }
    if (std::fprintf(manifest_, "\n") < 0) return fail("tier-b export: manifest io", err);
    ++windows_;
    return true;
}

inline bool TierBExport::close_one(std::FILE*& f, const char* which, bool flush, std::string& err) {
    if (f == nullptr) return true;
    bool ok = true;
    if (flush && std::fflush(f) != 0) ok = false;
    if (std::fclose(f) != 0) ok = false;
    f = nullptr;
    if (!ok) {
        failed_ = true;
        if (error_id_.empty()) error_id_ = which;
        err = std::string("tier-b export: ") + which;
        return false;
    }
    return true;
}

inline bool TierBExport::finish(bool complete, std::string& err) {
    if (!enabled()) return true;
    bool ok = close_one(rows_, "rows close", true, err);   // rows must be durable before success footer
    if (truncated_ && std::fprintf(manifest_, "# truncation 1\n") < 0) ok = fail("tier-b export: manifest io", err);
    if (failed_) {
        const char* id = error_id_.empty() ? "unknown" : error_id_.c_str();
        if (std::fprintf(manifest_, "# export_error %s\n", id) < 0) ok = fail("tier-b export: manifest io", err);
    }
    if (complete && !failed_ && ok) {
        if (std::fprintf(manifest_, "# complete 1\n") < 0) ok = fail("tier-b export: manifest io", err);
    }
    if (!close_one(manifest_, "manifest close", true, err)) ok = false;
    return ok && !failed_;
}

}  // namespace strata::program
