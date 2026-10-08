"""sta-osh.23: complete explicit expert-buffer accounting, offline only.

The entire benchmark is compiled against the dedicated expert fake CUDA. The
oracle uses checked-in ggml block declarations and the actual scratch-size helper
body read from iq_kernels.cu; no CUDA compiler/runtime/GPU or downloads are used.
Synthetic weight row sizes keep allocations small. Graph/driver allocations are
opaque, unmeasured, and deliberately outside the explicit cudaMalloc-buffer cap.

The `real` phase additionally plans the documented default geometry against the
production `iq_row_bytes` and `native_expert_layout` bodies read out of
iq_kernels.cu, so the stated auxiliary+guard overhead and the real manifest
weight arenas are checked against real exporter/kernel sizes, not the fake's
synthetic rows.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tools/tier_b/fixtures/expert_cleanup"

HARNESS = r'''
#define GGML_COMMON_DECL_CPP
#include "third_party/ggml/ggml-common.h"
#define EXPERT_CLEANUP_PROBE_MAIN unused_expert_cleanup_entry
#include "tools/tier_b/fixtures/expert_cleanup/expert_cleanup_probe.cpp"
static_assert(sizeof(block_q8_1) == 36, "Q8_1 allocation ABI changed");

// SOURCE_SCRATCH_HELPER

namespace strata::kernels {
// Production row-size and blob-layout bodies, renamed; the fake's synthetic rows stay for the
// lifecycle sweeps while the real manifest geometry is planned from real block sizes.
// SOURCE_REAL_GEOMETRY
}

namespace {
std::vector<size_t> oracle(const DevicePlan& p) {
    const size_t entries = static_cast<size_t>(p.groups) * p.tokens;
    const size_t h = static_cast<size_t>(p.L.n_embd);
    const size_t stride = (p.L.bytes + 255) / 256 * 256;
    const size_t arena = static_cast<size_t>(p.slots) * p.groups * stride;
    // Independent reproduction of the audit's ten-buffer accounting, each with its own guard.
    return {arena + 256, static_cast<size_t>(p.slots) * p.groups * sizeof(unsigned long long) + 256,
            (static_cast<size_t>(p.groups) + 1) * sizeof(int32_t) + 256, sizeof(int32_t) + 256,
            entries * sizeof(int32_t) + 256, entries * sizeof(int32_t) + 256,
            8 * h * sizeof(float) + 256, 8 * (h / 32) * sizeof(block_q8_1) + 256,
            source_scratch_bytes(static_cast<int64_t>(entries), p.L.n_ff) + 256,
            entries * h * sizeof(float) + 256};
}
void allocated_exactly(const DevicePlan& p, size_t cap) {
    expert_fake::reset(); g_fail = 0;
    {
        Guard guard; Stream s; ck(cudaStreamCreate(s.put()), "budget stream");
        std::mt19937 rng(2715); Bench b(p, rng, s, cap);
        const auto expected = oracle(p);
        require(expert_fake::allocation_sizes == expected, "actual cudaMalloc requests differ from oracle");
        const size_t sum = std::accumulate(expected.begin(), expected.end(), size_t(0));
        require(sum == p.total_bytes && expert_fake::buffer_peak_bytes == sum,
                "planned total differs from actual live bytes");
        require(sum <= cap, "accepted allocation exceeds cap");
    }
    require(expert_fake::empty() && expert_fake::buffer_live_bytes == 0 && g_fail == 0, "budget probe leaked");
}
DevicePlan defaults() {
    Cli c;
    require(c.max_bytes == (512ull << 20), "default cap changed");
    return device_plan(k::native_expert_layout(18, 20, 2560, 640), c.groups, c.tokens, c.slots);
}
int default_probe() {
    const auto p = defaults(); const auto sizes = oracle(p);
    require(p.total_bytes - p.arena_bytes == 406440, "default auxiliary overhead is wrong");
    require(p.scratch_bytes == 134400, "default scratch is wrong");
    allocated_exactly(p, 512ull << 20);
    std::printf("BUDGET default overhead=%zu scratch=%zu arena=%zu total=%zu requests=10\n",
                p.total_bytes - p.arena_bytes, p.scratch_bytes, p.arena_bytes, p.total_bytes);
    for (size_t i = 0; i < sizes.size(); ++i) std::printf("BUFFER %zu bytes=%zu\n", i, sizes[i]);
    return 0;
}
int boundary_probe() {
    const auto p = defaults();
    expert_fake::reset(); g_fail = 0; bool rejected = false;
    {
        Guard guard; Stream s; ck(cudaStreamCreate(s.put()), "boundary stream");
        std::mt19937 rng(2715);
        require(p.arena_bytes <= p.total_bytes - 1, "weight arena does not fit rejection cap");
        try { Bench b(p, rng, s, p.total_bytes - 1); }
        catch (const std::length_error&) { rejected = true; }
    }
    require(rejected && expert_fake::allocation_sizes.empty(), "over-cap constructor reached cudaMalloc");
    require(expert_fake::empty(), "rejected constructor leaked");
    allocated_exactly(p, p.total_bytes); allocated_exactly(p, p.total_bytes + 1);
    expert_fake::reset(); rejected = false;
    std::mt19937 rng(2715);
    try { Bench b(18, 20, 2560, 640, 64, 8, 1024, rng, nullptr); }
    catch (const std::length_error&) { rejected = true; }
    require(rejected && expert_fake::allocation_sizes.empty(), "default 512 MiB cap not enforced before cudaMalloc");
    std::printf("BUDGET boundary total=%zu cap_below_malloc=0 exact_fit=accepted cap_above=accepted default_cap_reject_malloc=0\n", p.total_bytes);
    return 0;
}
// Real manifest geometry: the documented default must carry exactly the audited auxiliary+guard
// overhead and still fit the default cap at the real IQ3_S-file 2560x640 blob sizes.
int real_probe() {
    Cli c;
    require(c.groups == 8 && c.tokens == 2 && c.slots == 8 && c.max_bytes == (512ull << 20),
            "documented default geometry changed");
    require(k::source_row_bytes(18, 2560) == 980 && k::source_row_bytes(20, 640) == 360 &&
            k::source_row_bytes(23, 2560) == 1360 && k::source_row_bytes(42, 640) == 180,
            "production row bytes differ from the checked-in block sizes");
    int checked = 0;
    for (auto pair : {std::pair<int, int>{18, 20}, {18, 42}, {22, 20}, {22, 42}, {21, 20}, {21, 42}, {23, 20}}) {
        const auto L = k::source_expert_layout(pair.first, pair.second, 2560, 640);
        require(L.bytes == 2 * static_cast<size_t>(640) * k::source_row_bytes(pair.first, 2560) +
                           static_cast<size_t>(2560) * k::source_row_bytes(pair.second, 640),
                "real blob layout is not gate|up|down");
        const auto p = device_plan(L, c.groups, c.tokens, c.slots);
        require(p.total_bytes - p.arena_bytes == 406440, "real auxiliary+guard overhead is not 406440");
        require(p.scratch_bytes == 134400 && p.n_ent == 16, "real default scratch/entries changed");
        require(p.arena_bytes == static_cast<size_t>(c.slots) * c.groups * ((L.bytes + 255) / 256 * 256),
                "real weight arena differs from slots x groups x aligned blob");
        require(p.total_bytes <= c.max_bytes, "real default geometry no longer fits the default cap");
        std::printf("REAL %d/%d blob=%zu arena=%zu overhead=%zu total=%zu\n", pair.first, pair.second,
                    L.bytes, p.arena_bytes, p.total_bytes - p.arena_bytes, p.total_bytes);
        ++checked;
    }
    std::printf("BUDGET real pairs=%d overhead=406440 fits_default_cap=1\n", checked);
    return 0;
}
int shapes_probe() {
    int shapes = 0;
    for (auto pair : {std::pair<int, int>{18, 20}, {18, 42}, {22, 20}, {22, 42}, {21, 20}, {21, 42}, {23, 20}}) {
        for (auto shape : {std::vector<int>{1, 1, 1}, {1, 8, 3}, {8, 2, 8}, {11, 3, 2}, {64, 8, 4}}) {
            const auto p = device_plan(k::native_expert_layout(pair.first, pair.second, 256, 32), shape[0], shape[1], shape[2]);
            allocated_exactly(p, p.total_bytes); ++shapes;
        }
    }
    // Largest legal CLI counts are planned without allocating a huge weight arena.
    const auto p = device_plan(k::native_expert_layout(18, 20, 2560, 640), 64, 8, 1024);
    require(p.n_ent == 512 && p.ptr_count == 65536 && p.total_bytes > (512ull << 20), "maximum shape plan is wrong");
    std::printf("BUDGET shapes tested=%d max_cli_plan=checked\n", shapes); return 0;
}
int cli_probe(const std::string& mode) {
    const auto p = defaults();
    size_t cap = p.total_bytes;
    int initial_slots = 8, max_slots = 8, l2 = 1;
    bool rejected = false;
    DevicePlan chosen = p;
    if (mode == "cli_below") { --cap; rejected = true; }
    else if (mode == "cli_above") ++cap;
    else if (mode == "cli_l2_equal") { l2 = static_cast<int>(p.arena_bytes); rejected = true; }
    else if (mode == "cli_l2_grow") {
        initial_slots = 1; max_slots = 2; l2 = static_cast<int>(p.slot_bytes);
        chosen = device_plan(p.L, p.groups, p.tokens, 2); cap = chosen.total_bytes;
    }
    expert_fake::reset(); g_fail = 0;
    // Narrow fake support to one manifest pair, without changing or exercising the separately-owned --pair parser.
    expert_fake::only_manifest_pair = true; expert_fake::device_l2 = l2;
    std::vector<std::string> args = {"expert", "--warmup", "1", "--samples", "1", "--slots", std::to_string(initial_slots),
        "--max-slots", std::to_string(max_slots), "--max-bytes", std::to_string(cap)};
    std::vector<char*> argv; for (auto& arg : args) argv.push_back(arg.data());
    const int rc = expert_benchmark_main(static_cast<int>(argv.size()), argv.data());
    // The six unsupported fixture pairs cause exit 1 even when the targeted pair times successfully.
    require(rc == 1, "CLI did not preserve unsupported-pair failure status");
    require(expert_fake::empty() && expert_fake::ns && expert_fake::old, "CLI unwind failed");
    if (rejected) {
        require(expert_fake::allocation_sizes.empty(), "rejected CLI case reached cudaMalloc");
    } else {
        require(expert_fake::allocation_sizes == oracle(chosen), "CLI allocation total differs from plan");
        require(expert_fake::buffer_peak_bytes == chosen.total_bytes, "CLI peak bytes are wrong");
    }
    std::printf("BUDGET %s cap=%zu malloc=%zu target_timed=%d\n", mode.c_str(), cap,
                expert_fake::allocation_sizes.size(), rejected ? 0 : 1);
    return 0;
}
template<typename F> void overflow(F fn) {
    bool caught = false; try { fn(); } catch (const std::overflow_error&) { caught = true; }
    require(caught, "overflow was not rejected");
}
int overflow_probe() {
    expert_fake::reset(); const size_t max = std::numeric_limits<size_t>::max();
    require(add_bytes(max, 0) == max && mul_bytes(max, 1) == max && mul_bytes(max, 0) == 0,
            "checked arithmetic rejects valid edges");
    require(buffer_bytes<uint8_t>(max - 256) == max && align256(max - 255) == max - 255, "valid guard/alignment edge rejected");
    overflow([&] { add_bytes(max, 1); }); overflow([&] { mul_bytes(max, 2); });
    overflow([&] { align256(max - 254); }); overflow([&] { buffer_bytes<float>(max / sizeof(float) + 1); });
    overflow([&] { buffer_bytes<uint8_t>(max); }); overflow([&] { auto a = dalloc<uint8_t>(max); });
    auto l = k::native_expert_layout(18, 20, 2560, 640);
    l.bytes = max; overflow([&] { device_plan(l, 1, 1, 1); });
    l.bytes = max - 255; overflow([&] { device_plan(l, 2, 1, 1); });
    l.bytes = max / 4; overflow([&] { device_plan(l, 1, 1, 8); });
    l.bytes = max - 511; overflow([&] { device_plan(l, 1, 1, 1); }); // total addition, not arena/guard overflow
    l = k::native_expert_layout(18, 20, 2560, 640);
    overflow([&] { device_plan(l, std::numeric_limits<int>::max(), 2, 1); });
    l.n_embd = std::numeric_limits<int64_t>::max() - 31;
    overflow([&] { device_plan(l, 1, 1, 1); });
    l.n_embd = 2560; l.n_ff = std::numeric_limits<int64_t>::max() - 31;
    overflow([&] { device_plan(l, 1, 1, 1); });
    require(expert_fake::allocation_sizes.empty() && expert_fake::calls.empty(), "overflow reached CUDA");
    std::printf("BUDGET overflow rejected=13 cuda_calls=0\n"); return 0;
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "budget phase required"); const std::string mode = argv[1];
        if (mode == "default") return default_probe();
        if (mode == "boundary") return boundary_probe();
        if (mode == "shapes") return shapes_probe();
        if (mode == "real") return real_probe();
        if (mode == "overflow") return overflow_probe();
        return cli_probe(mode);
    } catch (const std::exception& e) { std::fprintf(stderr, "BUDGET FAIL: %s\n", e.what()); return 1; }
}
'''


@unittest.skipUnless(shutil.which("g++"), "offline expert budget probes require g++")
class ExpertBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="expert_budget_")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.exe = Path(cls.tmp.name) / "expert_budget_probe"
        source = (ROOT / "src/kernels/cuda/iq_kernels.cu").read_text()
        match = re.search(r"size_t native_expert_scratch_bytes\(int64_t cap, int64_t n_ff\) \{.*?\n\}", source, re.S)
        if match is None:
            raise AssertionError("production scratch helper not found")
        scratch = match.group().replace("native_expert_scratch_bytes", "source_scratch_bytes", 1)
        rows = re.search(r"size_t iq_row_bytes\(int t, int64_t n\) noexcept \{.*?\n\}", source, re.S)
        layout = re.search(
            r"NativeExpertLayout native_expert_layout\(int gu_type, int d_type, int64_t n_embd, int64_t n_ff\) \{.*?\n\}",
            source, re.S)
        if rows is None or layout is None:
            raise AssertionError("production row-bytes/layout helpers not found")
        geometry = "\n".join([
            rows.group().replace("iq_row_bytes", "source_row_bytes", 1),
            layout.group().replace("native_expert_layout", "source_expert_layout", 1)
                          .replace("iq_row_bytes", "source_row_bytes"),
        ])
        cpp = Path(cls.tmp.name) / "expert_budget_probe.cpp"
        cpp.write_text(HARNESS.replace("// SOURCE_SCRATCH_HELPER", scratch)
                              .replace("// SOURCE_REAL_GEOMETRY", geometry))
        command = ["g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-I", str(FIXTURES),
                   "-I", str(ROOT / "include"), "-I", str(ROOT), str(cpp), "-o", str(cls.exe)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(f"budget compile exit={result.returncode}:\n{result.stderr}")

    def probe(self, mode):
        result = subprocess.run([str(self.exe), mode], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = next(line for line in result.stdout.splitlines() if line.startswith(f"BUDGET {mode} "))
        print(summary)
        return result

    def test_default_auxiliary_overhead_recomputed_from_actual_requests(self):
        result = self.probe("default")
        self.assertIn("overhead=406440 scratch=134400", result.stdout)
        sizes = [int(n) for n in re.findall(r"^BUFFER \d+ bytes=(\d+)$", result.stdout, re.M)]
        # Audit reproduction: includes the arena guard plus every other buffer and its guard.
        self.assertEqual(len(sizes), 10)
        self.assertEqual(256 + sum(sizes[1:]), 406440)

    def test_exact_fit_and_one_byte_below_above_constructor_boundaries(self):
        self.probe("boundary")

    def test_all_manifest_pairs_and_representative_shapes_match_every_request(self):
        self.probe("shapes")

    def test_real_manifest_geometry_overhead_is_406440_and_fits_default_cap(self):
        result = self.probe("real")
        self.assertIn("pairs=7 overhead=406440 fits_default_cap=1", result.stdout)
        rows = re.findall(r"^REAL (\d+)/(\d+) blob=(\d+) arena=(\d+) overhead=(\d+) total=(\d+)$",
                          result.stdout, re.M)
        self.assertEqual(len(rows), 7)
        for row in rows:
            print("REAL %s/%s blob=%s arena=%s overhead=%s total=%s" % row)
        # Real IQ3_XXS/IQ4_NL 2560x640: 2*640*980 + 2560*360 bytes per blob, 8 slots x 8 groups.
        self.assertEqual(rows[0][2:], ("2176000", "139264000", "406440", "139670440"))
        for _, _, _, arena, overhead, total in rows:
            self.assertEqual(int(overhead), 406440)
            self.assertEqual(int(arena) + 406440, int(total))
            self.assertLessEqual(int(total), 512 << 20)

    def test_weight_arena_fits_but_total_fails_before_any_cli_malloc(self):
        result = self.probe("cli_below")
        self.assertIn("rejected before device-buffer allocation", result.stdout)
        self.assertIn("malloc=0 target_timed=0", result.stdout)
        self.assertIn("7 parity failures, 0 timed pairs", result.stdout)

    def test_cli_exact_fit_admitted(self):
        result = self.probe("cli_fit")
        self.assertIn("malloc=10 target_timed=1", result.stdout)
        self.assertIn("6 parity failures, 1 timed pairs", result.stdout)
        self.assertIn("explicit device buffers", result.stdout)
        self.assertIn("sizes are not measured (not a total VRAM-use cap)", result.stdout)

    def test_cli_one_byte_extra_cap_admitted(self):
        self.probe("cli_above")

    def test_weight_working_set_equal_to_l2_rejected_without_malloc(self):
        result = self.probe("cli_l2_equal")
        self.assertIn("<= L2", result.stdout)
        self.assertIn("malloc=0 target_timed=0", result.stdout)

    def test_l2_auto_bump_counts_additional_weights_and_pointers(self):
        result = self.probe("cli_l2_grow")
        self.assertIn("2 slots x 8 groups", result.stdout)
        self.assertIn("malloc=10 target_timed=1", result.stdout)

    def test_checked_arithmetic_edges_and_overflow_never_reach_cuda(self):
        self.probe("overflow")


if __name__ == "__main__":
    unittest.main()
