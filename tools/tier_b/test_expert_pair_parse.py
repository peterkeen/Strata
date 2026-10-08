"""sta-osh.27: --pair gu,dt parsing regression, offline only.

The real parser body is compiled from src/kernels/tier_b_expert_bench.cpp against the
dedicated expert fake CUDA; no CUDA compiler/runtime/GPU or downloads are used. Invalid
values make the parser exit(2) by design, so every case runs in its own process and the
test asserts on exit status and the parsed pairs the probe prints.
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

# The seven manifest pairs the benchmark must still default to when --pair is absent.
MANIFEST = [(18, 20), (18, 42), (22, 20), (22, 42), (21, 20), (21, 42), (23, 20)]

HARNESS = r'''
#define EXPERT_CLEANUP_PROBE_MAIN unused_expert_cleanup_entry
#include "tools/tier_b/fixtures/expert_cleanup/expert_cleanup_probe.cpp"

// Parse exactly the argv the test passes and print what the real parser produced.
// No CUDA resource is acquired here: only the command line is exercised.
int main(int argc, char** argv) {
    const Cli c = parse(argc, argv);
    std::printf("PAIRS %zu\n", c.pairs.size());
    for (const auto& p : c.pairs) std::printf("PAIR %d %d\n", p.first, p.second);
    // Defaults must survive the parse unchanged.
    std::printf("DEFAULTS groups=%d tokens=%d slots=%d warmup=%d samples=%d max_slots=%d max_bytes=%zu\n",
                c.groups, c.tokens, c.slots, c.warmup, c.samples, c.max_slots, c.max_bytes);
    return 0;
}
'''


@unittest.skipUnless(shutil.which("g++"), "offline expert pair-parse probes require g++")
class ExpertPairParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="expert_pair_")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.exe = Path(cls.tmp.name) / "expert_pair_probe"
        cpp = Path(cls.tmp.name) / "expert_pair_probe.cpp"
        cpp.write_text(HARNESS)
        command = ["g++", "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror", "-I", str(FIXTURES),
                   "-I", str(ROOT / "include"), "-I", str(ROOT), str(cpp), "-o", str(cls.exe)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise AssertionError(f"pair-parse compile exit={result.returncode}:\n{result.stderr}")

    def parse(self, *args):
        return subprocess.run([str(self.exe), *args], capture_output=True, text=True, timeout=60)

    def pairs(self, *args):
        """Parse successfully and return the pairs the real parser built."""
        result = self.parse(*args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        count = int(re.search(r"^PAIRS (\d+)$", result.stdout, re.M).group(1))
        pairs = [(int(a), int(b)) for a, b in re.findall(r"^PAIR (\d+) (\d+)$", result.stdout, re.M)]
        self.assertEqual(len(pairs), count)
        return pairs

    def rejects(self, *args):
        """Malformed input must exit 2 with a message and never yield a pair."""
        result = self.parse(*args)
        self.assertEqual(result.returncode, 2, f"accepted {args!r}:\n{result.stdout}{result.stderr}")
        self.assertNotIn("PAIR ", result.stdout)
        self.assertTrue(result.stderr.strip(), f"no diagnostic for {args!r}")
        return result

    # ---------------------------------------------------------------- valid input
    def test_every_manifest_pair_is_accepted(self):
        for gu, dt in MANIFEST:
            self.assertEqual(self.pairs("--pair", f"{gu},{dt}"), [(gu, dt)],
                             f"valid manifest pair {gu},{dt} was rejected")

    def test_repeated_pair_flags_accumulate_in_order(self):
        self.assertEqual(self.pairs("--pair", "18,20", "--pair", "21,42", "--pair", "23,20"),
                         [(18, 20), (21, 42), (23, 20)])

    def test_duplicate_and_reversed_pairs_are_preserved_verbatim(self):
        self.assertEqual(self.pairs("--pair", "18,20", "--pair", "18,20"), [(18, 20), (18, 20)])
        self.assertEqual(self.pairs("--pair", "20,18"), [(20, 18)])

    def test_range_bounds_are_accepted(self):
        self.assertEqual(self.pairs("--pair", "1,1"), [(1, 1)])
        self.assertEqual(self.pairs("--pair", "1024,1024"), [(1024, 1024)])
        self.assertEqual(self.pairs("--pair", "1,1024"), [(1, 1024)])

    def test_pair_parsing_leaves_other_defaults_untouched(self):
        result = self.parse("--pair", "18,20")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DEFAULTS groups=8 tokens=2 slots=8 warmup=25 samples=30 "
                      "max_slots=64 max_bytes=536870912", result.stdout)

    def test_pair_combines_with_other_options(self):
        result = self.parse("--groups", "4", "--pair", "22,42", "--samples", "3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PAIR 22 42", result.stdout)
        self.assertIn("groups=4", result.stdout)
        self.assertIn("samples=3", result.stdout)

    # ---------------------------------------------------------------- default behaviour
    def test_no_pair_flag_leaves_the_list_empty_for_the_seven_manifest_defaults(self):
        self.assertEqual(self.pairs(), [])
        self.assertEqual(self.pairs("--groups", "2"), [])

    def test_default_run_still_reports_all_seven_manifest_pairs(self):
        # The unchanged default: the benchmark walks its seven built-in pairs when none is given.
        source = (ROOT / "src/kernels/tier_b_expert_bench.cpp").read_text()
        literal = re.search(r"const std::pair<int, int> kPairs\[\] = \{(.+?)\};", source, re.S).group(1)
        built_in = [(int(a), int(b)) for a, b in re.findall(r"\{(\d+), (\d+)\}", literal)]
        self.assertEqual(built_in, MANIFEST)

    # ---------------------------------------------------------------- malformed input
    def test_missing_comma_is_rejected(self):
        for bad in ("18", "1820", "18 20", "18;20", "18.20", "18-20", "18:20"):
            result = self.rejects("--pair", bad)
            if "," not in bad:
                self.assertIn("--pair needs gu,dt", result.stderr)

    def test_missing_value_after_flag_is_rejected(self):
        result = self.rejects("--pair")
        self.assertIn("--pair needs a value", result.stderr)

    def test_empty_and_whitespace_sides_are_rejected(self):
        for bad in ("", ",", "18,", ",20", " ,20", "18, ", "  ,  "):
            self.rejects("--pair", bad)

    def test_non_numeric_sides_are_rejected(self):
        for bad in ("a,20", "18,b", "x,y", "0x12,20", "18,0x14", "nan,20", "18,inf", "--pair,20"):
            self.rejects("--pair", bad)

    def test_out_of_range_sides_are_rejected(self):
        for bad in ("0,20", "18,0", "1025,20", "18,1025", "-1,20", "18,-1",
                    "99999999999999999999,20", "18,99999999999999999999"):
            self.rejects("--pair", bad)

    def test_trailing_garbage_and_extra_commas_are_rejected(self):
        for bad in ("18,20,", "18,20,21", "18,,20", ",18,20", "18,20x", "18x,20"):
            self.rejects("--pair", bad)

    def test_leading_sign_and_space_are_accepted_exactly_as_other_numeric_flags(self):
        # Pre-existing shared `num`/strtoll behaviour for every numeric option, not --pair-specific:
        # a leading '+' or spaces are consumed. Documented here so the comma fix is not blamed for it.
        self.assertEqual(self.pairs("--pair", "+18,20"), [(18, 20)])
        self.assertEqual(self.pairs("--pair", "18,+20"), [(18, 20)])
        self.assertEqual(self.pairs("--pair", " 18,20"), [(18, 20)])
        result = self.parse("--groups", "+8", "--slots", "+4")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("groups=8", result.stdout)
        self.assertIn("slots=4", result.stdout)

    def test_rejection_happens_even_when_a_later_pair_is_valid(self):
        self.rejects("--pair", "18,20", "--pair", "bad")
        self.rejects("--pair", "bad", "--pair", "18,20")


if __name__ == "__main__":
    unittest.main()
