"""sta-osh.30: actual IQ host failure policy, compiled without CUDA or a GPU.

Extracts the real TLS variable, check/argument helpers, policy accessors, guards,
type predicate and .22 SelectorGuard; uses the real API's IqHostErrorScope.
Default failures exit 1 in isolated subprocesses: that is fail-closed evidence,
NOT unwind. Only explicitly opted-in failures claim exception/guard unwind.
Kernel bodies/launches are NOT compiled; no GPU arithmetic or serving claim.
Reproduce: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tools.tier_b.test_iq_launch_errors -v
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/kernels/cuda/iq_kernels.cu"
CALLER = ROOT / "src/kernels/iq_multi_parity.cpp"
API = ROOT / "include/strata/kernels/iq_kernels.hpp"
ARG_HELPER = "template<typename... Args>\n[[noreturn]] void host_argument_error("
ARGUMENT_CASES = (
    ("mmvq_type", {"t": 999}, "iq_mmvq: type 999 is not supported"),
    ("f16", {"n": 257}, "iq_dequant_f16: bad arguments"),
    ("f32", {"t": -1}, "iq_dequant_f32: bad arguments"),
    ("embed", {"n": 257}, "iq_embed_rows: bad arguments"),
    ("gu_f16", {"t": -1}, "iq_dequant_gu_f16: type -1 / 256"),
    ("grouped", {"n": 33}, "native_expert_grouped: n_ff 33"),
    ("gu_type", {"t": -1}, "native_expert_grouped: gate/up type -1"),
    ("down_type", {"t": 999}, "native_expert_grouped: down type 999"),
)


def braced(text: str, marker: str) -> str:
    """Extract a uniquely identified definition, retaining its original body."""
    if text.count(marker) != 1:
        raise AssertionError(f"expected exactly one definition: {marker}")
    start = text.index(marker)
    opening = text.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    return text[start:end]


def prefix(text: str, marker: str, stop: str) -> str:
    definition = braced(text, marker)
    body = definition[definition.index("{") + 1:]
    if body.count(stop) != 1:
        raise AssertionError(f"missing/ambiguous guard boundary: {marker}: {stop}")
    return body[:body.index(stop)]


def rejection(text: str, message: str) -> str:
    matches = re.findall(r'host_argument_error\("' + re.escape(message) + r'[^;]*;', text)
    if len(matches) != 1:
        raise AssertionError(f"missing/ambiguous argument rejection: {message}")
    return matches[0]


PREAMBLE = r'''
#include "strata/kernels/iq_kernels.hpp"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>

static_assert(noexcept(strata::kernels::iq_host_errors_throw()));
static_assert(noexcept(strata::kernels::iq_set_host_errors_throw(false)));
static_assert(std::is_nothrow_default_constructible_v<strata::kernels::IqHostErrorScope>);
static_assert(std::is_nothrow_destructible_v<strata::kernels::IqHostErrorScope>);
static_assert(!std::is_copy_constructible_v<strata::kernels::IqHostErrorScope>);
static_assert(!std::is_move_constructible_v<strata::kernels::IqHostErrorScope>);
static_assert(!std::is_copy_assignable_v<strata::kernels::IqHostErrorScope>);
static_assert(!std::is_move_assignable_v<strata::kernels::IqHostErrorScope>);
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
thread_local int pending_error = 0, last_calls = 0, string_calls = 0;
cudaError_t cudaGetLastError() {
    ++last_calls;
    const int result = pending_error;
    pending_error = cudaSuccess;
    return result;
}
const char* cudaGetErrorString(cudaError_t error) {
    ++string_calls;
    if (error == 1) return "invalid configuration argument";
    if (error == 2) return "out of resources";
    return "unknown CUDA error";
}
namespace strata::kernels {
namespace {
bool g_old_kernels = false, g_no_sub16_gu = false;
}
namespace tier_b {
bool tracing = false;
bool enabled() { return tracing; }
template <class... Args> void emit(Args...) {}
}
'''

PROBE = r'''
int live = 0, released = 0;
bool expecting_default_exit = false;
std::array<int, 3> order{};
struct Resource {
    int id;
    explicit Resource(int i) : id(i) { ++live; }
    ~Resource() noexcept { --live; order[released++] = id; }
};
void default_exit_report() {
    if (expecting_default_exit)
        std::printf("DEFAULT_EXIT last=%d strings=%d live=%d released=%d policy=%d\n",
                    last_calls, string_calls, live, released, k::iq_host_errors_throw());
}
void require(bool value) { if (!value) throw std::logic_error("policy probe invariant"); }

void nested_policy(bool prior) {
    {
        k::IqHostErrorScope outer;
        require(k::iq_host_errors_throw());
        { k::IqHostErrorScope inner; require(k::iq_host_errors_throw()); }
        require(k::iq_host_errors_throw());
        bool caught = false;
        try {
            SelectorGuard inner_caller;
            pending_error = 1;
            k::check("nested caller");
        } catch (const std::runtime_error&) { caught = true; }
        require(caught && k::iq_host_errors_throw());
        k::iq_set_host_errors_throw(false);
        caught = false;
        try {
            k::IqHostErrorScope inner;
            require(k::iq_host_errors_throw());
            pending_error = 2;
            k::check("nested prior false");
        } catch (const std::runtime_error&) { caught = true; }
        require(caught && !k::iq_host_errors_throw());
        k::iq_set_host_errors_throw(true);
    }
    require(k::iq_host_errors_throw() == prior);
    {
        SelectorGuard caller;
        require(k::iq_host_errors_throw());
        { SelectorGuard inner; require(k::iq_host_errors_throw()); }
        require(k::iq_host_errors_throw());
    }
    require(k::iq_host_errors_throw() == prior);
}

void thread_policy(bool prior) {
    std::atomic<int> stage{0};
    bool worker_ok = false;
    {
        k::IqHostErrorScope main_scope;
        std::thread worker([&] {
            bool ok = !k::iq_host_errors_throw(); // starts false even while main is opted in
            {
                k::IqHostErrorScope own_scope;
                ok = ok && k::iq_host_errors_throw();
                stage.store(1);
                while (stage.load() == 1) std::this_thread::yield();
                ok = ok && k::iq_host_errors_throw(); // main's toggle cannot affect this thread
                try { pending_error = 1; k::check("worker checked launch"); ok = false; }
                catch (const std::runtime_error&) {}
            }
            worker_ok = ok && !k::iq_host_errors_throw();
        });
        while (stage.load() == 0) std::this_thread::yield();
        const bool main_still_enabled = k::iq_host_errors_throw();
        k::iq_set_host_errors_throw(false);
        stage.store(2);
        worker.join();
        require(main_still_enabled && worker_ok && !k::iq_host_errors_throw());
    }
    require(k::iq_host_errors_throw() == prior);
}

int main(int argc, char** argv) {
    if (argc != 12) return 90;
    const std::string action = argv[1];
    const char* label = argv[2];
    pending_error = std::stoi(argv[3]);
    const int t = std::stoi(argv[4]);
    const int64_t n = std::stoll(argv[5]), count = std::stoll(argv[6]);
    const bool prior_old = std::stoi(argv[7]), prior_layout = std::stoi(argv[8]);
    k::tier_b::tracing = std::stoi(argv[9]);
    const int mode = std::stoi(argv[10]); // 0 unmodified caller, 1 actual .22 guard, 2 API scope
    const bool prior_policy = std::stoi(argv[11]);
    if (k::iq_host_errors_throw()) return 96; // every fresh process must default false
    k::iq_set_host_errors_throw(prior_policy);
    if (action == "nested" || action == "threads") {
        try {
            if (action == "nested") nested_policy(prior_policy); else thread_policy(prior_policy);
        } catch (...) { return 97; }
        std::cout << "POLICY " << action << " restored=1 prior=" << prior_policy << '\n';
        return 0;
    }
    k::iq_set_old_kernels(prior_old);
    k::native_grouped_set_no_sub16_gu(prior_layout);
    if (mode == 0 || action == "thread_launch") {
        expecting_default_exit = true;
        if (std::atexit(default_exit_report)) return 98;
    }
    int result = 0;
    bool reached = false;
    std::string kind = "none", message;
    try {
        Resource first(1);
        std::optional<SelectorGuard> caller_guard;
        std::optional<k::IqHostErrorScope> api_scope;
        if (mode == 1) {
            caller_guard.emplace();
            k::iq_set_old_kernels(!prior_old);
            k::native_grouped_set_no_sub16_gu(!prior_layout);
        } else if (mode == 2) api_scope.emplace();
        Resource second(2), third(3);
        if (action == "launch") k::check(label);
        else if (action == "f16") k::probe_f16(t, n);
        else if (action == "f32") k::probe_f32(t, n);
        else if (action == "embed") k::probe_embed(t, n, count);
        else if (action == "quantize") k::probe_quantize(count, n);
        else if (action == "gu_f16") k::probe_gu_f16(t, n);
        else if (action == "grouped") k::probe_grouped(n, count, t);
        else if (action == "mmvq_type") k::probe_mmvq_type(t);
        else if (action == "gu_type") k::probe_gu_type(t);
        else if (action == "down_type") k::probe_down_type(t);
        else if (action == "format") k::host_argument_error("%s\n", label); // real formats end in \n
        else if (action == "thread_launch") {
            const int error = pending_error;
            std::thread worker([=] { pending_error = error; k::check(label); });
            worker.join();
        } else return 91;
        reached = true;
    } catch (const std::invalid_argument& error) {
        kind = "invalid_argument"; message = error.what(); result = 2;
    } catch (const std::runtime_error& error) {
        kind = "runtime_error"; message = error.what(); result = 2;
    } catch (...) { return 92; }
    expecting_default_exit = false;
    const bool restored = k::iq_old_kernels() == prior_old && k::native_grouped_no_sub16_gu() == prior_layout;
    const bool policy_restored = k::iq_host_errors_throw() == prior_policy;
    const int before_last = last_calls, before_strings = string_calls;
    // A subsequent success must not rethrow a consumed launch error, regardless of restored policy.
    if (action == "launch") {
        try { k::check("subsequent success"); } catch (...) { return 93; }
        if (last_calls != before_last + 1 || string_calls != before_strings) return 94;
    }
    std::cout << "IQ_HOST result=" << result << " kind=" << kind << " reached=" << reached
              << " live=" << live << " released=" << released << " restored=" << restored
              << " order=" << order[0] << order[1] << order[2]
              << " last=" << before_last << " strings=" << before_strings
              << " policy_restored=" << policy_restored << " policy=" << k::iq_host_errors_throw() << '\n';
    std::cout << "MESSAGE " << message << '\n';
    if (live != 0 || released != 3 || !restored || !policy_restored || order != std::array<int, 3>{3, 2, 1}) return 95;
    return 0;
}
'''


def probe_source(source: str) -> str:
    pieces = [PREAMBLE]
    tls = re.findall(r"^thread_local bool g_iq_host_errors_throw = false;", source, re.M)
    if len(tls) != 1:
        raise AssertionError("missing actual default-false TLS policy declaration")
    pieces.extend(tls)
    for marker in (ARG_HELPER, "void check(", "bool iq_host_errors_throw(", "void iq_set_host_errors_throw(",
                   "bool is_iq(", "bool embed_type_supported(", "void iq_set_old_kernels(", "bool iq_old_kernels(",
                   "void native_grouped_set_no_sub16_gu(", "bool native_grouped_no_sub16_gu("):
        pieces.append(braced(source, marker))
    guards = (
        ("void probe_f16(int t, int64_t n)", "void iq_dequant_f16(", "    dequant_flat_kernel"),
        ("void probe_f32(int t, int64_t n)", "void iq_dequant_f32(", "    dequant_flat_kernel"),
        ("void probe_embed(int t, int64_t n_embd, int64_t n_tok)", "void iq_embed_rows(", "    embed_rows_kernel"),
        ("void probe_gu_f16(int t, int64_t n_embd)", "void iq_dequant_gu_f16(", "    const int64_t per_row"),
        ("void probe_quantize(int64_t n_rows, int64_t n_cols)", "void quantize_q8_1_rows(", "    quantize_q8_1_kernel"),
    )
    for signature, marker, stop in guards:
        pieces.append(signature + " {" + prefix(source, marker, stop) + "\n}")
    pieces.append("void probe_grouped(int64_t n_ff, int64_t cap_groups, int64_t cap_entries) {\n"
                  "NativeExpertLayout L; L.n_ff = n_ff;\nconst int64_t grid_groups = 0;\nvoid* stream = nullptr;\n" +
                  prefix(source, "void native_expert_grouped(", "    cudaStream_t s =") + "\n}")
    # Default-switch rejection statements are compiled verbatim. Kernel dispatch itself is not compiled.
    pieces.append("void probe_mmvq_type(int t) {\n" + rejection(source, "iq_mmvq: type ") + "\n}")
    for name, field, message in (("gu", "gu_type", "native_expert_grouped: gate/up type "),
                                 ("down", "d_type", "native_expert_grouped: down type ")):
        pieces.append(f"void probe_{name}_type(int t) {{\nNativeExpertLayout L; L.{field} = t;\n" +
                      rejection(source, message) + "\n}")
    pieces.append("} // namespace strata::kernels\nnamespace k = strata::kernels;\n")
    pieces.append(braced(CALLER.read_text(), "struct SelectorGuard {") + ";\n")
    pieces.append(PROBE)
    return "\n".join(pieces)


class TestIqLaunchSource(unittest.TestCase):
    def test_process_exits_are_confined_to_policy_checked_failure_helpers(self):
        source = SOURCE.read_text()
        check = braced(source, "void check(")
        arguments = braced(source, ARG_HELPER)
        self.assertEqual(source.count("std::exit(1);"), 2)
        self.assertIn("if (g_iq_host_errors_throw) throw std::runtime_error", check)
        self.assertIn("if (!g_iq_host_errors_throw)", arguments)
        self.assertIn("std::exit(1);", check)
        self.assertIn("std::exit(1);", arguments)
        rest = source.replace(check, "").replace(arguments, "")
        self.assertNotRegex(rest, r"\b(?:exit|_Exit|_exit|quick_exit|abort|terminate|assert)\s*\(")
        # The two policy-checked helpers hold every throw statement in the file.  Match the keyword
        # itself, not the policy identifiers (g_iq_host_errors_throw, iq_set_host_errors_throw).
        self.assertEqual(len(re.findall(r"\bthrow\b", source)), 3)
        self.assertEqual(len(re.findall(r"\bthrow\b", check)), 1)
        self.assertEqual(len(re.findall(r"\bthrow\b", arguments)), 2)
        self.assertNotRegex(rest, r"\bthrow\b")
        self.assertEqual(rest.count("host_argument_error("), 8)

    def test_policy_api_is_noexcept_and_only_parity_explicitly_opts_in(self):
        api = API.read_text()
        self.assertIn("bool iq_host_errors_throw() noexcept;", api)
        self.assertIn("void iq_set_host_errors_throw(bool enabled) noexcept;", api)
        caller = CALLER.read_text()
        self.assertIn("k::IqHostErrorScope errors;", braced(caller, "struct SelectorGuard {"))
        self.assertTrue(braced(caller, "int parity_main(").split("{", 1)[1].lstrip().startswith("SelectorGuard mode;"))
        for name in ("quantize_q8_1_rows", "iq_mmvq", "iq_dequant_f16", "iq_dequant_f32",
                     "iq_embed_rows", "iq_dequant_gu_f16", "native_expert_grouped"):
            self.assertNotIn("IqHostErrorScope", braced(SOURCE.read_text(), "void " + name + "("))


class TestIqLaunchProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            raise AssertionError("a C++ compiler is required for the offline IQ host policy probe")
        cls.source = SOURCE.read_text()
        cls.temp = tempfile.TemporaryDirectory(prefix="iq_launch_errors_")
        cls.addClassCleanup(cls.temp.cleanup)
        path = Path(cls.temp.name)
        (path / "probe.cpp").write_text(probe_source(cls.source))
        cls.executables = []
        for q6 in (False, True):
            exe = path / f"probe_{int(q6)}"
            command = [compiler, "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "include")]
            if q6:
                command.append("-DSTRATA_Q6K_EXPERTS=1")
            command += [str(path / "probe.cpp"), "-o", str(exe)]
            run = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if run.returncode:
                raise AssertionError(f"compile exit={run.returncode}: {command}\n{run.stdout}\n{run.stderr}")
            cls.executables.append(exe)

    def probe(self, action, *, label="probe launch", error=0, t=18, n=256, count=1,
              old=False, layout=False, tracing=False, q6=False, mode=1, policy=False,
              expected="none", message=None):
        run = subprocess.run([str(self.executables[int(q6)]), action, label, str(error), str(t), str(n),
                              str(count), str(int(old)), str(int(layout)), str(int(tracing)), str(mode), str(int(policy))],
                             capture_output=True, text=True, timeout=10)
        if expected == "exit":
            self.assertEqual(run.returncode, 1, f"default failure: {run.stdout}\n{run.stderr}")
            self.assertEqual(run.stderr, message + "\n")
            self.assertNotIn("IQ_HOST", run.stdout)
            launch = action in ("launch", "thread_launch")
            self.assertIn(f"DEFAULT_EXIT last={int(launch)} strings={int(launch)} live=3 released=0 policy=0", run.stdout)
            return run # no destructor/unwind claim for default policy
        self.assertEqual(run.returncode, 0, f"{action}: {run.stdout}\n{run.stderr}")
        if action in ("nested", "threads"):
            self.assertIn(f"POLICY {action} restored=1 prior={int(policy)}", run.stdout)
            return run
        failed = expected != "none"
        self.assertIn(f"result={2 if failed else 0} kind={expected} reached={int(not failed)}", run.stdout)
        self.assertIn("live=0 released=3 restored=1 order=321", run.stdout)
        self.assertIn(f"last={int(action == 'launch')} strings={int(action == 'launch' and failed)}", run.stdout)
        self.assertIn(f"policy_restored=1 policy={int(policy)}", run.stdout)
        self.assertEqual(run.stderr, "")
        if message is not None:
            self.assertEqual(run.stdout.split("MESSAGE ", 1)[1].rstrip("\n"), message)
        return run

    def test_launch_success_default_api_scope_and_caller_guard(self):
        for mode, policy in ((0, False), (1, False), (1, True), (2, False), (2, True)):
            for old in (False, True):
                for layout in (False, True):
                    self.probe("launch", mode=mode, policy=policy, old=old, layout=layout, message="")

    def test_every_launch_context_throws_only_when_scoped(self):
        labels = sorted(set(re.findall(r'\bcheck\("([^"]+)"\)', self.source)))
        self.assertEqual(len(labels), 12)
        for label in labels:
            for error, text in ((1, "invalid configuration argument"), (2, "out of resources"), (999, "unknown CUDA error")):
                for mode, policy in ((1, False), (1, True), (2, False), (2, True)):
                    with self.subTest(label=label, error=error, mode=mode, policy=policy):
                        self.probe("launch", label=label, error=error, mode=mode, policy=policy,
                                   expected="runtime_error", message=f"{label}: {text}")

    def test_default_launch_failures_exit1_with_original_context(self):
        labels = sorted(set(re.findall(r'\bcheck\("([^"]+)"\)', self.source)))
        for label in labels:
            for error, text in ((1, "invalid configuration argument"), (2, "out of resources"), (999, "unknown CUDA error")):
                self.probe("launch", label=label, error=error, mode=0, expected="exit", message=f"{label}: {text}")

    def test_all_eight_argument_failures_default_exit_and_scoped_unwind(self):
        for action, args, message in ARGUMENT_CASES:
            for q6 in (False, True):
                self.probe(action, **args, q6=q6, mode=0, expected="exit", message=message)
                for mode, policy in ((1, False), (1, True), (2, False), (2, True)):
                    self.probe(action, **args, q6=q6, mode=mode, policy=policy,
                               expected="invalid_argument", message=message)

    def test_alignment_and_unsupported_type_guards_throw(self):
        for action, context in (("f16", "iq_dequant_f16"), ("f32", "iq_dequant_f32"),
                                ("embed", "iq_embed_rows"), ("gu_f16", "iq_dequant_gu_f16")):
            for t, n in ((18, 257), (-1, 256), (999, 256), (18, -257)):
                for q6 in (False, True):
                    run = self.probe(action, t=t, n=n, q6=q6, expected="invalid_argument")
                    self.assertIn(context + ":", run.stdout)
        for n in (33, -33):
            self.probe("grouped", n=n, expected="invalid_argument", message=f"native_expert_grouped: n_ff {n}")

    def test_successful_guards_all_supported_types_and_bf16(self):
        types = (2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18, 20, 21, 22, 23, 29, 42)
        for mode in (0, 1):
            for q6 in (False, True):
                for t in types + ((14,) if q6 else ()):
                    for action in ("f16", "f32", "embed", "gu_f16"):
                        self.probe(action, t=t, q6=q6, mode=mode)
                for action in ("f32", "embed"):
                    self.probe(action, t=30, q6=q6, mode=mode)
        for q6 in (False, True):
            for action in ("f16", "gu_f16"):
                self.probe(action, t=30, q6=q6, expected="invalid_argument")
            for action in ("f16", "f32", "embed", "gu_f16"):
                self.probe(action, t=14, q6=q6, expected="none" if q6 else "invalid_argument")
        self.probe("grouped", n=640)

    def test_original_no_work_returns_precede_validation_under_both_policies(self):
        for mode in (0, 1, 2):
            for count in (0, -1):
                for tracing in (False, True):
                    self.probe("embed", t=-1, n=257, count=count, tracing=tracing, mode=mode)
                    self.probe("grouped", n=33, count=count, tracing=tracing, mode=mode)
                    self.probe("grouped", n=33, count=1, t=count, tracing=tracing, mode=mode)
                    self.probe("quantize", count=count, mode=mode)
            self.probe("quantize", n=0, mode=mode)

    def test_default_switch_rejections_keep_context(self):
        for t in (-1, 999, 2147483647):
            for old in (False, True):
                for layout in (False, True):
                    for action, message in (("mmvq_type", f"iq_mmvq: type {t} is not supported"),
                                            ("gu_type", f"native_expert_grouped: gate/up type {t}"),
                                            ("down_type", f"native_expert_grouped: down type {t}")):
                        self.probe(action, t=t, old=old, layout=layout, expected="invalid_argument", message=message)

    def test_nested_guards_restore_prior_policy_on_success_and_exception(self):
        for policy in (False, True):
            self.probe("nested", policy=policy)

    def test_policy_is_thread_local_not_inherited_and_toggles_are_isolated(self):
        for policy in (False, True):
            self.probe("threads", policy=policy)

    def test_unmodified_worker_still_exits_even_when_parent_is_opted_in(self):
        for mode in (1, 2):
            self.probe("thread_launch", error=1, mode=mode, expected="exit",
                       message="probe launch: invalid configuration argument")

    def test_dimension_context_extrema_and_format_bound(self):
        for n in (-9223372036854775807, 9223372036854775807):
            self.probe("gu_f16", n=n, expected="invalid_argument", message=f"iq_dequant_gu_f16: type 18 / {n}")
            self.probe("grouped", n=n, expected="invalid_argument", message=f"native_expert_grouped: n_ff {n}")
        self.probe("format", label="x" * 300, mode=2, expected="runtime_error",
                   message="IQ host argument error formatting failed")
        self.probe("format", label="x" * 300, mode=0, expected="exit", message="x" * 300)


if __name__ == "__main__":
    unittest.main()
