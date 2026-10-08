// Compile the ENTIRE owned benchmark, using its real IQ declarations and an expert-only fake CUDA.
#define main expert_benchmark_main
#include "src/kernels/tier_b_expert_bench.cpp"
#undef main

namespace strata::kernels {
// Small synthetic byte counts keep the offline lifecycle sweep fast; no arithmetic/CUDA proof is claimed.
size_t iq_row_bytes(int t, int64_t n) noexcept { return static_cast<size_t>(n / (t == 20 ? 32 : t == 42 ? 64 : 256)) * 2; }
NativeExpertLayout native_expert_layout(int gu, int dt, int64_t h, int64_t ff) {
    NativeExpertLayout l;
    l.gu_type = gu; l.d_type = dt; l.n_embd = h; l.n_ff = ff;
    l.gu_row = iq_row_bytes(gu, h); l.d_row = iq_row_bytes(dt, ff);
    l.up_off = static_cast<size_t>(ff) * l.gu_row; l.down_off = 2 * l.up_off;
    l.bytes = l.down_off + static_cast<size_t>(h) * l.d_row;
    if (expert_fake::bad_layout) ++l.bytes;
    return l;
}
bool native_expert_supported(int gu, int dt, int64_t, int64_t) noexcept {
    return !expert_fake::only_manifest_pair || (gu == 18 && dt == 20);
}
size_t native_expert_scratch_bytes(int64_t n, int64_t ff) {
    const size_t f = static_cast<size_t>(n) * static_cast<size_t>(ff) * sizeof(float);
    const size_t q = static_cast<size_t>(n) * static_cast<size_t>(ff / 32) * 36;
    return 3 * ((f + 255) & ~size_t(255)) + ((q + 255) & ~size_t(255));
}
void quantize_q8_1_rows(const float*, int64_t, int64_t, void*, void*) { expert_fake::kernel_step("quantize"); }
void native_expert_grouped(const NativeExpertLayout& l, const unsigned long long*, const int32_t*,
                          const int32_t*, const int32_t*, const int32_t*, int64_t, int64_t n,
                          const void*, void*, float* out, void* stream, int64_t) {
    expert_fake::kernel_step("kernel");
    auto s = static_cast<cudaStream_t>(stream);
    if (s->capturing) { s->out = out; s->n = static_cast<size_t>(n * l.n_embd); }
    else expert_fake::output(out, static_cast<size_t>(n * l.n_embd));
}
void native_grouped_set_no_sub16_gu(bool ns) { expert_fake::ns = ns; }
bool native_grouped_no_sub16_gu() { return expert_fake::ns; }
void iq_set_old_kernels(bool old) { expert_fake::old = old; }
bool iq_old_kernels() { return expert_fake::old; }
// The sta-osh.30 opt-in policy accessors; IqHostErrorScope is header-inline and calls these.
bool iq_host_errors_throw() noexcept { return expert_fake::host_errors_throw; }
void iq_set_host_errors_throw(bool enabled) noexcept { expert_fake::host_errors_throw = enabled; }
}

namespace {
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
bool destructor_only(const std::string& name) {
    return name == "free" || name == "streamDestroy" || name == "eventDestroy" || name == "execDestroy";
}

// Injections start at each phase boundary, so constructor and graph-slot failures have independent evidence.
std::vector<std::string> phase(const std::string& which, int fault, bool partial, int selectors, int throw_kind = 0) {
    expert_fake::reset(); g_fail = 0;
    const bool ns = (selectors & 1) != 0, old = (selectors & 2) != 0;
    expert_fake::ns = ns; expert_fake::old = old;
    bool caught = false;
    std::vector<std::string> trace;
    int policy_steps_at_first_cuda = -1;
    try {
        Guard guard;
        // The scope must already be active before the first CUDA acquisition and before any kernel.
        require(expert_fake::host_errors_throw, "policy not enabled on guard entry");
        policy_steps_at_first_cuda = expert_fake::steps_without_policy;
        Stream s;
        Event e0, e1;
        if (which == "constructor") {
            expert_fake::fail_at = fault; expert_fake::partial_create = partial;
            expert_fake::kernel_throw = throw_kind;
        }
        ck(cudaStreamCreate(s.put()), "probe stream");
        ck(cudaEventCreate(e0.put()), "probe e0"); ck(cudaEventCreate(e1.put()), "probe e1");
        std::mt19937 rng(2715);
        Bench b(18, 20, 256, 32, 2, 1, 3, rng, s);
        if (which != "constructor") {
            expert_fake::calls.clear();
            expert_fake::fail_at = (which == "timing" || which == "graph_check") ? -1 : fault;
            expert_fake::partial_create = partial; expert_fake::kernel_throw = throw_kind;
        }
        if (which == "graphs") {
            auto a = graphs(b, false, s);
            auto r = graphs(b, true, s); // failure must also release the already-built first arm
            trace = expert_fake::calls;
        } else if (which == "capture") {
            capture_us(b, false, s); trace = expert_fake::calls;
        } else if (which == "parity") {
            require(parity(b, s) == 0, "fake parity failed"); trace = expert_fake::calls;
        } else if (which == "timing" || which == "graph_check") {
            auto a = graphs(b, false, s); auto r = graphs(b, true, s);
            // Replay/timing injections must occur AFTER graphs have been created.
            expert_fake::calls.clear(); expert_fake::fail_at = fault;
            if (which == "graph_check") {
                require(graph_check(b, a, false, s) == 0, "fake sub16 graph check failed");
                require(graph_check(b, r, true, s) == 0, "fake reference graph check failed");
            } else {
                warm_direct(b, 2, s); paired_direct(b, 2, s, e0, e1);
                warm_graph(b, a, 2, s); paired_graph(b, a, r, 2, s, e0, e1);
            }
            trace = expert_fake::calls;
        } else {
            trace = expert_fake::calls;
        }
        if (which == "constructor" && fault == -1) require(expert_fake::live.size() == 13, "not all ten Bench allocations owned");
    } catch (...) { caught = true; }
    require(expert_fake::empty(), "leaked allocation/stream/event/graph/exec");
    require(expert_fake::ns == ns && expert_fake::old == old, "selectors not restored");
    require(caught == (fault != -1), "injected failure not propagated");
    // Policy restored to this thread's prior value once the guard has unwound, on success and on error.
    require(expert_fake::host_errors_throw == expert_fake::prior_policy, "policy not restored after phase");
    require(policy_steps_at_first_cuda == 0 && expert_fake::steps_without_policy == 0,
            "a CUDA operation ran before the error policy was enabled");
    require(expert_fake::kernel_errors_without_policy == 0, "kernel error raised without the policy enabled");
    return trace;
}

int sweep(const std::string& which) {
    const auto trace = phase(which, -1, false, 3);
    int injections = 0;
    for (size_t i = 0; i < trace.size(); ++i) {
        if (destructor_only(trace[i])) continue;
        for (int selectors = 0; selectors < 4; ++selectors) {
            phase(which, static_cast<int>(i + 1), false, selectors); ++injections;
            if (trace[i] == "malloc" || trace[i] == "streamCreate" || trace[i] == "eventCreate" ||
                trace[i] == "end" || trace[i] == "instantiate") {
                phase(which, static_cast<int>(i + 1), true, selectors); ++injections;
            }
            if (trace[i] == "kernel" || trace[i] == "quantize") {
                phase(which, static_cast<int>(i + 1), false, selectors, 1); ++injections;
                phase(which, static_cast<int>(i + 1), false, selectors, 2); ++injections;
            }
        }
    }
    std::printf("PROBE %s injections=%d selectors=restored resources=0\n", which.c_str(), injections);
    return 0;
}

int entry_probe() {
    std::vector<std::string> args = {"expert", "--groups", "1", "--tokens", "1", "--slots", "2",
                                   "--max-slots", "2", "--warmup", "1", "--samples", "1"};
    std::vector<char*> argv; for (auto& arg : args) argv.push_back(arg.data());
    int probes = 0;
    for (const char* target : {"none", "props", "streamCreate", "eventCreate", "malloc", "memset", "begin",
                              "end", "instantiate", "kernel", "graphDestroy", "execDestroy", "eventDestroy", "streamDestroy"}) {
        expert_fake::reset(); g_fail = 0; expert_fake::ns = expert_fake::old = true;
        // Find each operation's index with one successful full entry run.
        int rc = expert_benchmark_main(static_cast<int>(argv.size()), argv.data());
        require(rc == 0, "fake benchmark baseline failed");
        auto it = std::find(expert_fake::calls.begin(), expert_fake::calls.end(), target);
        int index = static_cast<int>(it - expert_fake::calls.begin() + 1);
        if (std::string(target) != "none") require(it != expert_fake::calls.end(), "entry target not reached");
        expert_fake::reset(); g_fail = 0; expert_fake::ns = expert_fake::old = true;
        if (std::string(target) != "none") expert_fake::fail_at = index;
        rc = expert_benchmark_main(static_cast<int>(argv.size()), argv.data());
        require(rc == (std::string(target) == "none" ? 0 : 2), "top-level exit is not fail-closed");
        require(expert_fake::empty(), "entry leaked resources");
        require(expert_fake::ns && expert_fake::old, "entry selectors not restored"); ++probes;
    }
    // Test top-level bad_alloc and non-std exceptions rather than only CUDA runtime_error.
    for (int kind : {1, 2}) {
        expert_fake::reset(); g_fail = 0; expert_fake::ns = expert_fake::old = true;
        expert_benchmark_main(static_cast<int>(argv.size()), argv.data());
        auto it = std::find(expert_fake::calls.begin(), expert_fake::calls.end(), "kernel");
        int index = static_cast<int>(it - expert_fake::calls.begin() + 1);
        expert_fake::reset(); g_fail = 0; expert_fake::ns = expert_fake::old = true;
        expert_fake::fail_at = index; expert_fake::kernel_throw = kind;
        require(expert_benchmark_main(static_cast<int>(argv.size()), argv.data()) == 2, "exception escaped main");
        require(expert_fake::empty() && expert_fake::ns && expert_fake::old, "exception unwind failed"); ++probes;
    }
    std::printf("PROBE entry probes=%d selectors=restored resources=0\n", probes); return 0;
}

int cleanup_probe() {
    expert_fake::reset(); g_fail = 0;
    {
        Guard guard; Stream s; ck(cudaStreamCreate(s.put()), "cleanup stream");
        std::mt19937 rng(2715);
        Bench b(18, 20, 256, 32, 2, 1, 3, rng, s);
        expert_fake::all_cleanup_fail = true;
        // All ten frees report an error; no throwing/exit and no early termination of destruction.
    }
    require(expert_fake::empty(), "cleanup errors stopped later releases");
    require(g_fail == 10 && expert_fake::cleanup_attempts == 11, "cleanup errors not all reported");
    std::printf("PROBE cleanup failed_frees=10 attempted_releases=11 resources=0\n"); return 0;
}

int dual_probe() {
    for (bool partial : {false, true}) {
        expert_fake::reset(); g_fail = 0; expert_fake::ns = expert_fake::old = true;
        bool original = false;
        try {
            Guard guard; Stream s; ck(cudaStreamCreate(s.put()), "dual stream");
            std::mt19937 rng(2715); Bench b(18, 20, 256, 32, 2, 1, 3, rng, s);
            expert_fake::calls.clear(); expert_fake::fail_at = 2; // begin succeeds, kernel throws
            expert_fake::end_error = true; expert_fake::partial_create = partial;
            auto a = graphs(b, false, s);
        } catch (const std::exception& e) { original = std::string(e.what()) == "injected kernel exception"; }
        require(original && g_fail == 1, "cleanup masked original failure");
        require(expert_fake::captures_ended == 1 && expert_fake::empty(), "abandoned capture leaked");
        require(expert_fake::ns && expert_fake::old, "dual selectors not restored");
    }
    std::printf("PROBE dual probes=2 selectors=restored resources=0\n"); return 0;
}

int refused_probe() {
    expert_fake::reset(); g_fail = 0; expert_fake::ns = expert_fake::old = true;
    {
        Guard guard; Stream s; Event e0, e1;
        ck(cudaStreamCreate(s.put()), "refused stream");
        ck(cudaEventCreate(e0.put()), "refused e0"); ck(cudaEventCreate(e1.put()), "refused e1");
        std::mt19937 rng(2715); Bench b(18, 20, 256, 32, 2, 1, 3, rng, s);
        auto a = graphs(b, false, s); auto r = graphs(b, true, s);
        Graph g; Capture capture(s, "refused begin"); b.call(0, s); capture.end(g, "refused end");
        expert_fake::cleanup_attempts = 0; expert_fake::refuse_cleanup = true;
    }
    require(g_fail == 20 && expert_fake::cleanup_attempts == 20 && expert_fake::live.size() == 20,
            "refused destroy stopped other best-effort releases");
    require(expert_fake::ns && expert_fake::old, "refused destroy prevented selector restore");
    // The fake deliberately refused every release: only the fixture reclaims those remaining objects.
    expert_fake::refuse_cleanup = false;
    while (!expert_fake::empty()) {
        auto entry = *expert_fake::live.begin();
        if (entry.second == "allocation") cudaFree(entry.first);
        else if (entry.second == "stream") cudaStreamDestroy(static_cast<cudaStream_t>(entry.first));
        else if (entry.second == "event") cudaEventDestroy(static_cast<cudaEvent_t>(entry.first));
        else if (entry.second == "graph") cudaGraphDestroy(static_cast<cudaGraph_t>(entry.first));
        else cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(entry.first));
    }
    std::printf("PROBE refused attempted_releases=20 refused_resources=20 selectors=restored fixture_reclaimed_resources=0\n");
    return 0;
}

int blob_probe() {
    expert_fake::reset(); g_fail = 0; expert_fake::ns = expert_fake::old = true;
    bool caught = false;
    try {
        Guard guard; Stream s; ck(cudaStreamCreate(s.put()), "blob stream");
        std::mt19937 rng(2715); expert_fake::bad_layout = true;
        Bench b(18, 20, 256, 32, 2, 1, 3, rng, s);
    } catch (const std::exception& e) { caught = std::string(e.what()).find("blob ") == 0; }
    require(caught && expert_fake::empty(), "blob validation bypassed constructor unwind");
    require(expert_fake::ns && expert_fake::old, "blob selectors not restored");
    std::printf("PROBE blob probes=1 selectors=restored resources=0\n"); return 0;
}

// sta-osh.31: the scoped policy is enabled before any CUDA work, nests, and restores whatever this
// thread had - including an already-enabled prior policy - on normal and error exits of the whole CLI.
int policy_probe() {
    int checks = 0;
    std::vector<std::string> args = {"expert", "--groups", "1", "--tokens", "1", "--slots", "2",
                                   "--max-slots", "2", "--warmup", "1", "--samples", "1"};
    std::vector<char*> argv; for (auto& arg : args) argv.push_back(arg.data());
    // Both prior policies: default-off (the engine default) and already-on (a nesting caller).
    for (bool prior : {false, true}) {
        expert_fake::prior_policy = prior;
        // Success: policy active for every CUDA step, restored to `prior` afterwards.
        expert_fake::host_errors_throw = prior;
        expert_fake::reset(); g_fail = 0; expert_fake::ns = expert_fake::old = true;
        require(expert_benchmark_main(static_cast<int>(argv.size()), argv.data()) == 0, "policy baseline failed");
        require(expert_fake::host_errors_throw == prior, "policy not restored after success");
        require(expert_fake::steps_without_policy == 0 && expert_fake::steps_with_policy > 0,
                "CUDA ran outside the policy scope");
        const int total_steps = expert_fake::steps_with_policy;
        ++checks;
        // Error at every CUDA operation: the failure unwinds under the policy and still restores it.
        for (int at = 1; at <= total_steps; ++at) {
            expert_fake::host_errors_throw = prior;
            expert_fake::reset(); g_fail = 0; expert_fake::ns = expert_fake::old = true;
            expert_fake::fail_at = at;
            const int rc = expert_benchmark_main(static_cast<int>(argv.size()), argv.data());
            // Nonzero either way: 2 when an exception unwinds, 1 when only a noexcept release reported.
            require(rc == 1 || rc == 2, "policy error path is not fail-closed");
            require(expert_fake::host_errors_throw == prior, "policy not restored after error");
            require(expert_fake::steps_without_policy == 0, "error path ran CUDA outside the policy");
            require(expert_fake::empty() && expert_fake::ns && expert_fake::old, "policy error unwind leaked");
            ++checks;
        }
        // A throwing IQ kernel failure (what the policy exists for) unwinds with the policy enabled.
        for (int kind : {0, 1, 2}) {
            expert_fake::host_errors_throw = prior;
            expert_fake::reset(); g_fail = 0; expert_fake::ns = expert_fake::old = true;
            expert_benchmark_main(static_cast<int>(argv.size()), argv.data());
            auto it = std::find(expert_fake::calls.begin(), expert_fake::calls.end(), "kernel");
            require(it != expert_fake::calls.end(), "no kernel launch to fail");
            const int index = static_cast<int>(it - expert_fake::calls.begin() + 1);
            expert_fake::host_errors_throw = prior;
            expert_fake::reset(); g_fail = 0; expert_fake::ns = expert_fake::old = true;
            expert_fake::fail_at = index; expert_fake::kernel_throw = kind;
            require(expert_benchmark_main(static_cast<int>(argv.size()), argv.data()) == 2, "kernel error escaped");
            require(expert_fake::kernel_errors == 1 && expert_fake::kernel_errors_without_policy == 0,
                    "kernel error was raised without the policy enabled");
            require(expert_fake::host_errors_throw == prior, "policy not restored after kernel error");
            require(expert_fake::empty() && expert_fake::ns && expert_fake::old, "kernel error unwind leaked");
            ++checks;
        }
    }
    expert_fake::host_errors_throw = expert_fake::prior_policy = false;
    expert_fake::reset();
    // Nesting: an inner scope leaves the outer caller's enabled policy untouched.
    {
        k::IqHostErrorScope outer;
        require(expert_fake::host_errors_throw, "outer scope did not enable");
        { k::IqHostErrorScope inner; require(expert_fake::host_errors_throw, "inner scope disabled"); }
        require(expert_fake::host_errors_throw, "inner scope clobbered the outer policy");
        ++checks;
    }
    require(!expert_fake::host_errors_throw, "outer scope did not restore the default");
    std::printf("PROBE policy checks=%d policy=restored cuda_outside_policy=0 resources=0\n", checks);
    return 0;
}

int move_probe() {
    expert_fake::reset(); g_fail = 0;
    {
        auto a = dalloc<int>(1); auto b = dalloc<int>(1);
        b = std::move(a); require(expert_fake::live.size() == 1, "move assignment leaked destination");
        Device<int> c(std::move(b));
        auto& same = c; c = std::move(same);
    }
    require(expert_fake::empty() && expert_fake::cleanup_attempts == 2, "move double free/leak");
    std::printf("PROBE move attempted_releases=2 resources=0\n"); return 0;
}
}

#ifndef EXPERT_CLEANUP_PROBE_MAIN
#define EXPERT_CLEANUP_PROBE_MAIN main
#endif
int EXPERT_CLEANUP_PROBE_MAIN(int argc, char** argv) {
    try {
        require(argc == 2, "phase argument required"); const std::string which = argv[1];
        if (which == "entry") return entry_probe();
        if (which == "cleanup") return cleanup_probe();
        if (which == "move") return move_probe();
        if (which == "dual") return dual_probe();
        if (which == "refused") return refused_probe();
        if (which == "blob") return blob_probe();
        if (which == "policy") return policy_probe();
        return sweep(which);
    } catch (const std::exception& e) { std::fprintf(stderr, "PROBE FAIL: %s\n", e.what()); return 1; }
}
