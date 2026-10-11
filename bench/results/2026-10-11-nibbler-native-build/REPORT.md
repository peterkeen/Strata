# Frozen loader + fetch: native build verification

Scope: isolated CUDA build and GPU-free CTest only. No model load/unload/inference, frontend, native CUDA/parity/device fixture execution, production overwrite, config change, package installation, Beads mutation, commit or push by this helper. No source fixes were needed. RemoteShell used strict_host_key_checking:true, retries:0. Homelab preflight passed; equivalent read-only inspections used hl.

## Locations

Host root: `/data/llm/Strata-tests/perf-audit-20261011`
- Frozen sanitized sources: `candidate-src` (597 files, 11100157 bytes, no git/worktree/cache/model/secret files).
- Fresh native build: `candidate-build`.
- Complete unabridged configure/build/CTest output and state evidence: `candidate-verification`.
- Local downloaded evidence: `/tmp/str-2bg-native-verification/host-results`.

Source manifest: `candidate-src/SOURCE_SHA256SUMS`
SHA-256: `c9c32bd0a87161c87f350b8de7a63376578be1611cf9013e63a0f4ed71513d28`
Every source file passed SHA-256 verification after transfer and again after building/testing.
Fetch header dependency in `generate.cpp.o` was confirmed with Ninja's dependency database.

## Toolchain/configuration

CUDA 13.3.73, GCC/G++ 15.3.0, Ninja 1.13.2, CMake 4.3.4, Release, CUDA arch 120. C/C++/CUDA-host compiler paths match the existing host build. All build/test commands had `CUDA_VISIBLE_DEVICES=-1`. The compiler cache used `candidate-build/.ccache`, not a shared cache.

Shared read-only dependency: `/data/llm/Strata/third_party/llama.cpp` through STRATA_GGML_DIR. Its sibling ggml/VERSION.txt matches the frozen source pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`; GGML version 0.24.0. No .git/HEAD exists inside that extracted dependency; CMake's printed git commit can reflect the containing Strata repository, not the pinned dependency. Full GGML source hashes were recorded before the build and verified unchanged afterward.

Exact configure command (root abbreviates the host root above):

```sh
cmake -S "$root/candidate-src" -B "$root/candidate-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-15 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_BUILD_ARENA_LOAD_TESTS=ON \
  -DSTRATA_GGML_DIR=/data/llm/Strata/third_party/llama.cpp \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

nice -n 10 cmake --build "$root/candidate-build" \
  --target strata linux_arena_load_test linux_arena_parity \
  shared_kv_pages_test shared_kv_reservation_test batch_draft_coherence_test batch_mtp_policy_test \
  conversation_cache_test conversation_split_failure_test conversation_memory_test conversation_file_test \
  -j2
```

## Results

Native build: exit 0, all 166 Ninja steps completed. UTC start 2026-10-11T21:17:55Z, finish 2026-10-11T21:21:34Z (219 seconds). Compiled pinned.cu, linux_arena_load.cpp, expert_source.cpp, generate.cpp and the fetch header; linked strata and both arena executables. Nonfatal compiler warnings were emitted in other kernel/core code and generate.cpp:9641 (unused `w`); no compile/link errors and no fixes applied.

Binary SHA-256:
- strata: `6bbba11f5de003b43715c79cd4cba617373fe4d8920679dfcad5fa622ceaf68e`
- linux_arena_load_test: `aaf9b5e59dd08a612571729ada84b8b9a350ecdf644875d23c96dcbeafb137b1`
- linux_arena_parity: `9f7b7ca186b5e8ce4299642b4a3fb5e68438749dfcae931a2c037353e403aeb1`

Native parity executable compiled/linked ONLY, NOT executed.

Exact CPU-only CTest selection:

```sh
ctest --test-dir "$root/candidate-build" --parallel 1 --no-tests=error --output-on-failure \
  -R '^(linux_arena_load_test|shared_kv_pages_test|shared_kv_reservation_test|batch_draft_coherence_test|batch_mtp_policy_test|conversation_cache_test|conversation_split_failure_test|conversation_memory_test|conversation_file_test)$' \
  --output-junit "$root/candidate-verification/ctest-cpu.xml"
```

Output: `100% tests passed, 0 tests failed out of 9`; real time 0.54 seconds. One loader fixture and eight standalone conversation/KV tests. No GPU tests were selected. Host /tmp is tmpfs: the real-file loader cases truthfully selected buffered fallback; deterministic injected direct cases passed. This host run therefore does NOT establish real disk O_DIRECT/native arena/model parity. Full fixture output is in candidate-build/Testing/Temporary/LastTest.log (JUnit output is truncated for the first case).

Exact log hashes:
- configure.log: `e8d5f850975dc8029ddf5d4d7d6cfc6e0ebcf99fc06948b66e0766dac8d5d182`
- build.log: `e88160ad8f1cf4544ebcf377dc3799ec4458a405c1888829e32fdf960dba4346`
- ctest-cpu.log: `e23cee1c5f13d005d09502ba861a55c1a4e674c063bcdbda1c9eaba64de152a7`

## Production/dependency integrity

Recorded production binary, vision binary, BUILD.json, tracked/runtime configs, shared-settings file and relevant production source checksums all verified unchanged. Shared GGML source checksums unchanged. Production vision PID 2881000/start_ticks 65583992 and Qwen engine PID 2881053/start_ticks 65584100 unchanged. No production health inference beyond these integrity checks; no inference requests were sent.

Host summary before/after via hl: MemAvailable 24077017088 -> 24068550656 bytes; load averages 2.14/2.44/3.35 -> 1.95/2.41/2.94. These are snapshots, not a benchmark or proof of zero neighbor impact. Only isolated candidate/evidence/build artifacts and normal transient compiler resources were created.

## Skipped matrix and acceptance limits

HIP: hipcc absent from PATH and /opt/rocm/bin/hipcc. SYCL: icpx/dpcpp absent from PATH and the conventional oneAPI compiler/latest path. No toolchains installed. Windows build not performed. CUDA runtime, registered native arena parity, whole-model checksum/output parity, real-disk O_DIRECT and startup memory/latency A/B remain for the repaired measurement workflow in an authorized quiet window. This result establishes native compile/link compatibility and CPU fixture success only. Default production loader configuration remains unchanged.
