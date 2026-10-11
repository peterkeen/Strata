# Isolated nibbler native build — 2026-10-11

This archive preserves the **already-completed** CUDA compile/link and GPU-free
CTest verification of the frozen Linux loader and fetch-reporting changes.
`str-2bg.1` copied previously retrieved local evidence only; no new host operation,
build, model request, GPU test or production change was performed to create it.
It does not close the parent issue's runtime or performance acceptance gates.

## Result and scope

| Property | Recorded result |
|---|---|
| CUDA / host compilers | CUDA 13.3.73; GCC/G++ 15.3.0 |
| Build | Release, arch 120, `-j2` (process niceness 10) |
| Configure tools | CMake 4.3.4; Ninja 1.13.2 |
| Native result | 166 Ninja steps; 219 seconds; `build_exit=0` |
| UTC build interval | 2026-10-11T21:17:55Z–2026-10-11T21:21:34Z |
| Linked targets | `strata`, `linux_arena_load_test`, `linux_arena_parity`, eight standalone conversation/KV tests |
| Executed tests | Nine CPU CTests; zero failures/skips; 0.54 seconds |
| Native parity | **Compiled and linked; NEVER EXECUTED** |
| Source snapshot | 597 sanitized files, 11100157 bytes; loader and fetch header included |
| Shared dependency | Declared pin `3cf03257f219afbe7334045ff7c6a06ac68c627d`; GGML 0.24.0; recorded source hashes unchanged |
| Build environment | CUDA devices hidden; compiler cache isolated inside candidate build |

The build contained nonfatal compiler warnings; it is not a warning-free build.
No compile/link errors or source fixes were needed. The fetch header's dependency
in `generate.cpp.o` is explicitly recorded. Exact commands are in the original
[report](REPORT.md); complete compiler output is in [build.log](build.log).

### Immutable identities

Source-manifest SHA-256:

```text
c9c32bd0a87161c87f350b8de7a63376578be1611cf9013e63a0f4ed71513d28
```

Candidate `strata` SHA-256 (binary **not** copied):

```text
6bbba11f5de003b43715c79cd4cba617373fe4d8920679dfcad5fa622ceaf68e
```

Recorded production artifact/source hashes, shared GGML source hashes, and Qwen /
vision PID/start-tick identities were unchanged across the completed build. That
is integrity evidence for that interval, not a fresh readiness probe, a claim of
zero neighboring-service impact, or authorization for more host operations.

## Retained originals

All 18 originals below were safety-inspected and copied byte-for-byte from
`/tmp/str-2bg-native-verification/host-results`. [SHA256SUMS](SHA256SUMS) covers
them and this archive README. The original hash manifests retain their recorded
host paths; they are provenance, not instructions to access those paths now.

| Files | Purpose |
|---|---|
| [REPORT.md](REPORT.md) | Original build/test report, commands, limits and recorded binary hashes |
| [configure.log](configure.log), [build.log](build.log) | Complete unabridged CMake and compiler/link output |
| [build-status.txt](build-status.txt), [compiler-matrix.txt](compiler-matrix.txt) | UTC timing, exit status and compiler availability |
| [ctest-cpu.log](ctest-cpu.log) | Nine-test summary and 0.54-second total |
| [ctest-cpu-full.log](ctest-cpu-full.log) | Complete original `LastTest.log` output for the nine selected tests |
| [ctest-cpu.xml](ctest-cpu.xml) | Original JUnit output; first test's stdout is truncated by CTest, not by this archive |
| [SOURCE_SHA256SUMS](SOURCE_SHA256SUMS), [source-manifest.sha256](source-manifest.sha256) | All 597 transferred source-file hashes and manifest identity |
| [BINARY_SHA256SUMS](BINARY_SHA256SUMS), [LOG_SHA256SUMS](LOG_SHA256SUMS) | Recorded executable and configure/build/CTest-summary identities |
| [fetch-header-dependency.txt](fetch-header-dependency.txt) | Ninja dependency evidence for the fetch policy header |
| [ggml.before.sha256](ggml.before.sha256) | 1376 shared GGML source-file hashes |
| [production.before.sha256](production.before.sha256) | Ten production artifact/config/source hashes only |
| [production-processes.before.json](production-processes.before.json), [production-processes.after.json](production-processes.after.json) | Only executable paths, PIDs and start ticks; no command-line arguments |
| [state-audit.txt](state-audit.txt) | Original post-build hash and process-identity checks |

Key original SHA-256 values, compared with the previously recorded proof:

| Original | SHA-256 |
|---|---|
| `configure.log` | `e8d5f850975dc8029ddf5d4d7d6cfc6e0ebcf99fc06948b66e0766dac8d5d182` |
| `build.log` | `e88160ad8f1cf4544ebcf377dc3799ec4458a405c1888829e32fdf960dba4346` |
| `ctest-cpu.log` | `e23cee1c5f13d005d09502ba861a55c1a4e674c063bcdbda1c9eaba64de152a7` |
| `ctest-cpu-full.log` | `38f34140aac3f8325d9792577025e500b168a9da6c5a969fa40e99e28be23b37` |
| `REPORT.md` | `8b2d49a45ecd1553efe77215d91b9e8d1e69ce2007950b6ab76331ca6ceb3b01` |

All 597 files in the still-local frozen snapshot matched the original source
manifest during this documentation task. Thirteen current loader/fetch/CMake
inputs also matched that manifest. This is not a claim that the independently
maintained calibration tools or the entire current checkout remain at that
snapshot; the source manifest identifies exactly what was compiled.

### Safety and completeness

Inspection included credential/private-key/token, raw-environment and production
request-transcript patterns, plus review of the retained report, test output and
integrity records. No unsafe evidence was found in the selected originals, so no
selected original required redaction or omission. The production manifest names
`config.shared-settings.json` but contains its hash and path **only**, never its
contents. Controlled build settings in the report are not a raw environment dump.
Dependency/source manifests contain file names and hashes, not file contents.

No executables, frontend copies, model files, source trees, shared-settings
contents, secrets, raw environment, production request logs or unrelated large
logs are included. Original log whitespace is retained to preserve exact hashes.
Missing output has not been reconstructed. Use `ctest-cpu-full.log` for complete
fixture output rather than the truncated JUnit field. Its `1/123` labels refer to
registered suite positions; only the nine explicitly selected CPU tests ran.

## Acceptance still pending

- Host `/tmp` is tmpfs: real-file CPU cases reported **buffered fallback**.
  Injected direct cases test alignment/error handling through simulated direct
  handles; they are not actual storage-direct-I/O evidence.
- The primary reports a separate independent VM ext4 CPU audit observing real
  direct reads. That is not nibbler `/data` filesystem evidence, registered
  native-arena parity or production-model evidence. Its original audit output is
  not part of this native-build archive and is not reconstructed here.
- GPU runtime, registered native parity, full-arena checksums, model token/logit
  parity, lifecycle tests and startup memory/latency A/B remain unperformed.
  HIP/SYCL compilers were unavailable on the checked host paths; Windows was not
  built. This result does not establish cross-backend release validation.
- The primary reports independent reviews of the same code with no Blocker/High
  findings for nibbler readiness. That scope is **not release approval**; original
  review reports are not part of this archive. GPU validation and acceptance for
  all four audit issues (`str-pk5`, `str-inn`, `str-2bg`, `str-5p1`) remain pending.
- No deployment, default/production configuration change, admission enablement
  or performance-benefit claim follows from these results. Only the designated
  primary/operator owns subsequent host preflight and any newly authorized tests.

See [Linux arena loading](../../../docs/LINUX_ARENA_LOAD.md) and
[batch fetch assessment](../../../docs/BATCH_FETCH_ASSESSMENT.md).

Local archive verification (no host access):

```sh
cd bench/results/2026-10-11-nibbler-native-build
sha256sum --check SHA256SUMS
```
