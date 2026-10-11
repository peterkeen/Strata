# Opt-in Linux arena startup loading

`str-2bg` adds a bounded `O_DIRECT` reader for the resident arena loaded from
`experts.bin` on Linux CUDA/HIP builds. It does not change the default production
configuration. Reduced startup memory pressure is a hypothesis, not a measured
nibbler result.

## Selection and effective path

- Unset `STRATA_UNBUFFERED_LOAD`, or `STRATA_UNBUFFERED_LOAD=0`: the existing
  buffered startup reader, unchanged read/copy/hash loop.
- `STRATA_UNBUFFERED_LOAD=1`: request the new Linux startup reader. This existing
  variable also controls the separate file-tier reader; do not mistake file-tier
  serving results for startup results.
- Windows retains its existing selection, direct implementation and registration
  order. No Windows load algorithm or defaults were changed.
- Linux GGUF scatter loading is **not implemented by this change**. It remains
  buffered and logs that the override cannot be honored. `STRATA_ARENA_MMAP` and
  `--mmap-experts` do not select this resident `experts.bin` startup path.
- SYCL uses separate migrated loader sources; it does not gain this opt-in path.
  The shared header changes add statistics without requiring a new SYCL link
  dependency. A native SYCL build remains an unperformed portability check.

The Linux startup log reports `direct`, `buffered-fallback`, or
`mixed-direct-buffered-fallback`, with the requested policy, payload counts and
first fallback reason. A failure before any completed payload is reported as
`failed-before-payload`, followed by the caller's refusal message. Counts are
**successfully read requested bytes**, not physical disk I/O: rounded windows,
partial attempts and buffered re-reads are not separate payload counts. Existing
load timing fields retain wall time and summed worker read / copy-plus-FNV time;
this change does not claim a separately measured checksum time or add API metrics.

## Reader contract

Each worker holds one scratch allocation of at most 8 MiB + 8 KiB, aligned to
4096 bytes. At most 16 workers are used, also bounded by the requested threads and
layer count: at most 128 MiB + 128 KiB of scratch. Linux does not force the Windows
16-reader minimum. File descriptors are opened once per worker.

The reader rounds each source window, not the destination range. Unaligned
source offsets, range sizes, chunks, and destination pointers are supported. It
copies only requested bytes; arena gaps and padding stay untouched. Ranges must
be sorted, non-overlapping, and representable by `off_t` and pointer arithmetic.
The arena capacity is the caller's responsibility, as for the existing loader.

- Open, stat and read operations retry `EINTR` where applicable.
- Aligned direct partial reads continue with aligned offset, address and length.
  A sufficient EOF tail is accepted even if the rounded window exceeds the file.
- An insufficient non-aligned partial direct read switches that worker to
  buffered reads and re-reads the requested bytes before copying anything.
- Refused direct opens and direct support/alignment errors (`EINVAL`,
  `EOPNOTSUPP`, `ENOSYS`, `ENOTTY`, `EPERM`) switch to buffered `pread`, persistently
  for that worker. Buffered partial reads continue; `EINTR` retries.
- Real I/O failures such as `EIO`, premature zero-byte reads, truncation, and
  out-of-file requests refuse the load. They are not hidden by fallback or zero
  filling. Failed loads publish no per-layer checksums.
- Buffered and direct descriptors must identify the same inode/device/size.
  Model files must remain immutable during loading; this is not a snapshot
  mechanism for concurrent in-place modifications.
- tmpfs ignores `O_DIRECT`; it is explicitly reported as buffered fallback.
  Failure to inspect the filesystem also selects fallback. Other filesystems
  still need host validation of their direct-I/O semantics. A conservative 4 KiB
  alignment can be refused by a filesystem needing stricter alignment; fallback
  preserves correctness rather than promising cache bypass everywhere.

Layer L is not written until the existing `ready > L + 1` acquire fence permits
it (both its slice and the adjacent slice), or registration publishes its terminal
sentinel. Linux's default whole-arena registration is unchanged. FNV-1a uses the
existing seed `1469598103934665603` and multiplier `1099511628211`, in increasing
byte order within each layer, independent of worker scheduling.

There is no global cache drop, swap/hugepage policy change, or new cache-eviction
advice. In particular, per-file `POSIX_FADV_DONTNEED` is not used: it can evict
useful shared cache belonging to other readers. Direct I/O avoids creating the
usual buffered file-cache copy but does not remove cache that was already present,
shrink the pinned arena, or guarantee a lower host memory peak. Buffered fallback
can still create substantial file cache. Warm cached loads may be faster than
reading storage directly; this remains opt-in.

## GPU-free local fixtures

No model, GPU toolkit, CMake, or downloads are needed:

```sh
mkdir -p /tmp/str-2bg-tests
g++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread -Iinclude \
  src/platform/linux_arena_load.cpp tests/platform/linux_arena_load_test.cpp \
  -o /tmp/str-2bg-tests/linux_arena_load_test
/tmp/str-2bg-tests/linux_arena_load_test
```

Fixtures verify actual direct reads (or truthful filesystem fallback), exact
bytes against an independent file oracle, independent per-layer FNV, canaries,
aligned/unaligned ranges and EOF tails, injected interrupted/partial reads,
unsupported opens/reads, persistent and mixed fallback, fatal EOF/EIO,
pre/post-open truncation, input overflow, bounded chunks, and the registration
fence. Injected direct handles verify alignment but use buffered descriptors to
make failure cases deterministic; they are not evidence of device DMA.

The standalone CMake registration is separate from conversation/model/GPU tests:

```sh
cmake -S . -B build-arena-cpu -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_ARENA_LOAD_TESTS=ON
cmake --build build-arena-cpu --target linux_arena_load_test -j2
ctest --test-dir build-arena-cpu -R '^linux_arena_load_test$' --output-on-failure
```

## Primary-only host build and native gate

These commands are instructions for the primary/designated live helper, **not
permission to run them against production**. Use the isolated transferred source
checkout, not the production checkout. No install, deployment, router restart or
configuration write is included. Use the same CUDA 13.3 / GCC toolchain and local
llama.cpp source as the baseline.

```sh
# Run from the isolated source checkout on nibbler; third_party/llama.cpp must exist.
export PATH=/usr/local/cuda-13.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cmake -S . -B build-str2bg -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_BUILD_ARENA_LOAD_TESTS=ON \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build build-str2bg --target strata linux_arena_load_test linux_arena_parity -j4
ctest --test-dir build-str2bg -R '^linux_arena_(load_test|native_parity)$' --output-on-failure
```

`linux_arena_native_parity` uses a 131209-byte synthetic file, the actual unchanged
buffered loader, actual new Linux loader and a registered arena. It checks
per-range and whole-arena FNV, byte counts and padding canaries. It fails rather
than claiming a direct-I/O gate passed if registration or direct I/O is unavailable
(including a tmpfs `/tmp`). The GPU-free test checks exact bytes, not just hashes.

Native CUDA/HIP link/runtime tests, Windows and SYCL builds were not possible in
the implementation VM. HIP must rebuild `strata`, `linux_arena_load_test` and
`linux_arena_parity` using the existing backend/toolchain instructions in
`docs/AMD_HIP.md`; do not reuse a CUDA build directory. Windows/SYCL backend builds
are still needed before claiming cross-backend validation.

## Quiet-window Qwen parity and loader micro A/B

**Only the primary/designated helper may run these, after the approved quiet
window's preflight and stop gates. Do not run alongside the resident production
Qwen engine or other model trials.** The snapshot arena was 64800 MiB / 48 layers;
verify current pack geometry and file size before using those numbers.

```sh
# Full parity: ONE arena reused sequentially, not two 63 GiB allocations.
# The test splits bytes into equal ranges (last takes any remainder); it is not a pack-layout parser.
build-str2bg/linux_arena_parity \
  --file /data/llm/models/qwen-nvfp4-gptq/pack/experts.bin \
  --bytes 67947724800 --layers 48 --threads 8 --mode parity

# Separate-process load micro measurements; operator must recheck gates before EACH invocation.
/usr/bin/time -v build-str2bg/linux_arena_parity \
  --file /data/llm/models/qwen-nvfp4-gptq/pack/experts.bin \
  --bytes 67947724800 --layers 48 --threads 8 --mode buffered
/usr/bin/time -v build-str2bg/linux_arena_parity \
  --file /data/llm/models/qwen-nvfp4-gptq/pack/experts.bin \
  --bytes 67947724800 --layers 48 --threads 8 --mode direct
```

Every per-range checksum must match; the parity run's whole-arena FNV must match
as well. Fallback is a correctness result, not a successful direct-I/O performance
gate. Do not use the parity run's buffered-then-direct times as an unbiased A/B.
Micro modes report loader wall/read/copy-plus-hash time and checksums; `/usr/bin/time`
adds whole-process time and peak RSS. They do not measure server time to ready.

For full server A/B, the designated helper must launch the **same isolated binary
and private config** twice, changing only private `env.STRATA_UNBUFFERED_LOAD`
(`0` baseline, `1` candidate), with `STRATA_VERIFY_ARENA=1` on both parity runs.
Require equal `expert arena checksum` values and a successful ready state. Keep
model/template, vision, two slots, long context, reserve, pool and PCIe split
fixed. Whole-arena verification adds a full scan; time separate matched runs with
verification disabled only after the correctness gate passes. No production
config changes are required or recommended by this change.

Capture time to ready, load times, process RSS/VmPin/VmLck, host/cgroup file-cache
bytes, MemAvailable, swap-in/out deltas, pressure stalls and neighboring-service
latency. CUDA pin accounting is not always represented by VmPin; report the
engine's registration byte count too. Alternate A/B and B/A, distinguish naturally
cold from warm starts, and do not manufacture cold cache by eviction. Apply the
primary's agreed memory/swap/latency stop thresholds before and during each run.
No claim of lower pressure or acceptable warm-start cost is established by local
fixtures alone. Keep the option off until host evidence supports using it.

## Local verification recorded

- Optimized GCC C++20 `-Wall -Wextra -Werror` fixtures: passed.
- AddressSanitizer + UndefinedBehaviorSanitizer fixtures: passed.
- Existing `direct_file_async` standalone fixture: passed.
- `tools/test_setup_amd.py`: 36 tests passed (existing `/proc/cpuinfo` resource
  warning). Used the explicit Python 3.12.13 interpreter; the unconfigured `python`
  mise shim could not run it.
- Native parity source: GCC warning-clean compile to object only, **not linked or
  run**. Native CUDA/HIP/Windows/SYCL and CMake builds: not run; no toolchains/CMake
  in this VM, and no toolchain installation performed.
- Primary independently reported the loader fixtures and eight existing
  conversation/KV CPU tests passing.
- No host/production tests, Beads writes, commits, pushes, agents or worktrees were
  performed by this implementation helper.
