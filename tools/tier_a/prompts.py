"""tools/tier_a/prompts.py - prompt texts for the Tier-A benchmark arms.

Prompts are stored as plain strings.  The harness builds token-id lists from
them using the pack tokenizer (strata_tokenizer.Tokenizer) and writes a
temporary --tokens file; no tokenizer package beyond `regex` is required.

SHORT (~50 tokens prompt): a code question that yields a short but bounded
answer (Python merge-sort).  Useful for startup/warmup arms.

LONG (>8 K tokens prompt, >200 output tokens target): a reference document
followed by a detailed technical question.  The document is a lightly padded
excerpt about quicksort / introsort / timsort so it is stable across runs.

STORY (medium, ~200-512 output target): a creative continuation task that
tends to produce 200+ tokens before EOS without being unbounded.

All three prompts are formatted as Qwen3 chat turns with thinking disabled
(<think>\n\n</think>) so the output starts directly without reasoning tokens.
"""
from __future__ import annotations

_CHAT_TMPL = (
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

# ── short code prompt ─────────────────────────────────────────────────────────

SHORT_CODE = _CHAT_TMPL.format(user=(
    "Write a Python function `merge_sorted(a, b)` that merges two sorted lists "
    "into one sorted list. Include a docstring and two test cases."
))

# ── story / creative prompt (targets ~200-512 output tokens) ──────────────────

STORY = _CHAT_TMPL.format(user=(
    "Continue the following story in two paragraphs, then summarise the main "
    "character's motivation in one sentence:\n\n"
    "The lighthouse keeper had not left the island in eleven years. Every evening "
    "she climbed the iron stairs and lit the lamp, and every morning she climbed "
    "back down and wrote three sentences in a blue notebook. The notebook was "
    "almost full."
))

# ── long document + question (>8 K tokens, >200 output tokens) ───────────────
# Document: a technical overview of comparison-based sorting algorithms,
# padded to ~8 200 tokens with elaboration so the prefill path is exercised.

_SORT_DOC = """\
# Comparison-Based Sorting Algorithms: A Technical Reference

## 1. Introduction

Sorting is one of the most-studied problems in computer science. A comparison-based
sort determines the order of elements exclusively by comparing pairs; it cannot
exploit the structure of key values beyond the < / = / > relation. The information-
theoretic lower bound for comparison-based sorting of n elements is Omega(n log n)
comparisons in the worst case, proved by a decision-tree argument: a binary tree
with n! leaves needs height >= log2(n!), and Stirling's approximation gives
log2(n!) = n log2(n) - n log2(e) + O(log n).

## 2. Insertion Sort

Insertion sort builds the sorted array one element at a time. For position i it
scans leftward from i-1 while the scanned element exceeds a[i], shifting each
element one position right, then inserts a[i] into the gap. The invariant is that
a[0..i-1] is sorted after processing position i.

Complexity: O(n^2) worst-case and average, O(n) best-case (already sorted input).
Space: O(1). Cache behaviour: excellent, because every access is to nearby memory.
Stable: yes. The constant factor is small; insertion sort outperforms Quicksort
for n <= 10-20 in practice, which is why library implementations use it as a
base case.

## 3. Merge Sort

Merge sort divides the array into two halves, sorts each half recursively, and
merges the sorted halves. The merge step scans both halves in parallel, always
picking the smaller front element, in O(n) time.

Complexity: O(n log n) worst, average, and best. Space: O(n) for the scratch
buffer (an in-place merge is possible but costs O(n log^2 n) or O(n log n) with
a complex algorithm). Stable: yes. The merge step's sequential access pattern
is cache-friendly on large inputs, and merge sort is the algorithm of choice for
linked lists and for external sorting (data does not fit in RAM).

Top-down vs bottom-up: the top-down form recurses to length-1 subarrays and
merges on the way up. The bottom-up form iterates, doubling the run length each
pass: first merging runs of length 1, then 2, 4, 8 and so on. Both have the
same asymptotic cost; the bottom-up form avoids recursion overhead and is simpler
to implement for fixed-size integers.

## 4. Quicksort

Quicksort partitions the array around a pivot element so that every element left
of the pivot is <= pivot and every element right of the pivot is >= pivot, then
recurses on both sides. Lomuto partition places the pivot at its final position
in a single left-to-right scan; Hoare partition uses two converging pointers and
is roughly twice as fast in practice (fewer swaps).

Complexity: O(n log n) expected, O(n^2) worst-case. Space: O(log n) expected
stack depth. Not stable. Pivot selection determines worst-case frequency: a
sorted or reverse-sorted input degrades a naive first-element or last-element
pivot to O(n^2); median-of-three (the median of first, middle, last) avoids
this for common adversarial inputs; Tukey's ninther (median of three medians)
is used in some library implementations for large n.

Introsort (introspective sort, Musser 1997) starts as Quicksort and switches to
Heapsort when the recursion depth exceeds 2 * floor(log2(n)), bounding worst-case
to O(n log n) while retaining Quicksort's O(n log n) expected cache-friendly
behaviour for typical inputs. It switches to insertion sort for subarrays of
size <= 16. Introsort or a close variant is the sorting algorithm in C++ STL
std::sort, Rust's slice::sort_unstable, and most other systems languages.

## 5. Heapsort

Heapsort builds a max-heap from the array in O(n) time (Floyd's algorithm:
sift-down from n/2-1 down to 0), then repeatedly extracts the maximum,
swapping it to the end and sifting down, in O(n log n) time.

Complexity: O(n log n) worst, average, best. Space: O(1). Not stable. The
non-sequential memory access pattern (sift-down jumps to 2i+1 and 2i+2) causes
cache misses; in practice Heapsort is 2-5x slower than Quicksort for random data
despite identical asymptotic cost. It is valuable mainly as the fallback in
Introsort.

## 6. Timsort

Timsort (Peters 2002, Python's sort since 2.3, Java's Arrays.sort for objects
since 1.7) is a hybrid of merge sort and insertion sort designed for real-world
data, which frequently contains already-sorted or reverse-sorted runs.

Algorithm: scan the array for natural runs (monotone increasing or decreasing
sequences). Extend short runs to at least minrun (32-64 elements) with insertion
sort. Maintain a stack of pending runs; merge adjacent runs when the stack
violates the invariants run[-3] > run[-2] + run[-1] and run[-2] > run[-1]
(these ensure the stack stays O(log n) deep and the total merge cost is O(n log n)).

Complexity: O(n log n) worst, O(n) best (already sorted). Space: O(n). Stable.
The merge step uses galloping mode: if the same side wins many consecutive
comparisons, switch to exponential search to find how far it continues winning.
This amortises well on inputs with long runs and poorly on random inputs, where
Timsort falls back to ordinary merge.

## 7. Parallel Sorting

Parallel merge sort assigns subarrays to threads and merges in parallel. A naive
parallel merge is O(n) sequential; optimal parallel merge (Cole 1988) is O(log n)
with O(n) work. Sample sort partitions by selecting s-1 splitters from a random
sample of size s^2, distributes elements to s buckets, and sorts each bucket
independently; it is the standard parallel sort in distributed computing and HPC.

## 8. Cache-Oblivious Sorting

A cache-oblivious algorithm has optimal cache complexity without knowing the cache
size. Funnelsort (Brodal and Fagerberg 1999) achieves O(n log n / B * log_{M/B}(n/B))
cache misses, matching the optimal cache-aware lower bound, where B is the block
size and M is the cache size.

## 9. Radix Sort (Non-Comparison)

Radix sort processes key digits from least significant to most significant (LSD)
or vice versa (MSD). Each pass is a stable counting sort over one digit (typically
one byte). Complexity: O(nk) where k is the number of digits. For 32-bit integers
and byte-wide digits k=4, making it O(n) for bounded keys and often 2-4x faster
than comparison-based sorts in practice.

## 10. Selection in Linear Time

The selection problem (find the k-th smallest element) can be solved in O(n)
worst-case by the median-of-medians algorithm (Blum, Floyd, Pratt, Rivest,
Tarjan 1973): divide into groups of 5, find each group's median by insertion sort,
recurse on the medians to find their median, partition around that pivot. The
pivot is guaranteed to be between the 30th and 70th percentile, giving a
recurrence T(n) <= T(n/5) + T(7n/10) + O(n) whose solution is O(n). Introselect
(Musser 1997) applies the same depth-limiting idea as Introsort to nth_element.

## 11. Practical Considerations

Library sort benchmarks on modern hardware consistently show:
- For random data: Introsort (std::sort equivalent) ~ 1.5-2x faster than Timsort.
- For nearly-sorted data: Timsort ~ 5-10x faster than Introsort.
- For reverse-sorted data: Timsort ~ 3-5x faster.
- Radix sort for integers: 2-4x faster than any comparison-based sort at large n.
Branch prediction accuracy is a significant factor: Quicksort's partition loop
is highly predictable on random data; Heapsort's sift-down is not.

## 12. Stability and Its Applications

A stable sort preserves the relative order of equal elements. This matters when
sorting by multiple keys in succession (sort by secondary key, then primary key),
when sorting objects with identity (equal keys but distinct objects), and when
merging partially sorted streams. Merge sort, Timsort, and insertion sort are
stable; Heapsort, Quicksort, and most Introsort implementations are not. C++
std::stable_sort guarantees stability at O(n log^2 n) in-place or O(n log n)
with O(n) extra memory.

## 13. Summary Table

| Algorithm     | Best    | Average | Worst   | Space  | Stable |
|---------------|---------|---------|---------|--------|--------|
| Insertion sort| O(n)    | O(n^2)  | O(n^2)  | O(1)   | Yes    |
| Merge sort    | O(n lg) | O(n lg) | O(n lg) | O(n)   | Yes    |
| Quicksort     | O(n lg) | O(n lg) | O(n^2)  | O(lg)  | No     |
| Heapsort      | O(n lg) | O(n lg) | O(n lg) | O(1)   | No     |
| Timsort       | O(n)    | O(n lg) | O(n lg) | O(n)   | Yes    |
| Introsort     | O(n lg) | O(n lg) | O(n lg) | O(lg)  | No     |
| Radix sort    | O(nk)   | O(nk)   | O(nk)   | O(n+k) | Yes    |

(lg = log n; space is auxiliary space only)
"""  # ~1 950 words / ~2 500 tokens; repeated below to exceed 8 K

# The _SORT_DOC above is ~2 500 real tokens.  To reach ≥8 192 tokens we
# append three more sections: a worked-example walkthrough, an implementation
# notes section, and a complexity-analysis appendix.  Together these push the
# encoded token count well past 8 192 with the Qwen3 tokeniser.

_APPENDIX_A = """
## Appendix A — Worked Examples

### A.1 Merge Sort on [5, 2, 8, 1, 9, 3]

Divide: [5,2,8] and [1,9,3].
Recurse left: divide [5,2] and [8]; merge [2,5] + [8] = [2,5,8].
Recurse right: divide [1,9] and [3]; merge [1,9] + [3] = [1,3,9].
Final merge: compare 2 vs 1 → take 1; 2 vs 3 → take 2; 5 vs 3 → take 3;
5 vs 9 → take 5; 8 vs 9 → take 8; take 9.  Result: [1,2,3,5,8,9].
Total comparisons: 8 (worst-case for n=6 is ⌈n log n⌉ = 10; merge sort is
optimal for worst case but pays O(n) auxiliary space for every merge.

### A.2 Quicksort on [3, 6, 8, 10, 1, 2, 1] with Lomuto partition

Pivot = last element = 1.  Scan left-to-right: no element < 1 except the
final 1 itself.  After partition: [1, 6, 8, 10, 1, 2, 3] with pivot at
index 0 (or 1 depending on implementation — both are correct).  Recurse on
[6,8,10,1,2,3].  Pivot = 3; after partition: [1,2,3,10,6,8].  Continue
until all subarrays are length 1.

Note: the two 1s are equal keys.  Lomuto moves both to the left partition;
Hoare would handle them differently depending on initialisation.  Neither
guarantees stable relative order of equal keys.

### A.3 Timsort trace on [1, 3, 2, 4, 6, 5, 7, 8]

minrun = 4 (for n=8, minrun is chosen between 32 and 64 in the real
implementation; here we use 4 for illustration).  Scan:
  Run 1: [1,3] ascending, length 2 → extend with insertion sort to [1,2,3,4].
  Run 2: [6,5] descending → reverse to [5,6] → extend to [5,6,7,8].
Stack: [run1(4), run2(4)].  Invariant check: run[-2]=4, run[-1]=4;
run[-2] > run[-1] fails → merge → [1,2,3,4,5,6,7,8].  Done in one merge.

### A.4 Heapsort trace on [4, 10, 3, 5, 1]

Build max-heap (Floyd sift-down from index n//2-1 = 1 downward):
  Sift index 1: children are 3,5; max child 5 at index 3; swap → [4,10,3,5,1]
  (no swap needed for index 1=10; check: children are 5 and 1, 10>both; ok).
  Sift index 0: children are 10 and 3; 10>4 swap → [10,4,3,5,1].
  Sift index 0 again after swap: new subtree [4,5,1]; 5>4 swap → [10,5,3,4,1].
Heap: [10,5,3,4,1].  Extract max: swap 10↔1 → [1,5,3,4,10]; sift 1 at root;
5>1 swap → [5,1,3,4,10]; check subtree of 1: children 4; swap → [5,4,3,1,10].
Repeat: extract 5 → sorted suffix grows: [1,3,4,5,10].

### A.5 Radix Sort on [170, 45, 75, 90, 802, 24, 2, 66]

LSD pass on digit 0 (ones): bucket by ones digit:
  0: 170, 90  1: —  2: 802, 2  3: —  4: 24  5: 45, 75  6: 66  7: —  8: —  9: —
  Result: [170, 90, 802, 2, 24, 45, 75, 66]

LSD pass on digit 1 (tens): bucket by tens digit:
  0: 802, 2  2: 24  4: 45  6: 66  7: 170, 75  9: 90
  Result: [802, 2, 24, 45, 66, 170, 75, 90]

LSD pass on digit 2 (hundreds): bucket by hundreds digit:
  0: 2, 24, 45, 66, 75, 90  1: 170  8: 802
  Result: [2, 24, 45, 66, 75, 90, 170, 802].  Sorted.

### A.6 Insertion Sort as Timsort base case

For a run shorter than minrun, insertion sort brings it up to size.  On the
subarray [7, 3, 5, 2]:
  i=1: key=3; shift 7 right; insert 3 → [3,7,5,2]
  i=2: key=5; shift 7 right; insert 5 → [3,5,7,2]
  i=3: key=2; shift 7,5,3 right; insert 2 → [2,3,5,7]
3+2+1=6 comparisons; no extra allocation.
"""

_APPENDIX_B = """
## Appendix B — Implementation Notes

### B.1 Choosing minrun in Timsort

Python's Timsort computes minrun from n as follows: take the top 6 bits of n;
if any lower bit is set, add 1.  This keeps n/minrun close to a power of two,
which means the final merge pass touches nearly equal-sized runs and costs at
most ε extra comparisons.  For n=64 minrun=32; for n=100 minrun=26 (since
100=0b1100100, top 6 bits=011001=25, lower bits nonzero, so minrun=26).

### B.2 Galloping in Timsort

During a merge of run A and run B, if the same side wins k consecutive picks
(default k=7 in CPython), Timsort switches to galloping mode: it searches for
where B[0] would sit in A using exponential search (1, 2, 4, 8, … steps then
binary search within the final interval).  Galloping amortises well when one
run is much shorter than the other (e.g., merging a nearly-sorted array with a
small new run).  It performs poorly on random data and the algorithm exits
galloping mode whenever neither side wins k consecutive times.

### B.3 The Introsort depth limit

Musser's original paper used 2*log2(n) as the cutoff; GNU libstdc++ uses
2*floor(log2(n)) which for n=10^6 gives 39 recursion levels before falling
back to Heapsort.  In practice the Heapsort fallback is almost never triggered
on random data; it exists only to defend against adversarial or pathological
inputs.  Microsoft's STL uses a slightly different threshold; the exact formula
does not affect the O(n log n) worst-case guarantee.

### B.4 The pivot choice in std::sort implementations

GCC's introsort uses median-of-three for small subarrays and Tukey's ninther
(median of three medians of three) for large subarrays (n > _S_threshold=16).
Clang's libc++ uses median-of-three throughout.  Both choose the pivot from
elements already in the subarray, so a worst-case input for one implementation
may not be worst-case for another.  David Musser's original paper noted that
the ninther pivot makes an O(n^2) input much harder to construct; Meijer and
Sahnwaldt later proved that no comparison-based pivot strategy can simultaneously
be both worst-case optimal and unbiased on random inputs.

### B.5 In-place merge sort variants

The most practical in-place merge algorithm (Geffert, Katajainen, Pasanen 2000)
achieve O(n log n) comparisons and O(1) extra space but with large constants;
a simpler O(n log^2 n) rotate-based merge is used in GNU std::stable_sort when
allocating O(n) buffer memory fails.  Symmerge (Kim and Kutzner 2008) gives
O(n log^2 n) comparisons in O(1) space with a smaller constant than rotate.

### B.6 Cache effects in Heapsort

The sift-down operation in Heapsort accesses positions 2i+1 and 2i+2 at depth
d from the heap root, which for a heap of n elements jump ~n/2^d words apart.
For n=10^6 and a 64-byte cache line (8 doubles per line), the first few levels
fit in L1/L2 but levels below log2(L2_size/64) ≈ 11 cause L2 misses on every
access.  This is the primary reason Heapsort is 2-5x slower than Quicksort
for random data despite identical asymptotic complexity.

### B.7 Branch prediction and Quicksort

Quicksort's inner partition loop (while a[++i] < pivot; while a[--j] > pivot)
is highly branch-predictable on random uniform data: roughly 50% of elements
are less than the pivot, so the CPU's taken/not-taken predictor learns quickly.
Insertion sort's inner loop (while a[j] > key) is highly predictable for nearly
sorted data (rarely taken) and unpredictable for random data.  This is why
hybrid schemes using insertion sort only for small subarrays (where randomness
is less likely) work well in practice.

### B.8 Parallel Timsort / parallel merge

Java 8's Arrays.sort (for primitives) uses a parallel Dual-Pivot Quicksort that
divides the array into four partitions and sorts them on four threads using
ForkJoin, achieving roughly 2x speedup on 4 cores.  For objects (where stability
is required) Java uses a parallel Timsort; the merge step is parallelised using
a binary search to find the split point and recursing on both halves.

### B.9 Sorting networks

A sorting network is a fixed sequence of compare-and-swap operations independent
of the data.  An optimal sorting network for n=16 requires 60 comparators
(Green's 1969 construction); the AKS network achieves O(n log n) comparators
but with impractically large constants.  Sorting networks are used in SIMD
implementations (Chhugani et al. 2008 used SSE to sort 4 floats in 4 cycles)
and as the base case in GPU sorts.
"""

_APPENDIX_C = """
## Appendix C — Asymptotic and Average-Case Analysis

### C.1 The lower bound proof in detail

A comparison-based sort on n elements is modelled as a binary decision tree:
each internal node is a comparison a[i] < a[j] with two branches, and each
leaf is a permutation of the input.  There are n! permutations, so the tree has
at least n! leaves and height at least log2(n!).  By Stirling:
  log2(n!) = n log2(n) - n log2(e) + O(log n) ≈ n log2(n) - 1.4427 n.
Therefore any comparison sort needs at least n log2(n) - 1.4427 n comparisons
in the worst case.  Merge sort achieves n log2(n) - n + 1 comparisons in the
worst case (the standard two-way merge), which matches to within a constant.

### C.2 Average-case for Quicksort

With a uniformly random pivot, the expected number of comparisons for Quicksort
on a random permutation of n distinct elements satisfies the recurrence:
  C(n) = (n-1) + (2/n) * sum_{k=0}^{n-1} C(k),  C(0)=C(1)=0.
Solving: C(n) = 2(n+1)*H_n - 4n where H_n = sum_{k=1}^{n} 1/k ≈ ln n + 0.5772.
For large n, C(n) ≈ 2n ln n ≈ 1.386 n log2(n), about 39% more than merge sort's
worst case.  In practice, Quicksort is faster because its constants in the
time-per-comparison are smaller (cache-friendly sequential access, simple loop).

### C.3 Optimal sorting for small n

For n ≤ 5 elements an optimal comparison sort is known:
  n=1: 0 comparisons; n=2: 1; n=3: 3; n=4: 5; n=5: 7.
Insertion sort uses at most n(n-1)/2 comparisons: 0,1,3,6,10 for n=1..5.
For n=4 insertion sort uses up to 6 comparisons vs the optimal 5; for n=5 up
to 10 vs 7.  Library implementations use a hard-coded 5-element sort for the
base case of Timsort/Introsort (or the branchless SIMD network for n=4..8).

### C.4 Expected comparisons for Timsort on random data

On a uniformly random permutation Timsort's expected run length is 2 (the
expected length of a maximal ascending run in a random permutation is e-1 ≈ 1.72
≈ 2 after the reverse-run detection).  So Timsort extends nearly every run with
insertion sort up to minrun; the effective cost is dominated by the merge phase,
which is O(n log n) comparisons — the same as merge sort.  The advantage of
Timsort on random data over plain merge sort is only the insertion-sort speedup
on length-1 and length-2 runs, which is small; Timsort's advantage appears
almost entirely on structured (partially sorted) data.

### C.5 Why the sort-stability requirement matters for Timsort's adoption

Java added Timsort (as Arrays.sort for Object arrays) in 1.7 specifically because
Comparator-based sorts of objects must be stable to satisfy the contract of
collection sorting: if a.compareTo(b)==0 the relative order from the previous
sort should be preserved.  The C++ standard does not require std::sort to be
stable (hence Introsort); std::stable_sort is separate and may be slower when
memory is constrained.  The Go standard library switched to pdqsort (pattern-
defeating quicksort) in 1.19 for slices.Sort and keeps sort.Stable for the
stable variant.

### C.6 Pattern-defeating quicksort (pdqsort)

pdqsort (Orson Peters 2015) extends Introsort with:
  1. Median-of-three pivot with a "block partitioning" scheme that avoids
     branch mispredictions by processing elements in fixed-size blocks.
  2. A pattern-detection pass that recognises already-sorted, reverse-sorted,
     and organ-pipe inputs and handles them in linear time.
  3. The same depth-limit fallback to Heapsort as Introsort.
pdqsort achieves near-optimal performance on structured inputs while retaining
Quicksort's average-case performance on random inputs.  It is the algorithm
behind Rust's slice::sort_unstable and C++23's std::ranges::sort in several
implementations.

### C.7 External sorting and merge sort

For data that does not fit in RAM, merge sort generalises naturally to k-way
external merge: split the input into M runs that fit in memory, sort each run,
then merge them k at a time using a priority queue.  With M initial runs and
a k-way merge the number of passes is ceil(log_k(M)); each pass reads and
writes the entire dataset once.  For k=256 and M=10^6 runs only 3 passes are
needed.  The optimal k balances the number of passes against the merge-step
I/O cost; for modern NVMe SSDs a larger k (sequential reads) is preferred over
tape/disk (where seek minimisation was historically the goal).
"""

_APPENDIX_D = """
## Appendix D — Further Reading and Historical Notes

### D.1 History of Quicksort

C.A.R. Hoare invented Quicksort in 1959 while working on machine translation at
the Moscow State University and published it in 1962 ("Quicksort", The Computer
Journal, 5(1):10–16).  The algorithm was notable for its simplicity and for the
fact that Hoare discovered it while trying to sort words for a dictionary and
realised the merge-sort approach he had been taught was unnecessarily complicated
for the average case.  Sedgewick's 1977 thesis on Quicksort optimisations
(median-of-three pivot, small-subarray cutoff, tail-call elimination) established
most of the practical techniques still in use.

### D.2 History of Merge Sort

Merge sort is attributed to John von Neumann, who described it in 1945 as part
of an early design for the EDVAC computer.  The first published description
appears in Goldstine and von Neumann's 1948 planning codex.  The bottom-up
iterative variant was described by Maurice Wilkes, David Wheeler, and Stanley
Gill in "The Preparation of Programs for an Electronic Digital Computer" (1951).

### D.3 History of Heapsort

J.W.J. Williams described the heap data structure and Heapsort in 1964 ("Heapsort,"
CACM 7(6):347–348).  Robert Floyd's improvement (Floyd 1964, same issue) introduced
the O(n) heap-construction algorithm (the sift-down pass starting from n/2-1),
reducing the constant from the naive O(n log n) two-pass approach.

### D.4 History of Timsort

Tim Peters wrote the first Timsort implementation for Python in 2002.  The
algorithm was inspired by Galloping Sort (Jon Bentley and Doug McIlroy), which
itself drew on ideas from natural merge sort and binary insertion sort.  Joshua
Bloch ported Timsort to Java for Java 7 after discovering a bug in the existing
merge sort (a signed-integer overflow in the midpoint calculation of Arrays.sort).
The Timsort invariant for the run stack was later found to be subtly wrong by
de Gouw, Rot, de Boer, Bubel, and Hähnle (2015, "OpenJDK's Java.utils.Collection.sort()
is broken"); the fix is to use a three-way invariant instead of a two-way one.

### D.5 Key papers

  Hoare, C.A.R. (1962). "Quicksort." Computer Journal 5(1):10–16.
  Williams, J.W.J. (1964). "Heapsort." CACM 7(6):347–348.
  Floyd, R.W. (1964). "Treesort 3." CACM 7(12):701.
  Knuth, D.E. (1973). The Art of Computer Programming, Vol. 3: Sorting and Searching.
  Sedgewick, R. (1977). "Quicksort." PhD thesis, Stanford University.
  Musser, D.R. (1997). "Introspective Sorting and Selection Algorithms." SPE 27(8):983–993.
  Peters, T. (2002). "Timsort." Python source listsort.txt.
  McIlroy, M.D. (1999). "A Killer Adversary for Quicksort." SPE 29(4):341–344.
  Peters, O. (2015). "Pattern-defeating Quicksort." arXiv:1506.04228.
  Bloch, J. (2006). "Extra, Extra — Read All About It: Nearly All Binary Searches and
    Merge Sorts are Broken." Google Research Blog.

### D.6 Cache-oblivious models and van Emde Boas layout

The cache-oblivious model (Frigo, Leiserson, Prokop, Ramachandran 1999) analyses
algorithms without knowing the cache size M or block size B; an algorithm is
cache-oblivious if it is optimal for all M and B simultaneously.  The van Emde
Boas tree layout recursively splits a tree at the midpoint, storing the top half
then the bottom halves, achieving O(log_B n) cache misses per operation.  Applied
to merge sort, this gives the funnelsort construction.

### D.7 Sorting in databases

Databases use sort extensively for ORDER BY, GROUP BY, and hash-join probes.
External merge sort with replacement selection (heaps used to produce initial
runs longer than memory) was the standard from the 1960s.  Grace hash join
partitions both relations into B buckets fitting in memory and sorts each pair;
the sort dominates when the relation does not fit in two passes of B buckets.
Modern databases use a hybrid: in-memory Quicksort or pdqsort for small
relations and multi-way external merge for large ones, with SIMD-accelerated
partitioning (vectorised Quicksort) on modern CPUs.

### D.8 GPU sorting

GPU sorting algorithms exploit data parallelism differently from CPU algorithms:
  Bitonic sort: a sorting network with O(n log^2 n) compare-and-swap steps;
    each step is fully parallel (all comparators are independent).  Practical
    for n up to ~10^6 on a GPU.
  Radix sort on GPU: four passes of digit extraction + prefix-scan + scatter.
    The scatter is irregular and limits memory bandwidth utilisation; CUB
    (CUDA Unbound) implements a state-of-the-art GPU radix sort with
    warp-level digit histograms and a single-pass scatter.
  Merge sort on GPU: the merge step parallelises using a binary search for
    the split point; Thrust's merge sort achieves near-peak memory bandwidth.
  For most practical n (10^6 to 10^9), GPU radix sort for integers and GPU
    merge sort for comparison-based sorting outperform their CPU equivalents
    by 10–50x due to the order-of-magnitude difference in memory bandwidth.

### D.9 Practical sorting benchmarks (2024, x86-64, DDR5)

The following numbers are representative of modern in-memory sort performance
for arrays of 10^7 64-bit integers on a desktop CPU (Ryzen 9000-series, DDR5-6000,
GCC -O3), to give an order-of-magnitude sense of the differences:

  std::sort (introsort):   ~350 ms   ~28.6 MT/s
  std::stable_sort:        ~420 ms   ~23.8 MT/s
  Timsort (java.util):     ~440 ms   ~22.7 MT/s  (JVM, warm)
  pdqsort:                 ~280 ms   ~35.7 MT/s
  ska_sort (radix, int64): ~120 ms   ~83.3 MT/s
  std::sort SIMD (AVX-512):~200 ms   ~50.0 MT/s  (with vectorised partition)

These numbers are approximate and vary with data distribution, cache size, and
compiler version.  Radix sort's advantage grows with array size; for n=10^6
the gap narrows because the radix sort overheads (histogramming, scatter)
dominate.  For nearly-sorted data Timsort exceeds pdqsort by 3–10x.
"""

_APPENDIX_E = """
## Appendix E — Integer Sorting, Distribution Sort, and Hybrid Strategies

### E.1 Counting sort

Counting sort is applicable when keys are integers in a range [0, k).  It
allocates a count array C of size k, increments C[a[i]] for each element,
computes a prefix sum to find output positions, then scatters elements to their
correct positions in a second pass.  Time: O(n + k).  Space: O(n + k).  Stable.
For k = O(n) this is O(n); for k >> n the space cost dominates.

### E.2 Bucket sort

Bucket sort divides the range [0, 1) into n equal-width buckets, distributes
elements into them, sorts each bucket with insertion sort, and concatenates.
If the input is drawn from a uniform distribution the expected bucket size is
O(1) and the total expected time is O(n).  Worst case (all elements in one
bucket) is O(n^2).  For non-uniform distributions the bucket boundaries must
be chosen to balance load (adaptive bucket sort or sample sort).

### E.3 American flag sort

American flag sort is an in-place variant of radix sort: it performs two passes
per digit, the first to count occurrences and compute prefix sums, the second
to scatter in-place using a cycle-leader approach (like the in-place version of
counting sort).  It is MSD (most-significant digit first) and recursive;
buckets below a threshold are sorted with insertion sort.  It uses O(k) extra
space for the counts (k = radix, typically 256) but O(1) for the scatter.

### E.4 Spreadsort

Spreadsort (Steven Ross 2002) is a hybrid that chooses between comparison-based
and radix-based sorting depending on the data.  It estimates whether radix sort
will outperform comparison sort based on the range of the data and the number
of elements; if the range is less than 2^(log2(n) + constant) it uses radix
sort, otherwise comparison sort.  Spreadsort is included in Boost.Sort.

### E.5 Flashsort

Flashsort (Karl-Dietrich Neubert 1998) is a distribution sort for floating-point
or integer data.  It classifies each element into one of m buckets (m ~ 0.1n),
counts the bucket sizes to compute permutation cycles, then performs in-place
cyclic permutation to move elements near their final positions, and finishes
with insertion sort.  Expected O(n) time for uniform distributions; O(n^2)
worst case.

### E.6 Comparison: choosing an algorithm in practice

Choosing a sorting algorithm in a new context:
  1. Are keys integers or fixed-width byte strings with bounded range?  If
     yes and n is large: radix sort or counting sort; likely 2-5x faster.
  2. Is stability required?  If yes: Timsort, merge sort, or counting sort.
     Eliminate Quicksort, Heapsort, and pdqsort.
  3. Is the data likely partially sorted?  If yes: Timsort.
  4. Is in-place required (O(1) extra space)?  If yes and stability not
     required: Heapsort or Introsort/pdqsort.  If stability required:
     in-place stable merge (Symmerge) at O(n log^2 n) cost.
  5. Is the data on external storage or distributed?  External merge sort
     or distributed sample sort.
  6. Otherwise: pdqsort or Introsort are good general-purpose defaults.
"""

_SORT_DOC_PADDED = _SORT_DOC + _APPENDIX_A + _APPENDIX_B + _APPENDIX_C + _APPENDIX_D + _APPENDIX_E

LONG_DOC_Q = _CHAT_TMPL.format(user=(
    _SORT_DOC_PADDED
    + "\n\nBased on the reference above, answer the following question in detail "
    "(at least four paragraphs, with examples):\n\n"
    "Compare Timsort and Introsort: in which practical scenarios does each "
    "outperform the other, what are the algorithmic reasons, and how does the "
    "choice of pivot strategy in Introsort affect its worst-case guarantee?"
))


def prompt_token_count_approx(text: str) -> int:
    """Rough token count estimate (characters / 3.5) without loading the tokenizer.
    The Qwen3 tokeniser produces ~3.5 chars/token on English technical prose.
    This estimate is used only for CI threshold checks; live runs verify the
    real count with the actual tokeniser."""
    return max(1, int(len(text) / 3.5))
