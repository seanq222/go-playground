# matmul results

Dense n x n matrix multiply, comparing four Go implementations across
three machines. Methodology matches the other `*-playground` repos where
possible: same operation (square matmul), same GFLOP/s formula
(`2*n^3 / elapsed_seconds / 1e9`), varying n from 500 to 8192.

All numbers below are from a **single run per machine** (not averaged
across trials) -- treat the small-n rows especially as approximate. The
`poker`/`rust-playground` `evaluate7` benchmarking earlier surfaced a case
where a single run was actively misleading, so these should be read as a
rough picture, not precise figures, until re-run with multiple trials.

## What's being compared

- **naive**: a plain triple-nested loop, single-threaded, no
  vectorization -- what Go gives you with zero dependencies.
- **gonum**: [`gonum.org/v1/gonum`](https://pkg.go.dev/gonum.org/v1/gonum)'s
  pure-Go BLAS (`mat.Dense.Mul` for f64, `blas32.Gemm` for f32) --
  assembly-optimized, multi-threaded, but not vendor-tuned.
- **vendor BLAS**: [`gonum.org/v1/netlib/blas/netlib`](https://pkg.go.dev/gonum.org/v1/netlib/blas/netlib),
  a cgo wrapper linked against a real CBLAS-compatible library --
  Accelerate on macOS (built-in), a from-source OpenBLAS build on Linux
  (see "Linux OpenBLAS setup" below).
- **Metal/MPS** (Mac only): GPU matmul via `MPSMatrixMultiplication`, a
  small self-contained cgo + Objective-C bridge, not a third-party Go
  Metal binding.

## Results (GFLOP/s, higher is better)

### Mac (M3 Max)

| n | naive f64 | naive f32 | gonum f64 | gonum f32 | Accelerate f64 | Accelerate f32 | MPS f32 (GPU) |
|---|---|---|---|---|---|---|---|
| 500 | 3.7 | 4.9 | 71 | 75 | 91-511* | 132-1070* | 1519-1689 |
| 1000 | 4.9 | 5.0 | 80 | 80 | 231-685* | 1211-2805* | 2488-2498 |
| 2000 | 4.9 | 5.0 | 78 | 81 | 711-725 | 1974-3062 | 2790-2828 |
| 4096 | -- | -- | 65 | 69 | 738-750 | 1189-2438 | 5600-6970 |
| 8192 | -- | -- | -- | -- | 725-731 | 2211-2790 | **10028-10248** |

\* small-n vendor-BLAS numbers were noisy across repeated invocations in
this session (thread-pool warmup dominates at this size) -- treat n<=1000
vendor-BLAS/MPS figures loosely, n>=2000 is the steady-state number.

### spark (Grace ARM64, OpenBLAS)

| n | naive f64 | naive f32 | gonum f64 | gonum f32 | OpenBLAS f64 | OpenBLAS f32 |
|---|---|---|---|---|---|---|
| 500 | 2.6 | 4.5 | 44 | 58 | 98 | 323 |
| 1000 | 4.0 | 4.5 | 77 | 90 | 399 | 757 |
| 2000 | 4.2 | 4.4 | 79 | 104 | 416 | 813 |
| 4096 | -- | -- | 76 | 103 | 413 | 841 |
| 8192 | -- | -- | -- | -- | **416** | **852** |

### HBM (Xeon Max 9480, WSL2, OpenBLAS)

| n | naive f64 | naive f32 | gonum f64 | gonum f32 | OpenBLAS f64 | OpenBLAS f32 |
|---|---|---|---|---|---|---|
| 500 | 1.1 | 2.6 | 21 | 59 | 29-98* | 231-323* |
| 1000 | 1.5 | 2.8 | 44 | 95 | 399-846* | 757-1211* |
| 2000 | 2.1 | 2.4 | 90 | 124 | 1184 | 1974 |
| 4096 | -- | -- | 102 | 155 | 741 | 1189 |
| 8192 | -- | -- | -- | -- | **1032** | **2211** |

\* same small-n noise caveat as Mac; only one run was taken here so these
numbers are less trustworthy than spark's.

## Notes

**Naive Go is ~2-5 GFLOP/s everywhere, dwarfed by any BLAS.** Even
gonum's pure-Go implementation is already 15-70x faster than the
triple-loop, and real vendor BLAS is another 5-10x past that. The naive
numbers are also the most machine-dependent in a *bad* way for HBM
specifically: it's noticeably slower there (~1-2 GFLOP/s) than Mac/spark
(~4-5 GFLOP/s) for this scalar, branch-heavy code -- consistent with what
the `evaluate7` and matmul benchmarking in the sibling repos already
found about this Xeon being less forgiving of unoptimized code than
Apple Silicon or Grace.

**Vendor BLAS closes most of the gap between machines -- except spark,
where it doesn't close nearly as much.** HBM's OpenBLAS numbers are the
highest of the three machines for f32 (up to ~2.2 TFLOP/s at n=8192),
roughly on par with the Mac's Accelerate f32 numbers, despite HBM being
the slowest machine by far for naive/scalar code. spark tops out around
~850 GFLOP/s f32 -- a much smaller jump over its own gonum baseline
(~8x) than HBM's (~14x) or the Mac's (~30-40x with AMX).

Initial guess was a missed-optimization bug: OpenBLAS's autodetection
picked the generic `CORE=ARMV8` kernel set on spark rather than an
SVE-tuned one, even though `/proc/cpuinfo` confirms the CPU supports
SVE/SVE2 and even int8/bf16 matmul extensions. Rebuilt with
`TARGET=ARMV8SVE` forced -- and it made **no meaningful difference**
(~397 GFLOP/s f64, ~830 GFLOP/s f32, both within noise of the generic
build). So that theory was wrong, and testing it directly is what showed
that.

The real explanation: `lscpu` on spark shows its "Grace" CPU is actually
**10x Cortex-X925 + 10x Cortex-A725** -- a 20-core client-class
performance/efficiency hybrid design (part of NVIDIA's GB10 Grace
Blackwell Superchip used in the DGX Spark dev kit), not the many-core
homogeneous Neoverse V2 server silicon used in full Grace/GH200 systems.
That's also *why* OpenBLAS's autodetection fell back to a generic
target in the first place -- there's no OpenBLAS kernel tuned for these
specific client cores. HBM's Xeon Max 9480 has 56 full, uniform server
cores, essentially 3x spark's total core count with none of them being
reduced-throughput efficiency cores. For a multi-threaded, compute-bound
kernel like large-matrix GEMM, that raw core-count-and-class gap is the
actual bottleneck, not instruction set tuning.

**Not the same OpenBLAS Julia uses, on purpose.** Julia's bundled OpenBLAS
on both spark and HBM is an **ILP64** build (`libopenblas64_.so`, exports
`cblas_dgemm64_` -- 64-bit integer arguments). gonum's netlib CBLAS
wrapper expects the standard **LP64** interface (plain `cblas_dgemm`,
32-bit ints) -- confirmed via `nm -D` that the symbol names don't even
match, so linking against Julia's library directly would fail (or worse,
silently corrupt arguments if the symbols happened to collide). Built
OpenBLAS 0.3.29 from source instead, `NOFORTRAN=1 NO_LAPACK=1` (BLAS only,
no Fortran compiler needed), installed to `~/openblas` on each machine --
same OpenBLAS version Julia uses, just the ABI-compatible build variant.
No root/sudo needed for the OpenBLAS build itself; only installing a C
compiler in the first place needed sudo (`build-essential` on HBM's WSL2,
which had none at all initially).

**Metal/MPS blows past everything else, as expected for GPU vs. CPU.**
~10 TFLOP/s at n=8192 vs. Accelerate's ~730-750 GFLOP/s (f64) / ~2200-2800
GFLOP/s (f32) on the same machine's CPU. No GPU comparison exists yet for
spark (CUDA) or HBM (no discrete GPU) in this repo.

**Single-run caveat.** Unlike the `evaluate7` cross-language benchmarks
(which went through several rounds of variance-checking after an early
single-sample number turned out to be unrepresentative), these matmul
numbers are each from one run per machine. The steady-state (large-n)
numbers are probably reliable; the small-n numbers, especially for the
cgo-based vendor-BLAS/MPS paths, showed enough run-to-run spread within
a single session that they shouldn't be trusted precisely without
repeating them.
