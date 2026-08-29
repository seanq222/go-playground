# go-playground

Go benchmarks, in the same spirit as `julia-playground`, `jax-playground`,
`pytorch-playground`, and `rust-playground`.

## matmul

[`src/matmul.go`](src/matmul.go) benchmarks dense square matrix multiply.
Go has no built-in BLAS, so it compares:

- **naive**: a plain triple-nested loop, single-threaded, no vectorization
  -- what you get from Go with zero dependencies.
- **gonum**: [`gonum.org/v1/gonum/mat`](https://pkg.go.dev/gonum.org/v1/gonum/mat)
  (float64, via `mat.Dense.Mul`) and
  [`gonum.org/v1/gonum/blas/blas32`](https://pkg.go.dev/gonum.org/v1/gonum/blas/blas32)
  (float32, via `Gemm`) -- a pure-Go BLAS implementation with
  assembly-optimized, multi-threaded kernels for amd64/arm64. Not as fast
  as a tuned vendor BLAS.
- **vendor BLAS** (macOS/Linux, requires cgo): [`gonum.org/v1/netlib/blas/netlib`](https://pkg.go.dev/gonum.org/v1/netlib/blas/netlib),
  a cgo wrapper around any CBLAS-compatible C library. This is real
  vendor-tuned BLAS, the closest Go equivalent to what
  Julia/NumPy/PyTorch get from OpenBLAS/MKL/Accelerate.
  - On **macOS**: linked against the built-in Accelerate framework, no
    install needed. ~10x gonum's throughput for float64 and ~30-40x for
    float32 on an M3 Max (likely hitting the AMX matrix coprocessor).
  - On **Linux**: linked against a real OpenBLAS build. Not the system
    package (would need root/apt) -- built from source into
    `~/openblas` with `NOFORTRAN=1 NO_LAPACK=1` (BLAS only, no Fortran
    compiler needed). Deliberately *not* the same `.so` Julia uses on
    that machine: Julia's bundled OpenBLAS is an **ILP64** build
    (exports `cblas_dgemm64_`, 64-bit integer args), while gonum's
    netlib wrapper expects the standard **LP64** interface (plain
    `cblas_dgemm`, 32-bit int args) -- those calling conventions aren't
    ABI-compatible, so a from-source LP64 build was needed instead. Same
    OpenBLAS version (0.3.29) either way.

```
go run src/matmul.go

# on macOS, to include the vendor-BLAS benchmarks:
CGO_LDFLAGS="-framework Accelerate" go run ./src

# on Linux, against a from-source OpenBLAS install:
CGO_LDFLAGS="-L$HOME/openblas/lib -lopenblas" \
  LD_LIBRARY_PATH="$HOME/openblas/lib" \
  go run ./src
```

On asymmetric-core Linux machines (e.g. spark's Cortex-X925/A725 hybrid
design), also set `OPENBLAS_NUM_THREADS` to the number of *fast* cores
only -- OpenBLAS's default of using every detected core measurably hurts
throughput there (30-40% slower) since the slow efficiency cores become
stragglers. See [`RESULTS.md`](RESULTS.md) for the numbers.

- **mps** (macOS only): GPU matmul via
  [`MPSMatrixMultiplication`](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
  (Metal Performance Shaders) -- the same tuned kernel Julia's `Metal.jl`
  and PyTorch's MPS backend go through, not a hand-written compute
  shader. No Go Metal binding is mature/stable enough yet to depend on, so
  this is a small self-contained cgo + Objective-C bridge
  ([`metal_matmul_darwin.m`](src/metal_matmul_darwin.m)), the same pattern
  as the Accelerate benchmark above. Only reported number is GPU kernel
  execution time (excludes allocation/upload/readback). On an M3 Max this
  reaches ~10 TFLOP/s at n=8192 -- well beyond Accelerate's CPU/AMX
  numbers, as expected for GPU vs. CPU.

```
CGO_LDFLAGS="-framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation" go run ./src
```
