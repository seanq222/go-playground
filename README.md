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
- **accelerate** (macOS only): [`gonum.org/v1/netlib/blas/netlib`](https://pkg.go.dev/gonum.org/v1/netlib/blas/netlib),
  a cgo wrapper around any CBLAS-compatible C library, linked here against
  macOS's built-in Accelerate framework -- no separate OpenBLAS install
  needed. This is real vendor-tuned BLAS, the closest Go equivalent to what
  Julia/NumPy/PyTorch get from OpenBLAS/MKL/Accelerate. On an M3 Max this
  is ~10x gonum's throughput for float64 and ~30-40x for float32 (likely
  hitting the AMX matrix coprocessor).

```
go run src/matmul.go

# on macOS, to include the Accelerate-backed benchmarks:
CGO_LDFLAGS="-framework Accelerate" go run ./src
```

On Linux, the same `gonum.org/v1/netlib/blas/netlib` package can link
against a real OpenBLAS install instead
(`CGO_LDFLAGS="-L/path/to/openblas -lopenblas"`) -- not wired up here yet
since this Mac has no OpenBLAS installed to test against.

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
