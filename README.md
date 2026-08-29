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
