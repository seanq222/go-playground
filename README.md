# go-playground

Go benchmarks, in the same spirit as `julia-playground`, `jax-playground`,
`pytorch-playground`, and `rust-playground`.

## matmul

[`src/matmul.go`](src/matmul.go) benchmarks dense square matrix multiply.
Go has no built-in BLAS, so it compares:

- **naive**: a plain triple-nested loop, single-threaded, no vectorization
  -- what you get from Go with zero dependencies.
- **gonum**: [`gonum.org/v1/gonum/mat`](https://pkg.go.dev/gonum.org/v1/gonum/mat),
  a pure-Go BLAS implementation with assembly-optimized, multi-threaded
  kernels for amd64/arm64. The closest Go equivalent to what
  Julia/NumPy/PyTorch get for free from OpenBLAS/MKL, though not as fast as
  a tuned vendor BLAS.

```
go run src/matmul.go
```
