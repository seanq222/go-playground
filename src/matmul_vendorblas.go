//go:build (darwin || linux) && cgo

// Vendor BLAS via cgo (gonum.org/v1/netlib/blas/netlib), linked against
// whatever CBLAS-compatible library CGO_LDFLAGS points at. On macOS
// that's the built-in Accelerate framework (no install needed); on Linux
// it's a real OpenBLAS build. Build/run with, e.g.:
//
//	CGO_LDFLAGS="-framework Accelerate" go run ./src                        # macOS
//	CGO_LDFLAGS="-L$HOME/openblas/lib -lopenblas" go run ./src              # Linux
package main

import (
	"time"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/netlib/blas/netlib"
)

func benchVendorBLASF64(n int) time.Duration {
	a := randMatrixF64(n)
	b := randMatrixF64(n)
	c := make([]float64, n*n)
	start := time.Now()
	netlib.Implementation{}.Dgemm(blas.NoTrans, blas.NoTrans, n, n, n, 1, a, n, b, n, 0, c, n)
	return time.Since(start)
}

func benchVendorBLASF32(n int) time.Duration {
	a := randMatrixF32(n)
	b := randMatrixF32(n)
	c := make([]float32, n*n)
	start := time.Now()
	netlib.Implementation{}.Sgemm(blas.NoTrans, blas.NoTrans, n, n, n, 1, a, n, b, n, 0, c, n)
	return time.Since(start)
}

var vendorBLASAvailable = true
