//go:build darwin

// Vendor BLAS via cgo, linked against macOS's built-in Accelerate
// framework (CBLAS-compatible, no separate OpenBLAS install needed).
// Build/run with:
//
//	CGO_LDFLAGS="-framework Accelerate" go run ./src
package main

import (
	"time"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/netlib/blas/netlib"
)

func benchAccelerateF64(n int) time.Duration {
	a := randMatrixF64(n)
	b := randMatrixF64(n)
	c := make([]float64, n*n)
	start := time.Now()
	netlib.Implementation{}.Dgemm(blas.NoTrans, blas.NoTrans, n, n, n, 1, a, n, b, n, 0, c, n)
	return time.Since(start)
}

func benchAccelerateF32(n int) time.Duration {
	a := randMatrixF32(n)
	b := randMatrixF32(n)
	c := make([]float32, n*n)
	start := time.Now()
	netlib.Implementation{}.Sgemm(blas.NoTrans, blas.NoTrans, n, n, n, 1, a, n, b, n, 0, c, n)
	return time.Since(start)
}

var accelerateAvailable = true
