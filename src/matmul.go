// Dense matrix multiply benchmark, in the same spirit as the matmul
// benchmarks in julia-playground/jax-playground/pytorch-playground: fill
// two square matrices with random values, multiply them, and report
// GFLOP/s. Go has no built-in BLAS, so this compares two implementations:
//
//   - naive: a plain triple-nested loop, single-threaded, no vectorization.
//     This is "what you get from Go with zero dependencies."
//   - gonum: gonum.org/v1/gonum/mat, which uses a pure-Go BLAS
//     implementation with assembly-optimized kernels for amd64/arm64.
//     Multi-threaded internally. The closest Go equivalent to what
//     Julia/NumPy/PyTorch get for free from OpenBLAS/MKL, though gonum's
//     pure-Go BLAS is not as fast as a tuned vendor BLAS.
package main

import (
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/mat"
)

func gflops(n int, elapsed time.Duration) float64 {
	flops := 2.0 * float64(n) * float64(n) * float64(n)
	return flops / elapsed.Seconds() / 1e9
}

func randMatrixF64(n int) []float64 {
	m := make([]float64, n*n)
	for i := range m {
		m[i] = rand.Float64()
	}
	return m
}

func randMatrixF32(n int) []float32 {
	m := make([]float32, n*n)
	for i := range m {
		m[i] = rand.Float32()
	}
	return m
}

// naiveMatMulF64 computes c = a * b for n x n row-major matrices with a
// plain triple loop (ikj order, so the innermost loop walks b and c
// row-major for better cache behavior than the textbook ijk order).
func naiveMatMulF64(n int, a, b []float64) []float64 {
	c := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for k := 0; k < n; k++ {
			aik := a[i*n+k]
			for j := 0; j < n; j++ {
				c[i*n+j] += aik * b[k*n+j]
			}
		}
	}
	return c
}

func naiveMatMulF32(n int, a, b []float32) []float32 {
	c := make([]float32, n*n)
	for i := 0; i < n; i++ {
		for k := 0; k < n; k++ {
			aik := a[i*n+k]
			for j := 0; j < n; j++ {
				c[i*n+j] += aik * b[k*n+j]
			}
		}
	}
	return c
}

func benchNaiveF64(n int) time.Duration {
	a := randMatrixF64(n)
	b := randMatrixF64(n)
	start := time.Now()
	_ = naiveMatMulF64(n, a, b)
	return time.Since(start)
}

func benchNaiveF32(n int) time.Duration {
	a := randMatrixF32(n)
	b := randMatrixF32(n)
	start := time.Now()
	_ = naiveMatMulF32(n, a, b)
	return time.Since(start)
}

func benchGonumF64(n int) time.Duration {
	a := mat.NewDense(n, n, randMatrixF64(n))
	b := mat.NewDense(n, n, randMatrixF64(n))
	c := mat.NewDense(n, n, nil)
	start := time.Now()
	c.Mul(a, b)
	return time.Since(start)
}

// benchGonumF32 uses gonum's lower-level blas32.Gemm (a real Sgemm-style
// call), since mat.Dense is float64-only -- there is no float32 Dense type
// in current gonum.
func benchGonumF32(n int) time.Duration {
	a := blas32.General{Rows: n, Cols: n, Stride: n, Data: randMatrixF32(n)}
	b := blas32.General{Rows: n, Cols: n, Stride: n, Data: randMatrixF32(n)}
	c := blas32.General{Rows: n, Cols: n, Stride: n, Data: make([]float32, n*n)}
	start := time.Now()
	blas32.Gemm(blas.NoTrans, blas.NoTrans, 1, a, b, 0, c)
	return time.Since(start)
}

func main() {
	fmt.Println("=== naive triple-loop matmul ===")
	for _, n := range []int{500, 1000, 2000} {
		d64 := benchNaiveF64(n)
		fmt.Printf("naive f64  n=%-5d  %10s  %8.3f GFLOP/s\n", n, d64.Round(time.Millisecond), gflops(n, d64))
	}
	for _, n := range []int{500, 1000, 2000} {
		d32 := benchNaiveF32(n)
		fmt.Printf("naive f32  n=%-5d  %10s  %8.3f GFLOP/s\n", n, d32.Round(time.Millisecond), gflops(n, d32))
	}

	fmt.Println()
	fmt.Println("=== gonum BLAS (mat.Dense.Mul for f64, blas32.Gemm for f32) ===")
	for _, n := range []int{500, 1000, 2000, 4096} {
		d := benchGonumF64(n)
		fmt.Printf("gonum f64  n=%-5d  %10s  %8.3f GFLOP/s\n", n, d.Round(time.Millisecond), gflops(n, d))
	}
	for _, n := range []int{500, 1000, 2000, 4096} {
		d := benchGonumF32(n)
		fmt.Printf("gonum f32  n=%-5d  %10s  %8.3f GFLOP/s\n", n, d.Round(time.Millisecond), gflops(n, d))
	}

	if accelerateAvailable {
		fmt.Println()
		fmt.Println("=== Accelerate (vendor BLAS via cgo) ===")
		for _, n := range []int{500, 1000, 2000, 4096, 8192} {
			d := benchAccelerateF64(n)
			fmt.Printf("accel f64  n=%-5d  %10s  %8.3f GFLOP/s\n", n, d.Round(time.Millisecond), gflops(n, d))
		}
		for _, n := range []int{500, 1000, 2000, 4096, 8192} {
			d := benchAccelerateF32(n)
			fmt.Printf("accel f32  n=%-5d  %10s  %8.3f GFLOP/s\n", n, d.Round(time.Millisecond), gflops(n, d))
		}
	}

	runMetalBenchmarks()
}
