//go:build darwin

// GPU matmul via Metal Performance Shaders (MPSMatrixMultiplication), the
// same tuned kernel Julia's Metal.jl and PyTorch's MPS backend go through
// -- not a hand-written compute shader. See metal_matmul_darwin.m for the
// Objective-C bridge; cgo compiles it automatically as part of this
// package on Darwin.
package main

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
#include "metal_matmul_darwin.h"
*/
import "C"
import (
	"fmt"
	"time"
	"unsafe"
)

var metalAvailable = true

// benchMetalF32 returns GPU kernel execution time (excludes allocation,
// upload, and readback) and whether a Metal device was found.
func benchMetalF32(n int) (time.Duration, bool) {
	a := randMatrixF32(n)
	b := randMatrixF32(n)
	c := make([]float32, n*n)

	ns := C.mps_matmul_f32(
		C.int(n),
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&c[0])),
	)
	if ns < 0 {
		return 0, false
	}
	return time.Duration(ns), true
}

func runMetalBenchmarks() {
	fmt.Println()
	fmt.Println("=== Metal Performance Shaders (MPSMatrixMultiplication, float32) ===")
	for _, n := range []int{500, 1000, 2000, 4096, 8192} {
		d, ok := benchMetalF32(n)
		if !ok {
			fmt.Println("no Metal device found")
			return
		}
		fmt.Printf("mps   f32  n=%-5d  %10s  %8.3f GFLOP/s\n", n, d.Round(time.Millisecond), gflops(n, d))
	}
}
