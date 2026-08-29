//go:build !darwin

package main

import "time"

var accelerateAvailable = false
var metalAvailable = false

func benchAccelerateF64(n int) time.Duration { return 0 }
func benchAccelerateF32(n int) time.Duration { return 0 }
func runMetalBenchmarks()                    {}
