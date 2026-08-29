//go:build !((darwin || linux) && cgo)

package main

import "time"

var vendorBLASAvailable = false

func benchVendorBLASF64(n int) time.Duration { return 0 }
func benchVendorBLASF32(n int) time.Duration { return 0 }
