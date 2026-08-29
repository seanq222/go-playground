//go:build !(darwin && cgo)

package main

var metalAvailable = false

func runMetalBenchmarks() {}
