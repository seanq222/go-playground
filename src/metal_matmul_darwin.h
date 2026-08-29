#ifndef METAL_MATMUL_DARWIN_H
#define METAL_MATMUL_DARWIN_H

#ifdef __cplusplus
extern "C" {
#endif

// Multiplies two n x n row-major float32 matrices a and b via
// MPSMatrixMultiplication, writing the result into c (also n x n
// row-major, pre-allocated by the caller). Returns GPU execution time in
// nanoseconds (kernel execution only, not allocation/upload/readback), or
// -1 if no Metal device is available.
long long mps_matmul_f32(int n, const float* a, const float* b, float* c);

#ifdef __cplusplus
}
#endif

#endif
