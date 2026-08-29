#import "metal_matmul_darwin.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

long long mps_matmul_f32(int n, const float* a, const float* b, float* c) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return -1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        NSUInteger rowBytes = (NSUInteger)n * sizeof(float);
        NSUInteger dataSize = (NSUInteger)n * (NSUInteger)n * sizeof(float);

        id<MTLBuffer> bufA = [device newBufferWithBytes:a length:dataSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:b length:dataSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:dataSize options:MTLResourceStorageModeShared];

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:n rowBytes:rowBytes dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:n rowBytes:rowBytes dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:n rowBytes:rowBytes dataType:MPSDataTypeFloat32];

        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                          transposeLeft:NO
                                                                         transposeRight:NO
                                                                             resultRows:n
                                                                          resultColumns:n
                                                                        interiorColumns:n
                                                                                  alpha:1.0
                                                                                   beta:0.0];

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        [mm encodeToCommandBuffer:cmdBuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        double gpuSeconds = cmdBuf.GPUEndTime - cmdBuf.GPUStartTime;

        memcpy(c, bufC.contents, dataSize);

        return (long long)(gpuSeconds * 1e9);
    }
}
