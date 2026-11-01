#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// All exported functions return a cudaError_t as int (0 = success) so Rust can turn
// failures into errors without exceptions crossing the FFI boundary.
#define GS_EXPORT extern "C"

#define GS_CHECK(x)                                   \
    do {                                              \
        cudaError_t _e = (x);                         \
        if (_e != cudaSuccess) return (int) _e;       \
    } while (0)
