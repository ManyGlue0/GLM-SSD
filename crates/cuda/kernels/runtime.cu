// Runtime helpers: device info, memory, streams, events. Thin wrappers so the Rust side never
// has to link against CUDA headers directly.
#include "common.cuh"
#include <string.h>

struct GsDeviceInfo {
    char     name[256];
    int      cc_major, cc_minor;
    int      sm_count;
    uint64_t total_mem, free_mem;
    int      pci_bus, pci_device;
    int      async_engines;
    int      can_map_host;
};

GS_EXPORT int gs_device_info(int dev, GsDeviceInfo *out) {
    cudaDeviceProp p;
    GS_CHECK(cudaGetDeviceProperties(&p, dev));
    GS_CHECK(cudaSetDevice(dev));
    size_t fr = 0, tot = 0;
    GS_CHECK(cudaMemGetInfo(&fr, &tot));
    memset(out, 0, sizeof(*out));
    strncpy(out->name, p.name, sizeof(out->name) - 1);
    out->cc_major = p.major;
    out->cc_minor = p.minor;
    out->sm_count = p.multiProcessorCount;
    out->total_mem = tot;
    out->free_mem = fr;
    out->pci_bus = p.pciBusID;
    out->pci_device = p.pciDeviceID;
    out->async_engines = p.asyncEngineCount;
    out->can_map_host = p.canMapHostMemory;
    return 0;
}

GS_EXPORT const char *gs_error_string(int e) { return cudaGetErrorString((cudaError_t) e); }

GS_EXPORT int gs_malloc(void **p, size_t n) { return (int) cudaMalloc(p, n); }
GS_EXPORT int gs_free(void *p) { return (int) cudaFree(p); }
GS_EXPORT int gs_host_alloc(void **p, size_t n) { return (int) cudaHostAlloc(p, n, cudaHostAllocDefault); }
GS_EXPORT int gs_host_free(void *p) { return (int) cudaFreeHost(p); }
GS_EXPORT int gs_host_register(void *p, size_t n) { return (int) cudaHostRegister(p, n, cudaHostRegisterDefault); }
GS_EXPORT int gs_host_unregister(void *p) { return (int) cudaHostUnregister(p); }

GS_EXPORT int gs_stream_create(void **s) {
    return (int) cudaStreamCreateWithFlags((cudaStream_t *) s, cudaStreamNonBlocking);
}
GS_EXPORT int gs_stream_destroy(void *s) { return (int) cudaStreamDestroy((cudaStream_t) s); }
GS_EXPORT int gs_stream_sync(void *s) { return (int) cudaStreamSynchronize((cudaStream_t) s); }
GS_EXPORT int gs_device_sync() { return (int) cudaDeviceSynchronize(); }
// Destroy the CUDA context (frees everything the process holds on the GPU). All CUDA objects
// must already be released; the next CUDA call creates a fresh context.
GS_EXPORT int gs_device_reset() { return (int) cudaDeviceReset(); }

GS_EXPORT int gs_memcpy_h2d(void *dst, const void *src, size_t n, void *s) {
    return (int) cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, (cudaStream_t) s);
}
GS_EXPORT int gs_memcpy_d2h(void *dst, const void *src, size_t n, void *s) {
    return (int) cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, (cudaStream_t) s);
}
GS_EXPORT int gs_memcpy_d2d(void *dst, const void *src, size_t n, void *s) {
    return (int) cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, (cudaStream_t) s);
}

GS_EXPORT int gs_event_create(void **e) {
    return (int) cudaEventCreateWithFlags((cudaEvent_t *) e, cudaEventDisableTiming);
}
GS_EXPORT int gs_event_create_timed(void **e) { return (int) cudaEventCreate((cudaEvent_t *) e); }
GS_EXPORT int gs_event_destroy(void *e) { return (int) cudaEventDestroy((cudaEvent_t) e); }
GS_EXPORT int gs_event_record(void *e, void *s) { return (int) cudaEventRecord((cudaEvent_t) e, (cudaStream_t) s); }
GS_EXPORT int gs_event_sync(void *e) { return (int) cudaEventSynchronize((cudaEvent_t) e); }
GS_EXPORT int gs_event_query(void *e) { return (int) cudaEventQuery((cudaEvent_t) e); }
GS_EXPORT int gs_stream_wait_event(void *s, void *e) {
    return (int) cudaStreamWaitEvent((cudaStream_t) s, (cudaEvent_t) e, 0);
}
GS_EXPORT int gs_event_elapsed_ms(float *ms, void *a, void *b) {
    return (int) cudaEventElapsedTime(ms, (cudaEvent_t) a, (cudaEvent_t) b);
}

// ---- smoke test kernel ---------------------------------------------------------------
__global__ void k_saxpy(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

GS_EXPORT int gs_saxpy(int n, float a, const float *x, float *y, void *s) {
    k_saxpy<<<(n + 255) / 256, 256, 0, (cudaStream_t) s>>>(n, a, x, y);
    return (int) cudaGetLastError();
}

GS_EXPORT int gs_memset(void *p, int v, size_t n, void *s) { return (int) cudaMemsetAsync(p, v, n, (cudaStream_t) s); }
