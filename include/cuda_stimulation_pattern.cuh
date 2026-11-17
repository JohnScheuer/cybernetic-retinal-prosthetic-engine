#ifndef CUDA_STIMULATION_PATTERN_CUH
#define CUDA_STIMULATION_PATTERN_CUH

#include <cuda_runtime.h>

void launch_retinal_stimulation_kernel(
    const float* d_on_filtered,
    const float* d_off_filtered,
    const float* d_dvs_events,
    float* d_stimulation_currents,
    int width,
    int height,
    float gain,
    cudaStream_t stream
);

#endif // CUDA_STIMULATION_PATTERN_CUH
