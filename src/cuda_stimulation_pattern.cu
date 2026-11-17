#include "cuda_stimulation_pattern.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA Kernel: 1024-Channel Phased-Array Neural Injector with Lateral Inhibition Mesh
__global__ void retinal_stimulation_kernel(
    const float* __restrict__ on_filtered,
    const float* __restrict__ off_filtered,
    const float* __restrict__ dvs_events,
    float* __restrict__ stimulation_currents,
    int width,
    int height,
    float gain
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        float on_val = on_filtered[idx];
        float off_val = off_filtered[idx];
        float dvs_val = fabsf(dvs_events[idx]);

        // Lateral inhibition convolution (3x3 receptive field kernel)
        float spatial_sum = 0.0f;
        float weight_sum = 0.0f;

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float dist_sq = static_cast<float>(dx * dx + dy * dy);
                    float w = __expf(-dist_sq / 1.5f);
                    spatial_sum += (on_filtered[ny * width + nx] - off_filtered[ny * width + nx]) * w;
                    weight_sum += w;
                }
            }
        }

        float local_field = (weight_sum > 0.0f) ? (spatial_sum / weight_sum) : (on_val - off_val);
        
        // Phased-array current encoder
        float base_current_uA = 10.0f;
        float current_uA = base_current_uA + gain * (on_val * 35.0f - off_val * 20.0f + dvs_val * 15.0f + local_field * 5.0f);
        
        if (current_uA < 0.0f) current_uA = 0.0f;
        if (current_uA > 50.0f) current_uA = 50.0f; // Cybernetic safety cap
        
        stimulation_currents[idx] = current_uA;
    }
}

void launch_retinal_stimulation_kernel(
    const float* d_on_filtered,
    const float* d_off_filtered,
    const float* d_dvs_events,
    float* d_stimulation_currents,
    int width,
    int height,
    float gain,
    cudaStream_t stream
) {
    dim3 block(8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    retinal_stimulation_kernel<<<grid, block, 0, stream>>>(
        d_on_filtered,
        d_off_filtered,
        d_dvs_events,
        d_stimulation_currents,
        width,
        height,
        gain
    );
}
