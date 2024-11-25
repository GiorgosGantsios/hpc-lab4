#include <cuda.h>
#include <device_launch_parameters.h>

__global__ void histogramKernel(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin) {
    // Shared memory for block-local histogram
    extern __shared__ int shared_hist[];

    // Initialize shared histogram to 0
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tIdx = threadIdx.x;
    for (int i = tIdx; i < nbr_bin; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();

    // Process the image
    if (tid < img_size) {
        atomicAdd(&shared_hist[img_in[tid]], 1);
    }
    __syncthreads();

    // Reduce shared histogram to global histogram
    for (int i = tIdx; i < nbr_bin; i += blockDim.x) {
        atomicAdd(&hist_out[i], shared_hist[i]);
    }
}
