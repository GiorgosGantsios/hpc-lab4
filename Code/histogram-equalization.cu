#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

__global__ void histogramGPU(int * hist_out, unsigned char * img_in, int imageW, int imageH) {
    extern __shared__ int sharedMemory[];
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    if (tx < 256) {
        sharedMemory[tx] = 0;
    }

    __syncthreads();

    if (index < imageH*imageW)  {
        atomicAdd(&sharedMemory[img_in[index]], 1);
        __syncthreads();
        atomicAdd(&hist_out[tx], sharedMemory[tx]);
    }
    __syncthreads();
}

texture<int, cudaTextureType1D, cudaReadModeElementType> texRef; // Bind the 1D texture

__global__ void histogram_equalization_GPU(unsigned char * img_out, unsigned char * img_in, int * lut, int imageW, int imageH) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int y = index / imageW; // row
    int x = index % imageW; // col
    
    /* Get the result image */
    if ((y * imageW + x) < imageW * imageH)  {
        img_out[index] = tex1Dfetch(texRef, img_in[index]);
    }
}

int histogram_equalization_prep(unsigned char * img_out, unsigned char * img_in, int * hist_in, int imageW, int imageH, int nbr_bin, unsigned char * d_ImgIn) {
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d, index, *d_lut;
    int img_size = imageW * imageH;
    float millisecondsTransfers = 0;
    int extra_block = ((imageH*imageW)%256 != 0);
    cudaEvent_t startCuda, stopCuda;

    // Construct the LUT by calculating the CDF
    cdf = 0;
    min = 0;
    i = 0;
    
    // Finds the first value on the Histogram that isn't 0
    while(min == 0) {
        min = hist_in[i++];
    }
    index = i-1;

    // Calculate the look up table (lut)
    for (i = 0; i < index + 1; i++)  {
        lut[i] = 0;
    }

    d = img_size - min;
    for(i = index; i < nbr_bin; i++) {
        cdf += hist_in[i];
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
    }
    for(i = 0; i < nbr_bin; i++)  {
        if(lut[i] > 255) {
            lut[i] = 255;
        }
    }

    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    cudaEventRecord(startCuda, 0);

    cudaError_t err = cudaMalloc((void **)&d_lut, sizeof(int)*nbr_bin);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (lut) free(lut);
        return(-1);
    }

    cudaMemcpy(d_lut, lut, sizeof(int)*nbr_bin, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Memcopy error: %s\n", cudaGetErrorString(err));
        if (d_lut) cudaFree(d_lut);
        if (lut) free(lut);
        return(-1);
    }

    cudaBindTexture(0, texRef, d_lut, 256 * sizeof(int));

    histogram_equalization_GPU<<<(img_size/1024)+extra_block, 1024, 256* sizeof(int)>>>(img_out, d_ImgIn, d_lut, imageW, imageH);
    cudaDeviceSynchronize(); 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        if (d_lut) cudaFree(d_lut);
        if (lut) free(lut);
        return(-1);
    }
    cudaFree(d_lut);

    cudaEventRecord(stopCuda, 0);
    cudaEventSynchronize(stopCuda);
    cudaUnbindTexture(texRef);
    cudaEventElapsedTime(&millisecondsTransfers, startCuda, stopCuda);

    free(lut);
    return(millisecondsTransfers);
}