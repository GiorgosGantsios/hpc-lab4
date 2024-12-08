#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_GPU(PGM_IMG img_in)  {
    cudaEvent_t startCuda, stopCuda;
    PGM_IMG gpuResult;
    PGM_IMG result;
    float millisecondsTransfers = 0, time;
    int t_hist[256];
    int *d_hist;
    unsigned char * d_ImgIn;

    result.w = img_in.w;
    result.h = img_in.h;
    
    cudaMallocManaged(&gpuResult.img, result.w * result.h * sizeof(unsigned char));

    gpuResult.w = img_in.w;
    gpuResult.h = img_in.h;

    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    cudaEventRecord(startCuda);

    cudaError_t err = cudaMalloc((void**)&d_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        gpuResult.w = -1;
        return(gpuResult);
    }
    
    cudaMemset(d_hist, 0, sizeof(int) * 256);
    
    histogramGPU<<<((gpuResult.h*gpuResult.w)/256)+1, 256, 256*sizeof(int) >>>(d_hist, img_in.img, gpuResult.w, gpuResult.h);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        if (d_hist) cudaFree(d_hist);
        gpuResult.w = -1;
        return(gpuResult);
    }

    err = cudaMemcpy(t_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);  // Copy data from host to device
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        if (d_hist) cudaFree(d_hist);
        gpuResult.w = -1;
        return(gpuResult);
    }

    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&millisecondsTransfers, startCuda, stopCuda);

    time = histogram_equalization_prep(gpuResult.img, img_in.img, t_hist, gpuResult.w, gpuResult.h, 256, img_in.img);

    if (time == -1)  {
        if (gpuResult.img) cudaFree(gpuResult.img);
        if (d_hist) cudaFree(d_hist);
        gpuResult.w = -1;
        return(gpuResult);
    }

    time += millisecondsTransfers;

    cudaEventRecord(startCuda, 0);

    cudaFree(d_hist);

    cudaEventRecord(stopCuda, 0);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&millisecondsTransfers, startCuda, stopCuda);

    time += millisecondsTransfers;

    printf("\nGPU Execution time: %lf seconds\n", time/1000.0);

    return gpuResult;
}