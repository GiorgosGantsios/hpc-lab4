#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
/*#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            cleanup();
        } \
    } while (0)*/

/*void cleanup (void * gpuResult, )  {
    cudaFree(gpuResult);
}*/

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)  {
    PGM_IMG result;
    PGM_IMG gpuResult;
    int hist[256];
    int t_hist[256];
    int *d_hist;
    unsigned char * d_ImgIn;

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    gpuResult.w = img_in.w;
    gpuResult.h = img_in.h;
    cudaError_t err = cudaMalloc((void **)&gpuResult.img, gpuResult.w * gpuResult.h * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        free(result.img); 
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(result);
        //cleanup(gpuResult.img, result.img);
    }
    err = cudaMalloc((void **)&d_ImgIn, gpuResult.w * gpuResult.h * sizeof(unsigned char));
    /*cudaError_t err = cudaMemcpy(d_hist, hist, 256 * sizeof(int), cudaMemcpyHostToDevice);  // Copy data from host to device
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        free(result.img); 
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(result);
        //cleanup(gpuResult.img, result.img);
    }*/

    err = cudaMalloc((void**)&d_hist, 256 * sizeof(int));  // Allocate memory on the GPU

    
    //histogramGPU<<<1, 256>>>(d_hist);
    cudaMemset(d_hist, 0, sizeof(int) * 256);


    err = cudaMemcpy(d_ImgIn, img_in.img, gpuResult.w * gpuResult.h * sizeof(unsigned char), cudaMemcpyHostToDevice);  // Copy data from host to device

    histogramConstuctionGPU<<<((gpuResult.h*gpuResult.w)/1024)+1, 1024>>>(d_hist, d_ImgIn, gpuResult.w, gpuResult.h);

    err = cudaMemcpy(t_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);  // Copy data from host to device

    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

    printf("CPU\n");
    for (int i = 0; i < 256; i++)  {
        printf(" %d",hist[i]);
    }

    printf("\nGPU\n");
    for (int i = 0; i < 256; i++)  {
        printf(" %d",t_hist[i]);
    }
    return result;
}