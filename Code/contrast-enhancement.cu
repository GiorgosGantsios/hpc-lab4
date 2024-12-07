#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)  {
    PGM_IMG result;
    int hist[256];

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);

    return result;
}

PGM_IMG contrast_enhancement_GPU(PGM_IMG img_in)  {
    cudaEvent_t startCuda, stopCuda;
    PGM_IMG gpuResult;
    PGM_IMG result;
    float millisecondsTransfers = 0, time;
    //int hist[256];
    int t_hist[256];
    int *d_hist, *A0_hist, *A1_hist, *A2_hist, *A3_hist, *A4_hist, *A5_hist, *A6_hist, *A7_hist;
    unsigned char * d_ImgIn;
    unsigned char * img_A0,* img_A1,* img_A2,* img_A3,* img_A4,* img_A5,* img_A6,* img_A7;
    int SegSize = 114688;
    int t0_hist[256], t1_hist[256], t2_hist[256], t3_hist[256], t4_hist[256], t5_hist[256], t6_hist[256], t7_hist[256];
    cudaStream_t stream0, stream1, stream2, stream3, stream4, stream5, stream6, stream7;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    // cudaStreamCreate(&stream2);
    // cudaStreamCreate(&stream3);
    // cudaStreamCreate(&stream4);
    // cudaStreamCreate(&stream5);
    // cudaStreamCreate(&stream6);
    // cudaStreamCreate(&stream7);

    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    gpuResult.w = img_in.w;
    gpuResult.h = img_in.h;

    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    cudaEventRecord(startCuda);


    cudaError_t err = cudaMalloc((void **)&gpuResult.img, gpuResult.w * gpuResult.h * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
    }
    err = cudaMalloc((void **)&d_ImgIn, gpuResult.w * gpuResult.h * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
    }
    err = cudaMalloc((void**)&d_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
    }
    // err = cudaMalloc((void**)&A0_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&A1_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&A2_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&A3_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&A4_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&A5_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&A6_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&A7_hist, 256 * sizeof(int));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    err = cudaMalloc((void**)&img_A0, SegSize * sizeof(unsigned char));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
    }
    err = cudaMalloc((void**)&img_A1, SegSize * sizeof(unsigned char));  // Allocate memory on the GPU
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
        if (gpuResult.img) cudaFree(gpuResult.img);
        return(gpuResult);
    }
    // err = cudaMalloc((void**)&img_A2, SegSize * sizeof(unsigned char));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&img_A3, SegSize * sizeof(unsigned char));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&img_A4, SegSize * sizeof(unsigned char));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&img_A5, SegSize * sizeof(unsigned char));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&img_A6, SegSize * sizeof(unsigned char));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    // err = cudaMalloc((void**)&img_A7, SegSize * sizeof(unsigned char));  // Allocate memory on the GPU
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
    //     if (gpuResult.img) cudaFree(gpuResult.img);
    //     return(gpuResult);
    // }
    
    cudaMemset(d_hist, 0, sizeof(int) * 256);

    err = cudaMemcpy(d_ImgIn, img_in.img, gpuResult.w * gpuResult.h * sizeof(unsigned char), cudaMemcpyHostToDevice);  // Copy data from host to device
    //err = cudaMemcpy(img_A0, img_in.img, SegSize * sizeof(unsigned char), cudaMemcpyHostToDevice);  // Copy data from host to device

    //cudaMemcpyAsync(img_A0, d_ImgIn, SegSize*sizeof(unsigned char), cudaMemcpyHostToDevice,  stream0);
    //histogramGPU<<<((SegSize)/256), 256, 256*sizeof(int), stream0 >>>(A0_hist, img_A0, SegSize);
    //histogramGPU<<<((gpuResult.h*gpuResult.w)/256)+1, 256, 256*sizeof(int) >>>(d_hist, d_ImgIn, gpuResult.w* gpuResult.h);

    for (int i=0; i<gpuResult.h*gpuResult.w; i+=SegSize*2)  {
        cudaMemcpyAsync(img_A0, img_in.img+i, SegSize*sizeof(unsigned char), cudaMemcpyHostToDevice,  stream0);
        cudaMemcpyAsync(img_A1, img_in.img+i+SegSize, SegSize*sizeof(unsigned char), cudaMemcpyHostToDevice, stream1); 
        // cudaMemcpyAsync(img_A2, img_in.img+i+2 * SegSize, SegSize*sizeof(unsigned char), cudaMemcpyHostToDevice, stream2);
        // cudaMemcpyAsync(img_A3, img_in.img+i+3 * SegSize, SegSize*sizeof(unsigned char), cudaMemcpyHostToDevice, stream3);
        // cudaMemcpyAsync(img_A4, img_in.img+i+4 * SegSize, SegSize*sizeof(unsigned char), cudaMemcpyHostToDevice, stream4);
        // cudaMemcpyAsync(img_A5, img_in.img+i+5 * SegSize, SegSize*sizeof(unsigned char), cudaMemcpyHostToDevice, stream5);
        // cudaMemcpyAsync(img_A6, img_in.img+i+6 * SegSize, SegSize*sizeof(unsigned char), cudaMemcpyHostToDevice, stream6);
        // cudaMemcpyAsync(img_A7, img_in.img+i+7 * SegSize, SegSize*sizeof(unsigned char), cudaMemcpyHostToDevice, stream7);
        histogramGPU<<<((SegSize)/256), 256, 256*sizeof(int), stream0 >>>(d_hist, img_A0, SegSize);
        histogramGPU<<<((SegSize)/256), 256, 256*sizeof(int), stream1 >>>(d_hist, img_A1, SegSize);
        // histogramGPU<<<((SegSize)/256), 256, 256*sizeof(int), stream2 >>>(d_hist, img_A2, SegSize);
        // histogramGPU<<<((SegSize)/256), 256, 256*sizeof(int), stream3 >>>(d_hist, img_A3, SegSize);
        // histogramGPU<<<((SegSize)/256), 256, 256*sizeof(int), stream4 >>>(d_hist, img_A4, SegSize);
        // histogramGPU<<<((SegSize)/256), 256, 256*sizeof(int), stream5 >>>(d_hist, img_A5, SegSize);
        // histogramGPU<<<((SegSize)/256), 256, 256*sizeof(int), stream6 >>>(d_hist, img_A6, SegSize);
        // histogramGPU<<<((SegSize)/256), 256, 256*sizeof(int), stream7 >>>(d_hist, img_A7, SegSize);
        // cudaMemcpyAsync(t0_hist, A0_hist, 256*sizeof(int), cudaMemcpyDeviceToHost, stream0);
        // cudaMemcpyAsync(t1_hist, A1_hist, 256*sizeof(int), cudaMemcpyDeviceToHost, stream1);
        // cudaMemcpyAsync(t2_hist, A2_hist, 256*sizeof(int), cudaMemcpyDeviceToHost, stream2);
        // cudaMemcpyAsync(t3_hist, A3_hist, 256*sizeof(int), cudaMemcpyDeviceToHost, stream3);
        // cudaMemcpyAsync(t4_hist, A4_hist, 256*sizeof(int), cudaMemcpyDeviceToHost, stream4);
        // cudaMemcpyAsync(t5_hist, A5_hist, 256*sizeof(int), cudaMemcpyDeviceToHost, stream5);
        // cudaMemcpyAsync(t6_hist, A6_hist, 256*sizeof(int), cudaMemcpyDeviceToHost, stream6);
        // cudaMemcpyAsync(t7_hist, A7_hist, 256*sizeof(int), cudaMemcpyDeviceToHost, stream7);
        cudaStreamSynchronize(stream0);
        cudaStreamSynchronize(stream1);
        // cudaStreamSynchronize(stream2);
        // cudaStreamSynchronize(stream3);
        // cudaStreamSynchronize(stream4);
        // cudaStreamSynchronize(stream5);
        // cudaStreamSynchronize(stream6);
        // cudaStreamSynchronize(stream7);
        // for (int i = 0; i < 256; i++)  {
        //     t_hist[i] = t0_hist[i] + t1_hist[i] + t2_hist[i] + t3_hist[i] + t4_hist[i] + t5_hist[i] + t6_hist[i] + t7_hist[i];
        // }
    }
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KapCUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    //histogramGPU<<<((gpuResult.h*gpuResult.w)/256)+1, 256, 256*sizeof(int) >>>(d_hist, d_ImgIn, gpuResult.w, gpuResult.h);

    err = cudaMemcpy(t_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);  // Copy data from host to device

    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&millisecondsTransfers, startCuda, stopCuda);

    printf("\nGPU1 Execution time: %lf seconds\n", millisecondsTransfers/1000.0);
    time = histogram_equalization_prep(gpuResult.img, img_in.img, t_hist, gpuResult.w, gpuResult.h, 256, d_ImgIn);

    time += millisecondsTransfers;

    cudaEventRecord(startCuda, 0);

    err = cudaMemcpy(result.img, gpuResult.img, gpuResult.w * gpuResult.h * sizeof(unsigned char), cudaMemcpyDeviceToHost);  // Copy data from host to device

    cudaFree(d_ImgIn);  
    cudaFree(d_hist);
    cudaFree(gpuResult.img);

    cudaEventRecord(stopCuda, 0);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&millisecondsTransfers, startCuda, stopCuda);

    time += millisecondsTransfers;

    printf("\nGPU Execution time: %lf seconds\n", time/1000.0);

    return result;
}