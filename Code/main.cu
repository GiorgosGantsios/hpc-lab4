#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

#define null_img {0, 0, NULL}
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.01
#define CHECK_CUDA_ERROR(call, img_ibuf_g, img_obuf, d_img_in, gpu_img)          \
    do {                                                                      \
        cudaError_t __err = call;                                             \
        if (__err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n",      \
                    __FILE__, __LINE__, __err, cudaGetErrorString(__err));    \
            clean_exit_program(img_ibuf_g, img_obuf, d_img_in, gpu_img);         \
        }                                                                     \
    } while (0)

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename);
void clean_exit_program(PGM_IMG img_ibuf_g, PGM_IMG img_obuf, PGM_IMG *d_img_in, PGM_IMG gpu_img);

int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g, gpu_img;
    PGM_IMG *d_img_in;
    PGM_IMG *d_img_out;
    struct timespec  tv1, tv2;
    cudaEvent_t startCuda, stopCuda;
    float elapsed_time_CPU, millisecondsGPU;

	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(argv[1]);
    if(!img_ibuf_g.img) clean_exit_program(img_ibuf_g, null_img, d_img_in, gpu_img);   // Error check malloc in read_pgm function
    CHECK_CUDA_ERROR(cudaMalloc((void**) &d_img_in, sizeof(PGM_IMG)),
                                img_ibuf_g, null_img, d_img_in, gpu_img);
    CHECK_CUDA_ERROR(cudaMalloc((void**) &d_img_out, sizeof(PGM_IMG)),
                                img_ibuf_g, null_img, d_img_in, gpu_img);
    CHECK_CUDA_ERROR(cudaMemset(d_img_out, 0, sizeof(PGM_IMG)),
                                img_ibuf_g, null_img, d_img_in, gpu_img);

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    run_cpu_gray_test(img_ibuf_g, argv[2]);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    elapsed_time_CPU = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_nsec - tv1.tv_nsec) / 1e9;
    printf("CPU Execution time: %lf seconds\n", elapsed_time_CPU);

    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);
    cudaEventRecord(startCuda, 0);
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_img_in, &img_ibuf_g, sizeof(PGM_IMG), cudaMemcpyHostToDevice),
                                img_ibuf_g, null_img, d_img_in, gpu_img);
    
    

    CHECK_CUDA_ERROR(cudaMemcpy(&gpu_img, d_img_out, sizeof(PGM_IMG), cudaMemcpyDeviceToHost),
                                img_ibuf_g, null_img, d_img_in, gpu_img);

    cudaEventRecord(stopCuda, 0);
    cudaEventSynchronize(stopCuda);
    cudaEventElapsedTime(&millisecondsGPU, startCuda, stopCuda);

    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess) clean_exit_program(img_ibuf_g, null_img, d_img_in, gpu_img);
    printf("GPU Execution time: %lf seconds\n", millisecondsGPU/1000);

    // check_equality();

    // free_pgm(img_ibuf_g);
    clean_exit_program(img_ibuf_g, null_img, d_img_in, gpu_img);

	return 0;
}

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    unsigned int timer = 0;
    PGM_IMG img_obuf;
    
    
    printf("Starting CPU processing...\n");
    img_obuf = contrast_enhancement_g(img_in);
    write_pgm(img_obuf, out_filename);
    free_pgm(img_obuf);
}


PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

        
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}

void clean_exit_program(PGM_IMG img_ibuf_g, PGM_IMG img_obuf, PGM_IMG *d_img_in, PGM_IMG gpu_img){
    
    // Ensure kernels have finished
    cudaDeviceSynchronize();
    
    // Deallocate CPU memory
    if(img_ibuf_g.img) free_pgm(img_ibuf_g);
    if(img_obuf.img) free_pgm(img_obuf);
    if(gpu_img.img) free_pgm(gpu_img);

    // Deallocate GPU memory
    if(d_img_in){
        // if(d_img_in.img) cudaFree(d_img_in.img);
    cudaFree(d_img_in);
    }
    cudaError_t e;
    e = cudaGetLastError();
    if(e!=cudaSuccess){
      printf("ERROR: %s, FILE: %s, LINE: %d\n", cudaGetErrorString(e), __FILE__, __LINE__);
    }
    else{
      printf("cudaGetLastError() == cudaSuccess!\n");
    }

    cudaDeviceReset();


    exit(1);    // exit with failure
}
