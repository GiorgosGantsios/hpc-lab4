#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include "macros.h"

int error_flag = 0;

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename);

int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;
    struct timespec  tv1, tv2;
    float elapsed_time_CPU;

	if (argc != 3) {
		printf("Run with input file name and output file name as arguments\n");
		exit(1);
	}
	
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(argv[1]);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    run_cpu_gray_test(img_ibuf_g, argv[2]);
    if(error_flag) goto clean_up_main;
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    elapsed_time_CPU = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_nsec - tv1.tv_nsec) / 1e9;
    printf("CPU Execution time: %lf seconds\n", elapsed_time_CPU);
clean_up_main:
    free_pgm(img_ibuf_g);

	return 0;
}



void run_cpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    unsigned int timer = 0;
    unsigned char* gpu_img;
    PGM_IMG img_obuf;
    
    printf("Starting CPU processing...\n");
    img_obuf = contrast_enhancement_g(img_in);
    if(error_flag) return;
    gpu_img = (unsigned char*)malloc(img_obuf.w * img_obuf.h * sizeof(unsigned char));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_img, img_obuf.d_img, img_obuf.w * img_obuf.h * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    if(error_flag) return;
    for(int i = 0; i < img_obuf.w * img_obuf.h; i++){
        if(ABS(img_obuf.img[i] - gpu_img[i]) < accuracy){
            printf("Difference bigger than accuracy in index: %d: img_obuf.img[%d]=%d, gpu_img[%d]=%d\n", i, i, img_obuf.img[i], i, gpu_img[i]);
            error_flag = 1;
            break;
        }
    }
    free(gpu_img);
    if(!error_flag) write_pgm(img_obuf, out_filename);
    free_pgm(img_obuf);
    printf("Reset\n");
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
    CHECK_CUDA_ERROR(cudaMalloc((void**) &result.d_img, result.w * result.h *sizeof(unsigned char)));
    if(error_flag) goto clean_up_read_pgm;
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);
    CHECK_CUDA_ERROR(cudaMemcpy(result.d_img, result.img, result.w * result.h * sizeof(unsigned char), cudaMemcpyHostToDevice));
clean_up_read_pgm:
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
clean_up_free_pgm:
    cudaDeviceSynchronize();
    if(img.img) free(img.img);
    if(img.d_img)cudaFree(img.d_img);
    cudaDeviceReset();
}

