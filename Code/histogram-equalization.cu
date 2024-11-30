#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin) {
    int i;

    // Initialization
    for (i = 0; i < nbr_bin; i++) {
        hist_out[i] = 0;
    }

    // Constructs the Histogram Vector
    for (i = 0; i < img_size; i++) {
        //printf("CPUindex: %d, img: %d\n", i, img_in[i]);
        hist_out[img_in[i]]++;
    }
    
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin) {
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d, index;

    /* Construct the LUT by calculating the CDF */
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
    
    /* Get the result image */
    for(i = 0; i < img_size; i++) {
        img_out[i] = (unsigned char)lut[img_in[i]];
    }
}

__global__ void histogramGPU(int * hist_out) {
    int index = threadIdx.x;

    // Initialization
    hist_out[index] = 0;

    // Constructs the Histogram Vector
    /*for (i = 0; i < img_size; i++) {
        hist_out[img_in[i]]++;
    }*/
}
__global__ void histogramConstuctionGPU(int * hist_out, unsigned char * img_in, int imageW, int imageH) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int y = index / imageW; // row
    int x = index % imageW; // col
    
    //printf("GPUindex: %d, img: %d\n", index, img_in[index]);

    // Constructs the Histogram Vector
    atomicAdd(&hist_out[img_in[y * imageW + x]], 1);
    //__syncthreads();
}
