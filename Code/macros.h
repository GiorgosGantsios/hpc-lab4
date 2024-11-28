#ifndef SHARED_MACROS_H
#define SHARED_MACROS_H

#define ABS(x) ((x) < 0 ? -(x) : (x))
#define accuracy 0.05
#define CHECK_CUDA_ERROR(call)                                                \
    do {                                                                      \
        cudaError_t __err = call;                                             \
        if (__err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n",      \
                    __FILE__, __LINE__, __err, cudaGetErrorString(__err));    \
            error_flag = 1;                                                   \
        }                                                                     \
    } while (0)

#endif