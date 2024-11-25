# Compilers
NVCC = nvcc
CC = gcc

# CUDA Paths
CUDA_PATH = /usr/local/cuda
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart
INCLUDES = -I$(CUDA_PATH)/include

# Flags
CFLAGS = -Wall -O2
NVCCFLAGS = -g -O4 $(INCLUDES)

# Target
TARGET = histogram_equalization

# Source Files
CUDA_SRCS = kernel.cu
C_SRCS = main.c histogram-equalization.c contrast-enhancement.c

# Object Files
C_OBJS = $(C_SRCS:.c=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# Build Target
$(TARGET): $(C_OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(C_OBJS) $(CUDA_OBJS) $(CUDA_LIBS)

# Compile C Source Files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA Source Files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean Up
clean:
	rm -f $(TARGET) $(C_OBJS) $(CUDA_OBJS)



