NVCC = nvcc
CFLAGS = -g -O4 -Xptxas -v

# TARGET = Convolution2D

# SRC = Convolution2D.cu

TARGET = main

SRC = main.cu histogram-equalization.cu contrast-enhancement.cu
HEADERS = hist-equ.h

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

#LOG_FILE = experiment_12.log

# Clean up
clean:
	rm -f $(TARGET)



experiment_12: $(TARGET)
	@echo "Running $(TARGET) 12 times with filter radius $(FILTER_RADIUS) and image size $(IMAGE_SIZE), writing output to $(LOG_FILE)"
	@rm -f $(LOG_FILE)
	@for i in {1..12}; do \
    	echo "Run $$i:" >> $(LOG_FILE); \
    	./$(TARGET) $(FILTER_RADIUS) $(IMAGE_SIZE) >> $(LOG_FILE); \
    	echo "" >> $(LOG_FILE); \
    done

run_test:
	@rm -f output.txt
	@for i in $(shell seq 12); do \
        ./main ../Images/planet_surface.pgm ../Images/planet_surface_gpu.pgm >> output.txt; \
    done

run_test1:
	@rm -f output.txt
	@for i in $(shell seq 12); do \
        ./main ../Images/fort.pgm ../Images/fort_gpu.pgm >> output.txt; \
    done

run_test2:
	@rm -f output.txt
	@for i in $(shell seq 12); do \
        ./main ../Images/ship.pgm ../Images/ship_gpu.pgm >> output.txt; \
    done

run_test3:
	@rm -f output.txt
	@for i in $(shell seq 12); do \
        ./main ../Images/uth.pgm ../Images/uth_gpu.pgm >> output.txt; \
    done

run_test4:
	@rm -f output.txt
	@for i in $(shell seq 12); do \
        ./main ../Images/x_ray.pgm ../Images/x_ray_gpu.pgm >> output.txt; \
    done

run_test5:
	@rm -f output.txt
	@for i in $(shell seq 12); do \
        ./main ../Images/senator.pgm ../Images/senator_gpu.pgm >> output.txt; \
    done
