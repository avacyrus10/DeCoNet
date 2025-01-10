CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

TARGET := ifft_test
SRCS := main.cu ifft.cu cufft_impl.cu
INCLUDES := -I.
CFLAGS := -O3 -arch=sm_75
LDFLAGS := -lcufft -lcudart

.PHONY: all clean run generate validate

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

generate: $(TARGET)
	./$(TARGET) generate

validate: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

