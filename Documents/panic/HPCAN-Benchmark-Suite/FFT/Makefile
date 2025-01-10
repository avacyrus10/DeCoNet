
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

TARGET := fft
SRCS := main.cu fft.cu
CUFFT_SRCS := cufft_impl.cu
VALIDATE_FFT_SRCS := validate_fft.cu fft.cu
VALIDATE_CUFFT_SRCS := validate_cufft.cu cufft_impl.cu
INPUT_GENERATION_SRCS := input_generation.cu


INCLUDES := -I.
CFLAGS := -O3 -arch=sm_75
LDFLAGS := -lcufft -lcudart

.PHONY: all clean test_fft test_cufft validate_fft validate_cufft generate_input


all: $(TARGET)

$(TARGET): $(SRCS) $(CUFFT_SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

test_fft: validate_fft
	./validate_fft

test_cufft: validate_cufft
	./validate_cufft

validate_fft: $(VALIDATE_FFT_SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o validate_fft $^ $(LDFLAGS)

validate_cufft: $(VALIDATE_CUFFT_SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o validate_cufft $^ $(LDFLAGS)

generate_input: $(INPUT_GENERATION_SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o generate_input $^ $(LDFLAGS)

run_input_generation: generate_input
	./generate_input

clean:
	rm -f $(TARGET) validate_fft validate_cufft generate_input test_fft test_cufft data/*.txt

