CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

CFLAGS := -O3 -arch=sm_75
LDFLAGS := -lcufft -lcudart

all: dct_generate dct_validate

dct_generate: dct_generate.cu dct.cu
	$(NVCC) $(CFLAGS) -o dct_generate dct_generate.cu dct.cu $(LDFLAGS)

dct_validate: dct_validate.cu dct.cu
	$(NVCC) $(CFLAGS) -o dct_validate dct_validate.cu dct.cu $(LDFLAGS)

generate: dct_generate
	./dct_generate

validate: dct_validate
	./dct_validate

clean:
	rm -f dct_generate dct_validate data/*.txt

