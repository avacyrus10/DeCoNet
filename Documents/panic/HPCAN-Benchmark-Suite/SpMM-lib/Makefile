CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXX := g++
CXXFLAGS := -std=c++17
CUDAFLAGS := -arch=sm_60
LDFLAGS := -lcudart -lcusparse -L$(CUDA_PATH)/lib64

all: input_generate validate

input_generate: input_generate_exec
	./input_generate

validate: validate_exec
	./test_spmm

input_generate_exec: input_generate.cpp spmm.cu spmm.h
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) input_generate.cpp spmm.cu -o input_generate $(LDFLAGS)

validate_exec: test_spmm.cpp spmm.cu spmm.h
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) spmm.cu test_spmm.cpp -o test_spmm $(LDFLAGS)

clean:
	rm -f input_generate test_spmm
	rm -rf data

