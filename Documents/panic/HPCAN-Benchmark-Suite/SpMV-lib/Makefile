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
	./test_spmv

input_generate_exec: input_generate.cpp matrix_generate.cpp spmv.h
	$(CXX) $(CXXFLAGS) input_generate.cpp matrix_generate.cpp -o input_generate

validate_exec: test_spmv.cpp spmv.cu matrix_generate.cpp spmv.h
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) spmv.cu test_spmv.cpp matrix_generate.cpp -o test_spmv $(LDFLAGS)

clean:
	rm -f input_generate test_spmv
	rm -rf data

