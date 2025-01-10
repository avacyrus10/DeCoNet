CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXX := g++
CXXFLAGS := -std=c++17
CUDAFLAGS := -arch=sm_60
LDFLAGS := -lcudart -L$(CUDA_PATH)/lib64

all: input_generate validate

input_generate: input_generate_exec
	./input_generate

validate: validate_exec
	./test_sma

input_generate_exec: input_generate.cpp sma.cpp sma.h
	$(CXX) $(CXXFLAGS) input_generate.cpp sma.cpp -o input_generate

validate_exec: test_sma.cpp sma.cu sma.cpp sma.h
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) sma.cu sma.cpp test_sma.cpp -o test_sma $(LDFLAGS)

clean:
	rm -f input_generate test_sma
	rm -rf data

