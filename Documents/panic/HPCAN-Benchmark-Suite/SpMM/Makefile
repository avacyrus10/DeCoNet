CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
CXX := g++
CXXFLAGS := -std=c++17
CUDAFLAGS := -arch=sm_60
LDFLAGS := -lcudart -L$(CUDA_PATH)/lib64

BIN_DIR := bin
SRC_DIR := src
DATA_DIR := data

all: generate validate

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

generate: $(SRC_DIR)/generate.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $(BIN_DIR)/generate

validate: $(SRC_DIR)/validate.cpp $(SRC_DIR)/spmm.cu $(SRC_DIR)/spmm.h | $(BIN_DIR)
	$(NVCC) $(CUDAFLAGS) $(CXXFLAGS) $(SRC_DIR)/spmm.cu $< -o $(BIN_DIR)/validate $(LDFLAGS)

run-generate: generate
	./$(BIN_DIR)/generate $(DATA_DIR)

run-validate: validate
	./$(BIN_DIR)/validate $(DATA_DIR)

clean:
	rm -f $(BIN_DIR)/*
	rm -rf $(DATA_DIR)

