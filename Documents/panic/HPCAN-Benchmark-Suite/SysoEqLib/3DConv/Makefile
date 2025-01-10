NVCC = nvcc
CUDNN_FLAGS = -lcudnn

SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin

CUDA_OBJS = $(BUILD_DIR)/convolution3d.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/main.o
CUDNN_OBJS = $(BUILD_DIR)/cudnn_convolution3d.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/main_cudnn.o

dirs:
	mkdir -p $(BUILD_DIR) $(BIN_DIR)

$(BUILD_DIR)/convolution3d.o: $(SRC_DIR)/convolution3d.cu
	$(NVCC) -c $< -o $@

$(BUILD_DIR)/cudnn_convolution3d.o: $(SRC_DIR)/cudnn_convolution3d.cpp
	$(NVCC) -c $< -o $@

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp
	$(NVCC) -c $< -o $@

$(BUILD_DIR)/main_cudnn.o: $(SRC_DIR)/main.cpp
	$(NVCC) -DUSE_CUDNN -c $< -o $@

$(BUILD_DIR)/utils.o: $(SRC_DIR)/utils.cpp
	$(NVCC) -c $< -o $@

inputs:
	python3 generate_inputs.py

validation: inputs dirs $(CUDA_OBJS)
	@echo "Validating without CUDNN..."
	$(NVCC) $(CUDA_OBJS) -o $(BIN_DIR)/convolve3D
	python3 validate_outputs.py

validation-cudnn: inputs dirs $(CUDNN_OBJS)
	@echo "Validating with CUDNN..."
	$(NVCC) -DUSE_CUDNN $(CUDNN_OBJS) -o $(BIN_DIR)/cudnn_convolve3D $(CUDNN_FLAGS)
	python3 validate_outputs.py --cudnn

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) data

.PHONY: all dirs clean inputs validation validation-cudnn

