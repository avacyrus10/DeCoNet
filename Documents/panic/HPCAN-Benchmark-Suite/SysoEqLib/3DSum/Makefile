CC = nvcc
BIN_DIR = bin
DATA_DIR = data
CFLAGS = -O3 -arch=sm_52
LDFLAGS = -lcublas
PYTHON = python3

all: $(BIN_DIR)/matrixSum3D $(BIN_DIR)/matrixSum3D_cublas

$(BIN_DIR)/matrixSum3D: matrix_sum_3d.cu
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/matrixSum3D matrix_sum_3d.cu

$(BIN_DIR)/matrixSum3D_cublas: matrix_sum_3d_cublas.cu
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(BIN_DIR)/matrixSum3D_cublas matrix_sum_3d_cublas.cu

generate_inputs:
	mkdir -p $(DATA_DIR)
	$(PYTHON) input_generator_3d.py

validate: $(BIN_DIR)/matrixSum3D
	$(PYTHON) test_matrix_sum_3d.py

validate_cublas: $(BIN_DIR)/matrixSum3D_cublas
	$(PYTHON) test_matrix_sum_3d.py --use_cublas

clean:
	rm -rf $(BIN_DIR)/*
	rm -rf $(DATA_DIR)/*.txt

