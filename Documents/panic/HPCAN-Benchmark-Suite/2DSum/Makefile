CC = nvcc
BIN_DIR = bin
DATA_DIR = data
CFLAGS = -O3 -arch=sm_52
LDFLAGS = -lcublas
PYTHON = python3
TEST_SCRIPT = test_matrix_sum_2d.py

all: $(BIN_DIR)/matrixSum2D $(BIN_DIR)/matrixSum2D_cublas

$(BIN_DIR)/matrixSum2D: matrix_sum_2d.cu
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/matrixSum2D matrix_sum_2d.cu

$(BIN_DIR)/matrixSum2D_cublas: matrix_sum_2d_cublas.cu
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(BIN_DIR)/matrixSum2D_cublas matrix_sum_2d_cublas.cu

validate: $(BIN_DIR)/matrixSum2D
	$(PYTHON) $(TEST_SCRIPT)

validate_cublas: $(BIN_DIR)/matrixSum2D_cublas
	$(PYTHON) $(TEST_SCRIPT) --use_cublas

generate_inputs:
	mkdir -p $(DATA_DIR)
	$(PYTHON) $(TEST_SCRIPT) --generate_only

clean:
	rm -rf $(BIN_DIR)/*
	rm -rf $(DATA_DIR)/*.txt

