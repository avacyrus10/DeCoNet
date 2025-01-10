CC = nvcc
BIN_DIR = bin
DATA_DIR = data
CFLAGS = -O3 -arch=sm_52
LDFLAGS = -lcublas
PYTHON = python3
TEST_SCRIPT = test_matrix_mult_3d.py

all: $(BIN_DIR)/matrixMult3D $(BIN_DIR)/matrixMult3D_cublas

$(BIN_DIR)/matrixMult3D: matrix_mult_3d.cu
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/matrixMult3D matrix_mult_3d.cu

$(BIN_DIR)/matrixMult3D_cublas: matrix_mult_3d_cublas.cu
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(BIN_DIR)/matrixMult3D_cublas matrix_mult_3d_cublas.cu

generate_inputs:
	mkdir -p $(DATA_DIR)
	$(PYTHON) $(TEST_SCRIPT) --generate_only

validate: $(BIN_DIR)/matrixMult3D
	$(PYTHON) $(TEST_SCRIPT)

validate_cublas: $(BIN_DIR)/matrixMult3D_cublas
	$(PYTHON) $(TEST_SCRIPT) --use_cublas

clean:
	rm -rf $(BIN_DIR)/*
	rm -rf $(DATA_DIR)/*.txt

