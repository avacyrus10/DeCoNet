CC = nvcc
BIN_DIR = bin
DATA_DIR = data
CFLAGS = -O3 -arch=sm_52
PYTHON = python3

all: $(BIN_DIR)/matrixMult

$(BIN_DIR)/matrixMult: matrix_mult.cu
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/matrixMult matrix_mult.cu

generate_inputs:
	mkdir -p $(DATA_DIR)
	$(PYTHON) input_generator_mult.py

validate: all
	$(PYTHON) test_matrix_mult.py

clean:
	rm -rf $(BIN_DIR)/*
	rm -rf $(DATA_DIR)/*.txt

