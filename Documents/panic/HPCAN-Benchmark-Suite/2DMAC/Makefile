CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
TARGET = mac_program
SRC = mac.cu
OBJ = mac.o
CFLAGS = -O3
PYTHON = python3
DATA_DIR = data
GEN_SCRIPT = input_generator.py
TEST_SCRIPT = test_mac.py

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $^

generate_inputs:
	mkdir -p $(DATA_DIR)
	$(PYTHON) $(GEN_SCRIPT)

validate: all
	$(PYTHON) $(TEST_SCRIPT)

clean:
	rm -f $(TARGET) *.o $(DATA_DIR)/*.bin

