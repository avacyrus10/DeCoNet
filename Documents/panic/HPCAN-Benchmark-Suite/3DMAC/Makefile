CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
TARGET = mac_program
SRC = mac_3d.cu
CFLAGS = -O3
PYTHON = python3
GEN_SCRIPT = input_generator_3d.py
TEST_SCRIPT = test_mac_3d.py
DATA_DIR = data

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

