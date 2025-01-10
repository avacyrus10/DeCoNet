NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -arch=sm_60
LDFLAGS = -lcublas
TARGET_CUSTOM = matrix_norm
TARGET_CUBLAS = matrix_norm_cublas
OBJ_CUSTOM = matrix_norm.o
OBJ_CUBLAS = matrix_norm_cublas.o

all: $(TARGET_CUSTOM) $(TARGET_CUBLAS)

$(TARGET_CUSTOM): $(OBJ_CUSTOM)
	$(NVCC) $(CFLAGS) -o $(TARGET_CUSTOM) $(OBJ_CUSTOM)

$(TARGET_CUBLAS): $(OBJ_CUBLAS)
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $(TARGET_CUBLAS) $(OBJ_CUBLAS)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET_CUSTOM) $(TARGET_CUBLAS)
	rm -rf data

custom_matrix_norm_test: generate_matrices $(TARGET_CUSTOM)
	python3 test_matrix_norm.py custom

cublas_matrix_norm_test: generate_matrices $(TARGET_CUBLAS)
	python3 test_matrix_norm.py cublas

generate_matrices:
	python3 generate_matrices.py
