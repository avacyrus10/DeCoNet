
NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -arch=sm_60
LDFLAGS = -lcusolver -lcublas
TARGET_MANUAL = system_solver
TARGET_LIB = system_solver_cusolver
OBJ_MANUAL = system_solver.o
OBJ_LIB = system_solver_cusolver.o
PYTHON = python3

TEST_SCRIPT = test_system_of_equations.py

DATA_DIR = data
SOLUTION_FILE = solution.bin

all: $(TARGET_MANUAL) $(TARGET_LIB)

$(TARGET_MANUAL): $(OBJ_MANUAL)
	$(NVCC) $(CFLAGS) -o $(TARGET_MANUAL) $(OBJ_MANUAL)

$(TARGET_LIB): $(OBJ_LIB)
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $(TARGET_LIB) $(OBJ_LIB)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET_MANUAL) $(TARGET_LIB)
	rm -rf $(DATA_DIR) $(SOLUTION_FILE)


test_lib: $(TARGET_LIB)
	$(PYTHON) generate_matrices.py
	$(PYTHON) $(TEST_SCRIPT) lib

generate_matrices:
	$(PYTHON) generate_matrices.py

clean_data:
	rm -rf $(DATA_DIR) $(SOLUTION_FILE)

