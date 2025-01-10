NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -arch=sm_60
LDFLAGS = -lcusolver -lcublas
TARGET = system_solver
OBJ = system_solver_cusolver.o
PYTHON = python3


TEST_SCRIPT = test_system_of_equations.py

DATA_DIR = data
SOLUTION_FILE = solution.bin

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJ)

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
	rm -rf $(DATA_DIR) $(SOLUTION_FILE)

test: all generate_matrices validate

generate_matrices:
	$(PYTHON) generate_matrices.py

validate:
	$(PYTHON) $(TEST_SCRIPT)

clean_data:
	rm -rf $(DATA_DIR) $(SOLUTION_FILE)

