CC = nvcc
CFLAGS = -std=c++11 -lcudnn -lcuda -O2
TARGET = correlate2D
SRC = cudnn_correlation2d.cu main.cpp
OBJ = $(SRC:.cpp=.o)
PY_SCRIPT = test_script.py

all: $(TARGET)

$(TARGET): cudnn_correlation2d.o main.o
	$(CC) $(CFLAGS) -o $@ cudnn_correlation2d.o main.o

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

generate_input:
	python3 $(PY_SCRIPT) generate

test:
	python3 $(PY_SCRIPT) test ./$(TARGET)

clean:
	rm -f *.o $(TARGET)
	rm -rf data/

