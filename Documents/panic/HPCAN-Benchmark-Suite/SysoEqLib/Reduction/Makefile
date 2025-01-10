NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11

TARGET = reduce_program
COOPERATIVE_TARGET = cooperative_grid_reduce

OBJDIR = obj
DATADIR = data

CU_SOURCES = reduction.cu cooperative_grid_reduction.cu
CPP_SOURCES = main.cpp

CU_OBJECTS = $(OBJDIR)/reduction.o $(OBJDIR)/cooperative_grid_reduction.o
CPP_OBJECTS = $(OBJDIR)/main.o

OBJECTS = $(CU_OBJECTS) $(CPP_OBJECTS)

.PHONY: all generate_input test clean

all: $(OBJDIR) $(TARGET) $(COOPERATIVE_TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) -L/usr/local/cuda/lib64 -lcudart

$(COOPERATIVE_TARGET): $(CPP_OBJECTS) $(OBJDIR)/reduction.o $(OBJDIR)/cooperative_grid_reduction.o
	$(CXX) $(CPP_OBJECTS) $(OBJDIR)/reduction.o $(OBJDIR)/cooperative_grid_reduction.o -o $(COOPERATIVE_TARGET) -L/usr/local/cuda/lib64 -lcudart

$(OBJDIR)/reduction.o: reduction.cu | $(OBJDIR)
	$(NVCC) -c reduction.cu -o $(OBJDIR)/reduction.o

$(OBJDIR)/cooperative_grid_reduction.o: cooperative_grid_reduction.cu | $(OBJDIR)
	$(NVCC) -c cooperative_grid_reduction.cu -o $(OBJDIR)/cooperative_grid_reduction.o

$(OBJDIR)/main.o: main.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c main.cpp -o $(OBJDIR)/main.o

generate_input:
	python3 test_reduction.py generate

test: all
	python3 test_reduction.py test

clean:
	rm -rf $(OBJDIR) $(TARGET) $(COOPERATIVE_TARGET) $(DATADIR)/

