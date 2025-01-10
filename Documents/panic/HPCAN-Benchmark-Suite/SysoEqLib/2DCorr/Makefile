NVCC = nvcc
GXX = g++
CXXFLAGS = -std=c++11 -I/usr/local/cuda/include 

SRCDIR = .
OBJDIR = ./obj
BINDIR = ./bin

CU_SRCS = $(SRCDIR)/correlation2d_kernel.cu
CPP_SRCS = $(SRCDIR)/main.cpp
HDRS = $(SRCDIR)/correlation2d.h

CU_OBJS = $(OBJDIR)/correlation2d_kernel.o
CPP_OBJS = $(OBJDIR)/main.o
OBJS = $(CU_OBJS) $(CPP_OBJS)

TARGET = $(BINDIR)/correlate2D

CUDA_PATH = /usr/local/cuda
CUDA_LIB = -lcudart
CUDA_LIB_PATH = -L$(CUDA_PATH)/lib64

$(shell mkdir -p $(OBJDIR))
$(shell mkdir -p $(BINDIR))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(CUDA_LIB_PATH) $(CUDA_LIB)

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp $(HDRS)
	$(GXX) $(CXXFLAGS) -c $(SRCDIR)/main.cpp -o $@

$(OBJDIR)/correlation2d_kernel.o: $(SRCDIR)/correlation2d_kernel.cu $(HDRS)
	$(NVCC) -c $(SRCDIR)/correlation2d_kernel.cu -o $@

inputs:
	python3 generate_inputs.py

validation: all
	python3 validate_outputs.py

clean:
	rm -rf $(OBJDIR) $(BINDIR) data

.PHONY: all inputs validation clean

