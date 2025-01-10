NVCC = nvcc
GXX = g++
CXXFLAGS = -std=c++11 -I/usr/local/cuda/include

SRCDIR = .
OBJDIR = ./obj
BINDIR = ./bin

CU_SRCS = $(SRCDIR)/correlation3d.cu
CPP_SRCS = $(SRCDIR)/main.cpp
HDRS = $(SRCDIR)/correlation3d.h

CU_OBJS = $(OBJDIR)/correlation3d.o
CPP_OBJS = $(OBJDIR)/main.o
OBJS = $(CU_OBJS) $(CPP_OBJS)

TARGET = $(BINDIR)/correlate3D

CUDA_LIB = -lcudart
CUDA_LIB_PATH = -L/usr/local/cuda/lib64

dirs:
	mkdir -p $(OBJDIR) $(BINDIR)

all: dirs $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(CUDA_LIB_PATH) $(CUDA_LIB)

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp $(HDRS) | dirs
	$(GXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/correlation3d.o: $(SRCDIR)/correlation3d.cu $(HDRS) | dirs
	$(NVCC) -c $< -o $@

inputs: dirs
	python3 generate_inputs.py

validation: all
	python3 validate_outputs.py

clean:
	rm -rf $(OBJDIR) $(BINDIR) data

.PHONY: all inputs validation clean dirs

