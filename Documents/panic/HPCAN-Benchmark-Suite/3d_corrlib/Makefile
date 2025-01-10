NVCC = nvcc
GXX = g++
CXXFLAGS = -std=c++11 -I/usr/local/cuda/include

SRCDIR = .
OBJDIR = ./obj
BINDIR = ./bin

CU_SRCS = $(SRCDIR)/correlation3d.cu
CPP_SRCS = $(SRCDIR)/main.cpp $(SRCDIR)/fileio.cpp
HDRS = $(SRCDIR)/correlation3d.h

CU_OBJS = $(OBJDIR)/correlation3d.o
CPP_OBJS = $(OBJDIR)/main.o $(OBJDIR)/fileio.o
OBJS = $(CU_OBJS) $(CPP_OBJS)

TARGET = $(BINDIR)/correlate3D

CUDA_LIB = -lcudart -lcudnn
CUDA_LIB_PATH = -L/usr/local/cuda/lib64

$(shell mkdir -p $(OBJDIR))
$(shell mkdir -p $(BINDIR))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(CUDA_LIB_PATH) $(CUDA_LIB)

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp $(HDRS)
	$(GXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/fileio.o: $(SRCDIR)/fileio.cpp $(HDRS)
	$(GXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/correlation3d.o: $(SRCDIR)/correlation3d.cu $(HDRS)
	$(NVCC) -c $< -o $@

generate_input:
	python3 test_correlation3d.py generate

test: all
	python3 test_correlation3d.py test ./$(TARGET)

clean:
	rm -rf $(OBJDIR) $(BINDIR)
	rm -rf data/

.PHONY: all generate_input test clean

