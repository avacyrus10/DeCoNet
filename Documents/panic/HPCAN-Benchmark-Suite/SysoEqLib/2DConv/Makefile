NVCC = nvcc
GXX = g++
CXXFLAGS = -std=c++11 -I/usr/local/cuda/include  

SRCDIR = .
OBJDIR = ./obj
BINDIR = ./bin

CU_SRCS = $(SRCDIR)/convolution2d.cu
CU_CUDNN_SRCS = $(SRCDIR)/convolution2d_cudnn.cu
CPP_SRCS = $(SRCDIR)/main.cpp
HDRS = $(SRCDIR)/convolution2d.h

CU_OBJS = $(OBJDIR)/convolution2d.o
CU_CUDNN_OBJS = $(OBJDIR)/convolution2d_cudnn.o
CPP_OBJS = $(OBJDIR)/main.o
OBJS = $(CU_OBJS) $(CPP_OBJS)
CUDNN_OBJS = $(CU_CUDNN_OBJS) $(CU_OBJS) $(CPP_OBJS)

TARGET = $(BINDIR)/convolve2D
TARGET_CUDNN = $(BINDIR)/convolve2D_cudnn

CUDA_PATH = /usr/local/cuda
CUDA_LIB = -lcudart
CUDNN_LIB = -lcudart -lcudnn
CUDA_LIB_PATH = -L$(CUDA_PATH)/lib64

$(shell mkdir -p $(OBJDIR))
$(shell mkdir -p $(BINDIR))

all: $(TARGET)

cudnn: $(TARGET_CUDNN)

$(TARGET): $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(CU_OBJS) $(CPP_OBJS) -o $@ $(CUDA_LIB_PATH) $(CUDA_LIB)

$(TARGET_CUDNN): $(CU_CUDNN_OBJS) $(CU_OBJS) $(CPP_OBJS)
	$(NVCC) $(CUDNN_OBJS) -o $@ $(CUDA_LIB_PATH) $(CUDNN_LIB)

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp $(HDRS)
	$(GXX) $(CXXFLAGS) -c $(SRCDIR)/main.cpp -o $@

$(OBJDIR)/convolution2d.o: $(SRCDIR)/convolution2d.cu $(HDRS)
	$(NVCC) -c $(SRCDIR)/convolution2d.cu -o $@

$(OBJDIR)/convolution2d_cudnn.o: $(SRCDIR)/convolution2d_cudnn.cu $(HDRS)
	$(NVCC) -c $(SRCDIR)/convolution2d_cudnn.cu -o $@

inputs:
	python3 generate_inputs.py

validation: all
	python3 validate_outputs.py

clean:
	rm -rf $(OBJDIR) $(BINDIR) data

.PHONY: all cudnn inputs validation clean

