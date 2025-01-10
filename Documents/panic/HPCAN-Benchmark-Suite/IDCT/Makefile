CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
TARGET := dct_idct_test
SRCS := main.cu transform.cu
OBJS := $(SRCS:.cu=.o)
INCLUDES := -I.
CFLAGS := -O3 -arch=sm_75
LDFLAGS := -lcudart

.PHONY: all clean generate validate

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) -c $< -o $@

generate: $(TARGET)
	./$(TARGET) --generate

validate: $(TARGET)
	./$(TARGET) --validate

clean:
	rm -f $(TARGET) $(OBJS) data/*.txt

