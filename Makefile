# Compiler
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++14
NVCCFLAGS = -lcufft

# Source files
SOURCES = main.cpp fft_cuda.cu
HEADERS = fft_cuda.cuh

# Object files
OBJECTS = $(SOURCES:.cpp=.o)
OBJECTS := $(OBJECTS:.cu=.o)

# Executable name
EXECUTABLE = fft_program

# Make all
all: $(EXECUTABLE)

# Compile source files
%.o: %.cpp $(HEADERS)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

%.o: %.cu $(HEADERS)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Link object files to create executable
$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(OBJECTS) $(NVCCFLAGS) -o $@

# Clean targets
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

