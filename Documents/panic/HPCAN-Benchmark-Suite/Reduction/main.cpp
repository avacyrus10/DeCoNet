#include <iostream>  
#include <string>  
#include <fstream>    
#include <vector>     

void reduce(int* input, int* output, int n);
void reduceCooperative(int* input, int* output, int n);

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <inputFile> <outputFile> <size> <method>\n";
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];
    int size = std::stoi(argv[3]);
    std::string method = argv[4];

    std::ifstream inputFileStream(inputFile, std::ios::binary);
    if (!inputFileStream) {
        std::cerr << "Error: Unable to open input file " << inputFile << "\n";
        return 1;
    }

    std::vector<int> input(size);
    inputFileStream.read(reinterpret_cast<char*>(input.data()), size * sizeof(int));

    int result;

    if (method == "cooperative") {
        reduceCooperative(input.data(), &result, size);
    } else {
        reduce(input.data(), &result, size);
    }

    std::ofstream outputFileStream(outputFile, std::ios::binary);
    if (!outputFileStream) {
        std::cerr << "Error: Unable to open output file " << outputFile << "\n";
        return 1;
    }

    outputFileStream.write(reinterpret_cast<const char*>(&result), sizeof(int));
    return 0;
}

