#!/bin/bash

BASE_DIR="."

WORKLOADS=(
    "2DConv|make validation"
    "2DCorr|make validation"
    "2dcorr_lib|make test"
    "2DMAC|make validation"
    "2DMult|make validate"
    "2DMult_lib|make validate"
    "2DSum|make validate validate_cublas"
    "3DConv|make validation validation-cudnn"
    "3DCorr|make validation"
    "3d_corrlib|make test"
    "3DMAC|make validate"
    "3DMult|make validate validate_cublas"
    "3DSum|make validate validate_cublas"
    "DCT|make validate"
    "FFT|make test_fft test_cufft"
    "fix|make validate"
    "IDCT|make validate"
    "ifft|make validate"
    "MatrixInversion|make custom_inversion_test cublas_inversion_test"
    "MatrixNorm|make custom_matrix_norm_test cublas_matrix_norm_test"
    "Reduction|make test"
    "SMA|make validate"
    "SpMM|make validate_exec"
    "SpMM-lib|make validate"
    "SpMV|make validate_exec"
    "SpMV-lib|make validate"
    "SysoEq|make validate"
    "SysoEqLib|make test_lib"
)

generate_inputs() {
    echo "Generating inputs for all workloads..."
    for workload in "${WORKLOADS[@]}"; do
        IFS="|" read -r folder _ <<< "$workload"
        echo "Processing $folder..."
        if [ -d "$BASE_DIR/$folder" ]; then
            (cd "$BASE_DIR/$folder" && make generate_inputs 2>/dev/null || make generate_input 2>/dev/null || echo "No input generation target for $folder")
        else
            echo "Directory $folder does not exist. Skipping..."
        fi
    done
    echo "Input generation complete."
}

run_validations() {
    echo "Running validations for all workloads..."
    for workload in "${WORKLOADS[@]}"; do
        IFS="|" read -r folder commands <<< "$workload"
        echo "Processing $folder..."
        if [ -d "$BASE_DIR/$folder" ]; then
            (cd "$BASE_DIR/$folder" && eval "$commands" || echo "Failed to validate $folder")
        else
            echo "Directory $folder does not exist. Skipping..."
        fi
    done
    echo "Validation complete."
}

clean_all() {
    echo "Cleaning all workloads..."
    for workload in "${WORKLOADS[@]}"; do
        IFS="|" read -r folder _ <<< "$workload"
        echo "Processing $folder..."
        if [ -d "$BASE_DIR/$folder" ]; then
            (cd "$BASE_DIR/$folder" && make clean || echo "Failed to clean $folder")
        else
            echo "Directory $folder does not exist. Skipping..."
        fi
    done
    echo "Cleaning complete."
}

case $1 in
    generate)
        generate_inputs
        ;;
    test)
        run_validations
        ;;
    clean)
        clean_all
        ;;
    *)
        echo "Usage: $0 {generate|test|clean}"
        exit 1
        ;;
esac

