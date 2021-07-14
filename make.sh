#!/bin/bash

HALIDE_DIR=$HOME/opt/halide/ \
MKL_DIR=/opt/intel/oneapi/mkl/latest/ \
cmake -GNinja \
    -DCMAKE_CXX_COMPILER=clang++-11 \
    -DCMAKE_C_COMPILER=clang-11 \
    -DMKL_DIR=/opt/intel/oneapi/mkl/latest/ \
    -DUSE_MLIR=ON \
    -DUSE_CUDA=ON \
    -DUSE_CUBLAS=ON \
    -DUSE_OPENBLAS=ON \
    -B build .

cmake --build build

