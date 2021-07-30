#!/bin/bash

echo "Translation + Compilation test with matmult-gpu.mlir.in ..."

mlir-opt matmul-gpu.mlir.in \
    --linalg-tile-to-parallel-loops="linalg-tile-sizes=1,1" \
    --convert-linalg-to-parallel-loops \
    --test-gpu-greedy-parallel-loop-mapping \
    --convert-parallel-loops-to-gpu \
    --gpu-kernel-outlining \
    --lower-affine \
    --convert-scf-to-std \
    --canonicalize \
    --pass-pipeline="gpu.module(strip-debuginfo, convert-gpu-to-nvvm, gpu-to-cubin)" \
    --gpu-to-llvm 2>&1 >matmul-gpu.mlir.out 

mlir-translate matmul-gpu.mlir.out --mlir-to-llvmir \
    | opt -O3 -S | llc -O3 | as - -o matmul-gpu.mlir.o

clang++-11 matmul-gpu.mlir.o -lcuda \
    $HOME/opt/llvm/lib/libmlir_cuda_runtime.so \
    $HOME/opt/llvm/lib/libmlir_runner_utils.so \
    -o matmul-gpu

