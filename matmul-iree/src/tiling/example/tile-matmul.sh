# Test mhlo.dot
../../../../build/flatbuffers-install/bin/flatc -b ../compile_options.fbs matmul_mhlo.json && \
cat ../../matmul_MxNxK.mlir | sed 's@${M}@'"256"'@g' | sed 's@${N}@'"256"'@g' | sed 's@${K}@'"256"'@g' | sed 's@${TYPE}@'"f32"'@g' > matmul_mhlo.mlir && \
../../../../build/matmul-iree/src/tiling/add-tiling-attribute-pass matmul_mhlo.mlir --compile-options "./matmul_mhlo.bin"

# Test linalg.matmul
../../../../build/flatbuffers-install/bin/flatc -b ../compile_options.fbs matmul_linalg.json && \
cat ../../matmul_linalg_MxNxK.mlir | sed 's@${M}@'"256"'@g' | sed 's@${N}@'"256"'@g' | sed 's@${K}@'"256"'@g' > matmul_linalg.mlir && \
../../../../build/matmul-iree/src/tiling/add-tiling-attribute-pass matmul_linalg.mlir --compile-options "./matmul_linalg.bin"

# Test multiple linalg.matmul
../../../../build/flatbuffers-install/bin/flatc -b ../compile_options.fbs matmul_multi.json && \
../../../../build/matmul-iree/src/tiling/add-tiling-attribute-pass matmul_multi.mlir --compile-options "./matmul_multi.bin"

# Test linalg.matmul on IREE-CUDA
#../../../../build_cuda/flatbuffers-install/bin/flatc -b ../compile_options.fbs matmul_linalg_cuda.json && \
#cat ../../matmul_linalg_MxNxK.mlir | sed 's@${M}@'"256"'@g' | sed 's@${N}@'"256"'@g' | sed 's@${K}@'"256"'@g' > matmul_linalg_cuda.mlir && \
#../../../../build_cuda/matmul-iree/src/tiling/add-tiling-attribute-pass matmul_linalg_cuda.mlir --compile-options "./matmul_linalg_cuda.bin"

