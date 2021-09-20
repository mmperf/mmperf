../../../../build/flatbuffers-install/bin/flatc -b ../compile_options.fbs iree-tiling.json && \
cat ../../matmul_MxNxK.mlir | sed 's@${M}@'"256"'@g' | sed 's@${N}@'"256"'@g' | sed 's@${K}@'"256"'@g' > matmul.mlir && \
../../../../build/matmul-iree/src/tiling/add-tiling-attribute-pass matmul.mlir --compile-options "./iree-tiling.bin"
