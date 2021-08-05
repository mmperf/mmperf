# Matrix sizes to benchmark
message(STATUS "Reading matrix sizes from ... ${SIZE_FILE}")
file(READ ${SIZE_FILE} MATRIX_SIZES)
string(REGEX REPLACE "#[^\n]*\n" "" MATRIX_SIZES "${MATRIX_SIZES}")
string(REGEX REPLACE ";" "\\\\;" MATRIX_SIZES "${MATRIX_SIZES}")
string(REGEX REPLACE "\n" ";" MATRIX_SIZES "${MATRIX_SIZES}")
set(NUM_REPS ${NUM_REPS})

# Use `iree-translate` to transform an MLIR file into an VM bytcode module.
# Resolve the executable binary path for iree-translate from the target name.
set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:iree_tools_iree-translate>)

message(STATUS "Generating mhlo.dot mlir files...")
foreach(MATRIX_SIZE ${MATRIX_SIZES})
  if ("${MATRIX_SIZE}" STREQUAL "")
    continue()
  endif()

  string(CONCAT MATMUL "matmul_" ${MATRIX_SIZE})
  message(STATUS "Compiling ${MATMUL}")

  string(REPLACE "x" ";" SIZES ${MATRIX_SIZE})
  list(GET SIZES 0 M)
  list(GET SIZES 1 N)
  list(GET SIZES 2 K)

  compile_mlir(${MATMUL} ${M} ${N} ${K})

  set(MATMUL_MLIR_FILE ${CMAKE_BINARY_DIR}/mlir-objs/${MATMUL}.mlir)

  # Generate binary for each backend and specific matrix size
  if(${IREE_VMVX} STREQUAL "ON")
    generate_matmul_binary(${MATMUL_MLIR_FILE} ${MATRIX_SIZE} vmvx ${M} ${N} ${K} ${NUM_REPS})
  endif()
  if(${IREE_DYLIB} STREQUAL "ON")
    generate_matmul_binary(${MATMUL_MLIR_FILE} ${MATRIX_SIZE} dylib ${M} ${N} ${K} ${NUM_REPS})
  endif()
  if(${IREE_CUDA} STREQUAL "ON")
    generate_matmul_binary(${MATMUL_MLIR_FILE} ${MATRIX_SIZE} cuda ${M} ${N} ${K} ${NUM_REPS})
  endif()
  
endforeach()
