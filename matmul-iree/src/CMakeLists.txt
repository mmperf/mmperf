# Matrix sizes to benchmark
message(STATUS "Reading matrix sizes from ... ${SIZE_FILE}")
file(READ ${SIZE_FILE} MATRIX_SIZES)
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${SIZE_FILE})
string(REGEX REPLACE "#[^\n]*\n" "" MATRIX_SIZES "${MATRIX_SIZES}")
string(REGEX REPLACE ";" "\\\\;" MATRIX_SIZES "${MATRIX_SIZES}")
string(REGEX REPLACE "\n" ";" MATRIX_SIZES "${MATRIX_SIZES}")
set(NUM_REPS ${NUM_REPS})

if(IREE_CPU_MULTITHREADED)
  message(STATUS "Enabling local task")
  add_compile_definitions(USE_LOCAL_TASK)
else()
  message(STATUS "Enabling local sync")
endif()

# Use `iree-translate` to transform an MLIR file into an VM bytcode module.
# Resolve the executable binary path for iree-translate from the target name.
set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:iree-compile>)
set(_EMBEDDED_LINKER_TOOL_EXECUTABLE $<TARGET_FILE:lld>)

message(STATUS "Generating mhlo.dot mlir files...")
foreach(MATRIX_SIZE ${MATRIX_SIZES})
  if(${USE_FP16} STREQUAL "ON")
    set(TYPE "f16")
  else()
    set(TYPE "f32")
  endif()

  if ("${MATRIX_SIZE}" STREQUAL "")
    continue()
  endif()

  string(CONCAT MATMUL "matmul_" ${MATRIX_SIZE})
  message(STATUS "Compiling ${MATMUL} ${TYPE}")

  string(REPLACE "x" ";" SIZES ${MATRIX_SIZE})
  list(LENGTH SIZES LEN)
  if ("${LEN}" STREQUAL "4")
    list(GET SIZES 0 B)
    list(GET SIZES 1 M)
    list(GET SIZES 2 N)
    list(GET SIZES 3 K)
  else()
    list(GET SIZES 0 M)
    list(GET SIZES 1 N)
    list(GET SIZES 2 K)
    set(B "0")
  endif()

  compile_mlir(${MATMUL} ${B} ${M} ${N} ${K} ${TYPE})

  set(MATMUL_MLIR_FILE ${CMAKE_BINARY_DIR}/mlir-objs/${MATMUL}.mlir)

  # Generate binary for each backend and specific matrix size
  if(${IREE_VMVX} STREQUAL "ON")
    generate_matmul_binary(${MATMUL_MLIR_FILE} ${MATRIX_SIZE} vmvx ${B} ${M} ${N} ${K} ${TYPE} ${NUM_REPS})
  endif()
  if(${IREE_LLVMCPU} STREQUAL "ON")
    generate_matmul_binary(${MATMUL_MLIR_FILE} ${MATRIX_SIZE} dylib ${B} ${M} ${N} ${K} ${TYPE} ${NUM_REPS})
  endif()
  if(${IREE_CUDA} STREQUAL "ON")
    generate_matmul_binary(${MATMUL_MLIR_FILE} ${MATRIX_SIZE} cuda ${B} ${M} ${N} ${K} ${TYPE} ${NUM_REPS})
  endif()
  
endforeach()

add_subdirectory(tiling)
