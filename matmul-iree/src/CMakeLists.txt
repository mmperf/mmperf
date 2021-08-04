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

  #-------------------------------------------------------------------------------
  # Use `iree-translate` to transform an MLIR file into an VM bytcode module.
  #-------------------------------------------------------------------------------

  # Define arguments passed to iree-translate
  set(_ARGS)
  list(APPEND _ARGS "-iree-input-type=mhlo")
  list(APPEND _ARGS "-iree-mlir-to-vm-bytecode-module")
  # Backend here is just used to create bytecode module and doesn't reflect the
  # backend used to run the program. Could be set to any of the available ones.
  list(APPEND _ARGS "-iree-hal-target-backends=vmvx")
  list(APPEND _ARGS "${MATMUL_MLIR_FILE}")
  list(APPEND _ARGS "-o")
  list(APPEND _ARGS "${MATMUL}.vmfb")

  # Translate MLIR file to VM bytecode module
  add_custom_command(
    OUTPUT "${MATMUL}.vmfb"
    COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_ARGS}
    DEPENDS iree_tools_iree-translate
  )

  #-------------------------------------------------------------------------------
  # Embedd the VM bytcode module into a c file via `generate_embed_data`.
  #-------------------------------------------------------------------------------

  # Define arguments passed to generate_embed_data
  set(_ARGS)
  list(APPEND _ARGS "--output_header=${MATMUL}.h")
  list(APPEND _ARGS "--output_impl=${MATMUL}.c")
  list(APPEND _ARGS "--identifier=matmul")
  list(APPEND _ARGS "--flatten")
  list(APPEND _ARGS "${MATMUL}.vmfb")

  # Embed VM bytecode module into c source file
  add_custom_command(
    OUTPUT
      "${MATMUL}.h"
      "${MATMUL}.c"
    COMMAND generate_embed_data ${_ARGS}
    DEPENDS generate_embed_data ${MATMUL}.vmfb
  )

  #-------------------------------------------------------------------------------
  # Create a library and thus a CMake target.
  #-------------------------------------------------------------------------------

  string(CONCAT MLIR_LIB ${MATMUL} "_c")
  add_library(${MLIR_LIB} STATIC "")
  target_sources(${MLIR_LIB}
    PRIVATE
    ${MATMUL}.c
    ${MATMUL}.h
  )

  #-------------------------------------------------------------------------------
  # Build the excutable for each available backend.
  #-------------------------------------------------------------------------------
  if(${IREE_VMVX} STREQUAL "ON")
    generate_matmul_binary(${MATRIX_SIZE} vmvx ${M} ${N} ${K} ${MLIR_LIB} ${NUM_REPS})
  endif()
  if(${IREE_CUDA} STREQUAL "ON")
    generate_matmul_binary(${MATRIX_SIZE} cuda ${M} ${N} ${K} ${MLIR_LIB} ${NUM_REPS})
  endif()
  
endforeach()

# file(GLOB iree_binaries "${CMAKE_BINARY_DIR}/src/matmul_iree*")
# foreach(iree_binary ${iree_binaries})
#     # file(RENAME ${iree_binary}
#     #     ${CMAKE_BINARY_DIR}/../matmul/${iree_executable_name})
#     message("xXxXxX-----" ${iree_binary})
# endforeach()