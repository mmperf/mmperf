# Generate MLIR artifacts of specified sizes
function(compile_mlir mlir_prefix M N K)
    configure_file(${CMAKE_SOURCE_DIR}/src/matmul_MxNxK.mlir ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.mlir)
endfunction()

# Matrix sizes to benchmark
message(STATUS "Reading matrix sizes from ... ${SIZE_FILE}")
file(READ ${SIZE_FILE} MATRIX_SIZES)
string(REGEX REPLACE "#[^\n]*\n" "" MATRIX_SIZES "${MATRIX_SIZES}")
string(REGEX REPLACE ";" "\\\\;" MATRIX_SIZES "${MATRIX_SIZES}")
string(REGEX REPLACE "\n" ";" MATRIX_SIZES "${MATRIX_SIZES}")

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
  list(APPEND _ARGS "-iree-hal-target-backends=vmvx")
  # Uncomment the line below to use vulkan-spirv backend
  #list(APPEND _ARGS "-iree-hal-target-backends=vulkan-spirv")
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
  # Build the excutable.
  #-------------------------------------------------------------------------------
  add_executable(${MATMUL} "")
  target_sources(${MATMUL}
    PRIVATE
    matmul_generator.c
    device_vmvx.c
  )

  set_target_properties(${MATMUL} PROPERTIES OUTPUT_NAME ${MATMUL})

  target_include_directories(${MATMUL}
    PUBLIC
      ${CMAKE_CURRENT_BINARY_DIR}
  )

  target_link_libraries(${MATMUL}
    ${MLIR_LIB}
    iree_base_base
    iree_hal_hal
    iree_hal_vmvx_registration_registration
    iree_modules_hal_hal
    iree_vm_vm
    iree_vm_bytecode_module
  )

  target_compile_definitions(${MATMUL}
   PRIVATE "MATMUL_HEADER=\"${MATMUL}.h\"")
  target_compile_definitions(${MATMUL} PRIVATE MDIM=${M})
  target_compile_definitions(${MATMUL} PRIVATE NDIM=${N})
  target_compile_definitions(${MATMUL} PRIVATE KDIM=${K})

endforeach()