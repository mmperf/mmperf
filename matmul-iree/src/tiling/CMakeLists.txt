# Build flatbuffer tools
include_directories(${CMAKE_BINARY_DIR}/../flatbuffers-install/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/fbs/)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/fbs/compile_options_generated.h
  COMMAND ${CMAKE_BINARY_DIR}/../flatbuffers-install/bin/flatc
  --cpp --gen-object-api --raw-binary -o ${CMAKE_CURRENT_BINARY_DIR}/fbs/
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_options.fbs
  DEPENDS
    ${CMAKE_BINARY_DIR}/../flatbuffers-install/bin/flatc
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_options.fbs
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Build tool to compile mlir
list(APPEND CMAKE_MODULE_PATH "${PROJECT_BINARY_DIR}/iree/llvm-project/lib/cmake/llvm")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_BINARY_DIR}/lib/cmake/mlir")

include(AddLLVM)
include(AddMLIR)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

include_directories(${IREE_SOURCE_DIR})
include_directories(${IREE_SOURCE_DIR}/third_party/llvm-project/llvm/include)
include_directories(${IREE_SOURCE_DIR}/third_party/llvm-project/mlir/include)
include_directories(${IREE_SOURCE_DIR}/third_party/mlir-hlo)
include_directories(${IREE_BINARY_DIR}/third_party/llvm-project/llvm/include)
include_directories(${IREE_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir-hlo)
include_directories(${IREE_BINARY_DIR})

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRExecutionEngine
  LLVMX86AsmParser
  iree::compiler::Codegen
  iree::compiler::Codegen::Dialect::IREECodegenDialect
  iree::compiler::Codegen::Utils
  iree::compiler::Dialect::HAL::IR
  iree::compiler::Dialect::HAL::IR::HALDialect
  iree::compiler::Dialect::Util::IR
)

add_llvm_executable(add-tiling-attribute-pass
  add-tiling-attribute-pass.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/fbs/compile_options_generated.h)
llvm_update_compile_flags(add-tiling-attribute-pass)
target_link_libraries(add-tiling-attribute-pass PRIVATE stdc++fs ${LIBS})
mlir_check_all_link_libraries(add-tiling-attribute-pass)
