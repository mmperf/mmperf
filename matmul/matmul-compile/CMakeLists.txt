cmake_minimum_required(VERSION 3.18)
project(matmul_compile CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# Build flatbuffer tools
include_directories(${CMAKE_BINARY_DIR}/../flatbuffers-install/include)
include_directories(${CMAKE_BINARY_DIR}/../fbs/)
add_custom_command(OUTPUT compile_options_generated.h
  COMMAND ${CMAKE_BINARY_DIR}/../flatbuffers-install/bin/flatc
  --cpp --gen-object-api --raw-binary -o ${CMAKE_BINARY_DIR}/../fbs/
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_options.fbs
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/compile_options.fbs
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_custom_target(run ALL DEPENDS compile_options_generated.h)

# Build tool to compile mlir
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRExecutionEngine
  LLVMX86AsmParser
)

add_llvm_executable(matmul-compile matmul-compile.cpp)
add_dependencies(matmul-compile run)
llvm_update_compile_flags(matmul-compile)
target_link_libraries(matmul-compile PRIVATE ${LIBS})
mlir_check_all_link_libraries(matmul-compile)
