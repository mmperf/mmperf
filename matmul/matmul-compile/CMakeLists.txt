cmake_minimum_required(VERSION 3.18)
project(matmul_compile CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(LLVM_DIR_ ${LLVM_DIR})
find_package(MLIR REQUIRED CONFIG)
set(LLVM_DIR ${LLVM_DIR_})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Build tool to compile mlir
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
message(STATUS "${dialect_libs}")
if (DEFINED MLIR_BUILD)
  link_directories(${MLIR_BUILD})
  set(LLVM_X86ASM_PARSER ${MLIR_BUILD}/lib/libLLVMX86AsmParser.a)
endif()

if (DEFINED LLVM_DIR)
  if(EXISTS ${LLVM_DIR}/lib/libLLVMX86AsmParser.so)
    set(LLVM_X86ASM_PARSER ${LLVM_DIR}/lib/libLLVMX86AsmParser.so)
  else()
    if(EXISTS ${LLVM_DIR}/lib/libLLVMX86AsmParser.a)
      set(LLVM_X86ASM_PARSER ${LLVM_DIR}/lib/libLLVMX86AsmParser.a)
    else()
      message(FATAL_ERROR "Neither of libLLVMX86AsmParser.a nor \
                            libLLVMX86AsmParser.so was not found, aborting." )
    endif()
  endif()
endif()

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIRExecutionEngine
  ${LLVM_X86ASM_PARSER}
)
add_llvm_executable(matmul-compile matmul-compile.cpp)
llvm_update_compile_flags(matmul-compile)
target_link_libraries(matmul-compile PRIVATE ${LIBS})
mlir_check_all_link_libraries(matmul-compile)
