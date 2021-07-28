cmake_minimum_required(VERSION 3.13.4)

#-------------------------------------------------------------------------------
# Project configuration
#-------------------------------------------------------------------------------

project(matmul-iree VERSION 1.0 LANGUAGES CXX C)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# NOTE: The RTTI setting must match what LLVM was compiled with and IREE
#       defaults to RTTI disabled in `iree_copts.cmake`.
#       RTTI is disabled for `dialect` target, but you may want to consider to
#       to pass `-fno-rtti` as a global compile option by adding the line
# add_compile_options(-fno-rtti)

#-------------------------------------------------------------------------------
# Third-party dependencies
#-------------------------------------------------------------------------------

# Extend module path to allow submodules to find AddMLIR
list(APPEND CMAKE_MODULE_PATH "${PROJECT_BINARY_DIR}/lib/cmake/mlir")

add_subdirectory(${IREE_SRC} iree EXCLUDE_FROM_ALL)

#-------------------------------------------------------------------------------
# Top-level components
#-------------------------------------------------------------------------------

add_subdirectory(src)