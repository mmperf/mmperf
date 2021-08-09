# Backends
option(USE_ACCELERATE "Enable Apple Accelerate Framework" OFF)
option(USE_BLASFEO "Enable BLASFEO" OFF)
option(USE_BLIS "Enable OpenBLAS" OFF)
option(USE_CUBLAS "Enable cublas" OFF)
option(USE_HALIDE "Enable Halide" OFF)
option(USE_MLIR "Enable MLIR" OFF)
option(USE_MLIR_CUDA "Enable MLIR CUDA" OFF)
option(USE_MKL "Enable MKL" OFF)
option(USE_NAIVE "Enable naive implementation by c/c++ loop" OFF)
option(USE_OPENBLAS "Enable OpenBLAS" OFF)
option(USE_RUY "Enable Ruy" OFF)
option(USE_TVM "Enable TVM" OFF)
option(USE_TVM_CUDA "Enable cuda for TVM" OFF)
option(USE_IREE "Enable IREE" OFF)

# Backend specific options
option(USE_TVM_TUNED "Use TVM Autotuned ops" OFF)
option(USE_MATMUL_COMPILE "Use matmul-compile instead of mlir-opt for small and medium sizes" OFF)

# General options
option(USE_COLUMN_MAJOR "Matrix format" OFF)
option(ENABLE_CHECK "Enable verification by naive implementation" ON)
set(SIZE_FILE ../../benchmark_sizes/benchmark_all_sizes.txt CACHE FILEPATH "File containing matrix sizes to be benchmarked")
set(TILE_FILE "" CACHE FILEPATH "File containing association between matrix size and tile size")
set(TARGET_CPU "haswell" CACHE STRING "Target CPU for MLIR")
set(VECTOR_WIDTH "256" CACHE STRING "Vector width for MLIR")
set(TILE_SIZES "" CACHE STRING "Tile sizes for MLIR")
set(COL_MAJOR_TILE_SIZES "" CACHE INTERNAL "Column Major MatMul Tile sizes for MLIR")
set(REGISTER_TILE_SIZES "" CACHE STRING "Register Tile sizes for MLIR")
set(COPY_FILL_TILE_SIZES "" CACHE STRING "Copy and Fill Tile sizes for MLIR")
set(NUM_REPS "100" CACHE STRING "Number of times to repeat the test")

if ("${TILE_FILE}" STREQUAL "")
  set(USE_NODAI OFF CACHE INTERNAL "Enable Nod.AI")
else()
  set(USE_NODAI ON CACHE INTERNAL "Enable Nod.AI")
endif()
option(SEARCH_MODE "Read tile size file as permutations instead of associations" OFF)
if (NOT (USE_ACCELERATE OR USE_MLIR OR USE_MLIR_CUDA OR USE_MKL OR USE_OPENBLAS OR USE_HALIDE OR USE_BLASFEO OR USE_RUY OR USE_NAIVE OR USE_NODAI OR USE_ACCELERATE OR USE_BLIS OR USE_TVM OR USE_TVM_CUDA OR USE_CUBLAS OR USE_IREE))
  message(FATAL_ERROR "No backend was enabled!")
endif()

# Set IREE backends to use
if(${USE_IREE} STREQUAL "ON")
  option(IREE_VMVX "Enable IREE vmvx (CPU) backend" OFF)
  option(IREE_DYLIB "Enable IREE dylib (CPU) backend" ON)
  option(IREE_CUDA "Enable IREE cuda (GPU) backend" ON)
endif()

set(VARS_TO_COPY
    USE_ACCELERATE
    USE_BLASFEO
    USE_BLIS
    USE_CUBLAS
    USE_HALIDE
    USE_MLIR
    USE_MLIR_CUDA
    USE_MKL
    USE_NAIVE
    USE_OPENBLAS
    USE_RUY
    USE_TVM
    USE_TVM_CUDA
    USE_TVM_TUNED
    USE_MATMUL_COMPILE
    USE_IREE
    IREE_CUDA
    IREE_VMVX
    USE_COLUMN_MAJOR
    ENABLE_CHECK
    SIZE_FILE
    TILE_FILE
    TARGET_CPU
    VECTOR_WIDTH
    TILE_SIZES
    COL_MAJOR_TILE_SIZES
    REGISTER_TILE_SIZES
    COPY_FILL_TILE_SIZES
    SEARCH_MODE
    NUM_REPS)
