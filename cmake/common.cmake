option(USE_ACCELERATE "Enable Apple Accelerate Framework" OFF)
option(USE_MLIR "Enable MLIR" OFF)
option(USE_MKL "Enable MKL" OFF)
option(USE_OPENBLAS "Enable OpenBLAS" OFF)
option(USE_HALIDE "Enable Halide" OFF)
option(USE_RUY "Enable Ruy" OFF)
option(USE_NAIVE "Enable naive implementation by c/c++ loop" OFF)
option(USE_MATMUL_COMPILE "Use matmul-compile instead of mlir-opt for small and medium sizes" OFF)
option(USE_COLUMN_MAJOR "Matrix format" OFF)
if (NOT (USE_ACCELERATE OR USE_MLIR OR USE_MKL OR USE_OPENBLAS OR USE_HALIDE OR USE_RUY OR USE_NAIVE))
    message(FATAL_ERROR "No backend was enabled!")
endif()

option(ENABLE_CHECK "Enable verification by naive implementation" ON)

set(SIZE_FILE ../../benchmark_sizes/benchmark_all_sizes.txt CACHE FILEPATH "File containing matrix sizes to be benchmarked")
set(TILE_FILE "" CACHE FILEPATH "File containing association between matrix size and tile size")
set(TARGET_CPU "haswell" CACHE STRING "Target CPU for MLIR")
set(VECTOR_WIDTH "256" CACHE STRING "Vector width for MLIR")
set(TILE_SIZES "" CACHE STRING "Tile sizes for MLIR")
set(COL_MAJOR_TILE_SIZES "" CACHE INTERNAL "Column Major MatMul Tile sizes for MLIR")
set(REGISTER_TILE_SIZES "" CACHE STRING "Register Tile sizes for MLIR")
set(COPY_FILL_TILE_SIZES "" CACHE STRING "Copy and Fill Tile sizes for MLIR")

if ("${TILE_FILE}" STREQUAL "")
  set(USE_NODAI OFF CACHE INTERNAL "Enable Nod.AI")
else()
  set(USE_NODAI ON CACHE INTERNAL "Enable Nod.AI")
endif()

set(VARS_TO_COPY
    USE_ACCELERATE
    USE_MLIR
    USE_MKL
    USE_OPENBLAS
    USE_HALIDE
    USE_RUY
    USE_NAIVE
    USE_MATMUL_COMPILE
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
    ENABLE_CHECK)
