option(USE_MLIR "Enable MLIR" OFF)
option(USE_MKL "Enable MKL" OFF)
option(USE_OPENBLAS "Enable OpenBLAS" OFF)
option(USE_HALIDE "Enable Halide" OFF)
option(USE_RUY "Enable Ruy" OFF)
option(USE_NAIVE "Enable naive implementation by c/c++ loop" OFF)
if (NOT (USE_MLIR OR USE_MKL OR USE_OPENBLAS OR USE_HALIDE OR USE_RUY OR USE_NAIVE))
    message(FATAL_ERROR "No backend was enabled!")
endif()

option(ENABLE_CHECK "Enable verification by naive implementation" ON)

set(SIZE_FILE ${CMAKE_CURRENT_LIST_DIR}/../benchmark_sizes/benchmark_all_sizes.txt CACHE INTERNAL "File containing matrix sizes to be benchmarked")
set(TARGET_CPU "haswell" CACHE INTERNAL "Target CPU for MLIR")
set(VECTOR_WIDTH "256" CACHE INTERNAL "Vector width for MLIR")
set(TILE_SIZES "" CACHE INTERNAL "Tile sizes for MLIR")
set(REGISTER_TILE_SIZES "" CACHE INTERNAL "Register Tile sizes for MLIR")
set(COPY_FILL_TILE_SIZES "" CACHE INTERNAL "Copy and Fill Tile sizes for MLIR")

set(VARS_TO_COPY
    USE_MLIR
    USE_MKL
    USE_OPENBLAS
    USE_HALIDE
    USE_RUY
    USE_NAIVE
    ENABLE_CHECK
    SIZE_FILE
    TARGET_CPU
    VECTOR_WIDTH
    TILE_SIZES
    REGISTER_TILE_SIZES
    COPY_FILL_TILE_SIZES)
