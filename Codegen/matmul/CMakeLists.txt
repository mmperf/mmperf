cmake_minimum_required(VERSION 3.18)
project(matmul C CXX)

# Matrix sizes to benchmark
message(STATUS "Reading matrix sizes from ... ${SIZE_FILE}")
file(READ ${SIZE_FILE} MATRIX_SIZES)
string(REGEX REPLACE ";" "\\\\;" MATRIX_SIZES "${MATRIX_SIZES}")
string(REGEX REPLACE "\n" ";" MATRIX_SIZES "${MATRIX_SIZES}")

if(${USE_MLIR} STREQUAL "ON")
  function(compile_mlir mlir_prefix)
    set(OBJ ${CMAKE_BINARY_DIR}/${mlir_prefix}.o)
    add_custom_command(OUTPUT ${OBJ}
      COMMAND ${CMAKE_BINARY_DIR}/matmul-compile/matmul-compile ${CMAKE_CURRENT_LIST_DIR}/mlir/${mlir_prefix}.mlir
      COMMAND ${CMAKE_CXX_COMPILER} -O3 ${CMAKE_CURRENT_LIST_DIR}/${mlir_prefix}.ll
      -mllvm -enable-matrix -mllvm -matrix-allow-contract -mllvm -matrix-default-layout=row-major
      -c -o ${OBJ}
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
      DEPENDS matmul-compile
    )
  endfunction()

  # Build matmul-compile and pass
  add_subdirectory(matmul-compile)

  # Generate mlir files
  message(STATUS "Running ${CMAKE_CURRENT_LIST_DIR}/mlir/generate_mlir.py -matrix_sizes ${SIZE_FILE}")
  execute_process(
    COMMAND ${CMAKE_CURRENT_LIST_DIR}/mlir/generate_mlir.py -matrix_sizes ${SIZE_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/mlir
  )
endif()

if(${USE_HALIDE} STREQUAL "ON")
  list(APPEND CMAKE_PREFIX_PATH ${HALIDE_DIR}/lib/cmake/Halide)
  add_subdirectory(halide)
endif()

# Number of times to repeat the test
set(NUM_REPS 100)

# Create executables for each size and one for all
set(ALL_TARGETS "")
foreach(MATRIX_SIZE ${MATRIX_SIZES})
  set(SRC main.c)
  string(CONCAT MATMUL "matmul_" ${MATRIX_SIZE})
  message(STATUS "Compiling ${MATMUL}")
  if(${USE_MLIR} STREQUAL "ON")
    compile_mlir(${MATMUL})
    list(APPEND SRC ${CMAKE_BINARY_DIR}/${MATMUL}.o)
  endif()
  add_executable(${MATMUL} ${SRC})
  string(REPLACE "x" ";" SIZES ${MATRIX_SIZE})
  list(GET SIZES 0 M)
  list(GET SIZES 1 N)
  list(GET SIZES 2 K)
  target_compile_definitions(${MATMUL} PRIVATE MDIM=${M})
  target_compile_definitions(${MATMUL} PRIVATE NDIM=${N})
  target_compile_definitions(${MATMUL} PRIVATE KDIM=${K})
  target_compile_definitions(${MATMUL} PRIVATE NUM_REPS=${NUM_REPS})
  if(${USE_MKL} STREQUAL "ON")
    target_compile_definitions(${MATMUL} PRIVATE FILE_NAME=${MATMUL}_mkl_perf.out)
    target_compile_definitions(${MATMUL} PRIVATE MKL)
    target_include_directories(${MATMUL} PRIVATE ${MKL_DIR}/include)
    target_link_directories(${MATMUL} PRIVATE ${MKL_DIR}/lib/intel64)
    target_link_libraries(${MATMUL} PRIVATE mkl_intel_ilp64 mkl_gnu_thread mkl_core gomp)
  elseif(${USE_OPENBLAS} STREQUAL "ON")
    target_compile_definitions(${MATMUL} PRIVATE FILE_NAME=${MATMUL}_openblas_perf.out)
    target_compile_definitions(${MATMUL} PRIVATE OPENBLAS)
    target_link_libraries(${MATMUL} PRIVATE openblas)
  elseif(${USE_HALIDE} STREQUAL "ON")
    target_compile_definitions(${MATMUL} PRIVATE FILE_NAME=${MATMUL}_halide_perf.out)
    target_compile_definitions(${MATMUL} PRIVATE HALIDE)
    target_link_libraries(${MATMUL} PRIVATE halide_blas)
  elseif(${USE_MLIR} STREQUAL "ON")
    target_compile_definitions(${MATMUL} PRIVATE FILE_NAME=${MATMUL}_mlir_perf.out)
  endif()
  target_link_libraries(${MATMUL} PRIVATE m)
  list(APPEND ALL_TARGETS ${MATMUL})
endforeach()

set(args "")
foreach(target ${ALL_TARGETS})
  list(APPEND args COMMAND $<TARGET_FILE:${target}>)
endforeach()
add_custom_target(run_all_tests ${args})
