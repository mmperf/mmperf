cmake_minimum_required(VERSION 3.18)
project(matmul C)

# Build matmul-compile and pass
add_subdirectory(matmul-compile)

function(compile_mlir mlir_prefix)
  set(OBJ ${CMAKE_BINARY_DIR}/${mlir_prefix}.o)
  add_custom_command(OUTPUT ${OBJ}
    COMMAND ${CMAKE_BINARY_DIR}/matmul-compile/matmul-compile ${CMAKE_CURRENT_LIST_DIR}/mlir/${mlir_prefix}.mlir
    COMMAND ${CMAKE_CXX_COMPILER} -O3 ${CMAKE_CURRENT_LIST_DIR}/${mlir_prefix}.ll -c -o ${OBJ}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS matmul-compile
  )
endfunction()

# Matrix sizes to benchmark
set(MATRIX_SIZES
  "18x32x96" "24x64x96" "24x64x512" "48x64x128" "192x64x128" # SMALL
  "192x128x128" "192x256x256" "384x256x256" "480x512x16" "480x512x256" # MEDIUM
  "1024x1024x1024" "1020x1152x1152" "1920x2304x2304" "2304x2304x2560" # LARGE
)

# Number of times to repeat the test
set(NUM_REPS 100)

# Create executables for each size and one for all
set(ALL_TARGETS "")
foreach(MATRIX_SIZE ${MATRIX_SIZES})
  set(SRC main.c)
  string(CONCAT MATMUL "matmul_" ${MATRIX_SIZE})
  message(STATUS "Compiling ${MATMUL}")
  if(${USE_MKL} STREQUAL "OFF")
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
  else()
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
