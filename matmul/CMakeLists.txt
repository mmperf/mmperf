cmake_minimum_required(VERSION 3.18)
project(matmul C CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/../cmake/")
include(common)

# Matrix sizes to benchmark
message(STATUS "Reading matrix sizes from ... ${SIZE_FILE}")
file(READ ${SIZE_FILE} MATRIX_SIZES)
string(REGEX REPLACE "#[^\n]*\n" "" MATRIX_SIZES "${MATRIX_SIZES}")
string(REGEX REPLACE ";" "\\\\;" MATRIX_SIZES "${MATRIX_SIZES}")
string(REGEX REPLACE "\n" ";" MATRIX_SIZES "${MATRIX_SIZES}")

if(USE_NODAI)
  message(STATUS "Reading tile sizes from ... ${TILE_FILE}")
  file(READ "${TILE_FILE}" TILE_SIZE_LIST)
  string(REGEX REPLACE "#[^\n]*\n" "" TILE_SIZE_LIST "${TILE_SIZE_LIST}")
  string(REGEX REPLACE ";" "\\\\;" TILE_SIZE_LIST "${TILE_SIZE_LIST}")
  string(REGEX REPLACE "\n" ";" TILE_SIZE_LIST "${TILE_SIZE_LIST}")
  foreach(TILE_SIZE IN LISTS TILE_SIZE_LIST)
    if (TILE_SIZE)
      string(REPLACE "," ";" pair ${TILE_SIZE})
      list(GET pair 0 matrix_size)
      list(GET pair 1 tile_size)
      string(REPLACE "x" " " TILE_FOR_MATRIX_${matrix_size} ${tile_size})
      message(STATUS "Matrix ${matrix_size} has tile size ${tile_size}")
    endif()
  endforeach()
endif()

if(USE_MLIR OR USE_NODAI)
  function(compile_mlir mlir_prefix M N K nodai_mode)
    set(OBJ ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.o)
    if(nodai_mode)
      if(NOT DEFINED "TILE_FOR_MATRIX_${M}x${N}x${K}")
        message(FATAL_ERROR "A tile size for ${M}x${N}x${K} wasn't provided in ${TILE_FILE}")
      endif()
      message(STATUS "Using nodai tiles sizes ...")
      set(TILE_SIZES_LOCAL ${TILE_FOR_MATRIX_${M}x${N}x${K}})
    else()
      if("${TILE_SIZES}" STREQUAL "")
        message(STATUS "Using default tile sizes")
        if(M GREATER 1000)
          set(TILE_SIZES_LOCAL "128 128 256")
          set(REGISTER_TILE_SIZES "8 16 8")
          set(COPY_FILL_TILE_SIZES "4 16")
          set(COL_MAJOR_TILE_SIZES "16 0 32")
        else()
          set(TILE_SIZES_LOCAL "12 32 16")
          set(REGISTER_TILE_SIZES "")
          set(COPY_FILL_TILE_SIZES "4 16")
          set(COL_MAJOR_TILE_SIZES "16 0 32")
        endif()
      else()
        set(TILE_SIZES_LOCAL "${TILE_SIZES}")
        message(STATUS "Using custom tile sizes ...")
      endif()
    endif()
    message(STATUS "Tile sizes = ${TILE_SIZES_LOCAL}")
    message(STATUS "Register Tile sizes = ${REGISTER_TILE_SIZES}")
    message(STATUS "Copy Fill Tile sizes = ${COPY_FILL_TILE_SIZES}")
    if(${USE_COLUMN_MAJOR} STREQUAL "ON")
      message(STATUS "Compiling linalg.matmul_column_major ...")
      configure_file(${CMAKE_SOURCE_DIR}/matmul_column_major_mlir_MxNxK.mlir.in ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.mlir)
    else()
      message(STATUS "Compiling linalg.matmul ...")
      configure_file(${CMAKE_SOURCE_DIR}/matmul_mlir_MxNxK.mlir.in ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.mlir)
    endif()
    # For larger sizes, we will still use matmul-compile
    if((${USE_MATMUL_COMPILE} STREQUAL "ON") OR (M GREATER 1000))
      add_custom_command(OUTPUT ${OBJ}
        COMMAND ${CMAKE_BINARY_DIR}/matmul-compile/matmul-compile
        -M=${M} -N=${N} -K=${K} -target-cpu=${TARGET_CPU} -vector-width=${VECTOR_WIDTH}
        -tile-sizes=${TILE_SIZES_LOCAL} -register-tile-sizes=${REGISTER_TILE_SIZES}
        -copy-fill-tile-sizes=${COPY_FILL_TILE_SIZES}
        -promote=true -promote-full-tile-pad=true -vectorize=true
        -vectorize-to=outerproduct -split-vector-transfers-to=vector-transfers
        -unroll-vector-transfers=true
        ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.mlir
        COMMAND ${CMAKE_CXX_COMPILER} -O3 ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.ll
        -mllvm -enable-matrix -mllvm -matrix-allow-contract -mllvm -matrix-default-layout=row-major
        -c -o ${OBJ} -g
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mlir-objs
        DEPENDS matmul-compile
      )
    else()
      string(REPLACE " " "," TILE_SIZES_LOCAL "${TILE_SIZES_LOCAL}")
      string(REPLACE " " "," COL_MAJOR_TILE_SIZES "${COL_MAJOR_TILE_SIZES}")
      string(REPLACE " " "," COPY_FILL_TILE_SIZES "${COPY_FILL_TILE_SIZES}")
      set(MLIR_OPT ${CMAKE_BINARY_DIR}/../mlir-install/bin/mlir-opt)
      set(MLIR_TRANSLATE ${CMAKE_BINARY_DIR}/../mlir-install/bin/mlir-translate)
      add_custom_command(OUTPUT ${OBJ}
	COMMAND cat ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.mlir |
	${MLIR_OPT} -test-linalg-codegen-strategy=\"anchor-op=linalg.matmul_column_major register-tile-sizes=${COL_MAJOR_TILE_SIZES} vectorize\" |
	${MLIR_OPT} -test-linalg-codegen-strategy=\"anchor-op=linalg.matmul register-tile-sizes=${TILE_SIZES_LOCAL} vectorize\" |
	${MLIR_OPT} -test-linalg-codegen-strategy=\"anchor-op=linalg.fill register-tile-sizes=${COPY_FILL_TILE_SIZES} vectorize\" |
	${MLIR_OPT} -test-linalg-codegen-strategy=\"anchor-op=linalg.copy register-tile-sizes=${COPY_FILL_TILE_SIZES} vectorize\" |
	${MLIR_OPT} -canonicalize -convert-vector-to-scf -lower-affine -convert-linalg-to-loops -canonicalize -convert-scf-to-std -convert-vector-to-llvm |
	${MLIR_TRANSLATE} --mlir-to-llvmir -o=${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.ll
        COMMAND ${CMAKE_CXX_COMPILER} -O3 ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.ll
        -mllvm -enable-matrix -mllvm -matrix-allow-contract -mllvm -matrix-default-layout=row-major
        -c -o ${OBJ} -g
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/mlir-objs
      )
    endif()
  endfunction()

  # Build matmul-compile and pass
  add_subdirectory(matmul-compile)
endif()

if(${USE_HALIDE} STREQUAL "ON")
  list(APPEND CMAKE_PREFIX_PATH ${HALIDE_DIR}/lib/cmake/Halide)
  add_subdirectory(halide)
endif()

if(${USE_RUY} STREQUAL "ON")
  add_subdirectory(${RUY_SOURCE} ruy)
endif()

# Number of times to repeat the test
set(NUM_REPS 100)

set(BACKENDS "")
if(USE_ACCELERATE)
  list(APPEND BACKENDS accelerate)
endif()
if(USE_MKL)
  list(APPEND BACKENDS mkl)
endif()
if(USE_MLIR)
  list(APPEND BACKENDS mlir)
endif()
if(USE_NODAI)
  list(APPEND BACKENDS nodai)
endif()
if(USE_OPENBLAS)
  list(APPEND BACKENDS openblas)
endif()
if(USE_HALIDE)
  list(APPEND BACKENDS halide)
endif()
if(USE_RUY)
  list(APPEND BACKENDS ruy)
endif()
if(USE_NAIVE)
    list(APPEND BACKENDS naive)
endif()

# Create executables for each size and one for all
set(ALL_TARGETS "")
foreach(BACKEND ${BACKENDS})
  foreach(MATRIX_SIZE ${MATRIX_SIZES})
    set(SRC main.cc)
    string(CONCAT MATMUL "matmul_" ${BACKEND} "_" ${MATRIX_SIZE})
    message(STATUS "Compiling ${MATMUL}")

    string(REPLACE "x" ";" SIZES ${MATRIX_SIZE})
    list(GET SIZES 0 M)
    list(GET SIZES 1 N)
    list(GET SIZES 2 K)

    if(${BACKEND} STREQUAL "mlir")
      compile_mlir(${MATMUL} ${M} ${N} ${K} OFF)
      list(APPEND SRC ${CMAKE_BINARY_DIR}/mlir-objs/${MATMUL}.o)
    elseif(${BACKEND} STREQUAL "nodai")
      compile_mlir(${MATMUL} ${M} ${N} ${K} ON)
      list(APPEND SRC ${CMAKE_BINARY_DIR}/mlir-objs/${MATMUL}.o)
    endif()
    add_executable(${MATMUL} ${SRC})
    target_compile_definitions(${MATMUL} PRIVATE MDIM=${M})
    target_compile_definitions(${MATMUL} PRIVATE NDIM=${N})
    target_compile_definitions(${MATMUL} PRIVATE KDIM=${K})
    target_compile_definitions(${MATMUL} PRIVATE NUM_REPS=${NUM_REPS})
    target_compile_definitions(${MATMUL} PRIVATE FILE_NAME=${MATMUL}_perf.out)
    if(${USE_COLUMN_MAJOR} STREQUAL "ON")
      target_compile_definitions(${MATMUL} PRIVATE COLUMN_MAJOR)
    endif()
    if (ENABLE_CHECK)
        target_compile_definitions(${MATMUL} PRIVATE ENABLE_CHECK)
    endif()
    if(${BACKEND} STREQUAL "mkl")
      target_compile_definitions(${MATMUL} PRIVATE MKL)
      target_include_directories(${MATMUL} PRIVATE ${MKL_DIR}/include)
      target_link_directories(${MATMUL} PRIVATE ${MKL_DIR}/lib/intel64)
      target_link_libraries(${MATMUL} PRIVATE mkl_intel_ilp64 mkl_gnu_thread mkl_core gomp)
    elseif(${BACKEND} STREQUAL "accelerate")
      set(BLA_VENDOR Apple)
      find_package(BLAS)
      target_include_directories(${MATMUL} PRIVATE /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers)
      target_compile_definitions(${MATMUL} PRIVATE ACCELERATE)
      target_link_libraries(${MATMUL} PRIVATE ${BLAS_LIBRARIES})
    elseif(${BACKEND} STREQUAL "openblas")
      target_compile_definitions(${MATMUL} PRIVATE OPENBLAS)
      target_link_libraries(${MATMUL} PRIVATE openblas)
    elseif(${BACKEND} STREQUAL "halide")
      target_compile_definitions(${MATMUL} PRIVATE HALIDE)
      target_link_libraries(${MATMUL} PRIVATE halide_blas)
    elseif(${BACKEND} STREQUAL "ruy")
      target_compile_definitions(${MATMUL} PRIVATE RUY)
      target_include_directories(${MATMUL} PRIVATE ${RUY_SOURCE})
      target_link_libraries(${MATMUL} PRIVATE ruy)
    elseif(${BACKEND} STREQUAL "mlir")
      target_compile_definitions(${MATMUL} PRIVATE MLIR)
    elseif(${BACKEND} STREQUAL "nodai")
      target_compile_definitions(${MATMUL} PRIVATE MLIR)
    elseif(${BACKEND} STREQUAL "naive")
      target_compile_definitions(${MATMUL} PRIVATE NAIVE)
      target_compile_options(${MATMUL} PRIVATE -O3)
    endif()
    target_link_libraries(${MATMUL} PRIVATE m)
    list(APPEND ALL_TARGETS ${MATMUL})
  endforeach()
endforeach()

add_custom_target(run_all_tests
    ${CMAKE_SOURCE_DIR}/../mmperf.py
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/../results/${TARGET_CPU}
    DEPENDS ${ALL_TARGETS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..)
