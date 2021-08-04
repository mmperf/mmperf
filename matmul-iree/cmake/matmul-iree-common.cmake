# Method to generate matmul MLIR artifacts of specified sizes.
function(compile_mlir mlir_prefix M N K)
    configure_file(${CMAKE_SOURCE_DIR}/src/matmul_MxNxK.mlir ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.mlir)
endfunction()

# Method that generated a matmul binary for a specific backend.
function(generate_matmul_binary matrix_size backend M N K mlir_lib NUM_REPS)
    string(CONCAT iree_executable_name "matmul_iree" ${backend} "_" ${matrix_size})

    add_executable(${iree_executable_name} "")
    target_sources(${iree_executable_name}
    PRIVATE
    matmul_generator.c
    device_${backend}.c
    )

    set_target_properties(${iree_executable_name} PROPERTIES OUTPUT_NAME ${iree_executable_name})
    # Setting the output directory to matmul build directory to plot performance.
    set_target_properties(${iree_executable_name} PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/../matmul")

    target_include_directories(${iree_executable_name}
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
    )

    target_link_libraries(${iree_executable_name}
        ${mlir_lib}
        iree_base_base
        iree_hal_hal
        iree_hal_${backend}_registration_registration
        iree_modules_hal_hal
        iree_vm_vm
        iree_vm_bytecode_module
    )

    target_compile_definitions(${iree_executable_name}
    PRIVATE "MATMUL_HEADER=\"matmul_${matrix_size}.h\"")
    target_compile_definitions(${iree_executable_name} PRIVATE MDIM=${M})
    target_compile_definitions(${iree_executable_name} PRIVATE NDIM=${N})
    target_compile_definitions(${iree_executable_name} PRIVATE KDIM=${K})
    target_compile_definitions(${iree_executable_name} PRIVATE NUM_REPS=${NUM_REPS})
    target_compile_definitions(${iree_executable_name} PRIVATE FILE_NAME=${iree_executable_name}_perf.out)
    if (ENABLE_CHECK)
        target_compile_definitions(${iree_executable_name} PRIVATE ENABLE_CHECK)
    endif()

endfunction()