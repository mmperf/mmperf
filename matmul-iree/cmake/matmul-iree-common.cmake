# Method to generate matmul MLIR artifacts of specified sizes.
function(compile_mlir mlir_prefix M N K)
    configure_file(${CMAKE_SOURCE_DIR}/src/matmul_MxNxK.mlir ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.mlir)
endfunction()

# Method to generate a IREE matmul binary for a specific backend and matrix size 
function(generate_matmul_binary mlir_file matrix_size backend M N K NUM_REPS)

    string(CONCAT iree_executable_name "matmul_iree" ${backend} "_" ${matrix_size})
    string(CONCAT mlir_lib "matmul_" "_" ${backend} ${matrix_size} )

    #-------------------------------------------------------------------------------
    # Use `iree-translate` to transform an MLIR file into an VM bytcode module.
    #-------------------------------------------------------------------------------

    # Define arguments passed to iree-translate
    set(_ARGS)
    list(APPEND _ARGS "-iree-input-type=mhlo")
    list(APPEND _ARGS "-iree-mlir-to-vm-bytecode-module")
    list(APPEND _ARGS "-iree-hal-target-backends=${backend}")
    list(APPEND _ARGS "${mlir_file}")
    list(APPEND _ARGS "-o")
    list(APPEND _ARGS "${mlir_lib}.vmfb")

    # Translate MLIR file to VM bytecode module
    add_custom_command(
        OUTPUT "${mlir_lib}.vmfb"
        COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_ARGS}
        DEPENDS iree_tools_iree-translate
    )

    #-------------------------------------------------------------------------------
    # Embedd the VM bytcode module into a c file via `generate_embed_data`.
    #-------------------------------------------------------------------------------

    # Define arguments passed to generate_embed_data
    set(_ARGS)
    list(APPEND _ARGS "--output_header=${mlir_lib}.h")
    list(APPEND _ARGS "--output_impl=${mlir_lib}.c")
    list(APPEND _ARGS "--identifier=matmul")
    list(APPEND _ARGS "--flatten")
    list(APPEND _ARGS "${mlir_lib}.vmfb")

    # Embed VM bytecode module into c source file
    add_custom_command(
        OUTPUT
            "${mlir_lib}.h"
            "${mlir_lib}.c"
        COMMAND generate_embed_data ${_ARGS}
        DEPENDS generate_embed_data ${mlir_lib}.vmfb
    )

    #-------------------------------------------------------------------------------
    # Create a library and thus a CMake target.
    #-------------------------------------------------------------------------------

    string(CONCAT MLIR_LIB ${mlir_lib} "_c")
    add_library(${MLIR_LIB} STATIC "")
    target_sources(${MLIR_LIB}
    PRIVATE
        ${mlir_lib}.c
        ${mlir_lib}.h
    )

    #-------------------------------------------------------------------------------
    # Build the executable.
    #-------------------------------------------------------------------------------

    add_executable(${iree_executable_name} "")
    target_sources(${iree_executable_name}
        PRIVATE
        matmul_generator.c
        device_${backend}.c
    )

    # Set output directory to matmul build directory to plot performance.
    set_target_properties(${iree_executable_name} PROPERTIES OUTPUT_NAME ${iree_executable_name})
    set_target_properties(${iree_executable_name} PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/../matmul")

    target_include_directories(${iree_executable_name}
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
    )

    target_link_libraries(${iree_executable_name}
        ${MLIR_LIB}
        iree_base_base
        iree_hal_hal
        iree_hal_${backend}_registration_registration
        iree_modules_hal_hal
        iree_vm_vm
        iree_vm_bytecode_module
    )

    target_compile_definitions(${iree_executable_name}
    PRIVATE "MATMUL_HEADER=\"${mlir_lib}.h\"")
    target_compile_definitions(${iree_executable_name} PRIVATE MDIM=${M})
    target_compile_definitions(${iree_executable_name} PRIVATE NDIM=${N})
    target_compile_definitions(${iree_executable_name} PRIVATE KDIM=${K})
    target_compile_definitions(${iree_executable_name} PRIVATE NUM_REPS=${NUM_REPS})
    target_compile_definitions(${iree_executable_name} PRIVATE FILE_NAME=${iree_executable_name}_perf.out)
    if (ENABLE_CHECK)
        target_compile_definitions(${iree_executable_name} PRIVATE ENABLE_CHECK)
    endif()

endfunction()
