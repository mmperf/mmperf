# Method to generate matmul MLIR artifacts of specified sizes.
function(compile_mlir mlir_prefix B M N K TYPE)
    if ("${B}" STREQUAL "0")
        configure_file(${CMAKE_SOURCE_DIR}/src/matmul_MxNxK.mlir
          ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.mlir)
        set_property(
          DIRECTORY
          APPEND
          PROPERTY CMAKE_CONFIGURE_DEPENDS
          ${CMAKE_SOURCE_DIR}/src/matmul_MxNxK.mlir)
    else()
        configure_file(${CMAKE_SOURCE_DIR}/src/batch_matmul_BxMxNxK.mlir
          ${CMAKE_BINARY_DIR}/mlir-objs/${mlir_prefix}.mlir)
        set_property(
          DIRECTORY
          APPEND
          PROPERTY CMAKE_CONFIGURE_DEPENDS
          ${CMAKE_SOURCE_DIR}/src/batch_matmul_BxMxNxK.mlir)
    endif()
endfunction()

# Method to generate a IREE matmul binary for a specific backend and matrix size 
function(generate_matmul_binary mlir_file matrix_size backend B M N K TYPE NUM_REPS)

    string(CONCAT iree_executable_name "matmul_iree" ${backend} "_" ${matrix_size})
    string(CONCAT mlir_lib "matmul_" "_" ${backend} ${matrix_size} )

    #-------------------------------------------------------------------------------
    # Use `iree-translate` to transform an MLIR file into an VM bytcode module.
    #-------------------------------------------------------------------------------

    # Define arguments passed to iree-translate
    set(_ARGS)
    list(APPEND _ARGS "-iree-input-type=mhlo")
    list(APPEND _ARGS "--output-format=vm-bytecode")
    if(${backend} STREQUAL "dylib")
        list(APPEND _ARGS "-iree-hal-target-backends=llvm-cpu")
        list(APPEND _ARGS "-iree-llvmcpu-target-cpu-features=host")
        list(APPEND _ARGS "-iree-llvmcpu-link-embedded=true")
        list(APPEND _ARGS "-iree-llvmcpu-debug-symbols=false")
        list(APPEND _ARGS "-iree-vm-bytecode-module-strip-source-map=true")
        list(APPEND _ARGS "-iree-vm-emit-polyglot-zip=false")
    elseif(${backend} STREQUAL "cuda")
        list(APPEND _ARGS "-iree-hal-target-backends=cuda")
        list(APPEND _ARGS "-iree-hal-cuda-llvm-target-arch=sm_80")
    else()
        list(APPEND _ARGS "-iree-hal-target-backends=${backend}")
    endif()
    list(APPEND _ARGS "-iree-hal-benchmark-dispatch-repeat-count=100")
    list(APPEND _ARGS "${mlir_file}")
    list(APPEND _ARGS "-o")
    list(APPEND _ARGS "${mlir_lib}.vmfb")
    list(APPEND _ARGS "-iree-llvmcpu-embedded-linker-path=${_EMBEDDED_LINKER_TOOL_EXECUTABLE}")

    # Translate MLIR file to VM bytecode module
    add_custom_command(
        OUTPUT "${mlir_lib}.vmfb"
        COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_ARGS}
        DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE}
                ${_EMBEDDED_LINKER_TOOL_EXECUTABLE}
                ${mlir_file}
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
        matmul_generator.cc
        device_${backend}.c
    )

    # Set output directory to matmul build directory to plot performance.
    set_target_properties(${iree_executable_name} PROPERTIES OUTPUT_NAME ${iree_executable_name})
    set_target_properties(${iree_executable_name} PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/../matmul")
    set(BENCHMARK_INSTALL_DIR  ${CMAKE_BINARY_DIR}/../../benchmark-install)
    set(BENCHMARK_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../../external/benchmark)
    target_link_directories(${iree_executable_name} PRIVATE ${BENCHMARK_INSTALL_DIR}/lib)

    target_include_directories(${iree_executable_name}
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${BENCHMARK_SOURCE_DIR}/include
    )

    if((${backend} STREQUAL "dylib") OR (${backend} STREQUAL "vmvx"))
	set(IREE_CPU_DRIVER_LIB iree_hal_drivers_local_sync_sync_driver)
	if(IREE_CPU_MULTITHREADED)
	  set(IREE_CPU_DRIVER_LIB iree_hal_drivers_local_task_task_driver)
	endif()
        target_link_libraries(${iree_executable_name}
            ${MLIR_LIB}
            iree_base_base
            iree_hal_hal
            iree_hal_local_local
	    ${IREE_CPU_DRIVER_LIB}
            iree_hal_local_loaders_embedded_elf_loader
            iree_hal_local_loaders_vmvx_module_loader
            iree_modules_hal_hal
            iree_task_api
            iree_vm_vm
            iree_vm_bytecode_module
            benchmark
        )
    elseif(${backend} STREQUAL "cuda")
        target_link_libraries(${iree_executable_name}
            ${MLIR_LIB}
            iree_base_base
            iree_hal_hal
            iree_hal_drivers_cuda_registration_registration
            iree_modules_hal_hal
            iree_vm_vm
            iree_vm_bytecode_module
            benchmark
        )
    endif()

    target_compile_definitions(${iree_executable_name}
    PRIVATE "MATMUL_HEADER=\"${mlir_lib}.h\"")
    target_compile_definitions(${iree_executable_name} PRIVATE BDIM=${B})
    target_compile_definitions(${iree_executable_name} PRIVATE MDIM=${M})
    target_compile_definitions(${iree_executable_name} PRIVATE NDIM=${N})
    target_compile_definitions(${iree_executable_name} PRIVATE KDIM=${K})
    target_compile_definitions(${iree_executable_name} PRIVATE NUM_REPS=${NUM_REPS})
    target_compile_definitions(${iree_executable_name} PRIVATE FILE_NAME=${iree_executable_name}_perf.out)
    if (ENABLE_CHECK)
        target_compile_definitions(${iree_executable_name} PRIVATE ENABLE_CHECK)
    endif()
    if (USE_FP16)
        target_compile_definitions(${iree_executable_name} PRIVATE USE_FP16)
    endif()

endfunction()
