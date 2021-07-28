# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-------------------------------------------------------------------------------
# Use `iree-translate` to transform an MLIR file into an VM bytcode module.
#-------------------------------------------------------------------------------

# Resolve the executable binary path for iree-translate from the target name.
set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:iree_tools_iree-translate>)

# Define arguments passed to iree-translate
set(_ARGS)
list(APPEND _ARGS "-iree-input-type=mhlo")
list(APPEND _ARGS "-iree-mlir-to-vm-bytecode-module")
list(APPEND _ARGS "-iree-hal-target-backends=vmvx")
# Uncomment the line below to use vulkan-spirv backend
#list(APPEND _ARGS "-iree-hal-target-backends=vulkan-spirv")
list(APPEND _ARGS "${CMAKE_CURRENT_SOURCE_DIR}/matmul.mlir")
list(APPEND _ARGS "-o")
list(APPEND _ARGS "matmul.vmfb")

# Translate MLIR file to VM bytecode module
add_custom_command(
  OUTPUT "matmul.vmfb"
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_ARGS}
  DEPENDS iree_tools_iree-translate
)

#-------------------------------------------------------------------------------
# Embedd the VM bytcode module into a c file via `generate_embed_data`.
#-------------------------------------------------------------------------------

# Define arguments passed to generate_embed_data
set(_ARGS)
list(APPEND _ARGS "--output_header=matmul.h")
list(APPEND _ARGS "--output_impl=matmul.c")
list(APPEND _ARGS "--identifier=matmul")
list(APPEND _ARGS "--flatten")
list(APPEND _ARGS "matmul.vmfb")

# Embed VM bytecode module into c source file
add_custom_command(
  OUTPUT
    "matmul.h"
    "matmul.c"
  COMMAND generate_embed_data ${_ARGS}
  DEPENDS generate_embed_data matmul.vmfb
)


#-------------------------------------------------------------------------------
# Create a library and thus a CMake target.
#-------------------------------------------------------------------------------

add_library(matmul_c STATIC "")
target_sources(matmul_c
  PRIVATE
    matmul.c
    matmul.h
)


#-------------------------------------------------------------------------------
# Build the excutable.
#-------------------------------------------------------------------------------

add_executable(matmul_generator "")
target_sources(matmul_generator
  PRIVATE
  matmul_generator.c
    device_vmvx.c
)

set_target_properties(matmul_generator PROPERTIES OUTPUT_NAME matmul_generator)

target_include_directories(matmul_generator
  PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(matmul_generator
  matmul_c
  iree_base_base
  iree_hal_hal
  iree_hal_vmvx_registration_registration
  iree_modules_hal_hal
  iree_vm_vm
  iree_vm_bytecode_module
)
