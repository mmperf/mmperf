// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Forked from IREE (with modified includes).

// A example of setting up the HAL module to run simple pointwise array
// multiplication with the device implemented by different backends via
// create_sample_driver().

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"

#include MATMUL_HEADER

#define STRING(s) #s
#define TO_STRING(x) STRING(x)

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void init_matrix(float *a, int nrows, int ncols) {
  for (int j = 0; j < ncols; j++) {
    for (int i = 0; i < nrows; i++) {
      a[i + j * nrows] = ((float) rand() / (float) RAND_MAX);
    }
  }
}

void naive_matmul(const float *a, const float *b, float *c, size_t m, size_t k, size_t n) {
  // correctness check
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      size_t ci = i*n + j;
      c[ci] = 0.0f;
      for (size_t p = 0; p < k; p++) {
        c[ci] += a[i*k + p] * b[p*n + j];
      }
    }
  }
}

// A function to create the HAL device from the different backend targets.
// The HAL device is returned based on the implementation, and it must be
// released by the caller.
extern iree_status_t create_sample_device(iree_hal_device_t** device);

iree_status_t Run() {

  double t_start, t_end;
  t_start = rtclock();
  // TODO(benvanik): move to instance-based registration.
  IREE_RETURN_IF_ERROR(iree_hal_module_register_types());

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance));

  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(create_sample_device(&device), "create device");
  iree_vm_module_t* hal_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_create(device, iree_allocator_system(), &hal_module));

  // Note the setup here only supports native build. The bytecode is not built
  // for the cross-compile execution. The code can be compiled but it will
  // hit runtime error in a cross-compile environment.
  const struct iree_file_toc_t* module_file_toc =
      matmul_create();

  iree_vm_module_t* bytecode_module = NULL;
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      module_data, iree_allocator_null(), iree_allocator_system(),
      &bytecode_module));

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* context = NULL;
  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, &modules[0], IREE_ARRAYSIZE(modules), iree_allocator_system(),
      &context));
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  // Lookup the entry point function.
  // Note that we use the synchronous variant which operates on pure type/shape
  // erased buffers.
  const char kMainFunctionName[] = "module.matmul";
  iree_vm_function_t main_function;
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));
  
  // Allocate memory for input
  static iree_hal_dim_t arg0_shape[] = {MDIM, KDIM};
  static iree_hal_dim_t arg1_shape[] = {KDIM, NDIM};

  float *arg0 = (float *) malloc(MDIM * KDIM * sizeof(float));
  float *arg1 = (float *) malloc(KDIM * NDIM * sizeof(float));

  init_matrix(arg0, MDIM, KDIM);
  init_matrix(arg1, KDIM, NDIM);

  iree_hal_buffer_view_t* arg0_buffer_view = NULL;
  iree_hal_buffer_view_t* arg1_buffer_view = NULL;

  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_wrap_or_clone_heap_buffer(
      iree_hal_device_allocator(device), arg0_shape, IREE_ARRAYSIZE(arg0_shape),
      IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_ALL,
      iree_make_byte_span((void*)arg0, MDIM * KDIM * sizeof(float)),
      iree_allocator_null(), &arg0_buffer_view));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_wrap_or_clone_heap_buffer(
      iree_hal_device_allocator(device), arg1_shape, IREE_ARRAYSIZE(arg1_shape),
      IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_ALL,
      iree_make_byte_span((void*)arg1, KDIM * NDIM * sizeof(float)),
      iree_allocator_null(), &arg1_buffer_view));

  // Pass in the tensor as an expanded HAL buffer.
  iree_vm_list_t* inputs = NULL;

  IREE_RETURN_IF_ERROR(iree_vm_list_create(
                           /*element_type=*/NULL,
                           /*capacity=*/2, iree_allocator_system(), &inputs),
                       "can't allocate input vm list");

  iree_vm_ref_t arg0_input_buffer_view_ref =
      iree_hal_buffer_view_move_ref(arg0_buffer_view);
  IREE_RETURN_IF_ERROR(
      iree_vm_list_push_ref_move(inputs, &arg0_input_buffer_view_ref));

  iree_vm_ref_t arg1_input_buffer_view_ref =
      iree_hal_buffer_view_move_ref(arg1_buffer_view);
  IREE_RETURN_IF_ERROR(
      iree_vm_list_push_ref_move(inputs, &arg1_input_buffer_view_ref));

  iree_vm_list_t* outputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
                           /*element_type=*/NULL,
                           /*capacity=*/1, iree_allocator_system(), &outputs),
                       "can't allocate output vm list");
  
  for (int t = 0; t < NUM_REPS; ++t) {
    // Synchronously invoke the function.
    IREE_RETURN_IF_ERROR(iree_vm_invoke(context, main_function,
                                        /*policy=*/NULL, inputs, outputs,
                                        iree_allocator_system()));
  }

  // Get the result buffers from the invocation.
  iree_hal_buffer_view_t* ret_buffer_view =
      (iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
          outputs, 0, iree_hal_buffer_view_get_descriptor());
  if (ret_buffer_view == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "can't find return buffer view");
  }

  t_end = rtclock();

  // Read back the results and ensure we got the right values.
  iree_hal_buffer_mapping_t mapped_memory;
  float *C = (float *) malloc(MDIM * NDIM * sizeof(float));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(ret_buffer_view), IREE_HAL_MEMORY_ACCESS_READ,
      0, IREE_WHOLE_BUFFER, &mapped_memory));
  for (int i = 0; i < mapped_memory.contents.data_length / sizeof(float); ++i) {
    // Accessing the output elements
    float matmul_output_element = ((const float*)mapped_memory.contents.data)[i];
    C[i] = ((const float*)mapped_memory.contents.data)[i];
  }

  const char *filename = TO_STRING(FILE_NAME);
  FILE *file = fopen(filename, "w");
  fprintf(file, "%0.2lf GFLOPS\n", 2.0 * NUM_REPS * MDIM * NDIM * KDIM / (t_end - t_start) / 1E9);
  fclose(file);

#ifdef ENABLE_CHECK
  float *C2 = (float *) malloc(MDIM * NDIM * sizeof(float));
  size_t errors = 0;
  naive_matmul(arg0,arg1,C2,MDIM,KDIM,NDIM);
  for (size_t i = 0; i < MDIM; i++) {
    for (size_t j = 0; j < NDIM; j++) {
      size_t ci = i + j*MDIM;
      if (fabs(C[ci] - C2[ci]) > 0.01f) {
        errors++;
        }
    }
  }
  printf("Detected %ld errors.\n", errors);
#endif

  iree_hal_buffer_unmap_range(&mapped_memory);

  free(arg0);
  free(arg1);

  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return iree_ok_status();
}

int main() {
  const iree_status_t result = Run();
  if (!iree_status_is_ok(result)) {
    char* message;
    size_t message_length;
    iree_status_to_string(result, &message, &message_length);
    fprintf(stderr, "matmul execution failed: %s\n", message);
    iree_allocator_free(iree_allocator_system(), message);
    return -1;
  }
  printf("matmul execution succeeded\n");
  return 0;
}
