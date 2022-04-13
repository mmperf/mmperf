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
#include <string>
#include <benchmark/benchmark.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"

#include MATMUL_HEADER

#define STRING(s) #s
#define TO_STRING(x) STRING(x)

void init_matrix(uint16_t *a, int nrows, int ncols) {
  for (int j = 0; j < ncols; j++) {
    for (int i = 0; i < nrows; i++) {
      a[i + j * nrows] = ((uint16_t) rand() / (uint16_t) RAND_MAX);
    }
  }
}

void naive_matmul(const uint16_t *a, const uint16_t *b, uint16_t *c, size_t m, size_t k, size_t n) {
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
extern "C"{
  iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                     iree_hal_device_t** out_device);
}

static void BenchmarkFunction(int batch_size,
                              iree_vm_context_t* context,
                              iree_vm_function_t function,
                              iree_vm_list_t* inputs,
                              iree_vm_list_t* outputs,
                              uint16_t* arg0,
                              uint16_t* arg1,
                              iree_hal_device_t* device,
                              benchmark::State& state) {
  while (state.KeepRunningBatch(batch_size)) {
    IREE_CHECK_OK(iree_vm_list_create(
                           /*element_type=*/NULL,
                           /*capacity=*/1, iree_allocator_system(), &outputs));

    // Synchronously invoke the function.
    IREE_CHECK_OK(iree_vm_invoke(context, function,
                                 IREE_VM_INVOCATION_FLAG_NONE,
                                 /*policy=*/NULL, inputs, outputs,
                                 iree_allocator_system()));
  }

  // Force a full flush and get the device back to an idle state.
  IREE_CHECK_OK(iree_hal_device_wait_idle(device, iree_infinite_timeout()));

  // Get the result buffers from the invocation.
  iree_hal_buffer_view_t* ret_buffer_view =
      (iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
          outputs, 0, iree_hal_buffer_view_get_descriptor());

  // Read back the results and ensure we got the right values.
  uint16_t results[MDIM * NDIM];
  IREE_CHECK_OK(iree_hal_device_transfer_d2h(
      device, iree_hal_buffer_view_buffer(ret_buffer_view), 0, results,
      sizeof(results), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
#ifdef ENABLE_CHECK
  uint16_t *C2 = (uint16_t *) malloc(MDIM * NDIM * sizeof(uint16_t));
  size_t errors = 0;
  naive_matmul(arg0,arg1,C2,MDIM,KDIM,NDIM);
  for (size_t i = 0; i < MDIM; i++) {
    for (size_t j = 0; j < NDIM; j++) {
      size_t ci = i + j*MDIM;
      if (fabs(results[ci] - C2[ci]) > 0.01f) {
        //fprintf(stderr, "Incorrect result at index %ld,%ld: C=%0.2f C2=%0.2f\n", i, j, results[ci], C2[ci]);
        errors++;
        }
    }
  }
  printf("Detected %ld errors.\n", errors);
#endif
}

iree_status_t Run() {
  // TODO(benvanik): move to instance-based registration.
  IREE_RETURN_IF_ERROR(iree_hal_module_register_types());

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance));

  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(create_sample_device(iree_allocator_system(), &device));
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
      instance, IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION, &modules[0], IREE_ARRAYSIZE(modules), iree_allocator_system(),
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

  uint16_t *arg0 = (uint16_t *) malloc(MDIM * KDIM * sizeof(uint16_t));
  uint16_t *arg1 = (uint16_t *) malloc(KDIM * NDIM * sizeof(uint16_t));

  init_matrix(arg0, MDIM, KDIM);
  init_matrix(arg1, KDIM, NDIM);

  iree_hal_buffer_view_t* arg0_buffer_view = NULL;
  iree_hal_buffer_view_t* arg1_buffer_view = NULL;

  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer(
      iree_hal_device_allocator(device), arg0_shape, IREE_ARRAYSIZE(arg0_shape),
      IREE_HAL_ELEMENT_TYPE_FLOAT_16, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage =
              IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_TRANSFER,
      },
      iree_make_const_byte_span((void*)arg0, MDIM * KDIM * sizeof(uint16_t)), &arg0_buffer_view));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer(
      iree_hal_device_allocator(device), arg1_shape, IREE_ARRAYSIZE(arg1_shape),
      IREE_HAL_ELEMENT_TYPE_FLOAT_16, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .usage =
              IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_TRANSFER,
      },
      iree_make_const_byte_span((void*)arg1, KDIM * NDIM * sizeof(uint16_t)), &arg1_buffer_view));

  // Pass in the tensor as an expanded HAL buffer.
  iree_vm_list_t* inputs = NULL;
  iree_vm_list_t* outputs = NULL;

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

  int batch_size = 100;

  ::benchmark::RegisterBenchmark("BM_Matmul",
                               [batch_size, context, main_function, inputs, outputs, device, arg0, arg1](benchmark::State& state)
                               -> void {BenchmarkFunction(batch_size, context, main_function, inputs, outputs, arg0, arg1, device, state);
                               }) ->MeasureProcessCPUTime() ->UseRealTime();
  ::benchmark::RunSpecifiedBenchmarks();

  free(arg0);
  free(arg1);

  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return iree_ok_status();
}

int main(int argc, char **argv) {
  ::benchmark::Initialize(&argc, argv);
  const iree_status_t result = Run();
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
  }
  printf("matmul execution succeeded\n");
  return 0;
}
