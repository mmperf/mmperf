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
#include "iree/vm/bytecode/module.h"
#include "iree/base/internal/math.h"

#include MATMUL_HEADER

#define STRING(s) #s
#define TO_STRING(x) STRING(x)

#ifdef USE_FP16
  using dtype = uint16_t;
#else
  using dtype = float;
#endif

void init_matrix(dtype *a, int nrows, int ncols) {
  for (int j = 0; j < ncols; j++) {
    for (int i = 0; i < nrows; i++) {
      #ifdef USE_FP16
        a[i + j * nrows] = iree_math_f32_to_f16((float) rand() / (float) RAND_MAX);
      #else
        a[i + j * nrows] = ((float) rand() / (float) RAND_MAX);
      #endif
    }
  }
}

void init_matrix_batch(dtype *a, int batch, int nrows, int ncols) {
  for (int b = 0; b < batch; b++){
    for (int j = 0; j < ncols; j++) {
      for (int i = 0; i < nrows; i++) {
      #ifdef USE_FP16
        a[i + j * nrows + b * ncols * nrows] = iree_math_f32_to_f16((float) rand() / (float) RAND_MAX);
      #else
        a[i + j * nrows + b * ncols * nrows] = ((float) rand() / (float) RAND_MAX);
      #endif
      }
    }
  }
}

void naive_matmul(const dtype *a, const dtype *b, float *c, size_t m, size_t k, size_t n) {
  // correctness check
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      size_t ci = i*n + j;
      c[ci] = 0.0f;
      for (size_t p = 0; p < k; p++) {
        #ifdef USE_FP16
          c[ci] += iree_math_f16_to_f32(a[i*k + p]) * iree_math_f16_to_f32(b[p*n + j]);
        #else
          c[ci] += a[i*k + p] * b[p*n + j];
        #endif
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
                              dtype* arg0,
                              dtype* arg1,
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

  // Get the result buffers from the invocation.
  iree_hal_buffer_view_t* ret_buffer_view =
      (iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
          outputs, 0, iree_hal_buffer_view_get_descriptor());

  // Read back the results and ensure we got the right values.
  dtype *C;
  IREE_CHECK_OK(iree_allocator_malloc(iree_allocator_system(),
                MDIM * NDIM * sizeof(dtype), (void**)&C));
  IREE_CHECK_OK(iree_hal_device_transfer_d2h(
      device, iree_hal_buffer_view_buffer(ret_buffer_view), 0, C,
      MDIM * NDIM * sizeof(dtype), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

#ifdef ENABLE_CHECK
  float *C2 = (float *) malloc(MDIM * NDIM * sizeof(float));
  size_t errors = 0;
  naive_matmul(arg0,arg1,C2,MDIM,KDIM,NDIM);
  for (size_t i = 0; i < MDIM; i++) {
    for (size_t j = 0; j < NDIM; j++) {
      size_t ci = i + j*MDIM;
      #ifdef USE_FP16
        float C1 = iree_math_f16_to_f32(C[ci]);
        if (fabs(C1- C2[ci]) > 0.8f) {  // Difference could be large for mixed precision calculation
          //fprintf(stderr, "Incorrect result at index %ld,%ld: C=%0.2f C2=%0.2f\n", i, j, C1, C2[ci]);
          errors++;
        }
      #else
        if (fabs(C[ci] - C2[ci]) > 0.3f) {   // Allow precision difference between fp32 and tf32
          //fprintf(stderr, "Incorrect result at index %ld,%ld: C=%0.2f C2=%0.2f\n", i, j, C[ci], C2[ci]);
          errors++;
        }
      #endif
    }
  }
  printf("Detected %ld errors.\n", errors);
#endif
  iree_allocator_free(iree_allocator_system(), C);
}

iree_status_t Run() {
  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));

  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(create_sample_device(iree_allocator_system(), &device));
  iree_vm_module_t* hal_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_create(instance, device, IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
                             iree_allocator_system(), &hal_module));

  // Note the setup here only supports native build. The bytecode is not built
  // for the cross-compile execution. The code can be compiled but it will
  // hit runtime error in a cross-compile environment.
  const struct iree_file_toc_t* module_file_toc =
      matmul_create();

  iree_vm_module_t* bytecode_module = NULL;
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      instance, module_data, iree_allocator_null(), iree_allocator_system(),
      &bytecode_module));

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* context = NULL;
  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION, IREE_ARRAYSIZE(modules),
      &modules[0], iree_allocator_system(), &context));
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  // Lookup the entry point function.
  // Note that we use the synchronous variant which operates on pure type/shape
  // erased buffers.
  iree_vm_function_t main_function;
  const char* kMainFunctionName;
  kMainFunctionName = (BDIM == 0) ? "module.matmul" : "module.batch_matmul";
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));

  // Allocate memory for input
  iree_hal_dim_t* arg0_shape, *arg1_shape;
  iree_host_size_t arg0_size, arg1_size;
  dtype *arg0, *arg1;
  iree_const_byte_span_t initial_data0, initial_data1;

  iree_hal_dim_t *arg0_dim;
  iree_hal_dim_t *arg1_dim;

  if (BDIM == 0){
    arg0_dim = (iree_hal_dim_t *) malloc(2 * sizeof(iree_hal_dim_t));
    arg1_dim = (iree_hal_dim_t *) malloc(2 * sizeof(iree_hal_dim_t));
    arg0_dim[1] = arg1_dim[0] = KDIM;
    arg0_dim[0] = MDIM;
    arg1_dim[1] = NDIM;
    arg0_shape = arg0_dim;
    arg1_shape = arg1_dim;
    arg0_size = 2;
    arg1_size = 2;

    arg0 = (dtype *) malloc(MDIM * KDIM * sizeof(dtype));
    arg1 = (dtype *) malloc(KDIM * NDIM * sizeof(dtype));

    init_matrix(arg0, MDIM, KDIM);
    init_matrix(arg1, KDIM, NDIM);

    initial_data0 = iree_make_const_byte_span((void*)arg0, MDIM * KDIM * sizeof(dtype));
    initial_data1 = iree_make_const_byte_span((void*)arg1, KDIM * NDIM * sizeof(dtype));
  } else {
    arg0_dim = (iree_hal_dim_t *) malloc(4 * sizeof(iree_hal_dim_t));
    arg1_dim = (iree_hal_dim_t *) malloc(4 * sizeof(iree_hal_dim_t));
    arg0_dim[0] = arg1_dim[0] = 1;
    arg0_dim[1] = arg1_dim[1] = BDIM;
    arg0_dim[3] = arg1_dim[2] = KDIM;
    arg0_dim[2] = MDIM;
    arg1_dim[3] = NDIM;
    arg0_shape = arg0_dim;
    arg1_shape = arg1_dim;
    arg0_size = 4;
    arg1_size = 4;

    arg0 = (dtype *) malloc(BDIM * MDIM * KDIM * sizeof(dtype));
    arg1 = (dtype *) malloc(BDIM * KDIM * NDIM * sizeof(dtype));

    init_matrix_batch(arg0, BDIM, MDIM, KDIM);
    init_matrix_batch(arg1, BDIM, KDIM, NDIM);

    initial_data0 = iree_make_const_byte_span((void*)arg0, BDIM * MDIM * KDIM * sizeof(dtype));
    initial_data1 = iree_make_const_byte_span((void*)arg1, BDIM * KDIM * NDIM * sizeof(dtype));
  }

  iree_hal_buffer_view_t* arg0_buffer_view = NULL;
  iree_hal_buffer_view_t* arg1_buffer_view = NULL;

#ifdef USE_FP16
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
  printf("Using FP16 \n");
#else
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
  printf("Using FP32 \n");
#endif

  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer(
      iree_hal_device_allocator(device), arg0_size, arg0_shape,
      element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      },
      initial_data0, &arg0_buffer_view));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer(
      iree_hal_device_allocator(device), arg1_size, arg1_shape,
      element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      },
      initial_data1, &arg1_buffer_view));

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
  free(arg0_dim);
  free(arg1_dim);

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
