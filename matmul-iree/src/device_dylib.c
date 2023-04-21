
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A example of setting up the embedded-sync driver.

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/embedded_elf_loader.h"

#ifdef USE_LOCAL_TASK
#include "iree/hal/drivers/local_task/task_device.h"
#include "iree/task/api.h"
#else
#include "iree/hal/drivers/local_sync/sync_device.h"
#endif

#define NOOP ((void)0)
#define STRINGIFY(x) #x

#ifdef USE_LOCAL_TASK

#define IDENTIFIER STRINGIFY(local-task)
#define CREATE_DEVICE \
   iree_hal_task_device_create( \
        identifier, &params, executor, /*loader_count=*/1, &loader, \
	device_allocator, host_allocator, out_device)
#define PARAMS_T iree_hal_task_device_params_t
#define INIT_PARAMS iree_hal_task_device_params_initialize
#define CREATE_TASK_EXECUTOR \
  iree_task_executor_t* executor = NULL; \
  status = iree_task_executor_create_from_flags(host_allocator, &executor)
#define RELEASE_TASK_EXECUTOR iree_task_executor_release(executor)

#else

#define IDENTIFIER STRINGIFY(local-sync)
#define CREATE_DEVICE \
   iree_hal_sync_device_create( \
        identifier, &params, /*loader_count=*/1, &loader, \
	device_allocator, host_allocator, out_device)
#define PARAMS_T iree_hal_sync_device_params_t
#define INIT_PARAMS iree_hal_sync_device_params_initialize
#define CREATE_TASK_EXECUTOR NOOP
#define RELEASE_TASK_EXECUTOR NOOP

#endif

iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                   iree_hal_device_t** out_device) {
  // Set parameters for the device created in the next step.
  PARAMS_T params;
  INIT_PARAMS(&params);

  iree_hal_executable_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_embedded_elf_loader_create(
      /*plugin_manager=*/NULL, host_allocator, &loader));

  iree_status_t status;
  CREATE_TASK_EXECUTOR;

  // Use the default host allocator for buffer allocations.
  iree_string_view_t identifier = iree_make_cstring_view(IDENTIFIER);
  iree_hal_allocator_t* device_allocator = NULL;
  status = iree_hal_allocator_create_heap(
      identifier, host_allocator, host_allocator, &device_allocator);

  // Create the device and release the loader afterwards.
  if (iree_status_is_ok(status)) {
    status = CREATE_DEVICE;
  }

  iree_hal_allocator_release(device_allocator);
  RELEASE_TASK_EXECUTOR;
  iree_hal_executable_loader_release(loader);
  return status;
}
