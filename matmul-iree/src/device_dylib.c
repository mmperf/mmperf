
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
#include "iree/hal/drivers/local_sync/sync_device.h"

iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                   iree_hal_device_t** out_device) {
  // Set parameters for the device created in the next step.
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  iree_hal_executable_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_embedded_elf_loader_create(
      iree_hal_executable_import_provider_null(), host_allocator, &loader));

  // Use the default host allocator for buffer allocations.
  iree_string_view_t identifier = iree_make_cstring_view("dylib");
  iree_hal_allocator_t* device_allocator = NULL;
  iree_status_t status = iree_hal_allocator_create_heap(
      identifier, host_allocator, host_allocator, &device_allocator);

  // Create the synchronous device and release the loader afterwards.
  if (iree_status_is_ok(status)) {
    status = iree_hal_sync_device_create(
        identifier, &params, /*loader_count=*/1, &loader, device_allocator,
        host_allocator, out_device);
  }

  iree_hal_allocator_release(device_allocator);
  iree_hal_executable_loader_release(loader);
  return status;
}
