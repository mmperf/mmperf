// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Forked from IREE (modified to use VMVX).

// A example of setting up the vmvx-sync driver.

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/vmvx_module_loader.h"
#include "iree/hal/drivers/local_sync/sync_device.h"

iree_status_t create_sample_device(iree_allocator_t host_allocator,
                                   iree_hal_device_t** out_device) {
  // Set parameters for the device created in the next step.
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(host_allocator, &instance));

  iree_hal_executable_loader_t* loader = NULL;
  iree_status_t status = iree_hal_vmvx_module_loader_create(
      instance, iree_allocator_system(), &loader);
  iree_vm_instance_release(instance);

  // Use the default host allocator for buffer allocations.
  iree_string_view_t identifier = iree_make_cstring_view("vmvx");
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(identifier, host_allocator,
                                            host_allocator, &device_allocator);
  }

  if (iree_status_is_ok(status)) {
    // Create the synchronous device.
    status = iree_hal_sync_device_create(
        identifier, &params, /*loader_count=*/1, &loader, device_allocator,
        host_allocator, out_device);
  }

  iree_hal_allocator_release(device_allocator);
  iree_hal_executable_loader_release(loader);
  return status;
}
