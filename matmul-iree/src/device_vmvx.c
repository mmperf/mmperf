// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Forked from IREE (modified to use VMVX).

// A example of setting up the the vmvx driver.

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/vmvx/registration/driver_module.h"

iree_status_t create_sample_device(iree_hal_device_t** device) {
  // Only register the VMVX HAL driver.
  IREE_RETURN_IF_ERROR(iree_hal_vmvx_driver_module_register(
      iree_hal_driver_registry_default()));
  // Create the hal driver from the name.
  iree_hal_driver_t* driver = NULL;
  iree_string_view_t identifier = iree_make_cstring_view("vmvx");
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create_by_name(
      iree_hal_driver_registry_default(), identifier, iree_allocator_system(),
      &driver));
  IREE_RETURN_IF_ERROR(iree_hal_driver_create_default_device(
      driver, iree_allocator_system(), device));
  iree_hal_driver_release(driver);
  return iree_ok_status();
}
