#pragma once
#include "utils/Export.h"
#include <c10/core/Device.h>
#include <cstdint>

namespace torch::mlu {

/// Returns the number of MLU devices available.
c10::DeviceIndex TORCH_MLU_API device_count();

/// Returns true if at least one MLU device is available.
bool TORCH_MLU_API is_available();

/// Returns true if MLU is available, and CNNL is available.
bool TORCH_MLU_API cnnl_is_available();

/// Sets the seed for the current MLU.
void TORCH_MLU_API manual_seed(uint64_t seed);

/// Sets the seed for all available MLUs.
void TORCH_MLU_API manual_seed_all(uint64_t seed);

/// Waits for all kernels in all streams on a MLU device to complete.
void TORCH_MLU_API synchronize(int64_t device_index = -1);

} // namespace torch::mlu
