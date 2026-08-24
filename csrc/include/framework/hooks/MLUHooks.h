#pragma once

#include "framework/generator/generator_impl.h"
#include "utils/Export.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/core/Generator.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>

#include <cstddef>
#include <cstdint>
#include <optional>
// IWYU pragma: no_include "framework/core/MLUStream.h"

// No need to have this whole header, we can just put it all in
// the cpp file

namespace torch_mlu {

TORCH_MLU_API bool hasPrimaryContext(int64_t device_index);
TORCH_MLU_API std::optional<int64_t> getDeviceIndexWithSharedContext();
TORCH_MLU_API int register_hook();

struct MLUHooksArgs : public at::PrivateUse1HooksArgs {};

struct MLUHooksInterface : public at::PrivateUse1HooksInterface {
  ~MLUHooksInterface() override = default;
  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) const override {
    static auto device_gen = torch_mlu::getDefaultMLUGenerator(device_index);
    return device_gen;
  }
  at::Device getDeviceFromPtr(void* data) const override;

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override;

  at::Generator getNewGenerator(
      at::DeviceIndex device_index = -1) const override;

  void init() const override;

  bool isBuilt() const override {
    return true;
  }

  bool isAvailable() const override {
    return true;
  };

  bool isPinnedPtr(const void* data) const override;

  c10::Allocator* getPinnedMemoryAllocator() const override;

  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize)
      const override;
};

at::PrivateUse1HooksInterface* get_private_hooks();

} // namespace torch_mlu
