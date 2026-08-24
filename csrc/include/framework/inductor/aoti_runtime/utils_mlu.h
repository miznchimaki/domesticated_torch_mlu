#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include "framework/inductor/aoti_torch/c/shim.h"

#include "cnrt.h"

namespace torch_mlu::aot_inductor {

inline void delete_mlu_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_delete_mlu_guard(reinterpret_cast<MLUGuardHandle>(ptr)));
}

inline void delete_mlu_stream_guard(void* ptr) {
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_mlu_stream_guard(
      reinterpret_cast<MLUStreamGuardHandle>(ptr)));
}

union WrapDevice {
  int16_t int16;
  int8_t int8[2];
  WrapDevice(int8_t device_type, int8_t device_index) {
    int8[0] = device_type;
    int8[1] = device_index;
  }
};

class AOTIMluGuard {
 public:
  AOTIMluGuard(int32_t device_index) : guard_(nullptr, delete_mlu_guard) {
    MLUGuardHandle ptr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_mlu_guard(device_index, &ptr));
    guard_.reset(ptr);
  }

  void set_index(int32_t device_index) {
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_mlu_guard_set_index(guard_.get(), device_index));
  }

 private:
  std::unique_ptr<MLUGuardOpaque, torch::aot_inductor::DeleterFnPtr> guard_;
};

class AOTIMluStreamGuard {
 public:
  AOTIMluStreamGuard(cnrtQueue_t stream, int32_t device_index)
      : guard_(nullptr, delete_mlu_stream_guard) {
    MLUStreamGuardHandle ptr = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_create_mlu_stream_guard(stream, device_index, &ptr));
    guard_.reset(ptr);
  }

 private:
  std::unique_ptr<MLUStreamGuardOpaque, torch::aot_inductor::DeleterFnPtr>
      guard_;
};

} // namespace torch_mlu::aot_inductor
