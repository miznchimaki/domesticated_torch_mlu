#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.

// FIXME: Currently, CPU and MLU backend are mutually exclusive.
// This is a temporary workaround. We need a better way to support
// multi devices.

#include <cnrt.h>

#define AOTI_RUNTIME_MLU_CHECK(EXPR)                       \
  do {                                                     \
    const cnrtRet_t code = EXPR;                           \
    const char* msg = cnrtGetErrorStr(code);               \
    if (code != cnrtSuccess) {                             \
      throw std::runtime_error(                            \
          std::string("CNRT error: ") + std::string(msg)); \
    }                                                      \
  } while (0)

namespace torch::aot_inductor {

using DeviceStreamType = cnrtQueue_t;

} // namespace torch::aot_inductor
