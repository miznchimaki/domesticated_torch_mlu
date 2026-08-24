#pragma once
#include <stddef.h>
#include <stdint.h>

// This header defines a stable C API for certain ATen functionality in
// libtorch. The AOTInductor compiled model.so will only refer to this header
// instead of other headers from aten/c10, which means it will NOT be able to
// directly use any data structures or call functions from libtorch.
//
// What problems are we trying to solve here?  Direct use of aten/c10 APIs
// means use of C++ APIs on a library that doesn't have any ABI compatibility
// guarantees.  However, we want model.so to remain usable across updates
// to the PyTorch C++ libraries, which requires a stable ABI.  By introducing
// a C shim layer, we can minimize the surface that will cause breakage. The
// corresponding software stack can be illustrated as follows:
//
// |--------------------------------|
// |     inference service code     |
// |--------------------------------|
// |           model.so             |
// |--------------|-----------------|
// |           <c shim>             |
// |          libtorch.so           |
// |--------------------------------|
//
// The general guidelines for the C API:
//
//  - No exceptions, return an explicit error code to be checked at call site
//  - Only pointers (AtenTensorHandle counts), integers and floats in headers
//
// If you want to make changes to this header, you MUST MAINTAIN ABI
// compatibility.  Typically, this means you will have to add a _v2 version
// of a function that you, e.g., want to add a new function parameter to, and
// maintain the old and new versions of the APIs until all old model.so
// go out of use.

// The following files are implemented in a header-only way and are guarded by
// test/cpp/aoti_abi_check
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef __cplusplus
extern "C" {
#endif

struct MLUGuardOpaque;
using MLUGuardHandle = MLUGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_mlu_guard(
    int32_t device_index,
    MLUGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_mlu_guard(MLUGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_mlu_guard_set_index(MLUGuardHandle guard, int32_t device_index);

struct MLUStreamGuardOpaque;
using MLUStreamGuardHandle = MLUStreamGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_mlu_stream_guard(
    void* stream,
    int32_t device_index,
    MLUStreamGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_mlu_stream_guard(MLUStreamGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_mlu_stream(int32_t device_index, void** ret_stream);

#ifdef __cplusplus
} // extern "C"
#endif
