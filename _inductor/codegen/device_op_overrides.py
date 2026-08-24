from __future__ import annotations
from typing import Optional
import torch

from torch._inductor.codegen import cpu_device_op_overrides
from torch._inductor.utils import triton_version_uses_attrs_dict
from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_device_op_overrides,
    TritonScratchWorkspace,
)


class MLUDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from torch_mlu._MLUC import _mlu_getCurrentRawStream as {name}"

    def set_device(self, device_idx: int) -> str:
        return f"torch.mlu.set_device({device_idx})"

    def synchronize(self) -> str:
        return "torch.mlu.synchronize()"

    def device_guard(self, device_idx: int) -> str:
        return f"torch.mlu._DeviceGuard({device_idx})"

    def cpp_device_guard(self) -> str:
        return "torch_mlu::mlu::MLUGuard"

    def cpp_aoti_device_guard(self) -> str:
        return "torch_mlu::aot_inductor::AOTIMluGuard"

    def cpp_stream_guard(self) -> str:
        return "torch_mlu::mlu::MLUStreamGuard"

    def cpp_aoti_stream_guard(self) -> str:
        return "torch_mlu::aot_inductor::AOTIMluStreamGuard"

    def cpp_getStreamFromExternal(self) -> str:
        return "torch_mlu::getStreamFromExternal"

    def kernel_header(self) -> str:
        source_codes = """
        #include "framework/core/mlu_guard.h"
        #include "framework/core/MLUStream.h"
        """
        return source_codes

    def kernel_driver(self) -> str:
        source_codes = """
            #define MLU_DRIVER_CHECK(EXPR)                              \\
            do {                                                        \\
                CNresult code = EXPR;                                   \\
                const char *msg;                                        \\
                CNresult code_get_error = cnGetErrorString(code, &msg); \\
                if (code_get_error != CN_SUCCESS) {                     \\
                    throw std::runtime_error(                           \\
                        std::string("MLU driver error: ") +             \\
                        std::string("invalid error code!"));            \\
                }                                                       \\
                if (code != CN_SUCCESS) {                               \\
                    throw std::runtime_error(                           \\
                        std::string("MLU driver error: ") +             \\
                        std::string(msg));                              \\
                }                                                       \\
            } while (0);

            static inline CNkernel loadKernel(
                    std::string filePath,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cnbinDir = std::nullopt) {
                if (cnbinDir) {
                    std::filesystem::path p1{*cnbinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }

                CNmodule mod;
                CNkernel func;
                MLU_DRIVER_CHECK(cnModuleLoad(filePath.c_str(), &mod));
                MLU_DRIVER_CHECK(cnModuleGetKernel(mod, funcName.c_str(), &func));
                return func;
            }

            static inline CNkernel loadKernel(const void* start, const std::string &funcName, uint32_t sharedMemBytes) {
                CNmodule mod;
                CNkernel func;
                MLU_DRIVER_CHECK(cnModuleLoadData(start, &mod));
                MLU_DRIVER_CHECK(cnModuleGetKernel(mod, funcName.c_str(), &func));
                return func;
            }

            static inline void launchKernel(
                    CNkernel func,
                    uint32_t gridX,
                    uint32_t gridY,
                    uint32_t gridZ,
                    uint32_t numWarps,
                    uint32_t sharedMemBytes,
                    void* args[],
                    void* stream) {
                auto properties = torch_mlu::getCurrentDeviceProperties();
                uint32_t cluster_count = static_cast<uint32_t>(properties->cluster_count);
                uint32_t core_num_per_cluster = static_cast<uint32_t>(properties->core_num_per_cluster);
                uint32_t total_core_num = cluster_count * core_num_per_cluster;
                gridX *= numWarps;
                uint32_t func_type = ((numWarps == 1) && (gridX % core_num_per_cluster == 0) && (gridX * gridY * gridZ <= total_core_num))
                                ? core_num_per_cluster
                                : numWarps;
                if (gridX * gridY * gridZ > 0) {
                  MLU_DRIVER_CHECK(cnInvokeKernel(
                      func, gridX, gridY, gridZ, (KernelClass)func_type, 0, static_cast<cnrtQueue_t>(stream), args, NULL));
                }
            }
        """
        return source_codes

    def tma_descriptor_helpers(self) -> str:
        raise RuntimeError("Host-side TMA descriptors not supported on MLU.")

    def cpp_stream_type(self) -> str:
        return "void*"

    def aoti_get_stream(self) -> str:
        return "aoti_torch_get_current_mlu_stream"

    def cpp_kernel_type(self) -> str:
        return "CNkernel"

    def cpp_device_ptr(self) -> str:
        return "CNaddr"

    def cpp_global_scratch(
        self, idx: int, workspace: TritonScratchWorkspace
    ) -> Optional[tuple[list[str], str]]:
        if triton_version_uses_attrs_dict():
            var_name = f"global_scratch_{idx}"
            if workspace.size > 0:
                size_array = f"int64_t {var_name}_size[] = {{{workspace.size}}};"
                stride_array = f"int64_t {var_name}_stride[] = {{1}};"
                device_type = "cached_torch_device_type_mlu"
                device_idx = "device_idx_"

                return (
                    [
                        f"{size_array}",
                        f"{stride_array}",
                        f"AtenTensorHandle {var_name}_handle;",
                        (
                            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, {var_name}_size, {var_name}_stride, "
                            f"{workspace.generate_dtype_str()}, {device_type}, {device_idx}, &{var_name}_handle));"
                        ),
                        f"RAIIAtenTensorHandle {var_name}_tensor({var_name}_handle);",
                        f"CNaddr {var_name} = reinterpret_cast<CUdeviceptr>({var_name}_tensor.data_ptr());",
                    ],
                    var_name,
                )
            else:
                return [f"CNaddr {var_name} = 0;"], var_name
        return None


register_device_op_overrides("mlu", MLUDeviceOpOverrides())
