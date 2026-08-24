/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/pytorch/pytorch/graphs/contributors Redistribution and use in
source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "utils/Export.h"
#include "cnrt.h" //NOLINT
#include <c10/core/Device.h>
#include <string>
#include <cstdint>
#include <cstddef>

#define MLU_DEVICE_NUM_MAX 16

namespace torch_mlu {

struct DeviceProp {
  std::string name = "";
  cnrtUUID_t uuid;
  int major = -1;
  int minor = -1;
  int core_num_per_cluster = -1;
  int cluster_count = -1;
  int job_limit = -1;
  int multi_processor_count = -1;
  int isa_version = -1;
  long total_memory = -1;
  int ipu_frequency = -1;
  int dram_bandwidth = -1;
  bool supports_linear_memory = false;
  int nram_size = -1;
  int wram_size = -1;
  int sram_size = -1;
  int llc_size = -1;
  bool smlu_mode = false;
};

TORCH_MLU_API c10::DeviceIndex device_count();

TORCH_MLU_API c10::DeviceIndex device_count_ensure_non_zero();

TORCH_MLU_API c10::DeviceIndex current_device();

TORCH_MLU_API uint32_t getDeviceAttr(cnrtDeviceAttr_t attr);

TORCH_MLU_API size_t getUnifiedNramSize();

TORCH_MLU_API DeviceProp* getCurrentDeviceProperties();

TORCH_MLU_API DeviceProp* getDeviceProperties(int64_t device);

TORCH_MLU_API void synchronize();

TORCH_MLU_API int getMaxParallelJobNum(cnrtFunctionType_t type);

TORCH_MLU_API cnrtFunctionType_t getJobLimitCnrtFuncType();

} // namespace torch_mlu
