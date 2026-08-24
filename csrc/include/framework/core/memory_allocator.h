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
#include "framework/core/MLUStream.h"
#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <torch/headeronly/util/Deprecated.h>

#include <cstddef>

namespace torch_mlu {

/**
 * Note [HostMemoryAllocator]
 * ~~~~~~~~~~~~~~~~
 * A host caching allocator is to hold MLU host page-locked memory.
 * Which is designed for re-uses freed pinned (page-locked) memory,
 * and avoid too many time-used api call. Like cnrtHostMalloc, cnrtFreeHost.
 *
 * Also Caching allocator tries to avoid allocating and freeing memory for each
 * use for performance reasons. Resources only be freed by explicitly clearing
 * the cache or at the teardown of process.
 * https://discuss.pytorch.org/t/why-dont-explicit-free-cpu-resource-in-cachinghostallocator/189714
 *
 * Also can get more details from Note [HostAllocator design] in
 * pytorch/aten/src/ATen/core/CachingHostAllocator.h
 *
 * Note1: To ensure correct behavior, CachingHostAllocator_recordEvent must be
 * called anytime a pointer from this allocator is used.
 * Example:
 *   {
 *     at::DataPtr ptr = getCachingHostAllocator()->allocate(size);
 *     // do something
 *     CachingHostAllocator_recordEvent(ptr.get(), ptr.get_context(), stream);
 *   }
 *
 * Note2: when you add new public function in this class, you may
 * need add a lock guard protection.
 *
 * Note3: that this allocator does not split larger allocations into smaller
 * blocks, unlike the caching device allocator.
 *
 */

// To get MLUCachingHostAllocator
C10_DEPRECATED_MESSAGE(
    "torch_mlu::getCachingHostAllocator() is deprecated. Please use at::getHostAllocator(at::kPrivateUse1) instead.")
inline TORCH_MLU_API at::HostAllocator* getCachingHostAllocator() {
  return at::getHostAllocator(at::kPrivateUse1);
};

C10_DEPRECATED_MESSAGE(
    "torch_mlu::CachingHostAllocator_recordEvent(...) is deprecated. Please use at::getHostAllocator(at::kPrivateUse1)->record_event(...) instead.")
inline TORCH_MLU_API bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    torch_mlu::MLUStream stream) {
  return at::getHostAllocator(at::kPrivateUse1)
      ->record_event(ptr, ctx, stream.unwrap());
};

C10_DEPRECATED_MESSAGE(
    "torch_mlu::CachingHostAllocator_emptyCache() is deprecated. Please use at::getHostAllocator(at::kPrivateUse1)->empty_cache() instead.")
inline TORCH_MLU_API void CachingHostAllocator_emptyCache() {
  at::getHostAllocator(at::kPrivateUse1)->empty_cache();
};
// Not using now, but aligned with pytorch gpu host allocator.
C10_DEPRECATED_MESSAGE(
    "torch_mlu::HostAlloc(...) is deprecated. Please use at::getHostAllocator(at::kPrivateUse1)->allocate(...) instead.")
inline at::DataPtr HostAlloc(size_t size) {
  return at::getHostAllocator(at::kPrivateUse1)->allocate(size);
}

C10_DEPRECATED_MESSAGE(
    "torch_mlu::CachingHostAllocator_getStats() is deprecated. Please use at::getHostAllocator(at::kPrivateUse1)->get_stats() instead.")
inline TORCH_MLU_API at::HostStats CachingHostAllocator_getStats() {
  return at::getHostAllocator(at::kPrivateUse1)->get_stats();
}

C10_DEPRECATED_MESSAGE(
    "torch_mlu::CachingHostAllocator_resetAccumulatedStats() is deprecated. Please use at::getHostAllocator(at::kPrivateUse1)->reset_accumulated_stats() instead.")
inline TORCH_MLU_API void CachingHostAllocator_resetAccumulatedStats() {
  at::getHostAllocator(at::kPrivateUse1)->reset_accumulated_stats();
}

C10_DEPRECATED_MESSAGE(
    "torch_mlu::CachingHostAllocator_resetPeakStats() is deprecated. Please use at::getHostAllocator(at::kPrivateUse1)->reset_peak_stats() instead.")
inline TORCH_MLU_API void CachingHostAllocator_resetPeakStats() {
  at::getHostAllocator(at::kPrivateUse1)->reset_peak_stats();
}

} // namespace torch_mlu
