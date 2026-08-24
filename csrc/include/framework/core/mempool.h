#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <memory>

#include "framework/core/caching_allocator.h"
// This header file seems not be used, but we need to keep it to avoid BC.
#include "framework/core/MLUStream.h" // IWYU pragma: keep
#include "utils/Export.h"

namespace torch_mlu {

// MemPool represents a pool of memory in a caching allocator. Currently,
// it's just the ID of the pool object maintained in the MLUCachingAllocator.
//
// An allocator pointer can be passed to the MemPool to define how the
// allocations should be done in the pool. For example: using a different
// system allocator such as ncclMemAlloc.
struct TORCH_MLU_API MemPool {
  MemPool(
      std::shared_ptr<MLUCachingAllocator::MLUAllocator> allocator = nullptr,
      bool is_user_created = true,
      bool use_on_oom = false,
      bool no_split = false);
  MemPool(const MemPool&) = delete;
  MemPool(MemPool&&) = default;
  MemPool& operator=(const MemPool&) = delete;
  MemPool& operator=(MemPool&&) = default;
  ~MemPool();

  MempoolId_t id();
  int use_count();
  c10::DeviceIndex device();
  static MempoolId_t graph_pool_handle(bool is_user_created = true);

 private:
  static std::atomic<CaptureId_t> uid_;
  static std::atomic<CaptureId_t> uuid_;
  bool is_user_created_;
  MempoolId_t id_;
  c10::DeviceIndex device_;
};

} // namespace torch_mlu
