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

#include <c10/core/Allocator.h>
#include <c10/core/AllocatorConfig.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>
#include <c10/util/flat_hash_map.h>

#include "cnrt.h"
#include "framework/core/MLUStream.h"
#include "framework/graphs/MLUGraphUtils.h"
#include "utils/Export.h"
#include "aten/utils/exceptions.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <map>

extern ska::flat_hash_map<void*, const char*> ptrMap_ipc;

namespace torch_mlu::MLUCachingAllocator {

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class TORCH_MLU_API FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() = default;
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeMluMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeMluMemoryCallbacksRegistry, name, __VA_ARGS__)

// Preserved only for BC reasons
// NOLINTNEXTLINE(misc-unused-using-decls)
using c10::DataPtr;
using c10::CachingDeviceAllocator::AllocatorTraceTracker;
using c10::CachingDeviceAllocator::AnnotationEntry;
using c10::CachingDeviceAllocator::BlockInfo;
using c10::CachingDeviceAllocator::CreateContextFn;
using c10::CachingDeviceAllocator::DeviceStats;
using c10::CachingDeviceAllocator::RecordContext;
using c10::CachingDeviceAllocator::SegmentInfo;
using c10::CachingDeviceAllocator::TraceEntry;

struct AllocatorState {
  virtual ~AllocatorState() = default;
};

struct AllocatorConfigInfo {
  double garbage_collection_threshold;
  size_t max_split_size;
  size_t pinned_num_register_threads;
  bool expandable_segments;
  bool release_lock_on_malloc;
  bool pinned_use_host_register;
  bool graph_capture_record_stream_reuse;
  std::string last_allocator_settings;
  std::vector<size_t> roundup_power2_divisions;
};

struct SnapshotInfo {
  std::vector<SegmentInfo> segments;
  std::vector<std::vector<TraceEntry>> device_traces;
  std::vector<AnnotationEntry> external_annotations;
  AllocatorConfigInfo config_metadata;
};

// returns the pointers freed in the pool
// and the pointers allocated. Note: a pointer
// may appear in both freed and allocated
struct CheckpointDelta {
  std::vector<void*> ptrs_freed;
  std::vector<at::DataPtr> dataptrs_allocd;
};

using OutOfMemoryObserver = std::function<void(
    int64_t device,
    size_t allocated,
    size_t device_total,
    size_t device_free)>;

struct ShareableHandle {
  ptrdiff_t offset;
  std::string handle;
};

struct StreamSegmentSize {
  StreamSegmentSize(cnrtQueue_t s, bool small, size_t sz)
      : stream(s), is_small_pool(small), total_size(sz) {}
  cnrtQueue_t stream;
  bool is_small_pool;
  size_t total_size;
};

class TORCH_MLU_API MLUAllocator : public c10::DeviceAllocator {
 public:
  virtual void* raw_alloc(size_t nbytes) = 0;
  virtual void* raw_alloc_with_stream(size_t nbytes, cnrtQueue_t stream) = 0;
  virtual void raw_delete(void* ptr) = 0;
  virtual void init(int device_count) = 0;
  virtual double getMemoryFraction(c10::DeviceIndex device) = 0;
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) = 0;
  virtual std::vector<StreamSegmentSize> getExpandableSegmentSizes(
      c10::DeviceIndex device) = 0;
  virtual void enable(bool value) = 0;
  virtual bool isEnabled() const = 0;
  virtual void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) = 0;
  virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
  // Keep for BC only
  virtual void recordStream(const DataPtr& ptr, MLUStream stream) = 0;
  void recordStream(const DataPtr& ptr, c10::Stream stream) override {
    MLUStream mlu_stream = MLUStream(stream);
    recordStream(ptr, mlu_stream);
  }
  virtual SnapshotInfo snapshot(
      MempoolId_t mempool_id = {0, 0},
      bool include_traces = true) = 0;
  virtual void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(cnrtQueue_t)> filter) = 0;
  virtual void endAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id) = 0;
  virtual void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) = 0;
  virtual int getPoolUseCount(
      c10::DeviceIndex /*device*/,
      MempoolId_t /*mempool_id*/) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support getPoolUseCount. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void createOrIncrefPool(
      c10::DeviceIndex /*device*/,
      MempoolId_t /*mempool_id*/,
      std::shared_ptr<MLUAllocator> allocator = nullptr) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support createOrIncrefPool. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void setUseOnOOM(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      bool use_on_oom) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support setUseOnOOM. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void setNoSplit(c10::DeviceIndex device, MempoolId_t mempool_id) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support setNoSplit. "
        "If you need it, please file an issue describing your use case.");
  }

  // returns true if the allocated blocks are equal to expected live allocations
  virtual bool checkPoolLiveAllocations(
      c10::DeviceIndex /*device*/,
      MempoolId_t /*mempool_id*/,
      const std::unordered_set<void*>& /*expected_live_allocations*/) {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support checkPoolLiveAllocations. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual ShareableHandle shareIpcHandle(void* ptr) = 0;
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) = 0;
  virtual bool isHistoryEnabled() {
    TORCH_CHECK(
        false,
        name(),
        " does not yet support recordHistory. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when,
      bool clearHistory,
      const std::vector<std::string>& skip_actions) = 0;
  virtual void recordAnnotation(
      const std::vector<std::pair<std::string, std::string>>& /*md*/) {}
  virtual void pushCompileContext(std::string& md) {}
  virtual void popCompileContext() {}
  virtual void setUserMetadata(const std::string& metadata) {}
  virtual std::string getUserMetadata() {
    return "";
  }
  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) = 0;

  // Attached AllocatorTraceTracker callbacks will be called while the
  // per-device allocator lock is held. Any additional locks taken from within
  // the callback must be proven to always have the lock order that never
  // triggers a deadlock. In particular, Python's GIL may be held when
  // calling the allocator so it is unsafe to try to acquire the GIL in this
  // callback.
  virtual void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) = 0;

  virtual void enablePeerAccess(
      c10::DeviceIndex dev,
      c10::DeviceIndex dev_to_access) = 0;

  // memory not allocated from cnrtMalloc cannot be copied
  // across devices using cnrtMemcpyAsync if peer to peer access is disabled.
  // instead it requires cnrtMemcpyAsyncPeer
  //  with P2P Enabled, all combinations work
  //  with P2P Disabled:
  //                       cnrtMalloc cnrtMallocAsync/cnMemMap
  // cnrtMemcpyAsyncPeer   works      works
  // cnrtMemcpyAsync       works      error

  // This function performs chooses to use the Peer version of
  // memcpy if required based on where the allocated put dst/src.
  virtual cnrtRet_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cnrtQueue_t stream,
      bool p2p_enabled) = 0;
  virtual std::shared_ptr<AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      MempoolId_t id) = 0;
  virtual CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<AllocatorState> pps) = 0;
  virtual std::string name() = 0;
  std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override {
    c10::DeviceGuard device_guard({at::kPrivateUse1, device});
    size_t free = 0;
    size_t total = 0;
    TORCH_CNRT_CHECK(cnrtMemGetInfo(&free, &total));
    return {free, total};
  }
};

// Allocator object, statically initialized
// See BackendInitializer in MLUCachingAllocator.cpp.
// Atomic loads on x86 are just normal loads,
// (atomic stores are different), so reading this value
// is no different than loading a pointer.
TORCH_MLU_API extern std::atomic<MLUAllocator*> allocator;

TORCH_MLU_API MLUAllocator* get();

// Called directly by clients.
inline void* raw_alloc(size_t nbytes) {
  return get()->raw_alloc(nbytes);
}

inline void* raw_alloc_with_stream(size_t nbytes, cnrtQueue_t stream) {
  return get()->raw_alloc_with_stream(nbytes, stream);
}

inline void raw_delete(void* ptr) {
  get()->raw_delete(ptr);
}

inline void init(int device_count) {
  get()->init(device_count);
}

inline double getMemoryFraction(c10::DeviceIndex device) {
  return get()->getMemoryFraction(device);
}

inline void setMemoryFraction(double fraction, c10::DeviceIndex device) {
  get()->setMemoryFraction(fraction, device);
}

inline std::vector<StreamSegmentSize> getExpandableSegmentSizes(
    c10::DeviceIndex device) {
  return get()->getExpandableSegmentSizes(device);
}

inline void emptyCache(MempoolId_t mempool_id = {0, 0}) {
  get()->emptyCache(mempool_id);
}

inline void enable(bool value) {
  get()->enable(value);
}

inline bool isEnabled() {
  return get()->isEnabled();
}

inline void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) {
  get()->cacheInfo(device, largestBlock);
}

inline void* getBaseAllocation(void* ptr, size_t* size) {
  return get()->getBaseAllocation(ptr, size);
}

inline void recordStream(const DataPtr& dataPtr, MLUStream stream) {
  get()->recordStream(dataPtr, stream);
}

inline c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
    c10::DeviceIndex device) {
  return get()->getDeviceStats(device);
}

inline void resetAccumulatedStats(c10::DeviceIndex device) {
  get()->resetAccumulatedStats(device);
}

inline void resetPeakStats(c10::DeviceIndex device) {
  get()->resetPeakStats(device);
}

inline SnapshotInfo snapshot(
    MempoolId_t mempool_id = {0, 0},
    bool include_traces = true) {
  return get()->snapshot(mempool_id, include_traces);
}

inline std::shared_ptr<AllocatorState> getCheckpointState(
    c10::DeviceIndex device,
    MempoolId_t id) {
  return get()->getCheckpointState(device, id);
}

inline CheckpointDelta setCheckpointPoolState(
    c10::DeviceIndex device,
    std::shared_ptr<AllocatorState> pps) {
  return get()->setCheckpointPoolState(device, std::move(pps));
}

// MLUGraph interactions
inline void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(cnrtQueue_t)> filter) {
  get()->beginAllocateToPool(device, mempool_id, std::move(filter));
}

inline void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->endAllocateToPool(device, mempool_id);
}

inline void recordHistory(
    bool enabled,
    CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    RecordContext when,
    bool clearHistory,
    const std::vector<std::string>& skip_actions) {
  get()->recordHistory(
      enabled,
      context_recorder,
      alloc_trace_max_entries,
      when,
      clearHistory,
      skip_actions);
}

inline void recordAnnotation(
    const std::vector<std::pair<std::string, std::string>>& md) {
  get()->recordAnnotation(md);
}

inline void pushCompileContext(std::string& md) {
  get()->pushCompileContext(md);
}

inline void popCompileContext() {
  get()->popCompileContext();
}

inline bool isHistoryEnabled() {
  return get()->isHistoryEnabled();
}

inline bool checkPoolLiveAllocations(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    const std::unordered_set<void*>& expected_live_allocations) {
  return get()->checkPoolLiveAllocations(
      device, mempool_id, expected_live_allocations);
}

inline void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
  get()->attachOutOfMemoryObserver(std::move(observer));
}

inline void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) {
  get()->attachAllocatorTraceTracker(std::move(tracker));
}

inline void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->releasePool(device, mempool_id);
}
inline void createOrIncrefPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::shared_ptr<MLUAllocator> allocator_ptr = nullptr) {
  get()->createOrIncrefPool(device, mempool_id, std::move(allocator_ptr));
}
inline void setUseOnOOM(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    bool use_on_oom) {
  get()->setUseOnOOM(device, mempool_id, use_on_oom);
}
inline void setNoSplit(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->setNoSplit(device, mempool_id);
}
inline int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return get()->getPoolUseCount(device, mempool_id);
}

// Not part of MLU_ALLOCATOR_BACKEND_INTERFACE
inline std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  return get()->getIpcDevPtr(std::move(handle));
}

inline ShareableHandle shareIpcHandle(void* ptr) {
  return get()->shareIpcHandle(ptr);
}

inline std::string name() {
  return get()->name();
}

inline cnrtRet_t memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    cnrtQueue_t stream,
    bool p2p_enabled) {
  return get()->memcpyAsync(
      dst, dstDevice, src, srcDevice, count, stream, p2p_enabled);
}

inline void enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  get()->enablePeerAccess(dev, dev_to_access);
}

inline void setUserMetadata(const std::string& metadata) {
  get()->setUserMetadata(metadata);
}

inline std::string getUserMetadata() {
  return get()->getUserMetadata();
}

// C++ API
TORCH_MLU_API std::pair<size_t, size_t> MemGetInfo(int device);
TORCH_MLU_API std::map<std::string, int64_t> mlu_memory_stats(int device);
TORCH_MLU_API uint64_t currentMemoryAllocated(int device_id);
TORCH_MLU_API uint64_t currentMemoryCached(int device_id);
TORCH_MLU_API uint64_t maxMemoryAllocated(int device_id);
TORCH_MLU_API uint64_t maxMemoryCached(int device_id);

} // namespace torch_mlu::MLUCachingAllocator
