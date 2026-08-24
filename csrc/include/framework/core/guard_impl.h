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
may be used to endorse or promote products derived from this software without
specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
*/

#pragma once
#include "cnrt.h"
#include "framework/core/MLUStream.h"
#include "framework/core/caching_allocator.h"
#include "framework/core/device.h"
#include "framework/core/device_utils.h"
#include "utils/Export.h"

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <aten/utils/exceptions.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <optional>

namespace torch_mlu {
namespace mlu {

struct MLUGuardImpl : public c10::impl::DeviceGuardImplInterface {
  static constexpr at::DeviceType static_type = at::DeviceType::PrivateUse1;
  MLUGuardImpl() = default;
  explicit MLUGuardImpl(at::DeviceType t) {
    AT_ASSERT(t == at::DeviceType::PrivateUse1);
  }
  at::DeviceType type() const override {
    return at::DeviceType::PrivateUse1;
  }

  c10::Device exchangeDevice(c10::Device device) const override {
    AT_ASSERT(device.type() == at::DeviceType::PrivateUse1);
    c10::Device old_device = getDevice();
    if (old_device.index() != device.index()) {
      setDevice(device);
    }
    return old_device;
  }

  c10::Device getDevice() const override {
    return c10::Device(at::DeviceType::PrivateUse1, current_device());
  }

  std::optional<c10::Device> uncheckedGetDevice() const noexcept {
    int device;
    const auto err = TORCH_CNRT_ERROR_HANDLED(cnrtGetDevice(&device));
    TORCH_CNRT_WARN(err);
    if (err != cnrtSuccess) {
      return c10::nullopt;
    }
    return c10::Device(at::DeviceType::PrivateUse1, device);
  }

  void setDevice(c10::Device device) const override {
    TORCH_INTERNAL_ASSERT(device.is_privateuseone());
    torch_mlu::setDevice(device.index());
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    torch_mlu::setDevice(device.index());
  }

  c10::Stream getStream(c10::Device device) const override {
    return getCurrentMLUStream(device.index()).unwrap();
  }

  c10::Stream getDefaultStream(c10::Device device) const override {
    return getDefaultMLUStream(device.index()).unwrap();
  }

  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    return getStreamFromPool(priority, d.index());
  }

  c10::Stream getStreamFromGlobalPool(
      c10::Device device,
      bool isHighPriority = false) const override {
    return getStreamFromPool(isHighPriority, device.index());
  }

  c10::Stream exchangeStream(c10::Stream s) const override {
    MLUStream mlu_stream(s);
    auto old_stream = getCurrentMLUStream(s.device().index());
    setCurrentMLUStream(mlu_stream);
    return old_stream.unwrap();
  }

  void* getStreamNativeHandle(const c10::Stream s) const override {
    MLUStream stream{s};
    return reinterpret_cast<void*>(stream.stream());
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  // Event-related functions
  void createEvent(cnrtNotifier_t* mlu_event, const c10::EventFlag flag) const {
    // Maps PyTorch's Event::Flag to MLU flag
    auto mlu_flag = CNRT_NOTIFIER_DEFAULT;
    switch (flag) {
      // see https://github.com/pytorch/pytorch/issues/117341
      case c10::EventFlag::PYTORCH_DEFAULT:
        mlu_flag = CNRT_NOTIFIER_DISABLE_TIMING_ALL;
        break;
      case c10::EventFlag::BACKEND_DEFAULT:
        mlu_flag = CNRT_NOTIFIER_DEFAULT;
        break;
      default:
        TORCH_CHECK(false, "MLU event received unknown flag");
    }

    TORCH_CNRT_CHECK(cnrtNotifierCreateWithFlags(mlu_event, mlu_flag));
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    auto mlu_event = static_cast<cnrtNotifier_t>(event);
    int orig_device;
    TORCH_CNRT_WARN(cnrtGetDevice(&orig_device));
    TORCH_CNRT_WARN(cnrtSetDevice(device_index));
    TORCH_CNRT_WARN(cnrtNotifierDestroy(mlu_event));
    TORCH_CNRT_WARN(cnrtSetDevice(orig_device));
  }

  void record(
      void** event,
      const c10::Stream& stream,
      const c10::DeviceIndex device_index,
      const c10::EventFlag flag) const override {
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");

    cnrtNotifier_t mlu_event = static_cast<cnrtNotifier_t>(*event);
    MLUStream mlu_stream(stream);
    cnrtQueue_t mlu_queue = mlu_stream.stream();

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Create the Notifier
    if (!mlu_event)
      createEvent(&mlu_event, flag);
    TORCH_CNRT_CHECK(cnrtPlaceNotifier(mlu_event, mlu_queue));
    *event = mlu_event;

    // Resets device
    setDevice(orig_device);
  }

  void block(void* event, const c10::Stream& stream) const override {
    if (!event)
      return;
    cnrtNotifier_t mlu_event = static_cast<cnrtNotifier_t>(event);
    MLUStream mlu_stream(stream);
    const auto orig_device = getDevice();
    setDevice(stream.device());
    TORCH_CNRT_CHECK(cnrtQueueWaitNotifier(mlu_event, mlu_stream.stream(), 0));
    setDevice(orig_device);
  }

  // May be called from any device
  bool queryEvent(void* event) const override {
    if (!event)
      return true;
    cnrtNotifier_t mlu_event = static_cast<cnrtNotifier_t>(event);
    const auto err = TORCH_CNRT_ERROR_HANDLED(cnrtQueryNotifier(mlu_event));
    if (err != cnrtErrorNotReady) {
      TORCH_CNRT_CHECK(err);
    } else {
      // ignore and clear the error if not ready
      (void)cnrtGetLastError();
    }
    return (err == cnrtSuccess);
  }

  // Stream-related functions
  bool queryStream(const c10::Stream& stream) const override {
    MLUStream mlu_stream{stream};
    return mlu_stream.query();
  }

  void synchronizeStream(const c10::Stream& stream) const override {
    MLUStream mlu_stream{stream};
    mlu_stream.synchronize();
  }

  void synchronizeEvent(void* event) const override {
    if (!event)
      return;
    auto mlu_event = static_cast<cnrtNotifier_t>(event);
    TORCH_CNRT_CHECK(cnrtWaitNotifier(mlu_event));
  }

  // Note: synchronizeDevice can be safely called from any device
  void synchronizeDevice(const c10::DeviceIndex device_index) const override {
    int orig_device{-1};
    TORCH_CNRT_CHECK(cnrtGetDevice(&orig_device));
    TORCH_CNRT_CHECK(cnrtSetDevice(device_index));
    // TODO() add trace here
    TORCH_CNRT_CHECK(cnrtSyncDevice());
    TORCH_CNRT_CHECK(cnrtSetDevice(orig_device));
  }

  void recordDataPtrOnStream(
      const c10::DataPtr& data_ptr,
      const c10::Stream& stream) const override {
    MLUStream mlu_stream{stream};
    torch_mlu::MLUCachingAllocator::recordStream(data_ptr, mlu_stream);
  }

  double elapsedTime(
      void* event1,
      void* event2,
      const c10::DeviceIndex device_index) const override {
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");
    // The behavior of cnrt is consistent with its behavior prior to CUDA
    // RT 12.0, therefore it is hoped that the behavior of `elapsedTime` will
    // also be consistent with versions prior to CUDA RT 12.0.
    int orig_device{-1};
    TORCH_CNRT_CHECK(cnrtGetDevice(&orig_device));
    TORCH_CNRT_CHECK(cnrtSetDevice(static_cast<int>(device_index)));
    auto mlu_event1 = static_cast<cnrtNotifier_t>(event1);
    auto mlu_event2 = static_cast<cnrtNotifier_t>(event2);
    float time_ms = 0;
    TORCH_CNRT_CHECK(cnrtNotifierElapsedTime(mlu_event1, mlu_event2, &time_ms));
    TORCH_CNRT_CHECK(cnrtSetDevice(orig_device));
    return static_cast<double>(time_ms);
  }
};

} // namespace mlu
} // namespace torch_mlu
