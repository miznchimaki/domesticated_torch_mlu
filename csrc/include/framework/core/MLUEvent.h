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

#include "cnrt.h"
#include "framework/core/MLUStream.h"
#include "framework/core/mlu_guard.h"
#include "utils/Export.h"
#include "aten/utils/exceptions.h"

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <optional>
#include <utility>

/*
 * Cuda define cudaEventExternal(0x08) flag, it is a torch-specific flag that is
 * used to indicate that the CUDAEvent will be used only for synchronization
 * with work outside of the cuda graph, rather than creation of
 * cross-stream dependencies within a cuda graph. For MLU, we define
 * CNRT_NOTIFIER_EXTERNAL which has the same function. However, torch-specific
 * cudaEventExternal(0x08) may be conflict with cuda if cuda expand the flag and
 * for MLU 0x08 has already been used by cnrt. Therefore, We use a large number
 * that may be more safe for CNRT_NOTIFIER_EXTERNAL.
 */

#define CNRT_NOTIFIER_EXTERNAL (1U << 10)

namespace torch_mlu {
struct TORCH_MLU_API MLUEvent {
  MLUEvent() {}
  MLUEvent(unsigned int flags) : flags_{flags} {}

  MLUEvent(DeviceIndex device_index, const cnrtIpcNotifierHandle* handle) {
    device_index_ = device_index;
    torch_mlu::mlu::MLUGuard guard(device_index_);
    TORCH_CNRT_CHECK(cnrtIpcOpenNotifierHandle(&event_, *handle));
    is_created_ = true;
  }

  // Note: event destruction done on creating device to avoid creating a
  // MLU context on other devices.
  ~MLUEvent() {
    if (is_created_) {
      torch_mlu::mlu::MLUGuard guard(device_index_);
      TORCH_CNRT_CHECK(cnrtNotifierDestroy(event_));
    }
  }

  MLUEvent(const MLUEvent&) = delete;
  MLUEvent& operator=(const MLUEvent&) = delete;

  MLUEvent(MLUEvent&& other) noexcept {
    moveHelper(std::move(other));
  }
  MLUEvent& operator=(MLUEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator cnrtNotifier_t() const {
    return event();
  }

  // Less than operator (to allow use in sets)
  friend bool operator<(const MLUEvent& left, const MLUEvent& right) {
    return left.event_ < right.event_;
  }

  std::optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(at::kPrivateUse1, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const {
    return is_created_;
  }
  c10::DeviceIndex device_index() const {
    return device_index_;
  }
  cnrtNotifier_t event() const {
    return event_;
  }

  void place(const MLUStream& stream);

  void place() {
    place(getCurrentMLUStream());
  }

  void placeOnce(const MLUStream& stream);

  float elapsed_time(const MLUEvent& other) const;

  float hardware_time(const MLUEvent& other) const;

  void wait(const MLUStream& stream);

  bool query() const;

  void synchronize();

  void ipc_handle(cnrtIpcNotifierHandle* handle);

  void create(DeviceIndex device_index) {
    if (!is_created_) {
      createMLUEvent(device_index);
    }
  }

 private:
  unsigned int flags_ = CNRT_NOTIFIER_DISABLE_TIMING_ALL;
  int device_index_ = -1;
  cnrtNotifier_t event_{};
  bool is_created_ = false;
  bool was_placed_ = false;
  bool external_ = false;

  void moveHelper(MLUEvent&& other) {
    // Transfer ownership of all state from other to this
    flags_ = other.flags_;
    is_created_ = other.is_created_;
    was_placed_ = other.was_placed_;
    external_ = other.external_;
    device_index_ = other.device_index_;
    event_ = other.event_;

    // Reset other to a valid empty state to prevent double-free
    // The moved-from object must not attempt to destroy the event
    other.is_created_ = false;
    other.event_ = cnrtNotifier_t{};
  }

  void createMLUEvent(DeviceIndex device_index) {
    external_ = (flags_ & CNRT_NOTIFIER_EXTERNAL) != 0;
    flags_ &= ~CNRT_NOTIFIER_EXTERNAL;
    device_index_ = device_index;
    torch_mlu::mlu::MLUGuard guard(device_index_);
    TORCH_CNRT_CHECK(cnrtNotifierCreateWithFlags(&event_, flags_));
    is_created_ = true;
  }
};

} // namespace torch_mlu
