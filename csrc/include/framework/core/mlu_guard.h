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

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include "framework/core/MLUStream.h"
#include "framework/core/guard_impl.h"

namespace torch_mlu {
namespace mlu {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for MLU.  It accepts
/// integer indices (interpreting them as MLU devices) and is a little
/// more efficient than DeviceGuard; however, it can only be used
/// from code that links against MLU directly.
struct TORCH_MLU_API MLUGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit MLUGuard() = delete;

  /// Set the current MLU device to the passed device index.
  explicit MLUGuard(c10::DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current MLU device to the passed device.  Errors if the passed
  /// device is not a MLU device.
  explicit MLUGuard(c10::Device device) : guard_(device) {}

  // Copy is not allowed
  MLUGuard(const MLUGuard&) = delete;
  MLUGuard& operator=(const MLUGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  MLUGuard(MLUGuard&& other) = delete;
  MLUGuard& operator=(MLUGuard&& other) = delete;
  ~MLUGuard() = default;

  /// Sets the MLU device to the given device.  Errors if the given device
  /// is not a MLU device.
  void set_device(c10::Device device) {
    guard_.set_device(device);
  }

  /// Sets the MLU device to the given device.  Errors if the given device
  /// is not a MLU device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(c10::Device device) {
    guard_.reset_device(device);
  }

  /// Sets the MLU device to the given device index.
  void set_index(c10::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  c10::Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  c10::Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<MLUGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for MLU.  See
/// MLUGuard for when you can use this.
struct TORCH_MLU_API OptionalMLUGuard {
  /// Create an uninitialized OptionalMLUGuard.
  explicit OptionalMLUGuard() = default;

  /// Set the current MLU device to the passed Device, if it is not nullopt.
  explicit OptionalMLUGuard(std::optional<c10::Device> device_opt)
      : guard_(device_opt) {}

  /// Set the current MLU device to the passed device index, if it is not
  /// nullopt
  explicit OptionalMLUGuard(std::optional<c10::DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalMLUGuard(const OptionalMLUGuard&) = delete;
  OptionalMLUGuard& operator=(const OptionalMLUGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalMLUGuard(OptionalMLUGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalMLUGuard& operator=(OptionalMLUGuard&& other) = delete;
  ~OptionalMLUGuard() = default;

  /// Sets the MLU device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a MLU
  /// device.
  void set_device(c10::Device device) {
    guard_.set_device(device);
  }

  /// Sets the MLU device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a MLU device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(c10::Device device) {
    guard_.reset_device(device);
  }

  /// Sets the MLU device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(c10::DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  std::optional<c10::Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  std::optional<c10::Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original MLU device, resetting this guard to uninitialized
  /// state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<MLUGuardImpl> guard_;
};

/// A variant of StreamGuard that is specialized for MLU.  See MLUGuard
/// for when you can use this.
struct MLUStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit MLUStreamGuard() = delete;

  /// Set the current MLU device to the device associated with the passed
  /// stream, and set the current MLU stream on that device to the passed
  /// stream. Errors if the Stream is not a MLU stream.
  explicit MLUStreamGuard(c10::Stream stream) : guard_(stream) {}
  ~MLUStreamGuard() = default;

  /// Copy is disallowed
  MLUStreamGuard(const MLUStreamGuard&) = delete;
  MLUStreamGuard& operator=(const MLUStreamGuard&) = delete;

  /// Move is disallowed, as MLUStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  MLUStreamGuard(MLUStreamGuard&& other) = delete;
  MLUStreamGuard& operator=(MLUStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a MLU stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on MLU, use MLUMultiStreamGuard instead.
  void reset_stream(c10::Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the MLU stream that was set at the time the guard was
  /// constructed.
  MLUStream original_stream() const {
    return MLUStream(MLUStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent MLU stream that was set using this device guard,
  /// either from construction, or via set_stream.
  MLUStream current_stream() const {
    return MLUStream(MLUStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent MLU device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  c10::Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the MLU device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  c10::Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<MLUGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for MLU.  See
/// MLUGuard for when you can use this.
struct OptionalMLUStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalMLUStreamGuard() = default;

  /// Set the current MLU device to the device associated with the passed
  /// stream, and set the current MLU stream on that device to the passed
  /// stream. Errors if the Stream is not a MLU stream.
  explicit OptionalMLUStreamGuard(c10::Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalMLUStreamGuard(std::optional<c10::Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalMLUStreamGuard(const OptionalMLUStreamGuard&) = delete;
  OptionalMLUStreamGuard& operator=(const OptionalMLUStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalMLUStreamGuard(OptionalMLUStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalMLUStreamGuard& operator=(OptionalMLUStreamGuard&& other) = delete;
  ~OptionalMLUStreamGuard() = default;

  /// Resets the currently set MLU stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(c10::Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the MLU stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  std::optional<MLUStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return MLUStream(MLUStream::UNCHECKED, r.value());
    } else {
      return std::nullopt;
    }
  }

  /// Returns the most recent MLU stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  std::optional<MLUStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return MLUStream(MLUStream::UNCHECKED, r.value());
    } else {
      return std::nullopt;
    }
  }

  /// Restore the original MLU device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<MLUGuardImpl> guard_;
};

/// A variant of MultiStreamGuard that is specialized for MLU.
struct MLUMultiStreamGuard {
  explicit MLUMultiStreamGuard(c10::ArrayRef<MLUStream> streams)
      : guard_(unwrapStreams(streams)) {}

  /// Copy is disallowed
  MLUMultiStreamGuard(const MLUMultiStreamGuard&) = delete;
  MLUMultiStreamGuard& operator=(const MLUMultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  MLUMultiStreamGuard(MLUMultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  MLUMultiStreamGuard& operator=(MLUMultiStreamGuard&& other) = delete;
  ~MLUMultiStreamGuard() = default;

 private:
  c10::impl::InlineMultiStreamGuard<MLUGuardImpl> guard_;

  static std::vector<c10::Stream> unwrapStreams(
      c10::ArrayRef<MLUStream> mluStreams) {
    std::vector<c10::Stream> streams;
    streams.reserve(mluStreams.size());
    for (const MLUStream& mluStream : mluStreams) {
      streams.push_back(mluStream);
    }
    return streams;
  }
};

} // namespace mlu
} // namespace torch_mlu
