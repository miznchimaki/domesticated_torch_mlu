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
#include "c10/util/Exception.h"
#include <torch/headeronly/macros/Macros.h>
#include "utils/Export.h"
#include <cstdint>

#define TORCH_BANGC_CHECK(EXPR)                \
  do {                                         \
    const cnrtRet_t __err = EXPR;              \
    torch_mlu::mlu_bangc_check_implementation( \
        static_cast<int32_t>(__err),           \
        __FILE__,                              \
        __func__,                              \
        static_cast<uint32_t>(__LINE__),       \
        false);                                \
  } while (0);

#ifdef TORCH_MLU_USE_CHECKED_WRAPPERS
// When TORCH_MLU_USE_CHECKED_WRAPPERS is defined, CHECK macros route calls
// through auto-generated wrappers that print all parameter values on error.
// This flag is only set when compiling torch_mlu internally — external
// components see the original macros below for BC.
#include "aten/utils/api_checked.h"

#define TORCH_CNRT_CHECK(EXPR)                 \
  do {                                         \
    CNRT_USING_CHECKED                         \
    auto __err __attribute__((unused)) = EXPR; \
  } while (0)

#define TORCH_CNNL_CHECK(EXPR)                 \
  do {                                         \
    CNNL_USING_CHECKED                         \
    auto __err __attribute__((unused)) = EXPR; \
  } while (0)

#define TORCH_CNDEV_CHECK(EXPR)                \
  do {                                         \
    CNDEV_USING_CHECKED                        \
    auto __err __attribute__((unused)) = EXPR; \
  } while (0)

#define TORCH_MLUOP_CHECK(EXPR)                \
  do {                                         \
    MLUOP_USING_CHECKED                        \
    auto __err __attribute__((unused)) = EXPR; \
  } while (0)

#define TORCH_CNDRV_CHECK(EXPR)                \
  do {                                         \
    CNDRV_USING_CHECKED                        \
    auto __err __attribute__((unused)) = EXPR; \
  } while (0)

#else
// Original macros — used by external components and .mlu files.
// C10_UNLIKELY is:
// #define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
// The __builtin_expect is a macro function provided in GCC, a popular
// C compiler. It is used to provide the compiler with branch prediction
// information, which can be beneficial for optimizing performance-critical
// code.
#define TORCH_CNRT_CHECK(EXPR)                                 \
  do {                                                         \
    cnrtRet_t __err = EXPR;                                    \
    if (C10_UNLIKELY(__err != cnrtSuccess)) {                  \
      auto error_unused [[maybe_unused]] = cnrtGetLastError(); \
      (void)error_unused;                                      \
      TORCH_CHECK(                                             \
          false,                                               \
          "CNRT error: ",                                      \
          cnrtGetErrorStr(__err),                              \
          " (API: ",                                           \
          #EXPR,                                               \
          ")");                                                \
    }                                                          \
  } while (0);

#define TORCH_CNNL_CHECK(EXPR)                         \
  do {                                                 \
    cnnlStatus_t status = EXPR;                        \
    if (C10_UNLIKELY(status != CNNL_STATUS_SUCCESS)) { \
      TORCH_CHECK(                                     \
          false,                                       \
          "CNNL error: ",                              \
          cnnlGetErrorString(status),                  \
          " (API: ",                                   \
          #EXPR,                                       \
          ")");                                        \
    }                                                  \
  } while (0);

#define TORCH_CNDEV_CHECK(EXPR)                                 \
  do {                                                          \
    cndevRet_t status = EXPR;                                   \
    if (C10_UNLIKELY(status != CNDEV_SUCCESS)) {                \
      auto error_unused [[maybe_unused]] = cndevGetLastError(); \
      (void)error_unused;                                       \
      TORCH_CHECK(                                              \
          false,                                                \
          "CNDEV error: ",                                      \
          cndevGetErrorString(status),                          \
          " (API: ",                                            \
          #EXPR,                                                \
          ")");                                                 \
    }                                                           \
  } while (0);

#define TORCH_MLUOP_CHECK(EXPR)                         \
  do {                                                  \
    mluOpStatus_t status = EXPR;                        \
    if (C10_UNLIKELY(status != MLUOP_STATUS_SUCCESS)) { \
      TORCH_CHECK(                                      \
          false,                                        \
          "MLUOPS error: ",                             \
          mluOpGetErrorString(status),                  \
          " (API: ",                                    \
          #EXPR,                                        \
          ")");                                         \
    }                                                   \
  } while (0);

#define TORCH_CNDRV_CHECK(EXPR)                                              \
  do {                                                                       \
    CNresult __err = EXPR;                                                   \
    if (C10_UNLIKELY(__err != CN_SUCCESS)) {                                 \
      const char* err_str;                                                   \
      CNresult get_error_str_err = cnGetErrorString(__err, &err_str);        \
      if (get_error_str_err != CN_SUCCESS) {                                 \
        TORCH_CHECK(false, "CNDRV error: unknown error (API: ", #EXPR, ")"); \
      } else {                                                               \
        TORCH_CHECK(false, "CNDRV error: ", err_str, " (API: ", #EXPR, ")"); \
      }                                                                      \
    }                                                                        \
  } while (0)

#endif // TORCH_MLU_USE_CHECKED_WRAPPERS

#define TORCH_BANGC_KERNEL_LAUNCH_CHECK() TORCH_BANGC_CHECK(cnrtGetLastError())

#define TORCH_CNRT_WARN(EXPR)                                               \
  do {                                                                      \
    cnrtRet_t __err = EXPR;                                                 \
    if (C10_UNLIKELY(__err != cnrtSuccess)) {                               \
      auto error_unused [[maybe_unused]] = cnrtGetLastError();              \
      (void)error_unused;                                                   \
      TORCH_WARN(                                                           \
          "CNRT warning: ", cnrtGetErrorStr(__err), " (API: ", #EXPR, ")"); \
    }                                                                       \
  } while (0);

namespace torch_mlu {
// include_device_assertions not supported for now
TORCH_MLU_API void mlu_bangc_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const uint32_t line_number,
    const bool include_device_assertions);
} // namespace torch_mlu

// Indicates that a CNRT error is handled in a non-standard way
#define TORCH_CNRT_ERROR_HANDLED(EXPR) EXPR
