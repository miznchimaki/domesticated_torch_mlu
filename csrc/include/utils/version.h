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

#include <string>
#ifdef USE_CNCL
#include "cncl.h" // NOLINT
#endif
#include "cnnl.h"
#include "cnnl_extra.h"
#include "mlu_op.h"
#include "utils/Export.h"

#define VERSION_ENCODE(x, y, z) ((x) * 1000000 + (y) * 1000 + (z))

#define VERSION_RANGE_ASSERT(name, major, minor, patch) \
  static_assert(                                        \
      (major) >= 0 && (major) <= 999,                   \
      #name " major version out of range[0, 999]");     \
  static_assert(                                        \
      (minor) >= 0 && (minor) <= 999,                   \
      #name " minor version out of range[0, 999]");     \
  static_assert(                                        \
      (patch) >= 0 && (patch) <= 999,                   \
      #name " patch version out of range[0, 999]");

#ifndef NEUWARE_CNTOOLKIT_VERSION
#if defined(CNTOOLKIT_MAJOR) && defined(CNTOOLKIT_MINOR) && \
    defined(CNTOOLKIT_PATCHLEVEL)
VERSION_RANGE_ASSERT(
    CNTOOLKIT,
    CNTOOLKIT_MAJOR,
    CNTOOLKIT_MINOR,
    CNTOOLKIT_PATCHLEVEL)
#define NEUWARE_CNTOOLKIT_VERSION \
  VERSION_ENCODE(CNTOOLKIT_MAJOR, CNTOOLKIT_MINOR, CNTOOLKIT_PATCHLEVEL)
#endif
#endif

#ifndef NEUWARE_CNNL_VERSION
VERSION_RANGE_ASSERT(CNNL, CNNL_MAJOR, CNNL_MINOR, CNNL_PATCHLEVEL)
#define NEUWARE_CNNL_VERSION \
  VERSION_ENCODE(CNNL_MAJOR, CNNL_MINOR, CNNL_PATCHLEVEL)
#endif

#ifdef USE_CNCL
#ifndef NEUWARE_CNCL_VERSION
VERSION_RANGE_ASSERT(
    CNCL,
    CNCL_MAJOR_VERSION,
    CNCL_MINOR_VERSION,
    CNCL_PATCH_VERSION)
#define NEUWARE_CNCL_VERSION \
  VERSION_ENCODE(CNCL_MAJOR_VERSION, CNCL_MINOR_VERSION, CNCL_PATCH_VERSION)
#endif
#endif

#ifndef NEUWARE_MLUOP_VERSION
VERSION_RANGE_ASSERT(MLUOP, MLUOP_MAJOR, MLUOP_MINOR, MLUOP_PATCHLEVEL)
#define NEUWARE_MLUOP_VERSION \
  VERSION_ENCODE(MLUOP_MAJOR, MLUOP_MINOR, MLUOP_PATCHLEVEL)
#endif

#ifndef NEUWARE_CNNLEXTRA_VERSION
VERSION_RANGE_ASSERT(
    CNNLEXTRA,
    CNNL_EXTRA_MAJOR,
    CNNL_EXTRA_MINOR,
    CNNL_EXTRA_PATCHLEVEL)
#define NEUWARE_CNNLEXTRA_VERSION \
  VERSION_ENCODE(CNNL_EXTRA_MAJOR, CNNL_EXTRA_MINOR, CNNL_EXTRA_PATCHLEVEL)
#endif

namespace torch_mlu {
TORCH_MLU_API void checkRequirements();
TORCH_MLU_API std::string getVersion();
TORCH_MLU_API std::string getDriverVersion();
TORCH_MLU_API bool is_driver_version_ge(
    int req_major,
    int req_minor,
    int req_patch);
} // namespace torch_mlu
