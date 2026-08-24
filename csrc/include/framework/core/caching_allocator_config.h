#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>

#include "utils/Export.h"

namespace torch_mlu::MLUCachingAllocator {

// Environment config parser
class TORCH_MLU_API MLUAllocatorConfig {
 public:
  // Do not reuse AcceleratorAllocatorConfig::max_split_size,
  // use MLUAllocatorConfig::max_split_size
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }

  static double garbage_collection_threshold() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        garbage_collection_threshold();
  }

  static bool expandable_segments() {
    bool enabled = c10::CachingAllocator::AcceleratorAllocatorConfig::
        use_expandable_segments();
    return enabled;
  }

  static bool release_lock_on_cnrtmalloc() {
    return instance().m_release_lock_on_cnrtmalloc;
  }

  static bool graph_capture_record_stream_reuse() {
    return instance().m_graph_capture_record_stream_reuse;
  }

  static double per_process_memory_fraction() {
    return instance().m_per_process_memory_fraction;
  }

  /** Pinned memory allocator settings */
  static bool pinned_use_mlu_host_register() {
    return instance().m_pinned_use_mlu_host_register;
  }

  static size_t pinned_num_register_threads() {
    return instance().m_pinned_num_register_threads;
  }

  static size_t pinned_max_register_threads() {
    // Based on the benchmark results, we see better allocation performance
    // with 8 threads. However on future systems, we may need more threads
    // and limiting this to 128 threads.
    return 128;
  }

  static size_t roundup_power2_divisions(size_t size) {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        roundup_power2_divisions(size);
  }

  static std::vector<size_t> roundup_power2_divisions() {
    return c10::CachingAllocator::AcceleratorAllocatorConfig::
        roundup_power2_divisions();
  }

  // Do not reuse AcceleratorAllocatorConfig::max_non_split_rounding_size,
  // use MLUAllocatorConfig::max_non_split_rounding_size
  static size_t max_non_split_rounding_size() {
    return instance().m_max_non_split_rounding_size;
  }

  // mlu default large_segment_size is 64MB, pytorch
  // default large_segment_size is 20MB, we cannot use
  // pytorch default large_segment_size
  static size_t large_segment_size() {
    return instance().m_large_segment_size;
  }

  static std::string last_allocator_settings() {
    return c10::CachingAllocator::getAllocatorSettings();
  }

  static bool use_linear_memory() {
    return instance().m_use_linear_memory;
  }

  static double empty_cache_time_threshold() {
    return instance().m_empty_cache_time_threshold;
  }

  static MLUAllocatorConfig& instance() {
    static MLUAllocatorConfig* s_instance = ([]() {
      auto inst = new MLUAllocatorConfig();
      // Note: keep the parsing order and logic stable to avoid potential
      // performance regressions.
      auto env = c10::utils::get_env("PYTORCH_MLU_ALLOC_CONF");
      if (!env.has_value()) {
        env = c10::utils::get_env("PYTORCH_CUDA_ALLOC_CONF");
      }
      if (!env.has_value()) {
        env = c10::utils::get_env("PYTORCH_ALLOC_CONF");
      }
      if (env.has_value()) {
        // Parse MLU-specific options
        inst->parseArgs(env.value());
        // Also parse base class options (expandable_segments, etc.)
        // This is needed because AcceleratorAllocatorConfig only checks
        // PYTORCH_CUDA_ALLOC_CONF and PYTORCH_ALLOC_CONF, not
        // PYTORCH_MLU_ALLOC_CONF
        c10::CachingAllocator::AcceleratorAllocatorConfig::instance().parseArgs(
            env.value());
      }

      auto custom_env = c10::utils::get_env("TORCH_MLU_ALLOC_CUSTOM_CONF");
      if (custom_env.has_value()) {
        inst->parseArgs(custom_env.value());
      }
      return inst;
    })();
    return *s_instance;
  }

  // Returns the set of valid keys for MLU-specific allocator configuration.
  // Note: large_segment_size_mb, max_split_size_mb, max_non_split_rounding_mb
  // are already defined in AcceleratorAllocatorConfig::getKeys(), so we only
  // include MLU-specific keys here.
  static const std::unordered_set<std::string>& getKeys() {
    static std::unordered_set<std::string> keys{
        "backend",
        "release_lock_on_cudamalloc",
        "release_lock_on_cnrtmalloc",
        "graph_capture_record_stream_reuse",
        "per_process_memory_fraction",
        "use_linear_memory",
        "empty_cache_time_threshold",
        "pinned_use_mlu_host_register",
        "pinned_use_cuda_host_register",
        "pinned_num_register_threads"};
    return keys;
  }

  void parseArgs(const std::string& env);

 private:
  MLUAllocatorConfig() = default;
  size_t parseAllocatorConfig(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i,
      bool& used_cudaMallocAsync);
  size_t parseGraphCaptureRecordStreamReuse(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  double parsePerProcessMemoryFraction(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parseLargeSegmentSize(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parseMaxSplitSize(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parseMaxNonSplitRoundingSize(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parseUseLinearMemory(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parseEmptyCacheTimeThreshold(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parsePinnedUseMluHostRegister(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);
  size_t parsePinnedNumRegisterThreads(
      const c10::CachingAllocator::ConfigTokenizer& tokenizer,
      size_t i);

  std::atomic<size_t> m_pinned_num_register_threads{1};
  std::atomic<bool> m_release_lock_on_cnrtmalloc{false};
  std::atomic<bool> m_pinned_use_mlu_host_register{false};
  std::atomic<bool> m_graph_capture_record_stream_reuse{false};
  std::atomic<double> m_per_process_memory_fraction{1.0};
  // Default large buffer size for MLU (64 MiB).
  // Note: This is now configurable via
  // MLUAllocatorConfig::large_segment_size(). The actual value is determined by
  // environment variable or this default.
  std::atomic<size_t> m_large_segment_size{
      67108864}; // 64 MB by default for MLU
  // Maximum block size that can be split. Blocks larger than this won't be
  // split. Default is unlimited (max size_t).
  std::atomic<size_t> m_max_split_size{std::numeric_limits<size_t>::max()};
  // Maximum extra size allowed when rounding up a block without splitting.
  // Default matches large_segment_size (64 MB for MLU).
  std::atomic<size_t> m_max_non_split_rounding_size{
      67108864}; // 64 MB by default for MLU
  std::atomic<bool> m_use_linear_memory{false};
  std::atomic<double> m_empty_cache_time_threshold{0.0};
};

// Keep this for backwards compatibility
using c10::CachingAllocator::setAllocatorSettings;

} // namespace torch_mlu::MLUCachingAllocator
