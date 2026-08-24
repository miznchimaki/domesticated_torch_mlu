
#pragma once
#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <string>
#include <vector>
#include <iostream>
#include <memory>

namespace torch_mlu {

inline std::string getCvarString(
    const std::vector<std::string>& env,
    const char* def,
    bool with_warning = true) {
  std::string ret(def);

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (int i = env.size() - 1; i >= 0; i--) {
    auto val = c10::utils::get_env(env[i].c_str());
    if (!val.has_value()) {
      continue;
    } else if (i > 1) { // support read NCCL environment
      if (with_warning) {
        TORCH_WARN(
            "Environment variable " + env[i] + " is deprecated; use " + env[0] +
            " instead");
      }
    }

    ret = val.value();
  }

  return ret;
}

inline int getCvarInt(const std::vector<std::string>& env, int def) {
  int ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (int i = env.size() - 1; i >= 0; i--) {
    const auto val = c10::utils::get_env(env[i].c_str());
    if (!val.has_value()) {
      continue;
    } else if (i > 1) { // support read NCCL environment
      TORCH_WARN(
          "Environment variable " + env[i] + " is deprecated; use " + env[0] +
          " instead");
    }

    try {
      ret = std::stoi(val.value());
    } catch (std::exception&) {
      TORCH_CHECK(false, "Invalid value for environment variable: " + env[i]);
    }
  }

  return ret;
}

inline bool getCvarBool(const std::vector<std::string>& env, bool def) {
  bool ret = def;

  if (env.empty()) {
    TORCH_CHECK(false, "No environment variables passed");
    return ret;
  }

  /* parse environment variable in reverse order, so the early
   * versions of a variable get higher priority than the latter
   * versions of the same variable */
  for (int i = env.size() - 1; i >= 0; i--) {
    auto val = c10::utils::get_env(env[i].c_str());
    if (!val.has_value()) {
      continue;
    } else if (i > 1) { // support read NCCL environment
      TORCH_WARN(
          "Environment variable " + env[i] + " is deprecated; use " + env[0] +
          " instead");
    }

    for (auto& x : val.value()) {
      x = std::tolower(x);
    }

    if (val == "y" || val == "yes" || val == "1" || val == "t" ||
        val == "true") {
      ret = true;
    } else if (
        val == "n" || val == "no" || val == "0" || val == "f" ||
        val == "false") {
      ret = false;
    } else {
      TORCH_CHECK(false, "Invalid value for environment variable: " + env[i]);
      return ret;
    }
  }

  return ret;
}

class SilentScope {
 public:
  enum class OutputMode {
    SUPPRESS_ALL, // Suppress all standard output and error output
    SUPPRESS_COUT_ONLY, // Only suppress standard output (std::cout)
    SUPPRESS_CERR_ONLY, // Only suppress standard error output (std::cerr)
    CAPTURE_TO_BUFFER // Capture output content to internal buffer
  };

  SilentScope(OutputMode mode = OutputMode::SUPPRESS_ALL) {
    if (mode == OutputMode::SUPPRESS_ALL ||
        mode == OutputMode::SUPPRESS_COUT_ONLY) {
      original_cout = std::cout.rdbuf();
      std::cout.rdbuf(&null_buffer);
    }
    if (mode == OutputMode::SUPPRESS_ALL ||
        mode == OutputMode::SUPPRESS_CERR_ONLY) {
      original_cerr = std::cerr.rdbuf();
      std::cerr.rdbuf(&null_buffer);
    }
    if (mode == OutputMode::CAPTURE_TO_BUFFER) {
      original_cout = std::cout.rdbuf();
      std::cout.rdbuf(capture_cout.rdbuf());
      original_cerr = std::cerr.rdbuf();
      std::cerr.rdbuf(capture_cerr.rdbuf());
    }
  }

  ~SilentScope() {
    if (original_cout) {
      std::cout.rdbuf(original_cout);
    }
    if (original_cerr) {
      std::cerr.rdbuf(original_cerr);
    }
  }

  std::string getCapturedCout() const {
    return capture_cout.str();
  }

  std::string getCapturedCerr() const {
    return capture_cerr.str();
  }

 private:
  std::streambuf* original_cout = nullptr;
  std::streambuf* original_cerr = nullptr;
  std::ostringstream capture_cout;
  std::ostringstream capture_cerr;

  // Null buffer (discards all output data)
  class NullBuffer : public std::streambuf {
   public:
    int overflow(int c) override {
      return c == traits_type::eof() ? traits_type::not_eof(c) : c;
    }
    int sync() override {
      return 0; // Synchronization succeeded, return 0
    }
  } null_buffer;
};

} // namespace torch_mlu
