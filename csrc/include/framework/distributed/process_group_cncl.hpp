/*
All modification made by Cambricon Corporation: © 2025 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2025, the respective contributors
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

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <torch/all.h>
#include <unordered_map>
#include <future>

#include "cncl.h"
#include "framework/core/MLUEvent.h"
#include "utils/version.h"

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/logger.hpp>

namespace torch_mlu {

// Environment variable which controls whether or not wait() is blocking or
// non-blocking.
static std::vector<std::string> TORCH_CNCL_BLOCKING_WAIT = {
    "TORCH_CNCL_BLOCKING_WAIT",
    "TORCH_NCCL_BLOCKING_WAIT",
    "CNCL_BLOCKING_WAIT",
    "NCCL_BLOCKING_WAIT"};

// Environment variable which controls whether or not we perform Async Error
// Handling with CNCL.
static std::vector<std::string> TORCH_CNCL_ASYNC_ERROR_HANDLING = {
    "TORCH_CNCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "CNCL_ASYNC_ERROR_HANDLING",
    "NCCL_ASYNC_ERROR_HANDLING"};

// Environment Variable to control whether Desync Debug is enabled.
// This variable must be set together with CNCL_ASYNC_ERROR_HANDLING.
static std::vector<std::string> TORCH_CNCL_DESYNC_DEBUG = {
    "TORCH_CNCL_DESYNC_DEBUG",
    "TORCH_NCCL_DESYNC_DEBUG",
    "CNCL_DESYNC_DEBUG",
    "NCCL_DESYNC_DEBUG"};

// Control whether dumping debug info on watchdog
// timeout is enabled. This variable must be set together with
// TORCH_CNCL_ENABLE_MONITORING=1 and TORCH_CNCL_TRACE_BUFFER_SIZE > 0.
static std::vector<std::string> TORCH_CNCL_DUMP_ON_TIMEOUT = {
    "TORCH_CNCL_DUMP_ON_TIMEOUT",
    "TORCH_NCCL_DUMP_ON_TIMEOUT"};

// Whether to rethrow CUDA Errors in the watchdog (default true)
static std::vector<std::string> TORCH_CNCL_RETHROW_MLU_ERRORS = {
    "TORCH_CNCL_RETHROW_MLU_ERRORS",
    "TORCH_NCCL_RETHROW_CUDA_ERRORS"};

// Control whether to propagate CNCL errors to all ranks through TCPStore.
static std::vector<std::string> TORCH_CNCL_PROPAGATE_ERROR = {
    "TORCH_CNCL_PROPAGATE_ERROR",
    "TORCH_NCCL_PROPAGATE_ERROR"};

// Enable monitoring thread which aborts the process when the ProcessGroupCNCL
// Watchdog thread gets stuck and no heartbeat is detected after
// TORCH_CNCL_HEARTBEAT_TIMEOUT_SEC. This can happen due to calling MLU/CNCL
// APIs that may hang. It is Useful to prevent jobs being stuck for a prolonged
// time than necessary tying up cluster resources.
static std::vector<std::string> TORCH_CNCL_ENABLE_MONITORING = {
    "TORCH_CNCL_ENABLE_MONITORING",
    "TORCH_NCCL_ENABLE_MONITORING"};

// Enable recording start-events for all ProcessGroupCNCL collectives, and
// compute accurate collective timing per-collective. (Note: end-events are
// recorded by default. Turn on this flag can increase chances of a watchdog
// hang due to performing a MLU event query which eventually calls
// cudaEventElapsedTime() API.
static std::vector<std::string> TORCH_CNCL_ENABLE_TIMING = {
    "TORCH_CNCL_ENABLE_TIMING",
    "TORCH_NCCL_ENABLE_TIMING",
    "CNCL_ENABLE_TIMING",
    "NCCL_ENABLE_TIMING"};

// Control the watchdog heartbeat timeout period after which the monitoring
// thread will abort the process.
static std::vector<std::string> TORCH_CNCL_HEARTBEAT_TIMEOUT_SEC = {
    "TORCH_CNCL_HEARTBEAT_TIMEOUT_SEC",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"};

// Control the interval inside the watchdog thread to check the coordinated
// signal from other ranks, e.g. to dump the debugging information.
static std::vector<std::string> TORCH_CNCL_COORD_CHECK_MILSEC = {
    "TORCH_CNCL_COORD_CHECK_MILSEC",
    "TORCH_NCCL_COORD_CHECK_MILSEC"};

// Whether to log C++ stack traces on unclean shutdown (default true)
static std::vector<std::string> TORCH_CNCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN = {
    "TORCH_CNCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN",
    "TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN"};

// Control how much extra time we will wait for dumping the debugging info
// before we exit and throws timeout exception.
static std::vector<std::string> TORCH_CNCL_WAIT_TIMEOUT_DUMP_MILSEC = {
    "TORCH_CNCL_WAIT_TIMEOUT_DUMP_MILSEC",
    "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"};

// Whether to include only active collectives in the Flight Recorder trace
// (default false)
static std::vector<std::string> TORCH_CNCL_EXTRA_DUMP_ON_EXEC = {
    "TORCH_CNCL_EXTRA_DUMP_ON_EXEC",
    "TORCH_NCCL_EXTRA_DUMP_ON_EXEC"};

static std::vector<std::string> TORCH_CNCL_MLU_EVENT_CACHE = {
    "TORCH_CNCL_MLU_EVENT_CACHE",
    "TORCH_NCCL_CUDA_EVENT_CACHE"};

// The maximum number of events we store in the flight recorder's ring buffer.
// (One event could be the start or end of a collective, for example).
static std::vector<std::string> TORCH_CNCL_TRACE_BUFFER_SIZE = {
    "TORCH_FR_BUFFER_SIZE",
    "TORCH_CNCL_TRACE_BUFFER_SIZE",
    "TORCH_NCCL_TRACE_BUFFER_SIZE"};

static std::vector<std::string> TORCH_CNCL_DEBUG_INFO_PIPE_FILE = {
    "TORCH_CNCL_DEBUG_INFO_PIPE_FILE",
    "TORCH_NCCL_DEBUG_INFO_PIPE_FILE"};

constexpr const char* CNCL_BACKEND_NAME = "cncl";

constexpr const char* kStoreDumpKey = "exception_dump";

constexpr const char* kStoreErrorSignalKey = "remote_error";

constexpr const int kWorkStatusUpdatePeriodMs = 30 * 1000; // 30 seconds

constexpr auto kProcessGroupCNCLDefaultTimeout =
    std::chrono::milliseconds(10 * 60 * 1000);

// NoHandling: do not handle asynchronous CNCL errors
// TearDown: tear down process upon error, see `WorkCNCL::handleException`
// CleanUpOnly: just clean up collectives and abort communicators without
// tearing down process SkipCleanUp: (this is a temporary option and can be
// removed in future) tear down process without cleaning up CNCL communicators.
// This should be used as a last resort in case `cnclCommAbort` itself is
// hanging
enum ErrorHandlingMode {
  NoHandling = 0,
  TearDown = 1,
  CleanUpOnly = 2,
  SkipCleanUp = 3
};

#define SHOULD_CLEAN_UP(a) (a != NoHandling && a != SkipCleanUp)

#define SHOULD_TEAR_DOWN(a) (a != NoHandling && a != CleanUpOnly)

// If set, ProcessGroupCNCL doesn't use recordStream calls to ensure
// caching allocator safety for tensors used on both user-facing and
// internal comm streams.
// Instead, it stashes live references to those tensors until after
// user-facing streams are synced with comm streams.
// See stashed_for_allocator_safety_ below.
static std::vector<std::string> TORCH_CNCL_AVOID_RECORD_STREAMS = {
    "TORCH_CNCL_AVOID_RECORD_STREAMS",
    "TORCH_NCCL_AVOID_RECORD_STREAMS"};

// A shelf for stashing tensors between op call and `work.wait()`.
// Used in case of async ops.
class TORCH_MLU_API TensorShelf {
 public:
  // Stash tensors so that CachingAllocator cannot recycle them prematurely.
  void stash(std::vector<at::Tensor>& tensors);
  // Stash tensors from another shelf.
  void stash(TensorShelf& other);
  // Unstage the stashed tensors so that CachingAllocator can recycle them.
  // Same as `clear()`.
  void unstash();
  // Whether shelf is empty.
  bool empty();
  // Clear the shelf.
  void clear();

 protected:
  // Get the inner tensor vector. Use with caution as it is not protected by
  // mutex.
  std::vector<at::Tensor>& get();

 private:
  std::vector<at::Tensor> tVector_;
  // Need a mutex to protect `tVector_` because it can be potentially accessed
  // from both main thread and watchdog thread.
  std::mutex mutex_;
};

// RAII wrapper for CNCL communicator in a process
class TORCH_MLU_API CNCLComm {
  using MutexType = std::recursive_mutex;
  using LockType = std::unique_lock<MutexType>;

 public:
  explicit CNCLComm(cnclComm_t cnclComm) // NOSONAR
      : cncl_comm_(cnclComm),
        aborted_(false),
        cncl_async_err_(CNCL_RET_SUCCESS),
        comm_failure_reason_(c10::nullopt),
        initialized_(false) {}

  CNCLComm() : CNCLComm(nullptr) {}

  ~CNCLComm() noexcept;

  static std::shared_ptr<CNCLComm> create(
      int numRanks,
      int rank,
      int device,
      const cnclCliqueId_t clique_id);

  // Must not be copyable
  CNCLComm(const CNCLComm&) = delete;
  CNCLComm& operator=(const CNCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  CNCLComm& operator=(CNCLComm&& other) = delete;

  // Move constructable
  CNCLComm(CNCLComm&& other) { // NOSONAR
    // Using other's lock, as it reads other's states
    // Can not use this.mutex_, as this object is being constructed.
    LockType lock(other.mutex_);
    std::swap(cncl_comm_, other.cncl_comm_);
    std::swap(aborted_, other.aborted_);
    std::swap(cncl_async_err_, other.cncl_async_err_);
    std::swap(initialized_, other.initialized_);
    std::swap(deviceIndex_, other.deviceIndex_);
  }

  cnclComm_t getCnclComm() {
    LockType lock(mutex_);
    if (aborted_) {
      auto commFailureMsg = comm_failure_reason_ != c10::nullopt
          ? c10::str(
                " Original reason for failure was: ", *comm_failure_reason_)
          : "";
      TORCH_CHECK_WITH(
          DistBackendError,
          false,
          c10::str(
              "CNCL communicator was aborted on rank ",
              rank_,
              ". ",
              commFailureMsg));
    }
    return cncl_comm_;
  }

  // Wait for the communicator to be ready. This is a blocking function.
  // Useful in nonblocking mode: NCCL requires the communicator to be ready
  // before issuing a second command.
  // Arguments:
  //   longInterval: if true, wait with sleep of an interval; otherwise, wait
  //   with `sched_yield` which is faster (but acquires CPU more frequently).
  //   Use `longInterval=true` when waiting for initialization or finalize to
  //   complete. Use `longInterval=false` when waiting collective call to return
  //   ncclSuccess.
  void waitReady(bool longInterval);

  // Destroy a communicator. This is a blocking function.
  void destroy();

  void abort(std::optional<std::string> comm_failure_reason = c10::nullopt);

  bool isInitialized() const {
    LockType lock(mutex_);
    return initialized_;
  }

  bool isAborted() const {
    LockType lock(mutex_);
    return aborted_;
  }

  std::optional<std::string> getCnclCommFailureReason() const {
    LockType lock(mutex_);
    return comm_failure_reason_;
  }

  cnclResult_t checkForCnclError() {
    LockType lock(mutex_);
    if (cncl_async_err_ != CNCL_RET_SUCCESS) {
      return cncl_async_err_;
    }
    cncl_async_err_ = cnclGetCommAsyncError(cncl_comm_);
    return cncl_async_err_;
  }

  cnclCliqueId getCnclId() {
    return cncl_id_;
  }

  friend class ProcessGroupCNCL;

 protected:
  cnclComm_t cncl_comm_;
  bool aborted_;
  mutable MutexType mutex_;
  // Rank that this communicator corresponds to.
  int rank_;
  cnclResult_t cncl_async_err_;
  cnclCliqueId cncl_id_;
  // Optional reason for communicator failure, provided by ProcessGroupCNCL
  // for better error messaging.
  std::optional<std::string> comm_failure_reason_;
  bool initialized_{false};
  // Device index for which the CNCL comm is created
  at::DeviceIndex deviceIndex_{-1};
};

// ProcessGroupCNCL implements CNCL bindings for c10d.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// All CNCL functions provided by this class are asynchronous functions. More
// specifically, each CNCL call is scheduled on a separate MLU stream that is
// different from the current MLU stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibility to make sure that the MLU stream their
// code works on needs to wait for the CNCL operation from
// this class.
//
// This can be done by calling:
//
// either WorkCNCL::wait() or WorkCNCL::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Note that WorkCNCL::isSuccess() and WorkCNCL::isCompleted() will always
// return true since ProcessGroupCNCL is single threaded. Every single CNCL
// or MLU failure will simply raise std::runtime_error.
//
// Therefore, WorkCNCL::exception() is not supported since isSuccess() always
// returns true.
//
// Also note that WorkCNCL::finishedMLUExecution() is a helper function only
// provided by ProcessGroupCNCL to check if the CNCL operation of WorkCNCL has
// finished execution on the MLU (not just scheduled).
//
// Example on using the CNCL process group
//
//   ProcessGroupCNCL pg(store, rank, size);
//   std::shared_ptr<WorkCNCL> work = pg.allreduce(tensors);
//
//   // At this point, CNCL kernel has already by streamd successfully
//   // Now, let current stream wait for the CNCL to finish, originally this
//   function is
//   // async operation as well, but currently MLU is sync.
//
//   work->wait()
//
//   // Now continue on other work in the current stream.
class TORCH_MLU_API ProcessGroupCNCL : public c10d::Backend {
 public:
  class WorkCNCL : public c10d::Work,
                   public std::enable_shared_from_this<WorkCNCL> {
   public:
    // Constructor takes a list of MLU devices
    WorkCNCL(
        const std::string& pgUID,
        const std::string& pgDesc,
        at::Device& device,
        int rank,
        c10d::OpType opType,
        uint64_t seq,
        bool isP2P = false,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputs = c10::nullopt,
        bool enableTiming = false,
        bool mluEventCacheEnabled = false,
        c10d::DebugLevel distDebugLevel = c10d::DebugLevel::Off);

    // Copy constructor doing partial copy without outputs_. Cleanup thread
    // monitors and removes finished works. However it will deadlock when
    // destructs outputs_ tensors who are view tensors in autograd graph.
    WorkCNCL(const WorkCNCL& w);

    virtual ~WorkCNCL();

    // Checks if the CNCL kernel has started to execute.
    bool isStarted();

    // Checks if request has completed. In this specific case of CNCL, it checks
    // if the CNCL operation has completed on the MLU in its own CNCL stream.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for CNCL work if timeout is not set.
    // Otherwise, it will block the CPU thread until the CNCL work is completed
    // or timed out. If timeout, exception will be thrown.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    void blockCurrentStream() override {
      synchronize();
    }

    void abort() override;

    // Let current stream wait on the completing of the CNCL work
    // Throws on exceptions.
    void synchronize() override;

    // Synchronize stream by blocking each on the CNCL stream
    void synchronizeStream();

    // Helper function to handle exception (throw if needed).
    void handleException(ErrorHandlingMode asyncErrorHandling);

    // Helper function that checks if the CNCL kernels have finished
    // execution on the MLUs
    bool finishedMLUExecution();

    // Get a Future object that will be marked as completed internally.
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    // Get a Future result of each work (e.g. success, different error types).
    // instead of the tensor output.
    c10::intrusive_ptr<c10::ivalue::Future> getFutureResult() override;

    float getDuration() const override;

    uint64_t getSequencenumber() const override;

    const std::string& logPrefix() const;

    // Helper function that sets an exception_ptr on the WorkCNCL object.
    void setException(std::exception_ptr exception_ptr);

    // Helper function that returns True if the WorkCNCL object has timed out
    // and False otherwise.
    // In case of timeout, set exception on the WorkCNCL object.
    bool checkTimeout(
        std::optional<std::chrono::milliseconds> timeout = c10::nullopt);

    // Print the traceback of the collective at call time
    void printTraceback() const;

    std::string getTraceback() const;

    std::vector<at::Tensor> result() override;

    void addStashedTensor(std::vector<at::Tensor>& t);

   protected:
    // The process group unique id
    std::string pgUID_;

    // The process group description
    std::string pgDesc_;

    // The cached list of MLU devices to operate on
    at::Device device_;

    // The start MLU events of CNCL operator tracking this work item on
    // multiple MLU devices. These start MLU events are needed by desync
    // debugging if enabled.
    std::shared_ptr<torch_mlu::MLUEvent> cncl_start_event_;

    // The end MLU events of CNCL operator tracking this work item on
    // multiple MLU devices.
    std::shared_ptr<torch_mlu::MLUEvent> cncl_end_event_;

    // The CNCL communicators used for this work item.
    std::shared_ptr<CNCLComm> cnclComm_;

    // Whether this work is a barrier op
    bool isBarrierOp_{false};

    // Clone of blockingWait_ from ProcessGroupCNCL.
    bool blockingWait_{false};

    // Clone of opTimeout_ from ProcessGroupCNCL.
    std::chrono::milliseconds opTimeout_;

    // Ephemeral timeouts are owned by exactly one work,
    // and reset after that work completes.
    // There may be more than one ephemeral timeout active at the same time,
    // and this variable is used to track the ownership of ephemeral timeout.
    std::chrono::milliseconds ownedEphermeralTimeout_ =
        std::chrono::milliseconds(0);

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    // Record the collective sequential number of collective or p2p.
    uint64_t seq_;
    bool isP2P_;

    // Indicates if the cncl start event has been updated to the store trace.
    // This will be used by desync debug.
    bool startTraceUpdated_{false};

    // Record collective sizes for debug. We only record the size on the first
    // device as multi-device per process is deprecated
    size_t numelIn_ = 0;
    size_t numelOut_ = 0;

    // Wrapper method for the static checkForCNCLErrors which can be overridden
    // for tests.
    virtual std::exception_ptr checkForCNCLErrors();

    friend TORCH_MLU_API std::ostream& operator<<(
        std::ostream& output,
        const WorkCNCL& workCNCL);

   private:
    // Checks for CNCL errors and sets an appropriate exception_ptr.
    void checkAndSetException();

    // Just checks whether MLU execution has started, without modifying
    // exception_ptr.
    bool startedMLUExecutionInternal() const;

    // Just checks whether MLU execution has completed, without modifying
    // exception_ptr.
    bool finishedMLUExecutionInternal() const;

    // Reference to the store so that we can write aborted communicators
    // to the store.
    c10::intrusive_ptr<c10d::Store> store_;

    // c10d::Store a reference to CNCL collective's outputs, used by result and
    // to give a more descriptive message when representing the Work as a
    // string.
    std::shared_ptr<std::vector<at::Tensor>> outputs_;

    // TORCH_CNCL_AVOID_RECORD_STREAMS implementation helper.
    // c10d::Stores references to participating non-output tensors (ie inputs,
    // flattened intermediates).
    // We'll clear this list in synchronizeStream, just after user-facing
    // stream(s) are synced with the cncl work stream(s).
    // By keeping these refs (as well as outputs_) alive until after the
    // collective's work rejoins the user-facing streams, we achieve
    // caching allocator safety without any recordStream calls.
    // For in-place collectives, some refs stashed here may alias outputs_,
    // but that doesn't do any harm.
    std::shared_ptr<TensorShelf> stashed_for_allocator_safety_;

    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;

    // the future result (e.g., success or failure) of the work
    c10::intrusive_ptr<at::ivalue::Future> futureWorkResult_;

    bool timingEnabled_;
    // unique id used to tell the trace buffer that this
    // work has completed
    std::optional<uint64_t> trace_id_;
    c10d::DebugLevel distDebugLevel_;

    friend class ProcessGroupCNCL;
  };

  struct Options : c10d::Backend::Options {
    // NOTE: timeout in ProcessGroupCNCL::Options denote the timeout for
    // operations. This is only used when blockingWait_ is enabled.
    explicit Options(bool is_high_priority_stream = false);
    Options(const Options&) = default;
    Options(Options&&) noexcept = default;
    Options& operator=(const Options&) = delete;
    Options& operator=(Options&&) noexcept = delete;
    ~Options() override = default;

    // return intrusive_ptr of the object
    static c10::intrusive_ptr<Options> create(
        bool is_high_priority_stream = false) {
      return c10::make_intrusive<Options>(is_high_priority_stream);
    }

    // Schedule CNCL operations on high priority MLU streams
    bool is_high_priority_stream;

    std::string group_name;
  };

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  // Helper class related to TORCH_CNCL_DESYNC_DEBUG
  class DesyncDebugger {
   public:
    // Initialize and enable DesyncDebugger
    void init(
        int rank,
        int size,
        int globalRank,
        int pgId,
        c10::intrusive_ptr<c10d::Store> store);

    // Run desync debug. This function is called by watchdog at time of timeout.
    void run();

    // Log work start to store.
    void logWorkStart(WorkCNCL& work);

    // Log work end to store.
    void logWorkEnd(WorkCNCL& work);

   private:
    // Whether desync debug is enabled.
    // If false, all functions are no-op.
    bool enabled_{false};

    // From ProcessGroupNCCL
    int rank_;
    int size_;
    int globalRank_;
    int pgId_;

    // Reference to the store so that we can log start/end event.
    c10::intrusive_ptr<c10d::Store> store_;

    // The store keys to trace the last CNCL collective kernel MLU events -
    // start event and end event respectively. These are used to do desync root
    // cause analysis.
    std::string traceKeyStart_;
    std::string traceKeyEnd_;
  };

  // Class that runs as a separate thread aside from watchdog
  // thread because we need to check the heartbeat from watchdog thread
  // so that when we get stuck in some CNCL/MLU calls,
  // we can dump the debugging information and abort the process.
  class HeartbeatMonitor {
   public:
    HeartbeatMonitor(ProcessGroupCNCL* pg);
    virtual ~HeartbeatMonitor() = default;

    // Start the heartbeat monitor thread.
    void start();

    // Join the heartbeat monitor thread.
    void join();

    // Run the actual loop to check watchdog heartbeat.
    virtual void runLoop();

    // Set the terminal flag and notify the heartbeat monitor thread to stop.
    void stop();

    // Set the last update time of watchdog thread.
    void setLastWorkListUpdateTime(
        std::chrono::time_point<std::chrono::steady_clock> time);

    int getDumpTimeout() const;

    // Util function to get the timeout error message
    std::string getCNCLWatchdogTimeoutErrorMsg(const std::string& extraMsg);

    // Util function to get the timeout exit message
    std::string getCNCLWatchdogTimeoutExitMsg(const std::string& exitReason);

   protected:
    // We need to keep a reference to the PG instance so that we can access
    // the member functions of the PG instance. We store a raw pointer on
    // purpose because the heartbeat monitor thread now still lives within the
    // lifetime of the PG instance.
    ProcessGroupCNCL* pg_;

   private:
    // Whether or not to print C++ stack traces to logs on unclean shutdown.
    bool logCppStackOnUncleanShutdown_;

    // The time interval used for deciding whether there is no watchdog
    // heartbeat.
    int heartbeatTimeoutInSec_;

    // timeout for the dump to finish.
    int waitTimeoutDumpInMilSec_;

    // Interval of check coordinated signals in ProcessGroupCNCL from other
    // ranks e.g., trigger the dump of the debugging info for timeout when
    // notified.
    int coordCheckIntervalMilSec_;

    // We gate the heartbeat monitor thread so that we can roll it out
    // gradually.
    bool watchdogHeartbeatMonitorEnabled_;

    // Monitor thread which checks the heartbeat of Watchdog thread.
    // If the monitor thread finds there is no heartbeat, it will dump debug
    // info and then kill the watchdog thread to avoid hang.
    std::thread cnclHeartbeatMonitorThread_;

    // Whether or not we should terminate the heartbeat monitoring threads.
    std::atomic<bool> terminateHeartbeatMonitorThread_{false};

    // Condition Variable for monitor thread to wake up early
    std::condition_variable monitorWakeUpCV_;

    // Whether or not to dump debug info on exception including both watchdog
    // timeout and cncl errors.
    bool dumpOnTimeoutOrEx_;

    // Mutex to Guard monitorWakeUpCV_
    std::mutex monitorMutex_;

    // The last update time of WorkList inside watchdog thread.
    std::chrono::time_point<std::chrono::steady_clock> lastWorkListUpdateTime_;
  };

  // Class that runs as a side thread to check whether the CNCL collective
  // is timed out or errors on the cached CNCL communicators.
  class Watchdog {
   public:
    Watchdog(ProcessGroupCNCL* pg);
    virtual ~Watchdog() = default;

    // Start the watchdog thread.
    void start();

    // Join the watchdog thread.
    void join();

    // Function that runs as part of a separate thread and checks for errors on
    // CNCL communicators. We need a separate thread to check for CNCL errors
    // since we can't rely on the user calling certain methods like wait(),
    // isCompleted() etc. to detect and remediate errors. In addition to this,
    // we need a mechanism to safely abort and remove CNCL communicators from
    // our cache. This can be done cleanly by having a thread for the
    // ProcessGroupCNCL class. Attempting to modify the communicator cache from
    // the WorkCNCL class might run into issues with object lifetime since the
    // ProcessGroupCNCL object might get destroyed before the WorkCNCL object.
    void run();

    // Watchdog's inside loop.
    // Takes care of cleaning up completed work, and aborting upon failure or
    // timeout.
    void runLoop();

    // Notify the loop inside watchdog.
    void notify();

    void checkAndSetRemoteError();

    // A helper function to get the src rank of a signal from the Store. This is
    // nonblocking function returning -1 if the signal is not available yet.
    int getSignalSrcRank(
        c10::intrusive_ptr<c10d::Store>& store,
        const std::string& signal);

    uint64_t getHeartbt() const;

    void setDesyncDebug(bool desyncDebug);

   private:
    std::thread cnclCommWatchdogThread_;

    // We need to keep a reference to the PG instance so that we can access
    // the member functions of the PG instance. We store a raw pointer on
    // purpose because the watchdog thread now still lives within the
    // lifetime of the PG instance.
    ProcessGroupCNCL* pg_;

    // Whether the CNCL watchdog should rethrow MLU errors.
    bool rethrowMLUErrors_ = false;

    std::exception_ptr watchDogException_ = nullptr;

    // Condition Variable for watchdog thread sleep
    std::condition_variable workMetaListCV_;

    // Heartbeat of watchdog thread.
    std::atomic_uint64_t heartbeat_{};

    // Whether or not to propagate detected errors to all ranks in the same PG
    // through TCPStore.
    bool propagatePgError_;

    // Whether or not to enable timeout root cause analysis.
    bool desyncDebug_;

    DesyncDebugger desyncDebugger_;
  };

  // If you wish to create multiple process groups, each with a potentially
  // different rank and size, you can do so by passing a new store instance
  // to each one. If you have only a single store object, you can
  // use the `c10d::PrefixStore` to derive scoped instances.
  // This is also what the Python API in torch.distributed does.
  //
  // The process group instance keeps a reference to the store because
  // it may be used long after the constructor runs. In fact, the constructor
  // doesn't create any CNCL communicators. A single CNCL communicator can
  // only be used on a specific set of devices, and are therefore created
  // on-demand when a collective runs. If another collective is executed later,
  // against a different set of devices, the process group creates another CNCL
  // communicator. These CNCL communicators are cached and reused if possible.
  ProcessGroupCNCL(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  virtual ~ProcessGroupCNCL();

  // This function returns a local uid for ProcessGroupCNCL.
  uint64_t getUid() {
    return static_cast<uint64_t>(local_id_);
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return c10::static_intrusive_pointer_cast<Backend::Options>(options_);
  }

  const std::string getBackendName() const override {
    return std::string(CNCL_BACKEND_NAME);
  }

  void runHookLoop();

  // This function iterates through the list of WorkCNCL objects in the
  // workList_ corresponding to incomplete collectives and then aborts CNCL
  // communicators associated with timed out collectives.
  void abortTimedOutCollectives(
      std::unordered_set<std::string>& aborted_comm_ids);

  bool supportsCoalescing() const override {
    return true;
  }

  bool supportsSplitting() const override {
    return true;
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    options_->timeout = timeout;
  }

  void startCoalescing() override;

  c10::intrusive_ptr<c10d::Work> endCoalescing() override;

  // For specifying a composite optype, such as ALLGATHER and REDUCE_SCATTER
  c10::intrusive_ptr<c10d::Work> endCoalescing(c10d::OpType optype);

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<c10d::Work> _broadcast_oop(
      at::Tensor& output_tensors,
      at::Tensor& input_tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions());

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts =
          c10d::AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> _reduce_oop(
      at::Tensor& outputTensors,
      at::Tensor& inputTensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions());

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

  // Unsupported Ops
  c10::intrusive_ptr<c10d::Work> scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dst_rank,
      int tag) override;

  c10::intrusive_ptr<c10d::Work> recv(
      std::vector<at::Tensor>& tensors,
      int src_rank,
      int tag) override;

  c10::intrusive_ptr<c10d::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override;

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      std::vector<int64_t>& output_split_sizes,
      std::vector<int64_t>& input_split_sizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

#if (                                \
    defined(NEUWARE_CNCL_VERSION) && \
    NEUWARE_CNCL_VERSION > VERSION_ENCODE(1, 30, 1))
  c10::intrusive_ptr<c10d::Work> transpose_alltoall(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      const int64_t gather_dim,
      const int64_t shard_dim);
#else
#endif

  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  // Create a new ProcessGroupCNCL instance
  static c10::intrusive_ptr<c10d::Backend> createProcessGroupCNCL(
      const c10::intrusive_ptr<c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::milliseconds& timeout);

  int64_t getCommPtr();

  static void groupStart();

  static void groupEnd();

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store.
  void setSequenceNumberForGroup() override;

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override;

  void registerOnCompletionHook(
      std::function<void(std::shared_ptr<c10d::WorkInfo>)>&& hook) override;

  void waitForPendingWorks() override;

  void enableCollectivesTiming() override;

  c10::intrusive_ptr<c10d::Backend> split(
      const c10::intrusive_ptr<c10d::Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<c10d::Backend::Options>& opts) override;

  c10::intrusive_ptr<Backend> merge(
      const c10::intrusive_ptr<c10d::Store>& store,
      const c10::intrusive_ptr<c10d::Backend::Options>& opts,
      const int& rank,
      const int& size) override;

  // Helper function for iteratively aborting communicators in the provided map
  void abortCommsFromMap(
      std::unordered_map<std::string, std::shared_ptr<CNCLComm>>& cnclComms_map,
      std::optional<std::string> abortReason);

  void abort() override;

  // Destroy (shutdown) this backend --normal exit.
  void shutdown() override;

  // If all comms on this PG are fully initialized, return true.
  bool isInitialized();

  // This method adds a temporary extension for the timeout period,
  // applying to all collectives between the calling of this API and
  // the completion of the first collective on the MLU. While this feature
  // provides flexibility in specific scenarios, it introduces statefulness
  // to timeout setting. Therefore, it is advisable to use this API sparingly
  // and consider alternative approaches, such as directly setting the timeout
  // or utilizing a barrier collective (one can set any timeout to the barrier),
  // whenever feasible.
  void addEphemeralTimeout(const std::chrono::milliseconds& timeout);

  // This function is only intended for testing purposes because we don't
  // want to expose the `WorkCNCL` via pybind. It verifies whether the
  // `opTimeout_` of the provided WorkCNCL instance is the same as the specified
  // timeout.
  bool verifyWorkTimeoutForTest(
      const c10::intrusive_ptr<c10d::Work> work,
      const std::chrono::milliseconds& timeout);

  // Returns the global rank of the device. This function assumes that users
  // always create a default global process group(PG) which includes all
  // devices. It is called in the constructor of ProcessGroupCNCL, so it always
  // return the rank_ of the the very first PG created, aka, default global PG.
  const int& globalRank() const;

  const c10::intrusive_ptr<c10d::Store>& globalStore() const;

  // Returns the global ranks of a PG.
  const std::vector<uint64_t>& groupRanks() const;

  // Util function to assign timeout to each work.
  void assignTimeoutToWork(
      const c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>& work,
      const c10::intrusive_ptr<Options>& option);

  // get a cnclComm_t.
  int64_t getCnclComm(int rankid);

  // Broadcast flight-recorder dump signal
  void broadcastDumpSignal();

  c10d::ErrorType getError() override;

 protected:
  uint64_t getWatchdogHeartbt() const;

  // Instance of the heartbeat monitor thread.
  std::unique_ptr<HeartbeatMonitor> heartbeatMonitor_;

  // Instance of the watchdog thread.
  std::unique_ptr<Watchdog> watchdog_;

  // Function that directly trigger std::abort so that the whole process
  // gets terminated.
  virtual void terminateProcess(std::string errMsg);

  static const int64_t k_watchdog_thread_sleep_millis;
  static const int64_t k_work_cleanup_thread_sleep_millis;

  // A helper function to wait for a future to complete or timeout.
  // Returns true if the future completes before timeout, false otherwise.
  bool waitForFutureOrTimeout(
      std::future<bool>& fut,
      const std::chrono::milliseconds& timeOutMilSec,
      const std::string& futDescription,
      c10d::C10dLoggingData& debugLog,
      bool throwException = false);

  // A helper function to guess the device id of the current rank, based on
  // bounded device or used device. Do not use this function if you already know
  // the device id to operate on.
  c10::DeviceIndex guessDeviceId() const;

  // The store is used to broadcast the CNCL unique ID of rank 0. This store
  // comes with prefix and it is different across ProcessGroup CNCL instances
  // (aka, different ProcessGroups).
  c10::intrusive_ptr<c10d::Store> store_;

  // Reference to the store without prefix so that keys are same across all
  // ProcessGroup CNCL instances and (key, value) pairs written to the store are
  // global.
  c10::intrusive_ptr<c10d::Store> globalStore_;

  // Whether or not the workCleanupThread is used to perform async error
  // handling.
  ErrorHandlingMode async_error_handling_ = NoHandling;

  c10d::ErrorType error_ = c10d::ErrorType::SUCCESS;

  std::mutex errorMutex_;

  // Whether or not to create start MLUEvent and enable timing for start
  // and end events. Note that enableTiming_ is always true if desync_debug_
  // is set to true.
  std::atomic<bool> enableTiming_{};

  uint64_t seqCollective_{0};
  // Counting for the sequential number of NCCL P2P calls.
  uint64_t seqP2P_{0};

  // Incrementing counter for logical operations (collective or p2p) issued on
  // the ProcessGroup
  uint64_t op_id_{0};

  // Mutex for watchdog.
  std::mutex watchdog_cv_mutex_;

  // Map from cnclCliqueId to appropriate communicator.
  std::unordered_map<std::string, std::shared_ptr<CNCLComm>>
      cncl_id_to_comm_map_;

  // We gate the mluEventCache so that we can roll it out gradually.
  std::atomic<bool> mluEventCacheEnabled_{};

  std::thread onCompletionHookThread_;

  // Size of ring buffer where we store CNCL Traces for debugging.
  int traceBufferSize_;

  // This is the signal from watchdog threads to indicate whether the monitor
  // thread should dump. Making it static so that it is accessiable from all the
  // PGs. With this flag, monitor thread would dump debug info under any one of
  // the three conditions:
  //
  // 1: watchdog thread of any PG detects a collective timeout.
  // 2: timeout signal is received from other ranks through tcpstore
  // 3: current PG's watchdog heartbeat timeout occurs.
  //
  // Note that only the monitor thread from PG0 will dump the debug info for
  // case one and two so that the debug info is only dumped once.
  static std::atomic<bool> shouldDump_;

  // Mutex to Guard workMetaList_
  std::mutex workMetaListMutex_;

  // Vector to store WorkCNCL pointers
  std::list<ProcessGroupCNCL::WorkCNCL> workMetaList_;

  // Mutex to Guard workMetaList_
  std::mutex completedWorkListMutex_;

  // Condition Variable for watchdog thread sleep
  std::condition_variable completedWorkListCV_;

  std::list<ProcessGroupCNCL::WorkCNCL> completedWorkList_;

  // Thread that removes CNCL Work upon timeout
  std::thread work_cleanup_thread_;

  std::string logPrefix_;

  // Number of devices on this node.
  int localDeviceCount_{0};

  // Set of communicators that this process group has aborted and their
  // cnclCliqueId has been written to the store. We don't need a lock
  // for this map since only the watchdog thread accesses this set. The
  // set contains the string representation of cnclCliqueId.
  std::unordered_set<std::string> aborted_comms_;

  // Add Work Pointer to workVector
  void workEnqueue(c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL>);

  // Helper that broadcasts cncl clique ID to all ranks through the store
  void broadcastCNCLCliqueID(
      cnclCliqueId* cncl_id,
      const bool is_p2p_op,
      const std::string& p2p_key,
      const int p2p_rank);

  // Helper that either looks up the cached CNCL communicators only
  std::shared_ptr<CNCLComm> getCNCLComm(const std::string& device_key);

  std::shared_ptr<CNCLComm> initCNCLComm(
      const std::string& device_key,
      at::Device& device,
      c10d::OpType op_type,
      const int p2p_rank = 0,
      const bool is_send_recv_self = false);

  // Wrapper method which can be overridden for tests.
  virtual std::exception_ptr checkForCNCLErrors(
      std::shared_ptr<CNCLComm>& cncl_comms);

  // Ensure thaht if record is True, the work obj will be enqueued via
  // workEnqueue
  virtual c10::intrusive_ptr<ProcessGroupCNCL::WorkCNCL> initWork(
      at::Device device,
      int rank,
      c10d::OpType opType,
      bool isP2P,
      const char* profilingTitle = nullptr,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {},
      bool record = false);

  // The lock which protects the write/read of
  // ephemeralTimeoutActive_/ephemeralTimeoutInflight_.
  std::mutex mtxTimeoutExtension_;

  // The ephemeral timeout added on top of existing timeout for works issued
  // before first work finishes.
  std::chrono::milliseconds ephemeralTimeoutActive_ =
      std::chrono::milliseconds(0);

  // The ephemeral timeout addition which has been already applied to work.
  std::chrono::milliseconds ephemeralTimeoutInflight_ =
      std::chrono::milliseconds(0);

  const c10::intrusive_ptr<Options> options_;

  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminate_process_group_;

  // The number of ProcessGroupCNCL created on the current rank.
  size_t local_id_;

  // The number of CNCL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t cncl_comm_counter_{0};

  // The CNCL communicator that the process group has cached.
  // The key is a list of MLU devices that an operation is operating on
  // The MLU devices are stored in a device sequence and the cache CNCL
  // communicator is associated with this MLU device sequence
  //
  // e.g. If the process group op only uses device 0, then the value of
  // the used device string stored (value of the hashmap) would be "0".
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
  //      then the value of the used device string (key) stored would be
  //      "0,1,2,3,4,5,6,7"
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
  //      then the value of the used device string stored would be
  //      "0,4,5,6,7,1,2,3"
  //
  //      Note that the order of the device for the tensor list matters.
  std::unordered_map<std::string, std::shared_ptr<CNCLComm>> dev_cncl_comm_map_;

  // The MLU streams used by CNCL kernels
  std::unordered_map<std::string, torch_mlu::MLUStream> cncl_streams_;

  // The MLUEvents used to sync CNCL streams
  std::unordered_map<std::string, torch_mlu::MLUEvent> cncl_events_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  int waitTcdpDumpInMilSec_;

  std::string tcdpDumpCmd_;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  int coalescing_state_ = 0;

  // c10d::Stores device indexe for all collectives run inside a coalescing
  // block
  at::Device coalescedDevice_ = at::Device("mlu");

  // c10d::Stores communicator for all collectives run inside a coalescing block
  std::shared_ptr<CNCLComm> coalescedComm_ = nullptr;

  // Whether the coalesced calls are sync or async.
  bool coalescedAsync_;

  // keeps track of input and output tensors when coalescing is in flight.  Will
  // hand over these tensors to WorkCNCL's stash when coalescing is ended.
  TensorShelf coalescedTensors_;

  // Some ops may have completed, but user still hasn't called `work.wait()`.
  // When watchdog detects this, it transfers the TensorShelf from `work` to
  // this `shelves` structure. Next time we execute ProcessGroupCNCL's methods
  // on main thread, we clear the `shelves` in one shot. This is mainly because
  // watchdog (a side thread) unstashing the shelf directly seems to cause some
  // problem.
  std::vector<std::shared_ptr<TensorShelf>> shelvesToUnstash_;
  std::mutex shelvesMutex_;

  // In the timeout case and we will dump debug info such as the CNCL flight
  // recorder to storage. Down the road, if we have more complicated or blocking
  // operations, we might need to use a side thread to do it.
  bool dumpDebuggingInfo(
      bool includeStackTrace = true,
      bool onlyActive = false);

  void dumpExtraDebuggingInfo();

  // Abort all communicators on this rank.
  bool abortComms(std::optional<std::string> abortReason = std::nullopt);

 private:
  int globalRankStart_;
  int globalRankStride_;
  //  Helper that encapsulates work shared across all collective communication
  //  primitives.
  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      c10d::OpType op_type,
      bool asyncOp,
      const char* profilingTitle = nullptr);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<c10d::Work> collective(
      at::Tensor& inputs,
      at::Tensor& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      c10d::OpType op_type,
      bool asyncOp,
      const char* profilingTitle = nullptr);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<c10d::Work> collective(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      c10d::OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr);

  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> collectiveCoalesced(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      c10d::OpType op_type,
      bool asyncOp,
      const char* profilingTitle = nullptr);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<c10d::Work> collectiveCoalesced(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      c10d::OpType op_type,
      bool asyncOp,
      const char* profilingTitle = nullptr);

  // Helper that encapsulates work shared across point-to-point communication
  // primitives. It is the same structure as the helper used for collective
  // communicaiton primitives.
  template <typename Fn>
  c10::intrusive_ptr<c10d::Work> pointToPoint(
      at::Tensor& tensors,
      Fn fn,
      int peer,
      c10d::OpType op_type,
      const char* profilingTitle = nullptr);

  c10::intrusive_ptr<c10d::Work> allreduce_impl(
      at::Tensor& tensor,
      const char* profilingTitle = "cncl:all_reduce",
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions());

  // Checks for CNCL errors on each of the communicators and returns an
  // appropriate exception_ptr (nullptr if no errors).
  static std::exception_ptr checkForCNCLErrorsInternal(
      std::shared_ptr<CNCLComm>& cncl_comms);

  // Generates a prefix that is unique to this process group and rank, for
  // disambiguating logs
  std::string createLogPrefix() const;

  // Returns the unique prefix created in createLogPrefix
  const std::string& logPrefix() const;

  // A helper function to broadcast a signal (key) from a src rank to all other
  // ranks using the specified store.
  void broadcastSignal(
      c10::intrusive_ptr<c10d::Store>& store,
      const std::string& signal,
      int srcRank);

  // The number of active cnclGroupStart() calls. This counter will be increased
  // by 1 when cnclGroupStart() is called and decreased by 1 when cnclGroupEnd()
  // is called.
  static thread_local uint64_t cnclActiveGroupCounter_;

  // Mutex to guard maps like dev_cncl_comm_map_
  std::mutex mutex_;

  std::shared_ptr<c10d::ProcessGroupStatus> pgStatus_ =
      std::make_shared<c10d::ProcessGroupStatus>();
};

// Reset the flighrecorder recordings for the current rank.
TORCH_MLU_API void reset_cncl_trace();

TORCH_MLU_API std::string dump_cncl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive);

TORCH_MLU_API std::string dump_cncl_trace_json(
    bool includeCollectives,
    bool onlyActive);

// Gets a mutable reference to a global optional function.  Heartbeat Monitor
// will use this function to dump traces, if available. Inside fbcode, we store
// a function here that uses an internal tool for process tracing
TORCH_MLU_API std::optional<
    std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper();

// Similar to get_cpp_trace_dumper, this stores a function defined in
// torch-python layer that lets us check whether the GIL can be acquired,
// helpful for instrumenting in cases where a hang was observed.
typedef bool (*gil_checker_t)();

TORCH_MLU_API gil_checker_t& get_gil_checker();

} // namespace torch_mlu
