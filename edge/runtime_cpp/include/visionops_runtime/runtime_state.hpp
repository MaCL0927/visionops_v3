#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

namespace visionops::runtime {

struct RuntimeCounters {
  std::uint64_t frames_in{0};
  std::uint64_t frames_inferred{0};
  std::uint64_t frames_dropped{0};
  std::uint64_t errors{0};
};

struct RuntimeSnapshot {
  bool running{false};
  std::string mode{"idle"};
  std::string health{"ok"};
  double uptime_s{0.0};
  RuntimeCounters counters;
  std::optional<std::string> last_frame_id;
  std::optional<std::string> last_result_id;
  std::optional<std::string> latest_result_json;
};

struct InferenceIdentity {
  std::string frame_id;
  std::string result_id;
};

class RuntimeState {
 public:
  RuntimeState();

  RuntimeSnapshot snapshot() const;
  RuntimeSnapshot start_preview();
  RuntimeSnapshot stop_preview();
  InferenceIdentity begin_inference();
  void complete_inference(const InferenceIdentity& identity, std::string result_json);

 private:
  double uptime_seconds() const;

  mutable std::mutex mutex_;
  std::chrono::steady_clock::time_point started_at_;
  bool running_{false};
  std::string mode_{"idle"};
  std::string health_{"ok"};
  RuntimeCounters counters_;
  std::uint64_t frame_sequence_{0};
  std::uint64_t result_sequence_{0};
  std::optional<std::string> last_frame_id_;
  std::optional<std::string> last_result_id_;
  std::optional<std::string> latest_result_json_;
};

}  // namespace visionops::runtime
