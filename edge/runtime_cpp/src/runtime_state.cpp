#include "visionops_runtime/runtime_state.hpp"

#include <iomanip>
#include <sstream>
#include <utility>

namespace visionops::runtime {

namespace {

std::string sequence_id(const char* prefix, std::uint64_t value) {
  std::ostringstream stream;
  stream << prefix << '-' << std::setw(8) << std::setfill('0') << value;
  return stream.str();
}

}  // namespace

RuntimeState::RuntimeState() : started_at_(std::chrono::steady_clock::now()) {}

double RuntimeState::uptime_seconds() const {
  const auto elapsed = std::chrono::steady_clock::now() - started_at_;
  return std::chrono::duration<double>(elapsed).count();
}

RuntimeSnapshot RuntimeState::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  RuntimeSnapshot value;
  value.running = running_;
  value.mode = mode_;
  value.health = health_;
  value.uptime_s = uptime_seconds();
  value.counters = counters_;
  value.last_frame_id = last_frame_id_;
  value.last_result_id = last_result_id_;
  value.latest_result_json = latest_result_json_;
  return value;
}

RuntimeSnapshot RuntimeState::start_preview() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = true;
    mode_ = "preview";
    health_ = "ok";
  }
  return snapshot();
}

RuntimeSnapshot RuntimeState::stop_preview() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    mode_ = "idle";
    health_ = "ok";
  }
  return snapshot();
}

InferenceIdentity RuntimeState::begin_inference() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++frame_sequence_;
  ++result_sequence_;
  ++counters_.frames_in;
  running_ = true;
  mode_ = "detect";
  return {
      sequence_id("frame-mock", frame_sequence_),
      sequence_id("result-mock", result_sequence_),
  };
}

void RuntimeState::complete_inference(
    const InferenceIdentity& identity,
    std::string result_json) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++counters_.frames_inferred;
  last_frame_id_ = identity.frame_id;
  last_result_id_ = identity.result_id;
  latest_result_json_ = std::move(result_json);
  health_ = "ok";
}

}  // namespace visionops::runtime
