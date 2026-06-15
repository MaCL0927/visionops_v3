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
  value.sequence = sequence_;
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

InferenceIdentity RuntimeState::begin_inference(
    const std::string& frame_prefix,
    const std::string& result_prefix) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++sequence_;
  ++counters_.frames_in;
  running_ = true;
  mode_ = "detect";
  return {
      sequence_id(frame_prefix.c_str(), sequence_),
      sequence_id(result_prefix.c_str(), sequence_),
      sequence_,
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

void RuntimeState::complete_inference_error(
    const InferenceIdentity& identity,
    std::string error_json) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++counters_.errors;
  last_frame_id_ = identity.frame_id;
  last_result_id_ = identity.result_id;
  latest_result_json_ = std::move(error_json);
  health_ = "degraded";
  mode_ = "error";
}

void RuntimeState::record_error() {
  std::lock_guard<std::mutex> lock(mutex_);
  ++counters_.errors;
  health_ = "error";
  mode_ = "error";
}

}  // namespace visionops::runtime
