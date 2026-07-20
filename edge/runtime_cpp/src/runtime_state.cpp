#include "visionops_runtime/runtime_state.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <vector>
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

  // Report recent throughput instead of retaining a historical FPS forever
  // after inference has stopped.  Two seconds is long enough to smooth a 1 FPS
  // producer while still returning to 0 promptly when the pipeline is idle.
  const auto now = std::chrono::steady_clock::now();
  const auto fps_window_start = now - std::chrono::seconds(2);
  auto recent_begin = std::find_if(
      inference_completed_at_.begin(),
      inference_completed_at_.end(),
      [fps_window_start](const auto& completed_at) {
        return completed_at >= fps_window_start;
      });
  const auto recent_count = static_cast<std::size_t>(
      std::distance(recent_begin, inference_completed_at_.end()));
  if (recent_count >= 2) {
    const double elapsed = std::chrono::duration<double>(
        inference_completed_at_.back() - *recent_begin).count();
    if (elapsed > 0.0) {
      value.inference_fps = static_cast<double>(recent_count - 1) / elapsed;
    }
  }
  if (!inference_latency_ms_.empty()) {
    value.latency_latest_ms = inference_latency_ms_.back();
    value.latency_average_ms =
        std::accumulate(
            inference_latency_ms_.begin(),
            inference_latency_ms_.end(),
            0.0) /
        static_cast<double>(inference_latency_ms_.size());
    std::vector<double> ordered(
        inference_latency_ms_.begin(),
        inference_latency_ms_.end());
    std::sort(ordered.begin(), ordered.end());
    const std::size_t p95_index = std::min(
        ordered.size() - 1,
        static_cast<std::size_t>(
            std::ceil(static_cast<double>(ordered.size()) * 0.95) - 1.0));
    value.latency_p95_ms = ordered[p95_index];
  }
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
      std::chrono::steady_clock::now(),
  };
}

void RuntimeState::complete_inference(
    const InferenceIdentity& identity,
    std::string result_json) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++counters_.frames_inferred;
  const auto completed_at = std::chrono::steady_clock::now();
  inference_completed_at_.push_back(completed_at);
  inference_latency_ms_.push_back(
      std::chrono::duration<double, std::milli>(
          completed_at - identity.started_at).count());
  while (inference_completed_at_.size() > 100) inference_completed_at_.pop_front();
  while (inference_latency_ms_.size() > 100) inference_latency_ms_.pop_front();
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
  const auto completed_at = std::chrono::steady_clock::now();
  inference_completed_at_.push_back(completed_at);
  inference_latency_ms_.push_back(
      std::chrono::duration<double, std::milli>(
          completed_at - identity.started_at).count());
  while (inference_completed_at_.size() > 100) inference_completed_at_.pop_front();
  while (inference_latency_ms_.size() > 100) inference_latency_ms_.pop_front();
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
