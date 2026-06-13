#include "visionops_runtime/runtime_app.hpp"

#include <iomanip>
#include <sstream>
#include <utility>

#include "visionops_runtime/json_utils.hpp"
#include "visionops_runtime/preprocess.hpp"

namespace visionops::runtime {

namespace {

std::string optional_json(const std::optional<std::string>& value) {
  return value ? '"' + json_escape(*value) + '"' : "null";
}

}  // namespace

RuntimeApp::RuntimeApp(AppConfig config)
    : config_(std::move(config)), rknn_runner_(config_.mock_task_type) {
  validate_app_config(config_);
}

std::string RuntimeApp::health_json() const {
  const auto snapshot = state_.snapshot();
  const auto now = now_timestamp_ms();
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3)
         << "{\"schema_version\":\"1.0\",\"message_type\":\"runtime_health\""
         << ",\"device_id\":\"" << json_escape(config_.device_id) << '"'
         << ",\"component\":\"" << json_escape(config_.component) << '"'
         << ",\"timestamp_ms\":" << now
         << ",\"trace_id\":\"" << make_trace_id(now) << '"'
         << ",\"source\":\"runtime:mock\",\"status\":\"ok\""
         << ",\"health\":\"ok\",\"ready\":true,\"version\":\"0.1.0\""
         << ",\"uptime_s\":" << snapshot.uptime_s << '}';
  return stream.str();
}

std::string RuntimeApp::status_json() const { return status_json(state_.snapshot()); }

std::string RuntimeApp::status_json(const RuntimeSnapshot& snapshot) const {
  const auto now = now_timestamp_ms();
  const double inference_fps = snapshot.running && snapshot.mode == "detect" ? 1.0 : 0.0;
  const double snapshot_fps = snapshot.running && snapshot.mode == "preview" ? 2.0 : 0.0;
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3)
         << "{\"schema_version\":\"1.0\",\"message_type\":\"runtime_status\""
         << ",\"device_id\":\"" << json_escape(config_.device_id) << '"'
         << ",\"component\":\"" << json_escape(config_.component) << '"'
         << ",\"timestamp_ms\":" << now
         << ",\"trace_id\":\"" << make_trace_id(now) << '"'
         << ",\"source\":\"runtime:mock\",\"status\":\"ok\""
         << ",\"running\":" << json_bool(snapshot.running)
         << ",\"mode\":\"" << json_escape(snapshot.mode) << '"'
         << ",\"health\":\"" << json_escape(snapshot.health) << '"'
         << ",\"uptime_s\":" << snapshot.uptime_s
         << ",\"loaded_model\":{\"model_id\":\"model-mock-001\",\"model_name\":\"visionops-runtime-mock\",\"model_version\":\"1.0.0\",\"task_type\":\""
         << json_escape(config_.mock_task_type) << "\",\"backend\":\"mock\"}"
         << ",\"camera_connected\":true"
         << ",\"fps\":{\"camera_fps\":15.0,\"inference_fps\":" << inference_fps
         << ",\"snapshot_fps\":" << snapshot_fps << '}'
         << ",\"latency_ms\":{\"latest\":16.0,\"average\":16.0,\"p95\":16.0}"
         << ",\"counters\":{\"frames_in\":" << snapshot.counters.frames_in
         << ",\"frames_inferred\":" << snapshot.counters.frames_inferred
         << ",\"frames_dropped\":" << snapshot.counters.frames_dropped
         << ",\"errors\":" << snapshot.counters.errors << '}'
         << ",\"last_result_id\":" << optional_json(snapshot.last_result_id)
         << ",\"last_frame_id\":" << optional_json(snapshot.last_frame_id)
         << ",\"last_error\":null"
         << ",\"resources\":{\"cpu_percent\":0.0,\"memory_mb\":0.0,\"npu_percent\":0.0,\"temperature_c\":0.0}}";
  return stream.str();
}

std::string RuntimeApp::start_preview() {
  stream_worker_.start_preview();
  return status_json(state_.start_preview());
}

std::string RuntimeApp::stop_preview() {
  stream_worker_.stop_preview();
  return status_json(state_.stop_preview());
}

std::string RuntimeApp::infer_once() {
  const auto identity = state_.begin_inference();
  const auto frame = stream_worker_.next_frame(identity.sequence);
  const auto preprocess = preprocess_mock_frame(frame);
  const auto inference = rknn_runner_.infer(preprocess);
  const auto result = inference_result_json(identity, frame, preprocess, inference);
  state_.complete_inference(identity, result);
  return result;
}

std::string RuntimeApp::inference_result_json(
    const InferenceIdentity& identity,
    const MockFrame& frame,
    const PreprocessOutput& preprocess,
    const MockInferenceOutput& inference) const {
  const auto now = now_timestamp_ms();
  const double total_ms = preprocess.elapsed_ms + inference.inference_ms + inference.postprocess_ms;
  std::ostringstream stream;
  stream << "{\"schema_version\":\"1.0\",\"message_type\":\"inference_result\""
         << ",\"device_id\":\"" << json_escape(config_.device_id) << '"'
         << ",\"component\":\"" << json_escape(config_.component) << '"'
         << ",\"timestamp_ms\":" << now
         << ",\"trace_id\":\"" << make_trace_id(now) << '"'
         << ",\"frame_id\":\"" << json_escape(identity.frame_id) << '"'
         << ",\"source\":\"runtime:mock\",\"status\":\"ok\""
         << ",\"result_id\":\"" << json_escape(identity.result_id) << '"'
         << ",\"task_type\":\"" << json_escape(inference.task_type) << '"'
         << ",\"model\":{\"model_id\":\"model-mock-001\",\"model_name\":\"visionops-runtime-mock\",\"model_version\":\"1.0.0\",\"backend\":\"mock\",\"input_size\":{\"width\":"
         << preprocess.input_width << ",\"height\":" << preprocess.input_height << "}}"
         << ",\"image\":{\"width\":" << frame.width << ",\"height\":" << frame.height << '}'
         << ",\"timing\":{\"preprocess_ms\":" << preprocess.elapsed_ms
         << ",\"inference_ms\":" << inference.inference_ms
         << ",\"postprocess_ms\":" << inference.postprocess_ms
         << ",\"total_ms\":" << total_ms << '}'
         << inference.result_payload_json << '}';
  return stream.str();
}

std::optional<std::string> RuntimeApp::latest_result_json() const {
  return state_.snapshot().latest_result_json;
}

const std::vector<std::uint8_t>& RuntimeApp::snapshot_jpeg() const {
  return snapshot_provider_.snapshot_jpeg();
}

const AppConfig& RuntimeApp::config() const { return config_; }

std::string RuntimeApp::snapshot_frame_id() const {
  return state_.snapshot().last_frame_id.value_or("frame-mock-placeholder");
}

void RuntimeApp::record_error() { state_.record_error(); }

}  // namespace visionops::runtime
