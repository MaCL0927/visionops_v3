#include "visionops_runtime/runtime_app.hpp"

#include <iomanip>
#include <sstream>
#include <utility>

#include "visionops_runtime/json_utils.hpp"
#include "visionops_runtime/postprocess_detect.hpp"
#include "visionops_runtime/postprocess_obb.hpp"
#include "visionops_runtime/postprocess_seg.hpp"
#include "visionops_runtime/preprocess.hpp"

namespace visionops::runtime {

namespace {

std::string optional_json(const std::optional<std::string>& value) {
  return value ? '"' + json_escape(*value) + '"' : "null";
}

void append_error(std::string& target, const std::string& error) {
  if (error.empty()) return;
  if (!target.empty()) target += "; ";
  target += error;
}

std::string fallback_postprocess(const std::string& task_type) {
  if (task_type == "obb") return make_obb_payload_json();
  if (task_type == "segmentation") return make_segmentation_payload_json();
  if (task_type == "classification") return make_classification_payload_json();
  if (task_type == "roi_classification") return make_roi_classification_payload_json();
  return make_detection_payload_json();
}

}  // namespace

RuntimeApp::RuntimeApp(AppConfig config)
    : config_(std::move(config)),
      model_info_(load_model_package(config_)),
      rknn_runner_(create_rknn_runner(config_.backend, model_info_.task_type)) {
  validate_app_config(config_);
  model_info_.backend = rknn_runner_->backend_name();
  const RunnerModelConfig runner_config{
      model_info_.task_type,
      model_info_.input_width,
      model_info_.input_height};
  if (!rknn_runner_->load_model(model_info_.rknn_path, runner_config)) {
    append_error(model_info_.model_load_error, rknn_runner_->last_error());
  }
}

bool RuntimeApp::runtime_degraded() const {
  return model_info_.degraded() || !rknn_runner_->is_loaded();
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
         << ",\"health\":\"" << (runtime_degraded() ? "degraded" : "ok") << '"'
         << ",\"ready\":true,\"version\":\"0.1.0\""
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
         << ",\"health\":\""
         << json_escape(runtime_degraded() ? "degraded" : snapshot.health) << '"'
         << ",\"uptime_s\":" << snapshot.uptime_s
         << ",\"loaded_model\":" << loaded_model_json()
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

std::string RuntimeApp::loaded_model_json() const {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3)
         << "{\"model_id\":\"" << json_escape(model_info_.model_id) << '"'
         << ",\"model_name\":\"" << json_escape(model_info_.model_name) << '"'
         << ",\"model_version\":\"" << json_escape(model_info_.model_version) << '"'
         << ",\"task_type\":\"" << json_escape(model_info_.task_type) << '"'
         << ",\"backend\":\"" << json_escape(model_info_.backend) << '"'
         << ",\"runner_loaded\":" << json_bool(rknn_runner_->is_loaded())
         << ",\"rknn_compiled\":" << json_bool(rknn_backend_compiled())
         << ",\"runner_error\":"
         << (rknn_runner_->last_error().empty()
                 ? "null"
                 : '"' + json_escape(rknn_runner_->last_error()) + '"')
         << ",\"target_platform\":\"" << json_escape(model_info_.target_platform) << '"'
         << ",\"rknn_path\":\"" << json_escape(model_info_.rknn_path) << '"'
         << ",\"config_path\":\"" << json_escape(model_info_.config_path) << '"'
         << ",\"labels_count\":" << model_info_.labels_count
         << ",\"input_size\":{\"width\":" << model_info_.input_width
         << ",\"height\":" << model_info_.input_height << '}'
         << ",\"score_threshold\":" << model_info_.score_threshold
         << ",\"nms_threshold\":" << model_info_.nms_threshold
         << ",\"model_load_error\":"
         << (model_info_.model_load_error.empty()
                 ? "null"
                 : '"' + json_escape(model_info_.model_load_error) + '"')
         << '}';
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
  RknnInput input;
  input.width = model_info_.input_width;
  input.height = model_info_.input_height;
  if (config_.backend == "rknn") {
    // M9.2 尚未接真实相机，使用零值 NHWC 输入验证 Runner 调用链。
    const auto byte_count = static_cast<std::size_t>(input.width) *
        static_cast<std::size_t>(input.height) * static_cast<std::size_t>(input.channels);
    input.data.assign(byte_count, 0);
  }
  auto inference = rknn_runner_->infer(input);
  if (inference.result_payload_json.empty()) {
    inference.result_payload_json = fallback_postprocess(model_info_.task_type);
    inference.postprocess_ms = 2.0;
  }
  const auto result = inference_result_json(identity, frame, preprocess, inference);
  state_.complete_inference(identity, result);
  return result;
}

std::string RuntimeApp::inference_result_json(
    const InferenceIdentity& identity,
    const MockFrame& frame,
    const PreprocessOutput& preprocess,
    const RknnOutput& inference) const {
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
         << ",\"model\":" << loaded_model_json()
         << ",\"image\":{\"width\":" << frame.width << ",\"height\":" << frame.height << '}'
         << ",\"timing\":{\"preprocess_ms\":" << preprocess.elapsed_ms
         << ",\"inference_ms\":" << inference.inference_ms
         << ",\"postprocess_ms\":" << inference.postprocess_ms
         << ",\"total_ms\":" << total_ms << '}'
         << inference.result_payload_json;
  if (config_.backend == "rknn") {
    stream << ",\"debug\":{\"rknn_runner_called\":" << json_bool(inference.runner_called)
           << ",\"runner_success\":" << json_bool(inference.success)
           << ",\"raw_outputs_count\":" << inference.tensors.size()
           << ",\"runner_error\":"
           << (inference.error.empty()
                   ? "null"
                   : '"' + json_escape(inference.error) + '"')
           << '}';
  }
  stream << '}';
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
