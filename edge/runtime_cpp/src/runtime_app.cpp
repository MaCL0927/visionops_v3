#include "visionops_runtime/runtime_app.hpp"

#include <iomanip>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

#include "visionops_runtime/json_utils.hpp"
#include "visionops_runtime/postprocess_detect.hpp"
#include "visionops_runtime/postprocess_obb.hpp"
#include "visionops_runtime/postprocess_seg.hpp"
#include "visionops_runtime/preprocess.hpp"
#include "visionops_runtime/rga_preprocess.hpp"

namespace visionops::runtime {

namespace {

struct PreparedModelRuntime {
  LoadedModelInfo model_info;
  std::unique_ptr<RknnRunner> runner;
};

std::string optional_json(const std::optional<std::string>& value) {
  return value ? '"' + json_escape(*value) + '"' : "null";
}

void append_error(std::string& target, const std::string& error) {
  if (error.empty()) return;
  if (!target.empty()) target += "; ";
  target += error;
}

std::string runtime_source(const AppConfig& config) {
  return "runtime:" + config.backend;
}

std::string frame_prefix_for_source(const std::string& frame_source) {
  if (frame_source == "hp60c_bridge" || frame_source == "hp60c") return "frame-hp60c";
  if (frame_source == "v4l2") return "frame-v4l2";
  if (frame_source == "test_image") return "frame-test-image";
  return "frame-mock";
}

std::string result_prefix_for_backend(const std::string& backend) {
  return backend == "rknn" ? "result-rknn" : "result-mock";
}

std::string fallback_postprocess(const std::string& task_type) {
  if (task_type == "obb") return make_obb_payload_json();
  if (task_type == "segmentation") return make_segmentation_payload_json();
  if (task_type == "classification") return make_classification_payload_json();
  if (task_type == "roi_classification") return make_roi_classification_payload_json();
  return make_detection_payload_json();
}

std::optional<std::string> json_string_field(const std::string& text, const std::string& key) {
  const std::string marker = '"' + key + '"';
  const auto key_position = text.find(marker);
  if (key_position == std::string::npos) {
    return std::nullopt;
  }
  const auto colon = text.find(':', key_position + marker.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  auto position = colon + 1;
  while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) {
    ++position;
  }
  if (position >= text.size() || text[position] != '"') {
    return std::nullopt;
  }
  ++position;
  std::string value;
  bool escaped = false;
  for (; position < text.size(); ++position) {
    const char ch = text[position];
    if (escaped) {
      value.push_back(ch);
      escaped = false;
    } else if (ch == '\\') {
      escaped = true;
    } else if (ch == '"') {
      return value;
    } else {
      value.push_back(ch);
    }
  }
  return std::nullopt;
}

std::string format_ms(double value) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << value;
  return stream.str();
}

void replace_token(std::string& target, const std::string& token, const std::string& replacement) {
  std::size_t position = 0;
  while ((position = target.find(token, position)) != std::string::npos) {
    target.replace(position, token.size(), replacement);
    position += replacement.size();
  }
}


std::optional<std::pair<int, int>> infer_input_size_from_tensor_infos(
    const std::vector<TensorInfo>& input_infos) {
  if (input_infos.empty()) {
    return std::nullopt;
  }
  const auto& dims = input_infos.front().dimensions;
  if (dims.size() == 4) {
    // RKNN 常见两类输入维度：NHWC=[1,H,W,3]，NCHW=[1,3,H,W]。
    if ((dims[3] == 3 || dims[3] == 1) && dims[1] > 0 && dims[2] > 0) {
      return std::make_pair(static_cast<int>(dims[2]), static_cast<int>(dims[1]));
    }
    if ((dims[1] == 3 || dims[1] == 1) && dims[2] > 0 && dims[3] > 0) {
      return std::make_pair(static_cast<int>(dims[3]), static_cast<int>(dims[2]));
    }
  }
  if (dims.size() == 3) {
    // HWC=[H,W,C] 或 CHW=[C,H,W]，用于少数无 batch 维模型。
    if ((dims[2] == 3 || dims[2] == 1) && dims[0] > 0 && dims[1] > 0) {
      return std::make_pair(static_cast<int>(dims[1]), static_cast<int>(dims[0]));
    }
    if ((dims[0] == 3 || dims[0] == 1) && dims[1] > 0 && dims[2] > 0) {
      return std::make_pair(static_cast<int>(dims[2]), static_cast<int>(dims[1]));
    }
  }
  return std::nullopt;
}

PreparedModelRuntime prepare_model_runtime(const AppConfig& config) {
  PreparedModelRuntime prepared;
  prepared.model_info = load_model_package(config);
  prepared.runner = create_rknn_runner(config.backend, prepared.model_info.task_type);
  if (config.score_threshold_override >= 0.0) {
    prepared.model_info.score_threshold = config.score_threshold_override;
  }
  if (config.nms_threshold_override >= 0.0) {
    prepared.model_info.nms_threshold = config.nms_threshold_override;
  }
  prepared.model_info.backend = prepared.runner->backend_name();
  if (config.backend == "rknn" &&
      (prepared.model_info.rknn_path.empty() ||
       !std::filesystem::exists(prepared.model_info.rknn_path))) {
    append_error(
        prepared.model_info.model_load_error,
        "RKNN 模型文件不存在: " +
            (prepared.model_info.rknn_path.empty()
                 ? std::string("<未配置>")
                 : prepared.model_info.rknn_path));
  }
  const RunnerModelConfig runner_config{
      prepared.model_info.task_type,
      prepared.model_info.input_width,
      prepared.model_info.input_height,
      config.dump_rknn_io};
  if (!prepared.runner->load_model(prepared.model_info.rknn_path, runner_config)) {
    append_error(prepared.model_info.model_load_error, prepared.runner->last_error());
  } else {
    const auto actual_input_size = infer_input_size_from_tensor_infos(prepared.runner->input_infos());
    if (actual_input_size.has_value() &&
        actual_input_size->first > 0 && actual_input_size->second > 0 &&
        (prepared.model_info.input_width != actual_input_size->first ||
         prepared.model_info.input_height != actual_input_size->second)) {
      // model.yaml 是模型包元信息来源，但 RKNN 输入 tensor 尺寸是实际推理的硬约束。
      // 如果 YAML 里遗留了错误 input_size，优先使用 RKNN 真实输入尺寸，避免 rknn_inputs_set -5。
      prepared.model_info.input_width = actual_input_size->first;
      prepared.model_info.input_height = actual_input_size->second;
    }
  }
  return prepared;
}

FrameSourceConfig make_frame_source_config(const AppConfig& config) {
  FrameSourceConfig frame_config;
  frame_config.type = config.frame_source;
  frame_config.camera_device = config.camera_device;
  frame_config.camera_width = config.camera_width;
  frame_config.camera_height = config.camera_height;
  frame_config.camera_fps = config.camera_fps;
  frame_config.camera_pixel_format = config.camera_pixel_format;
  frame_config.hp60c_url = config.hp60c_url;
  frame_config.hp60c_snapshot_path = config.hp60c_snapshot_path;
  frame_config.hp60c_health_path = config.hp60c_health_path;
  frame_config.test_image = config.test_image;
  frame_config.snapshot_source = config.snapshot_source;
  frame_config.enable_camera_thread = config.enable_camera_thread;
  frame_config.camera_open_timeout_ms = config.camera_open_timeout_ms;
  frame_config.camera_read_timeout_ms = config.camera_read_timeout_ms;
  return frame_config;
}

std::string frame_source_json(const FrameSourceStatus& frame_source) {
  std::ostringstream stream;
  stream << "{\"type\":\"" << json_escape(frame_source.type) << '"'
         << ",\"camera_id\":\"" << json_escape(frame_source.camera_id) << '"'
         << ",\"device\":\"" << json_escape(frame_source.device) << '"'
         << ",\"opened\":" << json_bool(frame_source.opened)
         << ",\"width\":" << frame_source.width
         << ",\"height\":" << frame_source.height
         << ",\"fps\":" << frame_source.fps
         << ",\"pixel_format\":\"" << json_escape(frame_source.pixel_format) << '"'
         << ",\"latest_frame_id\":"
         << (frame_source.latest_frame_id.empty() ? "null" : '"' + json_escape(frame_source.latest_frame_id) + '"')
         << ",\"latest_timestamp_ms\":" << frame_source.latest_timestamp_ms
         << ",\"snapshot_encoder\":\"" << json_escape(frame_source.snapshot_encoder) << '"'
         << ",\"last_error\":"
         << (frame_source.last_error.empty() ? "null" : '"' + json_escape(frame_source.last_error) + '"')
         << '}';
  return stream.str();
}

}  // namespace

RuntimeApp::RuntimeApp(AppConfig config)
    : config_(std::move(config)),
      stream_worker_(make_frame_source_config(config_)) {
  validate_app_config(config_);
  auto prepared = prepare_model_runtime(config_);
  model_info_ = std::move(prepared.model_info);
  rknn_runner_ = std::move(prepared.runner);
}

bool RuntimeApp::runtime_degraded() const {
  std::lock_guard<std::recursive_mutex> lock(model_mutex_);
  const auto frame_source = stream_worker_.status();
  return model_info_.degraded() || !rknn_runner_->is_loaded() ||
         (config_.frame_source == "v4l2" && !frame_source.last_error.empty());
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
         << ",\"source\":\"" << json_escape(runtime_source(config_)) << "\",\"status\":\"ok\""
         << ",\"health\":\"" << (runtime_degraded() ? "degraded" : "ok") << '"'
         << ",\"ready\":true,\"version\":\"0.1.0\""
         << ",\"uptime_s\":" << snapshot.uptime_s << '}';
  return stream.str();
}

std::string RuntimeApp::status_json() const { return status_json(state_.snapshot()); }

std::string RuntimeApp::status_json(const RuntimeSnapshot& snapshot) const {
  const auto now = now_timestamp_ms();
  const auto frame_source = stream_worker_.status();
  const double inference_fps = snapshot.running && snapshot.mode == "detect" ? 1.0 : 0.0;
  const double snapshot_fps = snapshot.running && snapshot.mode == "preview" ? frame_source.fps : 0.0;
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3)
         << "{\"schema_version\":\"1.0\",\"message_type\":\"runtime_status\""
         << ",\"device_id\":\"" << json_escape(config_.device_id) << '"'
         << ",\"component\":\"" << json_escape(config_.component) << '"'
         << ",\"timestamp_ms\":" << now
         << ",\"trace_id\":\"" << make_trace_id(now) << '"'
         << ",\"source\":\"" << json_escape(runtime_source(config_)) << "\",\"status\":\"ok\""
         << ",\"running\":" << json_bool(snapshot.running)
         << ",\"mode\":\"" << json_escape(snapshot.mode) << '"'
         << ",\"health\":\""
         << json_escape(runtime_degraded() ? "degraded" : snapshot.health) << '"'
         << ",\"uptime_s\":" << snapshot.uptime_s
         << ",\"loaded_model\":" << loaded_model_json()
         << ",\"camera_connected\":" << json_bool(frame_source.opened && frame_source.last_error.empty())
         << ",\"frame_source\":" << frame_source_json(frame_source)
         << ",\"preprocess\":{\"backend_requested\":\"" << json_escape(config_.preprocess_backend)
         << "\",\"backend_active\":\"" << json_escape(config_.preprocess_backend == "auto" && rga_backend_compiled() ? "rga" : config_.preprocess_backend)
         << "\",\"rga_mode\":\"" << json_escape(config_.rga_mode)
         << "\",\"rga_available\":" << json_bool(rga_backend_compiled()) << '}'
         << ",\"fps\":{\"camera_fps\":" << frame_source.fps << ",\"inference_fps\":" << inference_fps
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
  std::lock_guard<std::recursive_mutex> lock(model_mutex_);
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3)
         << "{\"model_id\":\"" << json_escape(model_info_.model_id) << '"'
         << ",\"model_name\":\"" << json_escape(model_info_.model_name) << '"'
         << ",\"model_version\":\"" << json_escape(model_info_.model_version) << '"'
         << ",\"task_type\":\"" << json_escape(model_info_.task_type) << '"'
         << ",\"backend\":\"" << json_escape(model_info_.backend) << '"'
         << ",\"runner_loaded\":" << json_bool(rknn_runner_->is_loaded())
         << ",\"rknn_compiled\":" << json_bool(rknn_backend_compiled())
         << ",\"input_count\":" << rknn_runner_->input_count()
         << ",\"output_count\":" << rknn_runner_->output_count()
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
  const auto identity = state_.begin_inference(
      frame_prefix_for_source(config_.frame_source),
      result_prefix_for_backend(config_.backend));
  auto frame_result = stream_worker_.next_frame(identity.sequence);
  auto frame = frame_result.frame;
  if (!frame_result.ok) {
    const auto error = inference_error_json(
        identity,
        frame,
        "CAMERA_FRAME_UNAVAILABLE",
        frame_result.error,
        frame_result.capture_ms,
        frame_result.decode_ms,
        nullptr,
        nullptr,
        "frame_source_error");
    state_.complete_inference_error(identity, error);
    return error;
  }
  if (config_.backend == "rknn" && !rknn_runner_->is_loaded()) {
    const auto error = inference_error_json(
        identity,
        frame,
        "RKNN_MODEL_NOT_LOADED",
        rknn_runner_->last_error(),
        frame_result.capture_ms,
        frame_result.decode_ms);
    state_.complete_inference_error(identity, error);
    return error;
  }

  ImageBuffer source_image;
  std::string image_error;
  if ((config_.frame_source == "test_image" || !config_.test_image.empty()) &&
      !config_.test_image.empty()) {
    load_test_image(config_.test_image, source_image, image_error);
    source_image.sequence = identity.sequence;
    source_image.source = "frame_source:test_image";
  } else if (image_buffer_valid_rgb(frame_result.image)) {
    source_image = frame_result.image;
  } else {
    source_image = make_mock_image(frame);
  }
  if (!image_error.empty()) {
    const auto error = inference_error_json(
        identity, frame, "TEST_IMAGE_LOAD_FAILED", image_error, frame_result.capture_ms, frame_result.decode_ms);
    state_.complete_inference_error(identity, error);
    return error;
  }
  if (source_image.width > 0 && source_image.height > 0) {
    frame.width = source_image.width;
    frame.height = source_image.height;
  }

  std::lock_guard<std::recursive_mutex> model_lock(model_mutex_);
  const PreprocessOptions preprocess_options{config_.preprocess_backend, config_.rga_mode};
  const auto preprocess = preprocess_image(
      frame, source_image, model_info_.input_width, model_info_.input_height, preprocess_options);
  if (!preprocess.error.empty()) {
    const auto error = inference_error_json(
        identity,
        frame,
        "PREPROCESS_FAILED",
        preprocess.error,
        frame_result.capture_ms,
        frame_result.decode_ms,
        &preprocess);
    state_.complete_inference_error(identity, error);
    return error;
  }
  RknnInput input;
  input.width = preprocess.input.width;
  input.height = preprocess.input.height;
  input.channels = preprocess.input.channels;
  input.data = preprocess.input.data;
  auto inference = rknn_runner_->infer(input);
  if (!inference.success) {
    const auto error = inference_error_json(
        identity,
        frame,
        "RKNN_INFERENCE_FAILED",
        inference.error.empty() ? rknn_runner_->last_error() : inference.error,
        frame_result.capture_ms,
        frame_result.decode_ms,
        &preprocess,
        &inference);
    state_.complete_inference_error(identity, error);
    return error;
  }

  if (config_.backend == "rknn") {
    const auto postprocess_started = std::chrono::steady_clock::now();
    const PostprocessConfig postprocess_config{
        model_info_.class_names,
        static_cast<float>(model_info_.score_threshold),
        static_cast<float>(model_info_.nms_threshold),
        100};
    PostprocessResult postprocess;
    if (model_info_.task_type == "detection" || model_info_.task_type == "detect") {
      postprocess = postprocess_detection(inference.tensors, postprocess_config, preprocess.letterbox);
    } else if (model_info_.task_type == "obb") {
      postprocess = postprocess_obb(inference.tensors, postprocess_config, preprocess.letterbox);
    } else if (model_info_.task_type == "segmentation" || model_info_.task_type == "segment") {
      postprocess = postprocess_segmentation(inference.tensors, postprocess_config, preprocess.letterbox);
    } else {
      const auto error = inference_error_json(
          identity,
          frame,
          "UNSUPPORTED_TASK_TYPE",
          "RKNN 后处理暂不支持 task_type: " + model_info_.task_type,
          frame_result.capture_ms,
          frame_result.decode_ms,
          &preprocess,
          &inference,
          "unsupported_task_type");
      state_.complete_inference_error(identity, error);
      return error;
    }
    inference.postprocess_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - postprocess_started).count();
    if (!postprocess.success) {
      const auto error = postprocess_error_json(
          identity,
          frame,
          frame_result.capture_ms,
          frame_result.decode_ms,
          preprocess,
          inference,
          postprocess.error_code.empty() ? "POSTPROCESS_FAILED" : postprocess.error_code,
          postprocess.error_message,
          postprocess.error_code == "UNSUPPORTED_OUTPUT_SHAPE"
              ? "unsupported_output_shape"
              : "postprocess_failed");
      state_.complete_inference_error(identity, error);
      return error;
    }
    inference.result_payload_json = postprocess.payload_json;
    inference.error = postprocess.warning;
    if (!config_.save_debug_output.empty()) {
      std::error_code filesystem_error;
      std::filesystem::create_directories(config_.save_debug_output, filesystem_error);
      if (!filesystem_error) {
        const auto summary_path = std::filesystem::path(config_.save_debug_output) /
            (identity.result_id + ".json");
        std::ofstream summary(summary_path);
        if (summary) {
          summary << "{\"result_id\":\"" << json_escape(identity.result_id)
                  << "\",\"task_type\":\"" << json_escape(model_info_.task_type)
                  << "\",\"raw_outputs_count\":" << inference.tensors.size()
                  << ",\"postprocess_count\":" << postprocess.result_count
                  << ",\"warning\":\"" << json_escape(postprocess.warning) << "\"}";
        }
      }
    }
  } else if (inference.result_payload_json.empty()) {
    inference.result_payload_json = fallback_postprocess(model_info_.task_type);
    inference.postprocess_ms = 2.0;
  }
  const auto result = inference_result_json(
      identity,
      frame,
      frame_result.capture_ms,
      frame_result.decode_ms,
      preprocess,
      inference);
  state_.complete_inference(identity, result);
  return result;
}

RuntimeApiResult RuntimeApp::switch_model(const std::string& request_body) {
  const auto model_dir_value = json_string_field(request_body, "model_dir");
  if (!model_dir_value || model_dir_value->empty()) {
    return {
        400,
        make_error_json(
            config_.device_id,
            config_.component,
            "MODEL_DIR_REQUIRED",
            "请求体必须包含 model_dir 字段",
            true),
    };
  }

  std::lock_guard<std::recursive_mutex> lock(model_mutex_);
  AppConfig next_config = config_;
  next_config.model_dir = *model_dir_value;
  const std::filesystem::path package_dir = std::filesystem::path(next_config.model_dir).lexically_normal();
  if (!std::filesystem::exists(package_dir) || !std::filesystem::is_directory(package_dir)) {
    return {
        500,
        make_error_json(
            config_.device_id,
            config_.component,
            "MODEL_SWITCH_FAILED",
            "模型目录不存在: " + package_dir.string(),
            true),
    };
  }
  if (!std::filesystem::exists(package_dir / "model.rknn") ||
      !std::filesystem::exists(package_dir / "model.yaml")) {
    return {
        500,
        make_error_json(
            config_.device_id,
            config_.component,
            "MODEL_SWITCH_FAILED",
            "模型目录必须包含 model.rknn 和 model.yaml: " + package_dir.string(),
            true),
    };
  }

  PreparedModelRuntime prepared;
  try {
    prepared = prepare_model_runtime(next_config);
  } catch (const std::exception& error) {
    return {
        500,
        make_error_json(
            config_.device_id,
            config_.component,
            "MODEL_SWITCH_FAILED",
            std::string("模型切换失败: ") + error.what(),
            true),
    };
  }

  if (prepared.model_info.degraded() || !prepared.runner || !prepared.runner->is_loaded()) {
    const std::string error = prepared.model_info.model_load_error.empty()
        ? std::string("新模型加载失败")
        : prepared.model_info.model_load_error;
    return {
        500,
        make_error_json(
            config_.device_id,
            config_.component,
            "MODEL_SWITCH_FAILED",
            error,
            true),
    };
  }
  if (prepared.model_info.rknn_path.empty() || prepared.model_info.config_path.empty() ||
      !std::filesystem::exists(prepared.model_info.rknn_path) ||
      !std::filesystem::exists(prepared.model_info.config_path)) {
    return {
        500,
        make_error_json(
            config_.device_id,
            config_.component,
            "MODEL_SWITCH_FAILED",
            "新模型包不是 M15 标准模型目录，必须包含 model.rknn 和 model.yaml",
            true),
    };
  }

  config_.model_dir = next_config.model_dir;
  model_info_ = std::move(prepared.model_info);
  rknn_runner_ = std::move(prepared.runner);
  return {200, status_json()};
}

std::string RuntimeApp::inference_result_json(
    const InferenceIdentity& identity,
    const MockFrame& frame,
    double capture_ms,
    double decode_ms,
    const PreprocessOutput& preprocess,
    const RknnOutput& inference) const {
  const auto now = now_timestamp_ms();
  (void)frame;
  constexpr char kResultBuildToken[] = "__RESULT_BUILD_MS__";
  constexpr char kTotalToken[] = "__TOTAL_MS__";
  const auto build_started = std::chrono::steady_clock::now();
  std::ostringstream stream;
  stream << "{\"schema_version\":\"1.0\",\"message_type\":\"inference_result\""
         << ",\"device_id\":\"" << json_escape(config_.device_id) << '"'
         << ",\"component\":\"" << json_escape(config_.component) << '"'
         << ",\"timestamp_ms\":" << now
         << ",\"trace_id\":\"" << make_trace_id(now) << '"'
         << ",\"frame_id\":\"" << json_escape(identity.frame_id) << '"'
         << ",\"source\":\"runtime:" << json_escape(config_.backend) << "\",\"status\":\"ok\""
         << ",\"result_id\":\"" << json_escape(identity.result_id) << '"'
         << ",\"task_type\":\"" << json_escape(inference.task_type) << '"'
         << ",\"model\":" << loaded_model_json()
         << ",\"image\":{\"width\":" << preprocess.letterbox.orig_width
         << ",\"height\":" << preprocess.letterbox.orig_height << '}'
         << ",\"timing\":{\"capture_ms\":" << capture_ms
         << ",\"decode_ms\":" << decode_ms
         << ",\"preprocess_ms\":" << preprocess.elapsed_ms
         << ",\"inference_ms\":" << inference.inference_ms
         << ",\"postprocess_ms\":" << inference.postprocess_ms
         << ",\"result_build_ms\":" << kResultBuildToken
         << ",\"total_ms\":" << kTotalToken << '}'
         << ",\"timing_detail\":{\"capture_ms\":" << capture_ms
         << ",\"decode_ms\":" << decode_ms
         << ",\"preprocess_ms\":" << preprocess.elapsed_ms
         << ",\"rknn_set_input_ms\":" << inference.set_input_ms
         << ",\"rknn_run_ms\":" << inference.run_ms
         << ",\"rknn_get_output_ms\":" << inference.get_output_ms
         << ",\"postprocess_ms\":" << inference.postprocess_ms
         << ",\"result_build_ms\":" << kResultBuildToken
         << ",\"total_ms\":" << kTotalToken << '}'
         << inference.result_payload_json;
  if (config_.backend == "rknn") {
    stream << ",\"debug\":{\"rknn_runner_called\":" << json_bool(inference.runner_called)
           << ",\"runner_success\":" << json_bool(inference.success)
           << ",\"raw_outputs_count\":" << inference.tensors.size()
           << ",\"runner_error\":"
           << (inference.error.empty()
                   ? "null"
                   : '"' + json_escape(inference.error) + '"')
           << ",\"preprocess_same_size_fast_path\":" << json_bool(preprocess.same_size_fast_path)
           << ",\"preprocess_backend_requested\":\"" << json_escape(preprocess.backend_requested) << '"'
           << ",\"preprocess_backend_active\":\"" << json_escape(preprocess.backend) << '"'
           << ",\"rga_mode\":\"" << json_escape(preprocess.rga_mode) << '"'
           << ",\"rga_available\":" << json_bool(preprocess.rga_available)
           << ",\"rga_used\":" << json_bool(preprocess.rga_used)
           << ",\"letterbox\":{\"scale\":" << preprocess.letterbox.scale
           << ",\"pad_x\":" << preprocess.letterbox.pad_x
           << ",\"pad_y\":" << preprocess.letterbox.pad_y << "}}";
  }
  stream << '}';
  std::string body = stream.str();
  const double result_build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - build_started).count();
  const double total_ms = capture_ms + decode_ms + preprocess.elapsed_ms +
      inference.inference_ms + inference.postprocess_ms + result_build_ms;
  replace_token(body, kResultBuildToken, format_ms(result_build_ms));
  replace_token(body, kTotalToken, format_ms(total_ms));
  return body;
}


std::string RuntimeApp::postprocess_error_json(
    const InferenceIdentity& identity,
    const MockFrame& frame,
    double capture_ms,
    double decode_ms,
    const PreprocessOutput& preprocess,
    const RknnOutput& inference,
    const std::string& code,
    const std::string& message,
    const std::string& debug_key) const {
  const auto now = now_timestamp_ms();
  (void)frame;
  constexpr char kResultBuildToken[] = "__RESULT_BUILD_MS__";
  constexpr char kTotalToken[] = "__TOTAL_MS__";
  const auto build_started = std::chrono::steady_clock::now();
  std::ostringstream stream;
  stream << "{\"schema_version\":\"1.0\",\"message_type\":\"inference_result\""
         << ",\"device_id\":\"" << json_escape(config_.device_id) << '"'
         << ",\"component\":\"" << json_escape(config_.component) << '"'
         << ",\"timestamp_ms\":" << now
         << ",\"trace_id\":\"" << make_trace_id(now) << '"'
         << ",\"frame_id\":\"" << json_escape(identity.frame_id) << '"'
         << ",\"source\":\"runtime:" << json_escape(config_.backend) << "\",\"status\":\"error\""
         << ",\"result_id\":\"" << json_escape(identity.result_id) << '"'
         << ",\"task_type\":\"" << json_escape(model_info_.task_type) << '"'
         << ",\"model\":" << loaded_model_json()
         << ",\"image\":{\"width\":" << preprocess.letterbox.orig_width
         << ",\"height\":" << preprocess.letterbox.orig_height << '}'
         << ",\"timing\":{\"capture_ms\":" << capture_ms
         << ",\"decode_ms\":" << decode_ms
         << ",\"preprocess_ms\":" << preprocess.elapsed_ms
         << ",\"inference_ms\":" << inference.inference_ms
         << ",\"postprocess_ms\":" << inference.postprocess_ms
         << ",\"result_build_ms\":" << kResultBuildToken
         << ",\"total_ms\":" << kTotalToken << '}'
         << ",\"timing_detail\":{\"capture_ms\":" << capture_ms
         << ",\"decode_ms\":" << decode_ms
         << ",\"preprocess_ms\":" << preprocess.elapsed_ms
         << ",\"rknn_set_input_ms\":" << inference.set_input_ms
         << ",\"rknn_run_ms\":" << inference.run_ms
         << ",\"rknn_get_output_ms\":" << inference.get_output_ms
         << ",\"postprocess_ms\":" << inference.postprocess_ms
         << ",\"result_build_ms\":" << kResultBuildToken
         << ",\"total_ms\":" << kTotalToken << '}'
         << ",\"error\":{\"code\":\"" << json_escape(code)
         << "\",\"message\":\"" << json_escape(message)
         << "\",\"detail\":null,\"recoverable\":true}"
         << ",\"debug\":{\"rknn_runner_called\":" << json_bool(inference.runner_called)
         << ",\"runner_success\":" << json_bool(inference.success)
         << ",\"raw_outputs_count\":" << inference.tensors.size()
         << ",\"raw_outputs\":[";
  for (std::size_t index = 0; index < inference.tensors.size(); ++index) {
    if (index != 0) stream << ',';
    const auto& tensor = inference.tensors[index];
    stream << "{\"index\":" << index
           << ",\"name\":\"" << json_escape(tensor.info.name) << '"'
           << ",\"data_type\":\"" << json_escape(tensor.info.data_type) << '"'
           << ",\"layout\":\"" << json_escape(tensor.info.layout) << '"'
           << ",\"byte_size\":" << tensor.info.byte_size
           << ",\"data_bytes\":" << tensor.data.size()
           << ",\"dims\":[";
    for (std::size_t dim = 0; dim < tensor.info.dimensions.size(); ++dim) {
      if (dim != 0) stream << ',';
      stream << tensor.info.dimensions[dim];
    }
    stream << "]}";
  }
  stream << ']';
  if (!debug_key.empty()) stream << ",\"" << json_escape(debug_key) << "\":true";
  stream << ",\"letterbox\":{\"scale\":" << preprocess.letterbox.scale
         << ",\"pad_x\":" << preprocess.letterbox.pad_x
         << ",\"pad_y\":" << preprocess.letterbox.pad_y << "}}}";
  std::string body = stream.str();
  const double result_build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - build_started).count();
  const double total_ms = capture_ms + decode_ms + preprocess.elapsed_ms +
      inference.inference_ms + inference.postprocess_ms + result_build_ms;
  replace_token(body, kResultBuildToken, format_ms(result_build_ms));
  replace_token(body, kTotalToken, format_ms(total_ms));
  return body;
}

std::string RuntimeApp::inference_error_json(
    const InferenceIdentity& identity,
    const MockFrame& frame,
    const std::string& code,
    const std::string& message,
    double capture_ms,
    double decode_ms,
    const PreprocessOutput* preprocess,
    const RknnOutput* inference,
    const std::string& debug_key) const {
  const auto now = now_timestamp_ms();
  constexpr char kResultBuildToken[] = "__RESULT_BUILD_MS__";
  constexpr char kTotalToken[] = "__TOTAL_MS__";
  const double preprocess_ms = preprocess == nullptr ? 0.0 : preprocess->elapsed_ms;
  const double inference_ms = inference == nullptr ? 0.0 : inference->inference_ms;
  const double postprocess_ms = inference == nullptr ? 0.0 : inference->postprocess_ms;
  const double set_input_ms = inference == nullptr ? 0.0 : inference->set_input_ms;
  const double run_ms = inference == nullptr ? 0.0 : inference->run_ms;
  const double get_output_ms = inference == nullptr ? 0.0 : inference->get_output_ms;
  const bool runner_called = inference != nullptr && inference->runner_called;
  const std::size_t raw_outputs_count = inference == nullptr ? 0 : inference->tensors.size();
  const auto build_started = std::chrono::steady_clock::now();
  std::ostringstream stream;
  stream << "{\"schema_version\":\"1.0\",\"message_type\":\"inference_result\""
         << ",\"device_id\":\"" << json_escape(config_.device_id) << '"'
         << ",\"component\":\"" << json_escape(config_.component) << '"'
         << ",\"timestamp_ms\":" << now
         << ",\"trace_id\":\"" << make_trace_id(now) << '"'
         << ",\"frame_id\":\"" << json_escape(identity.frame_id) << '"'
         << ",\"task_type\":\"" << json_escape(model_info_.task_type) << '"'
         << ",\"source\":\"runtime:" << json_escape(config_.backend) << "\",\"status\":\"error\""
         << ",\"model\":" << loaded_model_json()
         << ",\"image\":{\"width\":" << frame.width << ",\"height\":" << frame.height << '}'
         << ",\"timing\":{\"capture_ms\":" << capture_ms
         << ",\"decode_ms\":" << decode_ms
         << ",\"preprocess_ms\":" << preprocess_ms
         << ",\"inference_ms\":" << inference_ms
         << ",\"postprocess_ms\":" << postprocess_ms
         << ",\"result_build_ms\":" << kResultBuildToken
         << ",\"total_ms\":" << kTotalToken << '}'
         << ",\"timing_detail\":{\"capture_ms\":" << capture_ms
         << ",\"decode_ms\":" << decode_ms
         << ",\"preprocess_ms\":" << preprocess_ms
         << ",\"rknn_set_input_ms\":" << set_input_ms
         << ",\"rknn_run_ms\":" << run_ms
         << ",\"rknn_get_output_ms\":" << get_output_ms
         << ",\"postprocess_ms\":" << postprocess_ms
         << ",\"result_build_ms\":" << kResultBuildToken
         << ",\"total_ms\":" << kTotalToken << '}'
         << ",\"error\":{\"code\":\"" << json_escape(code)
         << "\",\"message\":\"" << json_escape(message)
         << "\",\"detail\":null,\"recoverable\":true}"
         << ",\"debug\":{\"rknn_runner_called\":" << json_bool(runner_called)
         << ",\"raw_outputs_count\":" << raw_outputs_count;
  if (preprocess != nullptr) {
    stream << ",\"preprocess_backend_requested\":\"" << json_escape(preprocess->backend_requested) << '"'
           << ",\"preprocess_backend_active\":\"" << json_escape(preprocess->backend) << '"'
           << ",\"rga_mode\":\"" << json_escape(preprocess->rga_mode) << '"'
           << ",\"rga_available\":" << json_bool(preprocess->rga_available)
           << ",\"rga_used\":" << json_bool(preprocess->rga_used);
  }
  if (!debug_key.empty()) stream << ",\"" << json_escape(debug_key) << "\":true";
  stream << "}}";
  std::string body = stream.str();
  const double result_build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - build_started).count();
  const double total_ms = capture_ms + decode_ms + preprocess_ms +
      inference_ms + postprocess_ms + result_build_ms;
  replace_token(body, kResultBuildToken, format_ms(result_build_ms));
  replace_token(body, kTotalToken, format_ms(total_ms));
  return body;
}

std::optional<std::string> RuntimeApp::latest_result_json() const {
  return state_.snapshot().latest_result_json;
}

std::vector<std::uint8_t> RuntimeApp::snapshot_jpeg() {
  std::vector<std::uint8_t> bridge_jpeg;

  // 如果没有启动 preview 线程，则每次 snapshot 主动取一帧，避免一直返回首帧缓存。
  if (config_.frame_source != "mock" && !stream_worker_.preview_running()) {
    auto frame_result = stream_worker_.next_frame(static_cast<std::uint64_t>(now_timestamp_ms()));

    if (stream_worker_.latest_snapshot_jpeg(bridge_jpeg)) {
      return bridge_jpeg;
    }

    if (frame_result.ok && image_buffer_valid_rgb(frame_result.image)) {
      return snapshot_provider_.snapshot_jpeg(&frame_result.image);
    }
  }

  // preview 线程已启动时，返回后台线程持续更新的最新帧。
  if (stream_worker_.latest_snapshot_jpeg(bridge_jpeg)) {
    return bridge_jpeg;
  }

  ImageBuffer image;
  if (stream_worker_.latest_frame(image)) {
    return snapshot_provider_.snapshot_jpeg(&image);
  }

  return snapshot_provider_.snapshot_jpeg();
}

const AppConfig& RuntimeApp::config() const { return config_; }

std::string RuntimeApp::snapshot_frame_id() const {
  const auto frame_source = stream_worker_.status();
  if (!frame_source.latest_frame_id.empty()) return frame_source.latest_frame_id;
  return state_.snapshot().last_frame_id.value_or(frame_prefix_for_source(config_.frame_source) + "-placeholder");
}

void RuntimeApp::record_error() { state_.record_error(); }

}  // namespace visionops::runtime
