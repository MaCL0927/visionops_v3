#include "visionops_runtime/model_package.hpp"

#include <filesystem>
#include <string>

#include "visionops_runtime/model_config.hpp"

namespace visionops::runtime {

namespace {

namespace fs = std::filesystem;

void append_error(std::string& target, const std::string& error) {
  if (error.empty()) {
    return;
  }
  if (!target.empty()) {
    target += "; ";
  }
  target += error;
}

}  // namespace

LoadedModelInfo load_model_package(const AppConfig& app_config) {
  LoadedModelInfo info;
  info.task_type = app_config.mock_task_type;

  if (app_config.model_dir.empty()) {
    info.class_names = {"object"};
    info.labels_count = static_cast<int>(info.class_names.size());
    return info;
  }

  const fs::path model_dir = fs::path(app_config.model_dir).lexically_normal();
  info.model_id = model_dir.filename().string();
  info.model_name = model_dir.filename().string();
  info.model_version = "unknown";
  info.rknn_path = (model_dir / "model.rknn").lexically_normal().string();
  info.config_path = (model_dir / "model.yaml").lexically_normal().string();
  info.labels_path.clear();

  if (!fs::exists(model_dir) || !fs::is_directory(model_dir)) {
    append_error(info.model_load_error, "模型目录不存在: " + model_dir.string());
  }
  if (!fs::exists(model_dir / "model.rknn")) {
    append_error(info.model_load_error, "模型目录缺少 model.rknn: " + model_dir.string());
  }
  if (!fs::exists(model_dir / "model.yaml")) {
    append_error(info.model_load_error, "模型目录缺少 model.yaml: " + model_dir.string());
  }

  ModelConfigData yaml;
  if (fs::exists(model_dir / "model.yaml")) {
    std::string error;
    if (load_model_config_yaml(info.config_path, yaml, error)) {
      if (!yaml.model_id.empty()) info.model_id = yaml.model_id;
      if (!yaml.model_name.empty()) info.model_name = yaml.model_name;
      if (!yaml.model_version.empty()) info.model_version = yaml.model_version;
      if (!yaml.task_type.empty()) info.task_type = yaml.task_type;
      if (!yaml.target_platform.empty()) info.target_platform = yaml.target_platform;
      if (!yaml.runtime_preprocess.empty()) info.runtime_preprocess = yaml.runtime_preprocess;
      if (yaml.input_width > 0 && yaml.input_height > 0) {
        info.input_width = yaml.input_width;
        info.input_height = yaml.input_height;
      }
      if (yaml.score_threshold >= 0.0) info.score_threshold = yaml.score_threshold;
      if (yaml.nms_threshold >= 0.0) info.nms_threshold = yaml.nms_threshold;
      info.class_names = yaml.class_names;
      info.labels_count = static_cast<int>(info.class_names.size());
    } else {
      append_error(info.model_load_error, error);
    }
  }

  if (info.class_names.empty()) {
    info.class_names.push_back("object");
    info.labels_count = static_cast<int>(info.class_names.size());
  }
  if (!is_supported_mock_task_type(info.task_type)) {
    append_error(info.model_load_error, "不支持的模型 task_type: " + info.task_type);
    info.task_type = app_config.mock_task_type;
  }
  if ((info.task_type == "classification" || info.task_type == "classify") &&
      (info.runtime_preprocess.empty() || info.runtime_preprocess == "letterbox")) {
    // Old classification packages may have been generated with letterbox.
    // YOLOv8-cls should use direct resize on the edge side unless explicitly overridden.
    info.runtime_preprocess = "resize";
  }
  return info;
}

}  // namespace visionops::runtime
