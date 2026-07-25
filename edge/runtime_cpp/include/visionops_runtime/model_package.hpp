#pragma once

#include <string>
#include <vector>

#include "visionops_runtime/app_config.hpp"
#include "visionops_runtime/input_roi.hpp"

namespace visionops::runtime {

struct LoadedModelInfo {
  std::string model_id{"model-mock-001"};
  std::string model_name{"visionops-runtime-mock"};
  std::string model_version{"1.0.0"};
  std::string task_type{"detection"};
  std::string backend{"mock"};
  std::string target_platform;
  std::string runtime_preprocess{"letterbox"};
  std::string rknn_path;
  std::string config_path;
  std::string labels_path;
  std::vector<std::string> class_names;
  int labels_count{0};
  int input_width{640};
  int input_height{640};
  double score_threshold{0.5};
  double nms_threshold{0.45};
  int max_detections{100};
  int mask_max_points{160};
  InputRoiConfig input_roi;
  std::string model_load_error;
  // True only for model package/YAML validation errors. Runner availability
  // errors also make the Runtime degraded, but must keep their existing stable
  // inference error codes (for example RKNN_MODEL_NOT_LOADED).
  bool package_invalid{false};

  bool degraded() const { return !model_load_error.empty(); }
};

LoadedModelInfo load_model_package(const AppConfig& app_config);

}  // namespace visionops::runtime
