#pragma once

#include <string>
#include <vector>

#include "visionops_runtime/input_roi.hpp"

namespace visionops::runtime {

struct ModelConfigData {
  std::string model_id;
  std::string model_name;
  std::string model_version;
  std::string task_type;
  std::string target_platform;
  std::string runtime_preprocess{"letterbox"};
  int input_width{0};
  int input_height{0};
  std::vector<std::string> class_names;
  double score_threshold{-1.0};
  double nms_threshold{-1.0};
  int max_detections{100};
  int mask_max_points{160};
  InputRoiConfig input_roi;
};

bool load_model_config_yaml(
    const std::string& path,
    ModelConfigData& config,
    std::string& error_message);

}  // namespace visionops::runtime
