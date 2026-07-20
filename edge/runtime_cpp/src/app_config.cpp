#include "visionops_runtime/app_config.hpp"

#include <set>
#include <stdexcept>

namespace visionops::runtime {

bool is_supported_mock_task_type(const std::string& task_type) {
  static const std::set<std::string> supported = {
      "detection", "obb", "segmentation", "roi_classification", "classification"};
  return supported.count(task_type) != 0;
}

void validate_app_config(const AppConfig& config) {
  if (config.host.empty() || config.device_id.empty() || config.component.empty()) {
    throw std::invalid_argument("host、device-id 和 component 不能为空");
  }
  if (config.port == 0) {
    throw std::invalid_argument("端口必须位于 1 到 65535");
  }
  if (!is_supported_mock_task_type(config.mock_task_type)) {
    throw std::invalid_argument("不支持的 mock task type: " + config.mock_task_type);
  }
  if (config.backend != "mock" && config.backend != "rknn") {
    throw std::invalid_argument("backend 仅支持 mock 或 rknn");
  }
  if (config.preprocess_backend != "cpu" && config.preprocess_backend != "rga" &&
      config.preprocess_backend != "auto") {
    throw std::invalid_argument("preprocess-backend 仅支持 cpu、rga 或 auto");
  }
  if (config.rga_mode != "resize_rgb") {
    throw std::invalid_argument("rga-mode 当前仅支持 resize_rgb");
  }
  if (config.score_threshold_override > 1.0 || config.nms_threshold_override > 1.0) {
    throw std::invalid_argument("score-threshold 和 nms-threshold 必须位于 0 到 1");
  }
  if (config.max_detections_override < 0 || config.mask_max_points_override < 0) {
    throw std::invalid_argument("max-detections 和 mask-max-points 不得为负数");
  }
  if (config.frame_source != "mock" && config.frame_source != "test_image" &&
      config.frame_source != "v4l2" && config.frame_source != "hp60c_bridge" &&
      config.frame_source != "hp60c") {
    throw std::invalid_argument("frame-source 仅支持 mock、test_image、v4l2 或 hp60c_bridge");
  }
  if ((config.frame_source == "hp60c_bridge" || config.frame_source == "hp60c") &&
      config.hp60c_url.empty()) {
    throw std::invalid_argument("hp60c-url 不能为空");
  }
  if (config.camera_width <= 0 || config.camera_height <= 0 || config.camera_fps <= 0) {
    throw std::invalid_argument("camera-width、camera-height、camera-fps 必须为正数");
  }
  if (config.camera_open_timeout_ms <= 0 || config.camera_read_timeout_ms <= 0 ||
      config.stale_frame_timeout_ms <= 0) {
    throw std::invalid_argument("camera timeout 必须为正数");
  }
  if (config.reconnect_failure_threshold <= 0 || config.reconnect_initial_ms <= 0 ||
      config.reconnect_max_ms < config.reconnect_initial_ms) {
    throw std::invalid_argument(
        "camera reconnect 参数非法：failure-threshold/initial 必须为正数，max 不得小于 initial");
  }
}


}  // namespace visionops::runtime
