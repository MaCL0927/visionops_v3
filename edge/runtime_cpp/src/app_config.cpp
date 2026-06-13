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
}

}  // namespace visionops::runtime
