#pragma once

#include <cstdint>
#include <string>

namespace visionops::runtime {

struct AppConfig {
  std::string host{"0.0.0.0"};
  std::uint16_t port{18080};
  std::string device_id{"example-edge-001"};
  std::string component{"rknn_runtime"};
  std::string mock_task_type{"detection"};
};

bool is_supported_mock_task_type(const std::string& task_type);
void validate_app_config(const AppConfig& config);

}  // namespace visionops::runtime
