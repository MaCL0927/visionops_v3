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
  std::string backend{"mock"};
  std::string model_dir;
  std::string test_image;
  std::string save_debug_output;
  std::string preprocess_backend{"cpu"};  // cpu, rga, auto
  std::string rga_mode{"resize_rgb"};
  bool dump_rknn_io{false};
  double score_threshold_override{-1.0};
  double nms_threshold_override{-1.0};
  std::string frame_source{"mock"};
  std::string camera_device{"/dev/video0"};
  int camera_width{640};
  int camera_height{480};
  int camera_fps{30};
  std::string camera_pixel_format{"YUYV"};
  std::string hp60c_url{"http://127.0.0.1:18181"};
  std::string hp60c_snapshot_path{"/stream/snapshot.jpg"};
  std::string hp60c_health_path{"/health"};
  std::string snapshot_source{"latest_frame"};
  bool enable_camera_thread{true};
  int camera_open_timeout_ms{3000};
  int camera_read_timeout_ms{1000};
};

bool is_supported_mock_task_type(const std::string& task_type);
void validate_app_config(const AppConfig& config);

}  // namespace visionops::runtime
