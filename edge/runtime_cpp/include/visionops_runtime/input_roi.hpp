#pragma once

#include <string>

namespace visionops::runtime {

// Model-package input ROI. Coordinates refer to the full Runtime snapshot image
// before model preprocessing. x1/y1 are exclusive bounds.
struct InputRoiConfig {
  bool enabled{false};
  std::string coordinate_space{"runtime_snapshot"};
  int source_width{0};
  int source_height{0};
  int x0{0};
  int y0{0};
  int x1{0};
  int y1{0};
  double normalized_x0{0.0};
  double normalized_y0{0.0};
  double normalized_x1{1.0};
  double normalized_y1{1.0};
  bool has_pixel_xyxy{false};
  bool has_normalized_xyxy{false};
  int crop_width{0};
  int crop_height{0};
  std::string resize_mode{"letterbox"};
  int pad_value{114};
};

struct ResolvedInputRoi {
  bool enabled{false};
  int full_width{0};
  int full_height{0};
  int x{0};
  int y{0};
  int width{0};
  int height{0};
  bool scaled_from_normalized{false};
};

bool validate_input_roi_config(
    const InputRoiConfig& config,
    std::string& error_code,
    std::string& error_message);

bool resolve_input_roi(
    const InputRoiConfig& config,
    int actual_width,
    int actual_height,
    ResolvedInputRoi& resolved,
    std::string& error_code,
    std::string& error_message);

}  // namespace visionops::runtime
