#include "visionops_runtime/input_roi.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace visionops::runtime {

namespace {

bool allowed_coordinate_space(const std::string& value) {
  return value == "runtime_snapshot" || value == "bridge_output_image";
}

bool allowed_resize_mode(const std::string& value) {
  return value == "letterbox" || value == "resize";
}

double aspect_ratio(int width, int height) {
  return height > 0 ? width / static_cast<double>(height) : 0.0;
}

int normalized_to_pixel(double value, int size) {
  return static_cast<int>(std::lround(value * static_cast<double>(size)));
}

}  // namespace

bool validate_input_roi_config(
    const InputRoiConfig& config,
    std::string& error_code,
    std::string& error_message) {
  error_code.clear();
  error_message.clear();
  if (!config.enabled) return true;

  const auto fail = [&](const std::string& code, const std::string& message) {
    error_code = code;
    error_message = message;
    return false;
  };

  if (!allowed_coordinate_space(config.coordinate_space)) {
    return fail(
        "INPUT_ROI_COORDINATE_SPACE_UNSUPPORTED",
        "input_roi.coordinate_space 仅支持 runtime_snapshot 或 bridge_output_image");
  }
  if (config.source_width <= 0 || config.source_height <= 0) {
    return fail(
        "INPUT_ROI_SOURCE_RESOLUTION_INVALID",
        "input_roi.source_resolution 必须提供正数 width/height");
  }
  if (!config.has_pixel_xyxy && !config.has_normalized_xyxy) {
    return fail(
        "INPUT_ROI_COORDINATES_MISSING",
        "启用 input_roi 时必须提供 pixel_xyxy 或 normalized_xyxy");
  }
  if (config.has_pixel_xyxy) {
    if (config.x0 < 0 || config.y0 < 0 || config.x1 <= config.x0 || config.y1 <= config.y0 ||
        config.x1 > config.source_width || config.y1 > config.source_height) {
      return fail(
          "INPUT_ROI_PIXEL_BOUNDS_INVALID",
          "input_roi.pixel_xyxy 超出 source_resolution 或宽高非法");
    }
  }
  if (config.has_normalized_xyxy) {
    if (config.normalized_x0 < 0.0 || config.normalized_y0 < 0.0 ||
        config.normalized_x1 > 1.0 || config.normalized_y1 > 1.0 ||
        config.normalized_x1 <= config.normalized_x0 ||
        config.normalized_y1 <= config.normalized_y0) {
      return fail(
          "INPUT_ROI_NORMALIZED_BOUNDS_INVALID",
          "input_roi.normalized_xyxy 必须位于 [0,1] 且右下角大于左上角");
    }
  }
  if (config.has_pixel_xyxy && config.has_normalized_xyxy) {
    const int nx0 = normalized_to_pixel(config.normalized_x0, config.source_width);
    const int ny0 = normalized_to_pixel(config.normalized_y0, config.source_height);
    const int nx1 = normalized_to_pixel(config.normalized_x1, config.source_width);
    const int ny1 = normalized_to_pixel(config.normalized_y1, config.source_height);
    constexpr int kTolerancePx = 2;
    if (std::abs(nx0 - config.x0) > kTolerancePx ||
        std::abs(ny0 - config.y0) > kTolerancePx ||
        std::abs(nx1 - config.x1) > kTolerancePx ||
        std::abs(ny1 - config.y1) > kTolerancePx) {
      return fail(
          "INPUT_ROI_COORDINATES_INCONSISTENT",
          "input_roi.pixel_xyxy 与 normalized_xyxy 不一致");
    }
  }
  const int expected_width = config.has_pixel_xyxy
      ? config.x1 - config.x0
      : normalized_to_pixel(config.normalized_x1, config.source_width) -
            normalized_to_pixel(config.normalized_x0, config.source_width);
  const int expected_height = config.has_pixel_xyxy
      ? config.y1 - config.y0
      : normalized_to_pixel(config.normalized_y1, config.source_height) -
            normalized_to_pixel(config.normalized_y0, config.source_height);
  if (expected_width < 2 || expected_height < 2) {
    return fail("INPUT_ROI_TOO_SMALL", "input_roi 裁剪区域宽高必须至少为 2 像素");
  }
  if (config.crop_width > 0 && std::abs(config.crop_width - expected_width) > 1) {
    return fail(
        "INPUT_ROI_CROP_RESOLUTION_INCONSISTENT",
        "input_roi.crop_resolution.width 与坐标计算结果不一致");
  }
  if (config.crop_height > 0 && std::abs(config.crop_height - expected_height) > 1) {
    return fail(
        "INPUT_ROI_CROP_RESOLUTION_INCONSISTENT",
        "input_roi.crop_resolution.height 与坐标计算结果不一致");
  }
  if (!allowed_resize_mode(config.resize_mode)) {
    return fail(
        "INPUT_ROI_RESIZE_MODE_UNSUPPORTED",
        "input_roi.resize_mode 仅支持 letterbox 或 resize");
  }
  if (config.pad_value < 0 || config.pad_value > 255) {
    return fail("INPUT_ROI_PAD_VALUE_INVALID", "input_roi.pad_value 必须位于 0 到 255");
  }
  return true;
}

bool resolve_input_roi(
    const InputRoiConfig& config,
    int actual_width,
    int actual_height,
    ResolvedInputRoi& resolved,
    std::string& error_code,
    std::string& error_message) {
  resolved = {};
  resolved.full_width = actual_width;
  resolved.full_height = actual_height;
  error_code.clear();
  error_message.clear();

  if (actual_width <= 0 || actual_height <= 0) {
    error_code = "INPUT_IMAGE_SIZE_INVALID";
    error_message = "Runtime 输入图像宽高必须为正数";
    return false;
  }
  if (!config.enabled) {
    resolved.x = 0;
    resolved.y = 0;
    resolved.width = actual_width;
    resolved.height = actual_height;
    return true;
  }
  if (!validate_input_roi_config(config, error_code, error_message)) return false;

  resolved.enabled = true;
  if (actual_width == config.source_width && actual_height == config.source_height &&
      config.has_pixel_xyxy) {
    resolved.x = config.x0;
    resolved.y = config.y0;
    resolved.width = config.x1 - config.x0;
    resolved.height = config.y1 - config.y0;
  } else {
    const double configured_aspect = aspect_ratio(config.source_width, config.source_height);
    const double actual_aspect = aspect_ratio(actual_width, actual_height);
    // A different size is allowed only when the field of view keeps the same
    // aspect ratio. The normalized coordinates then preserve the ROI location.
    if (std::fabs(configured_aspect - actual_aspect) > 1e-3) {
      std::ostringstream stream;
      stream << "input_roi 采集分辨率为 " << config.source_width << 'x' << config.source_height
             << "，当前 Runtime 帧为 " << actual_width << 'x' << actual_height
             << "，宽高比不一致";
      error_code = "INPUT_ROI_SOURCE_ASPECT_MISMATCH";
      error_message = stream.str();
      return false;
    }
    if (!config.has_normalized_xyxy) {
      error_code = "INPUT_ROI_SOURCE_RESOLUTION_MISMATCH";
      error_message = "当前帧分辨率与 input_roi.source_resolution 不同，且未提供 normalized_xyxy";
      return false;
    }
    const int x0 = normalized_to_pixel(config.normalized_x0, actual_width);
    const int y0 = normalized_to_pixel(config.normalized_y0, actual_height);
    const int x1 = normalized_to_pixel(config.normalized_x1, actual_width);
    const int y1 = normalized_to_pixel(config.normalized_y1, actual_height);
    resolved.x = std::clamp(x0, 0, actual_width - 1);
    resolved.y = std::clamp(y0, 0, actual_height - 1);
    const int clipped_x1 = std::clamp(x1, resolved.x + 1, actual_width);
    const int clipped_y1 = std::clamp(y1, resolved.y + 1, actual_height);
    resolved.width = clipped_x1 - resolved.x;
    resolved.height = clipped_y1 - resolved.y;
    resolved.scaled_from_normalized = true;
  }

  if (resolved.x < 0 || resolved.y < 0 || resolved.width < 2 || resolved.height < 2 ||
      resolved.x + resolved.width > actual_width || resolved.y + resolved.height > actual_height) {
    error_code = "INPUT_ROI_RESOLVED_BOUNDS_INVALID";
    error_message = "input_roi 解析后的区域超出当前 Runtime 输入图像";
    return false;
  }
  return true;
}

}  // namespace visionops::runtime
