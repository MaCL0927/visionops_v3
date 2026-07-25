#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "visionops_runtime/image_buffer.hpp"
#include "visionops_runtime/input_roi.hpp"
#include "visionops_runtime/stream_worker.hpp"

namespace visionops::runtime {

struct LetterboxMeta {
  // Full camera/Bridge frame dimensions used by all public output coordinates.
  int orig_width{0};
  int orig_height{0};
  // Actual source rectangle sent to preprocessing. For legacy full-frame models
  // this is [0,0,orig_width,orig_height].
  bool input_roi_enabled{false};
  std::string roi_coordinate_space{"runtime_snapshot"};
  int roi_x{0};
  int roi_y{0};
  int roi_width{0};
  int roi_height{0};
  bool roi_scaled_from_normalized{false};
  int input_width{0};
  int input_height{0};
  int resized_width{0};
  int resized_height{0};
  float scale{1.0F};
  float scale_x{1.0F};
  float scale_y{1.0F};
  float pad_x{0.0F};
  float pad_y{0.0F};
};

struct PreprocessOptions {
  std::string backend{"cpu"};  // cpu, rga, auto
  std::string rga_mode{"resize_rgb"};
  std::string mode{"letterbox"};  // letterbox, resize
  InputRoiConfig input_roi;
};

struct PreprocessOutput {
  MockFrame frame;
  ImageBuffer input;
  LetterboxMeta letterbox;
  double elapsed_ms{0.0};
  double input_roi_resolve_ms{0.0};
  double crop_resize_ms{0.0};
  bool same_size_fast_path{false};
  std::string backend{"cpu"};
  std::string backend_requested{"cpu"};
  std::string rga_mode;
  std::string mode{"letterbox"};
  bool rga_available{false};
  bool rga_used{false};
  bool rga_fused_crop_resize{false};
  std::string error_code;
  std::string error;
  std::string warning;
};


inline float map_model_x_to_full_image_unclamped(float value, const LetterboxMeta& meta) {
  return (value - meta.pad_x) / std::max(meta.scale_x, 1e-6F) +
      static_cast<float>(meta.roi_x);
}

inline float map_model_y_to_full_image_unclamped(float value, const LetterboxMeta& meta) {
  return (value - meta.pad_y) / std::max(meta.scale_y, 1e-6F) +
      static_cast<float>(meta.roi_y);
}

inline float map_model_x_to_full_image(float value, const LetterboxMeta& meta) {
  return std::clamp(
      map_model_x_to_full_image_unclamped(value, meta),
      static_cast<float>(meta.roi_x),
      static_cast<float>(std::max(meta.roi_x, meta.roi_x + meta.roi_width - 1)));
}

inline float map_model_y_to_full_image(float value, const LetterboxMeta& meta) {
  return std::clamp(
      map_model_y_to_full_image_unclamped(value, meta),
      static_cast<float>(meta.roi_y),
      static_cast<float>(std::max(meta.roi_y, meta.roi_y + meta.roi_height - 1)));
}

ImageBuffer make_mock_image(const MockFrame& frame);
bool load_ppm_image(const std::string& path, ImageBuffer& image, std::string& error);
bool load_test_image(const std::string& path, ImageBuffer& image, std::string& error);
PreprocessOutput preprocess_image(
    const MockFrame& frame,
    const ImageBuffer& image,
    int input_width,
    int input_height,
    const PreprocessOptions& options);
PreprocessOutput preprocess_image(
    const MockFrame& frame,
    const ImageBuffer& image,
    int input_width,
    int input_height);
PreprocessOutput preprocess_mock_frame(const MockFrame& frame);

}  // namespace visionops::runtime
