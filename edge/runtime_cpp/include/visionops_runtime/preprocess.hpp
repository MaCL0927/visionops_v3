#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "visionops_runtime/image_buffer.hpp"
#include "visionops_runtime/stream_worker.hpp"

namespace visionops::runtime {

struct LetterboxMeta {
  int orig_width{0};
  int orig_height{0};
  int input_width{0};
  int input_height{0};
  int resized_width{0};
  int resized_height{0};
  float scale{1.0F};
  float pad_x{0.0F};
  float pad_y{0.0F};
};

struct PreprocessOutput {
  MockFrame frame;
  ImageBuffer input;
  LetterboxMeta letterbox;
  double elapsed_ms{0.0};
  std::string error;
};

ImageBuffer make_mock_image(const MockFrame& frame);
bool load_ppm_image(const std::string& path, ImageBuffer& image, std::string& error);
bool load_test_image(const std::string& path, ImageBuffer& image, std::string& error);
PreprocessOutput preprocess_image(
    const MockFrame& frame,
    const ImageBuffer& image,
    int input_width,
    int input_height);
PreprocessOutput preprocess_mock_frame(const MockFrame& frame);

}  // namespace visionops::runtime
