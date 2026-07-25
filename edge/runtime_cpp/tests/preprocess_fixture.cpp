#include <cstdint>
#include <iostream>

#include "visionops_runtime/preprocess.hpp"

int main() {
  using namespace visionops::runtime;

  ImageBuffer image;
  image.width = 8;
  image.height = 6;
  image.channels = 3;
  image.pixel_format = "RGB888";
  image.data.resize(static_cast<std::size_t>(image.width) * image.height * 3);
  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * 3;
      image.data[offset] = static_cast<std::uint8_t>(x);
      image.data[offset + 1] = static_cast<std::uint8_t>(y);
      image.data[offset + 2] = static_cast<std::uint8_t>(x + y);
    }
  }

  MockFrame frame;
  frame.width = image.width;
  frame.height = image.height;

  PreprocessOptions options;
  options.backend = "cpu";
  options.mode = "letterbox";
  options.input_roi.enabled = true;
  options.input_roi.source_width = image.width;
  options.input_roi.source_height = image.height;
  options.input_roi.x0 = 2;
  options.input_roi.y0 = 1;
  options.input_roi.x1 = 6;
  options.input_roi.y1 = 5;
  options.input_roi.has_pixel_xyxy = true;
  options.input_roi.crop_width = 4;
  options.input_roi.crop_height = 4;

  const auto output = preprocess_image(frame, image, 4, 4, options);
  if (!output.error.empty() || output.input.data.size() != 4u * 4u * 3u) {
    std::cerr << output.error << '\n';
    return 1;
  }

  const auto first = output.input.data.data();
  const auto* last = output.input.data.data() + (4u * 4u - 1u) * 3u;
  if (first[0] != 2 || first[1] != 1 || first[2] != 3 ||
      last[0] != 5 || last[1] != 4 || last[2] != 9) {
    std::cerr << "ROI pixel copy mismatch\n";
    return 1;
  }
  if (!output.letterbox.input_roi_enabled || output.letterbox.roi_x != 2 ||
      output.letterbox.roi_y != 1 || output.letterbox.roi_width != 4 ||
      output.letterbox.roi_height != 4 || output.letterbox.orig_width != 8 ||
      output.letterbox.orig_height != 6) {
    std::cerr << "ROI metadata mismatch\n";
    return 1;
  }

  std::cout << "{\"status\":\"ok\",\"image\":{\"width\":8,\"height\":6},"
            << "\"input_roi\":{\"pixel_xyxy\":[2,1,6,5],\"crop_resolution\":{\"width\":4,\"height\":4}},"
            << "\"model_input\":{\"width\":4,\"height\":4},"
            << "\"first_pixel\":[" << static_cast<int>(first[0]) << ','
            << static_cast<int>(first[1]) << ',' << static_cast<int>(first[2]) << "],"
            << "\"last_pixel\":[" << static_cast<int>(last[0]) << ','
            << static_cast<int>(last[1]) << ',' << static_cast<int>(last[2]) << "]}\n";
  return 0;
}
