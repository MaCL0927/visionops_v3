#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace visionops::runtime {

// Runtime 内部统一图像结构。当前生产链路优先使用 RGB888 三通道数据，
// V4L2 的 YUYV 输入会在帧源层转换为 RGB888 后再进入预处理。
struct ImageBuffer {
  int width{0};
  int height{0};
  int channels{3};
  std::string pixel_format{"RGB888"};
  std::vector<std::uint8_t> data;
  std::uint64_t timestamp_ms{0};
  std::uint64_t sequence{0};
  std::string camera_id;
  std::string source;
};

inline bool image_buffer_valid_rgb(const ImageBuffer& image) {
  return image.width > 0 && image.height > 0 && image.channels == 3 &&
         !image.data.empty();
}

}  // namespace visionops::runtime
