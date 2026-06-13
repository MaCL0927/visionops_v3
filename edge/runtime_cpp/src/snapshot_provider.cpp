#include "visionops_runtime/snapshot_provider.hpp"

#include <string_view>

namespace visionops::runtime {

namespace {

std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
  static constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<std::uint8_t> output;
  int accumulator = 0;
  int bits = -8;
  for (const char ch : encoded) {
    if (ch == '=') {
      break;
    }
    const auto position = alphabet.find(ch);
    if (position == std::string_view::npos) {
      continue;
    }
    accumulator = (accumulator << 6) + static_cast<int>(position);
    bits += 6;
    if (bits >= 0) {
      output.push_back(static_cast<std::uint8_t>((accumulator >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return output;
}

}  // namespace

const std::vector<std::uint8_t>& SnapshotProvider::snapshot_jpeg() const {
  // 1x1 像素 JPEG 内嵌于程序，不读取或提交图片文件。
  static const std::vector<std::uint8_t> image = decode_base64(
      "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
      "2wBDAf//////////////////////////////////////////////////////////////////////////////////////"
      "wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/"
      "9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAA"
      "AAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAA"
      "AAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABAf/8QA"
      "FBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPxB//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPxB//8QA"
      "FBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxB//9k=");
  return image;
}

}  // namespace visionops::runtime
