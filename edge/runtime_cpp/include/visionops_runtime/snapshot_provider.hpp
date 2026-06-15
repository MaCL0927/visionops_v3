#pragma once

#include <cstdint>
#include <vector>

#include "visionops_runtime/image_buffer.hpp"

namespace visionops::runtime {

class SnapshotProvider {
 public:
  // 返回内置 1x1 JPEG，用于没有真实帧或 JPEG 编码失败时的稳定兜底。
  const std::vector<std::uint8_t>& fallback_snapshot_jpeg() const;

  // 将最新 RGB888/BGR888 帧编码为 JPEG。失败时返回 fallback_snapshot_jpeg() 的拷贝。
  std::vector<std::uint8_t> snapshot_jpeg(const ImageBuffer* image = nullptr) const;
};

}  // namespace visionops::runtime
