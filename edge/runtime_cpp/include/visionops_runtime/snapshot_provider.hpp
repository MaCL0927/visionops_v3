#pragma once

#include <cstdint>
#include <vector>

namespace visionops::runtime {

class SnapshotProvider {
 public:
  const std::vector<std::uint8_t>& snapshot_jpeg() const;
};

}  // namespace visionops::runtime
