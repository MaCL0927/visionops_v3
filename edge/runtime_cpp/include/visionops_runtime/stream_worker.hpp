#pragma once

#include <cstdint>

namespace visionops::runtime {

struct MockFrame {
  std::uint64_t sequence{0};
  int width{1920};
  int height{1080};
};

class StreamWorkerMock {
 public:
  void start_preview();
  void stop_preview();
  MockFrame next_frame(std::uint64_t sequence) const;
  bool preview_running() const;

 private:
  bool preview_running_{false};
};

}  // namespace visionops::runtime
