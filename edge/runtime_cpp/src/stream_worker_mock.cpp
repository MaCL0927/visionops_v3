#include "visionops_runtime/stream_worker.hpp"

namespace visionops::runtime {

void StreamWorkerMock::start_preview() { preview_running_ = true; }
void StreamWorkerMock::stop_preview() { preview_running_ = false; }
MockFrame StreamWorkerMock::next_frame(std::uint64_t sequence) const {
  return MockFrame{sequence, 1920, 1080};
}
bool StreamWorkerMock::preview_running() const { return preview_running_; }

}  // namespace visionops::runtime
