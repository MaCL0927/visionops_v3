#include "visionops_runtime/preprocess.hpp"

namespace visionops::runtime {

PreprocessOutput preprocess_mock_frame(const MockFrame& frame) {
  return PreprocessOutput{frame, 640, 640, 2.0};
}

}  // namespace visionops::runtime
