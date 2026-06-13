#pragma once

#include "visionops_runtime/stream_worker.hpp"

namespace visionops::runtime {

struct PreprocessOutput {
  MockFrame frame;
  int input_width{640};
  int input_height{640};
  double elapsed_ms{2.0};
};

PreprocessOutput preprocess_mock_frame(const MockFrame& frame);

}  // namespace visionops::runtime
