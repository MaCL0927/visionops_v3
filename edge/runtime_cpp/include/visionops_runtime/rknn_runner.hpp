#pragma once

#include <string>

#include "visionops_runtime/preprocess.hpp"

namespace visionops::runtime {

struct MockInferenceOutput {
  std::string task_type;
  std::string result_payload_json;
  double inference_ms{12.0};
  double postprocess_ms{2.0};
};

class RknnRunnerMock {
 public:
  explicit RknnRunnerMock(std::string task_type);

  MockInferenceOutput infer(const PreprocessOutput& input) const;
  const std::string& task_type() const;

 private:
  std::string task_type_;
};

}  // namespace visionops::runtime
