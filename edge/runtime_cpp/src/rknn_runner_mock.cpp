#include "visionops_runtime/rknn_runner.hpp"

#include <stdexcept>
#include <utility>

#include "visionops_runtime/app_config.hpp"
#include "visionops_runtime/postprocess_detect.hpp"
#include "visionops_runtime/postprocess_obb.hpp"
#include "visionops_runtime/postprocess_seg.hpp"

namespace visionops::runtime {

RknnRunnerMock::RknnRunnerMock(std::string task_type) : task_type_(std::move(task_type)) {
  if (!is_supported_mock_task_type(task_type_)) {
    throw std::invalid_argument("不支持的 mock task type: " + task_type_);
  }
}

MockInferenceOutput RknnRunnerMock::infer(const PreprocessOutput&) const {
  std::string payload;
  if (task_type_ == "obb") {
    payload = make_obb_payload_json();
  } else if (task_type_ == "segmentation") {
    payload = make_segmentation_payload_json();
  } else if (task_type_ == "classification") {
    payload = make_classification_payload_json();
  } else if (task_type_ == "roi_classification") {
    payload = make_roi_classification_payload_json();
  } else {
    payload = make_detection_payload_json();
  }
  return MockInferenceOutput{task_type_, std::move(payload), 12.0, 2.0};
}

const std::string& RknnRunnerMock::task_type() const { return task_type_; }

}  // namespace visionops::runtime
