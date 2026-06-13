#pragma once

#include <string>
#include <vector>

#include "visionops_runtime/postprocess.hpp"

namespace visionops::runtime {

PostprocessResult postprocess_detection(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox);
std::string make_detection_payload_json();
std::string make_classification_payload_json();
std::string make_roi_classification_payload_json();

}  // namespace visionops::runtime
