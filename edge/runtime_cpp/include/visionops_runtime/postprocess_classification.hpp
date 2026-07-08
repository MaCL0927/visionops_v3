#pragma once

#include <string>
#include <vector>

#include "visionops_runtime/postprocess.hpp"

namespace visionops::runtime {

PostprocessResult postprocess_classification(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config);

std::string make_classification_payload_json();

}  // namespace visionops::runtime
