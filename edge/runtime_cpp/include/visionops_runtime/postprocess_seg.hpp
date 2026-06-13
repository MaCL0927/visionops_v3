#pragma once

#include <string>
#include <vector>

#include "visionops_runtime/postprocess.hpp"

namespace visionops::runtime {

PostprocessResult postprocess_segmentation(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox);
std::string make_segmentation_payload_json();

}  // namespace visionops::runtime
