#pragma once

#include <string>
#include <vector>

#include "visionops_runtime/postprocess.hpp"

namespace visionops::runtime {

PostprocessResult postprocess_obb(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox);
std::string make_obb_payload_json();

}  // namespace visionops::runtime
