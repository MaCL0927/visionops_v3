#pragma once

#include <string>

namespace visionops::runtime {

std::string make_detection_payload_json();
std::string make_classification_payload_json();
std::string make_roi_classification_payload_json();

}  // namespace visionops::runtime
