#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "visionops_runtime/runtime_state.hpp"

namespace visionops::runtime {

std::int64_t timestamp_ms();
std::string json_escape(const std::string& value);

std::string make_health_json(
    const std::string& device_id,
    const std::string& component,
    double uptime_s);

std::string make_runtime_status_json(
    const std::string& device_id,
    const std::string& component,
    const std::string& task_type,
    const RuntimeSnapshot& snapshot);

std::string make_inference_result_json(
    const std::string& device_id,
    const std::string& component,
    const std::string& task_type,
    const InferenceIdentity& identity);

std::string make_error_json(
    const std::string& device_id,
    const std::string& component,
    const std::string& code,
    const std::string& message,
    bool recoverable);

const std::vector<std::uint8_t>& placeholder_jpeg();

}  // namespace visionops::runtime
