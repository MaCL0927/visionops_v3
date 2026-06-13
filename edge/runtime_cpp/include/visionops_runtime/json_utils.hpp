#pragma once

#include <cstdint>
#include <string>

namespace visionops::runtime {

std::int64_t now_timestamp_ms();
std::string json_escape(const std::string& value);
std::string json_bool(bool value);
std::string make_trace_id(std::int64_t timestamp_ms);
std::string make_error_json(
    const std::string& device_id,
    const std::string& component,
    const std::string& code,
    const std::string& message,
    bool recoverable);

}  // namespace visionops::runtime
