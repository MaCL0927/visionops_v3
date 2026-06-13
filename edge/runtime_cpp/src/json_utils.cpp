#include "visionops_runtime/json_utils.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>

namespace visionops::runtime {

std::int64_t now_timestamp_ms() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

std::string json_escape(const std::string& value) {
  std::ostringstream stream;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"': stream << "\\\""; break;
      case '\\': stream << "\\\\"; break;
      case '\b': stream << "\\b"; break;
      case '\f': stream << "\\f"; break;
      case '\n': stream << "\\n"; break;
      case '\r': stream << "\\r"; break;
      case '\t': stream << "\\t"; break;
      default:
        if (ch < 0x20) {
          stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(ch) << std::dec;
        } else {
          stream << static_cast<char>(ch);
        }
    }
  }
  return stream.str();
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

std::string make_trace_id(std::int64_t timestamp_ms) {
  return "trace-mock-" + std::to_string(timestamp_ms);
}

std::string make_error_json(
    const std::string& device_id,
    const std::string& component,
    const std::string& code,
    const std::string& message,
    bool recoverable) {
  return "{\"schema_version\":\"1.0\",\"message_type\":\"runtime_error\",\"device_id\":\"" +
      json_escape(device_id) + "\",\"component\":\"" + json_escape(component) +
      "\",\"timestamp_ms\":" + std::to_string(now_timestamp_ms()) +
      ",\"trace_id\":\"trace-error-mock\",\"source\":\"http_api\",\"status\":\"error\",\"error\":{\"code\":\"" +
      json_escape(code) + "\",\"message\":\"" + json_escape(message) +
      "\",\"detail\":null,\"recoverable\":" + json_bool(recoverable) + "}}";
}

}  // namespace visionops::runtime
