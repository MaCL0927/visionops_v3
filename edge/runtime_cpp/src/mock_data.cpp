#include "visionops_runtime/mock_data.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace visionops::runtime {

namespace {

std::string bool_json(bool value) { return value ? "true" : "false"; }

std::string optional_json(const std::optional<std::string>& value) {
  return value ? '"' + json_escape(*value) + '"' : "null";
}

std::string trace_id(std::int64_t timestamp) {
  return "trace-mock-" + std::to_string(timestamp);
}

std::string task_payload(const std::string& task_type) {
  if (task_type == "classification") {
    return R"json(,"classifications":[{"class_id":0,"class_name":"ok","score":0.92,"rank":1}],"final_decision":{"code":"OK","label":"ok","ok":true,"reason":"Mock 分类结果"})json";
  }
  if (task_type == "obb") {
    return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"rotated_object","score":0.91,"bbox_xyxy":[420.0,220.0,900.0,720.0],"center_xy":[660.0,470.0],"obb":{"cx":660.0,"cy":470.0,"w":430.0,"h":220.0,"angle_deg":-12.0,"points":[[427.0,406.0],[847.0,316.0],[893.0,534.0],[473.0,624.0]]}}],"final_decision":{"code":"ORIENTATION_OK","label":"aligned","ok":true,"reason":"Mock OBB 结果"})json";
  }
  if (task_type == "segmentation") {
    return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"surface_region","score":0.89,"bbox_xyxy":[300.0,220.0,980.0,820.0],"mask":{"encoding":"polygon","size":[1080,1920],"polygon":[[[320.0,250.0],[940.0,230.0],[970.0,790.0],[350.0,810.0]]]}}],"measurements":{"mask_area_px":337900,"coverage_ratio":0.1629})json";
  }
  if (task_type == "roi_classification") {
    return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"part","score":0.96,"bbox_xyxy":[760.0,210.0,1180.0,870.0],"attributes":{"roi_mode":"relative_box"}}],"classifications":[{"class_id":1,"class_name":"ng","score":0.93,"rank":1,"detection_id":"det-mock-001"}],"final_decision":{"code":"NG","label":"ng","ok":false,"reason":"Mock ROI 分类结果"})json";
  }
  return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"object","score":0.94,"bbox_xyxy":[420.5,180.0,860.0,790.5],"center_xy":[640.25,485.25]}],"final_decision":{"code":"OBJECT_FOUND","label":"object","ok":true,"reason":"Mock 检测结果"})json";
}

std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
  static constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<std::uint8_t> output;
  int accumulator = 0;
  int bits = -8;
  for (const char ch : encoded) {
    if (ch == '=') {
      break;
    }
    const auto position = alphabet.find(ch);
    if (position == std::string_view::npos) {
      continue;
    }
    accumulator = (accumulator << 6) + static_cast<int>(position);
    bits += 6;
    if (bits >= 0) {
      output.push_back(static_cast<std::uint8_t>((accumulator >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return output;
}

}  // namespace

std::int64_t timestamp_ms() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

std::string json_escape(const std::string& value) {
  std::ostringstream stream;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"':
        stream << "\\\"";
        break;
      case '\\':
        stream << "\\\\";
        break;
      case '\b':
        stream << "\\b";
        break;
      case '\f':
        stream << "\\f";
        break;
      case '\n':
        stream << "\\n";
        break;
      case '\r':
        stream << "\\r";
        break;
      case '\t':
        stream << "\\t";
        break;
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

std::string make_health_json(
    const std::string& device_id,
    const std::string& component,
    double uptime_s) {
  const auto now = timestamp_ms();
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3)
         << "{\"schema_version\":\"1.0\",\"message_type\":\"runtime_health\""
         << ",\"device_id\":\"" << json_escape(device_id) << '"'
         << ",\"component\":\"" << json_escape(component) << '"'
         << ",\"timestamp_ms\":" << now
         << ",\"trace_id\":\"" << trace_id(now) << '"'
         << ",\"source\":\"runtime:mock\",\"status\":\"ok\""
         << ",\"health\":\"ok\",\"ready\":true,\"version\":\"0.1.0\""
         << ",\"uptime_s\":" << uptime_s << '}';
  return stream.str();
}

std::string make_runtime_status_json(
    const std::string& device_id,
    const std::string& component,
    const std::string& task_type,
    const RuntimeSnapshot& snapshot) {
  const auto now = timestamp_ms();
  const double inference_fps = snapshot.running && snapshot.mode == "detect" ? 1.0 : 0.0;
  const double snapshot_fps = snapshot.running && snapshot.mode == "preview" ? 2.0 : 0.0;
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3)
         << "{\"schema_version\":\"1.0\",\"message_type\":\"runtime_status\""
         << ",\"device_id\":\"" << json_escape(device_id) << '"'
         << ",\"component\":\"" << json_escape(component) << '"'
         << ",\"timestamp_ms\":" << now
         << ",\"trace_id\":\"" << trace_id(now) << '"'
         << ",\"source\":\"runtime:mock\",\"status\":\"ok\""
         << ",\"running\":" << bool_json(snapshot.running)
         << ",\"mode\":\"" << json_escape(snapshot.mode) << '"'
         << ",\"health\":\"" << json_escape(snapshot.health) << '"'
         << ",\"uptime_s\":" << snapshot.uptime_s
         << ",\"loaded_model\":{\"model_id\":\"model-mock-001\",\"model_name\":\"visionops-runtime-mock\",\"model_version\":\"1.0.0\",\"task_type\":\""
         << json_escape(task_type)
         << "\",\"backend\":\"mock\"}"
         << ",\"camera_connected\":true"
         << ",\"fps\":{\"camera_fps\":15.0,\"inference_fps\":" << inference_fps
         << ",\"snapshot_fps\":" << snapshot_fps << '}'
         << ",\"latency_ms\":{\"latest\":16.0,\"average\":16.0,\"p95\":16.0}"
         << ",\"counters\":{\"frames_in\":" << snapshot.counters.frames_in
         << ",\"frames_inferred\":" << snapshot.counters.frames_inferred
         << ",\"frames_dropped\":" << snapshot.counters.frames_dropped
         << ",\"errors\":" << snapshot.counters.errors << '}'
         << ",\"last_result_id\":" << optional_json(snapshot.last_result_id)
         << ",\"last_frame_id\":" << optional_json(snapshot.last_frame_id)
         << ",\"last_error\":null"
         << ",\"resources\":{\"cpu_percent\":0.0,\"memory_mb\":0.0,\"npu_percent\":0.0,\"temperature_c\":0.0}}";
  return stream.str();
}

std::string make_inference_result_json(
    const std::string& device_id,
    const std::string& component,
    const std::string& task_type,
    const InferenceIdentity& identity) {
  const auto now = timestamp_ms();
  std::ostringstream stream;
  stream << "{\"schema_version\":\"1.0\",\"message_type\":\"inference_result\""
         << ",\"device_id\":\"" << json_escape(device_id) << '"'
         << ",\"component\":\"" << json_escape(component) << '"'
         << ",\"timestamp_ms\":" << now
         << ",\"trace_id\":\"" << trace_id(now) << '"'
         << ",\"frame_id\":\"" << json_escape(identity.frame_id) << '"'
         << ",\"source\":\"runtime:mock\",\"status\":\"ok\""
         << ",\"result_id\":\"" << json_escape(identity.result_id) << '"'
         << ",\"task_type\":\"" << json_escape(task_type) << '"'
         << ",\"model\":{\"model_id\":\"model-mock-001\",\"model_name\":\"visionops-runtime-mock\",\"model_version\":\"1.0.0\",\"backend\":\"mock\",\"input_size\":{\"width\":640,\"height\":640}}"
         << ",\"image\":{\"width\":1920,\"height\":1080}"
         << ",\"timing\":{\"preprocess_ms\":2.0,\"inference_ms\":12.0,\"postprocess_ms\":2.0,\"total_ms\":16.0}"
         << task_payload(task_type) << '}';
  return stream.str();
}

std::string make_error_json(
    const std::string& device_id,
    const std::string& component,
    const std::string& code,
    const std::string& message,
    bool recoverable) {
  return "{\"schema_version\":\"1.0\",\"message_type\":\"runtime_error\",\"device_id\":\"" +
      json_escape(device_id) + "\",\"component\":\"" + json_escape(component) +
      "\",\"timestamp_ms\":" + std::to_string(timestamp_ms()) +
      ",\"trace_id\":\"trace-error-mock\",\"source\":\"http_api\",\"status\":\"error\",\"error\":{\"code\":\"" +
      json_escape(code) + "\",\"message\":\"" + json_escape(message) +
      "\",\"detail\":null,\"recoverable\":" + bool_json(recoverable) + "}}";
}

const std::vector<std::uint8_t>& placeholder_jpeg() {
  // 1x1 像素 JPEG，通过 Base64 内嵌，仓库中不保存图片文件。
  static const std::vector<std::uint8_t> image = decode_base64(
      "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
      "2wBDAf//////////////////////////////////////////////////////////////////////////////////////"
      "wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/"
      "9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAA"
      "AAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAA"
      "AAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABAf/8QA"
      "FBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPxB//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPxB//8QA"
      "FBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxB//9k=");
  return image;
}

}  // namespace visionops::runtime
