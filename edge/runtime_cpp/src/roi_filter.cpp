#include "visionops_runtime/roi_filter.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <utility>

#include "visionops_runtime/json_utils.hpp"

namespace visionops::runtime {
namespace {

std::uint64_t now_ms() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}

std::optional<std::size_t> value_position(const std::string& text, const std::string& key) {
  const std::string marker = '"' + key + '"';
  const auto key_position = text.find(marker);
  if (key_position == std::string::npos) return std::nullopt;
  const auto colon = text.find(':', key_position + marker.size());
  if (colon == std::string::npos) return std::nullopt;
  auto position = colon + 1;
  while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) ++position;
  return position;
}

std::optional<bool> bool_field(const std::string& text, const std::string& key) {
  const auto position = value_position(text, key);
  if (!position) return std::nullopt;
  if (text.compare(*position, 4, "true") == 0) return true;
  if (text.compare(*position, 5, "false") == 0) return false;
  return std::nullopt;
}

std::optional<double> number_field(const std::string& text, const std::string& key) {
  const auto position = value_position(text, key);
  if (!position || *position >= text.size()) return std::nullopt;
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(text.c_str() + *position, &end);
  if (end == text.c_str() + *position || errno == ERANGE || !std::isfinite(value)) return std::nullopt;
  return value;
}

bool valid_roi(const RoiFilterConfig& roi, std::string& error_message) {
  if (roi.mode != "center") {
    error_message = "ROI mode 当前仅支持 center";
    return false;
  }
  if (!(roi.x1 >= 0.0 && roi.x1 < roi.x2 && roi.x2 <= 1.0 &&
        roi.y1 >= 0.0 && roi.y1 < roi.y2 && roi.y2 <= 1.0)) {
    error_message = "ROI 归一化坐标必须满足 0<=x1<x2<=1 且 0<=y1<y2<=1";
    return false;
  }
  if ((roi.x2 - roi.x1) < 0.005 || (roi.y2 - roi.y1) < 0.005) {
    error_message = "ROI 区域过小";
    return false;
  }
  return true;
}

std::string roi_object_json(const RoiFilterConfig& roi, const std::string& path, int width, int height) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(6)
         << "{\"enabled\":" << json_bool(roi.enabled)
         << ",\"mode\":\"" << json_escape(roi.mode) << '"'
         << ",\"normalized_xyxy\":[" << roi.x1 << ',' << roi.y1 << ',' << roi.x2 << ',' << roi.y2 << ']'
         << ",\"x1\":" << roi.x1 << ",\"y1\":" << roi.y1
         << ",\"x2\":" << roi.x2 << ",\"y2\":" << roi.y2
         << ",\"updated_at_ms\":" << roi.updated_at_ms
         << ",\"config_path\":" << (path.empty() ? "null" : '"' + json_escape(path) + '"');
  if (width > 0 && height > 0) {
    stream << ",\"pixel_xyxy\":["
           << roi.x1 * width << ',' << roi.y1 * height << ','
           << roi.x2 * width << ',' << roi.y2 * height << ']';
  } else {
    stream << ",\"pixel_xyxy\":null";
  }
  stream << '}';
  return stream.str();
}

}  // namespace

bool roi_contains_center(
    const RoiFilterConfig& roi,
    double center_x,
    double center_y,
    int image_width,
    int image_height) {
  if (!roi.enabled) return true;
  if (image_width <= 0 || image_height <= 0) return false;
  const double nx = center_x / static_cast<double>(image_width);
  const double ny = center_y / static_cast<double>(image_height);
  return nx >= roi.x1 && nx <= roi.x2 && ny >= roi.y1 && ny <= roi.y2;
}

RoiFilterStore::RoiFilterStore(std::string path) : path_(std::move(path)) { load(); }

RoiFilterConfig RoiFilterStore::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return config_;
}

std::string RoiFilterStore::json(int image_width, int image_height) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return "{\"schema_version\":\"1.0\",\"message_type\":\"runtime_roi_config\",\"status\":\"ok\",\"roi\":" +
      roi_object_json(config_, path_, image_width, image_height) + "}";
}

std::string RoiFilterStore::value_json(int image_width, int image_height) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return roi_object_json(config_, path_, image_width, image_height);
}

bool RoiFilterStore::update_from_json(const std::string& body, std::string& error_message) {
  RoiFilterConfig next = snapshot();
  const auto enabled = bool_field(body, "enabled");
  if (!enabled) {
    error_message = "请求体必须包含布尔字段 enabled";
    return false;
  }
  next.enabled = *enabled;
  if (const auto x1 = number_field(body, "x1")) next.x1 = *x1;
  if (const auto y1 = number_field(body, "y1")) next.y1 = *y1;
  if (const auto x2 = number_field(body, "x2")) next.x2 = *x2;
  if (const auto y2 = number_field(body, "y2")) next.y2 = *y2;
  if (!valid_roi(next, error_message)) return false;
  next.updated_at_ms = now_ms();
  RoiFilterConfig previous;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    previous = config_;
    config_ = next;
  }
  if (!save(error_message)) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = previous;
    return false;
  }
  return true;
}

const std::string& RoiFilterStore::path() const { return path_; }

void RoiFilterStore::load() {
  if (path_.empty()) return;
  std::ifstream input(path_);
  if (!input) return;
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const auto text = buffer.str();
  RoiFilterConfig loaded;
  if (const auto value = bool_field(text, "enabled")) loaded.enabled = *value;
  if (const auto value = number_field(text, "x1")) loaded.x1 = *value;
  if (const auto value = number_field(text, "y1")) loaded.y1 = *value;
  if (const auto value = number_field(text, "x2")) loaded.x2 = *value;
  if (const auto value = number_field(text, "y2")) loaded.y2 = *value;
  if (const auto value = number_field(text, "updated_at_ms")) {
    loaded.updated_at_ms = static_cast<std::uint64_t>(std::max(0.0, *value));
  }
  std::string error;
  if (!valid_roi(loaded, error)) return;
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = loaded;
}

bool RoiFilterStore::save(std::string& error_message) const {
  if (path_.empty()) return true;
  RoiFilterConfig value;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    value = config_;
  }
  std::error_code error;
  const std::filesystem::path target(path_);
  if (!target.parent_path().empty()) {
    std::filesystem::create_directories(target.parent_path(), error);
    if (error) {
      error_message = "创建 ROI 配置目录失败: " + error.message();
      return false;
    }
  }
  const auto temporary = target.string() + ".tmp";
  {
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) {
      error_message = "写入 ROI 临时配置失败: " + temporary;
      return false;
    }
    output << roi_object_json(value, path_, 0, 0) << '\n';
  }
  std::filesystem::rename(temporary, target, error);
  if (error) {
    std::filesystem::remove(target, error);
    error.clear();
    std::filesystem::rename(temporary, target, error);
  }
  if (error) {
    error_message = "保存 ROI 配置失败: " + error.message();
    std::filesystem::remove(temporary, error);
    return false;
  }
  return true;
}

}  // namespace visionops::runtime
