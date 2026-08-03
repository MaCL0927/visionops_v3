#include "visionops_runtime/model_config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace visionops::runtime {

namespace {

std::string trim(std::string value) {
  const auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

std::string unquote(std::string value) {
  value = trim(std::move(value));
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

int leading_spaces(const std::string& line) {
  int count = 0;
  while (count < static_cast<int>(line.size()) && line[count] == ' ') ++count;
  return count;
}

std::vector<std::string> parse_list(const std::string& value) {
  const auto open = value.find('[');
  const auto close = value.rfind(']');
  if (open == std::string::npos || close == std::string::npos || close <= open) {
    return {};
  }
  std::vector<std::string> items;
  std::istringstream stream(value.substr(open + 1, close - open - 1));
  std::string item;
  while (std::getline(stream, item, ',')) {
    item = unquote(item);
    if (!item.empty()) items.push_back(std::move(item));
  }
  return items;
}

bool parse_input_size(const std::string& value, int& width, int& height) {
  std::string normalized = trim(value);
  if (normalized.empty()) return true;
  auto items = parse_list(normalized);
  if (items.empty()) {
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream stream(normalized);
    std::string item;
    while (stream >> item) items.push_back(unquote(item));
  }
  try {
    if (items.size() == 1) {
      const int size = std::stoi(items[0]);
      if (size <= 0) return false;
      width = size;
      height = size;
      return true;
    }
    if (items.size() >= 2) {
      width = std::stoi(items[0]);
      height = std::stoi(items[1]);
      return width > 0 && height > 0;
    }
  } catch (const std::exception&) {
    return false;
  }
  return false;
}

bool parse_bool(const std::string& value, bool& result) {
  std::string normalized = unquote(value);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (normalized == "true" || normalized == "yes" || normalized == "1" || normalized == "on") {
    result = true;
    return true;
  }
  if (normalized == "false" || normalized == "no" || normalized == "0" || normalized == "off") {
    result = false;
    return true;
  }
  return false;
}

bool is_input_size_key(const std::string& key) {
  return key == "input_size" || key == "imgsz" || key == "image_size" ||
         key == "input_shape" || key == "model_input_size";
}

bool starts_with_dash_item(const std::string& line) {
  return !line.empty() && line.front() == '-';
}

struct Section {
  int indent{0};
  std::string key;
};

std::vector<std::string> make_path(
    const std::vector<Section>& sections,
    const std::string& leaf) {
  std::vector<std::string> path;
  path.reserve(sections.size() + 1);
  for (const auto& section : sections) path.push_back(section.key);
  path.push_back(leaf);
  return path;
}

bool path_is(const std::vector<std::string>& path, std::initializer_list<const char*> expected) {
  if (path.size() != expected.size()) return false;
  std::size_t index = 0;
  for (const char* item : expected) {
    if (path[index++] != item) return false;
  }
  return true;
}

bool path_starts_with(
    const std::vector<std::string>& path,
    std::initializer_list<const char*> expected) {
  if (path.size() < expected.size()) return false;
  std::size_t index = 0;
  for (const char* item : expected) {
    if (path[index++] != item) return false;
  }
  return true;
}

bool parse_int_list4(const std::string& value, int output[4]) {
  const auto items = parse_list(value);
  if (items.size() != 4) return false;
  try {
    for (int index = 0; index < 4; ++index) output[index] = std::stoi(items[index]);
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

bool parse_double_list4(const std::string& value, double output[4]) {
  const auto items = parse_list(value);
  if (items.size() != 4) return false;
  try {
    for (int index = 0; index < 4; ++index) output[index] = std::stod(items[index]);
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

}  // namespace

bool load_model_config_yaml(
    const std::string& path,
    ModelConfigData& config,
    std::string& error_message) {
  std::ifstream input(path);
  if (!input) {
    error_message = "无法读取模型配置: " + path;
    return false;
  }

  std::string raw_line;
  int line_number = 0;
  bool collecting_class_names = false;
  int class_names_indent = -1;
  bool collecting_input_size = false;
  int input_size_indent = -1;
  std::vector<std::string> pending_input_size;
  enum class PendingRoiList { kNone, kPixelXyxy, kNormalizedXyxy };
  PendingRoiList pending_roi_list = PendingRoiList::kNone;
  int roi_list_indent = -1;
  std::vector<std::string> pending_roi_values;
  std::vector<Section> sections;

  const auto finalize_input_size = [&](int current_line) -> bool {
    if (!collecting_input_size) return true;
    try {
      if (pending_input_size.size() == 1) {
        const int size = std::stoi(pending_input_size[0]);
        if (size <= 0) throw std::invalid_argument("non-positive");
        config.input_width = size;
        config.input_height = size;
      } else if (pending_input_size.size() >= 2) {
        config.input_width = std::stoi(pending_input_size[0]);
        config.input_height = std::stoi(pending_input_size[1]);
        if (config.input_width <= 0 || config.input_height <= 0) {
          throw std::invalid_argument("non-positive");
        }
      }
    } catch (const std::exception&) {
      error_message = "模型配置 input_size 非法，行 " + std::to_string(current_line);
      return false;
    }
    collecting_input_size = false;
    pending_input_size.clear();
    input_size_indent = -1;
    return true;
  };

  const auto finalize_roi_list = [&](int current_line) -> bool {
    if (pending_roi_list == PendingRoiList::kNone) return true;
    if (pending_roi_values.size() != 4) {
      error_message = "模型配置 input_roi 列表必须包含4个元素，行 " +
                      std::to_string(current_line);
      return false;
    }
    try {
      auto& roi = config.input_roi;
      if (pending_roi_list == PendingRoiList::kPixelXyxy) {
        roi.x0 = std::stoi(pending_roi_values[0]);
        roi.y0 = std::stoi(pending_roi_values[1]);
        roi.x1 = std::stoi(pending_roi_values[2]);
        roi.y1 = std::stoi(pending_roi_values[3]);
        roi.has_pixel_xyxy = true;
      } else {
        roi.normalized_x0 = std::stod(pending_roi_values[0]);
        roi.normalized_y0 = std::stod(pending_roi_values[1]);
        roi.normalized_x1 = std::stod(pending_roi_values[2]);
        roi.normalized_y1 = std::stod(pending_roi_values[3]);
        roi.has_normalized_xyxy = true;
      }
    } catch (const std::exception&) {
      error_message = "模型配置 input_roi 列表非法，行 " + std::to_string(current_line);
      return false;
    }
    pending_roi_list = PendingRoiList::kNone;
    pending_roi_values.clear();
    roi_list_indent = -1;
    return true;
  };

  while (std::getline(input, raw_line)) {
    ++line_number;
    const auto comment = raw_line.find('#');
    if (comment != std::string::npos) raw_line.erase(comment);
    if (trim(raw_line).empty()) continue;

    const int indent = leading_spaces(raw_line);
    std::string line = trim(raw_line);

    if (collecting_input_size) {
      if (indent >= input_size_indent && starts_with_dash_item(line)) {
        std::string item = unquote(trim(line.substr(1)));
        if (!item.empty()) pending_input_size.push_back(std::move(item));
        continue;
      }
      if (!finalize_input_size(line_number)) return false;
    }
    if (pending_roi_list != PendingRoiList::kNone) {
      if (indent >= roi_list_indent && starts_with_dash_item(line)) {
        std::string item = unquote(trim(line.substr(1)));
        if (!item.empty()) pending_roi_values.push_back(std::move(item));
        continue;
      }
      if (!finalize_roi_list(line_number)) return false;
    }
    if (collecting_class_names) {
      if (indent >= class_names_indent && starts_with_dash_item(line)) {
        std::string item = unquote(trim(line.substr(1)));
        if (!item.empty()) config.class_names.push_back(std::move(item));
        continue;
      }
      collecting_class_names = false;
      class_names_indent = -1;
    }

    while (!sections.empty() && indent <= sections.back().indent) sections.pop_back();

    const auto separator = line.find(':');
    if (separator == std::string::npos) continue;
    const std::string key = trim(line.substr(0, separator));
    const std::string value = trim(line.substr(separator + 1));
    const auto path_parts = make_path(sections, key);

    try {
      if (key == "model_id" || key == "package_id") {
        config.model_id = unquote(value);
      } else if (key == "model_name" || key == "display_name" ||
                 (key == "name" && path_starts_with(path_parts, {"model"}))) {
        config.model_name = unquote(value);
      } else if (key == "model_version" ||
                 (key == "version" && (sections.empty() || path_starts_with(path_parts, {"model"})))) {
        config.model_version = unquote(value);
      } else if (key == "task_type" || key == "task") {
        config.task_type = unquote(value);
      } else if (key == "target_platform" || key == "platform") {
        config.target_platform = unquote(value);
      } else if (is_input_size_key(key)) {
        if (value.empty()) {
          collecting_input_size = true;
          input_size_indent = indent;
          pending_input_size.clear();
        } else if (!parse_input_size(value, config.input_width, config.input_height)) {
          error_message = "模型配置 input_size 非法，行 " + std::to_string(line_number);
          return false;
        }
      } else if (key == "class_names" || key == "names") {
        auto items = parse_list(value);
        if (!items.empty()) {
          config.class_names = std::move(items);
        } else if (value.empty()) {
          config.class_names.clear();
          collecting_class_names = true;
          class_names_indent = indent;
        }
      } else if (path_is(path_parts, {"runtime", "preprocess"}) ||
                 key == "preprocess_mode") {
        const auto mode = unquote(value);
        if (!mode.empty()) config.runtime_preprocess = mode;
      } else if ((key == "resize_mode") &&
                 !path_starts_with(path_parts, {"preprocess", "input_roi"})) {
        const auto mode = unquote(value);
        if (!mode.empty()) config.runtime_preprocess = mode;
      } else if (key == "score_threshold" || key == "conf_threshold" ||
                 key == "confidence_threshold") {
        config.score_threshold = std::stod(value);
      } else if (key == "nms_threshold" || key == "iou_threshold") {
        config.nms_threshold = std::stod(value);
      } else if (key == "max_detections" || key == "max_results" || key == "max_det") {
        config.max_detections = std::stoi(value);
        if (config.max_detections <= 0) {
          error_message = "模型配置 max_detections 必须为正数，行 " + std::to_string(line_number);
          return false;
        }
      } else if (key == "mask_max_points" || key == "polygon_max_points") {
        config.mask_max_points = std::stoi(value);
        if (config.mask_max_points < 4) {
          error_message = "模型配置 mask_max_points 不得小于4，行 " + std::to_string(line_number);
          return false;
        }
      } else if (key == "mask_decode_mode") {
        config.mask_decode_mode = unquote(value);
        std::transform(
            config.mask_decode_mode.begin(),
            config.mask_decode_mode.end(),
            config.mask_decode_mode.begin(),
            [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (config.mask_decode_mode != "ultralytics_highres" &&
            config.mask_decode_mode != "legacy_proto") {
          error_message = "模型配置 mask_decode_mode 仅支持 ultralytics_highres 或 legacy_proto，行 " +
              std::to_string(line_number);
          return false;
        }
      } else if (key == "mask_threshold") {
        config.mask_threshold = std::stod(value);
        if (!(config.mask_threshold > 0.0 && config.mask_threshold < 1.0)) {
          error_message = "模型配置 mask_threshold 必须位于 (0,1)，行 " + std::to_string(line_number);
          return false;
        }
      } else if (key == "mask_polygon_epsilon_px" || key == "polygon_epsilon_px") {
        config.mask_polygon_epsilon_px = std::stod(value);
        if (config.mask_polygon_epsilon_px < 0.0) {
          error_message = "模型配置 mask_polygon_epsilon_px 不得小于0，行 " +
              std::to_string(line_number);
          return false;
        }
      }

      if (path_starts_with(path_parts, {"preprocess", "input_roi"})) {
        auto& roi = config.input_roi;
        if (path_is(path_parts, {"preprocess", "input_roi", "enabled"})) {
          if (!parse_bool(value, roi.enabled)) throw std::invalid_argument("invalid bool");
        } else if (path_is(path_parts, {"preprocess", "input_roi", "coordinate_space"})) {
          roi.coordinate_space = unquote(value);
        } else if (path_is(path_parts, {"preprocess", "input_roi", "pixel_xyxy"})) {
          if (value.empty()) {
            pending_roi_list = PendingRoiList::kPixelXyxy;
            roi_list_indent = indent;
            pending_roi_values.clear();
          } else {
            int values[4]{};
            if (!parse_int_list4(value, values)) throw std::invalid_argument("invalid pixel_xyxy");
            roi.x0 = values[0]; roi.y0 = values[1]; roi.x1 = values[2]; roi.y1 = values[3];
            roi.has_pixel_xyxy = true;
          }
        } else if (path_is(path_parts, {"preprocess", "input_roi", "normalized_xyxy"})) {
          if (value.empty()) {
            pending_roi_list = PendingRoiList::kNormalizedXyxy;
            roi_list_indent = indent;
            pending_roi_values.clear();
          } else {
            double values[4]{};
            if (!parse_double_list4(value, values)) throw std::invalid_argument("invalid normalized_xyxy");
            roi.normalized_x0 = values[0]; roi.normalized_y0 = values[1];
            roi.normalized_x1 = values[2]; roi.normalized_y1 = values[3];
            roi.has_normalized_xyxy = true;
          }
        } else if (path_is(path_parts, {"preprocess", "input_roi", "resize_mode"})) {
          roi.resize_mode = unquote(value);
        } else if (path_is(path_parts, {"preprocess", "input_roi", "pad_value"})) {
          roi.pad_value = std::stoi(value);
        } else if (path_is(path_parts, {"preprocess", "input_roi", "source_resolution", "width"})) {
          roi.source_width = std::stoi(value);
        } else if (path_is(path_parts, {"preprocess", "input_roi", "source_resolution", "height"})) {
          roi.source_height = std::stoi(value);
        } else if (path_is(path_parts, {"preprocess", "input_roi", "crop_resolution", "width"})) {
          roi.crop_width = std::stoi(value);
        } else if (path_is(path_parts, {"preprocess", "input_roi", "crop_resolution", "height"})) {
          roi.crop_height = std::stoi(value);
        }
      }
    } catch (const std::exception&) {
      error_message = "模型配置字段解析失败，行 " + std::to_string(line_number) + ": " + key;
      return false;
    }

    if (value.empty() && !collecting_input_size && !collecting_class_names &&
        pending_roi_list == PendingRoiList::kNone) {
      sections.push_back({indent, key});
    }
  }

  if (!finalize_input_size(line_number + 1)) return false;
  if (!finalize_roi_list(line_number + 1)) return false;

  std::string roi_error_code;
  std::string roi_error_message;
  if (!validate_input_roi_config(config.input_roi, roi_error_code, roi_error_message)) {
    error_message = roi_error_code + ": " + roi_error_message;
    return false;
  }
  return true;
}

}  // namespace visionops::runtime
