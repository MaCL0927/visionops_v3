#include "visionops_runtime/model_config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
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
    if (!item.empty()) {
      items.push_back(std::move(item));
    }
  }
  return items;
}

bool parse_input_size(const std::string& value, int& width, int& height) {
  std::string normalized = trim(value);
  if (normalized.empty()) {
    return true;
  }

  // 兼容常见写法：input_size: [640, 640] / input_size: [640] / input_size: 640
  auto items = parse_list(normalized);
  if (items.empty()) {
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream stream(normalized);
    std::string item;
    while (stream >> item) {
      items.push_back(unquote(item));
    }
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

bool is_input_size_key(const std::string& key) {
  return key == "input_size" || key == "imgsz" || key == "image_size" ||
         key == "input_shape" || key == "model_input_size";
}

bool starts_with_dash_item(const std::string& line) {
  return !line.empty() && line.front() == '-';
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

  std::string line;
  int line_number = 0;
  bool collecting_class_names = false;
  bool collecting_input_size = false;
  std::vector<std::string> pending_input_size;
  while (std::getline(input, line)) {
    ++line_number;
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
      line.erase(comment);
    }
    line = trim(std::move(line));
    if (line.empty()) {
      continue;
    }

    if (collecting_input_size) {
      if (starts_with_dash_item(line)) {
        std::string item = unquote(trim(line.substr(1)));
        if (!item.empty()) pending_input_size.push_back(std::move(item));
        if (pending_input_size.size() >= 2) {
          try {
            config.input_width = std::stoi(pending_input_size[0]);
            config.input_height = std::stoi(pending_input_size[1]);
          } catch (const std::exception&) {
            error_message = "模型配置 input_size 非法，行 " + std::to_string(line_number);
            return false;
          }
          if (config.input_width <= 0 || config.input_height <= 0) {
            error_message = "模型配置 input_size 非法，行 " + std::to_string(line_number);
            return false;
          }
        }
        continue;
      }
      if (pending_input_size.size() == 1) {
        try {
          const int size = std::stoi(pending_input_size[0]);
          if (size <= 0) {
            error_message = "模型配置 input_size 非法，行 " + std::to_string(line_number);
            return false;
          }
          config.input_width = size;
          config.input_height = size;
        } catch (const std::exception&) {
          error_message = "模型配置 input_size 非法，行 " + std::to_string(line_number);
          return false;
        }
      }
      collecting_input_size = false;
      pending_input_size.clear();
    }

    if (collecting_class_names) {
      if (starts_with_dash_item(line)) {
        std::string item = unquote(trim(line.substr(1)));
        if (!item.empty()) {
          config.class_names.push_back(std::move(item));
        }
        continue;
      }
      collecting_class_names = false;
    }

    const auto separator = line.find(':');
    if (separator == std::string::npos) {
      continue;
    }
    const std::string key = trim(line.substr(0, separator));
    const std::string value = trim(line.substr(separator + 1));
    try {
      if (key == "model_id" || key == "package_id") {
        config.model_id = unquote(value);
      } else if (key == "model_name" || key == "display_name") {
        config.model_name = unquote(value);
      } else if (key == "model_version" || key == "version") {
        config.model_version = unquote(value);
      } else if (key == "task_type" || key == "task") {
        config.task_type = unquote(value);
      } else if (key == "target_platform" || key == "platform") {
        config.target_platform = unquote(value);
      } else if (is_input_size_key(key)) {
        if (value.empty()) {
          collecting_input_size = true;
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
        }
      } else if (key == "score_threshold" || key == "conf_threshold" ||
                 key == "confidence_threshold") {
        config.score_threshold = std::stod(value);
      } else if (key == "nms_threshold" || key == "iou_threshold") {
        config.nms_threshold = std::stod(value);
      }
    } catch (const std::exception&) {
      error_message = "模型配置字段解析失败，行 " + std::to_string(line_number);
      return false;
    }
  }
  if (collecting_input_size) {
    if (pending_input_size.size() == 1) {
      try {
        const int size = std::stoi(pending_input_size[0]);
        if (size <= 0) {
          error_message = "模型配置 input_size 非法";
          return false;
        }
        config.input_width = size;
        config.input_height = size;
      } catch (const std::exception&) {
        error_message = "模型配置 input_size 非法";
        return false;
      }
    } else if (pending_input_size.size() >= 2) {
      try {
        config.input_width = std::stoi(pending_input_size[0]);
        config.input_height = std::stoi(pending_input_size[1]);
      } catch (const std::exception&) {
        error_message = "模型配置 input_size 非法";
        return false;
      }
      if (config.input_width <= 0 || config.input_height <= 0) {
        error_message = "模型配置 input_size 非法";
        return false;
      }
    }
  }
  return true;
}

}  // namespace visionops::runtime
