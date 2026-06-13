#include "visionops_runtime/model_package.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string_view>

#include "visionops_runtime/model_config.hpp"

namespace visionops::runtime {

namespace {

namespace fs = std::filesystem;

struct ManifestData {
  std::string package_id;
  std::string model_name;
  std::string model_version;
  std::string task_type;
  std::string target_platform;
  std::string rknn_file;
  std::string yaml_file;
  std::string labels_file;
  int input_width{0};
  int input_height{0};
  double score_threshold{-1.0};
  double nms_threshold{-1.0};
};

std::string read_text_file(const fs::path& path) {
  std::ifstream input(path);
  if (!input) {
    return {};
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::size_t value_start(const std::string& text, const std::string& key) {
  const std::string marker = '"' + key + '"';
  const auto key_position = text.find(marker);
  if (key_position == std::string::npos) {
    return std::string::npos;
  }
  const auto colon = text.find(':', key_position + marker.size());
  if (colon == std::string::npos) {
    return std::string::npos;
  }
  auto position = colon + 1;
  while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position]))) {
    ++position;
  }
  return position;
}

std::optional<std::string> json_string(const std::string& text, const std::string& key) {
  auto position = value_start(text, key);
  if (position == std::string::npos || position >= text.size() || text[position] != '"') {
    return std::nullopt;
  }
  ++position;
  std::string value;
  bool escaped = false;
  for (; position < text.size(); ++position) {
    const char ch = text[position];
    if (escaped) {
      value.push_back(ch);
      escaped = false;
    } else if (ch == '\\') {
      escaped = true;
    } else if (ch == '"') {
      return value;
    } else {
      value.push_back(ch);
    }
  }
  return std::nullopt;
}

std::optional<double> json_number(const std::string& text, const std::string& key) {
  auto position = value_start(text, key);
  if (position == std::string::npos) {
    return std::nullopt;
  }
  auto end = position;
  while (end < text.size() &&
         (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == '-' ||
          text[end] == '+' || text[end] == '.' || text[end] == 'e' || text[end] == 'E')) {
    ++end;
  }
  if (end == position) {
    return std::nullopt;
  }
  try {
    return std::stod(text.substr(position, end - position));
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<std::string> json_container(
    const std::string& text,
    const std::string& key,
    char open,
    char close) {
  const auto start = value_start(text, key);
  if (start == std::string::npos || start >= text.size() || text[start] != open) {
    return std::nullopt;
  }
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (auto position = start; position < text.size(); ++position) {
    const char ch = text[position];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
    } else if (ch == open) {
      ++depth;
    } else if (ch == close && --depth == 0) {
      return text.substr(start, position - start + 1);
    }
  }
  return std::nullopt;
}

std::optional<std::string> json_object(const std::string& text, const std::string& key) {
  return json_container(text, key, '{', '}');
}

std::optional<std::string> json_array(const std::string& text, const std::string& key) {
  return json_container(text, key, '[', ']');
}

bool parse_size_array(const std::string& array, int& width, int& height) {
  std::string values = array;
  std::replace(values.begin(), values.end(), '[', ' ');
  std::replace(values.begin(), values.end(), ']', ' ');
  std::replace(values.begin(), values.end(), ',', ' ');
  std::istringstream stream(values);
  return (stream >> width >> height) && width > 0 && height > 0;
}

bool parse_manifest(const fs::path& path, ManifestData& manifest, std::string& error) {
  const std::string text = read_text_file(path);
  if (text.empty()) {
    error = "无法读取模型 manifest: " + path.string();
    return false;
  }
  if (text.find('{') == std::string::npos || text.find('}') == std::string::npos) {
    error = "模型 manifest 不是有效 JSON 对象: " + path.string();
    return false;
  }

  manifest.package_id = json_string(text, "package_id").value_or("");
  manifest.model_name = json_string(text, "model_name").value_or("");
  manifest.model_version = json_string(text, "model_version").value_or("");
  manifest.task_type = json_string(text, "task_type").value_or("");
  manifest.target_platform = json_string(text, "target_platform").value_or("");

  if (const auto files = json_object(text, "files")) {
    manifest.rknn_file = json_string(*files, "rknn").value_or("");
    manifest.yaml_file = json_string(*files, "yaml").value_or("");
    manifest.labels_file = json_string(*files, "labels").value_or("");
  }
  if (const auto input = json_object(text, "input")) {
    const auto size = json_array(*input, "size").value_or(
        json_array(*input, "input_size").value_or(""));
    if (!size.empty()) {
      parse_size_array(size, manifest.input_width, manifest.input_height);
    } else {
      manifest.input_width = static_cast<int>(json_number(*input, "width").value_or(0));
      manifest.input_height = static_cast<int>(json_number(*input, "height").value_or(0));
    }
  }
  if (const auto postprocess = json_object(text, "postprocess")) {
    manifest.score_threshold = json_number(*postprocess, "score_threshold").value_or(
        json_number(*postprocess, "confidence_threshold").value_or(-1.0));
    manifest.nms_threshold = json_number(*postprocess, "nms_threshold").value_or(-1.0);
  }
  return true;
}

fs::path resolve_path(const fs::path& base, const std::string& value) {
  if (value.empty()) {
    return {};
  }
  fs::path path(value);
  if (path.is_relative() && !base.empty()) {
    path = base / path;
  }
  return path.lexically_normal();
}

void append_error(std::string& target, const std::string& error) {
  if (error.empty()) {
    return;
  }
  if (!target.empty()) {
    target += "; ";
  }
  target += error;
}

int count_labels(const fs::path& path) {
  std::ifstream input(path);
  if (!input) {
    return 0;
  }
  int count = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (std::any_of(line.begin(), line.end(), [](unsigned char ch) {
          return std::isspace(ch) == 0;
        })) {
      ++count;
    }
  }
  return count;
}

}  // namespace

LoadedModelInfo load_model_package(const AppConfig& app_config) {
  LoadedModelInfo info;
  info.task_type = app_config.mock_task_type;

  const fs::path model_dir = app_config.model_dir.empty()
      ? fs::path{}
      : fs::path(app_config.model_dir).lexically_normal();
  fs::path manifest_path;
  if (!app_config.model_manifest.empty()) {
    manifest_path = resolve_path(model_dir, app_config.model_manifest);
  } else if (!model_dir.empty() && fs::exists(model_dir / "manifest.json")) {
    manifest_path = model_dir / "manifest.json";
  }

  ManifestData manifest;
  if (!manifest_path.empty()) {
    std::string error;
    if (parse_manifest(manifest_path, manifest, error)) {
      const fs::path package_dir = model_dir.empty() ? manifest_path.parent_path() : model_dir;
      if (!manifest.package_id.empty()) info.model_id = manifest.package_id;
      if (!manifest.model_name.empty()) info.model_name = manifest.model_name;
      if (!manifest.model_version.empty()) info.model_version = manifest.model_version;
      if (!manifest.task_type.empty()) info.task_type = manifest.task_type;
      info.target_platform = manifest.target_platform;
      info.rknn_path = resolve_path(package_dir, manifest.rknn_file).string();
      info.labels_path = resolve_path(package_dir, manifest.labels_file).string();
      if (manifest.input_width > 0 && manifest.input_height > 0) {
        info.input_width = manifest.input_width;
        info.input_height = manifest.input_height;
      }
      if (manifest.score_threshold >= 0.0) info.score_threshold = manifest.score_threshold;
      if (manifest.nms_threshold >= 0.0) info.nms_threshold = manifest.nms_threshold;
    } else {
      append_error(info.model_load_error, error);
    }
  } else if (!app_config.model_manifest.empty()) {
    append_error(info.model_load_error, "模型 manifest 路径为空");
  }

  fs::path config_path;
  if (!app_config.model_config.empty()) {
    config_path = resolve_path(model_dir, app_config.model_config);
  } else if (!manifest.yaml_file.empty()) {
    const fs::path package_dir = model_dir.empty() ? manifest_path.parent_path() : model_dir;
    config_path = resolve_path(package_dir, manifest.yaml_file);
  } else if (!model_dir.empty() && fs::exists(model_dir / "model.yaml")) {
    config_path = model_dir / "model.yaml";
  }
  info.config_path = config_path.string();

  ModelConfigData yaml;
  if (!config_path.empty()) {
    std::string error;
    if (load_model_config_yaml(config_path.string(), yaml, error)) {
      if (!yaml.model_name.empty()) info.model_name = yaml.model_name;
      if (!yaml.model_version.empty()) info.model_version = yaml.model_version;
      if (!yaml.task_type.empty()) info.task_type = yaml.task_type;
      if (yaml.input_width > 0 && yaml.input_height > 0) {
        info.input_width = yaml.input_width;
        info.input_height = yaml.input_height;
      }
      if (yaml.score_threshold >= 0.0) info.score_threshold = yaml.score_threshold;
      if (yaml.nms_threshold >= 0.0) info.nms_threshold = yaml.nms_threshold;
      info.labels_count = static_cast<int>(yaml.class_names.size());
    } else {
      append_error(info.model_load_error, error);
    }
  }

  if (!info.labels_path.empty()) {
    const int labels_file_count = count_labels(info.labels_path);
    if (labels_file_count > 0) {
      info.labels_count = labels_file_count;
    }
  }
  if (!is_supported_mock_task_type(info.task_type)) {
    append_error(info.model_load_error, "不支持的模型 task_type: " + info.task_type);
    info.task_type = app_config.mock_task_type;
  }
  return info;
}

}  // namespace visionops::runtime
