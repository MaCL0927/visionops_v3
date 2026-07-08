#include "visionops_runtime/postprocess_classification.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "visionops_runtime/json_utils.hpp"

namespace visionops::runtime {

namespace {

struct ClassificationItem {
  int class_id{0};
  std::string class_name;
  float score{0.0F};
  int rank{0};
};

std::string decision_code_from_label(std::string label) {
  if (label.empty()) return "CLASSIFIED";
  for (char& ch : label) {
    const auto uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch)) {
      ch = static_cast<char>(std::toupper(uch));
    } else {
      ch = '_';
    }
  }
  while (label.find("__") != std::string::npos) {
    label.replace(label.find("__"), 2, "_");
  }
  while (!label.empty() && label.front() == '_') label.erase(label.begin());
  while (!label.empty() && label.back() == '_') label.pop_back();
  return label.empty() ? "CLASSIFIED" : label;
}

bool looks_like_probability_vector(const std::vector<float>& values) {
  if (values.empty()) return false;
  float sum = 0.0F;
  for (const float value : values) {
    if (!std::isfinite(value) || value < -1e-4F || value > 1.0001F) {
      return false;
    }
    sum += value;
  }
  return sum >= 0.8F && sum <= 1.2F;
}

std::vector<float> softmax(const std::vector<float>& logits) {
  std::vector<float> probs(logits.size(), 0.0F);
  if (logits.empty()) return probs;
  const float max_value = *std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  for (std::size_t index = 0; index < logits.size(); ++index) {
    const double value = std::exp(static_cast<double>(logits[index] - max_value));
    probs[index] = static_cast<float>(value);
    sum += value;
  }
  if (sum <= 0.0 || !std::isfinite(sum)) {
    return std::vector<float>(logits.size(), 1.0F / std::max<std::size_t>(1, logits.size()));
  }
  for (float& value : probs) {
    value = static_cast<float>(value / sum);
  }
  return probs;
}

std::vector<float> classification_scores(const RuntimeTensor& tensor) {
  auto values = tensor_float_data(tensor);
  if (values.empty()) return values;

  // YOLOv8-cls 的 RKNN 输出常见形状为 [1,nc] / [nc] / [1,nc,1,1]。
  // rknn_outputs_get(want_float=1) 已经返回 float，本函数只负责拉平单 batch 输出。
  const auto& dims = tensor.info.dimensions;
  if (dims.size() >= 2 && dims[0] > 1) {
    // 多 batch 理论上不会用于边缘端实时推理；仅保留第一个 batch，避免误把 batch 维当类别维。
    std::size_t per_batch = values.size() / std::max<std::uint32_t>(1, dims[0]);
    if (per_batch > 0 && per_batch < values.size()) {
      values.resize(per_batch);
    }
  }
  return values;
}

std::string classifications_json(const std::vector<ClassificationItem>& items) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(6) << ",\"classifications\":[";
  for (std::size_t index = 0; index < items.size(); ++index) {
    if (index != 0) stream << ',';
    const auto& item = items[index];
    stream << "{\"class_id\":" << item.class_id
           << ",\"class_name\":\"" << json_escape(item.class_name) << '"'
           << ",\"label\":\"" << json_escape(item.class_name) << '"'
           << ",\"score\":" << std::clamp(item.score, 0.0F, 1.0F)
           << ",\"rank\":" << item.rank << '}';
  }
  stream << ']';

  if (items.empty()) {
    stream << ",\"final_decision\":{\"code\":\"NO_CLASS\",\"label\":\"no_class\",\"ok\":false,\"reason\":\"未得到分类结果\"}";
    return stream.str();
  }

  const auto& top = items.front();
  stream << ",\"final_decision\":{\"code\":\"" << json_escape(decision_code_from_label(top.class_name))
         << "\",\"label\":\"" << json_escape(top.class_name)
         << "\",\"ok\":true,\"reason\":\"分类结果: " << json_escape(top.class_name)
         << "\",\"score\":" << std::clamp(top.score, 0.0F, 1.0F) << '}';
  return stream.str();
}

}  // namespace

PostprocessResult postprocess_classification(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config) {
  PostprocessResult result;
  if (outputs.empty()) {
    result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
    result.error_message = "classification 没有输出 tensor";
    return result;
  }
  if (outputs.size() != 1) {
    result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
    result.error_message = "当前 classification 支持 YOLOv8-cls 单输出 [1,nc] / [nc] / [1,nc,1,1]";
    return result;
  }

  auto values = classification_scores(outputs.front());
  if (values.empty()) {
    result.error_code = "INVALID_OUTPUT_DATA";
    result.error_message = "classification 输出数据为空或不是 float";
    return result;
  }

  const int class_count = config.class_names.empty()
      ? static_cast<int>(values.size())
      : static_cast<int>(config.class_names.size());
  if (class_count <= 0) {
    result.error_code = "INVALID_CLASS_COUNT";
    result.error_message = "classification 类别数非法";
    return result;
  }
  if (static_cast<int>(values.size()) < class_count) {
    result.error_code = "INVALID_OUTPUT_DATA";
    result.error_message = "classification 输出长度小于类别数: output=" +
        std::to_string(values.size()) + ", classes=" + std::to_string(class_count);
    return result;
  }
  if (static_cast<int>(values.size()) > class_count) {
    values.resize(class_count);
  }

  std::vector<float> scores = looks_like_probability_vector(values) ? values : softmax(values);
  std::vector<int> order(class_count);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int left, int right) {
    return scores[left] > scores[right];
  });

  const int max_items = std::max(1, std::min({class_count, config.max_detections > 0 ? config.max_detections : class_count, 5}));
  std::vector<ClassificationItem> items;
  items.reserve(max_items);
  for (int rank_index = 0; rank_index < max_items; ++rank_index) {
    const int class_id = order[rank_index];
    const float score = scores[class_id];
    // 分类任务始终保留 Top-1；后续类别按阈值过滤，避免页面输出过多低置信度类别。
    if (rank_index > 0 && score < config.score_threshold) continue;
    ClassificationItem item;
    item.class_id = class_id;
    item.class_name = class_id < static_cast<int>(config.class_names.size())
        ? config.class_names[class_id]
        : std::to_string(class_id);
    item.score = score;
    item.rank = rank_index + 1;
    items.push_back(std::move(item));
  }

  result.success = true;
  result.result_count = static_cast<int>(items.size());
  result.payload_json = classifications_json(items);
  return result;
}

std::string make_classification_payload_json() {
  return R"json(,"classifications":[{"class_id":0,"class_name":"ok","label":"ok","score":0.920000,"rank":1}],"final_decision":{"code":"OK","label":"ok","ok":true,"reason":"Mock 分类结果","score":0.920000})json";
}

}  // namespace visionops::runtime
