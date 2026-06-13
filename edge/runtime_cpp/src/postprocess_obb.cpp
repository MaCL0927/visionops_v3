#include "visionops_runtime/postprocess_obb.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "visionops_runtime/json_utils.hpp"

namespace visionops::runtime {

namespace {

struct ObbItem {
  int class_id{0};
  std::string class_name;
  float score{0.0F};
  float cx{0.0F};
  float cy{0.0F};
  float width{0.0F};
  float height{0.0F};
  float angle{0.0F};
  float points[8]{};
  float x1{0.0F};
  float y1{0.0F};
  float x2{0.0F};
  float y2{0.0F};
};

float sigmoid(float value) { return 1.0F / (1.0F + std::exp(-value)); }
float clip(float value, float maximum) { return std::clamp(value, 0.0F, maximum); }

void finalize(ObbItem& item, const LetterboxMeta& meta) {
  item.cx = (item.cx - meta.pad_x) / std::max(meta.scale, 1e-6F);
  item.cy = (item.cy - meta.pad_y) / std::max(meta.scale, 1e-6F);
  item.width /= std::max(meta.scale, 1e-6F);
  item.height /= std::max(meta.scale, 1e-6F);
  if (std::fabs(item.angle) > 2.0F * static_cast<float>(M_PI)) {
    item.angle *= static_cast<float>(M_PI) / 180.0F;
  }
  const float cosine = std::cos(item.angle);
  const float sine = std::sin(item.angle);
  const float half_width = item.width * 0.5F;
  const float half_height = item.height * 0.5F;
  const float local[8] = {
      -half_width, -half_height, half_width, -half_height,
      half_width, half_height, -half_width, half_height};
  item.x1 = static_cast<float>(meta.orig_width);
  item.y1 = static_cast<float>(meta.orig_height);
  item.x2 = 0.0F;
  item.y2 = 0.0F;
  for (int point = 0; point < 4; ++point) {
    const float x = local[point * 2];
    const float y = local[point * 2 + 1];
    item.points[point * 2] = clip(
        x * cosine - y * sine + item.cx,
        static_cast<float>(std::max(0, meta.orig_width - 1)));
    item.points[point * 2 + 1] = clip(
        x * sine + y * cosine + item.cy,
        static_cast<float>(std::max(0, meta.orig_height - 1)));
    item.x1 = std::min(item.x1, item.points[point * 2]);
    item.y1 = std::min(item.y1, item.points[point * 2 + 1]);
    item.x2 = std::max(item.x2, item.points[point * 2]);
    item.y2 = std::max(item.y2, item.points[point * 2 + 1]);
  }
}

float bbox_iou(const ObbItem& a, const ObbItem& b) {
  const float intersection = std::max(0.0F, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
      std::max(0.0F, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  return intersection / (area_a + area_b - intersection + 1e-6F);
}

}  // namespace

PostprocessResult postprocess_obb(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox) {
  PostprocessResult result;
  if (outputs.size() != 1) {
    result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
    result.error_message = "当前 M9.3 OBB 仅支持单输出 YOLOv8 [1,C,N] 或 [1,N,C]";
    return result;
  }
  const int class_count = std::max(1, static_cast<int>(config.class_names.size()));
  const int expected_channels = 5 + class_count;
  const auto& dims = outputs[0].info.dimensions;
  int channels = 0;
  int candidates = 0;
  bool channel_first = true;
  if (dims.size() == 3 && static_cast<int>(dims[1]) >= expected_channels && dims[1] < dims[2]) {
    channels = dims[1]; candidates = dims[2];
  } else if (dims.size() == 3 && static_cast<int>(dims[2]) >= expected_channels) {
    candidates = dims[1]; channels = dims[2]; channel_first = false;
  } else if (dims.size() == 2 && static_cast<int>(dims[0]) >= expected_channels && dims[0] < dims[1]) {
    channels = dims[0]; candidates = dims[1];
  } else if (dims.size() == 2 && static_cast<int>(dims[1]) >= expected_channels) {
    candidates = dims[0]; channels = dims[1]; channel_first = false;
  } else {
    result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
    result.error_message = "unsupported_output_shape: OBB tensor";
    return result;
  }
  const auto values = tensor_float_data(outputs[0]);
  if (values.size() < static_cast<std::size_t>(channels) * candidates) {
    result.error_code = "INVALID_OUTPUT_DATA";
    result.error_message = "OBB tensor 数据长度不足";
    return result;
  }
  const auto at = [&](int candidate, int channel) {
    return channel_first ? values[channel * candidates + candidate]
                         : values[candidate * channels + channel];
  };
  std::vector<ObbItem> decoded;
  for (int index = 0; index < candidates; ++index) {
    int best_class = 0;
    float score = at(index, 4);
    for (int class_id = 1; class_id < class_count; ++class_id) {
      if (at(index, 4 + class_id) > score) {
        score = at(index, 4 + class_id);
        best_class = class_id;
      }
    }
    if (score < 0.0F || score > 1.0F) score = sigmoid(score);
    if (score < config.score_threshold) continue;
    ObbItem item;
    item.class_id = best_class;
    item.class_name = best_class < static_cast<int>(config.class_names.size())
        ? config.class_names[best_class] : std::to_string(best_class);
    item.score = std::clamp(score, 0.0F, 1.0F);
    item.cx = at(index, 0);
    item.cy = at(index, 1);
    item.width = at(index, 2);
    item.height = at(index, 3);
    item.angle = at(index, 4 + class_count);
    finalize(item, letterbox);
    if (item.x2 > item.x1 && item.y2 > item.y1) decoded.push_back(item);
  }
  std::sort(decoded.begin(), decoded.end(), [](const auto& a, const auto& b) {
    return a.score > b.score;
  });
  std::vector<ObbItem> kept;
  for (const auto& item : decoded) {
    bool suppressed = false;
    for (const auto& existing : kept) {
      if (item.class_id == existing.class_id && bbox_iou(item, existing) > config.nms_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push_back(item);
    if (static_cast<int>(kept.size()) >= config.max_detections) break;
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << ",\"detections\":[";
  for (std::size_t index = 0; index < kept.size(); ++index) {
    if (index != 0) stream << ',';
    const auto& item = kept[index];
    stream << "{\"id\":\"obb-rknn-" << index + 1 << "\",\"class_id\":" << item.class_id
           << ",\"class_name\":\"" << json_escape(item.class_name) << "\",\"score\":" << item.score
           << ",\"bbox_xyxy\":[" << item.x1 << ',' << item.y1 << ',' << item.x2 << ',' << item.y2 << ']'
           << ",\"center_xy\":[" << item.cx << ',' << item.cy << ']'
           << ",\"obb\":{\"cx\":" << item.cx << ",\"cy\":" << item.cy
           << ",\"w\":" << item.width << ",\"h\":" << item.height
           << ",\"angle_deg\":" << item.angle * 180.0F / static_cast<float>(M_PI)
           << ",\"points\":[";
    for (int point = 0; point < 4; ++point) {
      if (point != 0) stream << ',';
      stream << '[' << item.points[point * 2] << ',' << item.points[point * 2 + 1] << ']';
    }
    stream << "]}}";
  }
  stream << ']';
  result.success = true;
  result.result_count = static_cast<int>(kept.size());
  result.payload_json = stream.str();
  result.warning = "OBB 当前使用外接矩形 NMS";
  return result;
}

std::string make_obb_payload_json() {
  return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"rotated_object","score":0.91,"bbox_xyxy":[420.0,220.0,900.0,720.0],"center_xy":[660.0,470.0],"obb":{"cx":660.0,"cy":470.0,"w":430.0,"h":220.0,"angle_deg":-12.0,"points":[[427.0,406.0],[847.0,316.0],[893.0,534.0],[473.0,624.0]]}}],"final_decision":{"code":"ORIENTATION_OK","label":"aligned","ok":true,"reason":"Mock OBB 结果"})json";
}

}  // namespace visionops::runtime
