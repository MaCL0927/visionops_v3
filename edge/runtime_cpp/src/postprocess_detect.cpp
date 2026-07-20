#include "visionops_runtime/postprocess_detect.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

#include "visionops_runtime/json_utils.hpp"

namespace visionops::runtime {

namespace {

struct Detection {
  int class_id{0};
  std::string class_name;
  float score{0.0F};
  float x1{0.0F};
  float y1{0.0F};
  float x2{0.0F};
  float y2{0.0F};
};

float sigmoid(float value) { return 1.0F / (1.0F + std::exp(-value)); }

float map_x(float value, const LetterboxMeta& meta) {
  return std::clamp(
      (value - meta.pad_x) / std::max(meta.scale, 1e-6F),
      0.0F,
      static_cast<float>(std::max(0, meta.orig_width - 1)));
}

float map_y(float value, const LetterboxMeta& meta) {
  return std::clamp(
      (value - meta.pad_y) / std::max(meta.scale, 1e-6F),
      0.0F,
      static_cast<float>(std::max(0, meta.orig_height - 1)));
}

float iou(const Detection& a, const Detection& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
  const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
  const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
  return intersection / (area_a + area_b - intersection + 1e-6F);
}

std::vector<Detection> apply_nms(
    std::vector<Detection> detections,
    float threshold,
    int max_detections) {
  std::sort(detections.begin(), detections.end(), [](const auto& left, const auto& right) {
    return left.score > right.score;
  });
  std::vector<Detection> kept;
  for (const auto& candidate : detections) {
    bool suppressed = false;
    for (const auto& existing : kept) {
      if (candidate.class_id == existing.class_id && iou(candidate, existing) > threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(candidate);
      if (static_cast<int>(kept.size()) >= max_detections) break;
    }
  }
  return kept;
}

bool normalize_shape(
    const RuntimeTensor& tensor,
    int minimum_channels,
    int& channels,
    int& candidates,
    bool& channel_first) {
  const auto& dims = tensor.info.dimensions;
  if (dims.size() == 3) {
    const int first = static_cast<int>(dims[1]);
    const int second = static_cast<int>(dims[2]);
    if (first >= minimum_channels && first < second) {
      channels = first; candidates = second; channel_first = true; return true;
    }
    if (second >= minimum_channels) {
      candidates = first; channels = second; channel_first = false; return true;
    }
  } else if (dims.size() == 2) {
    const int first = static_cast<int>(dims[0]);
    const int second = static_cast<int>(dims[1]);
    if (first >= minimum_channels && first < second) {
      channels = first; candidates = second; channel_first = true; return true;
    }
    if (second >= minimum_channels) {
      candidates = first; channels = second; channel_first = false; return true;
    }
  }
  return false;
}

std::string detections_json(const std::vector<Detection>& detections) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << ",\"detections\":[";
  for (std::size_t index = 0; index < detections.size(); ++index) {
    if (index != 0) stream << ',';
    const auto& detection = detections[index];
    stream << "{\"id\":\"det-rknn-" << std::setw(3) << std::setfill('0') << index + 1
           << "\",\"class_id\":" << detection.class_id
           << ",\"class_name\":\"" << json_escape(detection.class_name) << '"'
           << ",\"score\":" << detection.score
           << ",\"bbox_xyxy\":[" << detection.x1 << ',' << detection.y1 << ','
           << detection.x2 << ',' << detection.y2 << ']'
           << ",\"center_xy\":[" << (detection.x1 + detection.x2) * 0.5F << ','
           << (detection.y1 + detection.y2) * 0.5F << "]}";
  }
  stream << "]";
  if (detections.empty()) {
    stream << ",\"final_decision\":{\"code\":\"NO_TARGET\",\"label\":\"no_target\",\"ok\":true,\"reason\":\"未检测到目标\"}";
  }
  return stream.str();
}

float dfl_expectation(
    const FloatTensorView& data,
    int side,
    int spatial_size,
    int index) {
  float maximum = data[(side * 16) * spatial_size + index];
  for (int bin = 1; bin < 16; ++bin) {
    maximum = std::max(maximum, data[(side * 16 + bin) * spatial_size + index]);
  }
  float sum = 0.0F;
  float weighted = 0.0F;
  for (int bin = 0; bin < 16; ++bin) {
    const float value = std::exp(data[(side * 16 + bin) * spatial_size + index] - maximum);
    sum += value;
    weighted += value * bin;
  }
  return weighted / std::max(sum, 1e-12F);
}

bool decode_split_dfl(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox,
    std::vector<Detection>& decoded) {
  const int class_count = std::max(1, static_cast<int>(config.class_names.size()));
  bool recognized = false;
  for (const auto& box_tensor : outputs) {
    const auto& box_dims = box_tensor.info.dimensions;
    if (box_dims.size() != 4 || box_dims[0] != 1 || box_dims[1] != 64) continue;
    const int height = box_dims[2];
    const int width = box_dims[3];
    const int spatial_size = height * width;
    const RuntimeTensor* class_tensor = nullptr;
    for (const auto& candidate : outputs) {
      const auto& dims = candidate.info.dimensions;
      if (dims.size() == 4 && dims[0] == 1 &&
          static_cast<int>(dims[1]) == class_count &&
          static_cast<int>(dims[2]) == height && static_cast<int>(dims[3]) == width) {
        if (&candidate != &box_tensor) {
          class_tensor = &candidate;
          break;
        }
      }
    }
    if (class_tensor == nullptr) continue;
    recognized = true;
    const auto boxes = tensor_float_data(box_tensor);
    const auto classes = tensor_float_data(*class_tensor);
    if (boxes.size() < static_cast<std::size_t>(64 * spatial_size) ||
        classes.size() < static_cast<std::size_t>(class_count * spatial_size)) {
      continue;
    }
    const bool apply_sigmoid = *std::min_element(classes.begin(), classes.end()) < 0.0F ||
        *std::max_element(classes.begin(), classes.end()) > 1.0F;
    const float stride = 0.5F * (
        letterbox.input_width / static_cast<float>(width) +
        letterbox.input_height / static_cast<float>(height));
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int index = y * width + x;
        int best_class = 0;
        float score = classes[index];
        for (int class_id = 1; class_id < class_count; ++class_id) {
          if (classes[class_id * spatial_size + index] > score) {
            score = classes[class_id * spatial_size + index];
            best_class = class_id;
          }
        }
        if (apply_sigmoid) score = sigmoid(score);
        if (score < config.score_threshold) continue;
        const float left = dfl_expectation(boxes, 0, spatial_size, index);
        const float top = dfl_expectation(boxes, 1, spatial_size, index);
        const float right = dfl_expectation(boxes, 2, spatial_size, index);
        const float bottom = dfl_expectation(boxes, 3, spatial_size, index);
        const float anchor_x = x + 0.5F;
        const float anchor_y = y + 0.5F;
        Detection detection;
        detection.class_id = best_class;
        detection.class_name = best_class < static_cast<int>(config.class_names.size())
            ? config.class_names[best_class] : std::to_string(best_class);
        detection.score = std::clamp(score, 0.0F, 1.0F);
        detection.x1 = map_x((anchor_x - left) * stride, letterbox);
        detection.y1 = map_y((anchor_y - top) * stride, letterbox);
        detection.x2 = map_x((anchor_x + right) * stride, letterbox);
        detection.y2 = map_y((anchor_y + bottom) * stride, letterbox);
        if (detection.x2 > detection.x1 && detection.y2 > detection.y1) decoded.push_back(detection);
      }
    }
  }
  return recognized;
}

void apply_roi_filter(
    std::vector<Detection>& detections,
    const RoiFilterConfig& roi,
    const LetterboxMeta& letterbox) {
  if (!roi.enabled) return;
  detections.erase(
      std::remove_if(
          detections.begin(),
          detections.end(),
          [&](const Detection& item) {
            return !roi_contains_center(
                roi,
                (item.x1 + item.x2) * 0.5F,
                (item.y1 + item.y2) * 0.5F,
                letterbox.orig_width,
                letterbox.orig_height);
          }),
      detections.end());
}

}  // namespace

PostprocessResult postprocess_detection(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox) {
  PostprocessResult result;
  std::vector<Detection> decoded;
  if (outputs.size() != 1) {
    if (decode_split_dfl(outputs, config, letterbox, decoded)) {
      decoded = apply_nms(std::move(decoded), config.nms_threshold, config.max_detections);
      result.raw_result_count = static_cast<int>(decoded.size());
      apply_roi_filter(decoded, config.roi, letterbox);
      result.success = true;
      result.result_count = static_cast<int>(decoded.size());
      result.roi_filtered_count = result.raw_result_count - result.result_count;
      result.payload_json = detections_json(decoded);
      return result;
    }
    result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
    result.error_message = "当前 detection 支持单输出 YOLOv8 或 Rockchip split-DFL 输出";
    return result;
  }
  const int class_count = std::max(1, static_cast<int>(config.class_names.size()));
  int channels = 0;
  int candidates = 0;
  bool channel_first = true;
  if (!normalize_shape(outputs[0], 4 + class_count, channels, candidates, channel_first)) {
    result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
    result.error_message = "unsupported_output_shape: detection tensor";
    return result;
  }
  const auto values = tensor_float_data(outputs[0]);
  if (values.size() < static_cast<std::size_t>(channels) * candidates) {
    result.error_code = "INVALID_OUTPUT_DATA";
    result.error_message = "detection tensor 数据长度不足";
    return result;
  }
  const auto at = [&](int candidate, int channel) {
    return channel_first ? values[channel * candidates + candidate]
                         : values[candidate * channels + channel];
  };
  bool normalized_boxes = true;
  for (int index = 0; index < std::min(candidates, 64); ++index) {
    for (int channel = 0; channel < 4; ++channel) {
      normalized_boxes = normalized_boxes && std::fabs(at(index, channel)) <= 2.0F;
    }
  }

  for (int index = 0; index < candidates; ++index) {
    int best_class = 0;
    float best_score = at(index, 4);
    for (int class_id = 1; class_id < class_count; ++class_id) {
      const float score = at(index, 4 + class_id);
      if (score > best_score) {
        best_score = score;
        best_class = class_id;
      }
    }
    if (best_score < 0.0F || best_score > 1.0F) best_score = sigmoid(best_score);
    if (best_score < config.score_threshold) continue;
    float cx = at(index, 0);
    float cy = at(index, 1);
    float width = at(index, 2);
    float height = at(index, 3);
    if (normalized_boxes) {
      cx *= letterbox.input_width;
      width *= letterbox.input_width;
      cy *= letterbox.input_height;
      height *= letterbox.input_height;
    }
    Detection detection;
    detection.class_id = best_class;
    detection.class_name = best_class < static_cast<int>(config.class_names.size())
        ? config.class_names[best_class]
        : std::to_string(best_class);
    detection.score = std::clamp(best_score, 0.0F, 1.0F);
    detection.x1 = map_x(cx - width * 0.5F, letterbox);
    detection.y1 = map_y(cy - height * 0.5F, letterbox);
    detection.x2 = map_x(cx + width * 0.5F, letterbox);
    detection.y2 = map_y(cy + height * 0.5F, letterbox);
    if (detection.x2 > detection.x1 && detection.y2 > detection.y1) decoded.push_back(detection);
  }
  decoded = apply_nms(std::move(decoded), config.nms_threshold, config.max_detections);
  result.raw_result_count = static_cast<int>(decoded.size());
  apply_roi_filter(decoded, config.roi, letterbox);
  result.success = true;
  result.result_count = static_cast<int>(decoded.size());
  result.roi_filtered_count = result.raw_result_count - result.result_count;
  result.payload_json = detections_json(decoded);
  return result;
}

std::string make_detection_payload_json() {
  return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"object","score":0.94,"bbox_xyxy":[420.5,180.0,860.0,790.5],"center_xy":[640.25,485.25]}],"final_decision":{"code":"OBJECT_FOUND","label":"object","ok":true,"reason":"Mock 检测结果"})json";
}


std::string make_roi_classification_payload_json() {
  return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"part","score":0.96,"bbox_xyxy":[760.0,210.0,1180.0,870.0],"attributes":{"roi_mode":"relative_box"}}],"classifications":[{"class_id":1,"class_name":"ng","score":0.93,"rank":1,"detection_id":"det-mock-001"}],"final_decision":{"code":"NG","label":"ng","ok":false,"reason":"Mock ROI 分类结果"})json";
}

}  // namespace visionops::runtime
