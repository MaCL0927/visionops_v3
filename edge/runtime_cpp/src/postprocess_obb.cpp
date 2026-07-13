#include "visionops_runtime/postprocess_obb.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
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
  float angle{0.0F};  // radians
  float points[8]{};
  float x1{0.0F};
  float y1{0.0F};
  float x2{0.0F};
  float y2{0.0F};
};

float sigmoid(float value) { return 1.0F / (1.0F + std::exp(-value)); }

float logit_threshold(float probability) {
  probability = std::clamp(probability, 1e-6F, 1.0F - 1e-6F);
  return std::log(probability / (1.0F - probability));
}

float clip(float value, float maximum) { return std::clamp(value, 0.0F, maximum); }

bool valid_tensor_data(const RuntimeTensor& tensor, std::size_t minimum_float_count) {
  return tensor.data.size() >= minimum_float_count * sizeof(float);
}

std::string shape_string(const RuntimeTensor& tensor) {
  std::ostringstream stream;
  stream << '[';
  for (std::size_t index = 0; index < tensor.info.dimensions.size(); ++index) {
    if (index != 0) stream << ',';
    stream << tensor.info.dimensions[index];
  }
  stream << ']';
  return stream.str();
}

std::string output_shapes_message(const std::vector<RuntimeTensor>& outputs) {
  std::ostringstream stream;
  stream << "unsupported OBB output shapes: ";
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (index != 0) stream << "; ";
    stream << "output[" << index << "] dims=" << shape_string(outputs[index])
           << " bytes=" << outputs[index].data.size();
  }
  return stream.str();
}

bool tensor_need_sigmoid(const std::vector<float>& values) {
  if (values.empty()) return false;
  auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
  return *min_it < 0.0F || *max_it > 1.0F;
}

bool tensor_need_sigmoid_sampled(
    const std::vector<float>& values,
    int base_channel,
    int channel_count,
    int spatial_size) {
  if (values.empty() || channel_count <= 0 || spatial_size <= 0) return true;
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
  const int step = std::max(1, spatial_size / 1024);
  for (int channel = 0; channel < channel_count; ++channel) {
    const int base = (base_channel + channel) * spatial_size;
    if (base < 0 || base >= static_cast<int>(values.size())) continue;
    for (int index = 0; index < spatial_size; index += step) {
      const int offset = base + index;
      if (offset < 0 || offset >= static_cast<int>(values.size())) continue;
      const float value = values[offset];
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
    }
  }
  if (!std::isfinite(minimum) || !std::isfinite(maximum)) return true;
  return minimum < 0.0F || maximum > 1.0F;
}

float dfl_expectation(
    const std::vector<float>& logits,
    int side,
    int spatial_size,
    int index) {
  float maximum = -std::numeric_limits<float>::infinity();
  float values[16]{};
  for (int bin = 0; bin < 16; ++bin) {
    const int channel = side * 16 + bin;
    const int offset = channel * spatial_size + index;
    const float value = offset >= 0 && offset < static_cast<int>(logits.size()) ? logits[offset] : 0.0F;
    values[bin] = value;
    maximum = std::max(maximum, value);
  }
  float sum = 0.0F;
  float weighted = 0.0F;
  for (int bin = 0; bin < 16; ++bin) {
    const float exp_value = std::exp(values[bin] - maximum);
    sum += exp_value;
    weighted += exp_value * static_cast<float>(bin);
  }
  return weighted / std::max(sum, 1e-12F);
}

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

bool finalize_valid(ObbItem& item, const LetterboxMeta& meta) {
  if (item.width <= 2.0F || item.height <= 2.0F) return false;
  finalize(item, meta);
  return item.width > 2.0F && item.height > 2.0F &&
      item.x2 > item.x1 + 2.0F && item.y2 > item.y1 + 2.0F;
}

float bbox_iou(const ObbItem& a, const ObbItem& b) {
  const float intersection = std::max(0.0F, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
      std::max(0.0F, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
  const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
  return intersection / (area_a + area_b - intersection + 1e-6F);
}

std::vector<ObbItem> apply_obb_nms(
    std::vector<ObbItem> decoded,
    float threshold,
    int max_detections) {
  std::sort(decoded.begin(), decoded.end(), [](const auto& left, const auto& right) {
    return left.score > right.score;
  });
  std::vector<ObbItem> kept;
  for (const auto& item : decoded) {
    bool suppressed = false;
    for (const auto& existing : kept) {
      if (item.class_id == existing.class_id && bbox_iou(item, existing) > threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push_back(item);
    if (static_cast<int>(kept.size()) >= max_detections) break;
  }
  return kept;
}

int rockchip_obb_head_channels(const RuntimeTensor& tensor, int class_count) {
  const auto& dims = tensor.info.dimensions;
  if (dims.size() != 4 || dims[0] != 1 || dims[2] == 0 || dims[3] == 0) return 0;
  const int channels = static_cast<int>(dims[1]);
  // Rockchip/Ultralytics YOLOv8-OBB split-DFL head is normally 64 + nc.
  // Some exported RKNN models keep one auxiliary channel in the head, giving
  // 64 + nc + 1, while angle is still emitted as a separate [1,1,N] tensor.
  // Do not tie recognition to a fixed input size or to exactly 64+nc.
  return channels >= 64 + class_count ? channels : 0;
}

long long spatial_count_after_batch_channel(const RuntimeTensor& tensor) {
  const auto& dims = tensor.info.dimensions;
  if (dims.size() < 3) return 0;
  long long count = 1;
  for (std::size_t index = 2; index < dims.size(); ++index) {
    count *= std::max<std::uint32_t>(1, dims[index]);
  }
  return count;
}

bool is_rockchip_obb_outputs(const std::vector<RuntimeTensor>& outputs, int class_count) {
  if (outputs.size() < 4) return false;
  int head_count = 0;
  long long head_spatial_total = 0;
  long long best_angle_count = 0;
  for (const auto& tensor : outputs) {
    const auto& dims = tensor.info.dimensions;
    const int head_channels = rockchip_obb_head_channels(tensor, class_count);
    if (head_channels > 0) {
      ++head_count;
      head_spatial_total += static_cast<long long>(dims[2]) * static_cast<long long>(dims[3]);
      continue;
    }
    if (dims.size() >= 3 && dims[0] == 1 && dims[1] == 1) {
      best_angle_count = std::max(best_angle_count, spatial_count_after_batch_channel(tensor));
    }
  }
  // Keep this dynamic: 640 -> 8400, 1280 -> 33600, and other input sizes should
  // work as long as the angle tensor covers all head grids.
  return head_count >= 3 && best_angle_count >= head_spatial_total && head_spatial_total > 0;
}

const RuntimeTensor* find_rockchip_obb_angle_output(const std::vector<RuntimeTensor>& outputs) {
  const RuntimeTensor* best = nullptr;
  long long best_count = 0;
  for (const auto& tensor : outputs) {
    const auto& dims = tensor.info.dimensions;
    if (dims.size() >= 3 && dims[0] == 1 && dims[1] == 1) {
      long long count = 1;
      for (std::size_t index = 2; index < dims.size(); ++index) {
        count *= std::max<std::uint32_t>(1, dims[index]);
      }
      if (count > best_count) {
        best = &tensor;
        best_count = count;
      }
    }
  }
  return best;
}

std::vector<ObbItem> decode_rockchip_obb_outputs(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox) {
  const int class_count = std::max(1, static_cast<int>(config.class_names.size()));
  const RuntimeTensor* angle_tensor = find_rockchip_obb_angle_output(outputs);
  if (angle_tensor == nullptr) return {};
  const auto angle_values = tensor_float_data(*angle_tensor);
  if (angle_values.empty()) return {};

  std::vector<const RuntimeTensor*> heads;
  for (const auto& tensor : outputs) {
    if (rockchip_obb_head_channels(tensor, class_count) > 0) {
      heads.push_back(&tensor);
    }
  }
  std::sort(heads.begin(), heads.end(), [](const RuntimeTensor* left, const RuntimeTensor* right) {
    return left->info.dimensions[2] > right->info.dimensions[2];
  });
  if (heads.empty()) return {};

  const bool angle_need_sigmoid = tensor_need_sigmoid(angle_values);
  std::vector<ObbItem> decoded;
  int angle_offset = 0;
  for (const RuntimeTensor* head : heads) {
    const auto& dims = head->info.dimensions;
    const int height = static_cast<int>(dims[2]);
    const int width = static_cast<int>(dims[3]);
    const int spatial_size = height * width;
    const int head_channels = static_cast<int>(dims[1]);
    const int class_channels_available = std::max(0, head_channels - 64);
    const int decode_class_count = std::min(class_count, class_channels_available);
    const auto values = tensor_float_data(*head);
    if (decode_class_count <= 0 ||
        !valid_tensor_data(*head, static_cast<std::size_t>(head_channels) * spatial_size) ||
        values.size() < static_cast<std::size_t>(head_channels) * spatial_size) {
      angle_offset += spatial_size;
      continue;
    }
    const float stride_x = letterbox.input_width / static_cast<float>(width);
    const float stride_y = letterbox.input_height / static_cast<float>(height);
    const float stride = (stride_x + stride_y) * 0.5F;
    const bool class_need_sigmoid = tensor_need_sigmoid_sampled(values, 64, decode_class_count, spatial_size);
    const float raw_threshold = class_need_sigmoid
        ? logit_threshold(config.score_threshold)
        : config.score_threshold;

    for (int y = 0; y < height; ++y) {
      const float anchor_y = static_cast<float>(y) + 0.5F;
      for (int x = 0; x < width; ++x) {
        const int index = y * width + x;
        const int global_index = angle_offset + index;
        if (global_index < 0 || global_index >= static_cast<int>(angle_values.size())) continue;

        int best_class = 0;
        float best_raw_score = values[(64 + 0) * spatial_size + index];
        for (int class_id = 1; class_id < decode_class_count; ++class_id) {
          const float raw_score = values[(64 + class_id) * spatial_size + index];
          if (raw_score > best_raw_score) {
            best_raw_score = raw_score;
            best_class = class_id;
          }
        }
        if (best_raw_score < raw_threshold) continue;
        float score = class_need_sigmoid ? sigmoid(best_raw_score) : best_raw_score;
        if (score < config.score_threshold) continue;
        score = std::clamp(score, 0.0F, 1.0F);

        const float left = dfl_expectation(values, 0, spatial_size, index);
        const float top = dfl_expectation(values, 1, spatial_size, index);
        const float right = dfl_expectation(values, 2, spatial_size, index);
        const float bottom = dfl_expectation(values, 3, spatial_size, index);

        const float angle_raw = angle_values[global_index];
        const float angle_sigmoid = angle_need_sigmoid ? sigmoid(angle_raw) : angle_raw;
        const float angle = (angle_sigmoid - 0.25F) * static_cast<float>(M_PI);

        const float anchor_x = static_cast<float>(x) + 0.5F;
        const float offset_x = (right - left) * 0.5F;
        const float offset_y = (bottom - top) * 0.5F;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float center_x = (offset_x * cosine - offset_y * sine + anchor_x) * stride;
        const float center_y = (offset_x * sine + offset_y * cosine + anchor_y) * stride;
        const float box_width = (left + right) * stride;
        const float box_height = (top + bottom) * stride;

        ObbItem item;
        item.class_id = best_class;
        item.class_name = best_class < static_cast<int>(config.class_names.size())
            ? config.class_names[best_class]
            : std::to_string(best_class);
        item.score = score;
        item.cx = center_x;
        item.cy = center_y;
        item.width = box_width;
        item.height = box_height;
        item.angle = angle;
        if (finalize_valid(item, letterbox)) decoded.push_back(item);
      }
    }
    angle_offset += spatial_size;
  }
  return decoded;
}

bool normalize_single_output_shape(
    const RuntimeTensor& tensor,
    int expected_channels,
    int& channels,
    int& candidates,
    bool& channel_first) {
  const auto& dims = tensor.info.dimensions;
  if (dims.size() == 3) {
    const int first = static_cast<int>(dims[1]);
    const int second = static_cast<int>(dims[2]);
    if (first >= expected_channels && first < second) {
      channels = first; candidates = second; channel_first = true; return true;
    }
    if (second >= expected_channels) {
      candidates = first; channels = second; channel_first = false; return true;
    }
  } else if (dims.size() == 2) {
    const int first = static_cast<int>(dims[0]);
    const int second = static_cast<int>(dims[1]);
    if (first >= expected_channels && first < second) {
      channels = first; candidates = second; channel_first = true; return true;
    }
    if (second >= expected_channels) {
      candidates = first; channels = second; channel_first = false; return true;
    }
  }
  return false;
}

std::vector<ObbItem> decode_single_obb_output(
    const RuntimeTensor& tensor,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox) {
  const int class_count = std::max(1, static_cast<int>(config.class_names.size()));
  const int expected_channels = 4 + class_count + 1;
  int channels = 0;
  int candidates = 0;
  bool channel_first = true;
  if (!normalize_single_output_shape(tensor, expected_channels, channels, candidates, channel_first)) {
    return {};
  }
  const auto values = tensor_float_data(tensor);
  if (values.size() < static_cast<std::size_t>(channels) * candidates) return {};
  const auto at = [&](int candidate, int channel) {
    return channel_first ? values[channel * candidates + candidate]
                         : values[candidate * channels + channel];
  };

  float class_min = std::numeric_limits<float>::infinity();
  float class_max = -std::numeric_limits<float>::infinity();
  const int probe = std::min(candidates, 2048);
  for (int candidate = 0; candidate < probe; ++candidate) {
    for (int class_id = 0; class_id < class_count; ++class_id) {
      const float value = at(candidate, 4 + class_id);
      class_min = std::min(class_min, value);
      class_max = std::max(class_max, value);
    }
  }
  const bool score_need_sigmoid = class_min < 0.0F || class_max > 1.0F;

  float max_box_abs = 0.0F;
  for (int candidate = 0; candidate < probe; ++candidate) {
    for (int channel = 0; channel < 4; ++channel) {
      max_box_abs = std::max(max_box_abs, std::fabs(at(candidate, channel)));
    }
  }
  const bool normalized_box = max_box_abs <= 2.0F;

  std::vector<ObbItem> decoded;
  for (int index = 0; index < candidates; ++index) {
    int best_class = 0;
    float raw_score = at(index, 4);
    for (int class_id = 1; class_id < class_count; ++class_id) {
      const float candidate_score = at(index, 4 + class_id);
      if (candidate_score > raw_score) {
        raw_score = candidate_score;
        best_class = class_id;
      }
    }
    float score = score_need_sigmoid ? sigmoid(raw_score) : raw_score;
    if (score < config.score_threshold) continue;

    ObbItem item;
    item.class_id = best_class;
    item.class_name = best_class < static_cast<int>(config.class_names.size())
        ? config.class_names[best_class]
        : std::to_string(best_class);
    item.score = std::clamp(score, 0.0F, 1.0F);
    item.cx = at(index, 0);
    item.cy = at(index, 1);
    item.width = at(index, 2);
    item.height = at(index, 3);
    if (normalized_box) {
      item.cx *= static_cast<float>(letterbox.input_width);
      item.width *= static_cast<float>(letterbox.input_width);
      item.cy *= static_cast<float>(letterbox.input_height);
      item.height *= static_cast<float>(letterbox.input_height);
    }
    item.angle = at(index, 4 + class_count);
    if (finalize_valid(item, letterbox)) decoded.push_back(item);
  }
  return decoded;
}

std::string detections_json(const std::vector<ObbItem>& detections) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << ",\"detections\":[";
  for (std::size_t index = 0; index < detections.size(); ++index) {
    if (index != 0) stream << ',';
    const auto& item = detections[index];
    stream << "{\"id\":\"obb-rknn-" << std::setw(3) << std::setfill('0') << index + 1
           << "\",\"class_id\":" << item.class_id
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
  if (detections.empty()) {
    stream << ",\"final_decision\":{\"code\":\"NO_TARGET\",\"label\":\"no_target\",\"ok\":true,\"reason\":\"未检测到旋转目标\"}";
  }
  return stream.str();
}

void apply_roi_filter(
    std::vector<ObbItem>& detections,
    const RoiFilterConfig& roi,
    const LetterboxMeta& letterbox) {
  if (!roi.enabled) return;
  detections.erase(
      std::remove_if(
          detections.begin(),
          detections.end(),
          [&](const ObbItem& item) {
            return !roi_contains_center(
                roi, item.cx, item.cy, letterbox.orig_width, letterbox.orig_height);
          }),
      detections.end());
}

}  // namespace

PostprocessResult postprocess_obb(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox) {
  PostprocessResult result;
  if (outputs.empty()) {
    result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
    result.error_message = "empty OBB outputs";
    return result;
  }

  std::vector<ObbItem> decoded;
  if (is_rockchip_obb_outputs(outputs, std::max(1, static_cast<int>(config.class_names.size())))) {
    decoded = decode_rockchip_obb_outputs(outputs, config, letterbox);
    result.warning = "OBB 使用 Rockchip YOLOv8-OBB split DFL 多输出后处理，NMS 当前基于外接矩形";
  } else if (outputs.size() == 1) {
    decoded = decode_single_obb_output(outputs[0], config, letterbox);
    result.warning = "OBB 使用单输出 YOLOv8-OBB 后处理，NMS 当前基于外接矩形";
  } else {
    result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
    result.error_message = output_shapes_message(outputs);
    return result;
  }

  decoded = apply_obb_nms(std::move(decoded), config.nms_threshold, config.max_detections);
  result.raw_result_count = static_cast<int>(decoded.size());
  apply_roi_filter(decoded, config.roi, letterbox);
  result.success = true;
  result.result_count = static_cast<int>(decoded.size());
  result.roi_filtered_count = result.raw_result_count - result.result_count;
  result.payload_json = detections_json(decoded);
  if (decoded.empty() && result.warning.empty()) {
    result.warning = "OBB 后处理完成，但没有通过阈值、NMS 和 ROI 的目标";
  }
  return result;
}

std::string make_obb_payload_json() {
  return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"rotated_object","score":0.91,"bbox_xyxy":[420.0,220.0,900.0,720.0],"center_xy":[660.0,470.0],"obb":{"cx":660.0,"cy":470.0,"w":430.0,"h":220.0,"angle_deg":-12.0,"points":[[427.0,406.0],[847.0,316.0],[893.0,534.0],[473.0,624.0]]}}],"final_decision":{"code":"ORIENTATION_OK","label":"aligned","ok":true,"reason":"Mock OBB 结果"})json";
}

}  // namespace visionops::runtime
