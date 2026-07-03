#include "visionops_runtime/postprocess_seg.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

#include "visionops_runtime/json_utils.hpp"

namespace visionops::runtime {

namespace {

constexpr int kDflRegMax = 16;
constexpr int kDflBoxChannels = 4 * kDflRegMax;

struct SegItem {
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
  stream << "unsupported segmentation output shapes: ";
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (index != 0) stream << "; ";
    stream << "output[" << index << "] dims=" << shape_string(outputs[index])
           << " bytes=" << outputs[index].data.size();
  }
  return stream.str();
}

bool values_need_sigmoid_sampled(
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
  float values[kDflRegMax]{};
  for (int bin = 0; bin < kDflRegMax; ++bin) {
    const int channel = side * kDflRegMax + bin;
    const int offset = channel * spatial_size + index;
    const float value = offset >= 0 && offset < static_cast<int>(logits.size())
        ? logits[offset]
        : 0.0F;
    values[bin] = value;
    maximum = std::max(maximum, value);
  }
  float sum = 0.0F;
  float weighted = 0.0F;
  for (int bin = 0; bin < kDflRegMax; ++bin) {
    const float exp_value = std::exp(values[bin] - maximum);
    sum += exp_value;
    weighted += exp_value * static_cast<float>(bin);
  }
  return weighted / std::max(sum, 1e-12F);
}

float iou(const SegItem& a, const SegItem& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
  const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
  const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
  return intersection / (area_a + area_b - intersection + 1e-6F);
}

std::vector<SegItem> apply_nms(
    std::vector<SegItem> detections,
    float threshold,
    int max_detections) {
  std::sort(detections.begin(), detections.end(), [](const auto& left, const auto& right) {
    return left.score > right.score;
  });
  std::vector<SegItem> kept;
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

bool is_4d_nchw(const RuntimeTensor& tensor) {
  const auto& dims = tensor.info.dimensions;
  return dims.size() == 4 && dims[0] == 1 && dims[1] > 0 && dims[2] > 0 && dims[3] > 0;
}

int tensor_channels(const RuntimeTensor& tensor) {
  return is_4d_nchw(tensor) ? static_cast<int>(tensor.info.dimensions[1]) : 0;
}

int tensor_height(const RuntimeTensor& tensor) {
  return is_4d_nchw(tensor) ? static_cast<int>(tensor.info.dimensions[2]) : 0;
}

int tensor_width(const RuntimeTensor& tensor) {
  return is_4d_nchw(tensor) ? static_cast<int>(tensor.info.dimensions[3]) : 0;
}

int tensor_area(const RuntimeTensor& tensor) {
  return tensor_height(tensor) * tensor_width(tensor);
}

const RuntimeTensor* find_split_proto_tensor(
    const std::vector<RuntimeTensor>& outputs,
    int class_count) {
  const RuntimeTensor* best = nullptr;
  int best_area = 0;
  for (const auto& tensor : outputs) {
    if (!is_4d_nchw(tensor)) continue;
    const int channels = tensor_channels(tensor);
    if (channels == kDflBoxChannels || channels == class_count || channels == 1) continue;
    const int area = tensor_area(tensor);
    // For Rockchip YOLOv8-seg split output, proto is normally [1, mask_dim,
    // input/4, input/4], larger than the per-scale mask coefficient maps.
    if (channels > 0 && area > best_area) {
      best = &tensor;
      best_area = area;
    }
  }
  return best;
}

const RuntimeTensor* find_tensor_by_shape(
    const std::vector<RuntimeTensor>& outputs,
    int channels,
    int height,
    int width,
    const RuntimeTensor* exclude = nullptr) {
  for (const auto& tensor : outputs) {
    if (&tensor == exclude) continue;
    if (!is_4d_nchw(tensor)) continue;
    if (tensor_channels(tensor) == channels &&
        tensor_height(tensor) == height &&
        tensor_width(tensor) == width) {
      return &tensor;
    }
  }
  return nullptr;
}

bool normalize_single_output_shape(
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
      channels = first;
      candidates = second;
      channel_first = true;
      return true;
    }
    if (second >= minimum_channels) {
      candidates = first;
      channels = second;
      channel_first = false;
      return true;
    }
  } else if (dims.size() == 2) {
    const int first = static_cast<int>(dims[0]);
    const int second = static_cast<int>(dims[1]);
    if (first >= minimum_channels && first < second) {
      channels = first;
      candidates = second;
      channel_first = true;
      return true;
    }
    if (second >= minimum_channels) {
      candidates = first;
      channels = second;
      channel_first = false;
      return true;
    }
  }
  return false;
}

std::string segmentation_json(const std::vector<SegItem>& detections, const LetterboxMeta& letterbox) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << ",\"detections\":[";
  for (std::size_t index = 0; index < detections.size(); ++index) {
    if (index != 0) stream << ',';
    const auto& item = detections[index];
    stream << "{\"id\":\"seg-rknn-" << std::setw(3) << std::setfill('0') << index + 1
           << "\",\"class_id\":" << item.class_id
           << ",\"class_name\":\"" << json_escape(item.class_name) << '"'
           << ",\"score\":" << item.score
           << ",\"bbox_xyxy\":[" << item.x1 << ',' << item.y1 << ',' << item.x2 << ',' << item.y2 << ']'
           << ",\"center_xy\":[" << (item.x1 + item.x2) * 0.5F << ','
           << (item.y1 + item.y2) * 0.5F << ']'
           << ",\"mask\":{\"encoding\":\"polygon\",\"size\":[" << letterbox.orig_height
           << ',' << letterbox.orig_width << "],\"polygon\":[[[" << item.x1 << ',' << item.y1 << "],["
           << item.x2 << ',' << item.y1 << "],[" << item.x2 << ',' << item.y2 << "],["
           << item.x1 << ',' << item.y2 << "]]]}}";
  }
  stream << "],\"measurements\":{\"mask_count\":" << detections.size() << '}';
  if (detections.empty()) {
    stream << ",\"final_decision\":{\"code\":\"NO_MASK\",\"label\":\"no_mask\",\"ok\":true,\"reason\":\"未检测到分割目标\"}";
  }
  return stream.str();
}

bool decode_split_dfl_segmentation(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox,
    std::vector<SegItem>& decoded,
    std::vector<std::uint32_t>& proto_shape,
    int& mask_dim) {
  const int class_count = std::max(1, static_cast<int>(config.class_names.size()));
  const RuntimeTensor* proto_tensor = find_split_proto_tensor(outputs, class_count);
  if (proto_tensor == nullptr) return false;
  mask_dim = tensor_channels(*proto_tensor);
  if (mask_dim <= 0) return false;
  proto_shape = proto_tensor->info.dimensions;

  bool recognized = false;
  std::vector<const RuntimeTensor*> box_tensors;
  for (const auto& tensor : outputs) {
    if (!is_4d_nchw(tensor)) continue;
    if (tensor_channels(tensor) == kDflBoxChannels) {
      box_tensors.push_back(&tensor);
    }
  }
  std::sort(box_tensors.begin(), box_tensors.end(), [](const RuntimeTensor* left, const RuntimeTensor* right) {
    return tensor_area(*left) > tensor_area(*right);
  });

  for (const RuntimeTensor* box_tensor : box_tensors) {
    const int height = tensor_height(*box_tensor);
    const int width = tensor_width(*box_tensor);
    const int spatial_size = height * width;
    if (spatial_size <= 0) continue;
    const RuntimeTensor* class_tensor = find_tensor_by_shape(outputs, class_count, height, width, box_tensor);
    const RuntimeTensor* object_tensor = find_tensor_by_shape(outputs, 1, height, width, box_tensor);
    const RuntimeTensor* coeff_tensor = find_tensor_by_shape(outputs, mask_dim, height, width, proto_tensor);
    if (class_tensor == nullptr || coeff_tensor == nullptr) continue;
    recognized = true;

    const auto boxes = tensor_float_data(*box_tensor);
    const auto classes = tensor_float_data(*class_tensor);
    std::vector<float> objectness;
    if (object_tensor != nullptr) objectness = tensor_float_data(*object_tensor);
    if (boxes.size() < static_cast<std::size_t>(kDflBoxChannels * spatial_size) ||
        classes.size() < static_cast<std::size_t>(class_count * spatial_size)) {
      continue;
    }
    if (object_tensor != nullptr && objectness.size() < static_cast<std::size_t>(spatial_size)) {
      continue;
    }
    // Decode and keep the mask coefficient tensor shape-recognition path even
    // though the current Runtime payload uses bbox polygon masks. This ensures
    // the 13-output YOLOv8-seg RKNN layout is genuinely recognized and keeps the
    // proto/mask_dim contract ready for future real mask rasterization.
    if (!valid_tensor_data(*coeff_tensor, static_cast<std::size_t>(mask_dim * spatial_size))) {
      continue;
    }

    const bool class_need_sigmoid = values_need_sigmoid_sampled(classes, 0, class_count, spatial_size);
    const bool object_need_sigmoid = object_tensor != nullptr
        ? values_need_sigmoid_sampled(objectness, 0, 1, spatial_size)
        : false;
    const float stride_x = letterbox.input_width / static_cast<float>(width);
    const float stride_y = letterbox.input_height / static_cast<float>(height);
    const float stride = (stride_x + stride_y) * 0.5F;

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int index = y * width + x;
        int best_class = 0;
        float best_score = classes[index];
        for (int class_id = 1; class_id < class_count; ++class_id) {
          const float score = classes[class_id * spatial_size + index];
          if (score > best_score) {
            best_score = score;
            best_class = class_id;
          }
        }
        if (class_need_sigmoid) best_score = sigmoid(best_score);
        float object_score = 1.0F;
        if (object_tensor != nullptr) {
          object_score = objectness[index];
          if (object_need_sigmoid) object_score = sigmoid(object_score);
        }
        const float final_score = std::clamp(best_score * object_score, 0.0F, 1.0F);
        if (final_score < config.score_threshold) continue;

        const float left = dfl_expectation(boxes, 0, spatial_size, index);
        const float top = dfl_expectation(boxes, 1, spatial_size, index);
        const float right = dfl_expectation(boxes, 2, spatial_size, index);
        const float bottom = dfl_expectation(boxes, 3, spatial_size, index);
        const float anchor_x = static_cast<float>(x) + 0.5F;
        const float anchor_y = static_cast<float>(y) + 0.5F;

        SegItem item;
        item.class_id = best_class;
        item.class_name = best_class < static_cast<int>(config.class_names.size())
            ? config.class_names[best_class]
            : std::to_string(best_class);
        item.score = final_score;
        item.x1 = map_x((anchor_x - left) * stride, letterbox);
        item.y1 = map_y((anchor_y - top) * stride, letterbox);
        item.x2 = map_x((anchor_x + right) * stride, letterbox);
        item.y2 = map_y((anchor_y + bottom) * stride, letterbox);
        if (item.x2 > item.x1 + 1.0F && item.y2 > item.y1 + 1.0F) decoded.push_back(item);
      }
    }
  }
  return recognized;
}

bool decode_fused_segmentation(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox,
    std::vector<SegItem>& decoded,
    std::vector<std::uint32_t>& proto_shape,
    int& mask_dim) {
  if (outputs.size() < 2) return false;
  const int class_count = std::max(1, static_cast<int>(config.class_names.size()));
  const auto& proto_dims = outputs[1].info.dimensions;
  mask_dim = 0;
  if (proto_dims.size() == 4 && proto_dims[0] == 1) {
    mask_dim = static_cast<int>(proto_dims[1]);
  } else if (proto_dims.size() == 3) {
    mask_dim = static_cast<int>(proto_dims[0]);
  }
  if (mask_dim <= 0) return false;
  proto_shape = proto_dims;

  const int minimum_channels = 4 + class_count + mask_dim;
  int channels = 0;
  int candidates = 0;
  bool channel_first = true;
  if (!normalize_single_output_shape(outputs[0], minimum_channels, channels, candidates, channel_first)) {
    return false;
  }
  const auto values = tensor_float_data(outputs[0]);
  if (values.size() < static_cast<std::size_t>(channels) * candidates) {
    return false;
  }
  const auto at = [&](int candidate, int channel) {
    return channel_first ? values[channel * candidates + candidate]
                         : values[candidate * channels + channel];
  };
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
    const float cx = at(index, 0);
    const float cy = at(index, 1);
    const float width = at(index, 2);
    const float height = at(index, 3);
    SegItem item;
    item.class_id = best_class;
    item.class_name = best_class < static_cast<int>(config.class_names.size())
        ? config.class_names[best_class]
        : std::to_string(best_class);
    item.score = std::clamp(score, 0.0F, 1.0F);
    item.x1 = map_x(cx - width * 0.5F, letterbox);
    item.y1 = map_y(cy - height * 0.5F, letterbox);
    item.x2 = map_x(cx + width * 0.5F, letterbox);
    item.y2 = map_y(cy + height * 0.5F, letterbox);
    if (item.x2 > item.x1 + 1.0F && item.y2 > item.y1 + 1.0F) decoded.push_back(item);
  }
  return true;
}

}  // namespace

PostprocessResult postprocess_segmentation(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox) {
  PostprocessResult result;
  std::vector<SegItem> decoded;
  std::vector<std::uint32_t> proto_shape;
  int mask_dim = 0;

  if (decode_split_dfl_segmentation(outputs, config, letterbox, decoded, proto_shape, mask_dim)) {
    decoded = apply_nms(std::move(decoded), config.nms_threshold, config.max_detections);
    result.success = true;
    result.result_count = static_cast<int>(decoded.size());
    result.mask_count = static_cast<int>(decoded.size());
    result.proto_shape = proto_shape;
    result.payload_json = segmentation_json(decoded, letterbox);
    result.warning = "segmentation 使用 Rockchip YOLOv8-seg split DFL 多输出后处理；mask 当前使用 bbox polygon 简化表示，proto 尚未栅格化";
    return result;
  }

  decoded.clear();
  proto_shape.clear();
  mask_dim = 0;
  if (decode_fused_segmentation(outputs, config, letterbox, decoded, proto_shape, mask_dim)) {
    decoded = apply_nms(std::move(decoded), config.nms_threshold, config.max_detections);
    result.success = true;
    result.result_count = static_cast<int>(decoded.size());
    result.mask_count = static_cast<int>(decoded.size());
    result.proto_shape = proto_shape;
    result.payload_json = segmentation_json(decoded, letterbox);
    result.warning = "M9.3 无 OpenCV 路径使用检测框简化 polygon；proto 已验证但未栅格化";
    return result;
  }

  result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
  result.error_message = output_shapes_message(outputs);
  return result;
}

std::string make_segmentation_payload_json() {
  return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"surface_region","score":0.89,"bbox_xyxy":[300.0,220.0,980.0,820.0],"mask":{"encoding":"polygon","size":[1080,1920],"polygon":[[[320.0,250.0],[940.0,230.0],[970.0,790.0],[350.0,810.0]]]}}],"measurements":{"mask_area_px":337900,"coverage_ratio":0.1629})json";
}

}  // namespace visionops::runtime
