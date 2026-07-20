#include "visionops_runtime/postprocess_seg.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <utility>

#include "visionops_runtime/json_utils.hpp"

namespace visionops::runtime {

namespace {

constexpr int kDflRegMax = 16;
constexpr int kDflBoxChannels = 4 * kDflRegMax;

struct PointF {
  float x{0.0F};
  float y{0.0F};
};

struct PointI {
  int x{0};
  int y{0};
  bool operator<(const PointI& other) const {
    if (x != other.x) return x < other.x;
    return y < other.y;
  }
  bool operator==(const PointI& other) const { return x == other.x && y == other.y; }
};

struct BoundarySegment {
  PointI start;
  PointI end;
};

struct SegItem {
  int class_id{0};
  std::string class_name;
  float score{0.0F};
  float x1{0.0F};
  float y1{0.0F};
  float x2{0.0F};
  float y2{0.0F};
  float input_x1{0.0F};
  float input_y1{0.0F};
  float input_x2{0.0F};
  float input_y2{0.0F};
  std::vector<float> mask_coeffs;
  std::vector<PointF> mask_polygon;
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


float polygon_area(const std::vector<PointI>& ring) {
  if (ring.size() < 3) return 0.0F;
  double area = 0.0;
  for (std::size_t i = 0; i < ring.size(); ++i) {
    const auto& a = ring[i];
    const auto& b = ring[(i + 1) % ring.size()];
    area += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
  }
  return static_cast<float>(0.5 * area);
}

float point_line_distance(const PointF& point, const PointF& start, const PointF& end) {
  const float dx = end.x - start.x;
  const float dy = end.y - start.y;
  const float denom = std::sqrt(dx * dx + dy * dy);
  if (denom < 1e-6F) {
    const float px = point.x - start.x;
    const float py = point.y - start.y;
    return std::sqrt(px * px + py * py);
  }
  return std::fabs(dy * point.x - dx * point.y + end.x * start.y - end.y * start.x) / denom;
}

void rdp_simplify_recursive(
    const std::vector<PointF>& points,
    int first,
    int last,
    float epsilon,
    std::vector<bool>& keep) {
  if (last <= first + 1) return;
  float max_distance = 0.0F;
  int max_index = first;
  for (int index = first + 1; index < last; ++index) {
    const float distance = point_line_distance(points[index], points[first], points[last]);
    if (distance > max_distance) {
      max_distance = distance;
      max_index = index;
    }
  }
  if (max_distance > epsilon) {
    keep[max_index] = true;
    rdp_simplify_recursive(points, first, max_index, epsilon, keep);
    rdp_simplify_recursive(points, max_index, last, epsilon, keep);
  }
}

std::vector<PointF> simplify_polygon(std::vector<PointF> polygon, float epsilon, int max_points) {
  if (polygon.size() <= 3) return polygon;
  if (polygon.size() > 1) {
    const auto& first = polygon.front();
    const auto& last = polygon.back();
    if (std::fabs(first.x - last.x) < 1e-4F && std::fabs(first.y - last.y) < 1e-4F) {
      polygon.pop_back();
    }
  }
  if (polygon.size() <= 3) return polygon;
  std::vector<PointF> closed = polygon;
  closed.push_back(polygon.front());
  std::vector<bool> keep(closed.size(), false);
  keep.front() = true;
  keep.back() = true;
  rdp_simplify_recursive(closed, 0, static_cast<int>(closed.size()) - 1, epsilon, keep);
  std::vector<PointF> simplified;
  for (std::size_t index = 0; index + 1 < closed.size(); ++index) {
    if (keep[index]) simplified.push_back(closed[index]);
  }
  if (simplified.size() < 3) simplified = polygon;
  if (max_points > 0 && static_cast<int>(simplified.size()) > max_points) {
    std::vector<PointF> downsampled;
    downsampled.reserve(max_points);
    for (int index = 0; index < max_points; ++index) {
      const std::size_t source = static_cast<std::size_t>(
          std::floor(index * simplified.size() / static_cast<float>(max_points)));
      downsampled.push_back(simplified[std::min(source, simplified.size() - 1)]);
    }
    simplified = std::move(downsampled);
  }
  return simplified;
}

std::vector<PointI> largest_boundary_loop(const std::vector<std::uint8_t>& binary, int width, int height) {
  if (width <= 0 || height <= 0 || binary.size() < static_cast<std::size_t>(width * height)) return {};
  const auto active = [&](int x, int y) {
    if (x < 0 || y < 0 || x >= width || y >= height) return false;
    return binary[y * width + x] != 0;
  };

  std::vector<BoundarySegment> segments;
  segments.reserve(static_cast<std::size_t>(width * height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (!active(x, y)) continue;
      if (!active(x, y - 1)) segments.push_back({{x, y}, {x + 1, y}});
      if (!active(x + 1, y)) segments.push_back({{x + 1, y}, {x + 1, y + 1}});
      if (!active(x, y + 1)) segments.push_back({{x + 1, y + 1}, {x, y + 1}});
      if (!active(x - 1, y)) segments.push_back({{x, y + 1}, {x, y}});
    }
  }
  if (segments.empty()) return {};

  std::map<PointI, std::vector<int>> by_start;
  for (int index = 0; index < static_cast<int>(segments.size()); ++index) {
    by_start[segments[index].start].push_back(index);
  }

  std::vector<bool> used(segments.size(), false);
  std::vector<PointI> best_loop;
  float best_area = 0.0F;
  for (int seed = 0; seed < static_cast<int>(segments.size()); ++seed) {
    if (used[seed]) continue;
    std::vector<PointI> loop;
    const PointI start = segments[seed].start;
    PointI current = start;
    int segment_index = seed;
    int guard = 0;
    while (segment_index >= 0 && !used[segment_index] && guard++ < static_cast<int>(segments.size()) + 4) {
      const auto& segment = segments[segment_index];
      used[segment_index] = true;
      if (loop.empty()) loop.push_back(segment.start);
      loop.push_back(segment.end);
      current = segment.end;
      if (current == start) break;
      segment_index = -1;
      auto it = by_start.find(current);
      if (it != by_start.end()) {
        for (const int candidate : it->second) {
          if (!used[candidate]) {
            segment_index = candidate;
            break;
          }
        }
      }
    }
    if (loop.size() >= 4 && loop.back() == loop.front()) loop.pop_back();
    if (loop.size() >= 3) {
      const float area = std::fabs(polygon_area(loop));
      if (area > best_area) {
        best_area = area;
        best_loop = std::move(loop);
      }
    }
  }
  return best_loop;
}

bool proto_geometry(const std::vector<std::uint32_t>& proto_shape, int& channels, int& height, int& width) {
  channels = 0;
  height = 0;
  width = 0;
  if (proto_shape.size() == 4 && proto_shape[0] == 1) {
    channels = static_cast<int>(proto_shape[1]);
    height = static_cast<int>(proto_shape[2]);
    width = static_cast<int>(proto_shape[3]);
    return channels > 0 && height > 0 && width > 0;
  }
  if (proto_shape.size() == 3) {
    channels = static_cast<int>(proto_shape[0]);
    height = static_cast<int>(proto_shape[1]);
    width = static_cast<int>(proto_shape[2]);
    return channels > 0 && height > 0 && width > 0;
  }
  return false;
}

void attach_proto_masks(
    std::vector<SegItem>& detections,
    const std::vector<float>& proto,
    const std::vector<std::uint32_t>& proto_shape,
    const LetterboxMeta& letterbox,
    int mask_max_points,
    float mask_threshold = 0.5F) {
  int channels = 0;
  int proto_h = 0;
  int proto_w = 0;
  if (!proto_geometry(proto_shape, channels, proto_h, proto_w)) return;
  const int proto_area = proto_h * proto_w;
  if (proto.size() < static_cast<std::size_t>(channels * proto_area)) return;

  for (auto& detection : detections) {
    if (detection.mask_coeffs.size() < static_cast<std::size_t>(channels)) continue;

    const float input_x1 = std::clamp(std::min(detection.input_x1, detection.input_x2), 0.0F, static_cast<float>(letterbox.input_width));
    const float input_y1 = std::clamp(std::min(detection.input_y1, detection.input_y2), 0.0F, static_cast<float>(letterbox.input_height));
    const float input_x2 = std::clamp(std::max(detection.input_x1, detection.input_x2), 0.0F, static_cast<float>(letterbox.input_width));
    const float input_y2 = std::clamp(std::max(detection.input_y1, detection.input_y2), 0.0F, static_cast<float>(letterbox.input_height));
    if (input_x2 <= input_x1 || input_y2 <= input_y1) continue;

    const int crop_x1 = std::clamp(static_cast<int>(std::floor(input_x1 * proto_w / std::max(1, letterbox.input_width))) - 1, 0, proto_w - 1);
    const int crop_y1 = std::clamp(static_cast<int>(std::floor(input_y1 * proto_h / std::max(1, letterbox.input_height))) - 1, 0, proto_h - 1);
    const int crop_x2 = std::clamp(static_cast<int>(std::ceil(input_x2 * proto_w / std::max(1, letterbox.input_width))) + 1, 0, proto_w);
    const int crop_y2 = std::clamp(static_cast<int>(std::ceil(input_y2 * proto_h / std::max(1, letterbox.input_height))) + 1, 0, proto_h);
    if (crop_x2 <= crop_x1 || crop_y2 <= crop_y1) continue;

    std::vector<std::uint8_t> binary(static_cast<std::size_t>(proto_area), 0);
    int active_count = 0;
    for (int y = crop_y1; y < crop_y2; ++y) {
      for (int x = crop_x1; x < crop_x2; ++x) {
        const int index = y * proto_w + x;
        float logit = 0.0F;
        for (int channel = 0; channel < channels; ++channel) {
          logit += detection.mask_coeffs[channel] * proto[channel * proto_area + index];
        }
        const float value = sigmoid(logit);
        if (value >= mask_threshold) {
          binary[index] = 1;
          ++active_count;
        }
      }
    }
    if (active_count < 3) continue;

    const auto loop = largest_boundary_loop(binary, proto_w, proto_h);
    if (loop.size() < 3) continue;

    std::vector<PointF> polygon;
    polygon.reserve(loop.size());
    for (const auto& point : loop) {
      const float input_x = point.x * letterbox.input_width / static_cast<float>(proto_w);
      const float input_y = point.y * letterbox.input_height / static_cast<float>(proto_h);
      polygon.push_back({map_x(input_x, letterbox), map_y(input_y, letterbox)});
    }
    polygon = simplify_polygon(
        std::move(polygon),
        2.0F,
        std::max(4, mask_max_points));
    if (polygon.size() >= 3) detection.mask_polygon = std::move(polygon);
  }
}

std::string segmentation_json(const std::vector<SegItem>& detections, const LetterboxMeta& letterbox) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << ",\"detections\":[";
  for (std::size_t index = 0; index < detections.size(); ++index) {
    if (index != 0) stream << ',';
    const auto& item = detections[index];
    const bool has_real_mask = item.mask_polygon.size() >= 3;
    stream << "{\"id\":\"seg-rknn-" << std::setw(3) << std::setfill('0') << index + 1
           << "\",\"class_id\":" << item.class_id
           << ",\"class_name\":\"" << json_escape(item.class_name) << '"'
           << ",\"score\":" << item.score
           << ",\"bbox_xyxy\":[" << item.x1 << ',' << item.y1 << ',' << item.x2 << ',' << item.y2 << ']'
           << ",\"center_xy\":[" << (item.x1 + item.x2) * 0.5F << ','
           << (item.y1 + item.y2) * 0.5F << ']'
           << ",\"mask\":{\"encoding\":\"polygon\",\"source\":\""
           << (has_real_mask ? "proto" : "bbox_fallback") << "\",\"size\":[" << letterbox.orig_height
           << ',' << letterbox.orig_width << "],\"polygon\":[[";
    if (has_real_mask) {
      for (std::size_t p = 0; p < item.mask_polygon.size(); ++p) {
        if (p != 0) stream << ',';
        stream << '[' << item.mask_polygon[p].x << ',' << item.mask_polygon[p].y << ']';
      }
    } else {
      stream << '[' << item.x1 << ',' << item.y1 << "],["
             << item.x2 << ',' << item.y1 << "],[" << item.x2 << ',' << item.y2 << "],["
             << item.x1 << ',' << item.y2 << ']';
    }
    stream << "]]}}";
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
    std::vector<float>& proto_values,
    int& mask_dim) {
  const int class_count = std::max(1, static_cast<int>(config.class_names.size()));
  const RuntimeTensor* proto_tensor = find_split_proto_tensor(outputs, class_count);
  if (proto_tensor == nullptr) return false;
  mask_dim = tensor_channels(*proto_tensor);
  if (mask_dim <= 0) return false;
  proto_shape = proto_tensor->info.dimensions;
  proto_values = tensor_float_data(*proto_tensor);

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
    const auto coeffs = tensor_float_data(*coeff_tensor);
    if (coeffs.size() < static_cast<std::size_t>(mask_dim * spatial_size)) {
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
        item.input_x1 = (anchor_x - left) * stride;
        item.input_y1 = (anchor_y - top) * stride;
        item.input_x2 = (anchor_x + right) * stride;
        item.input_y2 = (anchor_y + bottom) * stride;
        item.x1 = map_x(item.input_x1, letterbox);
        item.y1 = map_y(item.input_y1, letterbox);
        item.x2 = map_x(item.input_x2, letterbox);
        item.y2 = map_y(item.input_y2, letterbox);
        item.mask_coeffs.resize(mask_dim);
        for (int channel = 0; channel < mask_dim; ++channel) {
          item.mask_coeffs[channel] = coeffs[channel * spatial_size + index];
        }
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
    std::vector<float>& proto_values,
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
  proto_values = tensor_float_data(outputs[1]);

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
    item.input_x1 = cx - width * 0.5F;
    item.input_y1 = cy - height * 0.5F;
    item.input_x2 = cx + width * 0.5F;
    item.input_y2 = cy + height * 0.5F;
    item.x1 = map_x(item.input_x1, letterbox);
    item.y1 = map_y(item.input_y1, letterbox);
    item.x2 = map_x(item.input_x2, letterbox);
    item.y2 = map_y(item.input_y2, letterbox);
    item.mask_coeffs.resize(mask_dim);
    for (int channel = 0; channel < mask_dim; ++channel) {
      const int offset = 4 + class_count + channel;
      if (offset < channels) item.mask_coeffs[channel] = at(index, offset);
    }
    if (item.x2 > item.x1 + 1.0F && item.y2 > item.y1 + 1.0F) decoded.push_back(item);
  }
  return true;
}

void apply_roi_filter(
    std::vector<SegItem>& detections,
    const RoiFilterConfig& roi,
    const LetterboxMeta& letterbox) {
  if (!roi.enabled) return;
  detections.erase(
      std::remove_if(
          detections.begin(),
          detections.end(),
          [&](const SegItem& item) {
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

PostprocessResult postprocess_segmentation(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox) {
  PostprocessResult result;
  std::vector<SegItem> decoded;
  std::vector<std::uint32_t> proto_shape;
  std::vector<float> proto_values;
  int mask_dim = 0;

  if (decode_split_dfl_segmentation(outputs, config, letterbox, decoded, proto_shape, proto_values, mask_dim)) {
    decoded = apply_nms(std::move(decoded), config.nms_threshold, config.max_detections);
    result.raw_result_count = static_cast<int>(decoded.size());
    apply_roi_filter(decoded, config.roi, letterbox);
    attach_proto_masks(
        decoded,
        proto_values,
        proto_shape,
        letterbox,
        config.mask_max_points);
    result.success = true;
    result.result_count = static_cast<int>(decoded.size());
    result.roi_filtered_count = result.raw_result_count - result.result_count;
    result.mask_count = static_cast<int>(decoded.size());
    result.proto_shape = proto_shape;
    result.payload_json = segmentation_json(decoded, letterbox);
    result.warning = "segmentation 使用 Rockchip YOLOv8-seg split DFL 多输出后处理；mask 已由 coeff×proto 栅格化并转为 polygon，若 mask 为空则回退 bbox polygon";
    return result;
  }

  decoded.clear();
  proto_shape.clear();
  proto_values.clear();
  mask_dim = 0;
  if (decode_fused_segmentation(outputs, config, letterbox, decoded, proto_shape, proto_values, mask_dim)) {
    decoded = apply_nms(std::move(decoded), config.nms_threshold, config.max_detections);
    result.raw_result_count = static_cast<int>(decoded.size());
    apply_roi_filter(decoded, config.roi, letterbox);
    attach_proto_masks(
        decoded,
        proto_values,
        proto_shape,
        letterbox,
        config.mask_max_points);
    result.success = true;
    result.result_count = static_cast<int>(decoded.size());
    result.roi_filtered_count = result.raw_result_count - result.result_count;
    result.mask_count = static_cast<int>(decoded.size());
    result.proto_shape = proto_shape;
    result.payload_json = segmentation_json(decoded, letterbox);
    result.warning = "segmentation fused 输出后处理；mask 已由 coeff×proto 栅格化并转为 polygon，若 mask 为空则回退 bbox polygon";
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
