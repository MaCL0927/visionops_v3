#include "visionops_runtime/postprocess_seg.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "visionops_runtime/json_utils.hpp"

namespace visionops::runtime {

PostprocessResult postprocess_segmentation(
    const std::vector<RuntimeTensor>& outputs,
    const PostprocessConfig& config,
    const LetterboxMeta& letterbox) {
  PostprocessResult result;
  if (outputs.size() < 2) {
    result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
    result.error_message = "segmentation 需要 detection tensor 和 proto tensor";
    return result;
  }
  const int class_count = std::max(1, static_cast<int>(config.class_names.size()));
  const auto& proto_dims = outputs[1].info.dimensions;
  int mask_dim = 0;
  if (proto_dims.size() == 4 && proto_dims[0] == 1) {
    mask_dim = static_cast<int>(proto_dims[1]);
  } else if (proto_dims.size() == 3) {
    mask_dim = static_cast<int>(proto_dims[0]);
  }
  if (mask_dim <= 0) {
    result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
    result.error_message = "unsupported_output_shape: segmentation proto";
    return result;
  }
  const auto& dims = outputs[0].info.dimensions;
  const int minimum_channels = 4 + class_count + mask_dim;
  int channels = 0;
  int candidates = 0;
  bool channel_first = true;
  if (dims.size() == 3 && static_cast<int>(dims[1]) >= minimum_channels && dims[1] < dims[2]) {
    channels = dims[1]; candidates = dims[2];
  } else if (dims.size() == 3 && static_cast<int>(dims[2]) >= minimum_channels) {
    candidates = dims[1]; channels = dims[2]; channel_first = false;
  } else {
    result.error_code = "UNSUPPORTED_OUTPUT_SHAPE";
    result.error_message = "unsupported_output_shape: segmentation detection tensor";
    return result;
  }
  const auto values = tensor_float_data(outputs[0]);
  if (values.size() < static_cast<std::size_t>(channels) * candidates) {
    result.error_code = "INVALID_OUTPUT_DATA";
    result.error_message = "segmentation tensor 数据长度不足";
    return result;
  }
  const auto at = [&](int candidate, int channel) {
    return channel_first ? values[channel * candidates + candidate]
                         : values[candidate * channels + channel];
  };
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << ",\"detections\":[";
  int emitted = 0;
  for (int index = 0; index < candidates && emitted < config.max_detections; ++index) {
    int best_class = 0;
    float score = at(index, 4);
    for (int class_id = 1; class_id < class_count; ++class_id) {
      if (at(index, 4 + class_id) > score) {
        score = at(index, 4 + class_id);
        best_class = class_id;
      }
    }
    if (score < 0.0F || score > 1.0F) score = 1.0F / (1.0F + std::exp(-score));
    if (score < config.score_threshold) continue;
    const float cx = at(index, 0);
    const float cy = at(index, 1);
    const float width = at(index, 2);
    const float height = at(index, 3);
    const float x1 = std::clamp((cx - width * 0.5F - letterbox.pad_x) / letterbox.scale, 0.0F, static_cast<float>(letterbox.orig_width - 1));
    const float y1 = std::clamp((cy - height * 0.5F - letterbox.pad_y) / letterbox.scale, 0.0F, static_cast<float>(letterbox.orig_height - 1));
    const float x2 = std::clamp((cx + width * 0.5F - letterbox.pad_x) / letterbox.scale, 0.0F, static_cast<float>(letterbox.orig_width - 1));
    const float y2 = std::clamp((cy + height * 0.5F - letterbox.pad_y) / letterbox.scale, 0.0F, static_cast<float>(letterbox.orig_height - 1));
    if (x2 <= x1 || y2 <= y1) continue;
    if (emitted++ != 0) stream << ',';
    const std::string class_name = best_class < static_cast<int>(config.class_names.size())
        ? config.class_names[best_class] : std::to_string(best_class);
    stream << "{\"id\":\"seg-rknn-" << emitted << "\",\"class_id\":" << best_class
           << ",\"class_name\":\"" << json_escape(class_name) << "\",\"score\":" << score
           << ",\"bbox_xyxy\":[" << x1 << ',' << y1 << ',' << x2 << ',' << y2 << ']'
           << ",\"mask\":{\"encoding\":\"polygon\",\"size\":[" << letterbox.orig_height
           << ',' << letterbox.orig_width << "],\"polygon\":[[[" << x1 << ',' << y1 << "],["
           << x2 << ',' << y1 << "],[" << x2 << ',' << y2 << "],[" << x1 << ',' << y2 << "]]]}}";
  }
  stream << "],\"measurements\":{\"mask_count\":" << emitted << '}';
  result.success = true;
  result.result_count = emitted;
  result.mask_count = emitted;
  result.proto_shape = proto_dims;
  result.payload_json = stream.str();
  result.warning = "M9.3 无 OpenCV 路径使用检测框简化 polygon；proto 已验证但未栅格化";
  return result;
}

std::string make_segmentation_payload_json() {
  return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"surface_region","score":0.89,"bbox_xyxy":[300.0,220.0,980.0,820.0],"mask":{"encoding":"polygon","size":[1080,1920],"polygon":[[[320.0,250.0],[940.0,230.0],[970.0,790.0],[350.0,810.0]]]}}],"measurements":{"mask_area_px":337900,"coverage_ratio":0.1629})json";
}

}  // namespace visionops::runtime
