#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "visionops_runtime/preprocess.hpp"
#include "visionops_runtime/roi_filter.hpp"
#include "visionops_runtime/rknn_runner.hpp"

namespace visionops::runtime {

struct PostprocessConfig {
  std::vector<std::string> class_names;
  float score_threshold{0.5F};
  float nms_threshold{0.45F};
  int max_detections{100};
  RoiFilterConfig roi;
  int mask_max_points{160};
};

struct PostprocessResult {
  bool success{false};
  std::string payload_json;
  std::string error_code;
  std::string error_message;
  std::string warning;
  int result_count{0};
  int raw_result_count{0};
  int roi_filtered_count{0};
  int mask_count{0};
  std::vector<std::uint32_t> proto_shape;
};

class FloatTensorView {
 public:
  FloatTensorView() = default;
  FloatTensorView(const float* data, std::size_t size) : data_(data), size_(size) {}
  const float* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const float& operator[](std::size_t index) const { return data_[index]; }
  const float* begin() const { return data_; }
  const float* end() const { return data_ + size_; }

 private:
  const float* data_{nullptr};
  std::size_t size_{0};
};

FloatTensorView tensor_float_data(const RuntimeTensor& tensor);

}  // namespace visionops::runtime
