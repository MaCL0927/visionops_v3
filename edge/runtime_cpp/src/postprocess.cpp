#include "visionops_runtime/postprocess.hpp"

#include <cstring>

namespace visionops::runtime {

std::vector<float> tensor_float_data(const RuntimeTensor& tensor) {
  if (tensor.data.size() % sizeof(float) != 0) {
    return {};
  }
  std::vector<float> values(tensor.data.size() / sizeof(float));
  if (!values.empty()) {
    std::memcpy(values.data(), tensor.data.data(), tensor.data.size());
  }
  return values;
}

}  // namespace visionops::runtime
