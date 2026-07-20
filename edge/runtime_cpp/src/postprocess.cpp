#include "visionops_runtime/postprocess.hpp"

#include <cstdint>

namespace visionops::runtime {

FloatTensorView tensor_float_data(const RuntimeTensor& tensor) {
  if (tensor.data_size() == 0 || tensor.data_size() % sizeof(float) != 0) return {};
  const auto* bytes = tensor.data_ptr();
  if (bytes == nullptr || reinterpret_cast<std::uintptr_t>(bytes) % alignof(float) != 0) return {};
  return FloatTensorView(
      reinterpret_cast<const float*>(bytes), tensor.data_size() / sizeof(float));
}

}  // namespace visionops::runtime
