#include "visionops_runtime/rknn_runner.hpp"

#include <memory>
#include <stdexcept>

namespace visionops::runtime {

std::unique_ptr<RknnRunner> make_mock_runner(const std::string& task_type);
std::unique_ptr<RknnRunner> make_unavailable_runner(const std::string& task_type);

#ifdef VISIONOPS_HAS_RKNN
std::unique_ptr<RknnRunner> make_real_rknn_runner(const std::string& task_type);
#endif

std::unique_ptr<RknnRunner> create_rknn_runner(
    const std::string& backend,
    const std::string& task_type) {
  if (backend == "mock") {
    return make_mock_runner(task_type);
  }
  if (backend == "rknn") {
#ifdef VISIONOPS_HAS_RKNN
    return make_real_rknn_runner(task_type);
#else
    return make_unavailable_runner(task_type);
#endif
  }
  throw std::invalid_argument("未知 Runtime backend: " + backend);
}

bool rknn_backend_compiled() {
#ifdef VISIONOPS_HAS_RKNN
  return true;
#else
  return false;
#endif
}

}  // namespace visionops::runtime
