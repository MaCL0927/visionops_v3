#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace visionops::runtime {

struct TensorInfo {
  std::string name;
  std::vector<std::uint32_t> dimensions;
  std::string data_type;
  std::string layout;
  std::size_t byte_size{0};
};

struct RuntimeTensor {
  TensorInfo info;
  std::vector<std::uint8_t> data;
};

struct RknnInput {
  int width{0};
  int height{0};
  int channels{3};
  std::vector<std::uint8_t> data;
};

struct RknnOutput {
  bool success{false};
  bool runner_called{false};
  std::string task_type;
  std::string result_payload_json;
  std::vector<RuntimeTensor> tensors;
  double set_input_ms{0.0};
  double run_ms{0.0};
  double get_output_ms{0.0};
  double inference_ms{0.0};
  double postprocess_ms{0.0};
  std::string error;
};

struct RunnerModelConfig {
  std::string task_type{"detection"};
  int input_width{640};
  int input_height{640};
  bool dump_io{false};
};

class RknnRunner {
 public:
  virtual ~RknnRunner() = default;

  virtual bool load_model(const std::string& path, const RunnerModelConfig& config) = 0;
  virtual bool is_loaded() const = 0;
  virtual std::string backend_name() const = 0;
  virtual RknnOutput infer(const RknnInput& input) = 0;
  virtual std::string last_error() const = 0;
  virtual std::uint32_t input_count() const = 0;
  virtual std::uint32_t output_count() const = 0;
};

std::unique_ptr<RknnRunner> create_rknn_runner(
    const std::string& backend,
    const std::string& task_type);
bool rknn_backend_compiled();

}  // namespace visionops::runtime
