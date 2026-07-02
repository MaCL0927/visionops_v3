#include "visionops_runtime/rknn_runner.hpp"

#include <utility>

#include "visionops_runtime/app_config.hpp"
#include "visionops_runtime/postprocess_detect.hpp"
#include "visionops_runtime/postprocess_obb.hpp"
#include "visionops_runtime/postprocess_seg.hpp"

namespace visionops::runtime {

namespace {

std::string mock_payload(const std::string& task_type) {
  if (task_type == "obb") return make_obb_payload_json();
  if (task_type == "segmentation") return make_segmentation_payload_json();
  if (task_type == "classification") return make_classification_payload_json();
  if (task_type == "roi_classification") return make_roi_classification_payload_json();
  return make_detection_payload_json();
}

class RknnRunnerMock final : public RknnRunner {
 public:
  explicit RknnRunnerMock(std::string task_type) : task_type_(std::move(task_type)) {}

  bool load_model(const std::string&, const RunnerModelConfig& config) override {
    task_type_ = config.task_type;
    loaded_ = true;
    return true;
  }

  bool is_loaded() const override { return loaded_; }
  std::string backend_name() const override { return "mock"; }

  RknnOutput infer(const RknnInput&) override {
    RknnOutput output;
    output.success = true;
    output.runner_called = true;
    output.task_type = task_type_;
    output.result_payload_json = mock_payload(task_type_);
    output.set_input_ms = 0.0;
    output.run_ms = 12.0;
    output.get_output_ms = 0.0;
    output.inference_ms = 12.0;
    output.postprocess_ms = 2.0;
    return output;
  }

  std::string last_error() const override { return {}; }
  std::uint32_t input_count() const override { return 1; }
  std::uint32_t output_count() const override { return 1; }

 private:
  std::string task_type_;
  bool loaded_{false};
};

class RknnRunnerUnavailable final : public RknnRunner {
 public:
  explicit RknnRunnerUnavailable(std::string task_type) : task_type_(std::move(task_type)) {}

  bool load_model(const std::string&, const RunnerModelConfig& config) override {
    task_type_ = config.task_type;
    error_ = "当前构建未启用 RKNN，请使用 -DVISIONOPS_ENABLE_RKNN=ON 并配置 SDK";
    return false;
  }

  bool is_loaded() const override { return false; }
  std::string backend_name() const override { return "rknn"; }

  RknnOutput infer(const RknnInput&) override {
    RknnOutput output;
    output.success = false;
    output.runner_called = false;
    output.task_type = task_type_;
    output.result_payload_json = mock_payload(task_type_);
    output.postprocess_ms = 2.0;
    output.error = error_;
    return output;
  }

  std::string last_error() const override { return error_; }
  std::uint32_t input_count() const override { return 0; }
  std::uint32_t output_count() const override { return 0; }

 private:
  std::string task_type_;
  std::string error_;
};

}  // namespace

std::unique_ptr<RknnRunner> make_mock_runner(const std::string& task_type) {
  return std::make_unique<RknnRunnerMock>(task_type);
}

std::unique_ptr<RknnRunner> make_unavailable_runner(const std::string& task_type) {
  return std::make_unique<RknnRunnerUnavailable>(task_type);
}

}  // namespace visionops::runtime
