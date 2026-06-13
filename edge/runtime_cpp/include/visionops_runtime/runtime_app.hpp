#pragma once

#include <optional>
#include <string>
#include <vector>

#include "visionops_runtime/app_config.hpp"
#include "visionops_runtime/model_package.hpp"
#include "visionops_runtime/preprocess.hpp"
#include "visionops_runtime/rknn_runner.hpp"
#include "visionops_runtime/runtime_state.hpp"
#include "visionops_runtime/snapshot_provider.hpp"
#include "visionops_runtime/stream_worker.hpp"

namespace visionops::runtime {

class RuntimeApp {
 public:
  explicit RuntimeApp(AppConfig config);

  std::string health_json() const;
  std::string status_json() const;
  std::string start_preview();
  std::string stop_preview();
  std::string infer_once();
  std::optional<std::string> latest_result_json() const;
  const std::vector<std::uint8_t>& snapshot_jpeg() const;

  const AppConfig& config() const;
  std::string snapshot_frame_id() const;
  void record_error();

 private:
  std::string status_json(const RuntimeSnapshot& snapshot) const;
  std::string inference_result_json(
      const InferenceIdentity& identity,
      const MockFrame& frame,
      const PreprocessOutput& preprocess,
      const RknnOutput& inference) const;
  std::string loaded_model_json() const;
  bool runtime_degraded() const;
  std::string inference_error_json(
      const InferenceIdentity& identity,
      const MockFrame& frame,
      const std::string& code,
      const std::string& message,
      const std::string& debug_key = "") const;

  AppConfig config_;
  LoadedModelInfo model_info_;
  RuntimeState state_;
  std::unique_ptr<RknnRunner> rknn_runner_;
  StreamWorkerMock stream_worker_;
  SnapshotProvider snapshot_provider_;
};

}  // namespace visionops::runtime
