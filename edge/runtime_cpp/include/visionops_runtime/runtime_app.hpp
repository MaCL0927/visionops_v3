#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "visionops_runtime/app_config.hpp"
#include "visionops_runtime/model_package.hpp"
#include "visionops_runtime/preprocess.hpp"
#include "visionops_runtime/rknn_runner.hpp"
#include "visionops_runtime/roi_filter.hpp"
#include "visionops_runtime/runtime_state.hpp"
#include "visionops_runtime/snapshot_provider.hpp"
#include "visionops_runtime/stream_worker.hpp"

namespace visionops::runtime {

struct RuntimeApiResult {
  int status_code{200};
  std::string body;
};

class RuntimeApp {
 public:
  explicit RuntimeApp(AppConfig config);

  std::string health_json() const;
  std::string status_json() const;
  std::string start_preview();
  std::string stop_preview();
  std::string infer_once();
  RuntimeApiResult switch_model(const std::string& request_body);
  std::string roi_json() const;
  RuntimeApiResult update_roi(const std::string& request_body);
  std::optional<std::string> latest_result_json() const;
  std::vector<std::uint8_t> snapshot_jpeg();

  const AppConfig& config() const;
  std::string snapshot_frame_id() const;
  void record_error();

 private:
  std::string status_json(const RuntimeSnapshot& snapshot) const;
  std::string inference_result_json(
      const InferenceIdentity& identity,
      const MockFrame& frame,
      double capture_ms,
      double decode_ms,
      const PreprocessOutput& preprocess,
      const RknnOutput& inference) const;
  std::string loaded_model_json() const;
  bool runtime_degraded() const;
  std::string inference_error_json(
      const InferenceIdentity& identity,
      const MockFrame& frame,
      const std::string& code,
      const std::string& message,
      double capture_ms = 0.0,
      double decode_ms = 0.0,
      const PreprocessOutput* preprocess = nullptr,
      const RknnOutput* inference = nullptr,
      const std::string& debug_key = "") const;
  std::string postprocess_error_json(
      const InferenceIdentity& identity,
      const MockFrame& frame,
      double capture_ms,
      double decode_ms,
      const PreprocessOutput& preprocess,
      const RknnOutput& inference,
      const std::string& code,
      const std::string& message,
      const std::string& debug_key = "") const;

  AppConfig config_;
  LoadedModelInfo model_info_;
  RuntimeState state_;
  std::unique_ptr<RknnRunner> rknn_runner_;
  StreamWorkerMock stream_worker_;
  SnapshotProvider snapshot_provider_;
  RoiFilterStore roi_filter_;
  mutable std::recursive_mutex model_mutex_;
  std::mutex inference_mutex_;
};

}  // namespace visionops::runtime
