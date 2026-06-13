#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "visionops_runtime/postprocess_detect.hpp"
#include "visionops_runtime/postprocess_obb.hpp"
#include "visionops_runtime/postprocess_seg.hpp"

namespace {

visionops::runtime::RuntimeTensor make_tensor(
    std::vector<std::uint32_t> dimensions,
    const std::vector<float>& values) {
  visionops::runtime::RuntimeTensor tensor;
  tensor.info.dimensions = std::move(dimensions);
  tensor.info.data_type = "float32";
  tensor.info.byte_size = values.size() * sizeof(float);
  tensor.data.resize(tensor.info.byte_size);
  std::memcpy(tensor.data.data(), values.data(), tensor.data.size());
  return tensor;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "用法: visionops_postprocess_fixture detection|obb|segmentation\n";
    return 2;
  }
  const std::string task = argv[1];
  const visionops::runtime::LetterboxMeta meta{1280, 720, 640, 640, 640, 360, 0.5F, 0.0F, 140.0F};
  const visionops::runtime::PostprocessConfig config{{"tube"}, 0.5F, 0.45F, 100};
  visionops::runtime::PostprocessResult result;
  std::string output_task = task;
  if (task == "detection") {
    result = visionops::runtime::postprocess_detection(
        {make_tensor({1, 2, 5}, {320, 320, 160, 120, 0.9F, 322, 322, 160, 120, 0.4F})},
        config,
        meta);
  } else if (task == "detection_split") {
    std::vector<float> box(64, 0.0F);
    std::vector<float> classes{0.9F};
    result = visionops::runtime::postprocess_detection(
        {
            make_tensor({1, 64, 1, 1}, box),
            make_tensor({1, 1, 1, 1}, classes),
        },
        config,
        meta);
    output_task = "detection";
  } else if (task == "obb") {
    result = visionops::runtime::postprocess_obb(
        {make_tensor({1, 1, 6}, {320, 320, 180, 80, 0.92F, 0.25F})},
        config,
        meta);
  } else if (task == "segmentation") {
    result = visionops::runtime::postprocess_segmentation(
        {
            make_tensor({1, 1, 7}, {320, 320, 180, 120, 0.93F, 1.0F, -1.0F}),
            make_tensor({1, 2, 2, 2}, {1, 0, 0, 1, 0, 1, 1, 0}),
        },
        config,
        meta);
  } else {
    std::cerr << "未知 fixture task\n";
    return 2;
  }
  if (!result.success) {
    std::cout << "{\"status\":\"error\",\"error\":\"" << result.error_message << "\"}\n";
    return 1;
  }
  std::cout << "{\"schema_version\":\"1.0\",\"message_type\":\"inference_result\","
            << "\"device_id\":\"fixture\",\"component\":\"postprocess_fixture\","
            << "\"timestamp_ms\":1,\"trace_id\":\"fixture\",\"frame_id\":\"frame-fixture\","
            << "\"source\":\"fixture\",\"status\":\"ok\",\"result_id\":\"result-fixture\","
            << "\"task_type\":\"" << output_task << "\","
            << "\"model\":{\"model_id\":\"fixture\",\"model_name\":\"fixture\",\"model_version\":\"1\",\"backend\":\"rknn\",\"input_size\":{\"width\":640,\"height\":640}},"
            << "\"image\":{\"width\":1280,\"height\":720},"
            << "\"timing\":{\"preprocess_ms\":1,\"inference_ms\":1,\"postprocess_ms\":1,\"total_ms\":3}"
            << result.payload_json << "}\n";
  return 0;
}
