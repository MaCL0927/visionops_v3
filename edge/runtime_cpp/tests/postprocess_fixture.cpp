#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "visionops_runtime/postprocess_detect.hpp"
#include "visionops_runtime/postprocess_classification.hpp"
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

visionops::runtime::LetterboxMeta make_letterbox_meta(
    int orig_width,
    int orig_height,
    int input_width,
    int input_height,
    int resized_width,
    int resized_height,
    float scale,
    float pad_x,
    float pad_y) {
  visionops::runtime::LetterboxMeta meta;
  meta.orig_width = orig_width;
  meta.orig_height = orig_height;
  meta.roi_x = 0;
  meta.roi_y = 0;
  meta.roi_width = orig_width;
  meta.roi_height = orig_height;
  meta.input_width = input_width;
  meta.input_height = input_height;
  meta.resized_width = resized_width;
  meta.resized_height = resized_height;
  meta.scale = scale;
  meta.scale_x = scale;
  meta.scale_y = scale;
  meta.pad_x = pad_x;
  meta.pad_y = pad_y;
  return meta;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "用法: visionops_postprocess_fixture detection|detection_input_roi|obb|obb_input_roi|segmentation|segmentation_highres|segmentation_legacy|classification\n";
    return 2;
  }
  const std::string task = argv[1];
  auto meta = make_letterbox_meta(1280, 720, 640, 640, 640, 360, 0.5F, 0.0F, 140.0F);
  if (task == "detection_input_roi" || task == "obb_input_roi") {
    meta.input_roi_enabled = true;
    meta.roi_x = 850;
    meta.roi_y = 490;
    meta.roi_width = 427;
    meta.roi_height = 229;
    meta.resized_width = 640;
    meta.resized_height = 343;
    meta.scale = 640.0F / 427.0F;
    meta.scale_x = meta.scale;
    meta.scale_y = meta.scale;
    meta.pad_x = 0.0F;
    meta.pad_y = (640.0F - 343.0F) / 2.0F;
  }
  const visionops::runtime::PostprocessConfig config{{"tube", "bag"}, 0.5F, 0.45F, 100, {}};
  visionops::runtime::PostprocessResult result;
  std::string output_task = task;
  if (task == "detection" || task == "detection_roi" || task == "detection_input_roi") {
    auto detection_config = config;
    if (task == "detection_roi") {
      detection_config.roi.enabled = true;
      detection_config.roi.x1 = 0.75;
      detection_config.roi.y1 = 0.0;
      detection_config.roi.x2 = 1.0;
      detection_config.roi.y2 = 1.0;
      output_task = "detection";
    }
    if (task == "detection_input_roi") output_task = "detection";
    result = visionops::runtime::postprocess_detection(
        {make_tensor({1, 2, 6}, {
            320, 320, 160, 120, 0.9F, 0.1F,
            322, 322, 160, 120, 0.4F, 0.2F})},
        detection_config,
        meta);
  } else if (task == "detection_split") {
    std::vector<float> box(64, 0.0F);
    std::vector<float> classes{20.0F, -20.0F};
    result = visionops::runtime::postprocess_detection(
        {
            make_tensor({1, 64, 1, 1}, box),
            make_tensor({1, 2, 1, 1}, classes),
        },
        config,
        meta);
    output_task = "detection";
  } else if (task == "obb" || task == "obb_input_roi") {
    result = visionops::runtime::postprocess_obb(
        {make_tensor({1, 1, 7}, {320, 320, 180, 80, 0.92F, 0.08F, 0.25F})},
        config,
        meta);
    if (task == "obb_input_roi") output_task = "obb";
  } else if (task == "obb_rockchip") {
    std::vector<float> head(66, -20.0F);
    // 64 个 DFL 通道 + 1 个类别通道，单个 1x1 网格。
    // DFL 每个 side 的第 5 个 bin 最高，得到较稳定的旋转框宽高。
    for (int side = 0; side < 4; ++side) {
      head[side * 16 + 5] = 20.0F;
    }
    head[64] = 20.0F;
    head[65] = -20.0F;
    std::vector<float> angle(3, 0.5F);
    result = visionops::runtime::postprocess_obb(
        {
            make_tensor({1, 66, 1, 1}, head),
            make_tensor({1, 66, 1, 1}, head),
            make_tensor({1, 66, 1, 1}, head),
            make_tensor({1, 1, 3}, angle),
        },
        config,
        meta);
    output_task = "obb";
  } else if (task == "obb_rockchip_extra_channel") {
    const auto meta1280 = make_letterbox_meta(1280, 720, 1280, 1280, 1280, 720, 1.0F, 0.0F, 280.0F);
    const visionops::runtime::PostprocessConfig config2{{"bag", "point"}, 0.5F, 0.45F, 100, {}};
    std::vector<float> head(67, -20.0F);
    // 64 个 DFL 通道 + 2 个类别通道 + 1 个额外辅助通道。
    // 1280 OBB RKNN 常见输出为 [1,67,160,160] + [1,1,33600]。
    for (int side = 0; side < 4; ++side) {
      head[side * 16 + 5] = 20.0F;
    }
    head[64] = 20.0F;
    head[65] = -20.0F;
    head[66] = 0.0F;
    std::vector<float> angle(3, 0.5F);
    result = visionops::runtime::postprocess_obb(
        {
            make_tensor({1, 67, 1, 1}, head),
            make_tensor({1, 67, 1, 1}, head),
            make_tensor({1, 67, 1, 1}, head),
            make_tensor({1, 1, 3}, angle),
        },
        config2,
        meta1280);
    output_task = "obb";
  } else if (task == "classification") {
    result = visionops::runtime::postprocess_classification(
        {make_tensor({1, 2}, {0.08F, 0.92F})},
        config);
  } else if (task == "classification_logits") {
    result = visionops::runtime::postprocess_classification(
        {make_tensor({1, 2}, {-2.0F, 4.0F})},
        config);
    output_task = "classification";
  } else if (task == "segmentation") {
    result = visionops::runtime::postprocess_segmentation(
        {
            make_tensor({1, 1, 8}, {320, 320, 180, 120, 0.93F, 0.07F, 1.0F, -1.0F}),
            make_tensor({1, 2, 2, 2}, {1, 0, 0, 1, 0, 1, 1, 0}),
        },
        config,
        meta);
  } else if (task == "segmentation_highres" || task == "segmentation_legacy") {
    const auto square_meta = make_letterbox_meta(640, 640, 640, 640, 640, 640, 1.0F, 0.0F, 0.0F);
    visionops::runtime::PostprocessConfig mask_config{{"ring"}, 0.5F, 0.45F, 100, {}};
    mask_config.mask_max_points = 256;
    mask_config.mask_decode_mode = task == "segmentation_legacy" ? "legacy_proto" : "ultralytics_highres";
    mask_config.mask_threshold = 0.5F;
    mask_config.mask_polygon_epsilon_px = 0.5F;
    std::vector<float> proto(8 * 8, 0.0F);
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        const float dx = static_cast<float>(x) - 3.5F;
        const float dy = static_cast<float>(y) - 3.5F;
        proto[y * 8 + x] = 3.2F - std::sqrt(dx * dx + dy * dy);
      }
    }
    result = visionops::runtime::postprocess_segmentation(
        {
            make_tensor({1, 1, 6}, {320.0F, 320.0F, 600.0F, 600.0F, 0.95F, 1.0F}),
            make_tensor({1, 1, 8, 8}, proto),
        },
        mask_config,
        square_meta);
    output_task = "segmentation";
  } else if (task == "segmentation_split") {
    const visionops::runtime::PostprocessConfig seg_config{{"person", "bag"}, 0.5F, 0.45F, 100, {}};
    auto box_head = []() {
      std::vector<float> values(64, -20.0F);
      for (int side = 0; side < 4; ++side) {
        values[side * 16 + 3] = 20.0F;
      }
      return values;
    };
    result = visionops::runtime::postprocess_segmentation(
        {
            make_tensor({1, 64, 1, 1}, box_head()),
            make_tensor({1, 2, 1, 1}, {-20.0F, 20.0F}),
            make_tensor({1, 1, 1, 1}, {20.0F}),
            make_tensor({1, 32, 1, 1}, std::vector<float>(32, 0.1F)),
            make_tensor({1, 32, 2, 2}, std::vector<float>(128, 0.0F)),
        },
        seg_config,
        meta);
    output_task = "segmentation";
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
