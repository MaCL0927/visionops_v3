#include "visionops_runtime/preprocess.hpp"

#include "visionops_runtime/rga_preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <utility>

#ifdef VISIONOPS_HAS_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace visionops::runtime {

namespace {

double elapsed_ms_since(const std::chrono::steady_clock::time_point& started_at) {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started_at).count();
}

void copy_or_resize_cpu(
    const ImageBuffer& image,
    const ResolvedInputRoi& roi,
    int dst_width,
    int dst_height,
    int dst_x,
    int dst_y,
    ImageBuffer& output) {
  for (int y = 0; y < dst_height; ++y) {
    const int source_y = roi.y + std::min(
        roi.height - 1,
        static_cast<int>((y + 0.5F) * roi.height / static_cast<float>(dst_height)));
    for (int x = 0; x < dst_width; ++x) {
      const int source_x = roi.x + std::min(
          roi.width - 1,
          static_cast<int>((x + 0.5F) * roi.width / static_cast<float>(dst_width)));
      const std::size_t source =
          (static_cast<std::size_t>(source_y) * image.width + source_x) * 3;
      const std::size_t target =
          (static_cast<std::size_t>(dst_y + y) * output.width + dst_x + x) * 3;
      std::copy_n(image.data.data() + source, 3, output.data.data() + target);
    }
  }
}

void copy_roi_same_size(
    const ImageBuffer& image,
    const ResolvedInputRoi& roi,
    ImageBuffer& output) {
  const std::size_t row_bytes = static_cast<std::size_t>(roi.width) * 3;
  for (int y = 0; y < roi.height; ++y) {
    const auto* source = image.data.data() +
        (static_cast<std::size_t>(roi.y + y) * image.width + roi.x) * 3;
    auto* target = output.data.data() + static_cast<std::size_t>(y) * row_bytes;
    std::copy_n(source, row_bytes, target);
  }
}

}  // namespace

ImageBuffer make_mock_image(const MockFrame& frame) {
  ImageBuffer image;
  image.width = frame.width;
  image.height = frame.height;
  image.channels = 3;
  image.pixel_format = "RGB888";
  image.source = "mock";
  image.sequence = frame.sequence;
  image.data.assign(
      static_cast<std::size_t>(image.width) * image.height * image.channels,
      114);
  return image;
}

bool load_ppm_image(const std::string& path, ImageBuffer& image, std::string& error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "无法读取测试图片: " + path;
    return false;
  }
  std::string magic;
  int max_value = 0;
  input >> magic >> image.width >> image.height >> max_value;
  input.get();
  if (magic != "P6" || image.width <= 0 || image.height <= 0 || max_value != 255) {
    error = "无 OpenCV 构建仅支持 P6 PPM 测试图片";
    return false;
  }
  image.channels = 3;
  image.pixel_format = "RGB888";
  image.source = "test_image:ppm";
  image.data.resize(static_cast<std::size_t>(image.width) * image.height * image.channels);
  if (!input.read(reinterpret_cast<char*>(image.data.data()), image.data.size())) {
    error = "PPM 测试图片数据不完整";
    image.data.clear();
    return false;
  }
  return true;
}

bool load_test_image(const std::string& path, ImageBuffer& image, std::string& error) {
  if (path.size() >= 4 && path.substr(path.size() - 4) == ".ppm") {
    return load_ppm_image(path, image, error);
  }
#ifdef VISIONOPS_HAS_OPENCV
  const cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    error = "OpenCV 无法解码测试图片: " + path;
    return false;
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  image.width = rgb.cols;
  image.height = rgb.rows;
  image.channels = 3;
  image.pixel_format = "RGB888";
  image.source = "test_image:opencv";
  image.data.assign(rgb.data, rgb.data + rgb.total() * rgb.elemSize());
  return true;
#else
  error = "当前构建未启用 OpenCV；JPEG/PNG 需要 -DVISIONOPS_ENABLE_OPENCV=ON，默认仅支持 P6 PPM";
  return false;
#endif
}

PreprocessOutput preprocess_image(
    const MockFrame& frame,
    const ImageBuffer& image,
    int input_width,
    int input_height,
    const PreprocessOptions& options) {
  const auto started_at = std::chrono::steady_clock::now();
  PreprocessOutput output;
  output.frame = frame;
  output.backend_requested = options.backend;
  output.rga_mode = options.rga_mode;
  output.mode = options.mode.empty() ? "letterbox" : options.mode;
  output.rga_available = rga_backend_compiled();
  if (!image_buffer_valid_rgb(image)) {
    output.error_code = "INPUT_IMAGE_INVALID";
    output.error = "输入图像必须是非空 RGB 三通道 buffer";
    return output;
  }
  if (input_width <= 0 || input_height <= 0) {
    output.error_code = "MODEL_INPUT_SIZE_INVALID";
    output.error = "模型输入尺寸必须为正数";
    return output;
  }
  if (output.mode != "letterbox" && output.mode != "resize") {
    output.error_code = "PREPROCESS_MODE_UNSUPPORTED";
    output.error = "预处理模式仅支持 letterbox 或 resize";
    return output;
  }

  const auto roi_started = std::chrono::steady_clock::now();
  ResolvedInputRoi roi;
  if (!resolve_input_roi(
          options.input_roi,
          image.width,
          image.height,
          roi,
          output.error_code,
          output.error)) {
    output.input_roi_resolve_ms = elapsed_ms_since(roi_started);
    output.elapsed_ms = elapsed_ms_since(started_at);
    return output;
  }
  output.input_roi_resolve_ms = elapsed_ms_since(roi_started);

  auto& meta = output.letterbox;
  meta.orig_width = image.width;
  meta.orig_height = image.height;
  meta.input_roi_enabled = roi.enabled;
  meta.roi_coordinate_space = roi.enabled
      ? options.input_roi.coordinate_space
      : "runtime_snapshot";
  meta.roi_x = roi.x;
  meta.roi_y = roi.y;
  meta.roi_width = roi.width;
  meta.roi_height = roi.height;
  meta.roi_scaled_from_normalized = roi.scaled_from_normalized;
  meta.input_width = input_width;
  meta.input_height = input_height;

  if (output.mode == "resize") {
    meta.resized_width = input_width;
    meta.resized_height = input_height;
    meta.pad_x = 0.0F;
    meta.pad_y = 0.0F;
    meta.scale_x = input_width / static_cast<float>(roi.width);
    meta.scale_y = input_height / static_cast<float>(roi.height);
    meta.scale = meta.scale_x;
  } else {
    meta.scale = std::min(
        input_width / static_cast<float>(roi.width),
        input_height / static_cast<float>(roi.height));
    meta.scale_x = meta.scale;
    meta.scale_y = meta.scale;
    meta.resized_width = std::max(1, static_cast<int>(std::round(roi.width * meta.scale)));
    meta.resized_height = std::max(1, static_cast<int>(std::round(roi.height * meta.scale)));
    meta.pad_x = (input_width - meta.resized_width) / 2.0F;
    meta.pad_y = (input_height - meta.resized_height) / 2.0F;
  }

  output.input.width = input_width;
  output.input.height = input_height;
  output.input.channels = 3;
  output.input.pixel_format = "RGB888";
  output.input.source = roi.enabled
      ? "preprocess:input_roi_" + output.mode
      : "preprocess:" + output.mode;
  output.input.sequence = image.sequence;
  output.input.timestamp_ms = image.timestamp_ms;
  output.input.camera_id = image.camera_id;

  const int left = output.mode == "letterbox"
      ? static_cast<int>(std::round(meta.pad_x - 0.1F))
      : 0;
  const int top = output.mode == "letterbox"
      ? static_cast<int>(std::round(meta.pad_y - 0.1F))
      : 0;
  const int pad_value = options.input_roi.enabled ? options.input_roi.pad_value : 114;

  // Legacy full-frame same-size path avoids all preprocessing. ROI models can
  // still use a cheap row-copy path when crop size already equals model input.
  if (!roi.enabled && image.width == input_width && image.height == input_height) {
    output.same_size_fast_path = true;
    output.input.data = image.data;
    output.elapsed_ms = elapsed_ms_since(started_at);
    return output;
  }
  if (roi.width == input_width && roi.height == input_height && left == 0 && top == 0) {
    output.input.data.assign(
        static_cast<std::size_t>(input_width) * input_height * 3,
        static_cast<std::uint8_t>(pad_value));
    const auto transform_started = std::chrono::steady_clock::now();
    copy_roi_same_size(image, roi, output.input);
    output.crop_resize_ms = elapsed_ms_since(transform_started);
    output.same_size_fast_path = true;
    output.backend = "cpu";
    output.elapsed_ms = elapsed_ms_since(started_at);
    return output;
  }

  const bool request_rga = options.backend == "rga" || options.backend == "auto";
  const auto transform_started = std::chrono::steady_clock::now();
  if (request_rga && rga_backend_compiled()) {
    std::string rga_error;
    if (rga_crop_resize_rgb888(
            image,
            roi.x,
            roi.y,
            roi.width,
            roi.height,
            input_width,
            input_height,
            left,
            top,
            meta.resized_width,
            meta.resized_height,
            pad_value,
            output.input,
            rga_error)) {
      output.backend = "rga";
      output.rga_used = true;
      output.rga_fused_crop_resize = true;
      output.rga_mode = roi.enabled ? "crop_resize_rgb" : "resize_rgb";
      output.crop_resize_ms = elapsed_ms_since(transform_started);
      output.elapsed_ms = elapsed_ms_since(started_at);
      return output;
    }
    if (options.backend == "rga") {
      output.error_code = "RGA_CROP_RESIZE_FAILED";
      output.error = rga_error.empty() ? "RGA crop+resize 预处理失败" : rga_error;
      output.crop_resize_ms = elapsed_ms_since(transform_started);
      output.elapsed_ms = elapsed_ms_since(started_at);
      return output;
    }
    output.warning = rga_error.empty()
        ? "RGA crop+resize 失败，已回退 CPU ROI 预处理"
        : rga_error + "; 已回退 CPU ROI 预处理";
  } else if (options.backend == "rga" && !rga_backend_compiled()) {
    output.error_code = "RGA_NOT_COMPILED";
    output.error = "当前 Runtime 未编译 RGA 支持，请使用 -DVISIONOPS_ENABLE_RGA=ON";
    output.crop_resize_ms = elapsed_ms_since(transform_started);
    output.elapsed_ms = elapsed_ms_since(started_at);
    return output;
  }

  // CPU fallback samples directly from the full-frame ROI into the final model
  // canvas. It does not allocate or copy an intermediate cropped image.
  output.backend = "cpu";
  output.input.data.assign(
      static_cast<std::size_t>(input_width) * input_height * 3,
      static_cast<std::uint8_t>(pad_value));
  copy_or_resize_cpu(
      image,
      roi,
      meta.resized_width,
      meta.resized_height,
      left,
      top,
      output.input);
  output.crop_resize_ms = elapsed_ms_since(transform_started);
  output.elapsed_ms = elapsed_ms_since(started_at);
  return output;
}

PreprocessOutput preprocess_image(
    const MockFrame& frame,
    const ImageBuffer& image,
    int input_width,
    int input_height) {
  return preprocess_image(frame, image, input_width, input_height, PreprocessOptions{});
}

PreprocessOutput preprocess_mock_frame(const MockFrame& frame) {
  return preprocess_image(frame, make_mock_image(frame), 640, 640);
}

}  // namespace visionops::runtime
