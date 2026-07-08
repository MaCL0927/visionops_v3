#include "visionops_runtime/preprocess.hpp"

#include "visionops_runtime/rga_preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <utility>

#ifdef VISIONOPS_HAS_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace visionops::runtime {

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
    output.error = "输入图像必须是非空 RGB 三通道 buffer";
    return output;
  }

  auto& meta = output.letterbox;
  meta.orig_width = image.width;
  meta.orig_height = image.height;
  meta.input_width = input_width;
  meta.input_height = input_height;
  meta.scale = std::min(
      input_width / static_cast<float>(image.width),
      input_height / static_cast<float>(image.height));
  meta.resized_width = std::max(1, static_cast<int>(std::round(image.width * meta.scale)));
  meta.resized_height = std::max(1, static_cast<int>(std::round(image.height * meta.scale)));
  meta.pad_x = (input_width - meta.resized_width) / 2.0F;
  meta.pad_y = (input_height - meta.resized_height) / 2.0F;

  output.input.width = input_width;
  output.input.height = input_height;
  output.input.channels = 3;
  output.input.pixel_format = "RGB888";
  output.input.source = options.mode == "resize" ? "preprocess:resize" : "preprocess:letterbox";
  output.input.sequence = image.sequence;
  output.input.timestamp_ms = image.timestamp_ms;
  output.input.camera_id = image.camera_id;

  if (options.mode == "resize") {
    meta.resized_width = input_width;
    meta.resized_height = input_height;
    meta.pad_x = 0.0F;
    meta.pad_y = 0.0F;
    meta.scale = input_width / static_cast<float>(std::max(1, image.width));

    if (image.width == input_width && image.height == input_height) {
      output.same_size_fast_path = true;
      output.input.data = image.data;
      output.elapsed_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started_at).count();
      return output;
    }

    const bool request_rga = options.backend == "rga" || options.backend == "auto";
    if (request_rga && rga_backend_compiled()) {
      ImageBuffer resized;
      std::string rga_error;
      if (rga_resize_rgb888(image, input_width, input_height, resized, rga_error)) {
        output.input.data = std::move(resized.data);
        output.backend = "rga";
        output.rga_used = true;
        output.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started_at).count();
        return output;
      }
      if (options.backend == "rga") {
        output.error = rga_error.empty() ? "RGA resize 预处理失败" : rga_error;
        return output;
      }
    } else if (options.backend == "rga" && !rga_backend_compiled()) {
      output.error = "当前 Runtime 未编译 RGA 支持，请使用 -DVISIONOPS_ENABLE_RGA=ON";
      return output;
    }

    output.backend = "cpu";
    output.input.data.assign(static_cast<std::size_t>(input_width) * input_height * 3, 0);
    for (int y = 0; y < input_height; ++y) {
      const int source_y = std::min(
          image.height - 1,
          static_cast<int>((y + 0.5F) * image.height / static_cast<float>(input_height)));
      for (int x = 0; x < input_width; ++x) {
        const int source_x = std::min(
            image.width - 1,
            static_cast<int>((x + 0.5F) * image.width / static_cast<float>(input_width)));
        const std::size_t source = (static_cast<std::size_t>(source_y) * image.width + source_x) * 3;
        const std::size_t target = (static_cast<std::size_t>(y) * input_width + x) * 3;
        std::copy_n(image.data.data() + source, 3, output.input.data.data() + target);
      }
    }
    output.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started_at).count();
    return output;
  }

  if (image.width == input_width && image.height == input_height) {
    output.same_size_fast_path = true;
    output.input.data = image.data;
    output.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started_at).count();
    return output;
  }
  output.input.data.assign(static_cast<std::size_t>(input_width) * input_height * 3, 114);
  const int left = static_cast<int>(std::round(meta.pad_x - 0.1F));
  const int top = static_cast<int>(std::round(meta.pad_y - 0.1F));

  const bool request_rga = options.backend == "rga" || options.backend == "auto";
  if (request_rga && rga_backend_compiled()) {
    ImageBuffer resized;
    std::string rga_error;
    if (rga_resize_rgb888(image, meta.resized_width, meta.resized_height, resized, rga_error)) {
      for (int y = 0; y < meta.resized_height; ++y) {
        const auto* src_row = resized.data.data() +
            (static_cast<std::size_t>(y) * meta.resized_width * 3);
        auto* dst_row = output.input.data.data() +
            ((static_cast<std::size_t>(top + y) * input_width + left) * 3);
        std::copy_n(src_row, static_cast<std::size_t>(meta.resized_width) * 3, dst_row);
      }
      output.backend = "rga";
      output.rga_used = true;
      output.elapsed_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started_at).count();
      return output;
    }
    if (options.backend == "rga") {
      output.error = rga_error.empty() ? "RGA 预处理失败" : rga_error;
      return output;
    }
    // auto 模式下，RGA 失败时回退 CPU 最近邻 letterbox，保证现场可用。
  } else if (options.backend == "rga" && !rga_backend_compiled()) {
    output.error = "当前 Runtime 未编译 RGA 支持，请使用 -DVISIONOPS_ENABLE_RGA=ON";
    return output;
  }

  output.backend = "cpu";
  const float inverse_scale = 1.0F / std::max(meta.scale, 1e-6F);
  for (int y = 0; y < meta.resized_height; ++y) {
    const int source_y = std::min(
        image.height - 1,
        static_cast<int>(y * inverse_scale));
    for (int x = 0; x < meta.resized_width; ++x) {
      const int source_x = std::min(
          image.width - 1,
          static_cast<int>(x * inverse_scale));
      const std::size_t source = (static_cast<std::size_t>(source_y) * image.width + source_x) * 3;
      const std::size_t target =
          (static_cast<std::size_t>(top + y) * input_width + left + x) * 3;
      std::copy_n(image.data.data() + source, 3, output.input.data.data() + target);
    }
  }
  output.elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started_at).count();
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
