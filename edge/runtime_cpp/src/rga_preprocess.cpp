#include "visionops_runtime/rga_preprocess.hpp"

// 必须位于 Rockchip RGA 头文件之前
#include <cstring>

#ifdef VISIONOPS_HAS_RGA
#include <rga/im2d.hpp>
#include <rga/RgaUtils.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>

namespace visionops::runtime {

bool rga_backend_compiled() {
#ifdef VISIONOPS_HAS_RGA
  return true;
#else
  return false;
#endif
}

bool rga_crop_resize_rgb888(
    const ImageBuffer& src,
    int src_x,
    int src_y,
    int src_width,
    int src_height,
    int canvas_width,
    int canvas_height,
    int dst_x,
    int dst_y,
    int dst_width,
    int dst_height,
    int pad_value,
    ImageBuffer& dst,
    std::string& error) {
#ifndef VISIONOPS_HAS_RGA
  (void)src;
  (void)src_x;
  (void)src_y;
  (void)src_width;
  (void)src_height;
  (void)canvas_width;
  (void)canvas_height;
  (void)dst_x;
  (void)dst_y;
  (void)dst_width;
  (void)dst_height;
  (void)pad_value;
  (void)dst;
  error = "当前 Runtime 未启用 RGA，请使用 -DVISIONOPS_ENABLE_RGA=ON 重新构建";
  return false;
#else
  if (!image_buffer_valid_rgb(src)) {
    error = "RGA 输入图像必须是非空 RGB888 三通道 buffer";
    return false;
  }
  if (src_x < 0 || src_y < 0 || src_width < 2 || src_height < 2 ||
      src_x + src_width > src.width || src_y + src_height > src.height) {
    error = "RGA source ROI 超出输入图像或宽高小于 2";
    return false;
  }
  if (canvas_width <= 0 || canvas_height <= 0 || dst_width < 2 || dst_height < 2 ||
      dst_x < 0 || dst_y < 0 || dst_x + dst_width > canvas_width ||
      dst_y + dst_height > canvas_height) {
    error = "RGA destination rect 超出输出 canvas 或宽高小于 2";
    return false;
  }
  if (pad_value < 0 || pad_value > 255) {
    error = "RGA pad_value 必须位于 0 到 255";
    return false;
  }

  const std::size_t expected_src_bytes =
      static_cast<std::size_t>(src.width) * src.height * src.channels;
  if (src.data.size() < expected_src_bytes) {
    error = "RGA 输入图像数据大小不足";
    return false;
  }

  dst.width = canvas_width;
  dst.height = canvas_height;
  dst.channels = 3;
  dst.pixel_format = "RGB888";
  dst.source = "preprocess:rga_crop_resize";
  dst.sequence = src.sequence;
  dst.timestamp_ms = src.timestamp_ms;
  dst.camera_id = src.camera_id;
  dst.data.assign(
      static_cast<std::size_t>(canvas_width) * canvas_height * 3,
      static_cast<std::uint8_t>(pad_value));

  rga_buffer_t src_buffer = wrapbuffer_virtualaddr(
      const_cast<std::uint8_t*>(src.data.data()),
      src.width,
      src.height,
      RK_FORMAT_RGB_888,
      src.width,
      src.height);
  rga_buffer_t dst_buffer = wrapbuffer_virtualaddr(
      dst.data.data(),
      canvas_width,
      canvas_height,
      RK_FORMAT_RGB_888,
      canvas_width,
      canvas_height);

  im_rect src_rect{};
  src_rect.x = src_x;
  src_rect.y = src_y;
  src_rect.width = src_width;
  src_rect.height = src_height;

  im_rect dst_rect{};
  dst_rect.x = dst_x;
  dst_rect.y = dst_y;
  dst_rect.width = dst_width;
  dst_rect.height = dst_height;

  im_rect pat_rect{};
  rga_buffer_t pat_buffer{};

  // Validate only the actual source and destination operating rectangles.
  // This catches unsupported scale ratios and out-of-range rects before the
  // compound crop+resize call reaches the driver.
  IM_STATUS status = imcheck(src_buffer, dst_buffer, src_rect, dst_rect, 0);
  if (status != IM_STATUS_NOERROR) {
    std::ostringstream stream;
    stream << "RGA imcheck crop+resize 失败: " << imStrError(status);
    error = stream.str();
    dst.data.clear();
    return false;
  }

  // One RGA operation performs source ROI clipping and scaling directly into
  // the final model-input canvas. CPU only initializes letterbox padding.
  status = improcess(
      src_buffer,
      dst_buffer,
      pat_buffer,
      src_rect,
      dst_rect,
      pat_rect,
      0,
      nullptr,
      nullptr,
      0);
  if (status != IM_STATUS_SUCCESS) {
    std::ostringstream stream;
    stream << "RGA improcess crop+resize 失败: " << imStrError(status);
    error = stream.str();
    dst.data.clear();
    return false;
  }
  return true;
#endif
}

bool rga_resize_rgb888(
    const ImageBuffer& src,
    int dst_width,
    int dst_height,
    ImageBuffer& dst,
    std::string& error) {
  return rga_crop_resize_rgb888(
      src,
      0,
      0,
      src.width,
      src.height,
      dst_width,
      dst_height,
      0,
      0,
      dst_width,
      dst_height,
      0,
      dst,
      error);
}

}  // namespace visionops::runtime
