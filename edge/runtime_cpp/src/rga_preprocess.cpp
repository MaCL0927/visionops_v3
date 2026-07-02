#include "visionops_runtime/rga_preprocess.hpp"

#ifdef VISIONOPS_HAS_RGA
#include <rga/im2d.hpp>
#include <rga/RgaUtils.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <sstream>

namespace visionops::runtime {

bool rga_backend_compiled() {
#ifdef VISIONOPS_HAS_RGA
  return true;
#else
  return false;
#endif
}

bool rga_resize_rgb888(
    const ImageBuffer& src,
    int dst_width,
    int dst_height,
    ImageBuffer& dst,
    std::string& error) {
#ifndef VISIONOPS_HAS_RGA
  (void)src;
  (void)dst_width;
  (void)dst_height;
  (void)dst;
  error = "当前 Runtime 未启用 RGA，请使用 -DVISIONOPS_ENABLE_RGA=ON 重新构建";
  return false;
#else
  if (!image_buffer_valid_rgb(src)) {
    error = "RGA 输入图像必须是非空 RGB888 三通道 buffer";
    return false;
  }
  if (dst_width <= 0 || dst_height <= 0) {
    error = "RGA 输出尺寸必须为正数";
    return false;
  }

  const std::size_t expected_src_bytes =
      static_cast<std::size_t>(src.width) * src.height * src.channels;
  if (src.data.size() < expected_src_bytes) {
    error = "RGA 输入图像数据大小不足";
    return false;
  }

  dst.width = dst_width;
  dst.height = dst_height;
  dst.channels = 3;
  dst.pixel_format = "RGB888";
  dst.source = "preprocess:rga_resize";
  dst.sequence = src.sequence;
  dst.timestamp_ms = src.timestamp_ms;
  dst.camera_id = src.camera_id;
  dst.data.assign(static_cast<std::size_t>(dst_width) * dst_height * 3, 0);

  // 当前 Runtime 的 ImageBuffer 使用连续 RGB888 内存。这里显式传入
  // wstride/hstride，兼容部分 Rockchip librga 头文件中 wrapbuffer_virtualaddr
  // 宏在 4 参数形式下触发 zero-size array 的问题。
  rga_buffer_t src_buffer = wrapbuffer_virtualaddr(
      const_cast<std::uint8_t*>(src.data.data()),
      src.width,
      src.height,
      RK_FORMAT_RGB_888,
      src.width,
      src.height);
  rga_buffer_t dst_buffer = wrapbuffer_virtualaddr(
      dst.data.data(),
      dst_width,
      dst_height,
      RK_FORMAT_RGB_888,
      dst_width,
      dst_height);

  // 部分 Rockchip librga 头文件里的 imcheck 也是可变参数宏。
  // 不能使用 imcheck(src, dst, {}, {})，否则在 GCC 上会因为空
  // __VA_ARGS__ 生成 zero-size array。这里显式传入 src/dst rect
  // 和 mode_usage=0，兼容 LB3576 当前的 /usr/include/rga/im2d.h。
  im_rect src_rect{};
  src_rect.x = 0;
  src_rect.y = 0;
  src_rect.width = src.width;
  src_rect.height = src.height;

  im_rect dst_rect{};
  dst_rect.x = 0;
  dst_rect.y = 0;
  dst_rect.width = dst_width;
  dst_rect.height = dst_height;

  IM_STATUS status = imcheck(src_buffer, dst_buffer, src_rect, dst_rect, 0);
  if (status != IM_STATUS_NOERROR) {
    std::ostringstream stream;
    stream << "RGA imcheck 失败: " << imStrError(status);
    error = stream.str();
    dst.data.clear();
    return false;
  }

  status = imresize(src_buffer, dst_buffer);
  if (status != IM_STATUS_SUCCESS) {
    std::ostringstream stream;
    stream << "RGA imresize 失败: " << imStrError(status);
    error = stream.str();
    dst.data.clear();
    return false;
  }
  return true;
#endif
}

}  // namespace visionops::runtime
