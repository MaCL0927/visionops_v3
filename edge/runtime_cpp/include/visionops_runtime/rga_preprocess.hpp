#pragma once

#include <string>

#include "visionops_runtime/image_buffer.hpp"

namespace visionops::runtime {

bool rga_backend_compiled();

// Resize an RGB888 image with RGA. The destination ImageBuffer is always RGB888
// and is resized to dst_width x dst_height. The caller is responsible for
// applying letterbox padding around the resized image if needed.
bool rga_resize_rgb888(
    const ImageBuffer& src,
    int dst_width,
    int dst_height,
    ImageBuffer& dst,
    std::string& error);

}  // namespace visionops::runtime
