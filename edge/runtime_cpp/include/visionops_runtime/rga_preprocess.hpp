#pragma once

#include <string>

#include "visionops_runtime/image_buffer.hpp"

namespace visionops::runtime {

bool rga_backend_compiled();

// Crop a source rectangle and resize it directly into a destination rectangle
// on the final RGB888 canvas. The canvas is initialized with pad_value, so a
// letterbox path requires no intermediate ROI/resized ImageBuffer and no CPU
// paste after RGA.
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
    std::string& error);

// Backward-compatible full-frame resize helper.
bool rga_resize_rgb888(
    const ImageBuffer& src,
    int dst_width,
    int dst_height,
    ImageBuffer& dst,
    std::string& error);

}  // namespace visionops::runtime
