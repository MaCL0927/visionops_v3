#pragma once

#include <cstddef>
#include <cstdint>

namespace visionops::ipc {

constexpr std::uint64_t kSharedDepthMagic = 0x564F505344455031ULL;  // "VOPSDEP1"
constexpr std::uint32_t kSharedDepthVersion = 1;
constexpr std::uint32_t kSharedDepthBufferCount = 2;
constexpr std::uint32_t kSharedDepthPixelFormatUint16Mm = 1;
constexpr std::uint32_t kSharedDepthStateOffline = 0;
constexpr std::uint32_t kSharedDepthStateRunning = 1;
constexpr std::uint32_t kSharedDepthStateStale = 2;

// Cross-process contract for Bridge -> business-App D2C-aligned depth transport.
//
// The writer fills the inactive uint16 millimetre buffer, updates metadata, and
// publishes sequence with release ordering. Readers validate sequence before and
// after sampling. The effective intrinsics already account for optional bridge
// horizontal/vertical flips, so consumers can deproject displayed pixel
// coordinates with the pinhole equations directly.
struct alignas(64) SharedDepthHeader {
  std::uint64_t magic{0};
  std::uint32_t version{0};
  std::uint32_t header_size{0};
  std::uint64_t total_size{0};
  std::uint64_t frame_capacity{0};
  std::uint64_t frame_bytes{0};
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::uint32_t stride_bytes{0};
  std::uint32_t pixel_format{0};
  std::uint32_t buffer_count{0};
  std::uint32_t state{0};
  std::uint32_t active_buffer{0};
  std::uint32_t calibration_ready{0};
  std::uint32_t aligned_to_color{0};
  std::uint32_t flip_horizontal{0};
  std::uint32_t flip_vertical{0};
  std::uint32_t reserved0{0};
  std::uint64_t sequence{0};
  std::uint64_t timestamp_epoch_ms{0};
  std::uint64_t writer_pid{0};
  std::uint64_t publish_count{0};
  std::uint64_t dropped_count{0};
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
  std::uint64_t reserved[12]{};
};

static_assert(sizeof(SharedDepthHeader) == 256, "SharedDepthHeader ABI must remain 256 bytes");
static_assert(offsetof(SharedDepthHeader, sequence) == 88, "SharedDepthHeader sequence offset changed");
static_assert(offsetof(SharedDepthHeader, fx) == 128, "SharedDepthHeader intrinsics offset changed");

inline std::size_t shared_depth_total_size(std::size_t frame_capacity) {
  return sizeof(SharedDepthHeader) + frame_capacity * kSharedDepthBufferCount;
}

inline std::uint8_t* shared_depth_buffer(
    void* mapping,
    std::size_t frame_capacity,
    std::uint32_t index) {
  return static_cast<std::uint8_t*>(mapping) + sizeof(SharedDepthHeader) +
      frame_capacity * (index % kSharedDepthBufferCount);
}

inline const std::uint8_t* shared_depth_buffer(
    const void* mapping,
    std::size_t frame_capacity,
    std::uint32_t index) {
  return static_cast<const std::uint8_t*>(mapping) + sizeof(SharedDepthHeader) +
      frame_capacity * (index % kSharedDepthBufferCount);
}

inline std::uint64_t depth_atomic_load_u64(const std::uint64_t* value) {
  return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

inline std::uint32_t depth_atomic_load_u32(const std::uint32_t* value) {
  return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

inline void depth_atomic_store_u64(std::uint64_t* target, std::uint64_t value) {
  __atomic_store_n(target, value, __ATOMIC_RELEASE);
}

inline void depth_atomic_store_u32(std::uint32_t* target, std::uint32_t value) {
  __atomic_store_n(target, value, __ATOMIC_RELEASE);
}

inline bool valid_shared_depth_header(
    const SharedDepthHeader& header,
    std::size_t mapping_size) {
  if (header.magic != kSharedDepthMagic ||
      header.version != kSharedDepthVersion ||
      header.header_size != sizeof(SharedDepthHeader) ||
      header.buffer_count != kSharedDepthBufferCount ||
      header.pixel_format != kSharedDepthPixelFormatUint16Mm ||
      header.width == 0 || header.height == 0 ||
      header.stride_bytes < header.width * sizeof(std::uint16_t) ||
      header.frame_bytes == 0 || header.frame_bytes > header.frame_capacity) {
    return false;
  }
  return header.total_size <= mapping_size &&
      header.total_size == shared_depth_total_size(
          static_cast<std::size_t>(header.frame_capacity));
}

}  // namespace visionops::ipc
