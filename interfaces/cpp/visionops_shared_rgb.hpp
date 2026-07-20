#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace visionops::ipc {

constexpr std::uint64_t kSharedRgbMagic = 0x564F505352474231ULL;  // "VOPSRGB1"
constexpr std::uint32_t kSharedRgbVersion = 1;
constexpr std::uint32_t kSharedRgbBufferCount = 2;
constexpr std::uint32_t kSharedRgbPixelFormatRgb888 = 1;
constexpr std::uint32_t kSharedRgbStateOffline = 0;
constexpr std::uint32_t kSharedRgbStateRunning = 1;
constexpr std::uint32_t kSharedRgbStateStale = 2;

// Cross-process contract for Bridge -> Runtime RGB transport.
// Metadata is published with a sequence-counter release store after the selected
// frame buffer has been fully written. Readers validate sequence before/after copy.
struct alignas(64) SharedRgbHeader {
  std::uint64_t magic{0};
  std::uint32_t version{0};
  std::uint32_t header_size{0};
  std::uint64_t total_size{0};
  std::uint64_t frame_capacity{0};
  std::uint64_t frame_bytes{0};
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::uint32_t channels{0};
  std::uint32_t stride_bytes{0};
  std::uint32_t pixel_format{0};
  std::uint32_t buffer_count{0};
  std::uint32_t state{0};
  std::uint32_t active_buffer{0};
  std::uint64_t sequence{0};
  std::uint64_t timestamp_epoch_ms{0};
  std::uint64_t writer_pid{0};
  std::uint64_t publish_count{0};
  std::uint64_t dropped_count{0};
  std::uint64_t reserved[10]{};
};

inline std::size_t shared_rgb_total_size(std::size_t frame_capacity) {
  return sizeof(SharedRgbHeader) + frame_capacity * kSharedRgbBufferCount;
}

inline std::uint8_t* shared_rgb_buffer(void* mapping, std::size_t frame_capacity, std::uint32_t index) {
  return static_cast<std::uint8_t*>(mapping) + sizeof(SharedRgbHeader) +
      frame_capacity * (index % kSharedRgbBufferCount);
}

inline const std::uint8_t* shared_rgb_buffer(
    const void* mapping, std::size_t frame_capacity, std::uint32_t index) {
  return static_cast<const std::uint8_t*>(mapping) + sizeof(SharedRgbHeader) +
      frame_capacity * (index % kSharedRgbBufferCount);
}

inline std::uint64_t atomic_load_u64(const std::uint64_t* value) {
  return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

inline std::uint32_t atomic_load_u32(const std::uint32_t* value) {
  return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

inline void atomic_store_u64(std::uint64_t* target, std::uint64_t value) {
  __atomic_store_n(target, value, __ATOMIC_RELEASE);
}

inline void atomic_store_u32(std::uint32_t* target, std::uint32_t value) {
  __atomic_store_n(target, value, __ATOMIC_RELEASE);
}

inline bool valid_shared_rgb_header(const SharedRgbHeader& header, std::size_t mapping_size) {
  if (header.magic != kSharedRgbMagic || header.version != kSharedRgbVersion ||
      header.header_size != sizeof(SharedRgbHeader) ||
      header.buffer_count != kSharedRgbBufferCount ||
      header.pixel_format != kSharedRgbPixelFormatRgb888 ||
      header.channels != 3 || header.width == 0 || header.height == 0 ||
      header.frame_bytes == 0 || header.frame_bytes > header.frame_capacity) {
    return false;
  }
  return header.total_size <= mapping_size &&
      header.total_size == shared_rgb_total_size(static_cast<std::size_t>(header.frame_capacity));
}

}  // namespace visionops::ipc
