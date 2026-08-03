#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "interfaces/cpp/visionops_shared_rgb.hpp"
#include "visionops_runtime/stream_worker.hpp"

namespace {

std::uint64_t epoch_ms() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}

}  // namespace

int main() {
#ifndef __linux__
  std::cout << "{\"status\":\"skipped\",\"reason\":\"linux_only\"}\n";
  return 0;
#else
  using namespace visionops;
  constexpr int width = 32;
  constexpr int height = 24;
  constexpr std::size_t frame_bytes = static_cast<std::size_t>(width) * height * 3u;
  const std::string name = "/visionops_m362_fixture_" + std::to_string(::getpid());
  const std::size_t total_size = ipc::shared_rgb_total_size(frame_bytes);

  ::shm_unlink(name.c_str());
  const int fd = ::shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(total_size)) != 0) {
    std::cerr << "failed to create fixture shared memory\n";
    if (fd >= 0) ::close(fd);
    return 1;
  }
  void* mapping = ::mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapping == MAP_FAILED) {
    std::cerr << "failed to mmap fixture shared memory\n";
    ::close(fd);
    ::shm_unlink(name.c_str());
    return 1;
  }
  std::memset(mapping, 0, total_size);
  auto* header = static_cast<ipc::SharedRgbHeader*>(mapping);
  header->magic = ipc::kSharedRgbMagic;
  header->version = ipc::kSharedRgbVersion;
  header->header_size = sizeof(ipc::SharedRgbHeader);
  header->total_size = total_size;
  header->frame_capacity = frame_bytes;
  header->frame_bytes = frame_bytes;
  header->width = width;
  header->height = height;
  header->channels = 3;
  header->stride_bytes = width * 3;
  header->pixel_format = ipc::kSharedRgbPixelFormatRgb888;
  header->buffer_count = ipc::kSharedRgbBufferCount;
  header->writer_pid = static_cast<std::uint64_t>(::getpid());

  auto publish = [&](std::uint64_t sequence) {
    const std::uint32_t target = static_cast<std::uint32_t>(sequence % ipc::kSharedRgbBufferCount);
    auto* data = ipc::shared_rgb_buffer(mapping, frame_bytes, target);
    std::memset(data, static_cast<int>(sequence & 0xFFu), frame_bytes);
    ipc::atomic_store_u32(&header->state, ipc::kSharedRgbStateRunning);
    ipc::atomic_store_u32(&header->active_buffer, target);
    ipc::atomic_store_u64(&header->timestamp_epoch_ms, epoch_ms());
    ipc::atomic_store_u64(&header->publish_count, sequence);
    ipc::atomic_store_u64(&header->sequence, sequence);
  };

  publish(1);
  runtime::FrameSourceConfig config;
  config.type = "shared_memory";
  config.shared_memory_name = name;
  config.shared_memory_fallback_http = false;
  config.camera_fps = 30;
  config.camera_read_timeout_ms = 100;
  config.stale_frame_timeout_ms = 1000;
  config.reconnect_failure_threshold = 3;
  config.reconnect_initial_ms = 10;
  config.reconnect_max_ms = 50;

  runtime::StreamWorkerMock worker(config);
  worker.start_preview();

  std::atomic_bool stop{false};
  std::thread publisher([&]() {
    std::uint64_t sequence = 2;
    while (!stop.load()) {
      publish(sequence++);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(900));
  const auto status = worker.status();
  runtime::ImageBuffer image;
  const bool has_frame = worker.latest_frame(image);
  stop.store(true);
  publisher.join();
  worker.stop_preview();

  ipc::atomic_store_u32(&header->state, ipc::kSharedRgbStateOffline);
  ::munmap(mapping, total_size);
  ::close(fd);
  ::shm_unlink(name.c_str());

  if (!has_frame || image.width != width || image.height != height ||
      status.transport != "posix_shared_memory" ||
      status.configured_transport != "posix_shared_memory" ||
      status.fallback_active || status.shared_memory_fallback_count != 0 ||
      status.shared_memory_sequence < 2 || status.stale || !status.thread_alive) {
    std::cerr << "shared RGB status mismatch\n";
    return 1;
  }
  // The synthetic writer publishes at 100 FPS. Shared memory must not be
  // throttled by the configured 30 FPS HTTP pacing loop. The old behavior
  // remained around 20-30 FPS because it slept after already waiting for a new
  // sequence.
  if (status.fps < 45.0) {
    std::cerr << "shared RGB reader is unexpectedly paced: fps=" << status.fps << '\n';
    return 1;
  }

  std::cout << "{\"status\":\"ok\",\"transport\":\"" << status.transport
            << "\",\"configured_transport\":\"" << status.configured_transport
            << "\",\"fps\":" << status.fps
            << ",\"sequence\":" << status.shared_memory_sequence
            << ",\"width\":" << image.width
            << ",\"height\":" << image.height << "}\n";
  return 0;
#endif
}
