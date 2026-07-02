#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "visionops_runtime/image_buffer.hpp"

namespace visionops::runtime {

struct MockFrame {
  std::uint64_t sequence{0};
  int width{1920};
  int height{1080};
};

struct FrameSourceConfig {
  std::string type{"mock"};
  std::string camera_device{"/dev/video0"};
  int camera_width{640};
  int camera_height{480};
  int camera_fps{30};
  std::string camera_pixel_format{"YUYV"};
  std::string hp60c_url{"http://127.0.0.1:18181"};
  std::string hp60c_snapshot_path{"/stream/snapshot.jpg"};
  std::string hp60c_health_path{"/health"};
  std::string test_image;
  std::string snapshot_source{"latest_frame"};
  bool enable_camera_thread{true};
  int camera_open_timeout_ms{3000};
  int camera_read_timeout_ms{1000};
};

struct FrameSourceStatus {
  std::string type{"mock"};
  std::string camera_id{"mock-camera"};
  std::string device;
  bool opened{true};
  int width{1920};
  int height{1080};
  double fps{15.0};
  std::string pixel_format{"RGB888"};
  std::string latest_frame_id;
  std::uint64_t latest_timestamp_ms{0};
  std::string last_error;
  std::string snapshot_encoder{"mock_jpeg"};
  std::uint64_t frames_captured{0};
};

struct FrameReadResult {
  MockFrame frame;
  ImageBuffer image;
  bool ok{true};
  std::string error;
  bool from_cache{false};
  double capture_ms{0.0};
  double decode_ms{0.0};
};

class StreamWorkerMock {
 public:
  StreamWorkerMock();
  explicit StreamWorkerMock(FrameSourceConfig config);
  ~StreamWorkerMock();

  StreamWorkerMock(const StreamWorkerMock&) = delete;
  StreamWorkerMock& operator=(const StreamWorkerMock&) = delete;

  void start_preview();
  void stop_preview();
  FrameReadResult next_frame(std::uint64_t sequence);
  bool latest_frame(ImageBuffer& image) const;
  bool latest_snapshot_jpeg(std::vector<std::uint8_t>& jpeg) const;
  bool preview_running() const;
  FrameSourceStatus status() const;

 private:
  void set_error(std::string error);
  void clear_error();
  ImageBuffer make_mock_image_for_sequence(std::uint64_t sequence) const;
  bool open_source(std::string& error);
  void close_source();
  void camera_loop();
  bool read_frame_once(ImageBuffer& image, double& capture_ms, double& decode_ms, std::string& error);
  bool read_v4l2_frame(ImageBuffer& image, double& capture_ms, std::string& error);
  bool read_hp60c_bridge_frame(ImageBuffer& image, double& capture_ms, double& decode_ms, std::string& error);
  bool open_hp60c_bridge(std::string& error);
  bool open_v4l2(std::string& error);
  void close_v4l2();
  void update_latest(ImageBuffer image);

  FrameSourceConfig config_;
  mutable std::mutex mutex_;
  bool preview_running_{false};
  bool opened_{true};
  std::string last_error_;
  ImageBuffer latest_image_;
  std::vector<std::uint8_t> latest_jpeg_;
  std::uint64_t latest_jpeg_timestamp_ms_{0};
  std::uint64_t latest_sequence_{0};
  double measured_fps_{0.0};
  std::atomic_bool stop_thread_{false};
  std::thread camera_thread_;

#ifdef __linux__
  struct MmapBuffer {
    void* start{nullptr};
    std::size_t length{0};
  };
  int v4l2_fd_{-1};
  std::vector<MmapBuffer> v4l2_buffers_;
  std::uint32_t v4l2_pixel_format_{0};
  bool v4l2_streaming_{false};
#endif
};

}  // namespace visionops::runtime
