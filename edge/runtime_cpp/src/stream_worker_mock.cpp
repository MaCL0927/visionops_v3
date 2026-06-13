#include "visionops_runtime/stream_worker.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <utility>

#include "visionops_runtime/json_utils.hpp"

#ifdef __linux__
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace visionops::runtime {

namespace {

std::string normalize_pixel_format(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::toupper(ch));
  });
  if (value == "YUYV422") return "YUYV";
  if (value == "RGB888") return "RGB";
  if (value == "BGR888") return "BGR";
  return value;
}

std::uint8_t clamp_u8(int value) {
  return static_cast<std::uint8_t>(std::max(0, std::min(255, value)));
}

void yuyv_to_rgb888(const std::uint8_t* yuyv, std::size_t size, ImageBuffer& image) {
  image.channels = 3;
  image.pixel_format = "RGB888";
  image.data.resize(static_cast<std::size_t>(image.width) * image.height * 3);
  const std::size_t pixels = static_cast<std::size_t>(image.width) * image.height;
  const std::size_t max_pairs = std::min(size / 4, pixels / 2);
  for (std::size_t pair = 0; pair < max_pairs; ++pair) {
    const std::size_t source = pair * 4;
    const int y0 = static_cast<int>(yuyv[source + 0]);
    const int u = static_cast<int>(yuyv[source + 1]) - 128;
    const int y1 = static_cast<int>(yuyv[source + 2]);
    const int v = static_cast<int>(yuyv[source + 3]) - 128;
    const auto write_pixel = [&](std::size_t index, int y) {
      const int c = y - 16;
      const int r = (298 * c + 409 * v + 128) >> 8;
      const int g = (298 * c - 100 * u - 208 * v + 128) >> 8;
      const int b = (298 * c + 516 * u + 128) >> 8;
      const std::size_t target = index * 3;
      image.data[target + 0] = clamp_u8(r);
      image.data[target + 1] = clamp_u8(g);
      image.data[target + 2] = clamp_u8(b);
    };
    write_pixel(pair * 2, y0);
    if (pair * 2 + 1 < pixels) write_pixel(pair * 2 + 1, y1);
  }
}

}  // namespace

StreamWorkerMock::StreamWorkerMock() = default;

StreamWorkerMock::StreamWorkerMock(FrameSourceConfig config)
    : config_(std::move(config)) {
  if (config_.type.empty()) config_.type = "mock";
  opened_ = config_.type == "mock" || config_.type == "test_image";
}

StreamWorkerMock::~StreamWorkerMock() { stop_preview(); }

void StreamWorkerMock::set_error(std::string error) {
  std::lock_guard<std::mutex> lock(mutex_);
  last_error_ = std::move(error);
  opened_ = false;
}

void StreamWorkerMock::clear_error() {
  std::lock_guard<std::mutex> lock(mutex_);
  last_error_.clear();
}

ImageBuffer StreamWorkerMock::make_mock_image_for_sequence(std::uint64_t sequence) const {
  ImageBuffer image;
  image.width = 1920;
  image.height = 1080;
  image.channels = 3;
  image.pixel_format = "RGB888";
  image.timestamp_ms = now_timestamp_ms();
  image.sequence = sequence;
  image.camera_id = "mock-camera";
  image.source = "frame_source:mock";
  image.data.assign(static_cast<std::size_t>(image.width) * image.height * 3, 114);
  return image;
}

void StreamWorkerMock::start_preview() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (preview_running_) return;
    preview_running_ = true;
  }
  std::string error;
  if (!open_source(error)) {
    set_error(error);
    return;
  }
  clear_error();
  if (config_.type == "v4l2" && config_.enable_camera_thread) {
    stop_thread_.store(false);
    camera_thread_ = std::thread(&StreamWorkerMock::camera_loop, this);
  }
}

void StreamWorkerMock::stop_preview() {
  bool already_stopped = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    already_stopped = !preview_running_ && !camera_thread_.joinable();
    preview_running_ = false;
  }
  if (already_stopped) {
    close_source();
    return;
  }
  stop_thread_.store(true);
  if (camera_thread_.joinable()) {
    camera_thread_.join();
  }
  close_source();
}

bool StreamWorkerMock::preview_running() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return preview_running_;
}

FrameSourceStatus StreamWorkerMock::status() const {
  std::lock_guard<std::mutex> lock(mutex_);
  FrameSourceStatus status;
  status.type = config_.type;
  status.device = config_.camera_device;
  status.opened = opened_;
  status.width = config_.type == "mock" ? 1920 : config_.camera_width;
  status.height = config_.type == "mock" ? 1080 : config_.camera_height;
  status.fps = measured_fps_ > 0.0 ? measured_fps_ : (config_.type == "mock" ? 15.0 : config_.camera_fps);
  status.pixel_format = config_.type == "v4l2" ? normalize_pixel_format(config_.camera_pixel_format) : "RGB888";
  status.latest_timestamp_ms = latest_image_.timestamp_ms;
  status.last_error = last_error_;
  status.snapshot_encoder = "mock_jpeg";
  status.camera_id = config_.type == "v4l2" ? config_.camera_device : config_.type + "-camera";
  if (latest_sequence_ > 0) {
    std::ostringstream frame_id;
    frame_id << "frame-camera-" << latest_sequence_;
    status.latest_frame_id = frame_id.str();
    if (latest_image_.width > 0) status.width = latest_image_.width;
    if (latest_image_.height > 0) status.height = latest_image_.height;
    if (!latest_image_.pixel_format.empty()) status.pixel_format = latest_image_.pixel_format;
  }
  return status;
}

FrameReadResult StreamWorkerMock::next_frame(std::uint64_t sequence) {
  FrameReadResult result;
  result.frame.sequence = sequence;
  if (config_.type == "mock" || config_.type.empty()) {
    result.image = make_mock_image_for_sequence(sequence);
    result.frame.width = result.image.width;
    result.frame.height = result.image.height;
    update_latest(result.image);
    return result;
  }
  if (config_.type == "test_image") {
    result.image = make_mock_image_for_sequence(sequence);
    result.image.source = "frame_source:test_image_placeholder";
    result.frame.width = result.image.width;
    result.frame.height = result.image.height;
    update_latest(result.image);
    return result;
  }
  if (config_.type != "v4l2") {
    result.ok = false;
    result.error = "不支持的 frame-source: " + config_.type;
    set_error(result.error);
    return result;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (latest_image_.data.size() > 0 && latest_image_.width > 0) {
      result.image = latest_image_;
      result.from_cache = true;
      result.frame.width = result.image.width;
      result.frame.height = result.image.height;
      return result;
    }
  }

  std::string error;
  if (!open_source(error)) {
    result.ok = false;
    result.error = error;
    set_error(error);
    return result;
  }
  ImageBuffer image;
  if (!read_frame_once(image, error)) {
    result.ok = false;
    result.error = error;
    set_error(error);
    return result;
  }
  image.sequence = sequence;
  result.image = image;
  result.frame.width = image.width;
  result.frame.height = image.height;
  update_latest(image);
  clear_error();
  return result;
}

bool StreamWorkerMock::open_source(std::string& error) {
  if (config_.type == "mock" || config_.type == "test_image") {
    std::lock_guard<std::mutex> lock(mutex_);
    opened_ = true;
    return true;
  }
  if (config_.type != "v4l2") {
    error = "不支持的 frame-source: " + config_.type;
    return false;
  }
  return open_v4l2(error);
}

void StreamWorkerMock::close_source() {
  if (config_.type == "v4l2") close_v4l2();
}

void StreamWorkerMock::camera_loop() {
  std::uint64_t frames = 0;
  const auto started_at = std::chrono::steady_clock::now();
  while (!stop_thread_.load()) {
    ImageBuffer image;
    std::string error;
    if (read_frame_once(image, error)) {
      ++frames;
      update_latest(std::move(image));
      const auto elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started_at).count();
      if (elapsed > 0.5) {
        std::lock_guard<std::mutex> lock(mutex_);
        measured_fps_ = frames / elapsed;
      }
      clear_error();
    } else {
      set_error(error);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
}

bool StreamWorkerMock::read_frame_once(ImageBuffer& image, std::string& error) {
  if (config_.type != "v4l2") {
    image = make_mock_image_for_sequence(++latest_sequence_);
    return true;
  }
  return read_v4l2_frame(image, error);
}

void StreamWorkerMock::update_latest(ImageBuffer image) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (image.sequence == 0) image.sequence = ++latest_sequence_;
  latest_sequence_ = std::max(latest_sequence_, image.sequence);
  latest_image_ = std::move(image);
  opened_ = true;
}

bool StreamWorkerMock::open_v4l2(std::string& error) {
#ifndef __linux__
  (void)error;
  error = "当前平台不支持 V4L2";
  return false;
#else
  if (v4l2_fd_ >= 0 && v4l2_streaming_) {
    std::lock_guard<std::mutex> lock(mutex_);
    opened_ = true;
    return true;
  }
  const std::string normalized = normalize_pixel_format(config_.camera_pixel_format);
  if (normalized != "YUYV") {
    error = "当前 M10 一期仅支持 V4L2 YUYV；请求格式: " + normalized;
    opened_ = false;
    return false;
  }
  v4l2_fd_ = ::open(config_.camera_device.c_str(), O_RDWR | O_NONBLOCK, 0);
  if (v4l2_fd_ < 0) {
    error = "无法打开摄像头设备 " + config_.camera_device + ": " + std::strerror(errno);
    opened_ = false;
    return false;
  }

  v4l2_capability cap{};
  if (ioctl(v4l2_fd_, VIDIOC_QUERYCAP, &cap) < 0) {
    error = "VIDIOC_QUERYCAP 失败: " + std::string(std::strerror(errno));
    close_v4l2();
    return false;
  }
  if ((cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0 ||
      (cap.capabilities & V4L2_CAP_STREAMING) == 0) {
    error = "摄像头不支持 VIDEO_CAPTURE 或 STREAMING";
    close_v4l2();
    return false;
  }

  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = static_cast<std::uint32_t>(config_.camera_width);
  fmt.fmt.pix.height = static_cast<std::uint32_t>(config_.camera_height);
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
  fmt.fmt.pix.field = V4L2_FIELD_ANY;
  if (ioctl(v4l2_fd_, VIDIOC_S_FMT, &fmt) < 0) {
    error = "VIDIOC_S_FMT 失败: " + std::string(std::strerror(errno));
    close_v4l2();
    return false;
  }
  v4l2_pixel_format_ = fmt.fmt.pix.pixelformat;
  if (v4l2_pixel_format_ != V4L2_PIX_FMT_YUYV) {
    error = "摄像头未接受 YUYV 格式，当前格式暂不支持";
    close_v4l2();
    return false;
  }
  config_.camera_width = static_cast<int>(fmt.fmt.pix.width);
  config_.camera_height = static_cast<int>(fmt.fmt.pix.height);

  v4l2_streamparm parm{};
  parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  parm.parm.capture.timeperframe.numerator = 1;
  parm.parm.capture.timeperframe.denominator =
      static_cast<std::uint32_t>(std::max(1, config_.camera_fps));
  ioctl(v4l2_fd_, VIDIOC_S_PARM, &parm);

  v4l2_requestbuffers req{};
  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (ioctl(v4l2_fd_, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
    error = "VIDIOC_REQBUFS 失败或缓冲区不足: " + std::string(std::strerror(errno));
    close_v4l2();
    return false;
  }
  v4l2_buffers_.resize(req.count);
  for (std::uint32_t index = 0; index < req.count; ++index) {
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = index;
    if (ioctl(v4l2_fd_, VIDIOC_QUERYBUF, &buffer) < 0) {
      error = "VIDIOC_QUERYBUF 失败: " + std::string(std::strerror(errno));
      close_v4l2();
      return false;
    }
    v4l2_buffers_[index].length = buffer.length;
    v4l2_buffers_[index].start = mmap(
        nullptr,
        buffer.length,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        v4l2_fd_,
        buffer.m.offset);
    if (v4l2_buffers_[index].start == MAP_FAILED) {
      v4l2_buffers_[index].start = nullptr;
      error = "mmap 摄像头缓冲区失败: " + std::string(std::strerror(errno));
      close_v4l2();
      return false;
    }
    if (ioctl(v4l2_fd_, VIDIOC_QBUF, &buffer) < 0) {
      error = "VIDIOC_QBUF 失败: " + std::string(std::strerror(errno));
      close_v4l2();
      return false;
    }
  }
  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(v4l2_fd_, VIDIOC_STREAMON, &type) < 0) {
    error = "VIDIOC_STREAMON 失败: " + std::string(std::strerror(errno));
    close_v4l2();
    return false;
  }
  v4l2_streaming_ = true;
  opened_ = true;
  last_error_.clear();
  return true;
#endif
}

void StreamWorkerMock::close_v4l2() {
#ifdef __linux__
  if (v4l2_fd_ >= 0 && v4l2_streaming_) {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(v4l2_fd_, VIDIOC_STREAMOFF, &type);
  }
  v4l2_streaming_ = false;
  for (auto& buffer : v4l2_buffers_) {
    if (buffer.start != nullptr && buffer.length > 0) {
      munmap(buffer.start, buffer.length);
      buffer.start = nullptr;
      buffer.length = 0;
    }
  }
  v4l2_buffers_.clear();
  if (v4l2_fd_ >= 0) {
    ::close(v4l2_fd_);
    v4l2_fd_ = -1;
  }
#endif
  std::lock_guard<std::mutex> lock(mutex_);
  if (config_.type == "v4l2") opened_ = false;
}

bool StreamWorkerMock::read_v4l2_frame(ImageBuffer& image, std::string& error) {
#ifndef __linux__
  error = "当前平台不支持 V4L2";
  return false;
#else
  if (v4l2_fd_ < 0 || !v4l2_streaming_) {
    if (!open_v4l2(error)) return false;
  }
  pollfd descriptor{};
  descriptor.fd = v4l2_fd_;
  descriptor.events = POLLIN;
  const int ready = poll(&descriptor, 1, std::max(1, config_.camera_read_timeout_ms));
  if (ready <= 0) {
    error = ready == 0 ? "读取摄像头超时" : "poll 摄像头失败: " + std::string(std::strerror(errno));
    return false;
  }
  v4l2_buffer buffer{};
  buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buffer.memory = V4L2_MEMORY_MMAP;
  if (ioctl(v4l2_fd_, VIDIOC_DQBUF, &buffer) < 0) {
    error = "VIDIOC_DQBUF 失败: " + std::string(std::strerror(errno));
    return false;
  }
  if (buffer.index >= v4l2_buffers_.size()) {
    error = "摄像头返回非法 buffer index";
    return false;
  }

  image.width = config_.camera_width;
  image.height = config_.camera_height;
  image.timestamp_ms = now_timestamp_ms();
  image.sequence = latest_sequence_ + 1;
  image.camera_id = config_.camera_device;
  image.source = "frame_source:v4l2";
  const auto* data = static_cast<const std::uint8_t*>(v4l2_buffers_[buffer.index].start);
  if (v4l2_pixel_format_ == V4L2_PIX_FMT_YUYV) {
    yuyv_to_rgb888(data, buffer.bytesused, image);
  } else {
    error = "暂不支持当前 V4L2 像素格式";
  }

  if (ioctl(v4l2_fd_, VIDIOC_QBUF, &buffer) < 0) {
    error = "VIDIOC_QBUF 重新入队失败: " + std::string(std::strerror(errno));
    return false;
  }
  if (!error.empty()) return false;
  return image_buffer_valid_rgb(image);
#endif
}

}  // namespace visionops::runtime
