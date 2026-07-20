#include "visionops_runtime/stream_worker.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>

#include "visionops_runtime/json_utils.hpp"

#ifdef VISIONOPS_HAS_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#ifdef __linux__
#include <fcntl.h>
#include <linux/videodev2.h>
#include <netdb.h>
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

bool is_hp60c_source(const std::string& type) {
  return type == "hp60c_bridge" || type == "hp60c";
}

bool is_live_source(const std::string& type) {
  return type == "v4l2" || is_hp60c_source(type);
}

void sleep_interruptible(const std::atomic_bool& stop_requested, int total_ms) {
  int remaining = std::max(0, total_ms);
  while (remaining > 0 && !stop_requested.load()) {
    const int slice = std::min(remaining, 50);
    std::this_thread::sleep_for(std::chrono::milliseconds(slice));
    remaining -= slice;
  }
}

struct HttpUrlParts {
  std::string host{"127.0.0.1"};
  int port{80};
  std::string path{"/"};
};

std::string join_url_path(std::string base, const std::string& path) {
  if (base.empty()) base = "http://127.0.0.1:18181";
  if (base.rfind("http://", 0) != 0) base = "http://" + base;
  if (path.empty()) return base;
  const auto scheme = base.find("://");
  const auto path_pos = base.find('/', scheme == std::string::npos ? 0 : scheme + 3);
  if (path_pos != std::string::npos) {
    base = base.substr(0, path_pos);
  }
  if (path.front() == '/') return base + path;
  return base + "/" + path;
}

bool parse_http_url(const std::string& url, HttpUrlParts& out, std::string& error) {
  const std::string prefix = "http://";
  if (url.rfind(prefix, 0) != 0) {
    error = "仅支持 http:// URL: " + url;
    return false;
  }
  std::string rest = url.substr(prefix.size());
  const auto slash = rest.find('/');
  std::string host_port = slash == std::string::npos ? rest : rest.substr(0, slash);
  out.path = slash == std::string::npos ? "/" : rest.substr(slash);
  if (host_port.empty()) {
    error = "HTTP URL 缺少主机名: " + url;
    return false;
  }
  const auto colon = host_port.rfind(':');
  if (colon != std::string::npos) {
    out.host = host_port.substr(0, colon);
    try {
      out.port = std::stoi(host_port.substr(colon + 1));
    } catch (const std::exception&) {
      error = "HTTP URL 端口非法: " + url;
      return false;
    }
  } else {
    out.host = host_port;
    out.port = 80;
  }
  if (out.host.empty() || out.port <= 0 || out.port > 65535) {
    error = "HTTP URL 主机或端口非法: " + url;
    return false;
  }
  return true;
}

bool http_get_bytes(
    const std::string& url,
    int timeout_ms,
    std::vector<std::uint8_t>& body,
    int& status_code,
    std::string& content_type,
    std::string& error) {
#ifndef __linux__
  (void)url; (void)timeout_ms; (void)body; (void)status_code; (void)content_type;
  error = "当前平台未实现 HTTP 帧源";
  return false;
#else
  HttpUrlParts parts;
  if (!parse_http_url(url, parts, error)) return false;

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* addresses = nullptr;
  const std::string port_text = std::to_string(parts.port);
  const int lookup = getaddrinfo(parts.host.c_str(), port_text.c_str(), &hints, &addresses);
  if (lookup != 0) {
    error = "HTTP 主机解析失败: " + std::string(gai_strerror(lookup));
    return false;
  }

  int fd = -1;
  for (addrinfo* addr = addresses; addr != nullptr; addr = addr->ai_next) {
    fd = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (fd < 0) continue;
    timeval tv{};
    tv.tv_sec = std::max(1, timeout_ms) / 1000;
    tv.tv_usec = (std::max(1, timeout_ms) % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    if (connect(fd, addr->ai_addr, addr->ai_addrlen) == 0) break;
    ::close(fd);
    fd = -1;
  }
  freeaddrinfo(addresses);
  if (fd < 0) {
    error = "无法连接 HTTP Bridge: " + url + ": " + std::strerror(errno);
    return false;
  }

  std::ostringstream req;
  req << "GET " << parts.path << " HTTP/1.1\r\n"
      << "Host: " << parts.host << ':' << parts.port << "\r\n"
      << "Connection: close\r\n"
      << "Cache-Control: no-cache\r\n\r\n";
  const std::string text = req.str();
  if (send(fd, text.data(), text.size(), 0) < 0) {
    error = "HTTP 请求发送失败: " + std::string(std::strerror(errno));
    ::close(fd);
    return false;
  }

  std::string raw;
  char buffer[8192];
  while (true) {
    const ssize_t count = recv(fd, buffer, sizeof(buffer), 0);
    if (count == 0) break;
    if (count < 0) {
      error = "HTTP 响应读取失败: " + std::string(std::strerror(errno));
      ::close(fd);
      return false;
    }
    raw.append(buffer, static_cast<std::size_t>(count));
    if (raw.size() > 32 * 1024 * 1024) {
      error = "HTTP 响应过大";
      ::close(fd);
      return false;
    }
  }
  ::close(fd);

  const auto header_end = raw.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    error = "HTTP 响应缺少 header/body 分隔符";
    return false;
  }
  const std::string header = raw.substr(0, header_end);
  std::istringstream hs(header);
  std::string http_version;
  hs >> http_version >> status_code;
  std::string line;
  while (std::getline(hs, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string name = line.substr(0, colon);
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    std::string value = line.substr(colon + 1);
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
      return std::isspace(ch) == 0;
    }));
    if (name == "content-type") content_type = value;
  }
  body.assign(raw.begin() + static_cast<std::ptrdiff_t>(header_end + 4), raw.end());
  if (status_code < 200 || status_code >= 300) {
    error = "HTTP Bridge 返回状态码 " + std::to_string(status_code) + ": " + url;
    return false;
  }
  return true;
#endif
}

#ifdef VISIONOPS_HAS_OPENCV
bool decode_jpeg_to_rgb888(const std::vector<std::uint8_t>& jpeg, ImageBuffer& image, std::string& error) {
  if (jpeg.empty()) {
    error = "JPEG 数据为空";
    return false;
  }
  cv::Mat encoded(1, static_cast<int>(jpeg.size()), CV_8UC1, const_cast<std::uint8_t*>(jpeg.data()));
  cv::Mat bgr = cv::imdecode(encoded, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    error = "OpenCV 无法解码 HP60C JPEG";
    return false;
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  image.width = rgb.cols;
  image.height = rgb.rows;
  image.channels = 3;
  image.pixel_format = "RGB888";
  image.data.assign(rgb.data, rgb.data + rgb.total() * rgb.elemSize());
  return image_buffer_valid_rgb(image);
}
#endif

}  // namespace

StreamWorkerMock::StreamWorkerMock() = default;

StreamWorkerMock::StreamWorkerMock(FrameSourceConfig config)
    : config_(std::move(config)) {
  if (config_.type.empty()) config_.type = "mock";
  opened_ = config_.type == "mock" || config_.type == "test_image";
}

StreamWorkerMock::~StreamWorkerMock() { stop_preview(); }

void StreamWorkerMock::set_error(std::string error, bool mark_closed) {
  std::lock_guard<std::mutex> lock(mutex_);
  last_error_ = std::move(error);
  if (mark_closed) opened_ = false;
}

void StreamWorkerMock::clear_error() {
  std::lock_guard<std::mutex> lock(mutex_);
  last_error_.clear();
}

bool StreamWorkerMock::frame_is_fresh_locked(std::uint64_t now_ms) const {
  if (!is_live_source(config_.type)) return true;
  const std::uint64_t timestamp =
      latest_image_.timestamp_ms > 0 ? latest_image_.timestamp_ms : latest_jpeg_timestamp_ms_;
  if (timestamp == 0) return false;
  if (now_ms < timestamp) return true;
  return now_ms - timestamp <= static_cast<std::uint64_t>(config_.stale_frame_timeout_ms);
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
    consecutive_read_errors_ = 0;
  }

  std::string error;
  const bool opened = open_source(error);
  if (!opened) {
    // 首次打开失败不能让 preview 永久卡在“running 但没有采集线程”的状态。
    // 保留错误信息，同时仍启动后台线程，由 camera_loop 按退避策略自动重连。
    set_error(error);
  } else {
    clear_error();
    // 正常情况下仍同步抓取首帧，使 start_preview 返回后页面能尽快显示画面。
    if (is_live_source(config_.type)) {
      ImageBuffer initial_image;
      std::string initial_error;
      double capture_ms = 0.0;
      double decode_ms = 0.0;
      if (read_frame_once(initial_image, capture_ms, decode_ms, initial_error)) {
        update_latest(std::move(initial_image));
        clear_error();
      } else if (!is_hp60c_source(config_.type)) {
        set_error(initial_error, false);
      }
    }
  }

  if (is_live_source(config_.type) && config_.enable_camera_thread) {
    stop_thread_.store(false);
    camera_thread_alive_.store(true);
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

bool StreamWorkerMock::latest_frame(ImageBuffer& image) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!image_buffer_valid_rgb(latest_image_)) return false;
  if (!frame_is_fresh_locked(static_cast<std::uint64_t>(now_timestamp_ms()))) return false;
  image = latest_image_;
  return true;
}

bool StreamWorkerMock::latest_snapshot_jpeg(std::vector<std::uint8_t>& jpeg) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (latest_jpeg_.empty()) return false;
  if (!frame_is_fresh_locked(static_cast<std::uint64_t>(now_timestamp_ms()))) return false;
  jpeg = latest_jpeg_;
  return true;
}

FrameSourceStatus StreamWorkerMock::status() const {
  std::lock_guard<std::mutex> lock(mutex_);
  FrameSourceStatus status;
  status.type = config_.type;
  status.device = config_.camera_device;
  const auto now_ms = static_cast<std::uint64_t>(now_timestamp_ms());
  const auto frame_timestamp =
      latest_image_.timestamp_ms > 0 ? latest_image_.timestamp_ms : latest_jpeg_timestamp_ms_;
  status.latest_frame_age_ms =
      frame_timestamp > 0 && now_ms >= frame_timestamp ? now_ms - frame_timestamp : 0;
  status.stale = is_live_source(config_.type) && !frame_is_fresh_locked(now_ms);
  status.opened = opened_ && !status.stale;
  status.thread_alive = camera_thread_alive_.load();
  status.consecutive_read_errors = consecutive_read_errors_;
  status.reconnect_count = reconnect_count_;
  status.last_reconnect_timestamp_ms = last_reconnect_timestamp_ms_;
  status.width = config_.type == "mock" ? 1920 : config_.camera_width;
  status.height = config_.type == "mock" ? 1080 : config_.camera_height;
  status.fps = status.stale
      ? 0.0
      : (measured_fps_ > 0.0 ? measured_fps_ : (config_.type == "mock" ? 15.0 : config_.camera_fps));
  status.pixel_format = config_.type == "v4l2" ? normalize_pixel_format(config_.camera_pixel_format) : "RGB888";
  if (is_hp60c_source(config_.type)) status.pixel_format = "JPEG/RGB888";
  status.latest_timestamp_ms = latest_image_.timestamp_ms > 0 ? latest_image_.timestamp_ms : latest_jpeg_timestamp_ms_;
  status.last_error = last_error_;
  if (is_hp60c_source(config_.type) && !latest_jpeg_.empty()) {
    status.snapshot_encoder = "hp60c_bridge_jpeg";
  } else {
    status.snapshot_encoder = image_buffer_valid_rgb(latest_image_) ? "rgb888_jpeg" : "mock_jpeg";
  }
  status.camera_id = config_.type == "v4l2" ? config_.camera_device : config_.type + "-camera";
  if (is_hp60c_source(config_.type)) {
    status.camera_id = config_.hp60c_url;
    status.device = join_url_path(config_.hp60c_url, config_.hp60c_snapshot_path);
  }
  status.frames_captured = latest_sequence_;
  if (latest_sequence_ > 0) {
    std::ostringstream frame_id;
    if (is_hp60c_source(config_.type)) {
      frame_id << "frame-hp60c-";
    } else if (config_.type == "v4l2") {
      frame_id << "frame-v4l2-";
    } else {
      frame_id << "frame-camera-";
    }
    frame_id << latest_sequence_;
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
    result.frame.timestamp_ms = result.image.timestamp_ms;
    update_latest(result.image);
    return result;
  }
  if (config_.type == "test_image") {
    result.image = make_mock_image_for_sequence(sequence);
    result.image.source = "frame_source:test_image_placeholder";
    result.frame.width = result.image.width;
    result.frame.height = result.image.height;
    result.frame.timestamp_ms = result.image.timestamp_ms;
    update_latest(result.image);
    return result;
  }
  if (config_.type != "v4l2" && !is_hp60c_source(config_.type)) {
    result.ok = false;
    result.error = "不支持的 frame-source: " + config_.type;
    set_error(result.error);
    return result;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // 只有 preview 线程运行时，latest_image_ 才可以认为是持续更新的实时缓存。
    // 未启动 preview 时，next_frame 应主动从帧源读取新图，避免 infer_once/snapshot 永远使用首帧。
    if (preview_running_ && latest_image_.data.size() > 0 && latest_image_.width > 0 &&
        frame_is_fresh_locked(static_cast<std::uint64_t>(now_timestamp_ms()))) {
      result.image = latest_image_;
      result.from_cache = true;
      result.frame.width = result.image.width;
      result.frame.height = result.image.height;
      result.frame.timestamp_ms = result.image.timestamp_ms;
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
  if (!read_frame_once(image, result.capture_ms, result.decode_ms, error)) {
    result.ok = false;
    result.error = error;
    set_error(error);
    return result;
  }
  image.sequence = sequence;
  result.image = image;
  result.frame.width = image.width;
  result.frame.height = image.height;
  result.frame.timestamp_ms = image.timestamp_ms;
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
  if (is_hp60c_source(config_.type)) {
    return open_hp60c_bridge(error);
  }
  if (config_.type != "v4l2") {
    error = "不支持的 frame-source: " + config_.type;
    return false;
  }
  return open_v4l2(error);
}

void StreamWorkerMock::close_source() {
  if (config_.type == "v4l2") close_v4l2();
  if (is_hp60c_source(config_.type)) {
    std::lock_guard<std::mutex> lock(mutex_);
    opened_ = false;
  }
}

void StreamWorkerMock::camera_loop() {
  camera_thread_alive_.store(true);
  std::uint64_t frames = 0;
  auto fps_window_started = std::chrono::steady_clock::now();
  int consecutive_failures = 0;
  int reconnect_backoff_ms = config_.reconnect_initial_ms;
  const auto http_frame_period = std::chrono::microseconds(
      std::max<std::int64_t>(1, 1000000LL / std::max(1, config_.camera_fps)));
  auto next_http_capture_at = std::chrono::steady_clock::now();

  while (!stop_thread_.load()) {
    bool source_opened = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      source_opened = opened_;
    }

    if (!source_opened) {
      std::string open_error;
      if (!open_source(open_error)) {
        ++consecutive_failures;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          consecutive_read_errors_ = static_cast<std::uint64_t>(consecutive_failures);
        }
        set_error(open_error);
        sleep_interruptible(stop_thread_, reconnect_backoff_ms);
        reconnect_backoff_ms = std::min(reconnect_backoff_ms * 2, config_.reconnect_max_ms);
        continue;
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++reconnect_count_;
        last_reconnect_timestamp_ms_ = static_cast<std::uint64_t>(now_timestamp_ms());
      }
    }

    ImageBuffer image;
    std::string error;
    double capture_ms = 0.0;
    double decode_ms = 0.0;
    if (read_frame_once(image, capture_ms, decode_ms, error)) {
      const bool recovered_after_error = consecutive_failures > 0;
      if (recovered_after_error) {
        frames = 0;
        fps_window_started = std::chrono::steady_clock::now();
      }
      ++frames;
      consecutive_failures = 0;
      reconnect_backoff_ms = config_.reconnect_initial_ms;
      update_latest(std::move(image));
      {
        std::lock_guard<std::mutex> lock(mutex_);
        consecutive_read_errors_ = 0;
      }
      const auto now = std::chrono::steady_clock::now();
      const auto elapsed = std::chrono::duration<double>(now - fps_window_started).count();
      if (elapsed > 0.5) {
        std::lock_guard<std::mutex> lock(mutex_);
        measured_fps_ = frames / elapsed;
      }
      clear_error();

      if (is_hp60c_source(config_.type)) {
        // HTTP snapshot sources return immediately from a cached Bridge frame.
        // Pace this loop to the configured camera rate so Runtime does not open
        // and decode hundreds of duplicate JPEGs per second.  The remaining
        // sleep is calculated against an absolute deadline, so capture/decode
        // time is included in the target period.
        next_http_capture_at += http_frame_period;
        const auto after_work = std::chrono::steady_clock::now();
        if (next_http_capture_at > after_work) {
          const auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
              next_http_capture_at - after_work).count();
          if (remaining_ms > 0) {
            sleep_interruptible(stop_thread_, static_cast<int>(remaining_ms));
          }
        } else {
          next_http_capture_at = after_work;
        }
      }
      continue;
    }


    ++consecutive_failures;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      consecutive_read_errors_ = static_cast<std::uint64_t>(consecutive_failures);
    }

    if (consecutive_failures < config_.reconnect_failure_threshold) {
      set_error(error, false);
      sleep_interruptible(stop_thread_, 50);
      continue;
    }

    // 连续失败达到阈值后，彻底关闭并重建帧源。HTTP Bridge 模式下
    // 这会重新执行 /health 检查；V4L2 模式下会重新 open/STREAMON。
    set_error(error);
    close_source();
    sleep_interruptible(stop_thread_, reconnect_backoff_ms);
    reconnect_backoff_ms = std::min(reconnect_backoff_ms * 2, config_.reconnect_max_ms);
  }

  camera_thread_alive_.store(false);
}

bool StreamWorkerMock::read_frame_once(
    ImageBuffer& image,
    double& capture_ms,
    double& decode_ms,
    std::string& error) {
  capture_ms = 0.0;
  decode_ms = 0.0;
  if (is_hp60c_source(config_.type)) {
    return read_hp60c_bridge_frame(image, capture_ms, decode_ms, error);
  }
  if (config_.type != "v4l2") {
    image = make_mock_image_for_sequence(++latest_sequence_);
    return true;
  }
  return read_v4l2_frame(image, capture_ms, error);
}

void StreamWorkerMock::update_latest(ImageBuffer image) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (image.sequence == 0) image.sequence = ++latest_sequence_;
  latest_sequence_ = std::max(latest_sequence_, image.sequence);
  latest_image_ = std::move(image);
  opened_ = true;
  consecutive_read_errors_ = 0;
}


bool StreamWorkerMock::open_hp60c_bridge(std::string& error) {
  const std::string health_url = join_url_path(config_.hp60c_url, config_.hp60c_health_path);
  std::vector<std::uint8_t> body;
  int status_code = 0;
  std::string content_type;
  if (!http_get_bytes(health_url, config_.camera_open_timeout_ms, body, status_code, content_type, error)) {
    error = "HP60C SDK Bridge 健康检查失败: " + error;
    std::lock_guard<std::mutex> lock(mutex_);
    opened_ = false;
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  opened_ = true;
  last_error_.clear();
  return true;
}

bool StreamWorkerMock::read_hp60c_bridge_frame(
    ImageBuffer& image,
    double& capture_ms,
    double& decode_ms,
    std::string& error) {
#ifndef VISIONOPS_HAS_OPENCV
  (void)image;
  (void)decode_ms;
#endif
  const std::string snapshot_url = join_url_path(config_.hp60c_url, config_.hp60c_snapshot_path);
  std::vector<std::uint8_t> jpeg;
  int status_code = 0;
  std::string content_type;
  const auto capture_started = std::chrono::steady_clock::now();
  if (!http_get_bytes(snapshot_url, config_.camera_read_timeout_ms, jpeg, status_code, content_type, error)) {
    capture_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - capture_started).count();
    error = "HP60C SDK Bridge 快照读取失败: " + error;
    return false;
  }
  capture_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - capture_started).count();
  if (jpeg.size() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8) {
    error = "HP60C SDK Bridge 返回的不是 JPEG 图像";
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_jpeg_ = jpeg;
    latest_jpeg_timestamp_ms_ = now_timestamp_ms();
    if (latest_sequence_ == 0) ++latest_sequence_;
    opened_ = true;
  }

#ifdef VISIONOPS_HAS_OPENCV
  const auto decode_started = std::chrono::steady_clock::now();
  if (!decode_jpeg_to_rgb888(jpeg, image, error)) {
    decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - decode_started).count();
    return false;
  }
  decode_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - decode_started).count();
  image.timestamp_ms = now_timestamp_ms();
  image.sequence = latest_sequence_ + 1;
  image.camera_id = config_.hp60c_url;
  image.source = "frame_source:hp60c_bridge";
  return true;
#else
  error = "HP60C Bridge 已获取 JPEG，但当前 Runtime 未启用 OpenCV，无法解码为 RGB888 参与 RKNN 推理；请使用 -DVISIONOPS_ENABLE_OPENCV=ON 构建";
  return false;
#endif
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
    error = "当前仅支持 V4L2 YUYV；请求格式: " + normalized;
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

bool StreamWorkerMock::read_v4l2_frame(
    ImageBuffer& image,
    double& capture_ms,
    std::string& error) {
#ifndef __linux__
  (void)capture_ms;
  error = "当前平台不支持 V4L2";
  return false;
#else
  const auto started_at = std::chrono::steady_clock::now();
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
  capture_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started_at).count();
  return image_buffer_valid_rgb(image);
#endif
}

}  // namespace visionops::runtime
