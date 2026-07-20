// VisionOps v3 HP60C / HP60CN Angstrong SDK HTTP bridge.
//
// Dedicated port: 18181 (Orbbec Gemini 336L remains on 18182).
// API surface intentionally matches the Orbbec bridge where possible:
//   GET  /health
//   GET  /stream/profiles
//   GET  /stream/camera_info
//   GET  /stream/snapshot.jpg
//   GET  /stream/depth.png
//   GET  /stream/depth_vis.jpg
//   GET  /stream/depth_meta
//   GET  /stream.mjpeg, /stream/mjpeg, /stream.mjpg
//   POST /api/coordinate/deproject  (available when intrinsics are configured)
//
// The bridge detects stale RGB/depth frames, clears stale caches, destroys the
// old SDK listener/camera handles and periodically rebuilds them. A separate
// systemd watchdog remains the final fallback if a vendor SDK call blocks.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "as_camera_sdk_api.h"
#include "as_camera_sdk_def.h"

namespace {

std::atomic<bool> g_running{true};

static std::string getenv_str(const char *name, const std::string &fallback) {
    const char *value = std::getenv(name);
    return (value && *value) ? std::string(value) : fallback;
}

static int getenv_int(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (!value || !*value) return fallback;
    try { return std::stoi(value); } catch (...) { return fallback; }
}

static double getenv_double(const char *name, double fallback) {
    const char *value = std::getenv(name);
    if (!value || !*value) return fallback;
    try { return std::stod(value); } catch (...) { return fallback; }
}

static bool getenv_bool(const char *name, bool fallback) {
    std::string value = getenv_str(name, fallback ? "true" : "false");
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

static int64_t steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static int64_t epoch_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static std::string json_escape(const std::string &value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << c; break;
        }
    }
    return out.str();
}

static ssize_t send_all(int fd, const void *data, size_t length) {
    const char *cursor = static_cast<const char *>(data);
    size_t remaining = length;
    while (remaining > 0) {
        const ssize_t count = ::send(fd, cursor, remaining, MSG_NOSIGNAL);
        if (count <= 0) return count;
        cursor += count;
        remaining -= static_cast<size_t>(count);
    }
    return static_cast<ssize_t>(length);
}

static bool send_string(int fd, const std::string &value) {
    return send_all(fd, value.data(), value.size()) == static_cast<ssize_t>(value.size());
}

static std::string status_text(int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 503: return "Service Unavailable";
    default: return "Internal Server Error";
    }
}

static void send_bytes_response(int fd, int status, const std::string &content_type,
                                const void *data, size_t length) {
    std::ostringstream header;
    header << "HTTP/1.1 " << status << " " << status_text(status) << "\r\n"
           << "Content-Type: " << content_type << "\r\n"
           << "Content-Length: " << length << "\r\n"
           << "Cache-Control: no-store, no-cache, must-revalidate\r\n"
           << "Connection: close\r\n\r\n";
    send_string(fd, header.str());
    if (length > 0 && data) send_all(fd, data, length);
}

static void send_json_response(int fd, int status, const std::string &body) {
    send_bytes_response(fd, status, "application/json; charset=utf-8", body.data(), body.size());
}

struct FrameSnapshot {
    std::vector<unsigned char> jpeg;
    int width = 0;
    int height = 0;
    uint64_t frame_id = 0;
    int64_t steady_timestamp_ms = -1;
};

struct DepthSnapshot {
    std::vector<unsigned char> png;
    std::vector<unsigned char> visual_jpeg;
    int width = 0;
    int height = 0;
    uint64_t frame_id = 0;
    int64_t steady_timestamp_ms = -1;
};

class Hp60cSdkBridge {
public:
    Hp60cSdkBridge()
        : config_path_(getenv_str("VISIONOPS_HP60C_CONFIG", "")),
          bind_host_(getenv_str("VISIONOPS_HP60C_HTTP_HOST", "0.0.0.0")),
          http_port_(getenv_int("VISIONOPS_HP60C_HTTP_PORT", 18181)),
          jpeg_quality_(std::max(10, std::min(100, getenv_int("VISIONOPS_HP60C_JPEG_QUALITY", 85)))),
          mjpeg_fps_(std::max(1, std::min(30, getenv_int("VISIONOPS_HP60C_MJPEG_FPS", 10)))),
          flip_vertical_(getenv_bool("VISIONOPS_HP60C_FLIP_VERTICAL", true)),
          flip_horizontal_(getenv_bool("VISIONOPS_HP60C_FLIP_HORIZONTAL", false)),
          rgb_source_(getenv_str("VISIONOPS_HP60C_RGB_SOURCE", "auto")),
          rgb_order_(getenv_str("VISIONOPS_HP60C_RGB_ORDER", "bgr")),
          color_width_hint_(std::max(1, getenv_int("VISIONOPS_HP60C_COLOR_WIDTH", 640))),
          color_height_hint_(std::max(1, getenv_int("VISIONOPS_HP60C_COLOR_HEIGHT", 480))),
          depth_width_hint_(std::max(1, getenv_int("VISIONOPS_HP60C_DEPTH_WIDTH", 640))),
          depth_height_hint_(std::max(1, getenv_int("VISIONOPS_HP60C_DEPTH_HEIGHT", 480))),
          camera_fps_hint_(std::max(1, getenv_int("VISIONOPS_HP60C_FPS", 30))),
          stale_timeout_ms_(std::max(500, getenv_int("VISIONOPS_HP60C_STALE_TIMEOUT_MS", 3000))),
          first_frame_timeout_ms_(std::max(1000, getenv_int("VISIONOPS_HP60C_FIRST_FRAME_TIMEOUT_MS", 8000))),
          reconnect_initial_ms_(std::max(200, getenv_int("VISIONOPS_HP60C_RECONNECT_INITIAL_MS", 1000))),
          reconnect_max_ms_(std::max(reconnect_initial_ms_, getenv_int("VISIONOPS_HP60C_RECONNECT_MAX_MS", 30000))),
          alarm_after_ms_(std::max(1000, getenv_int("VISIONOPS_HP60C_RECONNECT_FAILURE_ALARM_SEC", 15) * 1000)),
          depth_aligned_to_color_(getenv_bool("VISIONOPS_HP60C_DEPTH_ALIGNED_TO_COLOR", true)),
          fx_(getenv_double("VISIONOPS_HP60C_FX", 0.0)),
          fy_(getenv_double("VISIONOPS_HP60C_FY", 0.0)),
          cx_(getenv_double("VISIONOPS_HP60C_CX", 0.0)),
          cy_(getenv_double("VISIONOPS_HP60C_CY", 0.0)) {
        std::transform(rgb_source_.begin(), rgb_source_.end(), rgb_source_.begin(), ::tolower);
        std::transform(rgb_order_.begin(), rgb_order_.end(), rgb_order_.begin(), ::tolower);
    }

    ~Hp60cSdkBridge() { stop(); }

    bool start() {
        set_state("starting", "");
        http_thread_ = std::thread(&Hp60cSdkBridge::http_loop, this);
        recovery_thread_ = std::thread(&Hp60cSdkBridge::recovery_loop, this);
        return true;
    }

    void stop() {
        if (stopped_.exchange(true)) return;
        g_running = false;
        reconnect_requested_ = true;
        recovery_cv_.notify_all();
        frame_cv_.notify_all();
        if (listen_fd_ >= 0) {
            ::shutdown(listen_fd_, SHUT_RDWR);
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
        if (recovery_thread_.joinable()) recovery_thread_.join();
        if (http_thread_.joinable()) http_thread_.join();
        shutdown_sdk();
    }

private:
    static void on_attached(AS_CAM_ATTR_S *attr, void *private_data) {
        auto *self = static_cast<Hp60cSdkBridge *>(private_data);
        if (self) self->handle_attached(attr);
    }

    static void on_detached(AS_CAM_ATTR_S *attr, void *private_data) {
        (void)attr;
        auto *self = static_cast<Hp60cSdkBridge *>(private_data);
        if (self) self->handle_detached();
    }

    static void on_frame(AS_CAM_PTR camera, const AS_SDK_Data_s *data, void *private_data) {
        auto *self = static_cast<Hp60cSdkBridge *>(private_data);
        if (self && data) self->handle_frame(camera, data);
    }

    void set_state(const std::string &state, const std::string &error) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        camera_state_ = state;
        if (!error.empty()) last_error_ = error;
        if (state == "running" && error.empty()) last_error_.clear();
    }

    std::string state() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return camera_state_;
    }

    std::string error_text() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return last_error_;
    }

    int64_t age_ms(const std::atomic<int64_t> &timestamp) const {
        const int64_t value = timestamp.load();
        return value < 0 ? -1 : std::max<int64_t>(0, steady_ms() - value);
    }

    bool color_fresh() const {
        const int64_t age = age_ms(last_color_ms_);
        return frame_valid_.load() && age >= 0 && age <= stale_timeout_ms_;
    }

    bool depth_fresh() const {
        const int64_t age = age_ms(last_depth_ms_);
        return depth_valid_.load() && age >= 0 && age <= stale_timeout_ms_;
    }

    bool camera_connected() const {
        return camera_opened_.load() && color_fresh() && depth_fresh();
    }

    void invalidate_frames() {
        frame_valid_ = false;
        depth_valid_ = false;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_ = FrameSnapshot{};
        }
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            latest_depth_ = DepthSnapshot{};
        }
        frame_cv_.notify_all();
    }

    void request_reconnect(const std::string &reason) {
        camera_opened_ = false;
        invalidate_frames();
        reconnect_requested_ = true;
        if (incident_started_epoch_ms_.load() <= 0) incident_started_epoch_ms_ = epoch_ms();
        set_state("reconnecting", reason);
        recovery_cv_.notify_all();
    }

    void handle_attached(AS_CAM_ATTR_S *attr) {
        if (!attr || stopped_.load() || recovery_shutdown_.load()) return;
        if (camera_setup_in_progress_.exchange(true)) return;
        std::cerr << "[INFO] HP60C camera attached" << std::endl;

        AS_CAM_PTR camera = nullptr;
        int result = AS_SDK_CreateCamHandle(camera, attr);
        if (result != 0 || !camera) {
            camera_setup_in_progress_ = false;
            request_reconnect("AS_SDK_CreateCamHandle failed: " + std::to_string(result));
            return;
        }

        AS_SDK_CAM_MODEL_E model = AS_SDK_CAM_MODEL_UNKNOWN;
        AS_SDK_GetCameraModel(camera, model);
        cam_model_ = static_cast<int>(model);

        result = AS_SDK_OpenCamera(camera, config_path_.c_str());
        if (result != 0) {
            AS_SDK_DestoryCamHandle(camera);
            camera_setup_in_progress_ = false;
            request_reconnect("AS_SDK_OpenCamera failed: " + std::to_string(result));
            return;
        }

        char serial[128] = {0};
        if (AS_SDK_GetSerialNumber(camera, serial, sizeof(serial)) == 0) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            serial_ = serial;
        }

        AS_CAM_Stream_Cb_s callback{};
        callback.callback = &Hp60cSdkBridge::on_frame;
        callback.privateData = this;
        result = AS_SDK_RegisterStreamCallback(camera, &callback);
        if (result == 0) result = AS_SDK_StartStream(camera, 0);
        if (result != 0) {
            AS_SDK_CloseCamera(camera);
            AS_SDK_DestoryCamHandle(camera);
            camera_setup_in_progress_ = false;
            request_reconnect("AS_SDK_StartStream/RegisterCallback failed: " + std::to_string(result));
            return;
        }

        {
            std::lock_guard<std::mutex> lock(camera_mutex_);
            cameras_.push_back(camera);
        }
        camera_opened_ = true;
        camera_setup_in_progress_ = false;
        set_state("starting", "");
        std::cerr << "[INFO] HP60C camera stream started" << std::endl;
    }

    void handle_detached() {
        if (stopped_.load() || recovery_shutdown_.load()) return;
        std::cerr << "[WARN] HP60C camera detached" << std::endl;
        request_reconnect("camera detached");
    }

    void update_depth(const AS_SDK_Data_s *data) {
        if (!(data->depthImg.size > 0 && data->depthImg.data && data->depthImg.width > 0 && data->depthImg.height > 0)) return;
        const int width = static_cast<int>(data->depthImg.width);
        const int height = static_cast<int>(data->depthImg.height);
        const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        cv::Mat depth;
        if (data->depthImg.size >= pixels * sizeof(uint16_t)) {
            depth = cv::Mat(height, width, CV_16UC1, data->depthImg.data).clone();
        } else if (data->depthImg.size >= pixels) {
            cv::Mat raw8(height, width, CV_8UC1, data->depthImg.data);
            raw8.convertTo(depth, CV_16UC1);
        }
        if (depth.empty()) return;
        // Apply the same image orientation to RGB and Depth so model-space pixels
        // continue to address the correct depth sample after Web camera settings change.
        if (flip_vertical_ && flip_horizontal_) cv::flip(depth, depth, -1);
        else if (flip_vertical_) cv::flip(depth, depth, 0);
        else if (flip_horizontal_) cv::flip(depth, depth, 1);

        std::vector<unsigned char> png;
        if (!cv::imencode(".png", depth, png, {cv::IMWRITE_PNG_COMPRESSION, 1})) return;

        double min_value = 0.0, max_value = 0.0;
        cv::minMaxLoc(depth, &min_value, &max_value, nullptr, nullptr, depth > 0);
        cv::Mat visual8, visual_color;
        if (max_value > min_value) {
            depth.convertTo(visual8, CV_8UC1, 255.0 / (max_value - min_value), -min_value * 255.0 / (max_value - min_value));
        } else {
            visual8 = cv::Mat::zeros(depth.size(), CV_8UC1);
        }
        cv::applyColorMap(visual8, visual_color, cv::COLORMAP_JET);
        visual_color.setTo(cv::Scalar(0, 0, 0), depth == 0);
        std::vector<unsigned char> visual_jpeg;
        cv::imencode(".jpg", visual_color, visual_jpeg, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_});

        const int64_t now = steady_ms();
        {
            std::lock_guard<std::mutex> lock(depth_mutex_);
            latest_depth_.png = std::move(png);
            latest_depth_.visual_jpeg = std::move(visual_jpeg);
            latest_depth_.width = width;
            latest_depth_.height = height;
            latest_depth_.frame_id = ++depth_frame_count_;
            latest_depth_.steady_timestamp_ms = now;
        }
        last_depth_ms_ = now;
        depth_valid_ = true;
    }

    cv::Mat decode_color(const AS_SDK_Data_s *data) {
        cv::Mat bgr;
        auto use_mjpeg = [&]() {
            if (!(data->mjpegImg.size > 0 && data->mjpegImg.data)) return cv::Mat{};
            std::vector<unsigned char> bytes(static_cast<unsigned char *>(data->mjpegImg.data),
                                     static_cast<unsigned char *>(data->mjpegImg.data) + data->mjpegImg.size);
            return cv::imdecode(bytes, cv::IMREAD_COLOR);
        };
        auto use_rgb = [&]() {
            if (!(data->rgbImg.size > 0 && data->rgbImg.data && data->rgbImg.width > 0 && data->rgbImg.height > 0)) return cv::Mat{};
            const int width = static_cast<int>(data->rgbImg.width);
            const int height = static_cast<int>(data->rgbImg.height);
            if (data->rgbImg.size < static_cast<size_t>(width) * static_cast<size_t>(height) * 3u) return cv::Mat{};
            cv::Mat raw(height, width, CV_8UC3, data->rgbImg.data);
            cv::Mat out;
            if (rgb_order_ == "rgb") cv::cvtColor(raw, out, cv::COLOR_RGB2BGR);
            else out = raw.clone();
            return out;
        };
        auto use_yuyv = [&]() {
            if (!(data->yuyvImg.size > 0 && data->yuyvImg.data && data->yuyvImg.width > 0 && data->yuyvImg.height > 0)) return cv::Mat{};
            const int width = static_cast<int>(data->yuyvImg.width);
            const int height = static_cast<int>(data->yuyvImg.height);
            cv::Mat raw(height, width, CV_8UC2, data->yuyvImg.data);
            cv::Mat out;
            cv::cvtColor(raw, out, cv::COLOR_YUV2BGR_YUYV);
            return out;
        };

        if (rgb_source_ == "mjpeg") bgr = use_mjpeg();
        else if (rgb_source_ == "rgb") bgr = use_rgb();
        else if (rgb_source_ == "yuyv") bgr = use_yuyv();
        else {
            bgr = use_mjpeg();
            if (bgr.empty()) bgr = use_rgb();
            if (bgr.empty()) bgr = use_yuyv();
        }
        if (!bgr.empty()) {
            if (flip_vertical_ && flip_horizontal_) cv::flip(bgr, bgr, -1);
            else if (flip_vertical_) cv::flip(bgr, bgr, 0);
            else if (flip_horizontal_) cv::flip(bgr, bgr, 1);
        }
        return bgr;
    }

    void handle_frame(AS_CAM_PTR, const AS_SDK_Data_s *data) {
        if (stopped_.load() || recovery_shutdown_.load()) return;
        update_depth(data);
        cv::Mat bgr = decode_color(data);
        if (bgr.empty()) return;
        std::vector<unsigned char> jpeg;
        if (!cv::imencode(".jpg", bgr, jpeg, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_})) return;
        const int64_t now = steady_ms();
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            latest_.jpeg = std::move(jpeg);
            latest_.width = bgr.cols;
            latest_.height = bgr.rows;
            latest_.frame_id = ++frame_count_;
            latest_.steady_timestamp_ms = now;
        }
        last_color_ms_ = now;
        frame_valid_ = true;
        frame_cv_.notify_all();
        if (camera_opened_.load() && depth_fresh()) {
            reconnect_requested_ = false;
            incident_started_epoch_ms_ = 0;
            set_state("running", "");
        }
    }

    bool initialize_sdk_listener() {
        recovery_shutdown_ = false;
        int result = AS_SDK_Init();
        if (result != 0) {
            set_state("offline", "AS_SDK_Init failed: " + std::to_string(result));
            return false;
        }
        sdk_initialized_ = true;
        char version[128] = {0};
        if (AS_SDK_GetSwVersion(version, sizeof(version)) == 0) {
            std::lock_guard<std::mutex> lock(state_mutex_);
            sdk_version_ = version;
        }
        AS_LISTENER_CALLBACK_S callback{};
        callback.onAttached = &Hp60cSdkBridge::on_attached;
        callback.onDetached = &Hp60cSdkBridge::on_detached;
        callback.privateData = this;
        result = AS_SDK_StartListener(callback, AS_LISTENNER_TYPE_USB, true);
        if (result != 0) {
            set_state("offline", "AS_SDK_StartListener failed: " + std::to_string(result));
            AS_SDK_Deinit();
            sdk_initialized_ = false;
            return false;
        }
        listener_started_ = true;
        return true;
    }

    void shutdown_sdk() {
        recovery_shutdown_ = true;
        camera_opened_ = false;
        camera_setup_in_progress_ = false;
        std::list<AS_CAM_PTR> cameras;
        {
            std::lock_guard<std::mutex> lock(camera_mutex_);
            cameras.swap(cameras_);
        }
        for (AS_CAM_PTR camera : cameras) {
            if (!camera) continue;
            AS_SDK_StopStream(camera);
            AS_SDK_CloseCamera(camera);
            AS_SDK_DestoryCamHandle(camera);
        }
        if (listener_started_) {
            AS_SDK_StopListener();
            listener_started_ = false;
        }
        if (sdk_initialized_) {
            AS_SDK_Deinit();
            sdk_initialized_ = false;
        }
        recovery_shutdown_ = false;
    }

    bool interruptible_wait(int wait_ms) {
        std::unique_lock<std::mutex> lock(recovery_mutex_);
        recovery_cv_.wait_for(lock, std::chrono::milliseconds(wait_ms), [&]() { return stopped_.load(); });
        return !stopped_.load();
    }

    bool wait_for_fresh_frames(int timeout_ms) {
        const int64_t deadline = steady_ms() + timeout_ms;
        while (!stopped_.load() && steady_ms() < deadline) {
            if (camera_connected()) return true;
            std::unique_lock<std::mutex> lock(recovery_mutex_);
            recovery_cv_.wait_for(lock, std::chrono::milliseconds(100));
        }
        return camera_connected();
    }

    void recovery_loop() {
        recovery_thread_alive_ = true;
        int backoff_ms = reconnect_initial_ms_;
        reconnect_requested_ = true;
        while (!stopped_.load()) {
            if (camera_connected() && !reconnect_requested_.load()) {
                set_state("running", "");
                std::unique_lock<std::mutex> lock(recovery_mutex_);
                recovery_cv_.wait_for(lock, std::chrono::milliseconds(250), [&]() {
                    return stopped_.load() || reconnect_requested_.load();
                });
                if (!stopped_.load() && !camera_connected()) {
                    request_reconnect("RGB/depth frame stale");
                }
                continue;
            }

            ++reconnect_attempt_count_;
            last_reconnect_epoch_ms_ = epoch_ms();
            set_state("reconnecting", error_text().empty() ? "waiting for HP60C camera" : error_text());
            invalidate_frames();
            shutdown_sdk();
            if (stopped_.load()) break;

            if (!initialize_sdk_listener()) {
                set_state("offline", error_text());
            } else if (wait_for_fresh_frames(first_frame_timeout_ms_)) {
                ++reconnect_success_count_;
                reconnect_requested_ = false;
                incident_started_epoch_ms_ = 0;
                backoff_ms = reconnect_initial_ms_;
                set_state("running", "");
                std::cerr << "[INFO] HP60C recovered, attempt=" << reconnect_attempt_count_.load() << std::endl;
                continue;
            } else {
                request_reconnect("HP60C first RGB/depth frame timeout");
                shutdown_sdk();
            }

            if (!interruptible_wait(backoff_ms)) break;
            backoff_ms = std::min(reconnect_max_ms_, std::max(reconnect_initial_ms_, backoff_ms * 2));
        }
        recovery_thread_alive_ = false;
    }

    FrameSnapshot frame_snapshot() const {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        return latest_;
    }

    DepthSnapshot depth_snapshot() const {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        return latest_depth_;
    }

    std::string health_json() const {
        const bool connected = camera_connected();
        const int64_t color_age = age_ms(last_color_ms_);
        const int64_t depth_age = age_ms(last_depth_ms_);
        const int64_t incident_age = incident_started_epoch_ms_.load() > 0 ? std::max<int64_t>(0, epoch_ms() - incident_started_epoch_ms_.load()) : 0;
        const bool alarm = !connected && incident_age >= alarm_after_ms_;
        std::string serial, sdk;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            serial = serial_;
            sdk = sdk_version_;
        }
        std::ostringstream out;
        out << std::boolalpha << std::fixed << std::setprecision(3)
            << "{\"ok\":true,\"backend\":\"angstrong-sdk\",\"camera_model\":\"hp60c\""
            << ",\"camera_started\":" << camera_opened_.load()
            << ",\"camera_opened\":" << camera_opened_.load()
            << ",\"camera_connected\":" << connected
            << ",\"camera_state\":\"" << json_escape(state()) << "\""
            << ",\"frame_stale\":" << (!connected)
            << ",\"camera_thread_alive\":" << recovery_thread_alive_.load()
            << ",\"last_color_age_ms\":" << color_age
            << ",\"last_depth_age_ms\":" << depth_age
            << ",\"latest_snapshot_age_ms\":" << color_age
            << ",\"frame_count\":" << frame_count_.load()
            << ",\"depth_frame_count\":" << depth_frame_count_.load()
            << ",\"reconnect_attempt_count\":" << reconnect_attempt_count_.load()
            << ",\"reconnect_success_count\":" << reconnect_success_count_.load()
            << ",\"last_reconnect_timestamp_ms\":" << last_reconnect_epoch_ms_.load()
            << ",\"alarm_active\":" << alarm
            << ",\"fault_numeric_code\":" << (connected ? 0 : 3101)
            << ",\"http_port\":" << http_port_
            << ",\"serial\":\"" << json_escape(serial) << "\""
            << ",\"sdk_version\":\"" << json_escape(sdk) << "\""
            << ",\"depth_aligned_to_color\":" << depth_aligned_to_color_
            << ",\"error\":";
        const std::string error = error_text();
        if (error.empty()) out << "null";
        else out << "\"" << json_escape(error) << "\"";
        out << "}";
        return out.str();
    }

    std::string profiles_json() const {
        const FrameSnapshot frame = frame_snapshot();
        const DepthSnapshot depth = depth_snapshot();
        const int color_width = frame.width > 0 ? frame.width : color_width_hint_;
        const int color_height = frame.height > 0 ? frame.height : color_height_hint_;
        const int depth_width = depth.width > 0 ? depth.width : depth_width_hint_;
        const int depth_height = depth.height > 0 ? depth.height : depth_height_hint_;
        std::ostringstream out;
        out << "{\"ok\":true,\"camera_model\":\"hp60c\",\"profiles\":{"
            << "\"color\":[{\"width\":" << color_width << ",\"height\":" << color_height
            << ",\"fps\":" << camera_fps_hint_ << ",\"formats\":[\"MJPEG\",\"RGB\",\"YUYV\"]}],"
            << "\"depth\":[{\"width\":" << depth_width << ",\"height\":" << depth_height
            << ",\"fps\":" << camera_fps_hint_ << ",\"formats\":[\"Y16\"]}]},"
            << "\"profile_control\":\"vendor_config_file\"}";
        return out.str();
    }

    bool intrinsics_ready() const { return fx_ > 0.0 && fy_ > 0.0; }

    std::string camera_info_json() const {
        const FrameSnapshot frame = frame_snapshot();
        const int width = frame.width > 0 ? frame.width : color_width_hint_;
        const int height = frame.height > 0 ? frame.height : color_height_hint_;
        std::ostringstream out;
        out << std::boolalpha << "{\"ok\":true,\"camera_model\":\"hp60c\","
            << "\"width\":" << width << ",\"height\":" << height
            << ",\"depth_aligned_to_color\":" << depth_aligned_to_color_
            << ",\"intrinsics_available\":" << intrinsics_ready()
            << ",\"intrinsics\":{\"fx\":" << fx_ << ",\"fy\":" << fy_
            << ",\"cx\":" << cx_ << ",\"cy\":" << cy_ << "},"
            << "\"note\":\"HP60C profile/exposure are controlled by the Angstrong encrypted config file\"}";
        return out.str();
    }

    std::string depth_meta_json() const {
        const DepthSnapshot depth = depth_snapshot();
        std::ostringstream out;
        out << std::boolalpha << "{\"ok\":" << (depth_valid_.load() && depth_fresh())
            << ",\"camera_model\":\"hp60c\",\"width\":" << depth.width
            << ",\"height\":" << depth.height << ",\"frame_id\":" << depth.frame_id
            << ",\"age_ms\":" << age_ms(last_depth_ms_)
            << ",\"unit\":\"mm\",\"depth_aligned_to_color\":" << depth_aligned_to_color_ << "}";
        return out.str();
    }

    static std::vector<double> parse_numbers_after_points(const std::string &body) {
        const size_t start = body.find("points");
        const std::string input = start == std::string::npos ? body : body.substr(start);
        std::vector<double> values;
        const char *cursor = input.c_str();
        while (*cursor) {
            char *end = nullptr;
            const double value = std::strtod(cursor, &end);
            if (end != cursor) {
                values.push_back(value);
                cursor = end;
            } else {
                ++cursor;
            }
        }
        return values;
    }

    std::pair<int, std::string> deproject_json(const std::string &body) const {
        if (!camera_connected()) return {503, "{\"ok\":false,\"error\":\"camera frame stale\"}"};
        if (!intrinsics_ready()) {
            return {503, "{\"ok\":false,\"error\":\"HP60C intrinsics are not configured; set VISIONOPS_HP60C_FX/FY/CX/CY\"}"};
        }
        const std::vector<double> values = parse_numbers_after_points(body);
        if (values.size() < 3 || values.size() % 3 != 0) return {400, "{\"ok\":false,\"error\":\"points must be [[u,v,depth_mm],...]\"}"};
        std::ostringstream out;
        out << std::boolalpha << std::fixed << std::setprecision(3)
            << "{\"ok\":true,\"camera_model\":\"hp60c\",\"coordinate_frame\":\"color_camera\",\"unit\":\"mm\",\"points\":[";
        for (size_t index = 0; index < values.size(); index += 3) {
            if (index) out << ',';
            const double u = values[index], v = values[index + 1], z = values[index + 2];
            const bool valid = z > 0.0;
            const FrameSnapshot frame = frame_snapshot();
            const double width = static_cast<double>(frame.width > 0 ? frame.width : color_width_hint_);
            const double height = static_cast<double>(frame.height > 0 ? frame.height : color_height_hint_);
            // fx/fy/cx/cy describe the unflipped camera image. Convert the displayed
            // pixel back to the sensor pixel before pinhole deprojection.
            const double sensor_u = flip_horizontal_ ? (width - 1.0 - u) : u;
            const double sensor_v = flip_vertical_ ? (height - 1.0 - v) : v;
            const double x = valid ? (sensor_u - cx_) * z / fx_ : 0.0;
            const double y = valid ? (sensor_v - cy_) * z / fy_ : 0.0;
            out << "{\"valid\":" << valid << ",\"position_camera\":[" << x << ',' << y << ',' << (valid ? z : 0.0) << "]}";
        }
        out << "]}";
        return {200, out.str()};
    }

    bool read_request(int fd, std::string &method, std::string &path, std::string &body) {
        std::string request;
        char buffer[4096];
        size_t content_length = 0;
        while (request.find("\r\n\r\n") == std::string::npos && request.size() < 1024 * 1024) {
            const ssize_t count = ::recv(fd, buffer, sizeof(buffer), 0);
            if (count <= 0) return false;
            request.append(buffer, static_cast<size_t>(count));
        }
        const size_t header_end = request.find("\r\n\r\n");
        if (header_end == std::string::npos) return false;
        const size_t first_end = request.find("\r\n");
        std::istringstream first(request.substr(0, first_end));
        first >> method >> path;
        std::string headers = request.substr(first_end + 2, header_end - first_end - 2);
        std::istringstream header_stream(headers);
        std::string line;
        while (std::getline(header_stream, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string key = line.substr(0, colon);
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);
            if (key == "content-length") {
                try { content_length = static_cast<size_t>(std::stoul(line.substr(colon + 1))); } catch (...) { content_length = 0; }
            }
        }
        body = request.substr(header_end + 4);
        while (body.size() < content_length && body.size() < 1024 * 1024) {
            const ssize_t count = ::recv(fd, buffer, sizeof(buffer), 0);
            if (count <= 0) break;
            body.append(buffer, static_cast<size_t>(count));
        }
        const size_t query = path.find('?');
        if (query != std::string::npos) path.resize(query);
        return !method.empty() && !path.empty();
    }

    void handle_mjpeg(int fd) {
        if (!camera_connected()) {
            send_json_response(fd, 503, "{\"ok\":false,\"error\":\"camera frame stale\"}");
            ::close(fd);
            return;
        }
        const std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Cache-Control: no-cache, no-store, must-revalidate\r\n"
            "Pragma: no-cache\r\n"
            "Connection: close\r\n\r\n";
        if (!send_string(fd, header)) { ::close(fd); return; }
        const auto frame_period = std::chrono::microseconds(
            std::max<int64_t>(1, 1000000LL / std::max(1, mjpeg_fps_)));
        auto next_send_at = std::chrono::steady_clock::now();
        uint64_t last_id = 0;
        while (!stopped_.load() && g_running.load() && camera_connected()) {
            const auto now = std::chrono::steady_clock::now();
            if (now < next_send_at) {
                std::this_thread::sleep_until(next_send_at);
            }

            FrameSnapshot frame;
            {
                std::unique_lock<std::mutex> lock(frame_mutex_);
                frame_cv_.wait_for(lock, std::chrono::milliseconds(500), [&]() {
                    return stopped_.load() || !frame_valid_.load() || latest_.frame_id != last_id;
                });
                frame = latest_;
            }
            if (!camera_connected() || frame.jpeg.empty()) break;
            if (frame.frame_id == last_id) continue;
            last_id = frame.frame_id;
            std::ostringstream part;
            part << "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " << frame.jpeg.size() << "\r\n\r\n";
            if (!send_string(fd, part.str()) || send_all(fd, frame.jpeg.data(), frame.jpeg.size()) <= 0 || !send_string(fd, "\r\n")) break;

            next_send_at += frame_period;
            const auto after_send = std::chrono::steady_clock::now();
            if (next_send_at < after_send) {
                // Sending time already consumed the period; do not add a full sleep.
                next_send_at = after_send;
            }
        }
        ::close(fd);
    }

    void handle_client(int fd) {
        std::string method, path, body;
        if (!read_request(fd, method, path, body)) { ::close(fd); return; }
        if (path == "/health" || path == "/stream/status") {
            send_json_response(fd, 200, health_json());
        } else if (path == "/stream/profiles") {
            send_json_response(fd, 200, profiles_json());
        } else if (path == "/stream/camera_info") {
            send_json_response(fd, 200, camera_info_json());
        } else if (path == "/stream/depth_meta") {
            send_json_response(fd, depth_fresh() ? 200 : 503, depth_meta_json());
        } else if (path == "/stream/snapshot.jpg") {
            const FrameSnapshot frame = frame_snapshot();
            if (!camera_connected() || frame.jpeg.empty()) send_json_response(fd, 503, "{\"ok\":false,\"error\":\"camera frame stale\"}");
            else send_bytes_response(fd, 200, "image/jpeg", frame.jpeg.data(), frame.jpeg.size());
        } else if (path == "/stream/depth.png") {
            const DepthSnapshot depth = depth_snapshot();
            if (!depth_fresh() || depth.png.empty()) send_json_response(fd, 503, "{\"ok\":false,\"error\":\"depth frame stale\"}");
            else send_bytes_response(fd, 200, "image/png", depth.png.data(), depth.png.size());
        } else if (path == "/stream/depth_vis.jpg") {
            const DepthSnapshot depth = depth_snapshot();
            if (!depth_fresh() || depth.visual_jpeg.empty()) send_json_response(fd, 503, "{\"ok\":false,\"error\":\"depth frame stale\"}");
            else send_bytes_response(fd, 200, "image/jpeg", depth.visual_jpeg.data(), depth.visual_jpeg.size());
        } else if (path == "/stream.mjpeg" || path == "/stream/mjpeg" || path == "/stream.mjpg") {
            handle_mjpeg(fd);
            return;
        } else if (path == "/api/coordinate/deproject" && method == "POST") {
            const auto result = deproject_json(body);
            send_json_response(fd, result.first, result.second);
        } else if ((path == "/stream/start" || path == "/stream/stop") && method == "POST") {
            send_json_response(fd, 200, "{\"ok\":true,\"message\":\"HP60C SDK bridge keeps camera streaming\"}");
        } else {
            send_json_response(fd, 404, "{\"ok\":false,\"error\":\"not found\"}");
        }
        ::close(fd);
    }

    void http_loop() {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) { set_state("offline", "socket failed"); return; }
        int reuse = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<uint16_t>(http_port_));
        if (bind_host_ == "0.0.0.0" || bind_host_.empty()) address.sin_addr.s_addr = INADDR_ANY;
        else if (::inet_pton(AF_INET, bind_host_.c_str(), &address.sin_addr) != 1) address.sin_addr.s_addr = INADDR_LOOPBACK;
        if (::bind(listen_fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 || ::listen(listen_fd_, 32) != 0) {
            set_state("offline", "HTTP bind/listen failed on port " + std::to_string(http_port_));
            ::close(listen_fd_);
            listen_fd_ = -1;
            return;
        }
        std::cerr << "[INFO] VisionOps HP60C Bridge HTTP listening on " << bind_host_ << ':' << http_port_ << std::endl;
        while (!stopped_.load() && g_running.load()) {
            sockaddr_in peer{};
            socklen_t peer_len = sizeof(peer);
            const int client = ::accept(listen_fd_, reinterpret_cast<sockaddr *>(&peer), &peer_len);
            if (client < 0) {
                if (stopped_.load() || !g_running.load()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            std::thread(&Hp60cSdkBridge::handle_client, this, client).detach();
        }
    }

private:
    std::string config_path_;
    std::string bind_host_;
    int http_port_;
    int jpeg_quality_;
    int mjpeg_fps_;
    bool flip_vertical_;
    bool flip_horizontal_;
    std::string rgb_source_;
    std::string rgb_order_;
    int color_width_hint_;
    int color_height_hint_;
    int depth_width_hint_;
    int depth_height_hint_;
    int camera_fps_hint_;
    int stale_timeout_ms_;
    int first_frame_timeout_ms_;
    int reconnect_initial_ms_;
    int reconnect_max_ms_;
    int alarm_after_ms_;
    bool depth_aligned_to_color_;
    double fx_, fy_, cx_, cy_;

    std::atomic<bool> stopped_{false};
    std::atomic<bool> camera_opened_{false};
    std::atomic<bool> camera_setup_in_progress_{false};
    std::atomic<bool> reconnect_requested_{true};
    std::atomic<bool> recovery_shutdown_{false};
    std::atomic<bool> recovery_thread_alive_{false};
    std::atomic<bool> frame_valid_{false};
    std::atomic<bool> depth_valid_{false};
    std::atomic<uint64_t> frame_count_{0};
    std::atomic<uint64_t> depth_frame_count_{0};
    std::atomic<uint64_t> reconnect_attempt_count_{0};
    std::atomic<uint64_t> reconnect_success_count_{0};
    std::atomic<int64_t> last_color_ms_{-1};
    std::atomic<int64_t> last_depth_ms_{-1};
    std::atomic<int64_t> last_reconnect_epoch_ms_{0};
    std::atomic<int64_t> incident_started_epoch_ms_{0};

    bool sdk_initialized_ = false;
    bool listener_started_ = false;
    int listen_fd_ = -1;
    std::thread http_thread_;
    std::thread recovery_thread_;

    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    FrameSnapshot latest_;
    mutable std::mutex depth_mutex_;
    DepthSnapshot latest_depth_;
    std::mutex camera_mutex_;
    std::list<AS_CAM_PTR> cameras_;
    std::mutex recovery_mutex_;
    std::condition_variable recovery_cv_;
    mutable std::mutex state_mutex_;
    std::string camera_state_ = "starting";
    std::string last_error_;
    int cam_model_ = 0;
    std::string serial_;
    std::string sdk_version_;
};

void signal_handler(int) {
    g_running = false;
}

}  // namespace

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    Hp60cSdkBridge bridge;
    if (!bridge.start()) return 1;
    while (g_running.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    bridge.stop();
    return 0;
}
