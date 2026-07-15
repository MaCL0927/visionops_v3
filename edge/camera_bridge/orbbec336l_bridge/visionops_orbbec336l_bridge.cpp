// VisionOps Orbbec Gemini 336L SDK bridge
// HTTP endpoints:
//   GET  /health
//   GET  /stream/status
//   GET  /stream/snapshot.jpg
//   GET  /stream/depth.png       16-bit PNG depth, millimeters when scale is available
//   GET  /stream/depth_vis.jpg   visualized depth JPEG
//   GET  /stream/depth_meta
//   GET  /stream/camera_info
//   POST /api/coordinate/deproject  {"points":[[u,v,depth_mm], ...]}
//   GET  /stream/profiles     SDK-supported color/depth profiles
//   GET  /stream.mjpeg, /stream/mjpeg, /stream.mjpg
//   POST /stream/start, /stream/stop  compatibility no-op endpoints

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "libobsensor/ObSensor.hpp"
#include "libobsensor/hpp/Utils.hpp"

namespace {

std::atomic<bool> g_running{true};
std::atomic<int> g_server_fd{-1};

static std::string getenv_str(const char *name, const std::string &fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::string(v);
}

static int getenv_int(const char *name, int fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) return fallback;
    try { return std::stoi(v); } catch (...) { return fallback; }
}

static bool getenv_bool(const char *name, bool fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) return fallback;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s == "1" || s == "true" || s == "yes" || s == "on");
}

static void prepare_runtime_workdir() {
    std::string dir = getenv_str("VISIONOPS_ORBBEC336L_RUNTIME_DIR", "/run/visionops-orbbec336l-bridge");
    if (dir.empty()) return;
    ::mkdir(dir.c_str(), 0755);
    if (::chdir(dir.c_str()) != 0) {
        std::cerr << "[WARN] failed to chdir to runtime dir " << dir << ": " << std::strerror(errno) << std::endl;
    }
}

static std::string now_string() {
    char buf[64] = {0};
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
    localtime_r(&t, &tmv);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
    return std::string(buf);
}

static int64_t epoch_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string json_escape(const std::string &s) {
    std::ostringstream os;
    for (char c : s) {
        switch (c) {
        case '\\': os << "\\\\"; break;
        case '"': os << "\\\""; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default: os << c; break;
        }
    }
    return os.str();
}

static bool encode_jpeg(const cv::Mat &bgr, int quality, std::vector<uchar> &out) {
    if (bgr.empty()) return false;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, std::max(1, std::min(100, quality))};
    return cv::imencode(".jpg", bgr, out, params);
}

static bool encode_png16(const cv::Mat &u16, std::vector<uchar> &out) {
    if (u16.empty() || u16.type() != CV_16UC1) return false;
    std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 3};
    return cv::imencode(".png", u16, out, params);
}

static cv::Mat depth_to_vis(const cv::Mat &depth_mm) {
    if (depth_mm.empty()) return cv::Mat();
    double minv = 0.0, maxv = 0.0;
    cv::Mat mask = depth_mm > 0;
    if (cv::countNonZero(mask) <= 0) {
        return cv::Mat(depth_mm.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    }
    cv::minMaxLoc(depth_mm, &minv, &maxv, nullptr, nullptr, mask);
    if (maxv <= minv) maxv = minv + 1.0;
    cv::Mat u8;
    depth_mm.convertTo(u8, CV_8U, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
    u8.setTo(0, ~mask);
    cv::Mat color;
    cv::applyColorMap(u8, color, cv::COLORMAP_JET);
    color.setTo(cv::Scalar(0, 0, 0), ~mask);
    return color;
}

static std::string frame_format_to_string(OBFormat f) {
    switch (f) {
    case OB_FORMAT_RGB: return "RGB";
    case OB_FORMAT_BGR: return "BGR";
    case OB_FORMAT_MJPG: return "MJPG";
    case OB_FORMAT_YUYV: return "YUYV";
    case OB_FORMAT_NV12: return "NV12";
    case OB_FORMAT_NV21: return "NV21";
    case OB_FORMAT_Y16: return "Y16";
    case OB_FORMAT_Y8: return "Y8";
    default: return "UNKNOWN";
    }
}

class OrbbecBridge {
private:
    enum class CameraState { Starting, Running, Stale, Reconnecting, Offline, Stopping };

public:
    OrbbecBridge()
        : http_host_(getenv_str("VISIONOPS_ORBBEC336L_HTTP_HOST", "127.0.0.1")),
          http_port_(getenv_int("VISIONOPS_ORBBEC336L_HTTP_PORT", 18182)),
          color_width_(getenv_int("VISIONOPS_ORBBEC336L_COLOR_WIDTH", 640)),
          color_height_(getenv_int("VISIONOPS_ORBBEC336L_COLOR_HEIGHT", 480)),
          depth_width_(getenv_int("VISIONOPS_ORBBEC336L_DEPTH_WIDTH", 640)),
          depth_height_(getenv_int("VISIONOPS_ORBBEC336L_DEPTH_HEIGHT", 480)),
          fps_(getenv_int("VISIONOPS_ORBBEC336L_FPS", 30)),
          jpeg_quality_(getenv_int("VISIONOPS_ORBBEC336L_JPEG_QUALITY", 85)),
          mjpeg_fps_(getenv_int("VISIONOPS_ORBBEC336L_MJPEG_FPS", 10)),
          stale_timeout_ms_(std::max(500, getenv_int("VISIONOPS_ORBBEC336L_STALE_TIMEOUT_MS", 3000))),
          first_frame_timeout_ms_(std::max(1000, getenv_int("VISIONOPS_ORBBEC336L_FIRST_FRAME_TIMEOUT_MS", 5000))),
          reconnect_initial_ms_(std::max(100, getenv_int("VISIONOPS_ORBBEC336L_RECONNECT_INITIAL_MS", 1000))),
          reconnect_max_ms_(std::max(
              std::max(100, getenv_int("VISIONOPS_ORBBEC336L_RECONNECT_INITIAL_MS", 1000)),
              getenv_int("VISIONOPS_ORBBEC336L_RECONNECT_MAX_MS", 30000))),
          reconnect_alarm_ms_(std::max(1000, getenv_int("VISIONOPS_ORBBEC336L_RECONNECT_FAILURE_ALARM_SEC", 15) * 1000)),
          flip_vertical_(getenv_bool("VISIONOPS_ORBBEC336L_FLIP_VERTICAL", false)),
          flip_horizontal_(getenv_bool("VISIONOPS_ORBBEC336L_FLIP_HORIZONTAL", false)),
          serial_(getenv_str("VISIONOPS_ORBBEC336L_SERIAL", "")) {}

    bool start_camera() {
        start_ms_ = now_ms();
        camera_stop_requested_ = false;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            set_camera_state_locked(CameraState::Starting, "CAMERA_STARTING", "waiting for Orbbec camera");
        }
        camera_thread_ = std::thread([this]() { this->camera_loop(); });
        return true;
    }

    void stop_camera() {
        camera_stop_requested_ = true;
        cv_.notify_all();
        if (camera_thread_.joinable()) camera_thread_.join();
        camera_started_ = false;
    }

    bool start_http() {
        server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        g_server_fd.store(server_fd_);
        if (server_fd_ < 0) {
            std::cerr << "[FATAL] socket failed" << std::endl;
            return false;
        }
        int yes = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(http_port_);
        if (http_host_ == "0.0.0.0") {
            addr.sin_addr.s_addr = INADDR_ANY;
        } else {
            if (inet_pton(AF_INET, http_host_.c_str(), &addr.sin_addr) != 1) {
                std::cerr << "[FATAL] invalid host: " << http_host_ << std::endl;
                return false;
            }
        }
        if (::bind(server_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
            std::cerr << "[FATAL] bind failed on port " << http_port_ << ": " << std::strerror(errno) << std::endl;
            g_server_fd.store(-1);
            return false;
        }
        if (::listen(server_fd_, 16) < 0) {
            std::cerr << "[FATAL] listen failed" << std::endl;
            return false;
        }
        std::cerr << "[INFO] HTTP listening on " << http_host_ << ":" << http_port_ << std::endl;
        while (g_running) {
            sockaddr_in cli{};
            socklen_t len = sizeof(cli);
            int fd = ::accept(server_fd_, reinterpret_cast<sockaddr *>(&cli), &len);
            if (fd < 0) {
                if (errno == EINTR) continue;
                if (!g_running) break;
                continue;
            }
            std::thread(&OrbbecBridge::handle_client, this, fd).detach();
        }
        g_server_fd.store(-1);
        return true;
    }

private:
    static int64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    std::shared_ptr<ob::VideoStreamProfile> enable_color_stream(
        const std::shared_ptr<ob::Pipeline> &pipeline,
        const std::shared_ptr<ob::Config> &cfg) {
        auto profiles = pipeline->getStreamProfileList(OB_SENSOR_COLOR);
        std::shared_ptr<ob::VideoStreamProfile> profile;
        try {
            if (color_width_ > 0 && color_height_ > 0) {
                profile = profiles->getVideoStreamProfile(color_width_, color_height_, OB_FORMAT_RGB, fps_);
            }
        } catch (...) {}
        if (!profile) {
            try {
                profile = profiles->getVideoStreamProfile(color_width_, color_height_, OB_FORMAT_MJPG, fps_);
            } catch (...) {}
        }
        if (!profile) profile = profiles->getProfile(0)->as<ob::VideoStreamProfile>();
        if (!profile) throw std::runtime_error("no usable color profile");
        cfg->enableStream(profile);
        std::cerr << "[INFO] color profile " << profile->width() << "x" << profile->height()
                  << " fps=" << profile->fps() << " fmt=" << frame_format_to_string(profile->format()) << std::endl;
        return profile;
    }

    std::shared_ptr<ob::VideoStreamProfile> enable_depth_stream(
        const std::shared_ptr<ob::Pipeline> &pipeline,
        const std::shared_ptr<ob::Config> &cfg) {
        auto profiles = pipeline->getStreamProfileList(OB_SENSOR_DEPTH);
        std::shared_ptr<ob::VideoStreamProfile> profile;
        try {
            if (depth_width_ > 0 && depth_height_ > 0) {
                profile = profiles->getVideoStreamProfile(depth_width_, depth_height_, OB_FORMAT_Y16, fps_);
            }
        } catch (...) {}
        if (!profile) profile = profiles->getProfile(0)->as<ob::VideoStreamProfile>();
        if (!profile) throw std::runtime_error("no usable depth profile");
        cfg->enableStream(profile);
        std::cerr << "[INFO] depth profile " << profile->width() << "x" << profile->height()
                  << " fps=" << profile->fps() << " fmt=" << frame_format_to_string(profile->format()) << std::endl;
        return profile;
    }

    cv::Mat color_frame_to_bgr(const std::shared_ptr<ob::ColorFrame> &frame) {
        if (!frame) return cv::Mat();
        int w = static_cast<int>(frame->width());
        int h = static_cast<int>(frame->height());
        OBFormat fmt = frame->format();
        const uint8_t *data = reinterpret_cast<const uint8_t *>(frame->data());
        if (!data || w <= 0 || h <= 0) return cv::Mat();
        cv::Mat bgr;
        try {
            if (fmt == OB_FORMAT_RGB) {
                cv::Mat rgb(h, w, CV_8UC3, const_cast<uint8_t *>(data));
                cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
            } else if (fmt == OB_FORMAT_BGR) {
                bgr = cv::Mat(h, w, CV_8UC3, const_cast<uint8_t *>(data)).clone();
            } else if (fmt == OB_FORMAT_MJPG) {
                std::vector<uint8_t> buf(data, data + frame->dataSize());
                bgr = cv::imdecode(buf, cv::IMREAD_COLOR);
            } else if (fmt == OB_FORMAT_YUYV) {
                cv::Mat yuyv(h, w, CV_8UC2, const_cast<uint8_t *>(data));
                cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);
            } else if (fmt == OB_FORMAT_NV12) {
                cv::Mat yuv(h + h / 2, w, CV_8UC1, const_cast<uint8_t *>(data));
                cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
            } else if (fmt == OB_FORMAT_NV21) {
                cv::Mat yuv(h + h / 2, w, CV_8UC1, const_cast<uint8_t *>(data));
                cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV21);
            } else if (fmt == OB_FORMAT_Y8) {
                cv::Mat gray(h, w, CV_8UC1, const_cast<uint8_t *>(data));
                cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
            }
        } catch (const std::exception &e) {
            std::cerr << "[WARN] color conversion failed: " << e.what() << std::endl;
        }
        if (!bgr.empty()) apply_flips(bgr);
        return bgr;
    }

    cv::Mat depth_frame_to_mm(const std::shared_ptr<ob::DepthFrame> &frame) {
        if (!frame) return cv::Mat();
        int w = static_cast<int>(frame->width());
        int h = static_cast<int>(frame->height());
        const uint16_t *data = reinterpret_cast<const uint16_t *>(frame->data());
        if (!data || w <= 0 || h <= 0) return cv::Mat();
        cv::Mat raw(h, w, CV_16UC1, const_cast<uint16_t *>(data));
        cv::Mat mm;
        float scale = 1.0f;
        try { scale = frame->getValueScale(); } catch (...) { scale = 1.0f; }
        if (scale > 0.0f && scale != 1.0f) {
            cv::Mat f32;
            raw.convertTo(f32, CV_32F, scale);
            f32.convertTo(mm, CV_16U);
        } else {
            mm = raw.clone();
        }
        if (!mm.empty()) apply_flips(mm);
        depth_scale_ = scale;
        return mm;
    }

    void apply_flips(cv::Mat &m) {
        if (flip_vertical_ && flip_horizontal_) cv::flip(m, m, -1);
        else if (flip_vertical_) cv::flip(m, m, 0);
        else if (flip_horizontal_) cv::flip(m, m, 1);
    }

    static const char *camera_state_name(CameraState state) {
        switch (state) {
        case CameraState::Starting: return "starting";
        case CameraState::Running: return "running";
        case CameraState::Stale: return "stale";
        case CameraState::Reconnecting: return "reconnecting";
        case CameraState::Offline: return "offline";
        case CameraState::Stopping: return "stopping";
        }
        return "offline";
    }

    bool run_requested() const {
        return g_running.load() && !camera_stop_requested_.load();
    }

    void set_camera_state_locked(CameraState state, const std::string &fault_code, const std::string &message) {
        const int64_t now = now_ms();
        const int64_t epoch = epoch_now_ms();
        if (camera_state_ != state) {
            camera_state_ = state;
            state_since_ms_ = now;
            state_since_epoch_ms_ = epoch;
        }
        if (state == CameraState::Running) {
            unhealthy_since_ms_ = 0;
            unhealthy_since_epoch_ms_ = 0;
        } else if (unhealthy_since_ms_ == 0) {
            unhealthy_since_ms_ = now;
            unhealthy_since_epoch_ms_ = epoch;
        }
        fault_code_ = fault_code;
        if (!message.empty()) last_error_ = message;
    }

    void set_camera_state(CameraState state, const std::string &fault_code, const std::string &message) {
        std::lock_guard<std::mutex> lk(mtx_);
        set_camera_state_locked(state, fault_code, message);
    }

    void record_error(const std::string &message) {
        std::lock_guard<std::mutex> lk(mtx_);
        last_error_ = message;
        last_error_epoch_ms_ = epoch_now_ms();
    }

    bool color_fresh_locked(int64_t now) const {
        return !latest_bgr_.empty() && last_color_ms_ > 0 && now - last_color_ms_ <= stale_timeout_ms_;
    }

    bool depth_fresh_locked(int64_t now) const {
        return !latest_depth_mm_.empty() && last_depth_ms_ > 0 && now - last_depth_ms_ <= stale_timeout_ms_;
    }

    bool camera_connected_locked(int64_t now) const {
        return camera_started_.load() && camera_state_ == CameraState::Running
            && color_fresh_locked(now) && depth_fresh_locked(now);
    }

    void invalidate_frames_locked() {
        latest_bgr_.release();
        latest_depth_mm_.release();
        last_color_ms_ = 0;
        last_depth_ms_ = 0;
        calibration_ready_ = false;
        color_w_ = 0;
        color_h_ = 0;
        depth_w_ = 0;
        depth_h_ = 0;
    }

    void sleep_interruptible(int delay_ms) {
        int waited = 0;
        while (run_requested() && waited < delay_ms) {
            const int chunk = std::min(100, delay_ms - waited);
            std::this_thread::sleep_for(std::chrono::milliseconds(chunk));
            waited += chunk;
        }
    }

    std::shared_ptr<ob::Pipeline> open_pipeline() {
        std::shared_ptr<ob::Context> context;
        std::shared_ptr<ob::Pipeline> pipeline;
        if (!serial_.empty()) {
            context = std::make_shared<ob::Context>();
            auto dev_list = context->queryDeviceList();
            bool found = false;
            for (uint32_t i = 0; i < dev_list->deviceCount(); ++i) {
                auto dev = dev_list->getDevice(i);
                auto info = dev->getDeviceInfo();
                std::string sn = info ? info->serialNumber() : "";
                if (sn == serial_) {
                    pipeline = std::make_shared<ob::Pipeline>(dev);
                    found = true;
                    break;
                }
            }
            if (!found) throw std::runtime_error("Orbbec serial not found: " + serial_);
        } else {
            // Reconstructing Pipeline is intentional: after a USB unplug, the old
            // SDK device handle is no longer reusable even when the cable is reinserted.
            pipeline = std::make_shared<ob::Pipeline>();
        }

        auto cfg = std::make_shared<ob::Config>();
        auto color_profile = enable_color_stream(pipeline, cfg);
        auto depth_profile = enable_depth_stream(pipeline, cfg);
        cfg->setAlignMode(ALIGN_D2C_SW_MODE);
        pipeline->start(cfg);
        auto calibration = pipeline->getCalibrationParam(cfg);

        {
            std::lock_guard<std::mutex> lk(mtx_);
            ctx_ = context;
            pipeline_ = pipeline;
            color_profile_ = color_profile;
            depth_profile_ = depth_profile;
            calibration_param_ = calibration;
            calibration_ready_ = true;
            camera_started_ = true;
            pipeline_started_ms_ = now_ms();
            invalidate_frames_locked();
            // invalidate_frames_locked clears calibration_ready_; restore it after
            // old image/depth buffers have been invalidated.
            calibration_param_ = calibration;
            calibration_ready_ = true;
            set_camera_state_locked(
                ever_connected_ ? CameraState::Reconnecting : CameraState::Starting,
                ever_connected_ ? "CAMERA_RECONNECTING" : "CAMERA_STARTING",
                ever_connected_ ? "pipeline started; waiting for fresh RGB/depth frames" : "waiting for first RGB/depth frames");
        }
        cv_.notify_all();
        return pipeline;
    }

    void close_pipeline(const std::shared_ptr<ob::Pipeline> &pipeline) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (pipeline_ == pipeline) {
                pipeline_.reset();
                ctx_.reset();
                color_profile_.reset();
                depth_profile_.reset();
                calibration_ready_ = false;
                camera_started_ = false;
                invalidate_frames_locked();
            }
        }
        cv_.notify_all();
        if (pipeline) {
            try {
                // Some SDK/USB failures can block here. The external systemd
                // watchdog is the final safety net and will kill/restart the process.
                pipeline->stop();
            } catch (const ob::Error &e) {
                record_error(std::string("pipeline stop failed: ") + e.getMessage());
            } catch (const std::exception &e) {
                record_error(std::string("pipeline stop failed: ") + e.what());
            } catch (...) {
                record_error("pipeline stop failed: unknown error");
            }
        }
    }

    void process_frames(const std::shared_ptr<ob::FrameSet> &frames) {
        if (!frames) return;
        cv::Mat bgr;
        cv::Mat depth_mm;
        auto c = frames->colorFrame();
        auto d = frames->depthFrame();
        if (c) bgr = color_frame_to_bgr(c);
        if (d) depth_mm = depth_frame_to_mm(d);
        const auto now = now_ms();
        bool became_running = false;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (!bgr.empty()) {
                latest_bgr_ = bgr;
                last_color_ms_ = now;
                ++color_frame_count_;
                color_format_ = frame_format_to_string(c->format());
                color_w_ = bgr.cols;
                color_h_ = bgr.rows;
            }
            if (!depth_mm.empty()) {
                latest_depth_mm_ = depth_mm;
                last_depth_ms_ = now;
                ++depth_frame_count_;
                depth_w_ = depth_mm.cols;
                depth_h_ = depth_mm.rows;
            }
            ++frame_count_;
            if (color_fresh_locked(now) && depth_fresh_locked(now) && camera_state_ != CameraState::Running) {
                set_camera_state_locked(CameraState::Running, "", "");
                last_error_.clear();
                fault_code_.clear();
                ever_connected_ = true;
                ++reconnect_success_count_;
                consecutive_reconnect_failures_ = 0;
                last_reconnect_success_epoch_ms_ = epoch_now_ms();
                became_running = true;
            }
        }
        if (became_running) {
            std::cerr << "[INFO] Orbbec RGB/depth frames healthy; camera running" << std::endl;
        }
        cv_.notify_all();
    }

    void camera_loop() {
        camera_thread_alive_ = true;
        int backoff_ms = reconnect_initial_ms_;
        while (run_requested()) {
            std::shared_ptr<ob::Pipeline> pipeline;
            bool reached_running = false;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                ++reconnect_attempt_count_;
                last_reconnect_attempt_epoch_ms_ = epoch_now_ms();
                set_camera_state_locked(
                    ever_connected_ ? CameraState::Reconnecting : CameraState::Starting,
                    ever_connected_ ? "CAMERA_RECONNECTING" : "CAMERA_STARTING",
                    ever_connected_ ? "rebuilding Orbbec pipeline" : "opening Orbbec camera");
            }
            try {
                pipeline = open_pipeline();
                std::cerr << "[INFO] Orbbec pipeline opened; waiting for RGB/depth frames" << std::endl;
                while (run_requested()) {
                    std::shared_ptr<ob::FrameSet> frames;
                    try {
                        frames = pipeline->waitForFrames(1000);
                        if (frames) process_frames(frames);
                    } catch (const ob::Error &e) {
                        record_error(std::string("wait frame failed: ") + e.getMessage());
                        std::cerr << "[WARN] Orbbec wait frame error: " << e.getMessage() << std::endl;
                    } catch (const std::exception &e) {
                        record_error(std::string("camera loop failed: ") + e.what());
                        std::cerr << "[WARN] camera loop error: " << e.what() << std::endl;
                    }

                    const int64_t now = now_ms();
                    bool stale = false;
                    std::string stale_message;
                    {
                        std::lock_guard<std::mutex> lk(mtx_);
                        reached_running = reached_running || camera_state_ == CameraState::Running;
                        const int64_t color_age = last_color_ms_ > 0 ? now - last_color_ms_ : now - pipeline_started_ms_;
                        const int64_t depth_age = last_depth_ms_ > 0 ? now - last_depth_ms_ : now - pipeline_started_ms_;
                        const int64_t limit = reached_running ? stale_timeout_ms_ : first_frame_timeout_ms_;
                        stale = color_age > limit || depth_age > limit;
                        if (stale) {
                            std::ostringstream message;
                            message << "RGB/depth frame stale: color_age_ms=" << color_age
                                    << ", depth_age_ms=" << depth_age << ", limit_ms=" << limit;
                            stale_message = message.str();
                            ++stale_event_count_;
                            set_camera_state_locked(CameraState::Stale, "CAMERA_FRAME_STALE", stale_message);
                            invalidate_frames_locked();
                        }
                    }
                    if (stale) {
                        std::cerr << "[WARN] " << stale_message << "; rebuilding pipeline" << std::endl;
                        cv_.notify_all();
                        break;
                    }
                }
            } catch (const ob::Error &e) {
                record_error(std::string("camera open failed: ") + e.getMessage());
                std::cerr << "[WARN] Orbbec camera open failed: " << e.getMessage() << std::endl;
            } catch (const std::exception &e) {
                record_error(std::string("camera open failed: ") + e.what());
                std::cerr << "[WARN] Orbbec camera open failed: " << e.what() << std::endl;
            } catch (...) {
                record_error("camera open failed: unknown error");
                std::cerr << "[WARN] Orbbec camera open failed: unknown error" << std::endl;
            }

            close_pipeline(pipeline);
            if (!run_requested()) break;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                ++consecutive_reconnect_failures_;
                set_camera_state_locked(
                    reached_running ? CameraState::Reconnecting : CameraState::Offline,
                    reached_running ? "CAMERA_RECONNECTING" : "CAMERA_OFFLINE",
                    last_error_.empty() ? "camera unavailable; retry scheduled" : last_error_);
            }
            cv_.notify_all();
            std::cerr << "[WARN] camera unavailable; retry in " << backoff_ms << " ms" << std::endl;
            sleep_interruptible(backoff_ms);
            if (reached_running) backoff_ms = reconnect_initial_ms_;
            else backoff_ms = std::min(reconnect_max_ms_, std::max(reconnect_initial_ms_, backoff_ms * 2));
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            set_camera_state_locked(CameraState::Stopping, "", "camera thread stopped");
            camera_started_ = false;
            invalidate_frames_locked();
        }
        cv_.notify_all();
        camera_thread_alive_ = false;
    }

    struct DeprojectInput {
        float u = 0.0f;
        float v = 0.0f;
        float depth_mm = 0.0f;
    };

    static bool parse_deproject_points(const std::string &body, std::vector<DeprojectInput> &points, std::string &error) {
        const auto key = body.find("\"points\"");
        if (key == std::string::npos) {
            error = "missing points";
            return false;
        }
        const auto begin = body.find('[', key);
        if (begin == std::string::npos) {
            error = "points must be array";
            return false;
        }
        int nesting = 0;
        size_t end = std::string::npos;
        for (size_t i = begin; i < body.size(); ++i) {
            if (body[i] == '[') ++nesting;
            else if (body[i] == ']') {
                --nesting;
                if (nesting == 0) {
                    end = i;
                    break;
                }
            }
        }
        if (end == std::string::npos) {
            error = "unterminated points array";
            return false;
        }
        std::vector<double> values;
        const char *cursor = body.c_str() + begin + 1;
        const char *limit = body.c_str() + end;
        while (cursor < limit) {
            while (cursor < limit && !((*cursor >= '0' && *cursor <= '9') || *cursor == '-' || *cursor == '+' || *cursor == '.')) ++cursor;
            if (cursor >= limit) break;
            char *next = nullptr;
            errno = 0;
            double value = std::strtod(cursor, &next);
            if (next == cursor || errno == ERANGE || !std::isfinite(value)) {
                error = "invalid number in points";
                return false;
            }
            values.push_back(value);
            cursor = next;
        }
        if (values.empty() || values.size() % 3 != 0) {
            error = "each point must contain u,v,depth_mm";
            return false;
        }
        if (values.size() / 3 > 512) {
            error = "too many points";
            return false;
        }
        for (size_t i = 0; i < values.size(); i += 3) {
            points.push_back(DeprojectInput{
                static_cast<float>(values[i]),
                static_cast<float>(values[i + 1]),
                static_cast<float>(values[i + 2]),
            });
        }
        return true;
    }

    std::string camera_info_json() {
        std::shared_ptr<ob::VideoStreamProfile> profile;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            profile = color_profile_;
        }
        if (!profile) return "{\"ok\":false,\"error\":\"color profile unavailable\"}";
        try {
            auto intrinsic = profile->getIntrinsic();
            std::ostringstream os;
            os << "{"
               << "\"ok\":true,"
               << "\"coordinate_frame\":\"color_camera\","
               << "\"unit\":\"mm\","
               << "\"depth_aligned_to_color\":true,"
               << "\"color_intrinsic\":{"
               << "\"fx\":" << intrinsic.fx << ","
               << "\"fy\":" << intrinsic.fy << ","
               << "\"cx\":" << intrinsic.cx << ","
               << "\"cy\":" << intrinsic.cy << ","
               << "\"width\":" << intrinsic.width << ","
               << "\"height\":" << intrinsic.height
               << "}}";
            return os.str();
        } catch (const std::exception &e) {
            return std::string("{\"ok\":false,\"error\":\"") + json_escape(e.what()) + "\"}";
        }
    }

    std::string deproject_json(const std::string &body, int &status_code) {
        std::vector<DeprojectInput> inputs;
        std::string parse_error;
        if (!parse_deproject_points(body, inputs, parse_error)) {
            status_code = 400;
            return std::string("{\"ok\":false,\"error\":\"") + json_escape(parse_error) + "\"}";
        }
        std::shared_ptr<ob::VideoStreamProfile> profile;
        OBCalibrationParam calibration_param{};
        bool calibration_ready = false;
        bool camera_connected = false;
        int width = 0;
        int height = 0;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            profile = color_profile_;
            calibration_param = calibration_param_;
            calibration_ready = calibration_ready_;
            camera_connected = camera_connected_locked(now_ms());
            width = color_w_ > 0 ? color_w_ : color_width_;
            height = color_h_ > 0 ? color_h_ : color_height_;
        }
        if (!camera_connected) {
            status_code = 503;
            return "{\"ok\":false,\"error\":\"camera frame stale or reconnecting\"}";
        }
        if (!profile) {
            status_code = 503;
            return "{\"ok\":false,\"error\":\"color profile unavailable\"}";
        }
        if (!calibration_ready) {
            status_code = 503;
            return "{\"ok\":false,\"error\":\"camera calibration unavailable\"}";
        }
        try {
            std::ostringstream os;
            os << "{\"ok\":true,\"coordinate_frame\":\"color_camera\",\"unit\":\"mm\",\"points\":[";
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (i) os << ",";
                float u = inputs[i].u;
                float v = inputs[i].v;
                const float depth_mm = inputs[i].depth_mm;
                if (flip_horizontal_ && width > 0) u = static_cast<float>(width - 1) - u;
                if (flip_vertical_ && height > 0) v = static_cast<float>(height - 1) - v;
                OBPoint3f point3d{};
                bool valid = false;
                if (depth_mm > 0.0f && u >= 0.0f && v >= 0.0f && u < width && v < height) {
                    OBPoint2f pixel{u, v};
                    valid = ob::CoordinateTransformHelper::calibration2dTo3d(
                        calibration_param,
                        pixel,
                        depth_mm,
                        OB_SENSOR_COLOR,
                        OB_SENSOR_COLOR,
                        &point3d);
                    valid = valid && std::isfinite(point3d.x) && std::isfinite(point3d.y) && std::isfinite(point3d.z);
                }
                os << "{\"valid\":" << (valid ? "true" : "false") << ",\"position_camera\":[";
                if (valid) os << point3d.x << "," << point3d.y << "," << point3d.z;
                else os << "0,0,0";
                os << "]}";
            }
            os << "]}";
            status_code = 200;
            return os.str();
        } catch (const std::exception &e) {
            status_code = 500;
            return std::string("{\"ok\":false,\"error\":\"") + json_escape(e.what()) + "\"}";
        }
    }

    bool read_http_request(int fd, std::string &method, std::string &path, std::string &body) {
        std::string request;
        char buffer[4096];
        size_t header_end = std::string::npos;
        while ((header_end = request.find("\r\n\r\n")) == std::string::npos) {
            ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
            if (n <= 0) return false;
            request.append(buffer, static_cast<size_t>(n));
            if (request.size() > 1024 * 1024) return false;
        }
        const std::string headers_text = request.substr(0, header_end);
        std::istringstream lines(headers_text);
        std::string request_line;
        std::getline(lines, request_line);
        if (!request_line.empty() && request_line.back() == '\r') request_line.pop_back();
        std::istringstream first(request_line);
        std::string proto;
        first >> method >> path >> proto;
        size_t content_length = 0;
        std::string line;
        while (std::getline(lines, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const auto colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string key = line.substr(0, colon);
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);
            if (key == "content-length") {
                try { content_length = static_cast<size_t>(std::stoul(line.substr(colon + 1))); }
                catch (...) { return false; }
            }
        }
        if (content_length > 1024 * 1024) return false;
        body = request.substr(header_end + 4);
        while (body.size() < content_length) {
            ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
            if (n <= 0) return false;
            body.append(buffer, static_cast<size_t>(n));
        }
        if (body.size() > content_length) body.resize(content_length);
        const auto query = path.find('?');
        if (query != std::string::npos) path.resize(query);
        return !method.empty() && !path.empty();
    }

    void send_all(int fd, const void *data, size_t len) {
        const char *p = reinterpret_cast<const char *>(data);
        while (len > 0) {
            ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
            if (n <= 0) return;
            p += n;
            len -= static_cast<size_t>(n);
        }
    }

    void send_text(int fd, int code, const std::string &ctype, const std::string &body) {
        std::ostringstream h;
        h << "HTTP/1.1 " << code << " " << (code == 200 ? "OK" : code == 404 ? "Not Found" : "Error") << "\r\n"
          << "Content-Type: " << ctype << "\r\n"
          << "Content-Length: " << body.size() << "\r\n"
          << "Connection: close\r\n\r\n";
        auto hs = h.str();
        send_all(fd, hs.data(), hs.size());
        send_all(fd, body.data(), body.size());
    }

    void send_binary(int fd, int code, const std::string &ctype, const std::vector<uchar> &body) {
        std::ostringstream h;
        h << "HTTP/1.1 " << code << " OK\r\n"
          << "Content-Type: " << ctype << "\r\n"
          << "Content-Length: " << body.size() << "\r\n"
          << "Connection: close\r\n\r\n";
        auto hs = h.str();
        send_all(fd, hs.data(), hs.size());
        if (!body.empty()) send_all(fd, body.data(), body.size());
    }

    std::string status_json() {
        std::lock_guard<std::mutex> lk(mtx_);
        const int64_t now = now_ms();
        const int64_t state_age_ms = state_since_ms_ > 0 ? now - state_since_ms_ : 0;
        const int64_t unhealthy_age_ms = unhealthy_since_ms_ > 0 ? now - unhealthy_since_ms_ : 0;
        const int64_t color_age_ms = last_color_ms_ > 0 ? now - last_color_ms_ : -1;
        const int64_t depth_age_ms = last_depth_ms_ > 0 ? now - last_depth_ms_ : -1;
        const bool connected = camera_connected_locked(now);
        const bool stale = !connected;
        std::string severity = "ok";
        bool alarm_active = false;
        int numeric_fault_code = 0;
        if (!connected) {
            alarm_active = unhealthy_age_ms >= reconnect_alarm_ms_;
            severity = alarm_active ? "error" : "warning";
            if (camera_state_ == CameraState::Stale) numeric_fault_code = 3101;
            else if (camera_state_ == CameraState::Reconnecting) numeric_fault_code = 3102;
            else if (camera_state_ == CameraState::Offline) numeric_fault_code = 3103;
            else numeric_fault_code = 3100;
        }
        std::ostringstream os;
        os << "{"
           << "\"ok\":true,"
           << "\"component\":\"orbbec336l_bridge\","
           << "\"health\":\"" << severity << "\","
           << "\"camera_started\":" << (camera_started_ ? "true" : "false") << ","
           << "\"camera_connected\":" << (connected ? "true" : "false") << ","
           << "\"camera_state\":\"" << camera_state_name(camera_state_) << "\","
           << "\"camera_thread_alive\":" << (camera_thread_alive_ ? "true" : "false") << ","
           << "\"frame_stale\":" << (stale ? "true" : "false") << ","
           << "\"fault_code\":\"" << json_escape(connected ? "" : fault_code_) << "\","
           << "\"fault_numeric_code\":" << numeric_fault_code << ","
           << "\"alarm_active\":" << (alarm_active ? "true" : "false") << ","
           << "\"fault_since_timestamp_ms\":" << (connected ? 0 : unhealthy_since_epoch_ms_) << ","
           << "\"state_age_ms\":" << state_age_ms << ","
           << "\"unhealthy_age_ms\":" << unhealthy_age_ms << ","
           << "\"frame_count\":" << frame_count_ << ","
           << "\"color_frame_count\":" << color_frame_count_ << ","
           << "\"depth_frame_count\":" << depth_frame_count_ << ","
           << "\"color_width\":" << color_w_ << ","
           << "\"color_height\":" << color_h_ << ","
           << "\"depth_width\":" << depth_w_ << ","
           << "\"depth_height\":" << depth_h_ << ","
           << "\"color_format\":\"" << json_escape(color_format_) << "\","
           << "\"depth_scale\":" << std::fixed << std::setprecision(6) << depth_scale_ << ","
           << "\"last_color_age_ms\":" << color_age_ms << ","
           << "\"last_depth_age_ms\":" << depth_age_ms << ","
           << "\"stale_timeout_ms\":" << stale_timeout_ms_ << ","
           << "\"reconnect_attempt_count\":" << reconnect_attempt_count_ << ","
           << "\"reconnect_success_count\":" << reconnect_success_count_ << ","
           << "\"consecutive_reconnect_failures\":" << consecutive_reconnect_failures_ << ","
           << "\"stale_event_count\":" << stale_event_count_ << ","
           << "\"last_reconnect_attempt_timestamp_ms\":" << last_reconnect_attempt_epoch_ms_ << ","
           << "\"last_reconnect_success_timestamp_ms\":" << last_reconnect_success_epoch_ms_ << ","
           << "\"last_error_timestamp_ms\":" << last_error_epoch_ms_ << ","
           << "\"uptime_ms\":" << (start_ms_ > 0 ? now - start_ms_ : 0) << ","
           << "\"time\":\"" << json_escape(now_string()) << "\","
           << "\"last_error\":\"" << json_escape(last_error_) << "\","
           << "\"alarm_interface\":{\"modbus_tcp_reserved\":true,\"implemented\":false}"
           << "}";
        return os.str();
    }

    std::string depth_meta_json() {
        std::lock_guard<std::mutex> lk(mtx_);
        const int64_t now = now_ms();
        const int64_t age = last_depth_ms_ > 0 ? now - last_depth_ms_ : -1;
        const bool fresh = !latest_depth_mm_.empty() && age >= 0 && age <= stale_timeout_ms_;
        std::ostringstream os;
        os << "{"
           << "\"ok\":" << (fresh ? "true" : "false") << ","
           << "\"fresh\":" << (fresh ? "true" : "false") << ","
           << "\"camera_state\":\"" << camera_state_name(camera_state_) << "\","
           << "\"width\":" << depth_w_ << ","
           << "\"height\":" << depth_h_ << ","
           << "\"encoding\":\"16UC1\","
           << "\"unit\":\"mm\","
           << "\"scale\":" << std::fixed << std::setprecision(6) << depth_scale_ << ","
           << "\"last_depth_ms\":" << last_depth_ms_ << ","
           << "\"last_depth_age_ms\":" << age
           << "}";
        return os.str();
    }



    void append_profile_array(std::ostringstream &os, OBSensorType sensor_type, const std::string &sensor_name) {
        std::set<std::string> seen;
        bool first = true;
        try {
            std::shared_ptr<ob::Pipeline> pipeline;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                pipeline = pipeline_;
            }
            if (!pipeline) {
                os << "]";
                return;
            }
            auto profiles = pipeline->getStreamProfileList(sensor_type);
            uint32_t count = profiles ? profiles->count() : 0;
            for (uint32_t i = 0; i < count; ++i) {
                std::shared_ptr<ob::VideoStreamProfile> profile;
                try {
                    profile = profiles->getProfile(i)->as<ob::VideoStreamProfile>();
                } catch (...) {
                    continue;
                }
                if (!profile) continue;
                const int w = profile->width();
                const int h = profile->height();
                const int fps = profile->fps();
                const std::string fmt = frame_format_to_string(profile->format());
                if (w <= 0 || h <= 0 || fps <= 0) continue;
                if (sensor_name == "depth" && fmt != "Y16") continue;
                if (sensor_name == "color" && fmt == "Y16") continue;
                std::ostringstream key;
                key << sensor_name << ":" << w << "x" << h << "@" << fps << ":" << fmt;
                if (!seen.insert(key.str()).second) continue;
                if (!first) os << ",";
                first = false;
                os << "{";
                os << "\"sensor\":\"" << json_escape(sensor_name) << "\",";
                os << "\"width\":" << w << ",";
                os << "\"height\":" << h << ",";
                os << "\"fps\":" << fps << ",";
                os << "\"format\":\"" << json_escape(fmt) << "\",";
                os << "\"id\":\"orbbec:" << w << "x" << h << "@" << fps << "\",";
                os << "\"label\":\"" << (sensor_name == "color" ? "RGB " : "Depth ") << w << "x" << h << " @ " << fps << " FPS (" << json_escape(fmt) << ")\"";
                os << "}";
            }
        } catch (const std::exception &e) {
            record_error(std::string("enumerate profiles failed: ") + e.what());
        } catch (...) {
            record_error("enumerate profiles failed");
        }
        os << "]";
    }

    std::string profiles_json() {
        std::ostringstream os;
        os << "{";
        os << "\"ok\":true,";
        os << "\"component\":\"orbbec336l_bridge\",";
        os << "\"profiles\":{";
        os << "\"color\":[";
        append_profile_array(os, OB_SENSOR_COLOR, "color");
        os << ",\"depth\":[";
        append_profile_array(os, OB_SENSOR_DEPTH, "depth");
        os << "},";
        os << "\"selected\":{";
        os << "\"color_width\":" << color_width_ << ",";
        os << "\"color_height\":" << color_height_ << ",";
        os << "\"depth_width\":" << depth_width_ << ",";
        os << "\"depth_height\":" << depth_height_ << ",";
        os << "\"fps\":" << fps_;
        os << "}";
        os << "}";
        return os.str();
    }

    bool copy_fresh_color(cv::Mat &img, uint64_t *frame_sequence = nullptr, int wait_ms = 0) {
        std::unique_lock<std::mutex> lk(mtx_);
        if (wait_ms > 0 && !camera_connected_locked(now_ms())) {
            cv_.wait_for(lk, std::chrono::milliseconds(wait_ms), [this]() {
                return !run_requested() || camera_connected_locked(now_ms());
            });
        }
        if (!camera_connected_locked(now_ms())) return false;
        img = latest_bgr_.clone();
        if (frame_sequence) *frame_sequence = color_frame_count_;
        return !img.empty();
    }

    bool copy_fresh_depth(cv::Mat &depth, int wait_ms = 0) {
        std::unique_lock<std::mutex> lk(mtx_);
        if (wait_ms > 0 && !camera_connected_locked(now_ms())) {
            cv_.wait_for(lk, std::chrono::milliseconds(wait_ms), [this]() {
                return !run_requested() || camera_connected_locked(now_ms());
            });
        }
        if (!camera_connected_locked(now_ms())) return false;
        depth = latest_depth_mm_.clone();
        return !depth.empty();
    }

    void send_snapshot(int fd) {
        cv::Mat img;
        if (!copy_fresh_color(img, nullptr, 1500)) {
            send_text(fd, 503, "application/json", "{\"ok\":false,\"error\":\"camera frame stale or reconnecting\"}");
            return;
        }
        std::vector<uchar> jpg;
        if (!encode_jpeg(img, jpeg_quality_, jpg)) {
            send_text(fd, 503, "application/json", "{\"ok\":false,\"error\":\"no color frame\"}");
            return;
        }
        send_binary(fd, 200, "image/jpeg", jpg);
    }

    void send_depth_png(int fd) {
        cv::Mat depth;
        if (!copy_fresh_depth(depth, 1500)) {
            send_text(fd, 503, "application/json", "{\"ok\":false,\"error\":\"depth frame stale or reconnecting\"}");
            return;
        }
        std::vector<uchar> png;
        if (!encode_png16(depth, png)) {
            send_text(fd, 503, "application/json", "{\"ok\":false,\"error\":\"no depth frame\"}");
            return;
        }
        send_binary(fd, 200, "image/png", png);
    }

    void send_depth_vis(int fd) {
        cv::Mat depth;
        if (!copy_fresh_depth(depth, 1500)) {
            send_text(fd, 503, "application/json", "{\"ok\":false,\"error\":\"depth frame stale or reconnecting\"}");
            return;
        }
        std::vector<uchar> jpg;
        if (!encode_jpeg(depth_to_vis(depth), jpeg_quality_, jpg)) {
            send_text(fd, 503, "application/json", "{\"ok\":false,\"error\":\"no depth frame\"}");
            return;
        }
        send_binary(fd, 200, "image/jpeg", jpg);
    }

    void send_mjpeg(int fd) {
        cv::Mat initial;
        uint64_t last_sequence = 0;
        if (!copy_fresh_color(initial, &last_sequence, 1500)) {
            send_text(fd, 503, "application/json", "{\"ok\":false,\"error\":\"camera stream unavailable\"}");
            return;
        }
        std::string boundary = "visionops-orbbec336l";
        std::ostringstream h;
        h << "HTTP/1.1 200 OK\r\n"
          << "Content-Type: multipart/x-mixed-replace; boundary=" << boundary << "\r\n"
          << "Cache-Control: no-cache\r\n"
          << "Connection: close\r\n\r\n";
        auto hs = h.str();
        send_all(fd, hs.data(), hs.size());

        const int delay_ms = std::max(10, 1000 / std::max(1, mjpeg_fps_));
        cv::Mat img = initial;
        while (run_requested()) {
            if (img.empty()) {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait_for(lk, std::chrono::milliseconds(std::max(100, delay_ms)), [this, last_sequence]() {
                    return !run_requested() || !camera_connected_locked(now_ms()) || color_frame_count_ != last_sequence;
                });
                if (!run_requested() || !camera_connected_locked(now_ms())) break;
                if (color_frame_count_ == last_sequence) continue;
                img = latest_bgr_.clone();
                last_sequence = color_frame_count_;
            }
            std::vector<uchar> jpg;
            if (encode_jpeg(img, jpeg_quality_, jpg)) {
                std::ostringstream ph;
                ph << "--" << boundary << "\r\n"
                   << "Content-Type: image/jpeg\r\n"
                   << "Content-Length: " << jpg.size() << "\r\n\r\n";
                auto ps = ph.str();
                if (::send(fd, ps.data(), ps.size(), MSG_NOSIGNAL) <= 0) break;
                if (::send(fd, jpg.data(), jpg.size(), MSG_NOSIGNAL) <= 0) break;
                const std::string tail = "\r\n";
                if (::send(fd, tail.data(), tail.size(), MSG_NOSIGNAL) <= 0) break;
            }
            img.release();
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }

    void handle_client(int fd) {
        std::string method;
        std::string path;
        std::string body;
        if (!read_http_request(fd, method, path, body)) {
            send_text(fd, 400, "application/json", "{\"ok\":false,\"error\":\"bad request\"}");
            ::close(fd);
            return;
        }

        if (path == "/health" || path == "/stream/status") {
            send_text(fd, 200, "application/json", status_json());
        } else if (path == "/stream/snapshot.jpg" || path == "/snapshot.jpg") {
            send_snapshot(fd);
        } else if (path == "/stream/depth.png") {
            send_depth_png(fd);
        } else if (path == "/stream/depth_vis.jpg" || path == "/stream/depth.jpg") {
            send_depth_vis(fd);
        } else if (path == "/stream/depth_meta") {
            send_text(fd, 200, "application/json", depth_meta_json());
        } else if (path == "/stream/camera_info" || path == "/api/camera/info") {
            send_text(fd, 200, "application/json", camera_info_json());
        } else if (method == "POST" && path == "/api/coordinate/deproject") {
            int code = 200;
            auto response = deproject_json(body, code);
            send_text(fd, code, "application/json", response);
        } else if (path == "/stream/profiles" || path == "/profiles") {
            send_text(fd, 200, "application/json", profiles_json());
        } else if (path == "/stream.mjpeg" || path == "/stream/mjpeg" || path == "/stream.mjpg") {
            send_mjpeg(fd);
        } else if ((method == "POST" || method == "GET") && (path == "/stream/start" || path == "/stream/stop")) {
            send_text(fd, 200, "application/json", "{\"ok\":true,\"note\":\"camera bridge keeps streaming\"}");
        } else {
            send_text(fd, 404, "application/json", "{\"ok\":false,\"error\":\"not found\"}");
        }
        ::close(fd);
    }

private:
    std::string http_host_;
    int http_port_;
    int color_width_;
    int color_height_;
    int depth_width_;
    int depth_height_;
    int fps_;
    int jpeg_quality_;
    int mjpeg_fps_;
    int stale_timeout_ms_;
    int first_frame_timeout_ms_;
    int reconnect_initial_ms_;
    int reconnect_max_ms_;
    int reconnect_alarm_ms_;
    bool flip_vertical_;
    bool flip_horizontal_;
    std::string serial_;

    std::shared_ptr<ob::Context> ctx_;
    std::shared_ptr<ob::Pipeline> pipeline_;
    OBCalibrationParam calibration_param_{};
    bool calibration_ready_ = false;
    std::shared_ptr<ob::VideoStreamProfile> color_profile_;
    std::shared_ptr<ob::VideoStreamProfile> depth_profile_;
    std::thread camera_thread_;
    std::atomic<bool> camera_started_{false};
    std::atomic<bool> camera_thread_alive_{false};
    std::atomic<bool> camera_stop_requested_{false};

    std::mutex mtx_;
    std::condition_variable cv_;
    cv::Mat latest_bgr_;
    cv::Mat latest_depth_mm_;
    int64_t last_color_ms_ = 0;
    int64_t last_depth_ms_ = 0;
    int64_t pipeline_started_ms_ = 0;
    int64_t start_ms_ = 0;
    uint64_t frame_count_ = 0;
    uint64_t color_frame_count_ = 0;
    uint64_t depth_frame_count_ = 0;
    CameraState camera_state_ = CameraState::Starting;
    int64_t state_since_ms_ = 0;
    int64_t state_since_epoch_ms_ = 0;
    int64_t unhealthy_since_ms_ = 0;
    int64_t unhealthy_since_epoch_ms_ = 0;
    int64_t last_error_epoch_ms_ = 0;
    int64_t last_reconnect_attempt_epoch_ms_ = 0;
    int64_t last_reconnect_success_epoch_ms_ = 0;
    uint64_t reconnect_attempt_count_ = 0;
    uint64_t reconnect_success_count_ = 0;
    uint64_t consecutive_reconnect_failures_ = 0;
    uint64_t stale_event_count_ = 0;
    bool ever_connected_ = false;
    int color_w_ = 0;
    int color_h_ = 0;
    int depth_w_ = 0;
    int depth_h_ = 0;
    float depth_scale_ = 1.0f;
    std::string color_format_ = "";
    std::string fault_code_ = "CAMERA_STARTING";
    std::string last_error_ = "";

    int server_fd_ = -1;
};

void signal_handler(int) {
    g_running = false;
    int fd = g_server_fd.exchange(-1);
    if (fd >= 0) {
        ::close(fd);
    }
}

} // namespace

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    prepare_runtime_workdir();
    std::cerr << "[INFO] VisionOps Orbbec Gemini 336L SDK bridge starting" << std::endl;
    OrbbecBridge bridge;
    if (!bridge.start_camera()) {
        std::cerr << "[FATAL] camera start failed" << std::endl;
        return 2;
    }
    bool ok = bridge.start_http();
    bridge.stop_camera();
    return ok ? 0 : 1;
}
