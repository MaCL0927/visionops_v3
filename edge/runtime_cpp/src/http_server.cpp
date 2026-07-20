#include "visionops_runtime/http_server.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

#include "visionops_runtime/json_utils.hpp"

namespace visionops::runtime {

namespace {

constexpr std::size_t kMaxHeaderBytes = 64 * 1024;
constexpr std::size_t kMaxBodyBytes = 1024 * 1024;

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

std::string trim(std::string value) {
  const auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char ch) {
                return !is_space(static_cast<unsigned char>(ch));
              }));
  value.erase(std::find_if(value.rbegin(), value.rend(), [&](char ch) {
                return !is_space(static_cast<unsigned char>(ch));
              }).base(), value.end());
  return value;
}

std::vector<std::uint8_t> bytes(std::string value) {
  return {value.begin(), value.end()};
}


int positive_env_int(const char* name, int fallback, int minimum, int maximum) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return fallback;
  try {
    const int value = std::stoi(raw);
    return std::max(minimum, std::min(maximum, value));
  } catch (const std::exception&) {
    return fallback;
  }
}

double elapsed_ms(std::chrono::steady_clock::time_point started) {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
}

bool send_iov_all(int fd, const std::string& header, const std::vector<std::uint8_t>& body) {
  iovec parts[2]{};
  parts[0].iov_base = const_cast<char*>(header.data());
  parts[0].iov_len = header.size();
  parts[1].iov_base = body.empty() ? nullptr : const_cast<std::uint8_t*>(body.data());
  parts[1].iov_len = body.size();
  int first = 0;
  int count = body.empty() ? 1 : 2;
  while (first < count) {
    msghdr message{};
    message.msg_iov = parts + first;
    message.msg_iovlen = static_cast<std::size_t>(count - first);
    const ssize_t sent = sendmsg(fd, &message, MSG_NOSIGNAL);
    if (sent < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (sent == 0) return false;
    std::size_t remaining = static_cast<std::size_t>(sent);
    while (first < count && remaining >= parts[first].iov_len) {
      remaining -= parts[first].iov_len;
      ++first;
    }
    if (first < count && remaining > 0) {
      auto* base = static_cast<std::uint8_t*>(parts[first].iov_base);
      parts[first].iov_base = base + remaining;
      parts[first].iov_len -= remaining;
    }
  }
  return true;
}
std::string status_reason(int status_code) {
  switch (status_code) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Error";
  }
}

}  // namespace

HttpServer::HttpServer(
    std::string host,
    std::uint16_t port,
    RuntimeApp& app,
    std::atomic_bool& stop_requested)
    : host_(std::move(host)),
      port_(port),
      app_(app),
      stop_requested_(stop_requested),
      worker_count_(positive_env_int("VISIONOPS_RUNTIME_HTTP_WORKERS", 4, 1, 32)),
      queue_capacity_(static_cast<std::size_t>(
          positive_env_int("VISIONOPS_RUNTIME_HTTP_QUEUE", 64, 4, 1024))) {}

HttpServer::~HttpServer() {
  close_listener();
  stop_workers();
}

bool HttpServer::open_listener() {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  addrinfo* addresses = nullptr;
  const std::string port_text = std::to_string(port_);
  const char* host = host_.empty() ? nullptr : host_.c_str();
  const int lookup = getaddrinfo(host, port_text.c_str(), &hints, &addresses);
  if (lookup != 0) {
    std::cerr << "监听地址解析失败: " << gai_strerror(lookup) << '\n';
    return false;
  }

  for (addrinfo* address = addresses; address != nullptr; address = address->ai_next) {
    const int candidate = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (candidate < 0) {
      continue;
    }
    int reuse = 1;
    setsockopt(candidate, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    if (bind(candidate, address->ai_addr, address->ai_addrlen) == 0 &&
        listen(candidate, 16) == 0) {
      listen_fd_ = candidate;
      break;
    }
    close(candidate);
  }

  freeaddrinfo(addresses);
  if (listen_fd_ < 0) {
    std::cerr << "无法监听 " << host_ << ':' << port_ << ": " << std::strerror(errno) << '\n';
    return false;
  }
  return true;
}

void HttpServer::close_listener() {
  if (listen_fd_ >= 0) {
    close(listen_fd_);
    listen_fd_ = -1;
  }
}

int HttpServer::run() {
  if (!open_listener()) {
    return 1;
  }
  start_workers();
  std::cout << "VisionOps Runtime Mock 正在监听 " << host_ << ':' << port_
            << "，task=" << app_.config().mock_task_type
            << "，http_workers=" << worker_count_ << '\n';

  while (!stop_requested_.load()) {
    pollfd descriptor{};
    descriptor.fd = listen_fd_;
    descriptor.events = POLLIN;
    const int ready = poll(&descriptor, 1, 200);
    if (ready < 0) {
      if (errno == EINTR) {
        continue;
      }
      std::cerr << "poll 失败: " << std::strerror(errno) << '\n';
      stop_workers();
      return 1;
    }
    if (ready == 0 || (descriptor.revents & POLLIN) == 0) {
      continue;
    }

    sockaddr_storage peer{};
    socklen_t peer_length = sizeof(peer);
    const int client_fd = accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &peer_length);
    if (client_fd < 0) {
      if (errno != EINTR && !stop_requested_.load()) {
        std::cerr << "accept 失败: " << std::strerror(errno) << '\n';
      }
      continue;
    }
    configure_client_socket(client_fd);
    const auto accepted_at = std::chrono::steady_clock::now();
    if (!enqueue_client(client_fd, accepted_at)) {
      auto response = error_response(
          503,
          "Service Unavailable",
          "HTTP_WORK_QUEUE_FULL",
          "Runtime HTTP 工作队列已满",
          true);
      response.headers.emplace_back("Retry-After", "1");
      write_response(client_fd, response);
      close(client_fd);
    }
  }

  stop_workers();
  std::cout << "VisionOps Runtime Mock 已停止\n";
  return 0;
}

void HttpServer::start_workers() {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  if (!workers_.empty()) return;
  workers_stopping_ = false;
  workers_.reserve(static_cast<std::size_t>(worker_count_));
  for (int index = 0; index < worker_count_; ++index) {
    workers_.emplace_back(&HttpServer::worker_loop, this);
  }
}

void HttpServer::stop_workers() {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (workers_.empty() && workers_stopping_) return;
    workers_stopping_ = true;
  }
  queue_cv_.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) worker.join();
  }
  workers_.clear();
  std::queue<QueuedClient> remaining;
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    remaining.swap(client_queue_);
  }
  while (!remaining.empty()) {
    if (remaining.front().fd >= 0) close(remaining.front().fd);
    remaining.pop();
  }
}

bool HttpServer::enqueue_client(
    int client_fd,
    std::chrono::steady_clock::time_point accepted_at) {
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (workers_stopping_ || client_queue_.size() >= queue_capacity_) return false;
    client_queue_.push(QueuedClient{client_fd, accepted_at});
  }
  queue_cv_.notify_one();
  return true;
}

void HttpServer::worker_loop() {
  while (true) {
    QueuedClient client;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [this]() {
        return workers_stopping_ || !client_queue_.empty();
      });
      if (workers_stopping_ && client_queue_.empty()) return;
      client = client_queue_.front();
      client_queue_.pop();
    }
    handle_client(client.fd, client.accepted_at);
    close(client.fd);
  }
}

void HttpServer::configure_client_socket(int client_fd) const {
  timeval timeout{};
  timeout.tv_sec = 3;
  setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  int one = 1;
  setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

void HttpServer::handle_client(
    int client_fd,
    std::chrono::steady_clock::time_point accepted_at) {
  const double queue_wait_ms = elapsed_ms(accepted_at);
  HttpRequest request;
  std::string error_message;
  if (!read_request(client_fd, request, error_message)) {
    const bool too_large = error_message == "请求体过大";
    auto response = error_response(
        too_large ? 413 : 400,
        too_large ? "Payload Too Large" : "Bad Request",
        "INVALID_HTTP_REQUEST",
        error_message,
        true);
    response.headers.emplace_back("X-VisionOps-Http-Queue-Ms", std::to_string(queue_wait_ms));
    write_response(client_fd, response);
    return;
  }

  const auto route_started = std::chrono::steady_clock::now();
  try {
    auto response = route(request);
    response.headers.emplace_back("X-VisionOps-Http-Queue-Ms", std::to_string(queue_wait_ms));
    response.headers.emplace_back("X-VisionOps-Http-Route-Ms", std::to_string(elapsed_ms(route_started)));
    write_response(client_fd, response);
  } catch (const std::exception& error) {
    app_.record_error();
    auto response = error_response(
        500, "Internal Server Error", "INTERNAL_ERROR", error.what(), true);
    response.headers.emplace_back("X-VisionOps-Http-Queue-Ms", std::to_string(queue_wait_ms));
    response.headers.emplace_back("X-VisionOps-Http-Route-Ms", std::to_string(elapsed_ms(route_started)));
    write_response(client_fd, response);
  }
}

bool HttpServer::read_request(
    int client_fd,
    HttpRequest& request,
    std::string& error_message) const {
  std::string raw;
  raw.reserve(4096);
  char buffer[4096];
  std::size_t header_end = std::string::npos;

  while ((header_end = raw.find("\r\n\r\n")) == std::string::npos) {
    const ssize_t count = recv(client_fd, buffer, sizeof(buffer), 0);
    if (count <= 0) {
      error_message = "请求头读取失败";
      return false;
    }
    raw.append(buffer, static_cast<std::size_t>(count));
    if (raw.size() > kMaxHeaderBytes) {
      error_message = "请求头过大";
      return false;
    }
  }

  const std::string header_text = raw.substr(0, header_end);
  std::istringstream stream(header_text);
  std::string request_line;
  if (!std::getline(stream, request_line)) {
    error_message = "缺少请求行";
    return false;
  }
  if (!request_line.empty() && request_line.back() == '\r') {
    request_line.pop_back();
  }
  std::istringstream request_line_stream(request_line);
  std::string version;
  if (!(request_line_stream >> request.method >> request.target >> version) ||
      version.rfind("HTTP/", 0) != 0) {
    error_message = "请求行格式错误";
    return false;
  }
  request.path = request.target.substr(0, request.target.find('?'));

  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const auto separator = line.find(':');
    if (separator == std::string::npos) {
      error_message = "请求头格式错误";
      return false;
    }
    request.headers[lower_copy(trim(line.substr(0, separator)))] = trim(line.substr(separator + 1));
  }

  std::size_t content_length = 0;
  const auto length_header = request.headers.find("content-length");
  if (length_header != request.headers.end()) {
    try {
      const unsigned long long parsed = std::stoull(length_header->second);
      if (parsed > kMaxBodyBytes || parsed > std::numeric_limits<std::size_t>::max()) {
        error_message = "请求体过大";
        return false;
      }
      content_length = static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
      error_message = "Content-Length 非法";
      return false;
    }
  }

  const std::size_t body_start = header_end + 4;
  request.body = raw.substr(body_start);
  while (request.body.size() < content_length) {
    const std::size_t remaining = content_length - request.body.size();
    const ssize_t count = recv(client_fd, buffer, std::min(remaining, sizeof(buffer)), 0);
    if (count <= 0) {
      error_message = "请求体读取失败";
      return false;
    }
    request.body.append(buffer, static_cast<std::size_t>(count));
  }
  if (request.body.size() > content_length) {
    request.body.resize(content_length);
  }
  return true;
}

HttpResponse HttpServer::route(const HttpRequest& request) {
  if (request.path == "/health") {
    return request.method == "GET" ? json_response(200, "OK", app_.health_json())
                                   : method_not_allowed("GET");
  }
  if (request.path == "/api/runtime/status") {
    return request.method == "GET" ? json_response(200, "OK", app_.status_json())
                                   : method_not_allowed("GET");
  }
  if (request.path == "/api/runtime/start_preview") {
    return request.method == "POST" ? json_response(200, "OK", app_.start_preview())
                                    : method_not_allowed("POST");
  }
  if (request.path == "/api/runtime/stop_preview") {
    return request.method == "POST" ? json_response(200, "OK", app_.stop_preview())
                                    : method_not_allowed("POST");
  }
  if (request.path == "/api/runtime/infer_once") {
    return request.method == "POST" ? json_response(200, "OK", app_.infer_once())
                                    : method_not_allowed("POST");
  }
  if (request.path == "/api/runtime/switch_model") {
    if (request.method != "POST") {
      return method_not_allowed("POST");
    }
    const auto result = app_.switch_model(request.body);
    return json_response(result.status_code, status_reason(result.status_code), result.body);
  }
  if (request.path == "/api/runtime/roi") {
    if (request.method == "GET") {
      return json_response(200, "OK", app_.roi_json());
    }
    if (request.method == "POST") {
      const auto result = app_.update_roi(request.body);
      return json_response(result.status_code, status_reason(result.status_code), result.body);
    }
    return method_not_allowed("GET, POST");
  }
  if (request.path == "/api/runtime/latest_result") {
    if (request.method != "GET") {
      return method_not_allowed("GET");
    }
    const auto result = app_.latest_result_json();
    return result ? json_response(200, "OK", *result)
                  : error_response(
                        404,
                        "Not Found",
                        "LATEST_RESULT_NOT_FOUND",
                        "尚未生成推理结果",
                        true);
  }
  if (request.path == "/api/runtime/snapshot.jpg") {
    if (request.method != "GET" && request.method != "HEAD") {
      return method_not_allowed("GET, HEAD");
    }
    auto snapshot = app_.snapshot_jpeg();
    if (snapshot.empty()) {
      auto response = error_response(
          503,
          "Service Unavailable",
          "CAMERA_FRAME_STALE",
          "实时画面暂不可用，Runtime 正在自动重连帧源",
          true);
      response.headers.emplace_back("X-Frame-Id", app_.snapshot_frame_id());
      response.headers.emplace_back("Cache-Control", "no-store");
      return response;
    }
    HttpResponse response;
    response.content_type = "image/jpeg";
    if (request.method == "GET") {
      response.body = std::move(snapshot);
    }
    response.headers.emplace_back("X-Frame-Id", app_.snapshot_frame_id());
    response.headers.emplace_back("X-Timestamp-Ms", std::to_string(now_timestamp_ms()));
    response.headers.emplace_back("Cache-Control", "no-store");
    return response;
  }
  return error_response(404, "Not Found", "ROUTE_NOT_FOUND", "接口不存在", true);
}

HttpResponse HttpServer::json_response(
    int status_code,
    std::string reason,
    std::string body) const {
  HttpResponse response;
  response.status_code = status_code;
  response.reason = std::move(reason);
  response.body = bytes(std::move(body));
  return response;
}

HttpResponse HttpServer::method_not_allowed(const std::string& expected_method) const {
  HttpResponse response = error_response(
      405,
      "Method Not Allowed",
      "METHOD_NOT_ALLOWED",
      "请求方法不支持，期望 " + expected_method,
      true);
  response.headers.emplace_back("Allow", expected_method);
  return response;
}

HttpResponse HttpServer::error_response(
    int status_code,
    std::string reason,
    const std::string& code,
    const std::string& message,
    bool recoverable) const {
  return json_response(
      status_code,
      std::move(reason),
      make_error_json(
          app_.config().device_id,
          app_.config().component,
          code,
          message,
          recoverable));
}

bool HttpServer::write_response(int client_fd, const HttpResponse& response) const {
  std::ostringstream header;
  header << "HTTP/1.1 " << response.status_code << ' '
         << (response.reason.empty() ? status_reason(response.status_code) : response.reason) << "\r\n"
         << "Content-Type: " << response.content_type << "\r\n"
         << "Content-Length: " << response.body.size() << "\r\n"
         << "Connection: close\r\n";
  for (const auto& [name, value] : response.headers) {
    header << name << ": " << value << "\r\n";
  }
  header << "\r\n";

  const std::string header_text = header.str();
  return send_iov_all(client_fd, header_text, response.body);
}

}  // namespace visionops::runtime
