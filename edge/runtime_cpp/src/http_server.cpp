#include "visionops_runtime/http_server.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string_view>

#include "visionops_runtime/mock_data.hpp"

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

std::string status_reason(int status_code) {
  switch (status_code) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 413:
      return "Payload Too Large";
    case 500:
      return "Internal Server Error";
    default:
      return "Error";
  }
}

}  // namespace

HttpServer::HttpServer(
    std::string host,
    std::uint16_t port,
    std::string device_id,
    std::string component,
    std::string mock_task_type,
    RuntimeState& state,
    std::atomic_bool& stop_requested)
    : host_(std::move(host)),
      port_(port),
      device_id_(std::move(device_id)),
      component_(std::move(component)),
      mock_task_type_(std::move(mock_task_type)),
      state_(state),
      stop_requested_(stop_requested) {}

HttpServer::~HttpServer() { close_listener(); }

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
  std::cout << "VisionOps Runtime Mock 正在监听 " << host_ << ':' << port_
            << "，task=" << mock_task_type_ << '\n';

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
      return 1;
    }
    if (ready == 0 || (descriptor.revents & POLLIN) == 0) {
      continue;
    }

    sockaddr_storage peer{};
    socklen_t peer_length = sizeof(peer);
    const int client_fd = accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &peer_length);
    if (client_fd < 0) {
      if (errno != EINTR) {
        std::cerr << "accept 失败: " << std::strerror(errno) << '\n';
      }
      continue;
    }
    handle_client(client_fd);
    close(client_fd);
  }

  std::cout << "VisionOps Runtime Mock 已停止\n";
  return 0;
}

void HttpServer::handle_client(int client_fd) {
  timeval timeout{};
  timeout.tv_sec = 3;
  setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

  HttpRequest request;
  std::string error_message;
  if (!read_request(client_fd, request, error_message)) {
    const auto response = json_response(
        error_message == "请求体过大" ? 413 : 400,
        error_message == "请求体过大" ? "Payload Too Large" : "Bad Request",
        make_error_json(device_id_, component_, "INVALID_HTTP_REQUEST", error_message, true));
    write_response(client_fd, response);
    return;
  }
  write_response(client_fd, route(request));
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
    if (request.method != "GET") {
      return method_not_allowed("GET");
    }
    const auto snapshot = state_.snapshot();
    return json_response(200, "OK", make_health_json(device_id_, component_, snapshot.uptime_s));
  }

  if (request.path == "/api/runtime/status") {
    if (request.method != "GET") {
      return method_not_allowed("GET");
    }
    return json_response(
        200,
        "OK",
        make_runtime_status_json(device_id_, component_, mock_task_type_, state_.snapshot()));
  }

  if (request.path == "/api/runtime/start_preview") {
    if (request.method != "POST") {
      return method_not_allowed("POST");
    }
    return json_response(
        200,
        "OK",
        make_runtime_status_json(device_id_, component_, mock_task_type_, state_.start_preview()));
  }

  if (request.path == "/api/runtime/stop_preview") {
    if (request.method != "POST") {
      return method_not_allowed("POST");
    }
    return json_response(
        200,
        "OK",
        make_runtime_status_json(device_id_, component_, mock_task_type_, state_.stop_preview()));
  }

  if (request.path == "/api/runtime/infer_once") {
    if (request.method != "POST") {
      return method_not_allowed("POST");
    }
    const auto identity = state_.begin_inference();
    const std::string result = make_inference_result_json(
        device_id_, component_, mock_task_type_, identity);
    state_.complete_inference(identity, result);
    return json_response(200, "OK", result);
  }

  if (request.path == "/api/runtime/latest_result") {
    if (request.method != "GET") {
      return method_not_allowed("GET");
    }
    const auto snapshot = state_.snapshot();
    if (!snapshot.latest_result_json) {
      return json_response(
          404,
          "Not Found",
          make_error_json(
              device_id_,
              component_,
              "LATEST_RESULT_NOT_FOUND",
              "尚未生成推理结果",
              true));
    }
    return json_response(200, "OK", *snapshot.latest_result_json);
  }

  if (request.path == "/api/runtime/snapshot.jpg") {
    if (request.method != "GET") {
      return method_not_allowed("GET");
    }
    HttpResponse response;
    response.content_type = "image/jpeg";
    response.body = placeholder_jpeg();
    const auto snapshot = state_.snapshot();
    response.headers.emplace_back(
        "X-Frame-Id",
        snapshot.last_frame_id.value_or("frame-mock-placeholder"));
    response.headers.emplace_back("X-Timestamp-Ms", std::to_string(timestamp_ms()));
    response.headers.emplace_back("Cache-Control", "no-store");
    return response;
  }

  return json_response(
      404,
      "Not Found",
      make_error_json(device_id_, component_, "ROUTE_NOT_FOUND", "接口不存在", true));
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
  HttpResponse response = json_response(
      405,
      "Method Not Allowed",
      make_error_json(
          device_id_,
          component_,
          "METHOD_NOT_ALLOWED",
          "请求方法不支持，期望 " + expected_method,
          true));
  response.headers.emplace_back("Allow", expected_method);
  return response;
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
  auto send_all = [&](const std::uint8_t* data, std::size_t size) {
    std::size_t sent = 0;
    while (sent < size) {
      const ssize_t count = send(client_fd, data + sent, size - sent, MSG_NOSIGNAL);
      if (count <= 0) {
        return false;
      }
      sent += static_cast<std::size_t>(count);
    }
    return true;
  };

  if (!send_all(reinterpret_cast<const std::uint8_t*>(header_text.data()), header_text.size())) {
    return false;
  }
  return response.body.empty() || send_all(response.body.data(), response.body.size());
}

}  // namespace visionops::runtime
