#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "visionops_runtime/runtime_state.hpp"

namespace visionops::runtime {

struct HttpRequest {
  std::string method;
  std::string target;
  std::string path;
  std::string body;
  std::unordered_map<std::string, std::string> headers;
};

struct HttpResponse {
  int status_code{200};
  std::string reason{"OK"};
  std::string content_type{"application/json; charset=utf-8"};
  std::vector<std::uint8_t> body;
  std::vector<std::pair<std::string, std::string>> headers;
};

class HttpServer {
 public:
  HttpServer(
      std::string host,
      std::uint16_t port,
      std::string device_id,
      std::string component,
      std::string mock_task_type,
      RuntimeState& state,
      std::atomic_bool& stop_requested);
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  int run();

 private:
  bool open_listener();
  void close_listener();
  void handle_client(int client_fd);
  bool read_request(int client_fd, HttpRequest& request, std::string& error_message) const;
  HttpResponse route(const HttpRequest& request);
  HttpResponse json_response(int status_code, std::string reason, std::string body) const;
  HttpResponse method_not_allowed(const std::string& expected_method) const;
  bool write_response(int client_fd, const HttpResponse& response) const;

  std::string host_;
  std::uint16_t port_;
  std::string device_id_;
  std::string component_;
  std::string mock_task_type_;
  RuntimeState& state_;
  std::atomic_bool& stop_requested_;
  int listen_fd_{-1};
};

}  // namespace visionops::runtime
