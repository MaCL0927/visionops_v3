#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "visionops_runtime/http_types.hpp"
#include "visionops_runtime/runtime_app.hpp"

namespace visionops::runtime {

class HttpServer {
 public:
  HttpServer(
      std::string host,
      std::uint16_t port,
      RuntimeApp& app,
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
  HttpResponse error_response(
      int status_code,
      std::string reason,
      const std::string& code,
      const std::string& message,
      bool recoverable) const;
  bool write_response(int client_fd, const HttpResponse& response) const;

  std::string host_;
  std::uint16_t port_;
  RuntimeApp& app_;
  std::atomic_bool& stop_requested_;
  int listen_fd_{-1};
};

}  // namespace visionops::runtime
