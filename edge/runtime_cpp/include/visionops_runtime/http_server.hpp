#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

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
  void start_workers();
  void stop_workers();
  void worker_loop(int worker_index);
  bool enqueue_client(int client_fd, std::chrono::steady_clock::time_point accepted_at);
  void configure_client_socket(int client_fd) const;
  void handle_client(
      int client_fd,
      std::chrono::steady_clock::time_point accepted_at);
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
  std::string server_status_json() const;

  std::string host_;
  std::uint16_t port_;
  RuntimeApp& app_;
  std::atomic_bool& stop_requested_;
  struct QueuedClient {
    int fd{-1};
    std::chrono::steady_clock::time_point accepted_at;
  };

  int listen_fd_{-1};
  int worker_count_{4};
  std::size_t queue_capacity_{64};
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::queue<QueuedClient> client_queue_;
  std::vector<std::thread> workers_;
  bool workers_stopping_{false};
  std::atomic<int> active_workers_{0};
  std::atomic<std::uint64_t> accepted_clients_{0};
  std::atomic<std::uint64_t> rejected_clients_{0};
  std::atomic<std::uint64_t> handled_clients_{0};
};

}  // namespace visionops::runtime
