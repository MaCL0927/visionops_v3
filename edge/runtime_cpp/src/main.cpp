#include <atomic>
#include <csignal>
#include <iostream>

#include "visionops_runtime/cli_args.hpp"
#include "visionops_runtime/http_server.hpp"
#include "visionops_runtime/runtime_app.hpp"

namespace {

std::atomic_bool* g_stop_requested = nullptr;

void signal_handler(int) {
  if (g_stop_requested != nullptr) {
    g_stop_requested->store(true);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto args = visionops::runtime::parse_cli_args(argc, argv);
    if (args.show_help) {
      std::cout << visionops::runtime::cli_help_text(argv[0]);
      return 0;
    }

    std::atomic_bool stop_requested{false};
    g_stop_requested = &stop_requested;
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    visionops::runtime::RuntimeApp app(args.config);
    visionops::runtime::HttpServer server(
        args.config.host,
        args.config.port,
        app,
        stop_requested);
    const int result = server.run();
    g_stop_requested = nullptr;
    return result;
  } catch (const std::exception& error) {
    std::cerr << "启动失败: " << error.what() << '\n'
              << visionops::runtime::cli_help_text(argv[0]);
    return 2;
  }
}
