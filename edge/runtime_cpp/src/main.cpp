#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

#include "visionops_runtime/http_server.hpp"
#include "visionops_runtime/runtime_state.hpp"

namespace {

std::atomic_bool* g_stop_requested = nullptr;

void signal_handler(int) {
  if (g_stop_requested != nullptr) {
    g_stop_requested->store(true);
  }
}

struct Options {
  std::string host{"0.0.0.0"};
  std::uint16_t port{18080};
  std::string device_id{"example-edge-001"};
  std::string component{"rknn_runtime"};
  std::string mock_task_type{"detection"};
};

void print_help(const char* program) {
  std::cout
      << "VisionOps v3 C++ Runtime Mock\n\n"
      << "用法: " << program << " [选项]\n\n"
      << "  --host <地址>             监听地址，默认 0.0.0.0\n"
      << "  --port <端口>             监听端口，默认 18080\n"
      << "  --device-id <标识>        Mock 设备标识\n"
      << "  --component <名称>        组件名称，默认 rknn_runtime\n"
      << "  --mock-task-type <类型>   detection、obb、segmentation、"
         "roi_classification 或 classification\n"
      << "  --help                    显示帮助\n";
}

std::string require_value(int argc, char* argv[], int& index) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string("参数缺少值: ") + argv[index]);
  }
  return argv[++index];
}

Options parse_options(int argc, char* argv[]) {
  Options options;
  const std::set<std::string> supported_tasks = {
      "detection", "obb", "segmentation", "roi_classification", "classification"};
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      print_help(argv[0]);
      std::exit(0);
    }
    if (argument == "--host") {
      options.host = require_value(argc, argv, index);
    } else if (argument == "--port") {
      const std::string value = require_value(argc, argv, index);
      const unsigned long parsed = std::stoul(value);
      if (parsed == 0 || parsed > 65535) {
        throw std::invalid_argument("端口必须位于 1 到 65535");
      }
      options.port = static_cast<std::uint16_t>(parsed);
    } else if (argument == "--device-id") {
      options.device_id = require_value(argc, argv, index);
    } else if (argument == "--component") {
      options.component = require_value(argc, argv, index);
    } else if (argument == "--mock-task-type") {
      options.mock_task_type = require_value(argc, argv, index);
    } else {
      throw std::invalid_argument("未知参数: " + argument);
    }
  }

  if (options.host.empty() || options.device_id.empty() || options.component.empty()) {
    throw std::invalid_argument("host、device-id 和 component 不能为空");
  }
  if (supported_tasks.count(options.mock_task_type) == 0) {
    throw std::invalid_argument("不支持的 mock task type: " + options.mock_task_type);
  }
  return options;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const Options options = parse_options(argc, argv);
    std::atomic_bool stop_requested{false};
    g_stop_requested = &stop_requested;
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    visionops::runtime::RuntimeState state;
    visionops::runtime::HttpServer server(
        options.host,
        options.port,
        options.device_id,
        options.component,
        options.mock_task_type,
        state,
        stop_requested);
    const int result = server.run();
    g_stop_requested = nullptr;
    return result;
  } catch (const std::exception& error) {
    std::cerr << "启动失败: " << error.what() << '\n';
    print_help(argv[0]);
    return 2;
  }
}
