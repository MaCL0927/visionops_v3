#include "visionops_runtime/cli_args.hpp"

#include <stdexcept>

namespace visionops::runtime {

namespace {

std::string require_value(int argc, char* argv[], int& index) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string("参数缺少值: ") + argv[index]);
  }
  return argv[++index];
}

}  // namespace

CliArgs parse_cli_args(int argc, char* argv[]) {
  CliArgs args;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      args.show_help = true;
    } else if (argument == "--host") {
      args.config.host = require_value(argc, argv, index);
    } else if (argument == "--port") {
      const unsigned long parsed = std::stoul(require_value(argc, argv, index));
      if (parsed == 0 || parsed > 65535) {
        throw std::invalid_argument("端口必须位于 1 到 65535");
      }
      args.config.port = static_cast<std::uint16_t>(parsed);
    } else if (argument == "--device-id") {
      args.config.device_id = require_value(argc, argv, index);
    } else if (argument == "--component") {
      args.config.component = require_value(argc, argv, index);
    } else if (argument == "--mock-task-type") {
      args.config.mock_task_type = require_value(argc, argv, index);
    } else if (argument == "--backend") {
      args.config.backend = require_value(argc, argv, index);
    } else if (argument == "--model-manifest") {
      args.config.model_manifest = require_value(argc, argv, index);
    } else if (argument == "--model-config") {
      args.config.model_config = require_value(argc, argv, index);
    } else if (argument == "--model-dir") {
      args.config.model_dir = require_value(argc, argv, index);
    } else {
      throw std::invalid_argument("未知参数: " + argument);
    }
  }

  if (!args.show_help) {
    validate_app_config(args.config);
  }
  return args;
}

std::string cli_help_text(const std::string& program) {
  return "VisionOps v3 C++ Runtime Mock\n\n"
         "用法: " + program + " [选项]\n\n"
         "  --host <地址>             监听地址，默认 0.0.0.0\n"
         "  --port <端口>             监听端口，默认 18080\n"
         "  --device-id <标识>        Mock 设备标识\n"
         "  --component <名称>        组件名称，默认 rknn_runtime\n"
         "  --mock-task-type <类型>   detection、obb、segmentation、"
         "roi_classification 或 classification\n"
         "  --backend <类型>          mock 或 rknn，默认 mock\n"
         "  --model-manifest <路径>  模型包 manifest JSON\n"
         "  --model-config <路径>    模型 YAML 配置\n"
         "  --model-dir <路径>       模型包目录及相对路径基准\n"
         "  --help                    显示帮助\n";
}

}  // namespace visionops::runtime
