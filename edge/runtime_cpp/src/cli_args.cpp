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
    } else if (argument == "--model-dir") {
      args.config.model_dir = require_value(argc, argv, index);
    } else if (argument == "--test-image") {
      args.config.test_image = require_value(argc, argv, index);
    } else if (argument == "--save-debug-output") {
      args.config.save_debug_output = require_value(argc, argv, index);
    } else if (argument == "--preprocess-backend") {
      args.config.preprocess_backend = require_value(argc, argv, index);
    } else if (argument == "--rga-mode") {
      args.config.rga_mode = require_value(argc, argv, index);
    } else if (argument == "--dump-rknn-io") {
      args.config.dump_rknn_io = true;
    } else if (argument == "--score-threshold") {
      args.config.score_threshold_override = std::stod(require_value(argc, argv, index));
    } else if (argument == "--nms-threshold") {
      args.config.nms_threshold_override = std::stod(require_value(argc, argv, index));
    } else if (argument == "--frame-source") {
      args.config.frame_source = require_value(argc, argv, index);
    } else if (argument == "--camera-device") {
      args.config.camera_device = require_value(argc, argv, index);
    } else if (argument == "--camera-width") {
      args.config.camera_width = std::stoi(require_value(argc, argv, index));
    } else if (argument == "--camera-height") {
      args.config.camera_height = std::stoi(require_value(argc, argv, index));
    } else if (argument == "--camera-fps") {
      args.config.camera_fps = std::stoi(require_value(argc, argv, index));
    } else if (argument == "--camera-pixel-format") {
      args.config.camera_pixel_format = require_value(argc, argv, index);
    } else if (argument == "--hp60c-url") {
      args.config.hp60c_url = require_value(argc, argv, index);
    } else if (argument == "--hp60c-snapshot-path") {
      args.config.hp60c_snapshot_path = require_value(argc, argv, index);
    } else if (argument == "--hp60c-health-path") {
      args.config.hp60c_health_path = require_value(argc, argv, index);
    } else if (argument == "--snapshot-source") {
      args.config.snapshot_source = require_value(argc, argv, index);
    } else if (argument == "--enable-camera-thread") {
      const auto value = require_value(argc, argv, index);
      args.config.enable_camera_thread = !(value == "false" || value == "0" || value == "no");
    } else if (argument == "--camera-open-timeout-ms") {
      args.config.camera_open_timeout_ms = std::stoi(require_value(argc, argv, index));
    } else if (argument == "--camera-read-timeout-ms") {
      args.config.camera_read_timeout_ms = std::stoi(require_value(argc, argv, index));
    } else if (argument == "--stale-frame-timeout-ms") {
      args.config.stale_frame_timeout_ms = std::stoi(require_value(argc, argv, index));
    } else if (argument == "--camera-reconnect-failure-threshold") {
      args.config.reconnect_failure_threshold = std::stoi(require_value(argc, argv, index));
    } else if (argument == "--camera-reconnect-initial-ms") {
      args.config.reconnect_initial_ms = std::stoi(require_value(argc, argv, index));
    } else if (argument == "--camera-reconnect-max-ms") {
      args.config.reconnect_max_ms = std::stoi(require_value(argc, argv, index));
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
         "  --model-dir <路径>       标准模型目录，必须包含 model.rknn 和 model.yaml\n"
         "  --test-image <路径>      本地 P6 PPM 测试图片（无 OpenCV 默认构建）\n"
         "  --save-debug-output <目录> 预留轻量调试输出目录\n"
         "  --preprocess-backend <cpu|rga|auto> 预处理后端，默认 cpu；启用 RGA 构建后可设为 rga\n"
         "  --rga-mode <resize_rgb>    RGA 预处理模式，当前支持 resize_rgb\n"
         "  --dump-rknn-io           启动时打印 RKNN 输入输出属性\n"
         "  --score-threshold <值>   覆盖模型置信度阈值\n"
         "  --nms-threshold <值>     覆盖模型 NMS 阈值\n"
         "  --frame-source <类型>     mock、test_image、v4l2 或 hp60c_bridge，默认 mock\n"
         "  --camera-device <设备>    V4L2 设备，默认 /dev/video0\n"
         "  --camera-width <宽>       V4L2 宽度，默认 640\n"
         "  --camera-height <高>      V4L2 高度，默认 480\n"
         "  --camera-fps <帧率>       V4L2 帧率，默认 30\n"
         "  --camera-pixel-format <格式> V4L2 像素格式，当前支持 YUYV\n"
         "  --hp60c-url <URL>         HP60C SDK HTTP Bridge 地址，默认 http://127.0.0.1:18181\n"
         "  --hp60c-snapshot-path <路径> HP60C 快照路径，默认 /stream/snapshot.jpg\n"
         "  --hp60c-health-path <路径> HP60C 健康检查路径，默认 /health\n"
         "  --snapshot-source <来源>  latest_frame 或 mock，当前 JPEG 编码默认仍为 mock\n"
         "  --enable-camera-thread <true/false> 是否开启取流线程，默认 true\n"
         "  --camera-open-timeout-ms <毫秒> 摄像头打开超时占位参数\n"
         "  --camera-read-timeout-ms <毫秒> 摄像头读取超时，默认 1000\n"
         "  --stale-frame-timeout-ms <毫秒> 实时帧过期阈值，默认 3000\n"
         "  --camera-reconnect-failure-threshold <次数> 连续失败多少次后重建帧源，默认 3\n"
         "  --camera-reconnect-initial-ms <毫秒> 自动重连初始退避，默认 200\n"
         "  --camera-reconnect-max-ms <毫秒> 自动重连最大退避，默认 2000\n"
         "  --help                    显示帮助\n";
}

}  // namespace visionops::runtime
