#pragma once

#include <string>

#include "visionops_runtime/app_config.hpp"

namespace visionops::runtime {

struct CliArgs {
  AppConfig config;
  bool show_help{false};
};

CliArgs parse_cli_args(int argc, char* argv[]);
std::string cli_help_text(const std::string& program);

}  // namespace visionops::runtime
