#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

}  // namespace visionops::runtime
