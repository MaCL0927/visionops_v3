#pragma once

#include <cstdint>
#include <mutex>
#include <string>

namespace visionops::runtime {

struct RoiFilterConfig {
  bool enabled{false};
  double x1{0.0};
  double y1{0.0};
  double x2{1.0};
  double y2{1.0};
  std::string mode{"center"};
  std::uint64_t updated_at_ms{0};
};

bool roi_contains_center(
    const RoiFilterConfig& roi,
    double center_x,
    double center_y,
    int image_width,
    int image_height);

class RoiFilterStore {
 public:
  explicit RoiFilterStore(std::string path = {});

  RoiFilterConfig snapshot() const;
  std::string json(int image_width = 0, int image_height = 0) const;
  std::string value_json(int image_width = 0, int image_height = 0) const;
  bool update_from_json(const std::string& request_body, std::string& error_message);
  const std::string& path() const;

 private:
  void load();
  bool save(std::string& error_message) const;

  std::string path_;
  mutable std::mutex mutex_;
  RoiFilterConfig config_;
};

}  // namespace visionops::runtime
