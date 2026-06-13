#include "visionops_runtime/postprocess_seg.hpp"

namespace visionops::runtime {

std::string make_segmentation_payload_json() {
  return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"surface_region","score":0.89,"bbox_xyxy":[300.0,220.0,980.0,820.0],"mask":{"encoding":"polygon","size":[1080,1920],"polygon":[[[320.0,250.0],[940.0,230.0],[970.0,790.0],[350.0,810.0]]]}}],"measurements":{"mask_area_px":337900,"coverage_ratio":0.1629})json";
}

}  // namespace visionops::runtime
