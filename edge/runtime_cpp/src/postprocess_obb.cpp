#include "visionops_runtime/postprocess_obb.hpp"

namespace visionops::runtime {

std::string make_obb_payload_json() {
  return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"rotated_object","score":0.91,"bbox_xyxy":[420.0,220.0,900.0,720.0],"center_xy":[660.0,470.0],"obb":{"cx":660.0,"cy":470.0,"w":430.0,"h":220.0,"angle_deg":-12.0,"points":[[427.0,406.0],[847.0,316.0],[893.0,534.0],[473.0,624.0]]}}],"final_decision":{"code":"ORIENTATION_OK","label":"aligned","ok":true,"reason":"Mock OBB 结果"})json";
}

}  // namespace visionops::runtime
