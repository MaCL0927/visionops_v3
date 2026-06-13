#include "visionops_runtime/postprocess_detect.hpp"

namespace visionops::runtime {

std::string make_detection_payload_json() {
  return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"object","score":0.94,"bbox_xyxy":[420.5,180.0,860.0,790.5],"center_xy":[640.25,485.25]}],"final_decision":{"code":"OBJECT_FOUND","label":"object","ok":true,"reason":"Mock 检测结果"})json";
}

std::string make_classification_payload_json() {
  return R"json(,"classifications":[{"class_id":0,"class_name":"ok","score":0.92,"rank":1}],"final_decision":{"code":"OK","label":"ok","ok":true,"reason":"Mock 分类结果"})json";
}

std::string make_roi_classification_payload_json() {
  return R"json(,"detections":[{"id":"det-mock-001","class_id":0,"class_name":"part","score":0.96,"bbox_xyxy":[760.0,210.0,1180.0,870.0],"attributes":{"roi_mode":"relative_box"}}],"classifications":[{"class_id":1,"class_name":"ng","score":0.93,"rank":1,"detection_id":"det-mock-001"}],"final_decision":{"code":"NG","label":"ng","ok":false,"reason":"Mock ROI 分类结果"})json";
}

}  // namespace visionops::runtime
