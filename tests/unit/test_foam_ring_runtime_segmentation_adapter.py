from __future__ import annotations

import pytest

from production.foam_ring_grasp.tasks.foam_ring_grasp_vision.segmentation import (
    runtime_result_to_segmentation_instances,
)


def _result(*, source: str = "proto", width: int = 64, height: int = 48):
    return {
        "message_type": "inference_result",
        "status": "ok",
        "task_type": "segmentation",
        "image": {"width": width, "height": height},
        "detections": [
            {
                "id": "seg-rknn-001",
                "class_id": 0,
                "class_name": "foam_ring",
                "score": 0.92,
                "mask": {
                    "source": source,
                    "size": [height, width],
                    "polygon": [[[10.2, 8.4], [30.6, 8.4], [30.6, 28.8], [10.2, 28.8]]],
                },
            }
        ],
    }


def test_runtime_proto_polygon_becomes_boolean_mask():
    adapted = runtime_result_to_segmentation_instances(_result(), (48, 64))
    assert adapted.accepted_count == 1
    assert adapted.rejected_count == 0
    assert adapted.polygon_point_count == 4
    instance = adapted.instances[0]
    assert instance.class_name == "foam_ring"
    assert instance.mask.dtype.name == "bool"
    assert instance.mask.shape == (48, 64)
    assert instance.area_px > 300
    assert instance.bbox_xyxy == (10, 8, 32, 30)


def test_bbox_fallback_is_rejected_from_geometry():
    adapted = runtime_result_to_segmentation_instances(
        _result(source="bbox_fallback"),
        (48, 64),
    )
    assert adapted.accepted_count == 0
    assert adapted.rejected_count == 1
    assert adapted.rejected[0]["reason"] == "bbox_fallback_rejected"


def test_runtime_image_and_exact_rgbd_shape_must_match():
    with pytest.raises(ValueError, match="尺寸不一致"):
        runtime_result_to_segmentation_instances(_result(), (47, 64))


def test_runtime_polygon_can_be_transformed_to_input_roi_coordinates():
    result = _result()
    adapted = runtime_result_to_segmentation_instances(
        result,
        (48, 64),
        geometry_roi_xyxy=(8, 4, 40, 36),
    )
    assert adapted.accepted_count == 1
    instance = adapted.instances[0]
    assert instance.mask.shape == (32, 32)
    assert instance.bbox_xyxy == (2, 4, 24, 26)
