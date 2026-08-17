"""M39.2 left-arm robot pose transform for the foam-ring grasp task.

Transform notation used throughout this module is ``T_A_B``: the homogeneous
matrix maps coordinates expressed in frame B into frame A.

M39.2 intentionally consumes the M38.6 visual grasp frame instead of rebuilding
any grasp geometry.  The chain is::

    T_base_grasp = T_base_camera @ T_camera_grasp
    T_base_hand_tcp = T_base_grasp @ T_grasp_hand_tcp
    T_base_left_link7 = T_base_hand_tcp @ T_hand_tcp_left_link7

The final hand_tcp->left_link7 transform is optional until an exact designer/
URDF fixed transform is supplied.  It must never be inferred from the old M38
collision approximation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np  # type: ignore


class RobotPoseTransformError(RuntimeError):
    """A transform/configuration contract is unsafe or internally inconsistent."""


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RobotPoseTransformError(f"{name}必须是对象")
    return value


def _resolve_path(config_path: Path, value: Any, name: str) -> Path:
    if not value:
        raise RobotPoseTransformError(f"{name}未配置")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix4(value: Any, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except Exception as error:  # pragma: no cover - numpy message varies
        raise RobotPoseTransformError(f"{name}无法转换为4x4矩阵: {error}") from error
    if matrix.shape != (4, 4):
        raise RobotPoseTransformError(f"{name}必须是4x4矩阵，当前shape={matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise RobotPoseTransformError(f"{name}包含NaN/Inf")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise RobotPoseTransformError(f"{name}最后一行必须为[0,0,0,1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5):
        raise RobotPoseTransformError(f"{name}旋转部分不是正交矩阵")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=2e-5):
        raise RobotPoseTransformError(f"{name}旋转行列式必须为+1，当前={determinant:.8f}")
    return matrix


def _rows(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix.tolist()]


def _vector3(value: Any, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception as error:  # pragma: no cover
        raise RobotPoseTransformError(f"{name}无法转换为3维向量: {error}") from error
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise RobotPoseTransformError(f"{name}必须是有限3维向量")
    return vector


def _rotation_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized xyzw quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / max(scale, 1e-12)
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / max(scale, 1e-12)
            qz = (rotation[0, 2] + rotation[2, 0]) / max(scale, 1e-12)
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / max(scale, 1e-12)
            qx = (rotation[0, 1] + rotation[1, 0]) / max(scale, 1e-12)
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / max(scale, 1e-12)
        else:
            scale = math.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / max(scale, 1e-12)
            qx = (rotation[0, 2] + rotation[2, 0]) / max(scale, 1e-12)
            qy = (rotation[1, 2] + rotation[2, 1]) / max(scale, 1e-12)
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise RobotPoseTransformError("旋转矩阵无法转换为有效四元数")
    quaternion /= norm
    # q and -q represent the same orientation. A deterministic sign makes logs
    # and robot-side regression checks stable.
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def _pose_document(matrix: np.ndarray, frame_id: str) -> Dict[str, Any]:
    position_mm = matrix[:3, 3]
    quaternion = _rotation_to_quaternion_xyzw(matrix[:3, :3])
    return {
        "frame_id": frame_id,
        "position_mm": [float(value) for value in position_mm],
        "position_m": [float(value) / 1000.0 for value in position_mm],
        "quaternion_xyzw": [float(value) for value in quaternion],
        "rotation_matrix_rows": [
            [float(value) for value in row] for row in matrix[:3, :3].tolist()
        ],
    }


@dataclass(frozen=True)
class RobotPoseTransformConfig:
    enabled: bool
    arm: str
    base_frame_id: str
    visual_grasp_frame_id: str
    hand_tcp_frame_id: str
    flange_frame_id: str
    camera_frame_aliases: tuple[str, ...]
    calibration_file: Path
    calibration_sha256: str | None
    required_calibration_type: str
    required_source_frame: str
    required_target_frame: str
    required_selected_method: str | None
    allowed_quality_status: tuple[str, ...]
    visual_to_tcp_enabled: bool
    T_grasp_hand_tcp_mm: np.ndarray | None
    visual_to_tcp_origin_policy: str
    flange_enabled: bool
    T_hand_tcp_flange_mm: np.ndarray | None


class RobotPoseTransformer:
    """Strict M39.2 camera->base->left-hand pose transformer."""

    def __init__(
        self,
        config: RobotPoseTransformConfig,
        calibration: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.calibration = dict(calibration)
        self.T_base_camera_mm = (
            _matrix4(self.calibration.get("T_base_camera_mm"), "handeye.T_base_camera_mm")
            if config.enabled
            else np.eye(4, dtype=np.float64)
        )
        self.selected_method = str(self.calibration.get("selected_method") or "")
        self.quality_status = str(self.calibration.get("quality_status") or "")
        self.sample_count_used = int(self.calibration.get("sample_count_used") or 0)
        self.metrics = (
            dict(self.calibration.get("metrics"))
            if isinstance(self.calibration.get("metrics"), Mapping)
            else {}
        )

    @classmethod
    def from_mapping(
        cls,
        raw_config: Mapping[str, Any],
        config_path: Path,
    ) -> "RobotPoseTransformer":
        section = raw_config.get("robot_pose_transform") or {}
        section = _as_mapping(section, "robot_pose_transform")
        enabled = bool(section.get("enabled", False))
        arm = str(section.get("arm") or "").strip().lower()
        if enabled and arm != "left":
            raise RobotPoseTransformError(
                f"M39.2当前版本只允许左手，robot_pose_transform.arm={arm!r}"
            )

        frames = _as_mapping(section.get("frames") or {}, "robot_pose_transform.frames")
        base_frame_id = str(frames.get("base") or "base_link")
        visual_grasp_frame_id = str(frames.get("visual_grasp") or "m38_6_visual_grasp")
        hand_tcp_frame_id = str(frames.get("hand_tcp") or "hand_tcp_link")
        flange_frame_id = str(frames.get("flange") or "left_link7")
        aliases_raw = frames.get("camera_aliases") or ["camera_color_optical_frame", "color_camera"]
        if not isinstance(aliases_raw, Sequence) or isinstance(aliases_raw, (str, bytes)):
            raise RobotPoseTransformError("robot_pose_transform.frames.camera_aliases必须是数组")
        camera_frame_aliases = tuple(str(value) for value in aliases_raw if str(value))
        if not camera_frame_aliases:
            raise RobotPoseTransformError("camera_aliases不能为空")

        # Generic/unit-test configs that predate M39.2 are allowed to omit the
        # entire section. They retain the M38 behavior with the transform layer
        # explicitly disabled. Production M39.2 line.yaml enables it and then
        # all left-arm/calibration contracts below become mandatory.
        if not enabled:
            config = RobotPoseTransformConfig(
                enabled=False,
                arm=arm or "left",
                base_frame_id=base_frame_id,
                visual_grasp_frame_id=visual_grasp_frame_id,
                hand_tcp_frame_id=hand_tcp_frame_id,
                flange_frame_id=flange_frame_id,
                camera_frame_aliases=camera_frame_aliases,
                calibration_file=config_path,
                calibration_sha256=None,
                required_calibration_type="eye_to_hand",
                required_source_frame="color_camera",
                required_target_frame=base_frame_id,
                required_selected_method=None,
                allowed_quality_status=(),
                visual_to_tcp_enabled=False,
                T_grasp_hand_tcp_mm=None,
                visual_to_tcp_origin_policy="disabled",
                flange_enabled=False,
                T_hand_tcp_flange_mm=None,
            )
            return cls(config, {})

        handeye = _as_mapping(section.get("handeye") or {}, "robot_pose_transform.handeye")
        calibration_file = _resolve_path(
            config_path, handeye.get("calibration_file"), "robot_pose_transform.handeye.calibration_file"
        )
        calibration_sha256 = str(handeye.get("sha256") or "").strip().lower() or None
        required_calibration_type = str(handeye.get("required_calibration_type") or "eye_to_hand")
        required_source_frame = str(handeye.get("required_source_frame") or "color_camera")
        required_target_frame = str(handeye.get("required_target_frame") or base_frame_id)
        required_selected_method = str(handeye.get("required_selected_method") or "").strip() or None
        quality_raw = handeye.get("allowed_quality_status") or ["PASS", "PROVISIONAL_PASS"]
        if not isinstance(quality_raw, Sequence) or isinstance(quality_raw, (str, bytes)):
            raise RobotPoseTransformError("allowed_quality_status必须是数组")
        allowed_quality_status = tuple(str(value) for value in quality_raw)

        visual = _as_mapping(
            section.get("visual_grasp_to_hand_tcp") or {},
            "robot_pose_transform.visual_grasp_to_hand_tcp",
        )
        visual_to_tcp_enabled = bool(visual.get("enabled", False))
        T_grasp_hand_tcp_mm = None
        if visual_to_tcp_enabled:
            T_grasp_hand_tcp_mm = _matrix4(
                visual.get("T_grasp_hand_tcp_rows"),
                "robot_pose_transform.visual_grasp_to_hand_tcp.T_grasp_hand_tcp_rows",
            )
        origin_policy = str(visual.get("origin_policy") or "exact_transform_not_supplied")

        flange = _as_mapping(
            section.get("hand_tcp_to_flange") or {},
            "robot_pose_transform.hand_tcp_to_flange",
        )
        flange_enabled = bool(flange.get("enabled", False))
        T_hand_tcp_flange_mm = None
        if flange_enabled:
            T_hand_tcp_flange_mm = _matrix4(
                flange.get("T_hand_tcp_flange_rows"),
                "robot_pose_transform.hand_tcp_to_flange.T_hand_tcp_flange_rows",
            )

        config = RobotPoseTransformConfig(
            enabled=enabled,
            arm=arm,
            base_frame_id=base_frame_id,
            visual_grasp_frame_id=visual_grasp_frame_id,
            hand_tcp_frame_id=hand_tcp_frame_id,
            flange_frame_id=flange_frame_id,
            camera_frame_aliases=camera_frame_aliases,
            calibration_file=calibration_file,
            calibration_sha256=calibration_sha256,
            required_calibration_type=required_calibration_type,
            required_source_frame=required_source_frame,
            required_target_frame=required_target_frame,
            required_selected_method=required_selected_method,
            allowed_quality_status=allowed_quality_status,
            visual_to_tcp_enabled=visual_to_tcp_enabled,
            T_grasp_hand_tcp_mm=T_grasp_hand_tcp_mm,
            visual_to_tcp_origin_policy=origin_policy,
            flange_enabled=flange_enabled,
            T_hand_tcp_flange_mm=T_hand_tcp_flange_mm,
        )
        calibration = cls._load_and_validate_calibration(config)
        return cls(config, calibration)

    @staticmethod
    def _load_and_validate_calibration(
        config: RobotPoseTransformConfig,
    ) -> Mapping[str, Any]:
        if not config.calibration_file.exists():
            raise RobotPoseTransformError(
                f"左手手眼标定文件不存在: {config.calibration_file}"
            )
        if config.calibration_sha256:
            actual = _sha256(config.calibration_file)
            if actual.lower() != config.calibration_sha256.lower():
                raise RobotPoseTransformError(
                    "左手手眼标定文件SHA256不匹配，拒绝继续: "
                    f"expected={config.calibration_sha256}, actual={actual}"
                )
        try:
            payload = json.loads(config.calibration_file.read_text(encoding="utf-8"))
        except Exception as error:
            raise RobotPoseTransformError(
                f"无法读取左手手眼标定文件: {config.calibration_file}: {error}"
            ) from error
        payload = _as_mapping(payload, "handeye calibration")

        actual_arm = str(payload.get("robot_arm") or "").strip().lower()
        if actual_arm != "left" or actual_arm != config.arm:
            raise RobotPoseTransformError(
                "手眼标定机械臂不匹配，M39.2拒绝加载: "
                f"config_arm={config.arm!r}, calibration_robot_arm={actual_arm!r}"
            )
        actual_type = str(payload.get("calibration_type") or "")
        if actual_type != config.required_calibration_type:
            raise RobotPoseTransformError(
                f"标定类型不匹配: expected={config.required_calibration_type!r}, actual={actual_type!r}"
            )
        actual_source = str(payload.get("source_frame") or "")
        if actual_source != config.required_source_frame:
            raise RobotPoseTransformError(
                f"标定source_frame不匹配: expected={config.required_source_frame!r}, actual={actual_source!r}"
            )
        actual_target = str(payload.get("target_frame") or "")
        if actual_target != config.required_target_frame:
            raise RobotPoseTransformError(
                f"标定target_frame不匹配: expected={config.required_target_frame!r}, actual={actual_target!r}"
            )
        selected = str(payload.get("selected_method") or "")
        if config.required_selected_method and selected != config.required_selected_method:
            raise RobotPoseTransformError(
                f"标定selected_method不匹配: expected={config.required_selected_method!r}, actual={selected!r}"
            )
        quality = str(payload.get("quality_status") or "")
        if config.allowed_quality_status and quality not in config.allowed_quality_status:
            raise RobotPoseTransformError(
                f"标定quality_status不允许: {quality!r}, allowed={config.allowed_quality_status}"
            )
        _matrix4(payload.get("T_base_camera_mm"), "handeye.T_base_camera_mm")
        return payload

    def status(self) -> Dict[str, Any]:
        config = self.config
        return {
            "stage": "M39.2",
            "enabled": bool(config.enabled),
            "arm": config.arm,
            "frames": {
                "base": config.base_frame_id,
                "camera_aliases": list(config.camera_frame_aliases),
                "visual_grasp": config.visual_grasp_frame_id,
                "hand_tcp": config.hand_tcp_frame_id,
                "flange": config.flange_frame_id,
            },
            "handeye": {
                "calibration_file": str(config.calibration_file),
                "robot_arm": str(self.calibration.get("robot_arm") or ""),
                "selected_method": self.selected_method,
                "quality_status": self.quality_status,
                "sample_count_used": self.sample_count_used,
                "source_frame": str(self.calibration.get("source_frame") or ""),
                "target_frame": str(self.calibration.get("target_frame") or ""),
                "sha256_verified": bool(config.calibration_sha256),
            },
            "visual_grasp_to_hand_tcp": {
                "configured": bool(config.visual_to_tcp_enabled),
                "origin_policy": config.visual_to_tcp_origin_policy,
                "axis_mapping_verified": {
                    "grasp_+X": "hand_tcp_+Z",
                    "grasp_+Y": "hand_tcp_-Y",
                    "grasp_+Z": "hand_tcp_+X",
                },
                "T_grasp_hand_tcp_rows": (
                    _rows(config.T_grasp_hand_tcp_mm)
                    if config.T_grasp_hand_tcp_mm is not None else None
                ),
                "reason": None if config.visual_to_tcp_enabled else (
                    "axis rotation is known, but the exact grasp-origin to hand_tcp origin translation "
                    "is not present in the supplied M38.6.2/calibration artifacts; hand_tcp output is gated"
                ),
            },
            "hand_tcp_to_flange": {
                "configured": bool(config.flange_enabled),
                "reason": None if config.flange_enabled else (
                    "exact T_hand_tcp_left_link7 is not supplied in M38.6.2/calibration package; "
                    "M39.2 will not infer it from collision geometry"
                ),
            },
        }

    def transform_candidate(self, candidate: Mapping[str, Any] | None) -> Dict[str, Any]:
        config = self.config
        base = {
            "schema_version": "1.0",
            "stage": "M39.2_left_robot_pose_transform",
            "arm": "left",
            "transform_notation": "T_A_B maps coordinates in B into A",
            "base_frame_id": config.base_frame_id,
            "hand_tcp_frame_id": config.hand_tcp_frame_id,
            "flange_frame_id": config.flange_frame_id,
            "calibration": {
                "robot_arm": str(self.calibration.get("robot_arm") or ""),
                "selected_method": self.selected_method,
                "quality_status": self.quality_status,
                "sample_count_used": self.sample_count_used,
                "T_base_camera_mm": _rows(self.T_base_camera_mm),
                "metrics": self.metrics,
            },
        }
        if not config.enabled:
            return {**base, "status": "disabled", "reason": "robot_pose_transform.enabled=false"}
        if not isinstance(candidate, Mapping):
            return {**base, "status": "not_applicable", "reason": "no_robot_candidate"}
        frame = candidate.get("grasp_frame_camera")
        if not isinstance(frame, Mapping):
            return {
                **base,
                "status": "not_applicable",
                "reason": "candidate_has_no_grasp_frame_camera",
                "grasp_branch": candidate.get("grasp_branch"),
            }

        camera_frame = str(frame.get("coordinate_frame") or "")
        if camera_frame not in config.camera_frame_aliases:
            raise RobotPoseTransformError(
                "M38.6 grasp camera frame与M39.2标定frame不兼容: "
                f"candidate={camera_frame!r}, aliases={config.camera_frame_aliases}"
            )
        length_unit = str(frame.get("length_unit") or "mm").lower()
        if length_unit != "mm":
            raise RobotPoseTransformError(
                f"M39.2要求M38.6 T_camera_grasp使用mm，当前={length_unit!r}"
            )

        T_camera_grasp = _matrix4(frame.get("T_camera_grasp_rows"), "candidate.T_camera_grasp")
        T_base_grasp = self.T_base_camera_mm @ T_camera_grasp

        pregrasp_camera_mm = _vector3(
            candidate.get("pregrasp_center_camera_mm"), "candidate.pregrasp_center_camera_mm"
        )
        T_camera_pregrasp = T_camera_grasp.copy()
        T_camera_pregrasp[:3, 3] = pregrasp_camera_mm
        T_base_pregrasp = self.T_base_camera_mm @ T_camera_pregrasp

        entry_camera_raw = candidate.get("entry_center_camera_mm")
        T_camera_entry = None
        T_base_entry = None
        if entry_camera_raw is not None:
            entry_camera_mm = _vector3(entry_camera_raw, "candidate.entry_center_camera_mm")
            T_camera_entry = T_camera_grasp.copy()
            T_camera_entry[:3, 3] = entry_camera_mm
            T_base_entry = self.T_base_camera_mm @ T_camera_entry

        result: Dict[str, Any] = {
            **base,
            "status": "base_grasp_ready",
            "camera_frame_id": camera_frame,
            "calibration_source_frame": str(self.calibration.get("source_frame") or ""),
            "camera_frame_alias_match": True,
            "grasp_branch": candidate.get("grasp_branch"),
            "chain": [
                "T_base_grasp = T_base_camera @ T_camera_grasp",
                "T_base_hand_tcp = T_base_grasp @ T_grasp_hand_tcp (when exact transform is configured)",
                "T_base_left_link7 = T_base_hand_tcp @ T_hand_tcp_left_link7 (when configured)",
            ],
            "static_tool_transform": {
                "configured": bool(config.visual_to_tcp_enabled),
                "T_grasp_hand_tcp_mm": (
                    _rows(config.T_grasp_hand_tcp_mm)
                    if config.T_grasp_hand_tcp_mm is not None else None
                ),
                "axis_mapping_verified": {
                    "grasp_+X": "hand_tcp_+Z",
                    "grasp_+Y": "hand_tcp_-Y",
                    "grasp_+Z": "hand_tcp_+X",
                },
                "origin_policy": config.visual_to_tcp_origin_policy,
            },
            "grasp": {
                "T_camera_grasp_mm": _rows(T_camera_grasp),
                "T_base_grasp_mm": _rows(T_base_grasp),
                "visual_grasp_pose_base": _pose_document(T_base_grasp, config.base_frame_id),
            },
            "pregrasp": {
                "source": "candidate.pregrasp_center_camera_mm with grasp orientation preserved",
                "T_camera_pregrasp_mm": _rows(T_camera_pregrasp),
                "T_base_pregrasp_mm": _rows(T_base_pregrasp),
                "visual_grasp_pose_base": _pose_document(T_base_pregrasp, config.base_frame_id),
            },
            "entry": (
                {
                    "source": "candidate.entry_center_camera_mm with grasp orientation preserved",
                    "T_camera_entry_mm": _rows(T_camera_entry),
                    "T_base_entry_mm": _rows(T_base_entry),
                    "visual_grasp_pose_base": _pose_document(T_base_entry, config.base_frame_id),
                }
                if T_camera_entry is not None and T_base_entry is not None
                else {"available": False, "reason": "candidate_has_no_entry_center_camera_mm"}
            ),
            "hand_tcp": {
                "available": False,
                "frame_id": config.hand_tcp_frame_id,
                "reason": (
                    "exact T_grasp_hand_tcp is not configured; verified axis mapping alone is insufficient "
                    "to infer the TCP origin translation"
                ),
            },
            "flange": {
                "available": False,
                "frame_id": config.flange_frame_id,
                "reason": (
                    "hand_tcp pose is unavailable and exact T_hand_tcp_left_link7 is not configured; "
                    "no M38 collision-model distance is used"
                ),
            },
        }

        if config.visual_to_tcp_enabled and config.T_grasp_hand_tcp_mm is not None:
            T_base_hand_tcp = T_base_grasp @ config.T_grasp_hand_tcp_mm
            T_base_hand_tcp_pregrasp = T_base_pregrasp @ config.T_grasp_hand_tcp_mm
            T_base_hand_tcp_entry = (
                T_base_entry @ config.T_grasp_hand_tcp_mm
                if T_base_entry is not None
                else None
            )
            result["status"] = "ok"
            result["grasp"]["T_base_hand_tcp_mm"] = _rows(T_base_hand_tcp)
            result["grasp"]["hand_tcp_pose_base"] = _pose_document(
                T_base_hand_tcp, config.base_frame_id
            )
            result["pregrasp"]["T_base_hand_tcp_mm"] = _rows(T_base_hand_tcp_pregrasp)
            result["pregrasp"]["hand_tcp_pose_base"] = _pose_document(
                T_base_hand_tcp_pregrasp, config.base_frame_id
            )
            if T_base_hand_tcp_entry is not None and isinstance(result.get("entry"), dict):
                result["entry"]["T_base_hand_tcp_mm"] = _rows(T_base_hand_tcp_entry)
                result["entry"]["hand_tcp_pose_base"] = _pose_document(
                    T_base_hand_tcp_entry, config.base_frame_id
                )
            result["hand_tcp"] = {
                "available": True,
                "frame_id": config.hand_tcp_frame_id,
                "grasp_pose_base": _pose_document(T_base_hand_tcp, config.base_frame_id),
                "pregrasp_pose_base": _pose_document(T_base_hand_tcp_pregrasp, config.base_frame_id),
                "entry_pose_base": (
                    _pose_document(T_base_hand_tcp_entry, config.base_frame_id)
                    if T_base_hand_tcp_entry is not None else None
                ),
            }

            if config.flange_enabled and config.T_hand_tcp_flange_mm is not None:
                T_base_flange = T_base_hand_tcp @ config.T_hand_tcp_flange_mm
                T_base_flange_pregrasp = T_base_hand_tcp_pregrasp @ config.T_hand_tcp_flange_mm
                T_base_flange_entry = (
                    T_base_hand_tcp_entry @ config.T_hand_tcp_flange_mm
                    if T_base_hand_tcp_entry is not None
                    else None
                )
                result["flange"] = {
                    "available": True,
                    "frame_id": config.flange_frame_id,
                    "T_hand_tcp_left_link7_mm": _rows(config.T_hand_tcp_flange_mm),
                    "grasp": {
                        "T_base_left_link7_mm": _rows(T_base_flange),
                        "pose_base": _pose_document(T_base_flange, config.base_frame_id),
                    },
                    "pregrasp": {
                        "T_base_left_link7_mm": _rows(T_base_flange_pregrasp),
                        "pose_base": _pose_document(T_base_flange_pregrasp, config.base_frame_id),
                    },
                    "entry": (
                        {
                            "T_base_left_link7_mm": _rows(T_base_flange_entry),
                            "pose_base": _pose_document(T_base_flange_entry, config.base_frame_id),
                        }
                        if T_base_flange_entry is not None
                        else {"available": False}
                    ),
                }
        return result
