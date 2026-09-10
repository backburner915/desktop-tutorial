from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DATASET_VERSION = "r1-bimanual-raw-v0.1"
INSTRUCTION = "Grasp the object with both hands, lift it, hold it steadily, and place it down."
DATASET_FPS = 10.0
ACTION_DIM = 16

# This list is intentionally explicit and is verified against the live
# articulation before any episode is executed.  It is not inferred from the
# conceptual labels at runtime.
ACTION_JOINT_NAMES = [
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_gripper_axis1",
    "left_gripper_axis2",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_gripper_axis1",
    "right_gripper_axis2",
]

FAILURE_MODES = {
    "none",
    "left_grasp_position_error",
    "right_grasp_position_error",
    "both_grasp_position_error",
    "left_approach_angle_error",
    "right_approach_angle_error",
    "asymmetric_approach_error",
    "left_gripper_underclose",
    "right_gripper_underclose",
    "left_gripper_early",
    "right_gripper_early",
    "left_gripper_late",
    "right_gripper_late",
    "phase_mismatch",
    "left_contact_missing",
    "right_contact_missing",
    "insufficient_lift",
    "slip",
}

COORDINATION_MODES = {"synchronous", "left_leads", "right_leads"}


def _array(value: Any, size: int, name: str) -> list[float]:
    if value is None:
        raise ValueError(f"{name} is required")
    values = [float(x) for x in value]
    if len(values) != size:
        raise ValueError(f"{name} must have length {size}, got {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return values


@dataclass
class ScenarioConfig:
    """All variables that define one reproducible episode design.

    ``left_start_pose`` and ``right_start_pose`` are six arm joint positions
    in the verified action-joint units (radians).  Gripper targets are the two
    real prismatic axis positions in meters.  Approach-angle offsets are XYZ
    Euler offsets in radians.  Grasp offsets are expressed in the local frame
    of the generated nominal grasp frame, in meters.
    """

    scenario_id: int
    intended_outcome: str
    object_position: list[float]
    object_orientation: list[float]  # roll, pitch, yaw, radians
    left_start_pose: list[float]
    right_start_pose: list[float]
    left_grasp_offset: list[float]
    right_grasp_offset: list[float]
    left_approach_angle: list[float]
    right_approach_angle: list[float]
    pregrasp_distance: float
    lift_vector: list[float]
    coordination_mode: str
    phase_offset: float
    gripper_target: dict[str, list[float]]
    failure_mode: str
    failure_magnitude: float
    seed: int
    instruction: str = INSTRUCTION
    target_object: str | None = None
    # Selects a reusable physical behavior implementation.  The legacy
    # value is retained only for old artifacts; all new R1 data must use the
    # contact-tracked fixed-base KLT transport behavior.
    behavior_template: str = "legacy"
    carry_vector: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if int(self.scenario_id) < 0:
            raise ValueError("scenario_id must be non-negative")
        if self.intended_outcome not in {"success", "failure"}:
            raise ValueError("intended_outcome must be success or failure")
        if self.failure_mode not in FAILURE_MODES:
            raise ValueError(f"unsupported failure_mode: {self.failure_mode}")
        if self.coordination_mode not in COORDINATION_MODES:
            raise ValueError(f"unsupported coordination_mode: {self.coordination_mode}")
        _array(self.object_position, 3, "object_position")
        _array(self.object_orientation, 3, "object_orientation")
        _array(self.left_start_pose, 6, "left_start_pose")
        _array(self.right_start_pose, 6, "right_start_pose")
        _array(self.left_grasp_offset, 3, "left_grasp_offset")
        _array(self.right_grasp_offset, 3, "right_grasp_offset")
        _array(self.left_approach_angle, 3, "left_approach_angle")
        _array(self.right_approach_angle, 3, "right_approach_angle")
        _array(self.lift_vector, 3, "lift_vector")
        _array(self.carry_vector, 3, "carry_vector")
        if not np.isfinite(float(self.pregrasp_distance)) or self.pregrasp_distance < 0:
            raise ValueError("pregrasp_distance must be finite and non-negative")
        if not np.isfinite(float(self.phase_offset)):
            raise ValueError("phase_offset must be finite")
        if not np.isfinite(float(self.failure_magnitude)) or self.failure_magnitude < 0:
            raise ValueError("failure_magnitude must be finite and non-negative")
        for side in ("left", "right"):
            self.gripper_target[side] = _array(
                self.gripper_target.get(side), 2, f"gripper_target.{side}"
            )
        if self.intended_outcome == "success" and self.failure_mode != "none":
            raise ValueError("success-oriented scenarios must use failure_mode='none'")
        if self.intended_outcome == "failure" and self.failure_mode == "none":
            raise ValueError("failure-oriented scenarios need one primary failure_mode")
        if self.behavior_template not in {"legacy", "physical_klt_transport_v1"}:
            raise ValueError(f"unsupported behavior_template: {self.behavior_template}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioConfig":
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "ScenarioConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )


def load_scene_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
