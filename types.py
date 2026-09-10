from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class Phase(str, Enum):
    INITIAL = "initial"
    PREGRASP = "pregrasp"
    APPROACH = "approach"
    CLOSE = "close"
    HOLD = "hold"
    LIFT = "lift"
    LIFT_HOLD = "lift_hold"
    PLACE = "place"
    RELEASE = "release"


class Outcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    INVALID = "INVALID"


@dataclass
class Pose:
    position: np.ndarray
    quaternion_wxyz: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.position, self.quaternion_wxyz]).astype(np.float32)


@dataclass
class NominalGrasp:
    left: Pose
    right: Pose
    object_center: np.ndarray
    object_dimensions: np.ndarray
    object_local_axis: int
    side_axis_world: np.ndarray
    clearance_m: float
    source: str = "bbox_local_coordinate_system"

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left.as_vector().tolist(),
            "right": self.right.as_vector().tolist(),
            "object_center": self.object_center.tolist(),
            "object_dimensions": self.object_dimensions.tolist(),
            "object_local_axis": self.object_local_axis,
            "side_axis_world": self.side_axis_world.tolist(),
            "clearance_m": self.clearance_m,
            "source": self.source,
        }


@dataclass
class EpisodeResult:
    actual_outcome: str
    success_flags: dict[str, bool]
    failure_reason: str | None
    failure_phase: str | None
    invalid_reason: str | None
    frame_count: int
    duration_s: float
    camera_missing_frames: int = 0
    controller_failures: int = 0
    ik_failures: int = 0
    reset_failures: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "actual_outcome": self.actual_outcome,
            "success_flags": self.success_flags,
            "failure_reason": self.failure_reason,
            "failure_phase": self.failure_phase,
            "invalid_reason": self.invalid_reason,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "camera_missing_frames": self.camera_missing_frames,
            "controller_failures": self.controller_failures,
            "ik_failures": self.ik_failures,
            "reset_failures": self.reset_failures,
        }
