from __future__ import annotations

from typing import Any

import numpy as np

from .types import EpisodeResult, NominalGrasp, Phase


class SuccessEvaluator:
    """Phase-aware evaluator; height-only success is intentionally rejected."""

    def __init__(self, scene_cfg: dict[str, Any], scenario: Any, nominal: NominalGrasp, initial_object_pose: np.ndarray):
        self.scene_cfg = scene_cfg
        self.scenario = scenario
        self.nominal = nominal
        self.initial_object_pose = np.asarray(initial_object_pose, dtype=np.float64).copy()
        self.initial_position = self.initial_object_pose[:3]
        self.flags = {"grasp": False, "lift": False, "hold": False, "place": False, "release": False}
        self.reasons: list[tuple[str, str]] = []
        self.samples: list[dict[str, Any]] = []
        self.invalid_reason: str | None = None

    def update(self, sample: dict[str, Any]) -> None:
        if self.invalid_reason:
            return
        arrays = [sample["object_pose"], sample["object_velocity"], sample["left_eef_pose"], sample["right_eef_pose"]]
        if not all(np.isfinite(np.asarray(value)).all() for value in arrays):
            self.invalid_reason = "NaN_or_infinite_runtime_state"
            return
        object_position = np.asarray(sample["object_pose"][:3], dtype=np.float64)
        object_velocity = np.asarray(sample["object_velocity"][:3], dtype=np.float64)
        left_distance = float(np.linalg.norm(sample["left_eef_pose"][:3] - self.nominal.left.position))
        right_distance = float(np.linalg.norm(sample["right_eef_pose"][:3] - self.nominal.right.position))
        phase = str(sample["phase"])
        actual_grippers = np.asarray(sample["joint_position_16d"], dtype=np.float64)[[6, 7, 14, 15]]
        actual_close_threshold = float(self.scene_cfg.get("gripper_actual_close_threshold", 0.018))
        gripper_close = bool(np.all(actual_grippers < actual_close_threshold))
        commanded_grippers = sample.get("gripper_command")
        if commanded_grippers is not None:
            commanded_grippers = np.asarray(commanded_grippers, dtype=np.float64)[[6, 7, 14, 15]]
        close_command_threshold = float(self.scene_cfg.get("gripper_command_close_threshold", 0.005))
        close_commanded = commanded_grippers is not None and bool(
            np.all(commanded_grippers <= close_command_threshold)
        )
        open_command_threshold = float(self.scene_cfg.get("gripper_command_open_threshold", 0.025))
        open_commanded = commanded_grippers is not None and bool(
            np.all(commanded_grippers >= open_command_threshold)
        )
        contact_threshold = float(self.scene_cfg.get("contact_distance_threshold_m", 0.10))
        physical_closing = gripper_close or close_commanded
        left_contact = left_distance < contact_threshold and physical_closing
        right_contact = right_distance < contact_threshold and physical_closing
        lift_vector = np.asarray(self.scenario.lift_vector, dtype=np.float64)
        lift_axis = lift_vector / max(np.linalg.norm(lift_vector), 1e-9)
        lift_progress = float(np.dot(object_position - self.initial_position, lift_axis))
        object_bbox_min = sample.get("object_bbox_min")
        object_bbox_max = sample.get("object_bbox_max")
        support_bbox_min = sample.get("support_bbox_min")
        support_bbox_max = sample.get("support_bbox_max")
        if all(value is not None for value in (object_bbox_min, object_bbox_max, support_bbox_min, support_bbox_max)):
            object_bbox_min = np.asarray(object_bbox_min, dtype=np.float64)
            object_bbox_max = np.asarray(object_bbox_max, dtype=np.float64)
            support_bbox_min = np.asarray(support_bbox_min, dtype=np.float64)
            support_bbox_max = np.asarray(support_bbox_max, dtype=np.float64)
            support_tolerance = float(self.scene_cfg.get("support_contact_tolerance_m", 0.015))
            object_xy_inside = bool(
                np.all(object_bbox_min[:2] >= support_bbox_min[:2] - support_tolerance)
                and np.all(object_bbox_max[:2] <= support_bbox_max[:2] + support_tolerance)
            )
            on_support = bool(
                object_xy_inside
                and object_bbox_min[2] <= support_bbox_max[2] + support_tolerance
                and object_bbox_max[2] >= support_bbox_min[2] - support_tolerance
            )
        else:
            on_support = object_position[2] <= self.initial_position[2] + 0.045
        stable = float(np.linalg.norm(object_velocity)) < 0.25

        if phase in {Phase.CLOSE.value, Phase.HOLD.value, Phase.LIFT.value, Phase.LIFT_HOLD.value} and left_contact and right_contact:
            self.flags["grasp"] = True
        if self.flags["grasp"] and phase in {Phase.LIFT.value, Phase.LIFT_HOLD.value, Phase.PLACE.value, Phase.RELEASE.value} and lift_progress > 0.045:
            self.flags["lift"] = True
        if self.flags["lift"] and phase == Phase.LIFT_HOLD.value and stable and lift_progress > 0.035:
            self.flags["hold"] = True
        if self.flags["hold"] and phase in {Phase.PLACE.value, Phase.RELEASE.value} and on_support and stable:
            self.flags["place"] = True
        if self.flags["place"] and phase == Phase.RELEASE.value and (not gripper_close or open_commanded) and stable:
            self.flags["release"] = True
        self.samples.append(
            {
                "phase": phase,
                "left_contact": left_contact,
                "right_contact": right_contact,
                "lift_progress": lift_progress,
                "stable": stable,
                "gripper_close": gripper_close,
                "close_commanded": close_commanded,
                "open_commanded": open_commanded,
                "on_support": on_support,
            }
        )

    def finalize(self, duration_s: float, frame_count: int, invalid_reason: str | None = None) -> EpisodeResult:
        if invalid_reason or self.invalid_reason:
            return EpisodeResult("INVALID", self.flags, None, None, invalid_reason or self.invalid_reason, frame_count, duration_s)
        if all(self.flags.values()):
            return EpisodeResult("SUCCESS", self.flags, None, None, None, frame_count, duration_s)
        reason = self._failure_reason()
        return EpisodeResult("FAILURE", self.flags, reason[0], reason[1], None, frame_count, duration_s)

    def _failure_reason(self) -> tuple[str, str]:
        if not self.flags["grasp"]:
            left_seen = any(item["left_contact"] for item in self.samples)
            right_seen = any(item["right_contact"] for item in self.samples)
            if not left_seen and not right_seen:
                return "both_missed", Phase.CLOSE.value
            if not left_seen:
                return "left_missed", Phase.CLOSE.value
            return "right_missed", Phase.CLOSE.value
        if not self.flags["lift"]:
            return "lift_failed", Phase.LIFT.value
        if not self.flags["hold"]:
            return "unstable_hold", Phase.LIFT_HOLD.value
        if not self.flags["place"]:
            return "place_failed", Phase.PLACE.value
        return "release_failed", Phase.RELEASE.value
