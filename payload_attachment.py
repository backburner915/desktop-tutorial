from __future__ import annotations

from typing import Any

import numpy as np

from .types import NominalGrasp, Phase


class RuntimeGraspAttachment:
    """Session-only payload constraint for imported scenes without repeatable contact.

    Some imported USD scenes expose the arm articulation and gripper joints but
    do not produce a stable dynamic grasp at the configured camera/physics
    rate.  For the normal expert template only, this adapter attaches the
    payload after both EEFs are at their generated grasp frames and the
    commanded grippers are closed.  The payload then follows the bimanual
    EEF midpoint through lift and a safe place-hover segment.  At release it
    is explicitly put back on the settled support pose captured at grasp
    time, so the low visual gripper pose cannot drive it through the table.
    Failure scenarios never call this adapter.

    This is deliberately kept outside the robot controller and is recorded in
    episode metadata so these episodes remain distinguishable from native
    PhysX-contact rollouts.
    """

    def __init__(self, robot: Any, scene_cfg: dict[str, Any], nominal: NominalGrasp, enabled: bool):
        self.robot = robot
        self.scene_cfg = scene_cfg
        self.nominal = nominal
        self.enabled = bool(enabled)
        self.active = False
        self.anchor_offset: np.ndarray | None = None
        self.object_quaternion: np.ndarray | None = None
        self.support_position: np.ndarray | None = None
        self.support_quaternion: np.ndarray | None = None
        self.attach_phase: str | None = None

    @property
    def used(self) -> bool:
        return bool(self.active or self.anchor_offset is not None)

    def maybe_attach(
        self,
        phase: Phase,
        left_tip: np.ndarray,
        right_tip: np.ndarray,
        commanded_action16: np.ndarray,
    ) -> bool:
        if not self.enabled or self.active or phase not in {Phase.CLOSE, Phase.HOLD}:
            return False
        close_threshold = float(self.scene_cfg.get("attachment_gripper_close_threshold_m", 0.02))
        if not np.all(np.asarray(commanded_action16, dtype=np.float64)[[6, 7, 14, 15]] <= close_threshold):
            return False
        left_error = float(np.linalg.norm(np.asarray(left_tip)[:3] - self.nominal.left.position))
        right_error = float(np.linalg.norm(np.asarray(right_tip)[:3] - self.nominal.right.position))
        max_distance = float(self.scene_cfg.get("scripted_grasp_attachment_distance_m", 0.11))
        if left_error > max_distance or right_error > max_distance:
            return False
        object_position, object_quaternion = self.robot.object_pose()
        midpoint = (np.asarray(left_tip, dtype=np.float64)[:3] + np.asarray(right_tip, dtype=np.float64)[:3]) / 2.0
        self.anchor_offset = np.asarray(object_position, dtype=np.float64) - midpoint
        self.object_quaternion = np.asarray(object_quaternion, dtype=np.float64).copy()
        # Capture the settled support pose before the scripted attachment.
        self.support_position = np.asarray(object_position, dtype=np.float64).copy()
        self.support_quaternion = np.asarray(object_quaternion, dtype=np.float64).copy()
        self.attach_phase = phase.value
        self.active = True
        print(
            f"R1 runner: runtime grasp attachment enabled at phase={phase.value} "
            f"left_error={left_error:.3f}m right_error={right_error:.3f}m",
            flush=True,
        )
        self.update(left_tip, right_tip)
        return True

    def update(self, left_tip: np.ndarray, right_tip: np.ndarray) -> None:
        if not self.active or self.anchor_offset is None or self.object_quaternion is None:
            return
        midpoint = (np.asarray(left_tip, dtype=np.float64)[:3] + np.asarray(right_tip, dtype=np.float64)[:3]) / 2.0
        position = midpoint + self.anchor_offset
        self.robot.object.set_world_poses(position.reshape(1, 3), self.object_quaternion.reshape(1, 4))
        self.robot.object.set_velocities(np.zeros((1, 6), dtype=np.float64))

    def release(self) -> None:
        if self.active:
            if self.support_position is not None and self.support_quaternion is not None:
                # Release from a safe hover and restore the payload to the
                # settled support pose instead of reusing a low EEF target as
                # the place pose.
                self.robot.object.set_world_poses(
                    self.support_position.reshape(1, 3),
                    self.support_quaternion.reshape(1, 4),
                )
                self.robot.object.set_velocities(np.zeros((1, 6), dtype=np.float64))
                print(
                    "R1 runner: payload placed on support before release "
                    f"z={self.support_position[2]:.3f}m",
                    flush=True,
                )
            print("R1 runner: runtime grasp attachment released", flush=True)
        self.active = False
