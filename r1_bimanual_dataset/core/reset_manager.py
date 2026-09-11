from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .robot_interface import quat_from_rpy, RobotInterface


class ResetManager:
    """Explicit reset protocol; no user Stop/Play interaction is required."""

    def __init__(
        self,
        robot: RobotInterface,
        scene_cfg: dict[str, Any],
        step_fn: Callable[[], None],
        play_fn: Callable[[], None] | None = None,
    ):
        self.robot = robot
        self.scene_cfg = scene_cfg
        self.step_fn = step_fn
        self.play_fn = play_fn

    def reset(self, scenario: Any, settle_steps: int = 30) -> None:
        open_target = list(self.scene_cfg["gripper_open"])
        object_quaternion = quat_from_rpy(*scenario.object_orientation)
        # Place the complete runtime state before the first physics tick.
        # Previously reset stepped the simulation for 10 frames before this
        # call, which allowed the imported floating-base robot to fall onto
        # the table in GUI mode.
        self.robot.reset_state(
            {"left": scenario.left_start_pose, "right": scenario.right_start_pose},
            {"left": open_target, "right": open_target},
            np.asarray(scenario.object_position, dtype=np.float64),
            object_quaternion,
        )
        # Apply the runtime locks while the timeline is still paused.  This
        # closes the one-frame window in which a floating-base imported USD
        # could receive gravity before the first controlled step.
        self.robot.hold_runtime_base()
        self.robot.hold_runtime_joints()
        if self.play_fn is not None:
            self.play_fn()
        for _ in range(settle_steps):
            self.step_fn()
        # Settling the payload must not leave the unanchored vehicle or its
        # arm drives in a displaced pose.  Reapply the requested initial
        # configuration after the settle window; this is runtime state only
        # and does not modify the source USD.
        self.robot.reset_state(
            {"left": scenario.left_start_pose, "right": scenario.right_start_pose},
            {"left": open_target, "right": open_target},
            np.asarray(scenario.object_position, dtype=np.float64),
            object_quaternion,
        )
        self.robot.hold_runtime_base()
        self.step_fn()
        position, quaternion = self.robot.object_pose()
        velocity = self.robot.object_velocity()
        root_position, _ = self.robot.root_pose()
        left_position, _ = self.robot.eef_link_pose("left")
        right_position, _ = self.robot.eef_link_pose("right")
        print(
            "R1 runner: reset state "
            f"root=({root_position[0]:.3f},{root_position[1]:.3f},{root_position[2]:.3f}) "
            f"object=({position[0]:.3f},{position[1]:.3f},{position[2]:.3f}) "
            f"distance={np.linalg.norm(position - root_position):.3f}m "
            f"eef_left=({left_position[0]:.3f},{left_position[1]:.3f},{left_position[2]:.3f}) "
            f"eef_right=({right_position[0]:.3f},{right_position[1]:.3f},{right_position[2]:.3f})",
            flush=True,
        )
        if not np.isfinite(np.concatenate([position, quaternion, velocity])).all():
            raise RuntimeError("reset produced NaN or infinity")
        if np.linalg.norm(velocity) > 0.35:
            raise RuntimeError(f"reset did not settle object velocity: {np.linalg.norm(velocity):.3f}")

    def emergency_stop(self) -> None:
        try:
            self.robot.stop_commands()
            self.step_fn()
        except Exception:
            pass
