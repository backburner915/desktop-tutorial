from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .nominal_grasp import perturb_grasp_pose, pregrasp_pose
from .robot_interface import IKError, RobotInterface
from .trajectory import JointTrajectory, Waypoint
from .types import NominalGrasp, Phase, Pose


def _failure_adjusted(config: Any, scene_cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply exactly one primary, interpretable perturbation."""

    values = {
        "left_grasp_offset": list(config.left_grasp_offset),
        "right_grasp_offset": list(config.right_grasp_offset),
        "left_approach_angle": list(config.left_approach_angle),
        "right_approach_angle": list(config.right_approach_angle),
        "phase_offset": float(config.phase_offset),
        "lift_vector": list(config.lift_vector),
        "gripper_target": deepcopy(config.gripper_target),
    }
    mode = config.failure_mode
    m = float(config.failure_magnitude)
    if mode == "left_grasp_position_error":
        values["left_grasp_offset"][0] += m
    elif mode == "right_grasp_position_error":
        values["right_grasp_offset"][0] += m
    elif mode == "both_grasp_position_error":
        values["left_grasp_offset"][0] += m
        values["right_grasp_offset"][0] -= m
    elif mode == "left_approach_angle_error":
        values["left_approach_angle"][2] += m
    elif mode == "right_approach_angle_error":
        values["right_approach_angle"][2] += m
    elif mode == "asymmetric_approach_error":
        values["left_approach_angle"][2] += m
        values["right_approach_angle"][2] -= m
    elif mode == "left_gripper_underclose":
        values["gripper_target"]["left"] = [min(0.03, max(m, x)) for x in values["gripper_target"]["left"]]
    elif mode == "right_gripper_underclose":
        values["gripper_target"]["right"] = [min(0.03, max(m, x)) for x in values["gripper_target"]["right"]]
    elif mode in {"left_gripper_early", "left_gripper_late", "right_gripper_early", "right_gripper_late", "phase_mismatch"}:
        sign = 1.0 if mode in {"left_gripper_early", "right_gripper_late"} else -1.0
        if mode == "phase_mismatch":
            sign = 1.0
        values["phase_offset"] = sign * max(abs(values["phase_offset"]), m)
    elif mode == "left_contact_missing":
        values["left_grasp_offset"][0] += max(m, 0.04)
    elif mode == "right_contact_missing":
        values["right_grasp_offset"][0] += max(m, 0.04)
    elif mode == "insufficient_lift":
        lift = np.asarray(values["lift_vector"], dtype=np.float64)
        if np.linalg.norm(lift) > 1e-9:
            lift *= min(0.35, max(0.05, m))
        values["lift_vector"] = lift.tolist()
    elif mode == "slip":
        slip_open = min(0.03, max(m, 0.008))
        values["gripper_target"] = {"left": [slip_open, slip_open], "right": [slip_open, slip_open]}
    return values


class BimanualController:
    """Build one reusable joint-target trajectory for a scenario."""

    def __init__(self, robot: RobotInterface, scene_cfg: dict[str, Any]):
        self.robot = robot
        self.scene_cfg = scene_cfg

    def build(self, config: Any, nominal: NominalGrasp) -> JointTrajectory:
        adjusted = _failure_adjusted(config, self.scene_cfg)
        left_grasp = perturb_grasp_pose(nominal.left, adjusted["left_grasp_offset"], adjusted["left_approach_angle"])
        right_grasp = perturb_grasp_pose(nominal.right, adjusted["right_grasp_offset"], adjusted["right_approach_angle"])
        left_pre = pregrasp_pose(left_grasp, nominal.object_center, config.pregrasp_distance)
        right_pre = pregrasp_pose(right_grasp, nominal.object_center, config.pregrasp_distance)

        q_initial = self.robot.full_positions()
        # Physical runs start from the measured settled articulation state.
        # The JSON home poses remain available for the legacy/configuration
        # runner, but overwriting a live imported USD pose makes the first IK
        # seed jump and can send the arm into an unrelated basin.
        if not bool(self.scene_cfg.get("use_live_initial_state", False)):
            q_initial[self.robot.arm_indices["left"]] = np.asarray(config.left_start_pose, dtype=np.float64)
            q_initial[self.robot.arm_indices["right"]] = np.asarray(config.right_start_pose, dtype=np.float64)
        open_targets = {
            "left": np.asarray(self.scene_cfg["gripper_open"], dtype=np.float64),
            "right": np.asarray(self.scene_cfg["gripper_open"], dtype=np.float64),
        }
        close_targets = {
            "left": np.asarray(adjusted["gripper_target"]["left"], dtype=np.float64),
            "right": np.asarray(adjusted["gripper_target"]["right"], dtype=np.float64),
        }
        q_initial[self.robot.gripper_indices["left"]] = open_targets["left"]
        q_initial[self.robot.gripper_indices["right"]] = open_targets["right"]

        ik_kwargs = {"max_iterations": int(self.scene_cfg.get("ik_max_iterations", 80))}
        q_pre = self._solve_with_scene_fallback(
            "left", left_pre, q_initial, ik_kwargs, "pregrasp"
        )
        q_pre = self._solve_with_scene_fallback(
            "right", right_pre, q_pre, ik_kwargs, "pregrasp"
        )
        q_grasp = self._solve_with_scene_fallback(
            "left", left_grasp, q_pre, ik_kwargs, "grasp"
        )
        q_grasp = self._solve_with_scene_fallback(
            "right", right_grasp, q_grasp, ik_kwargs, "grasp"
        )

        object_position, _ = self.robot.object_pose()
        lift = np.asarray(adjusted["lift_vector"], dtype=np.float64)
        left_lift = Pose(left_grasp.position + lift, left_grasp.quaternion_wxyz.copy())
        right_lift = Pose(right_grasp.position + lift, right_grasp.quaternion_wxyz.copy())
        q_lift = self._solve_with_scene_fallback(
            "left", left_lift, q_grasp, ik_kwargs, "lift"
        )
        q_lift = self._solve_with_scene_fallback(
            "right", right_lift, q_lift, ik_kwargs, "lift"
        )
        # Return the EEFs to the physical grasp height for place.  The
        # corrected-scene runner creates a real kinematic support whose top is
        # below the payload's settled rigid-body centre.  A positive hover
        # offset would therefore leave the object suspended and make the
        # release phase impossible to verify.
        place_hover_height = float(self.scene_cfg.get("place_hover_height_m", 0.0))
        place_clearance = np.asarray([0.0, 0.0, place_hover_height], dtype=np.float64)
        left_place = Pose(left_grasp.position + place_clearance, left_grasp.quaternion_wxyz.copy())
        right_place = Pose(right_grasp.position + place_clearance, right_grasp.quaternion_wxyz.copy())
        q_place = self._solve_with_scene_fallback(
            "left", left_place, q_lift, ik_kwargs, "place"
        )
        q_place = self._solve_with_scene_fallback(
            "right", right_place, q_place, ik_kwargs, "place"
        )

        # Restore the state that the reset manager established.  IK solving is
        # a precheck operation and must not leak its intermediate joint state
        # into the recorded episode.
        self.robot.robot.set_joint_positions(q_initial.reshape(1, -1))
        self.robot.robot.set_joint_velocities(np.zeros((1, self.robot.dof_count), dtype=np.float64))
        self.robot.apply_full_target(q_initial)

        def with_grippers(q: np.ndarray, close: bool) -> np.ndarray:
            result = q.copy()
            for side in ("left", "right"):
                result[self.robot.gripper_indices[side]] = close_targets[side] if close else open_targets[side]
            return result

        steady_place = self.scene_cfg.get("motion_profile") == "steady_place"
        if steady_place:
            # Give the imported arm time to settle into pregrasp/approach and
            # let the payload settle on the real support before opening. The
            # longer RELEASE phase also leaves the arms stationary after the
            # fingers are fully open, reducing release-induced tipping.
            durations = (1.2, 3.0, 2.5, 1.3, 1.3, 2.5, 2.0, 4.0, 4.5)
        else:
            durations = (0.6, 2.0, 1.8, 1.0, 1.0, 2.0, 1.5, 2.5, 3.0)
        waypoints = [
            Waypoint(Phase.INITIAL, durations[0], with_grippers(q_initial, False)),
            Waypoint(Phase.PREGRASP, durations[1], with_grippers(q_pre, False)),
            Waypoint(Phase.APPROACH, durations[2], with_grippers(q_grasp, False)),
            Waypoint(Phase.CLOSE, durations[3], with_grippers(q_grasp, True)),
            Waypoint(Phase.HOLD, durations[4], with_grippers(q_grasp, True)),
            Waypoint(Phase.LIFT, durations[5], with_grippers(q_lift, True)),
            Waypoint(Phase.LIFT_HOLD, durations[6], with_grippers(q_lift, True)),
            Waypoint(Phase.PLACE, durations[7], with_grippers(q_place, True)),
            Waypoint(Phase.RELEASE, durations[8], with_grippers(q_place, False)),
        ]
        close_start = sum(item.duration_s for item in waypoints[:3])
        release_start = sum(item.duration_s for item in waypoints[:8])
        phase_offset = float(adjusted["phase_offset"])
        if config.coordination_mode == "left_leads":
            phase_offset = abs(phase_offset) if abs(phase_offset) > 1e-6 else 0.15
        elif config.coordination_mode == "right_leads":
            phase_offset = -abs(phase_offset) if abs(phase_offset) > 1e-6 else -0.15
        else:
            phase_offset = 0.0
        return JointTrajectory(
            waypoints,
            open_targets,
            close_targets,
            self.robot.gripper_indices,
            close_start,
            release_start,
            phase_offset,
        )

    def _solve_with_scene_fallback(
        self,
        side: str,
        target: Pose,
        seed: np.ndarray,
        ik_kwargs: dict[str, Any],
        label: str,
    ) -> np.ndarray:
        try:
            return self.robot.solve_ik(
                side, target.position, target.quaternion_wxyz, seed=seed, **ik_kwargs
            )
        except IKError as first_error:
            fallback_seeds = self.scene_cfg.get("ik_fallback_seeds", {})
            fallback = fallback_seeds.get(label, self.scene_cfg.get("ik_fallback_seed"))
            if fallback is None:
                raise
            fallback_q = np.asarray(fallback, dtype=np.float64)
            if fallback_q.shape != (self.robot.dof_count,):
                raise ValueError(
                    f"ik_fallback_seed must contain {self.robot.dof_count} full-articulation values"
                ) from first_error
            print(
                f"R1 IK {side}: retrying {label} from calibrated reachable seed after local solve failed",
                flush=True,
            )
            return self.robot.solve_ik(
                side, target.position, target.quaternion_wxyz, seed=fallback_q, **ik_kwargs
            )
