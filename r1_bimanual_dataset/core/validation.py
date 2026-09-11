from __future__ import annotations

from typing import Any

import numpy as np

from ..config import ACTION_DIM, ACTION_JOINT_NAMES, COORDINATION_MODES, FAILURE_MODES


def validate_scene_config(scene_cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if len(scene_cfg.get("action_joint_names", [])) != ACTION_DIM:
        errors.append("scene action_joint_names length is not 16")
    if scene_cfg.get("action_joint_names") != ACTION_JOINT_NAMES:
        errors.append("scene action_joint_names do not match the verified fixed ordering")
    for key in ("robot_prim", "target_object", "left_eef", "right_eef", "cameras"):
        if not scene_cfg.get(key):
            errors.append(f"missing scene config field: {key}")
    if set(scene_cfg.get("cameras", {})) != {"front", "left_wrist", "right_wrist"}:
        errors.append("scene must define front, left_wrist, and right_wrist cameras")
    return errors


def validate_runtime_mapping(robot: Any, scene_cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    actual = list(robot.all_joint_names)
    expected = list(scene_cfg["action_joint_names"])
    if len(actual) < len(expected):
        errors.append(f"live articulation DOF count {len(actual)} is smaller than action dimension")
    missing = [name for name in expected if name not in actual]
    if missing:
        errors.append(f"live articulation missing action joints: {missing}")
    for side in ("left", "right"):
        if f"{side}_arm_link6" not in list(robot.robot.body_names):
            errors.append(f"live articulation missing {side}_arm_link6 body")
    return errors


def validate_scenario_for_execution(scenario: Any, scene_cfg: dict[str, Any], robot: Any) -> list[str]:
    errors: list[str] = []
    try:
        scenario.validate()
    except Exception as exc:
        errors.append(str(exc))
        return errors
    lower, upper = robot.limits()
    for side, values in (("left", scenario.left_start_pose), ("right", scenario.right_start_pose)):
        ids = robot.arm_indices[side]
        vector = np.asarray(values, dtype=np.float64)
        if np.any(vector < lower[ids] - 1e-5) or np.any(vector > upper[ids] + 1e-5):
            errors.append(f"{side} initial pose violates live joint limits")
    for side in ("left", "right"):
        gripper = np.asarray(scenario.gripper_target[side], dtype=np.float64)
        ids = robot.gripper_indices[side]
        if np.any(gripper < lower[ids] - 1e-5) or np.any(gripper > upper[ids] + 1e-5):
            errors.append(f"{side} gripper target violates live joint limits")
    if float(scene_cfg.get("support_z", -np.inf)) > float(scenario.object_position[2]) + 1e-4:
        errors.append("object center is below configured support height")
    if np.linalg.norm(np.asarray(scenario.lift_vector, dtype=np.float64)) < 1e-5:
        errors.append("lift_vector is too small for a lift episode")
    return errors
