from __future__ import annotations

import math
from typing import Any

import numpy as np

from .robot_interface import quat_from_rpy, quat_multiply, quat_normalize, quat_rotate
from .types import NominalGrasp, Pose


def _quat_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to Isaac Sim's scalar-first format.

    The previous implementation used the vector-first formulas in two of
    the diagonal branches.  That produced a plausible-looking quaternion for
    the right-hand grasp but not the requested look-at rotation, which made
    the live IK target inconsistent with the generated grasp point.
    """
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray([(s / 4.0), (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.asarray([(m[2, 1] - m[1, 2]) / s, s / 4.0, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    # Use >= for the tie case.  A side-on grasp commonly has equal Y/Z
    # diagonal entries; falling through to the Z branch there selects the
    # wrong rotation axis and points the right EEF away from the object.
    elif m[1, 1] >= m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.asarray([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4.0, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.asarray([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4.0])
    return quat_normalize(q)


def _look_at_quaternion(forward: np.ndarray, up_hint: np.ndarray = np.asarray([0.0, 0.0, 1.0])) -> np.ndarray:
    """Return a scalar-first quaternion whose local +Z points at the object."""

    z_axis = np.asarray(forward, dtype=np.float64)
    z_axis /= max(np.linalg.norm(z_axis), 1e-9)
    up = np.asarray(up_hint, dtype=np.float64)
    if abs(float(np.dot(up, z_axis))) > 0.96:
        up = np.asarray([1.0, 0.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= max(np.linalg.norm(x_axis), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return _quat_from_rotation_matrix(rotation)


class NominalGraspGenerator:
    """Generate opposing grasp frames from the target's local bounding box."""

    def __init__(self, stage: Any, scene_cfg: dict[str, Any], robot: Any):
        from pxr import Usd, UsdGeom

        self.stage = stage
        self.scene_cfg = scene_cfg
        self.robot = robot
        self._usd = Usd
        self._usd_geom = UsdGeom

    def _world_bbox(self, prim: Any) -> tuple[np.ndarray, np.ndarray, str]:
        cache = self._usd_geom.BBoxCache(
            self._usd.TimeCode.Default(),
            [self._usd_geom.Tokens.default_, self._usd_geom.Tokens.render, self._usd_geom.Tokens.proxy],
            useExtentsHint=True,
        )
        try:
            # On this scene T01 is an Xform/RigidBody whose local bound already
            # contains its authored translation. Adding that local center to
            # get_world_poses() would double the world transform. Compute the
            # bound in world space and use its center directly.
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            return np.asarray(box.GetMin(), dtype=np.float64), np.asarray(box.GetMax(), dtype=np.float64), "bbox_world_coordinate_system"
        except Exception as exc:
            raise ValueError(f"target world bounding box is unavailable: {exc}") from exc

    def generate(self) -> NominalGrasp:
        prim = self.stage.GetPrimAtPath(self.scene_cfg["target_object"])
        if not prim or not prim.IsValid():
            raise ValueError(f"target object prim is missing: {self.scene_cfg['target_object']}")
        world_min, world_max, source = self._world_bbox(prim)
        object_pose_center, object_quaternion = self.robot.object_pose()
        # T01 contains an authored mesh translation inside its rigid-body
        # prim. After runtime relocation, using only ComputeWorldBound for the
        # center can count that translation twice and put the grasp at the
        # object's top instead of its physical middle. The physical runner
        # supplies source bbox dimensions captured before relocation.
        if self.scene_cfg.get("nominal_object_center_from_rigid_pose", False):
            object_center = np.asarray(object_pose_center, dtype=np.float64)
            explicit_center = self.scene_cfg.get("nominal_object_center_override")
            if explicit_center is not None:
                object_center = np.asarray(explicit_center, dtype=np.float64).reshape(3)
            configured_dimensions = self.scene_cfg.get("object_bbox_dimensions_m")
            dimensions = np.asarray(
                configured_dimensions if configured_dimensions is not None else (world_max - world_min),
                dtype=np.float64,
            )
            source = "rigid_body_pose_center + source_bbox_dimensions"
        else:
            object_center = (world_min + world_max) / 2.0
            dimensions = np.maximum(world_max - world_min, 1e-6)
        dimensions = np.maximum(dimensions, 1e-6)

        root_position, root_quaternion = self.robot.root_pose()
        robot_lateral = quat_rotate(root_quaternion, np.asarray([0.0, 1.0, 0.0]))
        candidates = []
        for axis in (0, 1):
            axis_world = quat_rotate(object_quaternion, np.eye(3)[axis])
            candidates.append((abs(float(np.dot(axis_world, robot_lateral))), axis, axis_world))
        _, local_axis, side_axis_world = max(candidates, key=lambda x: x[0])
        if bool(self.scene_cfg.get("nominal_grasp_horizontal", False)):
            # The reference object is upright on a support.  Its authored
            # local frame can contain a small tilt, but using that tilt for
            # the side vector would move the two grasp points above/below the
            # physical object centre and cause a lift/slide.  Keep the
            # nominal opposing points in the support plane; angle variation
            # remains available through ScenarioConfig for non-nominal data.
            side_axis_world = np.asarray(side_axis_world, dtype=np.float64).copy()
            side_axis_world[2] = 0.0
            side_axis_world /= max(np.linalg.norm(side_axis_world), 1e-9)
        left_position, _ = self.robot.eef_link_pose("left")
        right_position, _ = self.robot.eef_link_pose("right")
        if float(np.dot(left_position - right_position, side_axis_world)) < 0.0:
            side_axis_world = -side_axis_world

        clearance = float(self.scene_cfg.get("nominal_grasp", {}).get("clearance_m", 0.008))
        # The grasp separation is determined by the live object's bbox plus a
        # configured side clearance for the real gripper geometry.  Do not use
        # a robot-task-specific fixed separation here: T01 in spacerobot.usd
        # is only about 0.102 m wide along its grasp axis, so +/-0.15 m would
        # place both EEFs outside the object and create a systematic miss.
        minimum_side_reach = float(
            self.scene_cfg.get("nominal_grasp", {}).get("minimum_side_reach_m", 0.0)
        )
        gripper_side_clearance = float(
            self.scene_cfg.get("nominal_grasp", {}).get("gripper_side_clearance_m", 0.008)
        )
        reach = max(
            float(dimensions[local_axis]) / 2.0 + clearance + gripper_side_clearance,
            minimum_side_reach,
        )
        left_grasp_position = object_center + side_axis_world * reach
        right_grasp_position = object_center - side_axis_world * reach
        left_quaternion = _look_at_quaternion(object_center - left_grasp_position)
        right_quaternion = _look_at_quaternion(object_center - right_grasp_position)

        return NominalGrasp(
            left=Pose(left_grasp_position, left_quaternion),
            right=Pose(right_grasp_position, right_quaternion),
            object_center=object_center,
            object_dimensions=dimensions,
            object_local_axis=local_axis,
            side_axis_world=side_axis_world,
            clearance_m=clearance,
            source=source,
        )


def perturb_grasp_pose(pose: Pose, offset_local: list[float], angle_rpy: list[float]) -> Pose:
    offset_local = np.asarray(offset_local, dtype=np.float64)
    angle_quaternion = quat_from_rpy(*np.asarray(angle_rpy, dtype=np.float64))
    return Pose(
        position=pose.position + quat_rotate(pose.quaternion_wxyz, offset_local),
        quaternion_wxyz=quat_normalize(quat_multiply(pose.quaternion_wxyz, angle_quaternion)),
    )


def pregrasp_pose(pose: Pose, object_center: np.ndarray, distance_m: float) -> Pose:
    direction_to_object = np.asarray(object_center, dtype=np.float64) - pose.position
    norm = max(np.linalg.norm(direction_to_object), 1e-9)
    direction_to_object /= norm
    return Pose(pose.position - direction_to_object * float(distance_m), pose.quaternion_wxyz.copy())
