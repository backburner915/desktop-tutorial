from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


class IKError(RuntimeError):
    pass


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.concatenate([[0.0], np.asarray(v, dtype=np.float64)])
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[1:]


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.asarray([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


class RobotInterface:
    """Thin adapter around the Isaac Sim articulation already in the USD.

    The adapter never creates or changes robot topology.  It only reads the
    live articulation and sends position targets to its existing drives.
    """

    def __init__(self, scene_cfg: dict[str, Any], update_fn: Callable[[], Any] | None = None):
        from isaacsim.core.prims import Articulation, RigidPrim, XFormPrim

        self.scene_cfg = scene_cfg
        self._update_fn = update_fn
        self.robot_path = scene_cfg["robot_prim"]
        self.robot = Articulation(self.robot_path, name="r1_dataset_robot")
        self.object = RigidPrim(scene_cfg["target_object"], name="r1_dataset_object")
        self.eef = {
            "left": XFormPrim(scene_cfg["left_eef"], name="r1_left_eef"),
            "right": XFormPrim(scene_cfg["right_eef"], name="r1_right_eef"),
        }
        self._initialized = False
        self.action_indices: np.ndarray | None = None
        self.arm_indices: dict[str, np.ndarray] = {}
        self.gripper_indices: dict[str, np.ndarray] = {}
        self._initial_root_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._initial_object_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._runtime_root_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._runtime_joint_pose: np.ndarray | None = None
        self._runtime_joint_state_lock_enabled = bool(scene_cfg.get("runtime_joint_state_lock", False))
        self._printed_view_diagnostics = False

    def set_update_fn(self, update_fn: Callable[[], Any] | None) -> None:
        self._update_fn = update_fn

    def set_runtime_joint_state_lock(self, enabled: bool) -> None:
        """Toggle the session-only joint teleport fallback.

        The fallback is useful while resetting and solving IK on this imported
        USD.  It is intentionally disabled for the physical rollout so the
        existing position drives, contacts, and payload dynamics remain active.
        """

        self._runtime_joint_state_lock_enabled = bool(enabled)
        if not enabled:
            self._runtime_joint_pose = None

    def initialize(self) -> None:
        self.robot.initialize()
        self.object.initialize()
        action_names = list(self.scene_cfg["action_joint_names"])
        actual_names = list(self.robot.dof_names)
        if len(action_names) != 16:
            raise ValueError(f"ACTION_JOINT_NAMES must contain 16 names, got {len(action_names)}")
        missing = [name for name in action_names if name not in actual_names]
        if missing:
            raise ValueError(f"action joints missing from live articulation: {missing}")
        self.action_indices = np.asarray([actual_names.index(name) for name in action_names], dtype=np.int64)
        for side in ("left", "right"):
            arm_names = [f"{side}_arm_joint{i}" for i in range(1, 7)]
            grip_names = [f"{side}_gripper_axis{i}" for i in range(1, 3)]
            self.arm_indices[side] = np.asarray([actual_names.index(name) for name in arm_names], dtype=np.int64)
            self.gripper_indices[side] = np.asarray([actual_names.index(name) for name in grip_names], dtype=np.int64)
        self._configure_runtime_drives(actual_names)
        self._initial_root_pose = self.root_pose()
        self._initial_object_pose = self.object_pose()
        self._initialized = True

    def _configure_runtime_drives(self, actual_names: list[str]) -> None:
        """Strengthen existing USD joint drives in the live session only.

        The supplied spacerobot USD contains the robot topology and physics,
        but its authored drive gains are not the high-PD gains used by the
        Galaxea R1 reference task.  With weak/absent drives, a reset pose is
        pulled away by gravity before the controller can read it.  Update the
        existing DriveAPI attributes in the current session layer; never save
        the stage or add/remove joints.
        """
        try:
            from omni.usd import get_context
            from pxr import UsdPhysics

            stage = get_context().get_stage()
            if stage is None:
                return
            arm_stiffness = float(self.scene_cfg.get("runtime_arm_stiffness", 400.0))
            arm_damping = float(self.scene_cfg.get("runtime_arm_damping", 80.0))
            gripper_stiffness = float(self.scene_cfg.get("runtime_gripper_stiffness", 1000.0))
            gripper_damping = float(self.scene_cfg.get("runtime_gripper_damping", 200.0))
            wheel_stiffness = float(self.scene_cfg.get("runtime_wheel_stiffness", 400.0))
            wheel_damping = float(self.scene_cfg.get("runtime_wheel_damping", 80.0))
            wanted = set(actual_names)
            matched: list[str] = []
            for prim in stage.Traverse():
                name = prim.GetName()
                if name not in wanted:
                    continue
                if "gripper" in name:
                    stiffness, damping = gripper_stiffness, gripper_damping
                elif "wheel" in name:
                    stiffness, damping = wheel_stiffness, wheel_damping
                else:
                    stiffness, damping = arm_stiffness, arm_damping
                updated = False
                for axis in ("angular", "linear"):
                    try:
                        drive = UsdPhysics.DriveAPI.Get(prim, axis)
                        stiffness_attr = drive.GetStiffnessAttr()
                        damping_attr = drive.GetDampingAttr()
                        if stiffness_attr.IsValid() and damping_attr.IsValid():
                            stiffness_attr.Set(stiffness)
                            damping_attr.Set(damping)
                            max_force_attr = drive.GetMaxForceAttr()
                            if max_force_attr.IsValid():
                                max_force_attr.Set(1.0e6)
                            updated = True
                    except Exception:
                        continue
                if updated:
                    matched.append(name)
            print(
                f"R1 runner: runtime drive gains applied to {len(matched)}/{len(wanted)} joints "
                f"(arm={arm_stiffness}/{arm_damping}, gripper={gripper_stiffness}/{gripper_damping})",
                flush=True,
            )
            missing = sorted(wanted - set(matched))
            if missing:
                print(f"R1 runner: joints without editable DriveAPI: {missing}", flush=True)
        except Exception as exc:
            raise RuntimeError(f"unable to configure runtime joint drives: {exc}") from exc

    def configure_stationary_base_collision_filter(self) -> None:
        """Disable only mobile-base collision response for stationary rollouts.

        The supplied spacerobot USD is an imported floating-base articulation.
        During the first dataset phase the base is deliberately held at a
        runtime pose and is not being trained or driven.  If its base/wheels
        overlap the authored support table, PhysX can apply a large contact
        impulse and topple the imported articulation before the arm trajectory
        starts.  Filter those base-only contacts in the current session layer;
        never save the USD and never disable arm/gripper collisions.
        """

        if not bool(self.scene_cfg.get("runtime_base_lock", False)):
            return
        try:
            from omni.usd import get_context
            from pxr import UsdPhysics

            stage = get_context().get_stage()
            if stage is None:
                return
            robot_prefix = self.robot_path + "/"
            base_tokens = (
                "/base_link",
                "/servo_link1",
                "/servo_link2",
                "/servo_link3",
                "/wheel_link1",
                "/wheel_link2",
                "/wheel_link3",
            )
            changed: list[str] = []
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith(robot_prefix):
                    continue
                relative = path[len(self.robot_path):]
                if not any(relative == token or relative.startswith(token + "/") for token in base_tokens):
                    continue
                collision = UsdPhysics.CollisionAPI.Apply(prim)
                attr = collision.GetCollisionEnabledAttr()
                if attr.IsValid():
                    attr.Set(False)
                    changed.append(path)
            print(
                f"R1 runner: stationary-base collision filter disabled on {len(changed)} base prims "
                "(session only; arm/gripper collisions unchanged)",
                flush=True,
            )
        except Exception as exc:
            raise RuntimeError(f"unable to configure stationary-base collision filter: {exc}") from exc

    @property
    def all_joint_names(self) -> list[str]:
        return list(self.robot.dof_names)

    @property
    def dof_count(self) -> int:
        return int(self.robot.num_dof)

    def _require_init(self) -> None:
        if not self._initialized or self.action_indices is None:
            raise RuntimeError("RobotInterface.initialize() has not completed")

    @staticmethod
    def _row(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        return array[0] if array.ndim > 1 else array

    def full_positions(self) -> np.ndarray:
        self._require_init()
        return self._row(self.robot.get_joint_positions())

    def full_velocities(self) -> np.ndarray:
        self._require_init()
        return self._row(self.robot.get_joint_velocities())

    def state16(self) -> tuple[np.ndarray, np.ndarray]:
        self._require_init()
        positions = self.full_positions()[self.action_indices]
        velocities = self.full_velocities()[self.action_indices]
        return positions.astype(np.float32), velocities.astype(np.float32)

    def root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, quaternion = self.robot.get_world_poses()
        return self._row(position), quat_normalize(self._row(quaternion))

    def object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position, quaternion = self.object.get_world_poses()
        return self._row(position), quat_normalize(self._row(quaternion))

    def object_velocity(self) -> np.ndarray:
        velocity = self.object.get_velocities()
        return self._row(velocity).astype(np.float32)

    def link_pose(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read an articulation link pose from the live PhysX view.

        The imported USD is a floating-base articulation. XFormPrim/Fabric
        transforms can retain the authored root transform for a tick after a
        runtime root teleport, so camera parents must use this same live view
        as the IK and joint state paths.
        """

        body_index = self.robot.get_body_index(link_name)
        view = getattr(self.robot, "_physics_view", None)
        if view is not None and hasattr(view, "get_link_transforms"):
            pose = np.asarray(view.get_link_transforms())[0, body_index]
            position = pose[:3]
            xyzw = pose[3:7]
            quaternion = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
            return position.astype(np.float64), quat_normalize(quaternion)
        raise RuntimeError(f"live PhysX link transform is unavailable for {link_name}")

    def eef_link_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        # Read the EEF from the same PhysX articulation view that supplies the
        # Jacobian. Reading a separate XFormPrim/Fabric view can be one tick
        # behind after a runtime joint-state write, which makes a valid IK
        # update appear to move in the opposite direction.
        try:
            return self.link_pose(f"{side}_arm_link6")
        except RuntimeError:
            # Keep the existing XFormPrim fallback for unusual articulation
            # wrappers; the normal spacerobot path uses link_pose above.
            pass
        position, quaternion = self.eef[side].get_world_poses(usd=False)
        return self._row(position), quat_normalize(self._row(quaternion))

    def eef_tip_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        position, quaternion = self.eef_link_pose(side)
        # The imported R1 scene does not author a universal TCP marker.  Use
        # the measured midpoint of the two physical gripper links whenever a
        # frame is not explicitly supplied; this keeps contact planning tied
        # to live collision geometry instead of a guessed link6-local offset.
        tip_mode = str(self.scene_cfg.get("eef_tip_mode", "configured_offset"))
        offsets = self.scene_cfg.get("eef_tip_offsets") or {}
        if tip_mode == "gripper_midpoint" or side not in offsets:
            midpoint, midpoint_q = self.gripper_midpoint_pose(side)
            return midpoint, midpoint_q
        offset = np.asarray(offsets[side], dtype=np.float64).reshape(3)
        if not np.isfinite(offset).all():
            raise ValueError(f"eef_tip_offsets[{side}] contains NaN or infinity")
        return position + quat_rotate(quaternion, offset), quaternion

    def measured_eef_tip_offset(self, side: str) -> np.ndarray:
        """Return the live link6-local vector to the finger midpoint."""

        link_position, link_quaternion = self.eef_link_pose(side)
        midpoint, _ = self.gripper_midpoint_pose(side)
        return quat_rotate(quat_conjugate(link_quaternion), midpoint - link_position)

    def gripper_midpoint_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the measured centre between the two physical finger links.

        ``*_arm_link6 + [0, 0, 0.15]`` is the Galaxea task-space TCP used by
        the upstream IK examples.  It is *not* the R1's actual two-finger
        contact centre in this imported scene.  The midpoint is measured from
        the live PhysX link transforms on every call, so it remains valid as
        the prismatic axes open and close.  It is used for contact evidence
        and geometric validation, never as a payload attachment point.
        """

        if side not in {"left", "right"}:
            raise ValueError(f"unknown R1 side: {side}")
        link1, _ = self.link_pose(f"{side}_gripper_link1")
        link2, _ = self.link_pose(f"{side}_gripper_link2")
        _, orientation = self.eef_link_pose(side)
        return ((link1 + link2) / 2.0).astype(np.float64), orientation

    def apply_full_target(self, target: np.ndarray) -> None:
        self._require_init()
        target = np.asarray(target, dtype=np.float64).reshape(self.dof_count)
        if not np.isfinite(target).all():
            raise ValueError("controller target contains NaN or infinity")
        # PhysX reduced-coordinate revolute drives reject targets outside
        # [-2*pi, 2*pi].  Imported wheel states can be unwrapped after a few
        # ticks even though the 16D arm/gripper action remains valid; clamp
        # the complete command vector before sending it to the articulation.
        target = np.clip(target, -2.0 * math.pi + 1e-6, 2.0 * math.pi - 1e-6)
        if self._runtime_joint_state_lock_enabled:
            self._runtime_joint_pose = target.copy()
            # Put the commanded pose into the live articulation before the
            # next physics tick.  This keeps Fabric/XFormPrim body poses and
            # the joint tensor on the same tick when this imported USD needs
            # the session-only kinematic fallback.
            self.robot.set_joint_positions(target.reshape(1, -1))
        self.robot.set_joint_position_targets(target.reshape(1, -1))

    def action_from_full(self, full_target: np.ndarray) -> np.ndarray:
        self._require_init()
        return np.asarray(full_target, dtype=np.float64)[self.action_indices].astype(np.float32)

    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        limits = np.asarray(self.robot.get_dof_limits(), dtype=np.float64)
        if limits.ndim == 3:
            limits = limits[0]
        return limits[:, 0], limits[:, 1]

    def reset_state(
        self,
        arm_positions: dict[str, list[float]],
        gripper_positions: dict[str, list[float]],
        object_position: np.ndarray,
        object_quaternion: np.ndarray,
    ) -> None:
        self._require_init()
        full = self.full_positions()
        for side in ("left", "right"):
            full[self.arm_indices[side]] = np.asarray(arm_positions[side], dtype=np.float64)
            full[self.gripper_indices[side]] = np.asarray(gripper_positions[side], dtype=np.float64)
        lower, upper = self.limits()
        if np.any(full < lower - 1e-5) or np.any(full > upper + 1e-5):
            raise ValueError("scenario initial joint state is outside live articulation limits")
        if self._initial_root_pose is not None:
            root_position = self._initial_root_pose[0].copy()
            root_quaternion = self._initial_root_pose[1].copy()
            if self.scene_cfg.get("base_movement") == "teleport_to_target":
                # Move only the live articulation root in the runtime/session
                # state. The source USD is never edited.
                offset_xy = np.asarray(
                    self.scene_cfg.get("base_target_offset_xy", [-0.85, 0.0]),
                    dtype=np.float64,
                ).reshape(2)
                root_position[:2] = np.asarray(object_position, dtype=np.float64)[:2] + offset_xy
                root_position[2] += float(self.scene_cfg.get("base_target_offset_z", 0.0))
            self.robot.set_world_poses(
                root_position.reshape(1, 3),
                root_quaternion.reshape(1, 4),
            )
            if self.scene_cfg.get("runtime_base_lock", False):
                self._runtime_root_pose = (root_position.copy(), root_quaternion.copy())
        self.robot.set_joint_positions(full.reshape(1, -1))
        self.robot.set_joint_velocities(np.zeros((1, self.dof_count), dtype=np.float64))
        if self._runtime_joint_state_lock_enabled:
            self._runtime_joint_pose = full.copy()
        self.robot.set_joint_position_targets(full.reshape(1, -1))
        self.object.set_world_poses(
            np.asarray(object_position, dtype=np.float64).reshape(1, 3),
            np.asarray(object_quaternion, dtype=np.float64).reshape(1, 4),
        )
        self.object.set_velocities(np.zeros((1, 6), dtype=np.float64))

    def hold_runtime_base(self) -> None:
        """Hold a teleported mobile base at its runtime pose without editing USD.

        This is deliberately a runtime/session-layer safeguard for the first
        stationary-base dataset phase.  It avoids wheel/gravity drift while
        the arms execute.  Dynamic wheel-base navigation remains a separate
        mode and is not silently mixed into these grasp episodes.
        """

        if not self.scene_cfg.get("runtime_base_lock", False) or self._runtime_root_pose is None:
            return
        position, quaternion = self._runtime_root_pose
        self.robot.set_world_poses(position.reshape(1, 3), quaternion.reshape(1, 4))

    def hold_runtime_joints(self) -> None:
        """Keep the live joint state at the latest commanded pose if enabled.

        This is a session-only fallback for USD scenes whose imported drives
        do not retain position targets.  It is intentionally opt-in and is
        useful for validating the trajectory/data pipeline before replacing
        it with a scene-specific actuator configuration.
        """
        if not self._runtime_joint_state_lock_enabled or self._runtime_joint_pose is None:
            return
        self.robot.set_joint_positions(self._runtime_joint_pose.reshape(1, -1))
        if self.scene_cfg.get("runtime_joint_state_lock_zero_velocity", True):
            self.robot.set_joint_velocities(np.zeros((1, self.dof_count), dtype=np.float64))

    def stop_commands(self) -> None:
        self.apply_full_target(self.full_positions())

    def jacobian_for_eef(self, side: str) -> np.ndarray:
        self._require_init()
        body_name = f"{side}_arm_link6"
        body_index = self.robot.get_body_index(body_name)
        if not self._printed_view_diagnostics:
            view = getattr(self.robot, "_physics_view", None)
            attrs = [] if view is None else [
                name for name in dir(view)
                if any(token in name.lower() for token in ("link", "body", "transform", "pose", "jacob"))
            ]
            print(
                f"R1 articulation view: type={type(view).__name__} body_index={body_index} "
                f"attrs={attrs[:80]}",
                flush=True,
            )
            self._printed_view_diagnostics = True
        jacobian = np.asarray(self.robot.get_jacobians())[0]
        # Match the Isaac Lab Galaxea reference implementation: a free-base
        # articulation keeps the body id as-is, while a fixed-base tensor
        # Jacobian omits the root body row.
        if bool(getattr(self.robot, "is_fixed_base", False)):
            body_index -= 1
        # Isaac Sim 5.1's low-level view for a floating-base articulation can
        # expose six root DOF columns before the articulation joint columns.
        # The high-level joint state still exposes only the 19 articulation
        # DOFs, so translate those indices before slicing the Jacobian.
        joint_offset = 0
        if not bool(getattr(self.robot, "is_fixed_base", False)):
            if jacobian.shape[-1] == self.dof_count + 6:
                joint_offset = 6
        joint_ids = self.arm_indices[side] + joint_offset
        return jacobian[body_index][:, joint_ids]

    @staticmethod
    def _skew(vector: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(vector, dtype=np.float64)
        return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)

    def _finite_difference_position_jacobian(
        self, side: str, q_reference: np.ndarray, epsilon: float = 0.002
    ) -> np.ndarray:
        """Calibrate the live EEF position Jacobian from the current USD scene.

        This is a diagnostic/compatibility path for imported articulations
        whose PhysX Jacobian buffer and body transform buffer use different
        conventions.  It is deliberately opt-in because it costs one tick per
        arm joint; the normal path remains the native PhysX Jacobian.
        """
        if self._update_fn is None:
            raise RuntimeError("finite-difference IK requires a simulation update callback")
        q_reference = np.asarray(q_reference, dtype=np.float64).copy()
        arm_ids = self.arm_indices[side]
        self.robot.set_joint_positions(q_reference.reshape(1, -1))
        self.apply_full_target(q_reference)
        self._update_fn()
        base_position = self.eef_tip_pose(side)[0]
        jacobian = np.zeros((3, len(arm_ids)), dtype=np.float64)
        for column, joint_id in enumerate(arm_ids):
            q_plus = q_reference.copy()
            q_plus[joint_id] += epsilon
            self.robot.set_joint_positions(q_plus.reshape(1, -1))
            self.apply_full_target(q_plus)
            self._update_fn()
            jacobian[:, column] = (self.eef_tip_pose(side)[0] - base_position) / epsilon
        self.robot.set_joint_positions(q_reference.reshape(1, -1))
        self.apply_full_target(q_reference)
        self._update_fn()
        return jacobian

    def solve_ik(
        self,
        side: str,
        target_tip_position: np.ndarray,
        target_tip_quaternion: np.ndarray,
        seed: np.ndarray | None = None,
        max_iterations: int = 80,
        position_tolerance: float = 0.025,
        orientation_tolerance: float = 0.60,
    ) -> np.ndarray:
        """Solve IK with the live PhysX Jacobian and the current articulation.

        This uses the same Jacobian-based control family as the checked-in
        Galaxea examples, but keeps the output as raw joint-position targets.
        """

        self._require_init()
        q = self.full_positions() if seed is None else np.asarray(seed, dtype=np.float64).copy()
        target_q = quat_normalize(target_tip_quaternion)
        orientation_weight = float(self.scene_cfg.get("ik_orientation_weight", 0.0))
        target_tip_position = np.asarray(target_tip_position, dtype=np.float64)
        arm_ids = self.arm_indices[side]
        lower, upper = self.limits()

        # ``seed`` is a real restart state, not only an initial local
        # variable.  Imported floating-base articulations can expose a poor
        # Jacobian basin from the authored Home pose; applying the seed to the
        # live articulation before the first iteration lets a scene-specific
        # calibrated reachable pose recover without changing the controller
        # or the action definition.
        if seed is not None:
            q = np.clip(q, lower, upper)
            self.robot.set_joint_positions(q.reshape(1, -1))
            self.apply_full_target(q)
            if self._update_fn is not None:
                self._update_fn()

        for iteration in range(max_iterations):
            # Always solve from the measured live joint state.  The position
            # drive is not an instantaneous teleport while PhysX is running,
            # so carrying a stale q estimate causes the solver to diverge.
            q = self.full_positions()
            # Keep auxiliary wheel DOFs at a bounded neutral target.  They are
            # deliberately outside the fixed 16D command ordering and their
            # authored revolute drives reject unwrapped angles.
            if self.action_indices is not None and len(self.action_indices) < len(q):
                auxiliary = np.ones(len(q), dtype=bool)
                auxiliary[self.action_indices] = False
                q[auxiliary] = 0.0
            current_link_position, current_q = self.eef_link_pose(side)
            if str(self.scene_cfg.get("eef_tip_mode", "configured_offset")) == "gripper_midpoint" or side not in (self.scene_cfg.get("eef_tip_offsets") or {}):
                current_tip_position = self.gripper_midpoint_pose(side)[0]
                tip_offset = quat_rotate(quat_conjugate(current_q), current_tip_position - current_link_position)
            else:
                tip_offset = np.asarray(self.scene_cfg["eef_tip_offsets"][side], dtype=np.float64).reshape(3)
                current_tip_position = current_link_position + quat_rotate(current_q, tip_offset)
            # Use the same root-frame convention as the checked-in Galaxea
            # DifferentialIKController examples: the PhysX Jacobian is used
            # with EEF positions expressed relative to the articulation root.
            root_position, root_q = self.root_pose()
            root_q_inv = quat_conjugate(root_q)
            current_tip_root = quat_rotate(root_q_inv, current_tip_position - root_position)
            target_tip_root = quat_rotate(root_q_inv, target_tip_position - root_position)
            position_error = target_tip_root - current_tip_root
            orientation_error_q = quat_multiply(target_q, quat_conjugate(current_q))
            if orientation_error_q[0] < 0:
                orientation_error_q = -orientation_error_q
            orientation_error = 2.0 * orientation_error_q[1:]
            link_jacobian = self.jacobian_for_eef(side)
            # Convert the link6 Jacobian into the same tip frame as the target
            # using the exact offset used by the Galaxea lift task.
            offset_world = quat_rotate(current_q, tip_offset)
            offset_root = quat_rotate(root_q_inv, offset_world)
            tip_jacobian = link_jacobian.copy()
            tip_jacobian[:3] = link_jacobian[:3] - self._skew(offset_root) @ link_jacobian[3:]
            if self.scene_cfg.get("ik_jacobian_source") == "finite_difference":
                # Finite differences are measured in world coordinates, but
                # the IK error above is expressed in the articulation-root
                # frame. Rotate every measured Jacobian column into that same
                # root frame before solving.
                finite_difference_world = self._finite_difference_position_jacobian(side, q)
                tip_jacobian[:3] = np.column_stack(
                    [quat_rotate(root_q_inv, finite_difference_world[:, column]) for column in range(finite_difference_world.shape[1])]
                )
            if iteration == 0:
                print(
                    f"R1 IK {side}: tip_target_root={np.round(target_tip_root, 4).tolist()} "
                    f"tip_now_root={np.round(current_tip_root, 4).tolist()} "
                    f"root_q={np.round(root_q, 4).tolist()} "
                    f"jacobian_shape={link_jacobian.shape} "
                    f"orientation_weight={orientation_weight}",
                    flush=True,
                )
            orientation_ok = orientation_weight <= 0.0 or np.linalg.norm(orientation_error) <= orientation_tolerance
            if np.linalg.norm(position_error) <= position_tolerance and orientation_ok:
                return np.clip(q, lower, upper)
            if orientation_weight <= 0.0:
                damping = float(self.scene_cfg.get("ik_damping", 0.05))
                jj = tip_jacobian[:3] @ tip_jacobian[:3].T + (damping**2) * np.eye(3)
                dq = tip_jacobian[:3].T @ np.linalg.solve(jj, position_error)
            else:
                damping = float(self.scene_cfg.get("ik_damping", 0.05))
                jacobian = np.vstack([tip_jacobian[:3], orientation_weight * tip_jacobian[3:]])
                twist = np.concatenate([position_error, orientation_weight * orientation_error])
                jj = jacobian @ jacobian.T + (damping**2) * np.eye(6)
                dq = jacobian.T @ np.linalg.solve(jj, twist)
            step_limit = float(self.scene_cfg.get("ik_step_limit", 0.08))
            dq = np.clip(dq, -step_limit, step_limit)
            if not np.all(np.isfinite(dq)):
                raise IKError(f"{side} IK produced a non-finite joint step")
            q_before_step = q.copy()
            if bool(self.scene_cfg.get("ik_line_search", False)):
                # The imported 5.1 articulation can become numerically
                # unstable when a finite-difference step crosses a bad
                # configuration. Evaluate the actual FK after every trial
                # and backtrack before the next PhysX tick.
                current_error_norm = float(np.linalg.norm(position_error))
                accepted = False
                for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                    candidate = q_before_step.copy()
                    candidate[arm_ids] = np.clip(
                        candidate[arm_ids] + scale * dq,
                        lower[arm_ids],
                        upper[arm_ids],
                    )
                    if not np.all(np.isfinite(candidate)):
                        continue
                    self.robot.set_joint_positions(candidate.reshape(1, -1))
                    self.apply_full_target(candidate)
                    if self._update_fn is not None:
                        self._update_fn()
                    trial_tip = self.eef_tip_pose(side)[0]
                    trial_tip_root = quat_rotate(root_q_inv, trial_tip - root_position)
                    trial_error = target_tip_root - trial_tip_root
                    trial_error_norm = float(np.linalg.norm(trial_error))
                    if np.all(np.isfinite(trial_tip_root)) and trial_error_norm < current_error_norm:
                        q = candidate
                        accepted = True
                        break
                if not accepted:
                    q = q_before_step
                    self.robot.set_joint_positions(q.reshape(1, -1))
                    self.apply_full_target(q)
                    if self._update_fn is not None:
                        self._update_fn()
            else:
                q[arm_ids] = np.clip(q[arm_ids] + dq, lower[arm_ids], upper[arm_ids])
                self.robot.set_joint_positions(q.reshape(1, -1))
                # Keep PhysX's active position drives at the trial
                # configuration while the next Jacobian is evaluated.
                self.apply_full_target(q)
                if self._update_fn is not None:
                    self._update_fn()
            if iteration == 0:
                print(
                    f"R1 IK {side}: dq0={np.round(dq, 5).tolist()} "
                    f"q0={np.round(self.full_positions()[arm_ids], 5).tolist()} "
                    f"q_trial={np.round(q[arm_ids], 5).tolist()} "
                    f"J_first3={np.round(tip_jacobian[:3], 5).tolist()} "
                    f"J_last3={np.round(tip_jacobian[3:], 5).tolist()}",
                    flush=True,
                )
            if iteration == 0:
                after_q = self.full_positions()[arm_ids]
                after_tip = self.eef_tip_pose(side)[0]
                print(
                    f"R1 IK {side}: after_tick_q={np.round(after_q, 5).tolist()} "
                    f"tip_before={np.round(current_tip_position, 5).tolist()} "
                    f"after_tick_tip={np.round(after_tip, 5).tolist()} "
                    f"root={np.round(root_position, 5).tolist()}",
                    flush=True,
                )

            if iteration in {19, 39, max_iterations - 1}:
                print(
                    f"R1 IK {side}: iteration={iteration + 1}/{max_iterations} "
                    f"position_error={np.linalg.norm(position_error):.4f}m",
                    flush=True,
                )

        accept_tolerance = float(
            self.scene_cfg.get("ik_accept_position_error", max(0.05, 2.0 * position_tolerance))
        )
        if np.linalg.norm(position_error) <= accept_tolerance:
            # For this bimanual gripper, position/contact alignment is the
            # primary grasp condition. Keep the best reachable joint target
            # when the requested look-at orientation is not fully reachable.
            print(
                f"R1 IK {side}: accepted reachable target with position_error="
                f"{np.linalg.norm(position_error):.4f}m tolerance={accept_tolerance:.4f}m",
                flush=True,
            )
            return np.clip(q, lower, upper)
        raise IKError(
            f"{side} IK did not converge: position_error={np.linalg.norm(position_error):.4f} m, "
            f"orientation_error={np.linalg.norm(orientation_error):.4f} rad"
        )
