"""SimAdapter: the only layer allowed to know about a physics engine.

episode_runner.py, detectors.py and recovery.py never import Isaac Sim —
they only see Observation/AnomalyEvent. That is what makes the FSM
testable in a plain Python sandbox (MockSimAdapter, used by
tests/test_r11_recovery_loop.py) and portable to the real robot
(IsaacLabR1Adapter, a grounded skeleton — see class docstring for what
is verified vs. still TODO).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from episode_gen.fsm import Phase
from episode_gen.scenario import JOINT_ORDER, ScenarioConfig
from episode_gen.types import Observation
from r1_bimanual_dataset.core.contact_tracker import ContactSample

CAMERA_NAMES = ("front", "left_wrist", "right_wrist")

# Paths verified in the active spacerobot.usd scene.  These are runtime
# references only; the adapter never edits or flattens the source USD.
# T03 is the target selected for the current data-generation task.
R1_STAGE_PATH = r"D:\Galaxea_Lab-galaxea-main\spacerobot.usd"
R1_ROBOT_PRIM_PATH = "/World/garobot2_driveable_final/r1_DVT_colored"
R1_TARGET_OBJECT_PRIM_PATH = "/World/TaskSetup/MovablePayloads/T03"
R1_SUPPORT_PRIM_PATH = "/World/TaskSetup/Fixtures/StorageRack/Top"
R1_EEF_PRIM_PATHS: dict[str, str] = {
    "left": R1_ROBOT_PRIM_PATH + "/left_arm_link6",
    "right": R1_ROBOT_PRIM_PATH + "/right_arm_link6",
}

# Verified against userguide-galaxea/galaxea_lab,
# omni/isaac/lab_assets/galaxea_robots.py :: GALAXEA_R1_CFG.init_state.joint_pos
DEFAULT_QPOS: dict[str, float] = {
    "left_arm_joint1": 0.0, "left_arm_joint2": 2.0, "left_arm_joint3": -1.57,
    "left_arm_joint4": 0.0, "left_arm_joint5": 1.57, "left_arm_joint6": 0.0,
    "left_gripper_axis1": 0.03, "left_gripper_axis2": 0.03,
    "right_arm_joint1": 0.0, "right_arm_joint2": 2.0, "right_arm_joint3": -1.57,
    "right_arm_joint4": 0.0, "right_arm_joint5": -1.57, "right_arm_joint6": 0.0,
    "right_gripper_axis1": 0.03, "right_gripper_axis2": 0.03,
}
DEFAULT_QPOS_VEC: list[float] = [DEFAULT_QPOS[j] for j in JOINT_ORDER]

# Public, immutable-by-convention dataset mapping.  It is intentionally
# copied from the verified R1 names rather than inferred from articulation
# index order (the live imported articulation also contains base/wheel DOFs).
ACTION_JOINT_NAMES: list[str] = list(JOINT_ORDER)
if len(ACTION_JOINT_NAMES) != 16:
    raise RuntimeError(f"R1 ACTION_JOINT_NAMES must contain 16 names, got {len(ACTION_JOINT_NAMES)}")

# Verified against galaxea_lab,
# .../galaxea/manager_based/lift/config/joint_pos_env_cfg.py :: R1LiftEnvCfg
CAMERA_PRIM_PATHS: dict[str, str] = {
    "front": R1_ROBOT_PRIM_PATH + "/torso_link4/front_camera",
    "left_wrist": R1_ROBOT_PRIM_PATH + "/left_arm_link6/left_wrist_camera",
    "right_wrist": R1_ROBOT_PRIM_PATH + "/right_arm_link6/right_wrist_camera",
}

# Contact tokens are a task contract, not basename guesses. The taxonomy's
# historical ``table`` token means this exact T03 support prim. Keep the
# mapping at full prim-path granularity so an unrelated ``Top`` cannot become
# a table contact by coincidence.
CONTACT_TOKEN_BY_PRIM_ROOT: dict[str, str] = {
    R1_TARGET_OBJECT_PRIM_PATH: "object",
    R1_SUPPORT_PRIM_PATH: "table",
}
R1_CONTACT_TOKEN_BY_WATCH_ROOT: dict[str, str] = {
    **CONTACT_TOKEN_BY_PRIM_ROOT,
    R1_ROBOT_PRIM_PATH + "/left_arm_link1": "left_arm_link1",
    R1_ROBOT_PRIM_PATH + "/left_arm_link2": "left_arm_link2",
    R1_ROBOT_PRIM_PATH + "/left_arm_link3": "left_arm_link3",
    R1_ROBOT_PRIM_PATH + "/left_arm_link4": "left_arm_link4",
    R1_ROBOT_PRIM_PATH + "/left_arm_link5": "left_arm_link5",
    R1_ROBOT_PRIM_PATH + "/left_arm_link6": "left_arm_link6",
    R1_ROBOT_PRIM_PATH + "/right_arm_link1": "right_arm_link1",
    R1_ROBOT_PRIM_PATH + "/right_arm_link2": "right_arm_link2",
    R1_ROBOT_PRIM_PATH + "/right_arm_link3": "right_arm_link3",
    R1_ROBOT_PRIM_PATH + "/right_arm_link4": "right_arm_link4",
    R1_ROBOT_PRIM_PATH + "/right_arm_link5": "right_arm_link5",
    R1_ROBOT_PRIM_PATH + "/right_arm_link6": "right_arm_link6",
    R1_ROBOT_PRIM_PATH + "/left_gripper_link1": "left_gripper_link1",
    R1_ROBOT_PRIM_PATH + "/left_gripper_link2": "left_gripper_link2",
    R1_ROBOT_PRIM_PATH + "/right_gripper_link1": "right_gripper_link1",
    R1_ROBOT_PRIM_PATH + "/right_gripper_link2": "right_gripper_link2",
}


class ObservationUnavailableError(RuntimeError):
    """A real-adapter observation cannot be read from the live scene."""


class ContactRootTokenError(ObservationUnavailableError):
    """A tracker watch root lacks an explicitly frozen taxonomy token."""

class SimAdapter(ABC):
    """Everything episode_runner needs from a simulator."""

    control_hz: float = 30.0

    @abstractmethod
    def reset(self, scenario: ScenarioConfig) -> Observation: ...

    @abstractmethod
    def step(self, phase: Phase, scenario: ScenarioConfig) -> tuple[Observation, list[float]]:
        """Realize one control step of `phase` (approach/grasp/lift/...).

        Owns the part that is genuinely robot- and scene-specific: turning
        a bounding-box-derived nominal grasp frame + scenario offsets into
        an actual 16D absolute joint target (IK, collision-aware planning,
        etc.), then advancing the simulator by one control period.
        Returns (resulting Observation, the 16D action actually commanded
        this step) — dataset_writer needs both.
        """

    @abstractmethod
    def render_cameras(self) -> dict[str, Any]:
        """Return {'front': rgb, 'left_wrist': rgb, 'right_wrist': rgb}."""

    @abstractmethod
    def check_place_success(self, obs: Observation, scenario: ScenarioConfig) -> bool:
        """TRA-01 §7 success criteria, evaluated after RELEASE settles."""

    @abstractmethod
    def check_timeout(self, obs: Observation, scenario: ScenarioConfig) -> bool: ...


class IsaacLabR1Adapter(SimAdapter):
    """Isaac Lab bridge for the verified Galaxea R1 ``spacerobot.usd`` scene.

    The preferred input is an Isaac Lab ``ManagerBasedEnv`` or an object that
    exposes its ``scene`` handles.  For the native Isaac Sim scene used here,
    the adapter also binds the active stage directly when those handles are
    absent.  In that mode it resolves the exact R1 articulation, T03 payload,
    EEF links and three camera prims listed above.  No guessed ``Robot`` path,
    new robot topology, payload attachment, or source-USD write is performed.

    The caller still owns AppLauncher and physics initialization.  This is
    intentional: importing this module remains safe in the dependency-free
    MockSimAdapter tests, while a real run must start Isaac Sim before creating
    this adapter.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.control_hz = 30.0
        self._dt = 1.0 / self.control_hz
        self._scene = self._lookup(env, ("scene",)) or env
        self._app = self._first(
            self._lookup(env, ("app", "simulation_app")),
            env if callable(getattr(env, "update", None)) else None,
        )
        self._robot: Any = None
        self._object: Any = None
        self._target_ring: Any = None
        self._target_ring_path: str | None = None
        self._eef_handles: dict[str, Any] = {}
        self._created_native_robot = False
        self._created_native_object = False
        self._stage: Any = None
        self._known_r1_bound = False
        self._r1_backend = self._first(
            self._lookup(env, ("r1_robot_interface", "robot_interface", "physical_robot")),
            self._lookup(self._scene, ("r1_robot_interface", "robot_interface", "physical_robot")),
        )
        self._joint_name_to_index: dict[str, int] = {}
        self._camera_cache: dict[str, Any] = {}
        self._native_handles_initialized = False
        self._refresh_handles()

        self._scenario: ScenarioConfig | None = None
        self._t = 0.0
        self._last_phase: Phase | None = None
        self._phase_progress = 0
        self._commanded_qpos = list(DEFAULT_QPOS_VEC)
        self._last_obs: Observation | None = None
        self._last_object_pos: tuple[float, float, float] | None = None
        self._object_start_z = 0.0
        self._ever_lifted = False
        self._grasp_valid = {"left": False, "right": False}
        self._release_stable_elapsed = 0.0
        self._release_last_pos: tuple[float, float, float] | None = None

    @staticmethod
    def _lookup(container: Any, names: tuple[str, ...]) -> Any:
        if container is None:
            return None
        for name in names:
            if isinstance(container, dict) and name in container:
                return container[name]
            try:
                value = getattr(container, name)
            except AttributeError:
                value = None
            if value is not None:
                return value
            try:
                return container[name]
            except (KeyError, IndexError, TypeError, AttributeError):
                pass
        return None

    @staticmethod
    def _first(*values: Any) -> Any:
        return next((value for value in values if value is not None), None)

    @staticmethod
    def _plain(value: Any) -> Any:
        if value is None:
            return None
        try:
            value = value.detach().cpu()
        except AttributeError:
            pass
        try:
            value = value.numpy()
        except AttributeError:
            pass
        try:
            value = value.tolist()
        except AttributeError:
            pass
        return value

    @classmethod
    def _vector(cls, value: Any, size: int) -> list[float] | None:
        value = cls._plain(value)
        while isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            value = value[0]
        if not isinstance(value, (list, tuple)) or len(value) < size:
            return None
        try:
            return [float(item) for item in value[:size]]
        except (TypeError, ValueError):
            return None

    @classmethod
    def _point(cls, value: Any) -> tuple[float, float, float] | None:
        vector = cls._vector(value, 3)
        return tuple(vector) if vector is not None else None

    @staticmethod
    def _norm(value: Any) -> float:
        import math

        value = IsaacLabR1Adapter._plain(value)
        while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
            value = value[0]
        if isinstance(value, (list, tuple)):
            try:
                return math.sqrt(sum(float(item) ** 2 for item in value))
            except (TypeError, ValueError):
                return 0.0
        try:
            return abs(float(value))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _invoke(fn: Any, variants: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...]) -> tuple[bool, Any]:
        for args, kwargs in variants:
            try:
                return True, fn(*args, **kwargs)
            except TypeError:
                continue
        return False, None

    def _call_first(
        self,
        owners: tuple[Any, ...],
        names: tuple[str, ...],
        variants: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...],
    ) -> tuple[bool, Any]:
        for owner in owners:
            if owner is None:
                continue
            for name in names:
                fn = getattr(owner, name, None)
                if callable(fn):
                    called, result = self._invoke(fn, variants)
                    if called:
                        return True, result
        return False, None

    def _data(self, handle: Any) -> Any:
        return self._lookup(handle, ("data",))

    def _refresh_handles(self) -> None:
        self._robot = self._first(
            self._lookup(self._scene, ("robot",)), self._lookup(self.env, ("robot",))
        )
        self._object = self._first(
            self._lookup(self._scene, ("object", "bag", "crew_lock_bag", "crew_lock_bag_object")),
            self._lookup(self.env, ("object", "bag", "crew_lock_bag", "crew_lock_bag_object")),
        )
        self._target_ring = self._first(
            self._lookup(self._scene, ("target_ring", "green_target_ring", "ring")),
            self._lookup(self.env, ("target_ring", "green_target_ring", "ring")),
        )
        self._bind_known_r1_scene()

    def _active_stage(self) -> Any:
        stage = self._first(
            self._lookup(self.env, ("stage",)),
            self._lookup(self._scene, ("stage",)),
            self._stage,
        )
        if stage is not None:
            return stage
        try:
            import omni.usd

            return omni.usd.get_context().get_stage()
        except (ImportError, AttributeError, RuntimeError):
            return None

    @staticmethod
    def _valid_prim(stage: Any, path: str) -> bool:
        try:
            prim = stage.GetPrimAtPath(path)
            return bool(prim and prim.IsValid())
        except (AttributeError, RuntimeError, TypeError):
            return False

    def _bind_known_r1_scene(self) -> None:
        """Bind the actual USD handles when the caller did not inject them.

        Isaac imports are lazy and every wrapper is optional.  This keeps the
        adapter usable with Isaac Lab scene dictionaries and with test
        doubles, while preventing the old implementation's silent fallback to
        an unrelated/default robot when the real stage is already open.
        """
        stage = self._active_stage()
        if stage is None or not self._valid_prim(stage, R1_ROBOT_PRIM_PATH):
            return
        self._stage = stage
        self._known_r1_bound = True
        if self._robot is None or self._object is None:
            try:
                from isaacsim.core.prims import Articulation, RigidPrim, XFormPrim

                if self._robot is None:
                    self._robot = Articulation(R1_ROBOT_PRIM_PATH, name="r1_dataset_robot")
                    self._created_native_robot = True
                if self._object is None and self._valid_prim(stage, R1_TARGET_OBJECT_PRIM_PATH):
                    self._object = RigidPrim(R1_TARGET_OBJECT_PRIM_PATH, name="r1_dataset_object")
                    self._created_native_object = True
                for arm, path in R1_EEF_PRIM_PATHS.items():
                    if arm not in self._eef_handles and self._valid_prim(stage, path):
                        self._eef_handles[arm] = XFormPrim(path, name=f"r1_{arm}_eef")
            except (ImportError, AttributeError, RuntimeError, TypeError):
                # A supplied Isaac Lab handle may be sufficient; do not make
                # importing the adapter fail just because native wrappers are
                # unavailable in a plain test process.
                pass
        if self._target_ring is None:
            ring_path = self._discover_target_ring_path(stage)
            if ring_path is not None:
                self._target_ring_path = ring_path
                try:
                    from isaacsim.core.prims import XFormPrim

                    self._target_ring = XFormPrim(ring_path, name="r1_target_ring")
                except (ImportError, AttributeError, RuntimeError, TypeError):
                    pass

    @staticmethod
    def _discover_target_ring_path(stage: Any) -> str | None:
        """Find an authored green/target ring without assuming its hierarchy."""
        try:
            candidates: list[str] = []
            for prim in stage.Traverse():
                name = str(prim.GetName()).lower()
                if ("ring" in name and ("green" in name or "target" in name)) or name in {
                    "target_ring", "green_target_ring"
                }:
                    candidates.append(str(prim.GetPath()))
            return sorted(candidates, key=lambda item: (item.count("/"), item))[0] if candidates else None
        except (AttributeError, RuntimeError, TypeError):
            return None

    def _initialize_native_handles(self) -> None:
        if not self._known_r1_bound or self._native_handles_initialized:
            return
        handles = []
        if self._created_native_robot:
            handles.append((self._robot, "R1 articulation"))
        if self._created_native_object:
            handles.append((self._object, "T03 rigid body"))
        for handle, label in handles:
            initialize = getattr(handle, "initialize", None)
            if not callable(initialize):
                continue
            try:
                initialize()
            except (AttributeError, RuntimeError, TypeError) as exc:
                raise RuntimeError(
                    f"{label} could not initialize on active stage {R1_STAGE_PATH!r}: {exc}"
                ) from exc
        self._native_handles_initialized = True

    def _validate_known_r1_scene(self) -> None:
        """Fail fast instead of producing a plausible-looking wrong episode."""
        if not self._known_r1_bound:
            return
        if self._stage is None or not self._valid_prim(self._stage, R1_ROBOT_PRIM_PATH):
            raise RuntimeError(f"active stage does not contain R1 at {R1_ROBOT_PRIM_PATH}")
        if not self._valid_prim(self._stage, R1_TARGET_OBJECT_PRIM_PATH):
            raise RuntimeError(f"active stage does not contain current target T03 at {R1_TARGET_OBJECT_PRIM_PATH}")
        names = self._joint_names()
        missing = [joint for joint in JOINT_ORDER if joint not in names]
        if missing:
            raise RuntimeError(
                "active R1 articulation is missing dataset action joints: "
                + ", ".join(missing)
            )
        missing_cameras = [
            f"{name}={path}"
            for name, path in CAMERA_PRIM_PATHS.items()
            if not self._valid_prim(self._stage, path)
        ]
        if missing_cameras:
            raise RuntimeError("active R1 stage is missing cameras: " + "; ".join(missing_cameras))
        stage_identifier = "unknown"
        try:
            stage_identifier = str(self._stage.GetRootLayer().identifier)
        except (AttributeError, RuntimeError):
            pass
        print(
            "R1 adapter: bound verified scene "
            f"stage={stage_identifier} robot={R1_ROBOT_PRIM_PATH} "
            f"target={R1_TARGET_OBJECT_PRIM_PATH} action_dofs={len(names)} "
            f"cameras={list(CAMERA_PRIM_PATHS.values())}",
            flush=True,
        )

    def _joint_names(self) -> list[str]:
        for owner in (self._robot, self._data(self._robot), self.env):
            value = self._plain(self._lookup(owner, ("dof_names", "joint_names", "names")))
            if isinstance(value, (list, tuple)):
                names = [str(item).split("/")[-1] for item in value]
                if all(name for name in names):
                    self._joint_name_to_index = {name: index for index, name in enumerate(names)}
                    return names
        return []

    def _asset_path(self, asset: str) -> str | None:
        names = {
            "object": ("object_prim_path", "bag_prim_path", "crew_lock_bag_prim_path"),
            "target_ring": ("target_ring_prim_path", "ring_prim_path"),
        }[asset]
        value = self._first(self._lookup(self.env, names), self._lookup(self._scene, names))
        if value is not None:
            return str(value)
        if asset == "object" and self._known_r1_bound:
            return R1_TARGET_OBJECT_PRIM_PATH
        if asset == "target_ring" and self._target_ring_path is not None:
            return self._target_ring_path
        return None

    @staticmethod
    def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
        import math

        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        return (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )

    def _reset_backend(self) -> None:
        owners = (self.env,) if self.env is not None else ()
        called, _ = self._call_first(owners, ("reset",), (((), {}),))
        if called:
            return
        sim = self._lookup(self.env, ("sim", "simulation"))
        called, _ = self._call_first((sim,), ("reset",), (((), {}),))
        if not called:
            self._call_first((self._scene,), ("reset",), (((), {}),))

    def _backend_vector(self, values: list[float], handle: Any = None) -> Any:
        reference = self._lookup(self._data(handle or self._robot), ("joint_pos", "joint_positions"))
        device = getattr(reference, "device", None)
        try:
            import torch

            return torch.tensor([values], dtype=torch.float32, device=device)
        except (ImportError, RuntimeError):
            return [list(values)]

    def _expand_action_to_live_dofs(self, action: list[float]) -> list[float]:
        """Map the fixed 16D dataset action into the live DOF ordering.

        The imported spacerobot articulation is not a 16-DOF articulation:
        it also contains mobile-base/wheel DOFs.  The dataset action remains
        frozen at ``JOINT_ORDER``; only the command sent to the simulator is
        expanded using the live articulation's names.
        """
        names = self._joint_names()
        if len(names) <= 16:
            return list(action[:len(names)]) if names else list(action)
        current = self._vector(
            self._lookup(self._data(self._robot), ("joint_pos", "joint_positions", "qpos")),
            len(names),
        )
        if current is None:
            current = [0.0] * len(names)
        expanded = list(current)
        for action_index, joint in enumerate(JOINT_ORDER):
            live_index = self._joint_name_to_index.get(joint)
            if live_index is not None:
                expanded[live_index] = float(action[action_index])
        return expanded

    def _set_root_pose(
        self,
        asset_name: str,
        handle: Any,
        position: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
    ) -> None:
        if handle is None:
            return
        called, _ = self._call_first(
            (self.env,), ("set_asset_pose", "set_object_pose"),
            (
                ((asset_name, position, orientation), {}),
                ((), {"asset": asset_name, "position": position, "orientation": orientation}),
            ),
        )
        if called:
            return
        for name in ("set_world_pose", "set_pose"):
            fn = getattr(handle, name, None)
            if not callable(fn):
                continue
            called, _ = self._invoke(
                fn,
                (
                    ((), {"position": position, "orientation": orientation}),
                    ((position, orientation), {}),
                ),
            )
            if called:
                return

        pose_values = [*position, *orientation]
        pose: Any = pose_values
        reference = self._lookup(self._data(handle), ("root_pos_w", "root_state_w"))
        device = getattr(reference, "device", None)
        try:
            import torch

            pose = torch.tensor([pose_values], dtype=torch.float32, device=device)
        except (ImportError, RuntimeError):
            pass
        for name in ("write_root_pose_to_sim", "write_root_state_to_sim"):
            fn = getattr(handle, name, None)
            if callable(fn):
                called, _ = self._invoke(fn, (((pose,), {}), ((pose_values,), {})))
                if called:
                    return

    def _write_joint_state(self, qpos: list[float]) -> None:
        if self._robot is None:
            return
        live_qpos = self._expand_action_to_live_dofs(qpos)
        target = self._backend_vector(live_qpos, self._robot)
        zero_vel = self._backend_vector([0.0] * len(live_qpos), self._robot)
        for name in ("write_joint_state_to_sim", "set_joint_state"):
            fn = getattr(self._robot, name, None)
            if not callable(fn):
                continue
            called, _ = self._invoke(
                fn,
                (
                    ((), {"joint_pos": target, "joint_vel": zero_vel}),
                    ((target, zero_vel), {}),
                    ((target,), {}),
                ),
            )
            if called:
                return

        data = self._data(self._robot)
        value = self._lookup(data, ("joint_pos", "joint_positions"))
        if value is not None:
            try:
                value[:] = target
            except (TypeError, ValueError, RuntimeError):
                pass

    def _advance_sim(self, action: list[float] | None, *, update_time: bool) -> None:
        sim = self._lookup(self.env, ("sim", "simulation"))
        called = False
        step_sim = getattr(self.env, "step_sim", None)
        if callable(step_sim):
            called, _ = self._invoke(step_sim, (((self._dt,), {}), ((), {})))
        if not called and sim is not None:
            step_fn = getattr(sim, "step", None)
            if callable(step_fn):
                called, _ = self._invoke(step_fn, (((), {"render": True}), ((), {})))
                if called:
                    update = getattr(self._scene, "update", None)
                    if callable(update):
                        self._invoke(update, (((self._dt,), {}), ((), {})))
        if not called and self._app is not None:
            update = getattr(self._app, "update", None)
            if callable(update):
                called, _ = self._invoke(update, (((), {}),))
        if not called:
            step_fn = getattr(self.env, "step", None)
            if callable(step_fn):
                payload = self._backend_vector(action or self._commanded_qpos, self._robot)
                called, _ = self._invoke(step_fn, (((payload,), {}), ((action or self._commanded_qpos,), {})))
        if not called:
            step_fn = getattr(self._scene, "step", None)
            if callable(step_fn):
                self._invoke(step_fn, (((self._dt,), {}), ((), {})))
        if update_time:
            self._t += self._dt

    def _start_qpos(self, scenario: ScenarioConfig) -> list[float]:
        qpos = list(DEFAULT_QPOS_VEC)
        for overrides, prefix in (
            (scenario.left_start_pose or {}, "left_"),
            (scenario.right_start_pose or {}, "right_"),
        ):
            for joint, value in overrides.items():
                if joint not in JOINT_ORDER:
                    raise ValueError(f"unknown R1 joint {joint!r} in {scenario.scenario_id}")
                if not joint.startswith(prefix):
                    raise ValueError(f"{joint!r} is not a {prefix} joint")
                qpos[JOINT_ORDER.index(joint)] = float(value)
        return qpos

    def reset(self, scenario: ScenarioConfig) -> Observation:
        if abs(float(scenario.control_hz) - self.control_hz) > 1.0e-6:
            raise ValueError("IsaacLabR1Adapter requires ScenarioConfig.control_hz == 30.0")
        self._scenario = scenario
        self._t = 0.0
        self._last_phase = Phase.RESET
        self._phase_progress = 0
        self._last_obs = None
        self._last_object_pos = None
        self._ever_lifted = False
        self._grasp_valid = {"left": False, "right": False}
        self._release_stable_elapsed = 0.0
        self._release_last_pos = None
        self._commanded_qpos = self._start_qpos(scenario)

        # Rebind after AppLauncher/Isaac Lab has created the stage.  This is
        # also what makes a raw native-USD caller work without inventing a
        # second scene configuration layer.
        self._bind_known_r1_scene()
        self._initialize_native_handles()
        self._validate_known_r1_scene()
        self._reset_backend()
        self._scene = self._lookup(self.env, ("scene",)) or self.env
        self._refresh_handles()
        configured, _ = self._call_first(
            (self.env,), ("configure_scenario", "set_scenario"), (((scenario,), {}),)
        )
        if not configured:
            # The verified physical runner positions T03 and its support in a
            # reachable workspace before constructing the adapter.  The
            # framework's historical ScenarioConfig defaults (0.40, 0, 1.0)
            # are not coordinates in spacerobot.usd.  Preserve that prepared
            # live pose unless the caller explicitly opts into world-space
            # scenario pose application.
            apply_world_pose = self._lookup(self.env, ("apply_scenario_object_pose",))
            if not self._known_r1_bound or bool(apply_world_pose):
                self._set_root_pose(
                    "object", self._object,
                    (scenario.object_x, scenario.object_y, scenario.object_z),
                    self._quat_from_rpy(scenario.object_roll, scenario.object_pitch, scenario.object_yaw),
                )
            elif self._known_r1_bound:
                print(
                    "R1 adapter: preserving prepared live T03/support pose; "
                    "set env.apply_scenario_object_pose=True to use world-space scenario coordinates",
                    flush=True,
                )
            self._set_root_pose(
                "target_ring", self._target_ring, tuple(scenario.place_target_pose),
                self._quat_from_rpy(0.0, 0.0, 0.0),
            )
        self._write_joint_state(self._commanded_qpos)
        self._advance_sim(self._commanded_qpos, update_time=False)
        obs = self._observe(scenario)
        self._object_start_z = obs.object_pos[2]
        self._last_obs = obs
        return obs

    def step(self, phase: Phase, scenario: ScenarioConfig) -> tuple[Observation, list[float]]:
        if abs(float(scenario.control_hz) - self.control_hz) > 1.0e-6:
            raise ValueError("IsaacLabR1Adapter requires ScenarioConfig.control_hz == 30.0")
        if self._scenario is None:
            self.reset(scenario)
        if phase != self._last_phase:
            self._phase_progress = 0
            if phase == Phase.RELEASE:
                self._release_stable_elapsed = 0.0
                self._release_last_pos = None
        self._last_phase = phase
        self._phase_progress += 1
        action = self._make_action(phase, scenario)
        self._set_joint_target(action)
        self._advance_sim(action, update_time=True)
        obs = self._observe(scenario)
        self._last_obs = obs
        return obs, list(action)

    def _read_qpos(self) -> list[float]:
        if self._r1_backend is not None:
            called, value = self._call_first(
                (self._r1_backend,), ("state16",), (((), {}),)
            )
            if called and isinstance(value, (tuple, list)) and len(value) >= 1:
                vector = self._vector(value[0], 16)
                if vector is not None:
                    return vector
        for owner in (self.env, self._robot):
            owners = (owner,) if owner is not None else ()
            called, value = self._call_first(owners, ("get_joint_positions", "get_qpos"), (((), {}),))
            if called:
                vector = self._project_live_joint_vector(value)
                if vector is not None:
                    return vector
        vector = self._project_live_joint_vector(
            self._lookup(self._data(self._robot), ("joint_pos", "joint_positions", "qpos"))
        )
        return vector or list(self._commanded_qpos)

    def _read_qvel(self) -> list[float]:
        if self._r1_backend is not None:
            called, value = self._call_first(
                (self._r1_backend,), ("state16",), (((), {}),)
            )
            if called and isinstance(value, (tuple, list)) and len(value) >= 2:
                vector = self._vector(value[1], 16)
                if vector is not None:
                    return vector
        vector = self._project_live_joint_vector(
            self._lookup(self._data(self._robot), ("joint_vel", "joint_velocity", "joint_velocities", "qvel"))
        )
        if vector is not None:
            return vector
        for owner in (self.env, self._robot):
            owners = (owner,) if owner is not None else ()
            called, value = self._call_first(owners, ("get_joint_velocities", "get_qvel"), (((), {}),))
            if called:
                vector = self._project_live_joint_vector(value)
                if vector is not None:
                    return vector
        return [0.0] * 16

    def _project_live_joint_vector(self, value: Any) -> list[float] | None:
        """Read a live vector by name and return exactly the dataset order."""
        names = self._joint_names()
        if names and len(names) >= 16:
            raw = self._vector(value, len(names))
            if raw is None:
                return None
            if not all(joint in self._joint_name_to_index for joint in JOINT_ORDER):
                return None
            return [raw[self._joint_name_to_index[joint]] for joint in JOINT_ORDER]
        return self._vector(value, 16)

    def _body_names(self) -> list[str]:
        for owner in (self._robot, self._data(self._robot)):
            value = self._plain(self._lookup(owner, ("body_names", "link_names", "prim_names")))
            if isinstance(value, (list, tuple)):
                return [str(item).replace("\\", "/").split("/")[-1] for item in value]
        return []

    def _body_position(self, arm: str) -> tuple[float, float, float] | None:
        link = f"{arm}_arm_link6"
        if self._r1_backend is not None:
            called, value = self._call_first(
                (self._r1_backend,), ("eef_tip_pose", "eef_link_pose"),
                (((arm,), {}),),
            )
            if called:
                if isinstance(value, (tuple, list)) and value:
                    value = value[0]
                point = self._point(value)
                if point is not None:
                    return point
        eef = self._eef_handles.get(arm)
        if eef is not None:
            called, value = self._call_first(
                (eef,), ("get_world_poses", "get_world_pose"),
                (((), {"usd": False}), ((), {})),
            )
            if called:
                if isinstance(value, (tuple, list)) and value:
                    value = value[0]
                point = self._point(value)
                if point is not None:
                    return point
        called, value = self._call_first(
            (self.env, self._robot), ("get_ee_position", "get_end_effector_position"),
            (((arm,), {}), ((), {"arm": arm})),
        )
        if called:
            point = self._point(value)
            if point is not None:
                return point
        mapping = self._first(
            self._lookup(self.env, ("ee_positions", "end_effector_positions")),
            self._lookup(self._robot, ("ee_positions", "end_effector_positions")),
        )
        if mapping is not None:
            point = self._point(self._lookup(mapping, (arm, link)))
            if point is not None:
                return point

        value = self._plain(self._lookup(self._data(self._robot), ("ee_pos_w", "end_effector_pos_w", "body_pos_w", "link_pos_w")))
        while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
            value = value[0]
        names = self._body_names()
        if isinstance(value, list) and value and isinstance(value[0], list):
            if link in names and names.index(link) < len(value):
                return self._point(value[names.index(link)])
            if len(value) >= 2:
                return self._point(value[0 if arm == "left" else 1])
        return None

    def _read_object_pose(self, scenario: ScenarioConfig) -> tuple[float, float, float]:
        if self._r1_backend is not None:
            called, value = self._call_first(
                (self._r1_backend,), ("object_pose",), (((), {}),)
            )
            if called:
                if isinstance(value, (tuple, list)) and value:
                    value = value[0]
                point = self._point(value)
                if point is not None:
                    return point
        called, value = self._call_first(
            (self.env, self._object), ("get_object_pose", "get_world_pose"), (((), {}),)
        )
        if called:
            if isinstance(value, (tuple, list)) and len(value) == 2:
                value = value[0]
            point = self._point(value)
            if point is not None:
                return point
        point = self._point(self._lookup(self._data(self._object), ("root_pos_w", "position", "pos_w")))
        if point is not None:
            return point
        state = self._vector(self._lookup(self._data(self._object), ("root_state_w",)), 3)
        if state is not None:
            return tuple(state[:3])
        raise ObservationUnavailableError(
            "T03 object pose is unavailable from the live backend or scene; "
            "refusing ScenarioConfig/YAML pose fallback"
        )

    def _read_object_velocity(self) -> tuple[float, float, float]:
        if self._r1_backend is not None:
            called, value = self._call_first(
                (self._r1_backend,), ("object_velocity",), (((), {}),)
            )
            if called:
                point = self._point(value)
                if point is not None:
                    return point
        data = self._data(self._object)
        point = self._point(self._lookup(data, ("root_lin_vel_w", "linear_velocity", "lin_vel_w")))
        if point is not None:
            return point
        state = self._vector(self._lookup(data, ("root_state_w",)), 10)
        return tuple(state[7:10]) if state is not None else (0.0, 0.0, 0.0)

    @staticmethod
    def _contacts_from_tracker_sample(sample: ContactSample) -> set[tuple[str, str]]:
        """Map tracker-owned roots to taxonomy tokens without parsing bodies.

        Raw PhysX body paths are audit evidence only.  The tracker resolves
        collision-proxy paths to watch roots before the adapter sees them, so
        canonicalization can never depend on a basename such as ``mesh_0``.
        """

        if not isinstance(sample, ContactSample):
            raise TypeError(
                "T03 contact source must provide ContactSample with watch-root ownership"
            )
        contacts: set[tuple[str, str]] = set()
        for watch_pairs in sample.pairs.values():
            for pair in watch_pairs:
                try:
                    query_token = R1_CONTACT_TOKEN_BY_WATCH_ROOT[pair.query_root]
                    counterpart_token = R1_CONTACT_TOKEN_BY_WATCH_ROOT[pair.counterpart_root]
                except KeyError as exc:
                    raise ContactRootTokenError(
                        "T03 tracker reported an unmapped watch root: "
                        f"{exc.args[0]!r}"
                    ) from exc
                contacts.add((query_token, counterpart_token))
        return contacts

    def _read_contacts(self) -> set[tuple[str, str]]:
        tracker = self._first(
            self._lookup(self.env, ("gripper_contact_tracker",)),
            self._lookup(self._scene, ("gripper_contact_tracker",)),
        )
        if tracker is None:
            raise ObservationUnavailableError(
                "T03 contact tracker is not injected; refusing unowned raw-contact parsing"
            )
        sample = tracker.sample()
        return self._contacts_from_tracker_sample(sample)

    def _read_grip_force(self, arm: str) -> float:
        if self._r1_backend is not None:
            called, value = self._call_first(
                (self._r1_backend,), ("get_grip_force", "get_gripper_force"),
                (((arm,), {}), ((), {"arm": arm})),
            )
            if called:
                try:
                    if isinstance(value, dict):
                        value = value.get(arm, 0.0)
                    return abs(float(value))
                except (TypeError, ValueError):
                    pass
        called, value = self._call_first(
            (self.env, self._robot), ("get_grip_force", "get_gripper_force"),
            (((arm,), {}), ((), {"arm": arm})),
        )
        if called:
            if isinstance(value, dict):
                value = value.get(arm)
            try:
                return abs(float(value))
            except (TypeError, ValueError):
                pass
        mapping = self._first(
            self._lookup(self.env, ("grip_force", "gripper_force", "grip_forces")),
            self._lookup(self._robot, ("grip_force", "gripper_force", "grip_forces")),
        )
        if mapping is not None:
            try:
                return abs(float(self._lookup(mapping, (arm,))))
            except (TypeError, ValueError):
                pass
        values = self._lookup(
            self._data(self._robot),
            ("applied_torque", "computed_torque", "joint_effort", "joint_force", "effort"),
        )
        names = self._joint_names()
        if names:
            values = self._project_live_joint_vector(values)
        else:
            values = self._vector(values, 16)
        if values is not None:
            indices = (6, 7) if arm == "left" else (14, 15)
            return max(abs(values[index]) for index in indices)
        return 0.0

    def _read_joint_limit_margin(self, qpos: list[float]) -> dict[str, float]:
        limits = self._plain(self._lookup(self._data(self._robot), ("soft_joint_pos_limits", "joint_pos_limits", "joint_limits")))
        while isinstance(limits, list) and len(limits) == 1 and isinstance(limits[0], list):
            limits = limits[0]
        result: dict[str, float] = {}
        names = self._joint_names()
        if isinstance(limits, list) and len(limits) >= 16:
            for index, joint in enumerate(JOINT_ORDER):
                limit_index = self._joint_name_to_index.get(joint, index)
                q_index = index
                bound = limits[limit_index] if limit_index < len(limits) else None
                if isinstance(bound, (list, tuple)) and len(bound) >= 2:
                    result[joint] = min(qpos[q_index] - float(bound[0]), float(bound[1]) - qpos[q_index])
        return result or {joint: 1.0 for joint in JOINT_ORDER}

    def _extra_from_backend(self) -> dict[str, Any]:
        called, value = self._call_first(
            (self.env,), ("get_observation_extra", "get_diagnostics"), (((), {}),)
        )
        if isinstance(value, dict):
            return dict(value)
        value = self._lookup(self.env, ("observation_extra", "diagnostics"))
        return dict(value) if isinstance(value, dict) else {}

    def _grip_threshold(self, scenario: ScenarioConfig) -> float:
        if scenario.anomaly is not None:
            try:
                return float(scenario.anomaly.extra.get("grip_force_min", 5.0))
            except (TypeError, ValueError):
                pass
        value = self._lookup(self.env, ("grip_force_min", "minimum_grip_force"))
        try:
            return float(value) if value is not None else 5.0
        except (TypeError, ValueError):
            return 5.0

    def _is_closed(self, arm: str, qpos: list[float]) -> bool:
        index = 6 if arm == "left" else 14
        close = self._gripper_value(arm, True)
        return max(abs(qpos[index] - close), abs(qpos[index + 1] - close)) <= 0.01

    def _phase_steps(self, phase: Phase, scenario: ScenarioConfig) -> int:
        configured = self._lookup(self.env, ("phase_steps", "phase_durations"))
        if isinstance(configured, dict):
            value = configured.get(phase, configured.get(phase.value))
            if value is not None:
                value = float(value)
                return max(1, round(value * self.control_hz) if value < 10 else round(value))
        return {
            Phase.RESET: 1,
            Phase.PREGRASP: 10,
            Phase.APPROACH: 30,
            Phase.CONTACT_CHECK: 1,
            Phase.GRASP: 2,
            Phase.LIFT: 20,
            Phase.HOLD: max(1, round(scenario.hold_duration_s * self.control_hz)),
            Phase.MOVE_TO_TARGET: 30,
            Phase.PLACE: 10,
            Phase.RELEASE: 15,
            Phase.RETREAT: 15,
        }.get(phase, 1)

    def _object_bbox_center(self, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        path = self._asset_path("object")
        variants = (((path,), {}), ((self._object,), {}), ((), {})) if path else (((self._object,), {}), ((), {}))
        called, value = self._call_first(
            (self.env, self._object), ("compute_obb", "compute_object_obb", "get_object_bbox", "get_object_bounding_box"), variants
        )
        if called:
            center = self._parse_bbox_center(value)
            if center is not None:
                return center
        for name in ("compute_obb", "get_bounding_box", "get_world_bounding_box"):
            fn = getattr(self._object, name, None)
            if callable(fn):
                called, value = self._invoke(fn, (((), {}),))
                if called:
                    center = self._parse_bbox_center(value)
                    if center is not None:
                        return center

        stage = self._first(self._lookup(self.env, ("stage",)), self._lookup(self._scene, ("stage",)))
        if stage is None:
            try:
                import omni.usd

                stage = omni.usd.get_context().get_stage()
            except (ImportError, AttributeError, RuntimeError):
                stage = None
        if stage is not None and path:
            try:
                from pxr import Usd, UsdGeom

                prim = stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    minimum = tuple(float(value) for value in bounds.GetMin())
                    maximum = tuple(float(value) for value in bounds.GetMax())
                    return tuple((a + b) / 2.0 for a, b in zip(minimum, maximum))
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                pass
        return fallback

    @staticmethod
    def _parse_bbox_center(value: Any) -> tuple[float, float, float] | None:
        if isinstance(value, dict):
            for key in ("center", "position", "centroid", "translation"):
                point = IsaacLabR1Adapter._point(value.get(key))
                if point is not None:
                    return point
            minimum = IsaacLabR1Adapter._point(value.get("min", value.get("minimum")))
            maximum = IsaacLabR1Adapter._point(value.get("max", value.get("maximum")))
            if minimum is not None and maximum is not None:
                return tuple((a + b) / 2.0 for a, b in zip(minimum, maximum))
            for key in ("bbox", "bounds", "obb"):
                point = IsaacLabR1Adapter._parse_bbox_center(value.get(key))
                if point is not None:
                    return point
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            minimum = IsaacLabR1Adapter._point(value[0])
            maximum = IsaacLabR1Adapter._point(value[1])
            if minimum is not None and maximum is not None:
                return tuple((a + b) / 2.0 for a, b in zip(minimum, maximum))
        return IsaacLabR1Adapter._point(value)

    @staticmethod
    def _unit(value: tuple[float, float, float]) -> tuple[float, float, float]:
        import math

        length = math.sqrt(sum(item * item for item in value))
        return (1.0, 0.0, 0.0) if length < 1.0e-8 else tuple(item / length for item in value)

    @staticmethod
    def _rotate_z(value: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
        import math

        cosine, sine = math.cos(angle), math.sin(angle)
        return (cosine * value[0] - sine * value[1], sine * value[0] + cosine * value[1], value[2])

    def _approach_vector(self, arm: str, center: tuple[float, float, float], scenario: ScenarioConfig) -> tuple[float, float, float]:
        current = None
        if self._last_obs is not None:
            current = self._last_obs.left_ee_pos if arm == "left" else self._last_obs.right_ee_pos
        if current is not None:
            candidate = tuple(current[index] - center[index] for index in range(3))
            base = self._unit(candidate) if self._norm(candidate) > 1.0e-8 else (0.0, 1.0 if arm == "left" else -1.0, 0.0)
        else:
            base = (0.0, 1.0 if arm == "left" else -1.0, 0.0)
        angle = scenario.left_approach_angle if arm == "left" else scenario.right_approach_angle
        return self._unit(self._rotate_z(base, angle))

    def _grasp_targets(self, phase: Phase, scenario: ScenarioConfig) -> dict[str, tuple[float, float, float]]:
        object_pos = self._read_object_pose(scenario)
        center = self._object_bbox_center(object_pos)
        lift = self._unit(tuple(float(value) for value in scenario.lift_direction))
        ring = tuple(float(value) for value in scenario.place_target_pose)
        targets: dict[str, tuple[float, float, float]] = {}
        for arm, offset in (("left", scenario.left_grasp_offset), ("right", scenario.right_grasp_offset)):
            grasp = tuple(center[index] + float(offset[index]) for index in range(3))
            approach = self._approach_vector(arm, center, scenario)
            if phase == Phase.PREGRASP:
                target = tuple(grasp[index] + approach[index] * scenario.pregrasp_distance for index in range(3))
            elif phase in (Phase.LIFT, Phase.HOLD):
                target = tuple(grasp[index] + lift[index] * scenario.lift_height for index in range(3))
            elif phase in (Phase.MOVE_TO_TARGET, Phase.PLACE, Phase.RELEASE):
                delta = tuple(ring[index] - center[index] for index in range(3))
                target = tuple(grasp[index] + delta[index] for index in range(3))
            elif phase == Phase.RETREAT:
                current = None
                if self._last_obs is not None:
                    current = self._last_obs.left_ee_pos if arm == "left" else self._last_obs.right_ee_pos
                origin = current or grasp
                distance = scenario.anomaly.retreat_distance if scenario.anomaly else 0.10
                target = tuple(origin[index] + approach[index] * distance for index in range(3))
            else:
                target = grasp
            targets[arm] = target
        return targets

    def _gripper_value(self, arm: str, closed: bool) -> float:
        names = ("gripper_closed_position", "closed_gripper_position") if closed else ("gripper_open_position", "open_gripper_position")
        value = self._first(self._lookup(self.env, names), self._lookup(self._robot, names))
        if isinstance(value, dict):
            value = value.get(arm)
        return float(value) if value is not None else (0.0 if closed else DEFAULT_QPOS[f"{arm}_gripper_axis1"])

    def _merge_ik(self, action: list[float], arm: str, result: Any) -> None:
        if result is None:
            return
        if isinstance(result, dict):
            nested = self._first(result.get(arm), result.get("joint_positions"), result.get("qpos"), result.get("action"))
            if nested is not None:
                self._merge_ik(action, arm, nested)
                return
            for joint, value in result.items():
                if joint in JOINT_ORDER:
                    action[JOINT_ORDER.index(joint)] = float(value)
            return
        values = self._plain(result)
        while isinstance(values, list) and values and isinstance(values[0], list):
            values = values[0]
        if not isinstance(values, (list, tuple)):
            return
        try:
            values = [float(value) for value in values]
        except (TypeError, ValueError):
            return
        if len(values) > 16 and self._r1_backend is not None:
            called, projected = self._call_first(
                (self._r1_backend,), ("action_from_full",), (((values,), {}),)
            )
            if called:
                projected_values = self._vector(projected, 16)
                if projected_values is not None:
                    action[:] = projected_values
                    return
        if len(values) >= 16:
            if len(values) == 16:
                action[:] = values[:16]
            else:
                # A full native-articulation result is only safe to project
                # by live names; never assume its first 16 entries are the
                # dataset action order.
                names = self._joint_names()
                if names and all(joint in self._joint_name_to_index for joint in JOINT_ORDER):
                    action[:] = [values[self._joint_name_to_index[joint]] for joint in JOINT_ORDER]
        elif len(values) >= 6:
            start = 0 if arm == "left" else 8
            action[start : start + 6] = values[:6]

    def _ik_result(self, arm: str, target: tuple[float, float, float], phase: Phase, scenario: ScenarioConfig) -> Any:
        orientation = self._quat_from_rpy(0.0, 0.0, scenario.left_approach_angle if arm == "left" else scenario.right_approach_angle)
        variants = (
            ((), {"arm": arm, "position": target, "orientation": orientation, "phase": phase, "scenario": scenario}),
            ((arm, target, orientation), {}),
            ((arm, target), {}),
            ((target, arm), {}),
            ((target,), {}),
        )
        called, result = self._call_first(
            (self._r1_backend, self.env, self._robot),
            ("solve_ik", "compute_ik", "get_ik_solution"),
            variants,
        )
        if called and result is not None:
            return result
        solvers = self._first(self._lookup(self.env, ("ik_solvers", "ik_controllers")), self._lookup(self._scene, ("ik_solvers", "ik_controllers")))
        solver = self._lookup(solvers, (arm, f"{arm}_ik", f"{arm}_controller"))
        if solver is not None:
            called, result = self._call_first(
                (solver,), ("solve", "compute", "command"),
                (((target, orientation), {}), ((target,), {}), ((), {"position": target, "orientation": orientation})),
            )
            if called and result is not None:
                return result
        return None

    def _make_action(self, phase: Phase, scenario: ScenarioConfig) -> list[float]:
        action = list(self._commanded_qpos)
        if phase not in (Phase.RESET, Phase.SUCCESS, Phase.FAILURE):
            targets = self._grasp_targets(phase, scenario)
            left_ik = self._ik_result("left", targets["left"], phase, scenario)
            right_ik = self._ik_result("right", targets["right"], phase, scenario)
            if self._known_r1_bound and (left_ik is None or right_ik is None):
                raise RuntimeError(
                    "active R1 scene has no usable left/right IK solution; "
                    "refusing to send a default-home action"
                )
            self._merge_ik(action, "left", left_ik)
            self._merge_ik(action, "right", right_ik)
        closed = phase in (Phase.GRASP, Phase.LIFT, Phase.HOLD, Phase.MOVE_TO_TARGET, Phase.PLACE)
        if phase == Phase.RETREAT:
            closed = bool(self._grasp_valid["left"] or self._grasp_valid["right"])
        for arm, start in (("left", 6), ("right", 14)):
            value = self._gripper_value(arm, closed)
            action[start] = value
            action[start + 1] = value
        self._commanded_qpos = list(action)
        return action

    def _set_joint_target(self, action: list[float]) -> None:
        if self._r1_backend is not None:
            called, _ = self._call_first(
                (self._r1_backend,), ("apply_full_target",),
                (((action,), {}),),
            )
            if called:
                return
        live_action = self._expand_action_to_live_dofs(action)
        target = self._backend_vector(live_action, self._robot)
        called, _ = self._call_first(
            (self.env, self._robot), ("set_joint_target", "apply_joint_position_target", "set_joint_position_target"),
            (
                ((target,), {}),
                ((live_action,), {}),
                ((action,), {}),
                ((), {"joint_pos": target}),
                ((), {"target": target}),
            ),
        )
        if not called:
            self._call_first((self.env,), ("apply_action",), (((target,), {}), ((action,), {})))

    def _observe(self, scenario: ScenarioConfig) -> Observation:
        qpos = self._read_qpos()
        qvel = self._read_qvel()
        object_pos = self._read_object_pose(scenario)
        object_vel = self._read_object_velocity()
        left_ee = self._body_position("left") or (0.0, 0.0, 0.0)
        right_ee = self._body_position("right") or (0.0, 0.0, 0.0)
        contacts = self._read_contacts()
        grip_force = {"left": self._read_grip_force("left"), "right": self._read_grip_force("right")}
        extra = self._extra_from_backend()
        closed = {arm: self._is_closed(arm, qpos) for arm in ("left", "right")}
        if self._last_phase == Phase.GRASP and self._phase_progress < 2:
            closed = {"left": False, "right": False}
        extra.setdefault("gripper_closed", closed)
        for arm in ("left", "right"):
            if closed[arm] and grip_force[arm] >= self._grip_threshold(scenario):
                self._grasp_valid[arm] = True
        if isinstance(extra.get("grasp_valid"), dict):
            self._grasp_valid.update({arm: bool(extra["grasp_valid"].get(arm, False)) for arm in ("left", "right")})
        extra.setdefault("grasp_valid", dict(self._grasp_valid))
        if self._last_phase in (Phase.LIFT, Phase.HOLD, Phase.MOVE_TO_TARGET, Phase.PLACE, Phase.RELEASE):
            if object_pos[2] > self._object_start_z + max(0.5 * scenario.lift_height, 0.02):
                self._ever_lifted = True
        extra.setdefault("object_left_support", self._ever_lifted)
        extra.setdefault("object_dropped", False)
        extra.setdefault("object_in_target", self._inside_target(object_pos, scenario))
        extra.setdefault("phase_complete", self._phase_progress >= self._phase_steps(self._last_phase or Phase.RESET, scenario))
        self._update_release_stability(scenario, object_pos, object_vel, extra)
        if self._last_object_pos is not None and self._last_phase in (Phase.LIFT, Phase.HOLD, Phase.MOVE_TO_TARGET):
            if object_pos[2] < self._object_start_z - 0.03:
                extra["object_dropped"] = True
        self._last_object_pos = object_pos
        return Observation(
            t=self._t, qpos=qpos, qvel=qvel, left_ee_pos=left_ee, right_ee_pos=right_ee,
            object_pos=object_pos, contacts=contacts,
            dist_to_grasp_frame=self._grasp_distances(left_ee, right_ee, scenario),
            grip_force=grip_force, joint_limit_margin=self._read_joint_limit_margin(qpos),
            object_vel=object_vel, extra=extra,
        )

    def _grasp_distances(self, left: tuple[float, float, float], right: tuple[float, float, float], scenario: ScenarioConfig) -> dict[str, float]:
        center = self._object_bbox_center(self._read_object_pose(scenario))
        targets = {
            "left": tuple(center[index] + float(scenario.left_grasp_offset[index]) for index in range(3)),
            "right": tuple(center[index] + float(scenario.right_grasp_offset[index]) for index in range(3)),
        }
        return {
            "left": self._norm(tuple(left[index] - targets["left"][index] for index in range(3))),
            "right": self._norm(tuple(right[index] - targets["right"][index] for index in range(3))),
        }

    def _configure_camera(self, sensor: Any) -> None:
        if sensor is None:
            return
        config = self._first(self._lookup(sensor, ("cfg", "config")), sensor)
        for target in (config, self._lookup(config, ("spawn",))):
            if target is None:
                continue
            for name, value in (("width", 640), ("height", 480), ("update_period", self._dt)):
                try:
                    setattr(target, name, value)
                except (AttributeError, TypeError):
                    pass
        fn = getattr(sensor, "set_resolution", None)
        if callable(fn):
            self._invoke(fn, (((640, 480), {}), ((), {"width": 640, "height": 480})))

    def _create_native_camera(self, name: str) -> Any:
        """Create a wrapper for an already-authored camera prim only."""
        path = CAMERA_PRIM_PATHS[name]
        if self._stage is None or not self._valid_prim(self._stage, path):
            return None
        try:
            from isaacsim.sensors.camera import Camera

            camera = Camera(path, name=f"r1_dataset_{name}", resolution=(640, 480))
            camera.initialize(attach_rgb_annotator=True)
            return camera
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            raise RuntimeError(
                f"camera {name!r} exists at {path!r} but could not initialize: {exc}"
            ) from exc

    def render_cameras(self) -> dict[str, Any]:
        sensors = self._first(self._lookup(self._scene, ("sensors",)), self._lookup(self.env, ("sensors",)))
        result: dict[str, Any] = {}
        for name in CAMERA_NAMES:
            sensor = self._camera_cache.get(name)
            if sensor is None:
                sensor = self._first(
                    self._lookup(sensors, (name, f"{name}_camera")),
                    self._lookup(self._scene, (name, f"{name}_camera")),
                    self._lookup(self.env, (name, f"{name}_camera")),
                )
                if sensor is None and self._known_r1_bound:
                    sensor = self._create_native_camera(name)
                self._camera_cache[name] = sensor
            self._configure_camera(sensor)
            data = self._data(sensor)
            rgb = self._lookup(self._lookup(data, ("output",)), ("rgb",))
            if rgb is None:
                rgb = self._lookup(data, ("rgb",))
            if rgb is None:
                called, rgb = self._call_first((sensor, self.env), ("get_rgb", "render_rgb", "read_rgb"), (((), {}), ((name,), {})))
                if not called:
                    raise RuntimeError(
                        f"camera {name!r} is missing; expected {name!r}/{name}_camera "
                        f"or prim path {CAMERA_PRIM_PATHS[name]!r}"
                    )
            shape = getattr(rgb, "shape", None)
            if shape is not None and len(shape) == 4 and shape[0] == 1:
                rgb = rgb[0]
            result[name] = rgb
        return result

    def _target_radius(self) -> float:
        value = self._first(
            self._lookup(self.env, ("target_ring_radius", "target_radius", "place_tolerance")),
            self._lookup(self._target_ring, ("radius", "target_radius")),
        )
        try:
            return max(0.01, float(value)) if value is not None else 0.08
        except (TypeError, ValueError):
            return 0.08

    def _inside_target(self, object_pos: tuple[float, float, float], scenario: ScenarioConfig) -> bool:
        import math

        ring = tuple(float(value) for value in scenario.place_target_pose)
        radial = math.sqrt((object_pos[0] - ring[0]) ** 2 + (object_pos[1] - ring[1]) ** 2)
        tolerance = self._first(self._lookup(self.env, ("target_z_tolerance",)), 0.08)
        return radial <= self._target_radius() and abs(object_pos[2] - ring[2]) <= float(tolerance)

    def _update_release_stability(
        self,
        scenario: ScenarioConfig,
        object_pos: tuple[float, float, float],
        object_vel: tuple[float, float, float],
        extra: dict[str, Any],
    ) -> None:
        if self._last_phase != Phase.RELEASE:
            self._release_stable_elapsed = 0.0
            self._release_last_pos = None
            return
        inside = extra.get("object_in_target", self._inside_target(object_pos, scenario))
        if bool(inside) and self._norm(object_vel) <= 0.10:
            movement = 0.0
            if self._release_last_pos is not None:
                movement = self._norm(tuple(object_pos[index] - self._release_last_pos[index] for index in range(3)))
            self._release_stable_elapsed = 0.0 if movement > 0.02 else self._release_stable_elapsed + self._dt
        else:
            self._release_stable_elapsed = 0.0
        self._release_last_pos = object_pos
        extra["release_stable_s"] = self._release_stable_elapsed

    def check_place_success(self, obs: Observation, scenario: ScenarioConfig) -> bool:
        if self._last_phase != Phase.RELEASE:
            return False
        extra = obs.extra or {}
        grasp_valid = extra.get("grasp_valid", self._grasp_valid)
        if not isinstance(grasp_valid, dict) or not all(bool(grasp_valid.get(arm, False)) for arm in ("left", "right")):
            return False
        if not bool(extra.get("object_left_support", self._ever_lifted)):
            return False
        if bool(extra.get("object_dropped", False)):
            return False
        if not bool(extra.get("object_in_target", self._inside_target(obs.object_pos, scenario))):
            return False
        released = extra.get("released")
        if released is None:
            released = not all(self._is_closed(arm, obs.qpos) for arm in ("left", "right"))
        if not bool(released):
            return False
        try:
            stable_s = float(extra.get("release_stable_s", self._release_stable_elapsed))
        except (TypeError, ValueError):
            stable_s = 0.0
        return stable_s >= 0.5 and self._norm(obs.object_vel) <= 0.10

    def check_timeout(self, obs: Observation, scenario: ScenarioConfig) -> bool:
        configured = self._first(
            self._lookup(self.env, ("timeout_s", "episode_timeout_s")),
            self._lookup(scenario.anomaly.extra if scenario.anomaly else None, ("timeout_s",)),
            30.0,
        )
        try:
            return float(obs.t) > float(configured)
        except (TypeError, ValueError):
            return True

class MockSimAdapter(SimAdapter):
    """Deterministic, physics-free stand-in for testing the FSM/detector/
    recovery/dataset-writer control flow without Isaac Sim.

    Not a physics simulator: qpos values are linearly-interpolated
    placeholders (valid floats, not kinematically meaningful). What it
    *does* faithfully reproduce is the fault-then-recovery timing this
    framework needs to prove out:

      - `scenario.anomaly` (the primary fault, R11-R19): injected exactly
        once, on the first attempt through its trigger phase, then the
        adapter behaves nominally on every retry — a correct
        episode_runner should show exactly one RETREAT then SUCCESS.
      - `scenario.secondary_anomaly` (F02 only): injected exactly once,
        on the *second* attempt through its trigger phase (i.e. while
        already recovering from the primary fault) — a correct
        episode_runner should go straight to FAILURE without a second
        RETREAT, since F02 is a compounding failure, not something to
        retry (taxonomy §3).

    Currently only the APPROACH-phase closures (R11/R12/R14/R16/R17/R18)
    are wired up as a *secondary* anomaly, since that's enough to
    exercise the F02 path end-to-end; R13/R15/R19 are only exercised as
    primary faults for now.
    """

    def __init__(self, approach_steps: int = 6, fault_at_progress: float = 0.4) -> None:
        self.control_hz = 30.0
        self._dt = 1.0 / self.control_hz
        self._approach_steps = approach_steps
        self._fault_at_progress = fault_at_progress
        self._t = 0.0
        self._last_phase: Phase | None = None
        self._approach_attempt = 0
        self._lift_attempt = 0
        self._phase_progress = 0
        self._fault_already_used = False
        self._secondary_fault_already_used = False
        self._qpos = list(DEFAULT_QPOS_VEC)
        self._gripper_close_steps = 0
        self._nominal_object_pos: tuple[float, float, float] = (0.4, 0.0, 1.0)

    def reset(self, scenario: ScenarioConfig) -> Observation:
        self._t = 0.0
        self._last_phase = None
        self._approach_attempt = 0
        self._lift_attempt = 0
        self._phase_progress = 0
        self._fault_already_used = False
        self._secondary_fault_already_used = False
        self._qpos = list(DEFAULT_QPOS_VEC)
        self._gripper_close_steps = 0
        # scenario-grounded, not a hardcoded placeholder — R16 (and the
        # default reported object_pos below) must displace *relative to
        # wherever this scenario's object actually is, whatever scene it's
        # calibrated against.
        self._nominal_object_pos = (scenario.object_x, scenario.object_y, scenario.object_z)
        return self._observe(dist=1.0, contacts=set(), extra={})

    # scenario classes whose fault signature fires during APPROACH and is
    # supported both as a primary and as a secondary (F02) anomaly
    _APPROACH_FAULT_CLASSES = ("R11", "R12", "R14", "R16", "R17", "R18")

    def _approach_fault_signature(
        self, an, dist: float
    ) -> tuple[float, set[tuple[str, str]], dict[str, Any]]:
        """What APPROACH should report if `an`'s fault is firing this step."""
        contacts_add: set[tuple[str, str]] = set()
        extra_add: dict[str, Any] = {}
        arms = ["left", "right"] if an.contact_arm == "both" else [an.contact_arm]

        if an.scenario_class == "R11":
            dist = max(dist, an.object_distance_at_contact + 0.05)
            for arm in arms:
                contacts_add.add((f"{arm}_arm_link6", an.surface))
        elif an.scenario_class == "R12":
            dist = max(dist, an.object_distance_at_contact + 0.05)
            for arm in arms:
                contacts_add.add((f"{arm}_gripper_link1", "object"))
                extra_add[f"{arm}_grasp_alignment_error"] = 0.10
        elif an.scenario_class == "R14":
            contacts_add.add(("left_arm_link6", "right_arm_link6"))
        elif an.scenario_class == "R16":
            threshold = an.extra.get("displacement_threshold_m", 0.05)
            nx, ny, nz = self._nominal_object_pos
            extra_add["_object_pos_override"] = (nx + threshold + 0.05, ny, nz)
        elif an.scenario_class == "R17":
            stall_max = an.extra.get("stall_time_s", 1.0)
            for arm in arms:
                extra_add[f"{arm}_approach_stall_s"] = stall_max + 0.5
        elif an.scenario_class == "R18":
            margin_min = an.extra.get("margin_min", 0.05)
            extra_add["_joint_margin_override"] = {"left_arm_joint2": margin_min - 0.01}
        return dist, contacts_add, extra_add

    def step(self, phase: Phase, scenario: ScenarioConfig) -> tuple[Observation, list[float]]:
        if phase != self._last_phase:
            self._phase_progress = 0
            if phase == Phase.APPROACH:
                self._approach_attempt += 1
            elif phase == Phase.LIFT:
                self._lift_attempt += 1
        self._last_phase = phase
        self._phase_progress += 1
        self._t += self._dt

        an = scenario.anomaly
        sec = scenario.secondary_anomaly
        fault_progress_hit = self._phase_progress / self._approach_steps >= self._fault_at_progress

        contacts: set[tuple[str, str]] = set()
        extra: dict[str, Any] = {}
        dist = max(0.0, 1.0 - self._phase_progress / self._approach_steps)

        if phase == Phase.APPROACH:
            extra["phase_complete"] = self._phase_progress >= self._approach_steps
            if (
                an is not None and an.scenario_class in self._APPROACH_FAULT_CLASSES
                and not self._fault_already_used and self._approach_attempt == 1
                and fault_progress_hit
            ):
                self._fault_already_used = True
                dist, c_add, e_add = self._approach_fault_signature(an, dist)
                contacts |= c_add
                extra.update(e_add)
            if (
                sec is not None and sec.scenario_class in self._APPROACH_FAULT_CLASSES
                and not self._secondary_fault_already_used and self._approach_attempt == 2
                and fault_progress_hit
            ):
                self._secondary_fault_already_used = True
                dist, c_add, e_add = self._approach_fault_signature(sec, dist)
                contacts |= c_add
                extra.update(e_add)
        elif phase == Phase.GRASP:
            self._gripper_close_steps += 1
            # coordination_mode/phase_offset (N08/P24): only the gripper
            # axes are staggered — arm waypoints still go out together, and
            # RELEASE is always treated as joint (see sim_data_request_v1.md
            # §5, matching r1_bimanual_dataset/core/bimanual_controller.py's
            # actual behavior). base_close_step is when the leading side
            # closes; the lagging side closes phase_offset seconds later.
            offset_steps = max(0, round(abs(scenario.phase_offset) * self.control_hz))
            base_close_step = 2
            if scenario.coordination_mode == "left_leads":
                left_close_step, right_close_step = base_close_step, base_close_step + offset_steps
            elif scenario.coordination_mode == "right_leads":
                left_close_step, right_close_step = base_close_step + offset_steps, base_close_step
            else:
                left_close_step = right_close_step = base_close_step
            closed_map = {
                "left": self._gripper_close_steps >= left_close_step,
                "right": self._gripper_close_steps >= right_close_step,
            }
            closed = closed_map["left"] and closed_map["right"]
            extra["gripper_closed"] = closed_map
            extra["phase_complete"] = closed
            if closed:
                if (
                    an is not None
                    and an.scenario_class == "R13"
                    and self._approach_attempt == 1
                    and not self._fault_already_used
                ):
                    self._fault_already_used = True
                    weak_arm = an.contact_arm if an.contact_arm != "both" else "left"
                    grip_force = {"left": 20.0, "right": 20.0}
                    grip_force[weak_arm] = 0.0
                else:
                    grip_force = {"left": 20.0, "right": 20.0}
                extra["_grip_force"] = grip_force
        elif phase == Phase.LIFT:
            extra["phase_complete"] = True
            if self._lift_attempt == 1 and an is not None and not self._fault_already_used:
                if an.scenario_class == "R19":
                    self._fault_already_used = True
                    extra["lift_stalled"] = True
                elif an.scenario_class == "R15":
                    self._fault_already_used = True
                    threshold = an.extra.get("slip_threshold_m", 0.02)
                    for arm in (["left", "right"] if an.contact_arm == "both" else [an.contact_arm]):
                        extra[f"{arm}_grasp_slip_m"] = threshold + 0.01
        elif phase == Phase.HOLD:
            hold_steps = max(1, round(scenario.hold_duration_s * self.control_hz))
            extra["phase_complete"] = self._phase_progress >= hold_steps

        grip_force = extra.pop("_grip_force", {"left": 20.0, "right": 20.0})
        object_pos_override = extra.pop("_object_pos_override", None)
        joint_margin_override = extra.pop("_joint_margin_override", None)
        obs = self._observe(
            dist=dist, contacts=contacts, extra=extra, grip_force=grip_force,
            object_pos_override=object_pos_override, joint_margin_override=joint_margin_override,
        )
        action = list(self._qpos)  # mock: hold pose; real adapter returns the IK solution
        return obs, action

    def render_cameras(self) -> dict[str, Any]:
        return {name: {"shape": (480, 640, 3), "dtype": "uint8"} for name in CAMERA_NAMES}

    def check_place_success(self, obs: Observation, scenario: ScenarioConfig) -> bool:
        return self._last_phase == Phase.RELEASE and self._phase_progress >= 2

    def check_timeout(self, obs: Observation, scenario: ScenarioConfig) -> bool:
        return obs.t > 30.0

    def _observe(
        self,
        dist: float,
        contacts: set[tuple[str, str]],
        extra: dict[str, Any],
        grip_force: dict[str, float] | None = None,
        object_pos_override: tuple[float, float, float] | None = None,
        joint_margin_override: dict[str, float] | None = None,
    ) -> Observation:
        # Control-flow fixture only: these explicit values keep missing-input
        # checks testable without letting detector-level defaults turn absent
        # data into a normal result. They are not real-physics evidence.
        extra = {
            "left_grasp_alignment_error": 0.0,
            "right_grasp_alignment_error": 0.0,
            "left_grasp_slip_m": 0.0,
            "right_grasp_slip_m": 0.0,
            "left_approach_stall_s": 0.0,
            "right_approach_stall_s": 0.0,
            "lift_stalled": False,
            "phase_complete": True,
            **extra,
        }
        joint_limit_margin = {j: 1.0 for j in JOINT_ORDER}
        if joint_margin_override:
            joint_limit_margin.update(joint_margin_override)
        nx, ny, nz = self._nominal_object_pos
        return Observation(
            t=self._t,
            qpos=list(self._qpos),
            qvel=[0.0] * len(JOINT_ORDER),
            left_ee_pos=(nx - dist, ny + 0.2, nz),
            right_ee_pos=(nx - dist, ny - 0.2, nz),
            object_pos=object_pos_override or self._nominal_object_pos,
            contacts=contacts,
            dist_to_grasp_frame={"left": dist, "right": dist},
            grip_force=grip_force or {"left": 0.0, "right": 0.0},
            joint_limit_margin=joint_limit_margin,
            extra=extra,
        )
