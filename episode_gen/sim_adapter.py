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

CAMERA_NAMES = ("front", "left_wrist", "right_wrist")

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

# Verified against galaxea_lab,
# .../galaxea/manager_based/lift/config/joint_pos_env_cfg.py :: R1LiftEnvCfg
CAMERA_PRIM_PATHS: dict[str, str] = {
    "front": "{ENV_REGEX_NS}/Robot/torso_link4/front_camera",
    "left_wrist": "{ENV_REGEX_NS}/Robot/left_arm_link6/left_wrist_camera",
    "right_wrist": "{ENV_REGEX_NS}/Robot/right_arm_link6/right_wrist_camera",
}


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
    """Real adapter skeleton for the Galaxea R1 in Isaac Lab.

    NOT runnable in this sandbox — there is no Isaac Sim here to verify
    against. What's pre-filled below is grounded in the actual
    userguide-galaxea/galaxea_lab source (joint names, camera prim
    paths, default pose); everything marked TODO is where this needs to
    be finished against your running Isaac Lab install and, once
    available, the scene USD (object/table/target-ring prim paths and
    poses — currently unknown here, see spacerobot.usd handoff).

    Suggested wiring, following the pattern already used in
    galaxea/manager_based/lift/config/joint_pos_env_cfg.py:
      - build an InteractiveScene with GALAXEA_R1_HIGH_PD_GRIPPER_CFG as
        `robot`, plus `table`/`object`/`target_ring` AssetBaseCfg/
        RigidObjectCfg entries once the USD prim paths are known
      - override GALAXEA_CAMERA_CFG per docs/dataset_spec_v0.1.md §3
        (640x480, update_period=1/30) for all three cameras above
      - use ContactSensorCfg on the arm/gripper links referenced by
        detectors.py (`{arm}_arm_link6`, table/object/cabinet prims) to
        populate Observation.contacts
      - compute nominal grasp frames from the object's bounding box
        (Isaac Sim `compute_obb` on the object prim) + scenario
        left/right_grasp_offset, solve IK (e.g. omni.isaac.lab's
        DifferentialIKController, already used elsewhere in galaxea_lab)
        to get the 16D joint target for APPROACH/GRASP/LIFT/etc.
    """

    def __init__(self, env: Any) -> None:
        # `env` is expected to be an Isaac Lab ManagerBasedEnv / a raw
        # InteractiveScene handle built by the caller — deliberately not
        # constructed here, since that requires a running Isaac Sim app.
        self.env = env
        self.control_hz = 30.0

    def reset(self, scenario: ScenarioConfig) -> Observation:
        raise NotImplementedError(
            "TODO: reset InteractiveScene, spawn/move object to "
            "(scenario.object_x/y/z, roll/pitch/yaw), set robot to "
            "DEFAULT_QPOS or scenario.left_start_pose/right_start_pose, "
            "step sim once, and return the resulting Observation."
        )

    def step(self, phase: Phase, scenario: ScenarioConfig) -> tuple[Observation, list[float]]:
        raise NotImplementedError(
            "TODO: compute the 16D absolute joint target for `phase` "
            "(bounding-box grasp frame + IK for APPROACH/GRASP/LIFT/"
            "MOVE_TO_TARGET/PLACE; binary/continuous gripper command for "
            "GRASP/RELEASE), write it to the articulation, step the sim, "
            "read back contacts/forces/joint state into an Observation."
        )

    def render_cameras(self) -> dict[str, Any]:
        raise NotImplementedError(
            "TODO: read scene.sensors['front_camera'/'left_wrist_camera'/"
            "'right_wrist_camera'].data.output['rgb'] once cameras are "
            "configured per docs/dataset_spec_v0.1.md §3."
        )

    def check_place_success(self, obs: Observation, scenario: ScenarioConfig) -> bool:
        raise NotImplementedError("TODO: TRA-01 §7 criteria against the real object/ring prims.")

    def check_timeout(self, obs: Observation, scenario: ScenarioConfig) -> bool:
        raise NotImplementedError("TODO: compare obs.t against an episode time budget.")


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
                contacts_add.add((f"{arm}_arm_link6", "object"))
                extra_add[f"{arm}_grasp_alignment_error"] = 0.10
        elif an.scenario_class == "R14":
            contacts_add.add(("left_arm_link6", "right_arm_link6"))
        elif an.scenario_class == "R16":
            threshold = an.extra.get("displacement_threshold_m", 0.05)
            extra_add["_object_pos_override"] = (0.40 + threshold + 0.05, 0.0, 1.0)
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
            closed = self._gripper_close_steps >= 2
            extra["gripper_closed"] = {"left": closed, "right": closed}
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
        joint_limit_margin = {j: 1.0 for j in JOINT_ORDER}
        if joint_margin_override:
            joint_limit_margin.update(joint_margin_override)
        return Observation(
            t=self._t,
            qpos=list(self._qpos),
            qvel=[0.0] * len(JOINT_ORDER),
            left_ee_pos=(0.4 - dist, 0.2, 1.0),
            right_ee_pos=(0.4 - dist, -0.2, 1.0),
            object_pos=object_pos_override or (0.4, 0.0, 1.0),
            contacts=contacts,
            dist_to_grasp_frame={"left": dist, "right": dist},
            grip_force=grip_force or {"left": 0.0, "right": 0.0},
            joint_limit_margin=joint_limit_margin,
            extra=extra,
        )
