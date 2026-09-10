from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .bimanual_controller import BimanualController
from .camera_recorder import CameraRecorder
from ..config import ScenarioConfig
from .episode_recorder import EpisodeRecorder
from .lighting import _row_rotation_from_quaternion, ensure_fixed_dataset_lighting, set_gui_task_view
from .nominal_grasp import NominalGraspGenerator
from .payload_attachment import RuntimeGraspAttachment
from .reset_manager import ResetManager
from .robot_interface import RobotInterface
from .runtime import IsaacRuntime
from .success_evaluator import SuccessEvaluator
from .types import EpisodeResult, Phase
from .validation import validate_runtime_mapping, validate_scenario_for_execution, validate_scene_config


def _pose_vector(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    return np.concatenate([position, quaternion]).astype(np.float32)


def run_episode(
    simulation_app: Any,
    scenario: ScenarioConfig,
    scene_cfg: dict[str, Any],
    output_dir: str | Path,
    *,
    record_cameras: bool = True,
    save_data: bool = True,
    gui_preview: bool = False,
) -> EpisodeResult:
    """Run exactly one scenario.  Invalid configurations are recorded with zero frames."""

    from .reference_policy import reject_legacy_execution
    reject_legacy_execution()
    scenario.validate()
    runtime = IsaacRuntime(simulation_app, scene_cfg)
    print("R1 runner: creating robot interface", flush=True)
    runtime.open()
    if not gui_preview:
        ensure_fixed_dataset_lighting(runtime.stage, scenario.object_position)
        print("R1 runner: fixed session lighting enabled for camera renders", flush=True)
    else:
        print("R1 runner: GUI preview uses the viewport's own lighting", flush=True)
    robot = RobotInterface(scene_cfg)
    robot.set_update_fn(
        lambda: (
            simulation_app.update(),
            robot.hold_runtime_base(),
            robot.hold_runtime_joints(),
        )
    )
    # Collision schema edits must happen before the global PhysX tensor view
    # is initialized.  Doing this after prepare_physics would invalidate the
    # view on Isaac Sim 5.1 and make articulation reads fail.
    robot.configure_stationary_base_collision_filter()
    runtime.prepare_physics()
    print("R1 runner: initializing robot articulation and target object", flush=True)
    robot.initialize()
    print(f"R1 runner: articulation ready ({robot.dof_count} DOF)", flush=True)
    # prepare_physics intentionally leaves the timeline paused. Start it only
    # immediately before reset_state writes the protected root/joint pose.
    runtime.stop()
    scene_errors = validate_scene_config(scene_cfg) + validate_runtime_mapping(robot, scene_cfg)
    camera_recorder = None
    gui_view_set = False

    def controlled_step() -> None:
        # Write the session-only reset/hold state before advancing PhysX.  In
        # this floating-base USD this prevents even the first visible frame
        # from starting with a gravity-induced drop.
        robot.hold_runtime_base()
        robot.hold_runtime_joints()
        runtime.step()
        robot.hold_runtime_base()
        robot.hold_runtime_joints()

    reset_manager = ResetManager(robot, scene_cfg, controlled_step, runtime.play)
    nominal = None
    recorder = None
    evaluator = None
    invalid_reason: str | None = None
    try:
        print(f"R1 runner: resetting scenario {scenario.scenario_id}", flush=True)
        reset_manager.reset(scenario)
        print("R1 runner: generating nominal grasp and joint trajectory", flush=True)
        nominal = NominalGraspGenerator(runtime.stage, scene_cfg, robot).generate()
        recorder = None
        if save_data:
            recorder = EpisodeRecorder(output_dir, scenario, scene_cfg, nominal, scene_cfg.get("fps", 10.0))
            recorder.start(runtime.metadata(robot))
        scenario_errors = validate_scenario_for_execution(scenario, scene_cfg, robot)
        root_position, _ = robot.root_pose()
        object_position, object_quaternion = robot.object_pose()
        if scene_cfg.get("base_movement") == "disabled":
            distance = float(np.linalg.norm(object_position - root_position))
            max_distance = float(scene_cfg.get("max_object_base_distance", 1.6))
            if distance > max_distance:
                scenario_errors.append(
                    f"target object is {distance:.3f} m from robot base; base movement is disabled and conservative reach limit is {max_distance:.3f} m"
                )
        if scene_errors or scenario_errors:
            invalid_reason = "INVALID_CONFIGURATION: " + "; ".join(scene_errors + scenario_errors)
            print(f"R1 runner: precheck failed: {invalid_reason}", flush=True)
            result = EpisodeResult("INVALID", {}, None, None, invalid_reason, 0, runtime.sim_time)
            if recorder is not None:
                recorder.save_result(result)
            return result

        controller = BimanualController(robot, scene_cfg)
        behavior = (
            __import__("r1_bimanual_dataset.behaviors.normal_grasp", fromlist=["build_plan"])
            if scenario.failure_mode == "none"
            else __import__("r1_bimanual_dataset.behaviors.perturbed_grasp", fromlist=["build_plan"])
        )
        plan = behavior.build_plan(controller, scenario, nominal)
        # IK above uses a session-only state lock to make the imported USD
        # deterministic while solving.  Rollout must use the authored drives
        # and real contacts; otherwise repeatedly writing joint positions and
        # zero velocities prevents a dynamic object from being lifted.
        if scene_cfg.get("physical_rollout", True):
            robot.set_runtime_joint_state_lock(False)
            print("R1 runner: physical rollout enabled; joint state lock released", flush=True)
        attachment = RuntimeGraspAttachment(
            robot,
            scene_cfg,
            nominal,
            enabled=(
                bool(scene_cfg.get("scripted_grasp_attachment", False))
                and scenario.failure_mode == "none"
            ),
        )
        if record_cameras:
            camera_recorder = CameraRecorder(scene_cfg)
            camera_recorder.initialize()
        # Query the live camera pose after reset/initialization.  USD xform
        # caches do not include the PhysX articulation root pose, so aiming
        # from an authored-stage transform can point into empty space.
        controlled_step()

        def update_camera_views() -> None:
            nonlocal gui_view_set
            front_position, front_quaternion = robot.link_pose("torso_link4")
            left_position, left_quaternion = robot.eef_link_pose("left")
            right_position, right_quaternion = robot.eef_link_pose("right")
            parent_poses = {
                "front": (front_position, _row_rotation_from_quaternion(front_quaternion)),
                "left_wrist": (left_position, _row_rotation_from_quaternion(left_quaternion)),
                "right_wrist": (right_position, _row_rotation_from_quaternion(right_quaternion)),
            }
            if camera_recorder is not None:
                camera_recorder.update_runtime_views(
                    parent_poses,
                    nominal.object_center,
                    scene_cfg.get("camera_parent_local_offsets", {}),
                )
            if gui_preview:
                # A fixed oblique overview keeps both arms, T01 and the
                # support surface visible after runtime base teleportation.
                if not gui_view_set:
                    set_gui_task_view(
                        nominal.object_center + np.asarray([1.35, -1.55, 0.95]),
                        nominal.object_center + np.asarray([0.0, 0.0, 0.30]),
                    )
                    gui_view_set = True

        update_camera_views()
        # Newly-created camera annotators do not always have a rendered frame
        # on their first read in Isaac Sim 5.1.  Warm all three sensors using
        # the same controlled simulation step before recording.  These frames
        # are deliberately not part of the episode, so the first saved frame
        # is still phase-aligned with the commanded trajectory.
        camera_warmup_steps = max(1, int(scene_cfg.get("camera_warmup_steps", 8)))
        for _ in range(camera_warmup_steps):
            warmup_action, _ = plan.sample(runtime.sim_time)
            robot.apply_full_target(warmup_action)
            controlled_step()
            update_camera_views()
        if gui_preview:
            # The viewport can restore its authored stage camera while the
            # USD/PhysX scene finishes loading. Reapply the task framing after
            # warmup, immediately before the first visible action phase.
            set_gui_task_view(
                nominal.object_center + np.asarray([1.35, -1.55, 0.95]),
                nominal.object_center + np.asarray([0.0, 0.0, 0.30]),
            )
        initial_object_position, initial_object_quaternion = robot.object_pose()
        evaluator = SuccessEvaluator(
            scene_cfg,
            scenario,
            nominal,
            _pose_vector(initial_object_position, initial_object_quaternion),
        )
        obs_every = max(1, int(round((1.0 / float(scene_cfg.get("fps", 10.0))) / runtime.physics_dt)))
        steps = int(math.ceil(plan.duration_s / runtime.physics_dt)) + 1
        frame_count = 0
        # The first GPU render product can still be empty even after the
        # camera warmup on a cold Isaac Sim process.  Allow a few observation
        # periods for that initial render without writing a partial frame;
        # once a valid frame exists, any later missing camera is an INVALID
        # alignment error as before.
        initial_camera_retry_steps = max(obs_every * 3, 1)
        last_phase = None
        for step_index in range(steps):
            action_full, phase = plan.sample(runtime.sim_time)
            if phase.value != last_phase:
                print(
                    f"R1 runner: phase={phase.value} step={step_index}/{steps} "
                    f"sim_time={runtime.sim_time:.2f}s",
                    flush=True,
                )
                last_phase = phase.value
            robot.apply_full_target(action_full)
            controlled_step()
            update_camera_views()
            left_tip_for_attachment = _pose_vector(*robot.eef_tip_pose("left"))
            right_tip_for_attachment = _pose_vector(*robot.eef_tip_pose("right"))
            action16_for_attachment = robot.action_from_full(action_full)
            attachment.maybe_attach(
                phase,
                left_tip_for_attachment,
                right_tip_for_attachment,
                action16_for_attachment,
            )
            if phase in {Phase.CLOSE, Phase.HOLD, Phase.LIFT, Phase.LIFT_HOLD, Phase.PLACE}:
                attachment.update(left_tip_for_attachment, right_tip_for_attachment)
            elif phase == Phase.RELEASE:
                attachment.release()
            if step_index % obs_every != 0 and step_index != steps - 1:
                continue
            if camera_recorder is not None:
                camera_sample = camera_recorder.capture_after_step(step_index, runtime.sim_time)
            else:
                camera_sample = {
                    "frames": {},
                    "missing": [],
                    "aligned": True,
                    "timestamp": float(runtime.sim_time),
                }
            if camera_sample["missing"]:
                print(
                    f"R1 runner: camera frame unavailable at step={step_index}; "
                    f"missing={camera_sample['missing']}",
                    flush=True,
                )
                if frame_count == 0 and step_index < initial_camera_retry_steps:
                    print(
                        "R1 runner: skipping empty initial camera sample; "
                        "waiting for the first complete RGB frame",
                        flush=True,
                    )
                    continue
            state16, velocity16 = robot.state16()
            all_positions = robot.full_positions()
            all_velocities = robot.full_velocities()
            left_tip = _pose_vector(*robot.eef_tip_pose("left"))
            right_tip = _pose_vector(*robot.eef_tip_pose("right"))
            object_position, object_quaternion = robot.object_pose()
            object_pose = _pose_vector(object_position, object_quaternion)
            object_velocity = robot.object_velocity()
            contact_state = {
                "source": "eef_distance_and_gripper_proxy",
                "left_distance_m": float(np.linalg.norm(left_tip[:3] - nominal.left.position)),
                "right_distance_m": float(np.linalg.norm(right_tip[:3] - nominal.right.position)),
                "left_contact_proxy": bool(np.linalg.norm(left_tip[:3] - nominal.left.position) < 0.07),
                "right_contact_proxy": bool(np.linalg.norm(right_tip[:3] - nominal.right.position) < 0.07),
            }
            if not camera_sample["aligned"]:
                invalid_reason = "camera_unavailable_or_frame_mismatch"
            if not all(np.isfinite(np.asarray(x)).all() for x in [state16, velocity16, all_positions, all_velocities, left_tip, right_tip, object_pose, object_velocity]):
                invalid_reason = "NaN_or_infinite_recorded_state"
            frame_count += 1
            if recorder is not None:
                recorder.append(
                    camera_sample,
                    state16,
                    velocity16,
                    robot.action_from_full(action_full),
                    all_positions,
                    all_velocities,
                    left_tip,
                    right_tip,
                    object_pose,
                    object_velocity,
                    phase.value,
                    contact_state,
                )
            evaluator.update(
                {
                    "object_pose": object_pose,
                    "object_velocity": object_velocity,
                    "left_eef_pose": left_tip,
                    "right_eef_pose": right_tip,
                    "joint_position_16d": state16,
                    "phase": phase.value,
                }
            )
            if invalid_reason:
                break
        runtime.stop()
        result = evaluator.finalize(runtime.sim_time, frame_count, invalid_reason)
        if recorder is not None:
            recorder.save_result(result)
        return result
    except Exception as exc:
        invalid_reason = f"runtime_exception: {type(exc).__name__}: {exc}"
        counters = {
            "ik_failures": int("IK" in type(exc).__name__ or "IK" in str(exc)),
            "reset_failures": int("reset" in str(exc).lower()),
            "controller_failures": int("controller" in str(exc).lower()),
        }
        if recorder is None:
            # A reset or scene binding failure happens before a nominal grasp
            # can be constructed; still emit a traceable invalid episode.
            try:
                fallback = EpisodeRecorder(output_dir, scenario, scene_cfg, None, scene_cfg.get("fps", 10.0))
                fallback.start({"runtime_exception": invalid_reason})
                result = EpisodeResult("INVALID", {}, None, None, invalid_reason, 0, runtime.sim_time, **counters)
                fallback.save_result(result)
                return result
            except Exception:
                return EpisodeResult("INVALID", {}, None, None, invalid_reason, 0, runtime.sim_time)
        recorded_frames = len(recorder.frames) if recorder is not None else 0
        result = EpisodeResult("INVALID", {}, None, None, invalid_reason, recorded_frames, runtime.sim_time, **counters)
        if recorder is not None:
            recorder.save_result(result)
        return result
    finally:
        reset_manager.emergency_stop()
        runtime.stop()
