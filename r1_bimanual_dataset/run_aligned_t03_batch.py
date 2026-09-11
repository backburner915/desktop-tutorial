"""Collect aligned T03 RAW episodes with the verified physical R1 runner.

This is deliberately a new entry point.  ``run_physical_grasp.py`` remains
the Golden Reference and is not modified.  The batch runner reuses its exact
stage placement, fixed-base Session-Layer setup, finite-difference IK and
position-drive control, then adds the dataset recorder and synchronized
camera reads around that physical rollout.

The output is RAW data, not a LeRobot database.  Each episode contains
``observation.state``/``action`` with the fixed 16D ordering, all 19 live
joint states, three PNG camera streams, timestamps and scenario metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from r1_bimanual_dataset.config import ACTION_DIM, DATASET_FPS, ScenarioConfig, load_scene_config
from r1_bimanual_dataset.core.camera_recorder import CameraRecorder
from r1_bimanual_dataset.core.episode_recorder import EpisodeRecorder
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.core.robot_interface import (
    quat_conjugate,
    quat_from_rpy,
    quat_rotate,
)
from r1_bimanual_dataset.core.types import EpisodeResult, Phase
from r1_bimanual_dataset.run_physical_grasp import (
    _ensure_runtime_target_pose_ops,
    _move_existing_support_top,
    _physical_scene_config,
)
from r1_bimanual_dataset.tools.physical_reference_probe import (
    _make_existing_r1_fixed,
    _open_stage,
    _place_existing_robot_in_session,
    _source_base_link_world_translation,
    _world_bbox,
    select_long_edge_pose,
)


T03_TARGET = "/World/TaskSetup/MovablePayloads/T03"
SUPPORT_PATH = "/World/TaskSetup/Fixtures/StorageRack/Top"
ROBOT_SOURCE = "/World/garobot2_driveable_final/r1_DVT_colored"
CONTROL_HZ = 30.0
PHYSICS_DT = 1.0 / 120.0


def _aligned_scene_config(base: dict) -> dict:
    """Return one explicit dataset config without changing scene_config.json."""

    cfg = dict(base)
    cfg.update(
        {
            "stage_path": r"D:\Galaxea_Lab-galaxea-main\spacerobot.usd",
            "stage_url": r"D:\Galaxea_Lab-galaxea-main\spacerobot.usd",
            "robot": "Galaxea R1 DVT",
            "robot_prim": ROBOT_SOURCE,
            "target_object": T03_TARGET,
            "left_eef": f"{ROBOT_SOURCE}/left_arm_link6",
            "right_eef": f"{ROBOT_SOURCE}/right_arm_link6",
            "fps": CONTROL_HZ,
            "control_hz": CONTROL_HZ,
            "physics_dt_fallback": PHYSICS_DT,
            "camera_resolution": [640, 480],
            "physical_rollout": True,
            "scripted_grasp_attachment": False,
            "base_movement": "disabled",
            "runtime_base_lock": False,
            "runtime_joint_state_lock": False,
            "motion_profile": "steady_place",
            "nominal_grasp_horizontal": True,
            "nominal_object_center_from_rigid_pose": True,
            "use_live_initial_state": False,
            "place_hover_height_m": 0.0,
            "eef_tip_offsets": {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]},
            "ik_jacobian_source": "finite_difference",
            "ik_max_iterations": 80,
            "ik_damping": 0.05,
            "ik_step_limit": 0.08,
            "ik_line_search": True,
            "ik_accept_position_error": 0.035,
            "camera_alignment_rule": "one_physics_control_step_then_all_three_reads",
        }
    )
    # The fallback seeds belong to an older, locked-root pipeline.  A failed
    # physical IK solve is an honest INVALID episode, never a hidden teleport.
    cfg.pop("ik_fallback_seed", None)
    cfg["ik_fallback_seeds"] = {}
    return cfg


def _gf_quaternion(value) -> np.ndarray:
    return np.asarray(
        [
            float(value.GetReal()),
            float(value.GetImaginary()[0]),
            float(value.GetImaginary()[1]),
            float(value.GetImaginary()[2]),
        ],
        dtype=np.float64,
    )


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    norm = max(float(np.linalg.norm([w, x, y, z])), 1e-12)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _camera_parent_poses(robot) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    front_position, front_q = robot.link_pose("torso_link4")
    left_position, left_q = robot.eef_link_pose("left")
    right_position, right_q = robot.eef_link_pose("right")
    return {
        "front": (front_position, _quat_to_matrix(front_q)),
        "left_wrist": (left_position, _quat_to_matrix(left_q)),
        "right_wrist": (right_position, _quat_to_matrix(right_q)),
    }


def _scenario_runtime_target(stage, scenario: ScenarioConfig):
    """Calibrate a scenario pose from the authored T03 geometry.

    Scenario object_position is a small perturbation around the authored
    T03 bbox centre.  The robot/support calibration remains the known-good
    physical workspace, so only that delta is transferred to the runtime
    workspace.  The rigid-body origin offset is rotated with the requested
    object orientation instead of assuming the body origin equals its bbox
    centre.
    """

    from pxr import UsdGeom

    obj_min, obj_max = _world_bbox(stage, T03_TARGET)
    bbox_center = (obj_min + obj_max) / 2.0
    dimensions = obj_max - obj_min
    target_prim = stage.GetPrimAtPath(T03_TARGET)
    cache = UsdGeom.XformCache()
    source_world = cache.GetLocalToWorldTransform(target_prim).RemoveScaleShear()
    source_origin = np.asarray(source_world.ExtractTranslation(), dtype=np.float64)
    source_q = _gf_quaternion(source_world.ExtractRotationQuat())
    source_local_offset = quat_rotate(quat_conjugate(source_q), bbox_center - source_origin)
    target_q = quat_from_rpy(*np.asarray(scenario.object_orientation, dtype=np.float64))
    delta = np.asarray(scenario.object_position, dtype=np.float64) - bbox_center
    return bbox_center, dimensions, source_local_offset, target_q, delta


def _contact_state(robot, nominal, command, tracker=None) -> dict:
    left_midpoint, _ = robot.gripper_midpoint_pose("left")
    right_midpoint, _ = robot.gripper_midpoint_pose("right")
    left_distance = float(np.linalg.norm(left_midpoint - nominal.left.position))
    right_distance = float(np.linalg.norm(right_midpoint - nominal.right.position))
    closed = np.asarray(command, dtype=np.float64)[[6, 7, 14, 15]] <= 0.005
    proxy_left = bool(left_distance < 0.10 and np.all(closed[:2]))
    proxy_right = bool(right_distance < 0.10 and np.all(closed[2:]))
    value = {
        "source": "physx_raw_contact_plus_geometric_proxy" if tracker is not None else "geometric_proxy",
        "left_contact_proxy": proxy_left,
        "right_contact_proxy": proxy_right,
        "left_distance_m": left_distance,
        "right_distance_m": right_distance,
        "grip_force": {"left": None, "right": None},
    }
    if tracker is not None:
        sample = tracker.sample()
        value.update(
            {
                "left_contact_raw": bool(sample.left),
                "right_contact_raw": bool(sample.right),
                "raw_pairs": sample.pairs,
            }
        )
    else:
        value.update({"left_contact_raw": None, "right_contact_raw": None, "raw_pairs": {}})
    return value


def _fallback_invalid(output_root: Path, scenario: ScenarioConfig, cfg: dict, reason: str) -> EpisodeResult:
    recorder = EpisodeRecorder(output_root, scenario, cfg, None, CONTROL_HZ)
    recorder.start(
        {
            "runner": "run_aligned_t03_batch.py",
            "status": "invalid_before_rollout",
            "invalid_reason": reason,
        }
    )
    result = EpisodeResult(
        "INVALID", {}, None, None, reason, 0, 0.0,
    )
    recorder.save_result(result)
    return result


def run_one(app, scenario: ScenarioConfig, base_cfg: dict, output_root: Path, use_cameras: bool, use_contact_tracker: bool) -> EpisodeResult:
    """Run one scenario in real PhysX and save it regardless of outcome."""

    cfg = _aligned_scene_config(base_cfg)
    stage = None
    timeline = None
    recorder = None
    evaluator = None
    frame_count = 0
    start_wall = time.perf_counter()
    try:
        stage, timeline = _open_stage(app, cfg["stage_path"], warmup_updates=0)
        authored_min, authored_max = _world_bbox(stage, T03_TARGET)
        authored_center, dimensions, local_offset, target_q, delta = _scenario_runtime_target(stage, scenario)
        desired_base, base_rot, placement = select_long_edge_pose(
            authored_min, authored_max, side="near", stand_off_m=0.72
        )
        source_base_offset = _source_base_link_world_translation(stage, ROBOT_SOURCE)
        desired_link = desired_base.copy()
        desired_link[2] = source_base_offset[2]
        _place_existing_robot_in_session(stage, ROBOT_SOURCE, desired_link - source_base_offset, base_rot)

        workspace_center = desired_link + quat_rotate(base_rot, np.asarray([0.40, 0.0, 1.06]))
        workspace_center[:2] += delta[:2]
        runtime_object_position = workspace_center - quat_rotate(target_q, local_offset)
        support_min, support_max = _move_existing_support_top(
            stage, SUPPORT_PATH, workspace_center, dimensions
        )
        _ensure_runtime_target_pose_ops(stage, T03_TARGET)
        fixed_root_path = _make_existing_r1_fixed(
            stage, ROBOT_SOURCE, relocate_articulation_root=False
        )

        # Contact reporting must be authored before PhysX initialization.
        tracker = None
        contact_status = "disabled"
        if use_contact_tracker:
            from r1_bimanual_dataset.core.contact_tracker import GripperContactTracker

            tracker = GripperContactTracker(stage, ROBOT_SOURCE, T03_TARGET)
            tracker.prepare()
            contact_status = "prepared"

        from isaacsim.core.simulation_manager import SimulationManager
        from r1_bimanual_dataset.core.bimanual_controller import BimanualController
        from r1_bimanual_dataset.core.nominal_grasp import NominalGraspGenerator
        from r1_bimanual_dataset.core.robot_interface import RobotInterface
        from r1_bimanual_dataset.core.success_evaluator import SuccessEvaluator

        timeline.play()
        SimulationManager.initialize_physics()
        for _ in range(3):
            app.update()
        timeline.stop()

        cfg = _physical_scene_config(cfg, fixed_root_path)
        cfg["target_object"] = T03_TARGET
        cfg["object_bbox_dimensions_m"] = dimensions.tolist()
        cfg["nominal_object_center_override"] = workspace_center.tolist()
        robot = RobotInterface(cfg, update_fn=app.update)
        robot.initialize()
        if tracker is not None:
            tracker.initialize()
            contact_status = "active"

        # Keep the same post-initialize synchronization window as the Golden
        # Reference.  On this imported USD, the RigidPrim world-pose write is
        # ignored if it is issued immediately after tensor initialization.
        for _ in range(3):
            app.update()
        robot.object.set_world_poses(
            runtime_object_position.reshape(1, 3), target_q.reshape(1, 4)
        )
        robot.object.set_velocities(np.zeros((1, 6), dtype=np.float64))
        immediate_position, _ = robot.object_pose()
        print(
            f"T03 dataset episode {scenario.scenario_id}: target set "
            f"position={np.round(immediate_position, 4).tolist()} "
            f"requested={np.round(runtime_object_position, 4).tolist()}",
            flush=True,
        )
        for _ in range(15):
            app.update()
        settled_position, _ = robot.object_pose()
        settled_velocity = robot.object_velocity()
        print(
            f"T03 dataset episode {scenario.scenario_id}: pre-plan settle "
            f"position={np.round(settled_position, 4).tolist()} "
            f"requested_body={np.round(runtime_object_position, 4).tolist()} "
            f"velocity={np.round(settled_velocity, 4).tolist()} "
            f"horizontal_delta={np.linalg.norm(settled_position[:2] - runtime_object_position[:2]):.4f}m "
            f"speed={np.linalg.norm(settled_velocity[:3]):.4f}m/s",
            flush=True,
        )
        if (
            not np.all(np.isfinite(settled_position))
            or not np.all(np.isfinite(settled_velocity))
            or float(np.linalg.norm(settled_position[:2] - runtime_object_position[:2])) > 0.03
            or float(np.linalg.norm(settled_velocity[:3])) > 0.05
        ):
            raise RuntimeError(
                "T03 did not return to a stable supported runtime pose before planning"
            )

        # IK is deliberately planned using the same physical controller and
        # local finite-difference FK as the successful Golden Reference.
        timeline.play()
        for _ in range(5):
            app.update()
        nominal = NominalGraspGenerator(stage, cfg, robot).generate()
        trajectory = BimanualController(robot, cfg).build(scenario, nominal)

        # IK trial configurations can touch the payload.  Reset both pose and
        # velocity, then let real support contact settle before recording.
        robot.object.set_world_poses(
            runtime_object_position.reshape(1, 3), target_q.reshape(1, 4)
        )
        robot.object.set_velocities(np.zeros((1, 6), dtype=np.float64))
        for _ in range(30):
            app.update()
        reset_position, _ = robot.object_pose()
        reset_velocity = robot.object_velocity()
        if (
            not np.all(np.isfinite(reset_position))
            or not np.all(np.isfinite(reset_velocity))
            or float(np.linalg.norm(reset_position[:2] - runtime_object_position[:2])) > 0.03
            or float(np.linalg.norm(reset_velocity[:3])) > 0.05
        ):
            raise RuntimeError("T03 reset after IK planning is not physically stable")

        if tracker is not None:
            # Contact reporting was prepared before initialization; this first
            # read confirms the interface is live before the first frame.
            tracker.sample()

        camera_recorder = None
        if use_cameras:
            camera_recorder = CameraRecorder(cfg, width=640, height=480)
            camera_recorder.initialize()

        initial_object_position, initial_object_quaternion = robot.object_pose()
        initial_object_pose = np.concatenate([initial_object_position, initial_object_quaternion])
        evaluator = SuccessEvaluator(cfg, scenario, nominal, initial_object_pose)
        recorder = EpisodeRecorder(output_root, scenario, cfg, nominal, CONTROL_HZ)
        recorder.start(
            {
                "runner": "run_aligned_t03_batch.py",
                "reference_runner": "run_physical_grasp.py",
                "control_hz": CONTROL_HZ,
                "physics_dt": PHYSICS_DT,
                "robot_dof_count": robot.dof_count,
                "all_joint_names": robot.all_joint_names,
                "action_dim": ACTION_DIM,
                "action_joint_names_verified": list(cfg["action_joint_names"]),
                "fixed_root_path": fixed_root_path,
                "target_runtime_body_position": runtime_object_position.tolist(),
                "target_runtime_bbox_center": workspace_center.tolist(),
                "target_runtime_quaternion_wxyz": target_q.tolist(),
                "support_path": SUPPORT_PATH,
                "support_bbox": {"min": support_min.tolist(), "max": support_max.tolist()},
                "base_placement": placement,
                "camera_resolution": [640, 480] if use_cameras else None,
                "contact_tracker": contact_status,
            }
        )

        if camera_recorder is not None:
            # Warm up annotators without writing frames.  All three camera
            # reads still happen only after the same shared simulation step.
            for _ in range(20):
                app.update()

        # Ensure the first recorded command is the trajectory's settled
        # initial target, not a transient planning pose.
        sim_time = 0.0
        dt = 1.0 / CONTROL_HZ
        last_phase = None
        last_bbox = _world_bbox(stage, T03_TARGET)
        bbox_refresh = 4
        for frame_index, t in enumerate(np.arange(0.0, trajectory.duration_s + 1e-9, dt)):
            command, phase = trajectory.sample(float(t))
            if command.shape != (robot.dof_count,):
                raise RuntimeError(f"full command shape mismatch: {command.shape}")
            robot.apply_full_target(command)
            # One observation/control period consists of four 120Hz physics
            # updates, then one synchronized state+camera read.
            for _ in range(max(1, int(round(dt / PHYSICS_DT)))):
                app.update()
            sim_time += dt

            if camera_recorder is not None:
                target_position, _ = robot.object_pose()
                camera_recorder.update_runtime_views(
                    _camera_parent_poses(robot),
                    target_position,
                    cfg.get("camera_parent_local_offsets", {}),
                )
                camera_sample = camera_recorder.capture_after_step(frame_index, sim_time)
            else:
                camera_sample = {
                    "step_index": frame_index,
                    "timestamp": sim_time,
                    "frames": {},
                    "missing": [],
                    "aligned": True,
                }

            object_pos, object_quaternion = robot.object_pose()
            object_velocity = robot.object_velocity()
            left_eef, left_quaternion = robot.eef_tip_pose("left")
            right_eef, right_quaternion = robot.eef_tip_pose("right")
            measured_state, measured_velocity = robot.state16()
            action16 = robot.action_from_full(command)
            if measured_state.shape != (ACTION_DIM,) or action16.shape != (ACTION_DIM,):
                raise RuntimeError(
                    f"16D alignment mismatch: state={measured_state.shape}, action={action16.shape}"
                )
            if frame_index % bbox_refresh == 0:
                last_bbox = _world_bbox(stage, T03_TARGET)
            object_bbox_min, object_bbox_max = last_bbox
            contact = _contact_state(robot, nominal, command, tracker)
            evaluator.update(
                {
                    "object_pose": np.concatenate([object_pos, object_quaternion]),
                    "object_velocity": object_velocity,
                    "left_eef_pose": np.concatenate([left_eef, left_quaternion]),
                    "right_eef_pose": np.concatenate([right_eef, right_quaternion]),
                    "joint_position_16d": measured_state,
                    "joint_velocity_16d": measured_velocity,
                    "action_16d": action16,
                    "gripper_command": action16,
                    "phase": phase.value,
                    "object_bbox_min": object_bbox_min,
                    "object_bbox_max": object_bbox_max,
                    "support_bbox_min": support_min,
                    "support_bbox_max": support_max,
                }
            )
            recorder.append(
                camera_sample=camera_sample,
                state16=measured_state,
                velocity16=measured_velocity,
                action16=action16,
                all_joint_positions=robot.full_positions(),
                all_joint_velocities=robot.full_velocities(),
                left_eef_pose=np.concatenate([left_eef, left_quaternion]),
                right_eef_pose=np.concatenate([right_eef, right_quaternion]),
                object_pose=np.concatenate([object_pos, object_quaternion]),
                object_velocity=object_velocity,
                phase=phase.value,
                contact_state=contact,
            )
            frame_count += 1
            if phase != last_phase:
                print(
                    f"T03 dataset episode {scenario.scenario_id}: phase={phase.value} "
                    f"object={np.round(object_pos, 4).tolist()}",
                    flush=True,
                )
                last_phase = phase

        # A missing camera frame is data-invalid even if the physical task
        # happened to finish.  The frames are retained for diagnosis.
        result = evaluator.finalize(
            trajectory.duration_s,
            frame_count,
            invalid_reason="camera_missing_frame" if recorder.missing_camera_frames else None,
        )
        recorder.save_result(result)
        timeline.stop()
        print(
            f"T03 dataset episode {scenario.scenario_id}: {result.actual_outcome} "
            f"frames={frame_count} wall_s={time.perf_counter() - start_wall:.1f}",
            flush=True,
        )
        return result
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        print(f"T03 dataset episode {scenario.scenario_id}: INVALID {reason}", flush=True)
        traceback.print_exc()
        if recorder is not None:
            result = EpisodeResult("INVALID", {}, None, None, reason, frame_count, 0.0)
            recorder.save_result(result)
            return result
        return _fallback_invalid(output_root, scenario, cfg, reason)
    finally:
        if timeline is not None:
            try:
                timeline.stop()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect aligned T03 R1 bimanual RAW episodes")
    parser.add_argument("--scenarios-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-config", type=Path, default=WORKSPACE_ROOT / "scene_config.json")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--gui", action="store_true", help="show the same run in Isaac Sim; headless is recommended")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--no-contact-tracker", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    scenario_paths = sorted(args.scenarios_dir.glob("scenario_*.json"))[: args.limit]
    if len(scenario_paths) < args.limit:
        raise FileNotFoundError(
            f"expected at least {args.limit} scenario_*.json files in {args.scenarios_dir}, found {len(scenario_paths)}"
        )
    base_cfg = load_scene_config(args.scene_config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "collection_config.json").write_text(
        json.dumps(
            {
                "runner": str(Path(__file__).resolve()),
                "stage": r"D:\Galaxea_Lab-galaxea-main\spacerobot.usd",
                "target_object": T03_TARGET,
                "control_hz": CONTROL_HZ,
                "physics_dt": PHYSICS_DT,
                "camera_resolution": [640, 480] if not args.no_cameras else None,
                "action_dim": ACTION_DIM,
                "scenario_files": [str(path.resolve()) for path in scenario_paths],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=not args.gui, renderer="RayTracedLighting" if args.gui else "None"))
    results: list[dict] = []
    consecutive_invalid = 0
    try:
        for index, path in enumerate(scenario_paths, start=1):
            scenario = ScenarioConfig.from_json(path)
            if scenario.target_object not in {None, T03_TARGET}:
                raise ValueError(f"scenario {path} is not aligned to T03: {scenario.target_object}")
            result = run_one(
                app,
                scenario,
                base_cfg,
                args.output,
                use_cameras=not args.no_cameras,
                use_contact_tracker=not args.no_contact_tracker,
            )
            results.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_file": str(path),
                    **result.as_dict(),
                }
            )
            if result.actual_outcome == "INVALID":
                consecutive_invalid += 1
                if consecutive_invalid >= 3:
                    print("T03 dataset batch stopped: 3 consecutive INVALID episodes", flush=True)
                    break
            else:
                consecutive_invalid = 0
            print(
                f"T03 dataset batch progress: {len(results)}/{len(scenario_paths)} "
                f"success={sum(item['actual_outcome'] == 'SUCCESS' for item in results)} "
                f"failure={sum(item['actual_outcome'] == 'FAILURE' for item in results)} "
                f"invalid={sum(item['actual_outcome'] == 'INVALID' for item in results)}",
                flush=True,
            )
            if not app.is_running():
                print("T03 dataset batch stopped: SimulationApp is no longer running", flush=True)
                break
        (args.output / "collection_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    finally:
        app.close()
    return 0 if len(results) == len(scenario_paths) else 2


if __name__ == "__main__":
    raise SystemExit(main())
