"""Dynamic Gate B: independent single-gripper contact trials.

This tool is deliberately a gate diagnostic, not a dataset runner.  It uses
the Session-Layer physical scene, a real fixed joint, authored position drives,
and the independent PhysX contact-report tracker.  It never attaches the
payload, filters collisions, locks the root/joints during motion, or writes an
episode.  A temporary tabletop stand can be selected explicitly while the
production table height is being resolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "r1_bimanual_dataset"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import load_scene_config
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.core.reference_policy import require_candidate
from r1_bimanual_dataset.core.physical_scene import prepare_fixed_target_scene
from r1_bimanual_dataset.core.types import Pose


ASSETS = {
    "sugar_box": PKG / "assets" / "004_sugar_box_physics.usd",
    "cracker_box": PKG / "assets" / "003_cracker_box_physics.usd",
}


def _row(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array[0] if array.ndim > 1 else array


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_gate_a(candidate: str) -> tuple[dict[str, object], dict[str, object]]:
    stem = "004_sugar_box_physics" if candidate == "sugar_box" else "003_cracker_box_physics"
    geometry = _read_json(ROOT / "reports" / f"candidate_{stem}.json")
    static = _read_json(ROOT / "reports" / f"candidate_{stem}_static_audit.json")
    if geometry.get("geometry_pass") is not True or geometry.get("gripper_fit_pass") is not True:
        raise RuntimeError(f"Gate A geometry/aperture is not PASS for {candidate}")
    if static.get("gate_a_candidate_pass") is not True:
        raise RuntimeError(f"Gate A static settle is not PASS for {candidate}")
    return geometry, static


def _trial(
    app: object,
    timeline: object,
    robot: object,
    tracker: object,
    station_frames: dict[str, dict[str, np.ndarray]],
    side: str,
    baseline_object: tuple[np.ndarray, np.ndarray],
    *,
    approach_steps: int,
    close_steps: int,
    hold_steps: int,
    settle_steps: int,
    trial_index: int,
) -> dict[str, object]:
    """Run one physical close-and-hold from a paused, reset setup state."""

    # All state writes occur while paused, before the next physics tick.  The
    # subsequent close/hold loop sends a complete full-articulation target on
    # every physics step and reads only measured state/contact reports.
    # ``stop()`` tears down Isaac Sim 5.1's articulation tensor view.  Keep
    # the view alive between trials with pause; all state writes below still
    # happen before the next physics tick.
    timeline.pause()
    # Every Gate-B trial starts from the same free-body target pose.  A prior
    # finger-contact trial may have nudged the rigid body; carrying that pose
    # into the opposite side would make the two sides non-independent.
    baseline_position, baseline_quaternion = baseline_object
    robot.object.set_world_poses(
        baseline_position.reshape(1, 3), baseline_quaternion.reshape(1, 4)
    )
    robot.object.set_velocities(np.zeros((1, 6), dtype=np.float64))
    trial_start_object = baseline_position.copy()
    q = robot.full_positions()
    # Seed from the authored Galaxea home configuration.  The imported
    # vehicle can expose unwrapped wheel/revolute values after the initial
    # PhysX warmup; those are auxiliary DOFs and are kept neutral here so the
    # arm IK starts from the same bounded state as the reachability gate.
    for index, name in enumerate(robot.all_joint_names):
        if name in robot.scene_cfg.get("home_joint_positions", {}):
            q[index] = float(robot.scene_cfg["home_joint_positions"][name])
    for index in range(len(q)):
        if index not in set(int(item) for item in robot.action_indices):
            q[index] = 0.0
    open_targets = {
        "left": np.asarray(robot.scene_cfg["gripper_open"], dtype=np.float64),
        "right": np.asarray(robot.scene_cfg["gripper_open"], dtype=np.float64),
    }
    for hand in ("left", "right"):
        q[robot.gripper_indices[hand]] = open_targets[hand]
    # The reachability gate established vertical pregrasp stations above the
    # target.  Use those independent long-axis station centers for the live
    # contact test.  A side-wall offset would move the midpoint outside the
    # imported R1 workspace and does not represent the two-finger TCP.
    frame = station_frames[side]
    grasp = Pose(frame["grasp_position"].copy(), frame["quaternion"].copy())
    pre = Pose(frame["pregrasp_position"].copy(), frame["quaternion"].copy())
    q_pre = robot.solve_ik(
        side,
        pre.position,
        pre.quaternion_wxyz,
        seed=q,
        max_iterations=int(robot.scene_cfg.get("gate_b_ik_pregrasp_max_iterations", 20)),
        position_tolerance=float(robot.scene_cfg.get("gate_b_ik_position_tolerance_m", 0.025)),
    )
    # Descend through measured approach waypoints.  A single large IK jump
    # from the high pregrasp to the object centre lands in a poor local basin
    # on this imported articulation even though the endpoints are close to the
    # reachable workspace.  Each waypoint remains a real commanded pose and
    # is evaluated with the live PhysX state.
    q_grasp = q_pre.copy()
    approach_joint_waypoints: list[np.ndarray] = [q_pre.copy()]
    approach_segments = max(1, int(robot.scene_cfg.get("gate_b_approach_segments", 8)))
    for segment in range(1, approach_segments + 1):
        alpha = float(segment) / float(approach_segments)
        waypoint = pre.position + alpha * (grasp.position - pre.position)
        q_grasp = robot.solve_ik(
            side,
            waypoint,
            grasp.quaternion_wxyz,
            seed=q_grasp,
            max_iterations=int(robot.scene_cfg.get("gate_b_ik_grasp_max_iterations", 80)),
            position_tolerance=float(robot.scene_cfg.get("gate_b_ik_position_tolerance_m", 0.025)),
        )
        approach_joint_waypoints.append(q_grasp.copy())
    # Start physically at the solved pregrasp, then execute the approach with
    # full 16D actions.  Setting a grasp q directly would create an impulse
    # before the contact window and is forbidden for the reference runner.
    q_pre_open = q_pre.copy()
    q_pre_open[robot.gripper_indices["left"]] = open_targets["left"]
    q_pre_open[robot.gripper_indices["right"]] = open_targets["right"]
    q_grasp_open = q_grasp.copy()
    q_grasp_open[robot.gripper_indices["left"]] = open_targets["left"]
    q_grasp_open[robot.gripper_indices["right"]] = open_targets["right"]
    q_start = q_grasp_open.copy()
    q_end = q_start.copy()
    q_end[robot.gripper_indices[side]] = np.zeros(2, dtype=np.float64)

    initial_object = trial_start_object
    robot.robot.set_joint_positions(q_pre_open.reshape(1, -1))
    robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
    robot.apply_full_target(q_pre_open)
    timeline.play()
    segment_count = max(1, len(approach_joint_waypoints) - 1)
    steps_per_segment = max(1, int(np.ceil(float(max(1, approach_steps)) / segment_count)))
    open_waypoints = []
    for waypoint in approach_joint_waypoints:
        item = waypoint.copy()
        item[robot.gripper_indices["left"]] = open_targets["left"]
        item[robot.gripper_indices["right"]] = open_targets["right"]
        open_waypoints.append(item)
    for segment in range(1, len(open_waypoints)):
        start = open_waypoints[segment - 1]
        end = open_waypoints[segment]
        for step in range(steps_per_segment):
            alpha = min(1.0, float(step + 1) / steps_per_segment)
            approach_command = start + alpha * (end - start)
            robot.apply_full_target(approach_command)
            app.update()
    for _ in range(max(1, settle_steps)):
        robot.apply_full_target(q_start)
        app.update()

    contact_steps = 0
    pair_history: set[str] = set()
    max_object_drift = 0.0
    finite = True
    total_steps = max(1, close_steps) + max(0, hold_steps)
    for step in range(total_steps):
        alpha = min(1.0, float(step + 1) / max(1, close_steps))
        command = q_start + alpha * (q_end - q_start)
        robot.apply_full_target(command)
        app.update()
        sample = tracker.sample()
        side_contact = bool(sample.left if side == "left" else sample.right)
        contact_steps += int(side_contact)
        pair_history.update(f"{a}|{b}" for a, b in sample.pairs[side])
        object_position, _ = robot.object_pose()
        object_velocity = robot.object_velocity()
        finite = finite and bool(np.isfinite(object_position).all() and np.isfinite(object_velocity).all())
        max_object_drift = max(max_object_drift, float(np.linalg.norm(object_position - initial_object)))

    final_object, _ = robot.object_pose()
    velocity = robot.object_velocity()
    final_midpoint, _ = robot.gripper_midpoint_pose(side)
    final_fingers = {
        f"{side}_gripper_link1": robot.link_pose(f"{side}_gripper_link1")[0].tolist(),
        f"{side}_gripper_link2": robot.link_pose(f"{side}_gripper_link2")[0].tolist(),
    }
    timeline.pause()
    hold_required = max(3, min(hold_steps, 10))
    passed = bool(
        finite
        and contact_steps >= hold_required
        and max_object_drift <= float(robot.scene_cfg.get("gate_b_max_object_drift_m", 0.03))
    )
    return {
        "trial": int(trial_index),
        "side": side,
        "result": "PASS" if passed else "FAIL",
        "contact_steps": int(contact_steps),
        "approach_steps": int(max(1, approach_steps)),
        "contact_window_required_steps": int(hold_required),
        "contact_pairs": sorted(pair_history),
        "max_object_drift_m": float(max_object_drift),
        "final_object_position_m": _row(final_object).tolist(),
        "final_object_velocity": _row(velocity).tolist(),
        "final_gripper_midpoint_m": _row(final_midpoint).tolist(),
        "final_gripper_midpoint_error_m": float(np.linalg.norm(_row(final_midpoint) - grasp.position)),
        "final_finger_link_positions_m": final_fingers,
        "finite": finite,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate B independent single-gripper contact trials")
    parser.add_argument("--candidate", choices=sorted(ASSETS), default="sugar_box")
    parser.add_argument("--trials-per-side", type=int, default=10)
    parser.add_argument("--target-stand-height", type=float, default=0.42)
    parser.add_argument("--base-stand-off", type=float, default=0.45)
    parser.add_argument("--approach-steps", type=int, default=48)
    parser.add_argument("--close-steps", type=int, default=60)
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=24)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.trials_per_side <= 0 or args.approach_steps <= 0 or args.close_steps <= 0 or args.hold_steps < 0 or args.settle_steps < 0:
        raise ValueError("trial and step counts must be positive (hold may be zero)")
    asset = ASSETS[args.candidate]
    require_candidate(asset)
    geometry, static = _candidate_gate_a(args.candidate)
    report_path = args.report or ROOT / "reports" / f"gate_b_{args.candidate}_single_gripper_contact.json"
    scene_source = Path(load_scene_config(ROOT / "scene_config.json")["stage_path"])
    source_before = hashlib.sha256(scene_source.read_bytes()).hexdigest()
    report: dict[str, object] = {
        "report_version": "r1-gate-b-single-gripper-contact-v1",
        "gate": "B",
        "candidate": args.candidate,
        "asset": str(asset),
        "status": "INVALID",
        "dataset_episodes_written": 0,
        "dataset_generation_enabled": False,
        "source_usd_modified": False,
        "production_approved": False,
        "environment_variant": {
            "target_stand_height_m": float(args.target_stand_height),
            "meaning": "explicit temporary Session-Layer tabletop stand for workspace validation; production choice pending",
        },
        "gate_a_geometry_report": f"reports/candidate_{asset.stem}.json",
        "gate_a_static_report": f"reports/candidate_{asset.stem}_static_audit.json",
        "trials_required_per_side": int(args.trials_per_side),
        "execution_steps": {
            "approach": int(args.approach_steps),
            "close": int(args.close_steps),
            "hold": int(args.hold_steps),
            "settle": int(args.settle_steps),
        },
    }
    app = None
    try:
        cfg = dict(load_scene_config(ROOT / "scene_config.json"))
        cfg.update(
            {
                "target_asset_url": str(asset),
                "target_stand_height_m": float(args.target_stand_height),
                "fixed_base_stand_off_m": float(args.base_stand_off),
                "target_spawn_xy": None,
                "target_yaw_rad": 0.0,
                "runtime_base_lock": False,
                "runtime_joint_state_lock": False,
                "scripted_grasp_attachment": False,
                "physical_rollout": True,
                "use_live_initial_state": True,
                # Match the validated stand-height pregrasp mapping: native
                # PhysX Jacobians with the Galaxea-compatible larger step.
                "ik_jacobian_source": "native",
                # The imported articulation's native Jacobian does not yet
                # expose a stable orientation solution for the measured
                # contact frame.  Keep the diagnostic position-only while
                # the arm reaches the station; contact-frame orientation is
                # validated separately from the live pad normals.
                "ik_orientation_weight": 0.0,
                "ik_line_search": False,
                "ik_max_iterations": 20,
                # The mirrored station is close to the edge of the imported
                # arm's workspace.  Keep this as a diagnostic reachability
                # tolerance; a Gate-B PASS still requires measured finger
                # contact and the 30 mm payload-drift limit.
                "ik_accept_position_error": 0.06,
                "physical_ik_accept_position_error": 0.03,
                "physical_ik_step_limit": 0.2,
                "physical_ik_damping": 0.02,
                "gate_b_ik_pregrasp_max_iterations": 40,
                "gate_b_ik_grasp_max_iterations": 80,
                "gate_b_approach_segments": 4,
                "gate_b_ik_position_tolerance_m": 0.06,
            }
        )
        prepare_isaac_environment()
        from isaacsim import SimulationApp

        app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
        _stage, timeline, robot, _target_path, setup = prepare_fixed_target_scene(app, cfg)
        tracker = setup["contact_tracker"]
        baseline_object = tuple(value.copy() for value in robot.object_pose())
        # Build the runtime stations from the live target bbox.  A few warmup
        # ticks after PhysX initialization can settle the replacement asset by
        # millimetres; using the live bound keeps the IK target aligned with
        # the object that the contact sensor will actually observe.
        from r1_bimanual_dataset.tools.physical_reference_probe import _world_bbox

        # Build the runtime stations from the settled target bbox.  The
        # station centers are separated along the measured long axis; the
        # closing axes come from the Gate-A collision audit and are recorded
        # with the trial for later pad-contact validation.
        bbox_min, bbox_max = _world_bbox(_stage, _target_path)
        bbox_min = np.asarray(bbox_min, dtype=np.float64)
        bbox_max = np.asarray(bbox_max, dtype=np.float64)
        dimensions = bbox_max - bbox_min
        object_center = (bbox_min + bbox_max) / 2.0
        long_axis_index = int(np.argmax(dimensions[:2]))
        long_axis = np.eye(3, dtype=np.float64)[long_axis_index]
        # Use a conservative near-centre pair while the imported arm's
        # workspace is being calibrated.  The spacing remains above the
        # Gate-A minimum station footprint plus safety margin and keeps the
        # two fingers on distinct long-axis stations.
        station_spacing = max(
            min(float(dimensions[long_axis_index] * 0.5), 0.050),
            float(geometry.get("minimum_station_spacing_m", 0.0)) + 0.008,
        )
        approach_distance = float(cfg.get("gate_b_pregrasp_distance_m", 0.08))
        pregrasp_height = float(cfg.get("gate_b_pregrasp_height_m", 0.14))
        # The midpoint of the two physical finger links is the centre of the
        # grasp station.  Do not offset that midpoint by half the object
        # thickness: the prismatic pads close around the object from this
        # centre, and adding a thickness offset moves one pad through the
        # payload before the close phase begins.
        grasp_offset = 0.0
        station_frames: dict[str, dict[str, np.ndarray]] = {}
        # The authored R1 arm pair reaches the near-centre stations with the
        # left arm on +long-axis and the right arm on -long-axis.  Keep these
        # independent stations separated; this mapping is measured from the
        # live link layout rather than inferred from the list order.
        for index, side in enumerate(("left", "right")):
            sign = 1.0 if index == 0 else -1.0
            station = object_center + sign * long_axis * station_spacing / 2.0
            closing_axis = np.asarray(geometry["grippers"][side]["closing_axis"], dtype=np.float64)
            closing_axis /= max(float(np.linalg.norm(closing_axis)), 1e-9)
            approach_normal = np.asarray(geometry["grippers"][side]["approach_normal"], dtype=np.float64)
            approach_normal /= max(float(np.linalg.norm(approach_normal)), 1e-9)
            grasp_position = station - approach_normal * grasp_offset
            # The imported R1 reaches this tabletop only from its validated
            # high pregrasp envelope.  Keep the measured approach normal in
            # the frame record, while descending from a collision-free high
            # point through segmented full-action waypoints.
            pregrasp_position = grasp_position + np.asarray([0.0, 0.0, pregrasp_height])
            station_frames[side] = {
                "station_center": station,
                "grasp_position": grasp_position,
                "pregrasp_position": pregrasp_position,
                "quaternion": robot.eef_tip_pose(side)[1].copy(),
                "closing_axis": closing_axis,
                "approach_normal": approach_normal,
            }
        report["setup"] = {
            key: value for key, value in setup.items() if key != "contact_tracker"
        }
        report["contact_tracker_query_paths"] = {
            side: list(getattr(tracker, "_finger_paths", {}).get(side, []))
            for side in ("left", "right")
        }
        report["nominal"] = {
            "object_center_m": object_center.tolist(),
            "object_dimensions_m": dimensions.tolist(),
            "long_axis_index": long_axis_index,
            "station_spacing_m": station_spacing,
            "approach_distance_m": approach_distance,
            "grasp_offset_m": grasp_offset,
            "pregrasp_height_m": pregrasp_height,
            "station_center_m": {
                side: station_frames[side]["station_center"].tolist() for side in ("left", "right")
            },
            "left_grasp_m": station_frames["left"]["grasp_position"].tolist(),
            "right_grasp_m": station_frames["right"]["grasp_position"].tolist(),
            "left_pregrasp_m": station_frames["left"]["pregrasp_position"].tolist(),
            "right_pregrasp_m": station_frames["right"]["pregrasp_position"].tolist(),
            "closing_axis_world": {
                side: station_frames[side]["closing_axis"].tolist() for side in ("left", "right")
            },
            "approach_normal_world": {
                side: station_frames[side]["approach_normal"].tolist() for side in ("left", "right")
            },
            "frame_source": "runtime target bbox + measured long-axis stations; Gate-A collision geometry",
            "baseline_object_pose_m": {
                "position": baseline_object[0].tolist(),
                "quaternion_wxyz": baseline_object[1].tolist(),
            },
        }
        trials: list[dict[str, object]] = []
        for side in ("left", "right"):
            for trial_index in range(args.trials_per_side):
                print(f"Gate B: {args.candidate} {side} trial {trial_index + 1}/{args.trials_per_side}", flush=True)
                trials.append(
                    _trial(
                        app,
                        timeline,
                        robot,
                        tracker,
                        station_frames,
                        side,
                        baseline_object,
                        approach_steps=args.approach_steps,
                        close_steps=args.close_steps,
                        hold_steps=args.hold_steps,
                        settle_steps=args.settle_steps,
                        trial_index=trial_index,
                    )
                )
        report["trials"] = trials
        by_side = {
            side: [item for item in trials if item["side"] == side]
            for side in ("left", "right")
        }
        report["successes"] = {side: sum(item["result"] == "PASS" for item in values) for side, values in by_side.items()}
        report["status"] = (
            "PASS"
            if all(report["successes"][side] >= args.trials_per_side for side in ("left", "right"))
            else "FAIL"
        )
        report["gate_b_pass"] = report["status"] == "PASS"
        return_code = 0 if report["status"] == "PASS" else 2
    except Exception as exc:
        report["status"] = "INVALID"
        report["reason"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        return_code = 2
    finally:
        report["source_usd_sha256_before"] = source_before
        report["source_usd_sha256_after"] = hashlib.sha256(scene_source.read_bytes()).hexdigest()
        report["source_usd_modified"] = report["source_usd_sha256_before"] != report["source_usd_sha256_after"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print("GATE_B_SINGLE_GRIPPER=" + json.dumps(report, ensure_ascii=False), flush=True)
        if app is not None:
            app.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
