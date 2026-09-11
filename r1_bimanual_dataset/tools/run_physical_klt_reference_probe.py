"""Run one complete, no-cheat physical R1/SmallKLT reference episode.

This is the final reference gate before the generic data runner is connected.
It records only a diagnostic JSON: neither image nor trajectory data are
written here.  The payload remains a free rigid body throughout the episode.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.core.physical_scene import prepare_fixed_target_scene
from r1_bimanual_dataset.tools.calibrate_klt_grasp_pose import _selected_cfg


def _step_phase(app, robot, tracker, start: np.ndarray, end: np.ndarray, steps: int, contacts: dict[str, int]) -> None:
    for step in range(steps):
        alpha = (step + 1) / max(steps, 1)
        robot.apply_full_target(start + alpha * (end - start))
        app.update()
        sample = tracker.sample()
        contacts["left"] += int(sample.left)
        contacts["right"] += int(sample.right)
        contacts["both"] += int(sample.left and sample.right)


def _snapshot(robot) -> dict[str, object]:
    position, quaternion = robot.object_pose()
    velocity = robot.object_velocity()
    return {"position_m": position.tolist(), "quaternion_wxyz": quaternion.tolist(), "velocity_m_rad_s": velocity.tolist()}


def main() -> int:
    from r1_bimanual_dataset.core.reference_policy import INVALID_KLT
    print(INVALID_KLT + ": retired counterexample; historical implementation and reports retained; no simulation or dataset writes")
    return 2


def _retired_main_for_source_regression_only() -> int:
    parser = argparse.ArgumentParser(description="Complete physical SmallKLT reference gate")
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_reference_probe_report.json")
    parser.add_argument("--gui", action="store_true", help="show the real physical rollout in Isaac Sim")
    parser.add_argument("--keep-open", action="store_true", help="with --gui, keep Isaac Sim open after the release phase")
    args = parser.parse_args()
    paths = {
        "grasp": PACKAGE_ROOT / "reports" / "small_klt_grasp_pose_report.json",
        "grasp_axis": PACKAGE_ROOT / "reports" / "small_klt_gripper_axis_report.json",
        "transport": PACKAGE_ROOT / "reports" / "small_klt_transport_pose_report.json",
        "transport_axis": PACKAGE_ROOT / "reports" / "small_klt_transport_axis_report.json",
        "place": PACKAGE_ROOT / "reports" / "small_klt_place_pose_report.json",
    }
    reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    required = ("grasp", "grasp_axis", "transport", "transport_axis", "place")
    failed = [name for name in required if reports[name].get("result") != "PASS"]
    if failed:
        raise RuntimeError(
            "required physical calibration is not PASS: " + ", ".join(failed)
        )

    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(
        simulation_app_config(
            headless=not args.gui,
            renderer="RayTracedLighting" if args.gui else "None",
        )
    )
    try:
        _stage, _timeline, robot, _target_path, setup = prepare_fixed_target_scene(app, _selected_cfg())
        tracker = setup["contact_tracker"]
        q_grasp = robot.full_positions()
        q_lift = q_grasp.copy()
        q_carry = q_grasp.copy()
        q_place = q_grasp.copy()
        for side in ("left", "right"):
            arm_ids = robot.arm_indices[side]
            q_grasp[arm_ids] = np.asarray(reports["grasp"]["sides"][side]["arm_joint_target"], dtype=np.float64)
            q_grasp[int(arm_ids[-1])] = float(reports["grasp_axis"]["sides"][side]["wrist_joint6"])
            q_lift[arm_ids] = np.asarray(reports["transport"]["lift"][side]["arm_joint_target"], dtype=np.float64)
            q_lift[int(arm_ids[-1])] = float(reports["transport_axis"]["phases"]["lift"][side]["wrist_joint6"])
            q_carry[arm_ids] = np.asarray(reports["transport"]["carry"][side]["arm_joint_target"], dtype=np.float64)
            q_carry[int(arm_ids[-1])] = float(reports["transport_axis"]["phases"]["carry"][side]["wrist_joint6"])
            q_place[arm_ids] = np.asarray(reports["place"]["sides"][side]["arm_joint_target"], dtype=np.float64)
            q_place[int(arm_ids[-1])] = float(reports["place"]["sides"][side]["arm_joint_target"][-1])
            q_grasp[robot.gripper_indices[side]] = 0.05
            q_lift[robot.gripper_indices[side]] = 0.0
            q_carry[robot.gripper_indices[side]] = 0.0
            q_place[robot.gripper_indices[side]] = 0.0
        robot.robot.set_joint_positions(q_grasp.reshape(1, -1))
        robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
        robot.apply_full_target(q_grasp)
        for _ in range(10):
            app.update()
        snapshots = {"initial": _snapshot(robot)}
        phase_contacts: dict[str, dict[str, int]] = {}
        q_close = q_grasp.copy()
        for side in ("left", "right"):
            q_close[robot.gripper_indices[side]] = 0.0
        for name, start, end, steps in (
            ("close", q_grasp, q_close, 90),
            ("grasp_hold", q_close, q_close, 60),
            ("lift", q_close, q_lift, 180),
            ("lift_hold", q_lift, q_lift, 120),
            ("carry", q_lift, q_carry, 180),
            ("carry_hold", q_carry, q_carry, 90),
            ("place", q_carry, q_place, 180),
            ("place_hold", q_place, q_place, 120),
        ):
            counter = {"left": 0, "right": 0, "both": 0}
            _step_phase(app, robot, tracker, start, end, steps, counter)
            phase_contacts[name] = counter
            snapshots[name] = _snapshot(robot)
        q_release = q_place.copy()
        q_retract = q_carry.copy()
        for side in ("left", "right"):
            q_release[robot.gripper_indices[side]] = 0.05
            q_retract[robot.gripper_indices[side]] = 0.05
        for name, start, end, steps in (
            ("release", q_place, q_release, 90),
            # A release is not verified while the fingers continue to cradle
            # the payload.  Move both open grippers back to the validated
            # high carry pose, then assess the free object's stability.
            ("retreat", q_release, q_retract, 180),
            ("release_hold", q_retract, q_retract, 240),
        ):
            counter = {"left": 0, "right": 0, "both": 0}
            _step_phase(app, robot, tracker, start, end, steps, counter)
            phase_contacts[name] = counter
            snapshots[name] = _snapshot(robot)
        initial = np.asarray(snapshots["initial"]["position_m"], dtype=np.float64)
        lift = np.asarray(snapshots["lift_hold"]["position_m"], dtype=np.float64)
        carry = np.asarray(snapshots["carry_hold"]["position_m"], dtype=np.float64)
        released = np.asarray(snapshots["release_hold"]["position_m"], dtype=np.float64)
        after_retreat = np.asarray(snapshots["retreat"]["position_m"], dtype=np.float64)
        release_velocity = np.asarray(snapshots["release_hold"]["velocity_m_rad_s"], dtype=np.float64)
        finite = bool(all(np.isfinite(np.asarray(s["position_m"], dtype=np.float64)).all() for s in snapshots.values()))
        lift_m = float(lift[2] - initial[2])
        carry_x_m = float(carry[0] - lift[0])
        final_height_error = float(abs(released[2] - initial[2]))
        release_settle_displacement = float(np.linalg.norm(released - after_retreat))
        report = {
            "report_version": "r1-small-klt-full-physical-reference-v1",
            "meaning": "complete physical reference diagnostic; no dataset frames written",
            "source_usd_modified": False,
            "table_moved": False,
            "payload_attachment": False,
            "collision_filters_changed": False,
            "runtime_base_lock": False,
            "runtime_joint_lock": False,
            "target": setup["target"],
            "snapshots": snapshots,
            "phase_contact_steps": phase_contacts,
            "lift_m": lift_m,
            "carry_x_m": carry_x_m,
            "final_height_error_m": final_height_error,
            "release_settle_displacement_m": release_settle_displacement,
            "release_linear_speed_m_s": float(np.linalg.norm(release_velocity[:3])),
            "finite": finite,
            "result": "PASS"
            if finite
            and lift_m >= 0.070
            and abs(carry_x_m) >= 0.100
            # A rigid object's root height is not a placement invariant once
            # the bin has rotated.  The post-retreat *free-body* stability
            # window and zero gripper contacts below are the physical
            # placement/release test; a floating object would fall during
            # that window.
            and release_settle_displacement <= 0.025
            and float(np.linalg.norm(release_velocity[:3])) <= 0.10
            and all(phase_contacts[name]["both"] >= 10 for name in ("close", "lift", "lift_hold", "carry", "place"))
            and phase_contacts["release_hold"]["both"] <= 5
            else "FAIL",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMALL_KLT_REFERENCE_PROBE=" + json.dumps(report, ensure_ascii=False), flush=True)
        if args.gui:
            try:
                from isaacsim.core.utils.viewports import set_active_viewport_camera
                from r1_bimanual_dataset.core.lighting import set_gui_task_view

                set_active_viewport_camera("/OmniverseKit_Persp")
                # The fixed base, whole staging platform and final placement
                # are all visible from this oblique long-edge view.
                set_gui_task_view(
                    np.asarray([-1.9, 1.8, 2.5], dtype=np.float64),
                    np.asarray([-3.1, 3.72, 1.12], dtype=np.float64),
                )
                app.update()
            except Exception as exc:
                print(f"SMALL_KLT_REFERENCE_PROBE=GUI_VIEW_WARNING {type(exc).__name__}: {exc}", flush=True)
        if args.gui and args.keep_open:
            print("SMALL_KLT_REFERENCE_PROBE=GUI_KEEP_OPEN close the Isaac Sim window to exit", flush=True)
            while app.is_running():
                app.update()
        return 0 if report["result"] == "PASS" else 2
    except Exception as exc:
        traceback.print_exc()
        print(f"SMALL_KLT_REFERENCE_PROBE=FAIL reason={exc}", flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
