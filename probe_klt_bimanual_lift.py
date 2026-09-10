"""Physical bilateral close, lift, and hold gate for the selected KLT.

No attachment, root/joint state lock, collision filtering, teleport during the
rollout, carry, place, or data recording is used.  This is the first dynamic
gate after verified bilateral contact.
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


def _accumulate(tracker, counters: dict[str, int]) -> None:
    sample = tracker.sample()
    for side in ("left", "right"):
        counters[side] += int(getattr(sample, side))
    counters["both"] += int(sample.left and sample.right)


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical no-attachment bilateral SmallKLT lift gate")
    parser.add_argument(
        "--grasp-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_grasp_pose_report.json"
    )
    parser.add_argument(
        "--axis-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_gripper_axis_report.json"
    )
    parser.add_argument(
        "--transport-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_transport_pose_report.json"
    )
    parser.add_argument(
        "--transport-axis-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_transport_axis_report.json"
    )
    parser.add_argument("--close-steps", type=int, default=90)
    parser.add_argument("--pre-lift-hold-steps", type=int, default=60)
    parser.add_argument("--lift-steps", type=int, default=180)
    parser.add_argument("--hold-steps", type=int, default=120)
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_bimanual_lift_report.json")
    args = parser.parse_args()
    grasp = json.loads(args.grasp_report.read_text(encoding="utf-8"))
    grasp_axis = json.loads(args.axis_report.read_text(encoding="utf-8"))
    transport = json.loads(args.transport_report.read_text(encoding="utf-8"))
    transport_axis = json.loads(args.transport_axis_report.read_text(encoding="utf-8"))

    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        _stage, _timeline, robot, _target_path, setup = prepare_fixed_target_scene(app, _selected_cfg())
        tracker = setup["contact_tracker"]
        grasp_q = robot.full_positions()
        lift_q = grasp_q.copy()
        for side in ("left", "right"):
            grasp_q[robot.arm_indices[side]] = np.asarray(grasp["sides"][side]["arm_joint_target"], dtype=np.float64)
            grasp_q[int(robot.arm_indices[side][-1])] = float(grasp_axis["sides"][side]["wrist_joint6"])
            grasp_q[robot.gripper_indices[side]] = 0.05
            lift_q[robot.arm_indices[side]] = np.asarray(transport["lift"][side]["arm_joint_target"], dtype=np.float64)
            lift_q[int(robot.arm_indices[side][-1])] = float(transport_axis["phases"]["lift"][side]["wrist_joint6"])
            lift_q[robot.gripper_indices[side]] = 0.0
        robot.robot.set_joint_positions(grasp_q.reshape(1, -1))
        robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
        robot.apply_full_target(grasp_q)
        for _ in range(10):
            app.update()
        initial_position, initial_quaternion = robot.object_pose()
        contacts = {"left": 0, "right": 0, "both": 0}
        close_q = grasp_q.copy()
        for side in ("left", "right"):
            close_q[robot.gripper_indices[side]] = 0.0
        for step in range(args.close_steps):
            alpha = (step + 1) / args.close_steps
            robot.apply_full_target(grasp_q + alpha * (close_q - grasp_q))
            app.update()
            _accumulate(tracker, contacts)
        for _ in range(args.pre_lift_hold_steps):
            robot.apply_full_target(close_q)
            app.update()
            _accumulate(tracker, contacts)
        before_lift, _ = robot.object_pose()
        lift_contacts = {"left": 0, "right": 0, "both": 0}
        for step in range(args.lift_steps):
            alpha = (step + 1) / args.lift_steps
            robot.apply_full_target(close_q + alpha * (lift_q - close_q))
            app.update()
            _accumulate(tracker, lift_contacts)
        for _ in range(args.hold_steps):
            robot.apply_full_target(lift_q)
            app.update()
            _accumulate(tracker, lift_contacts)
        final_position, final_quaternion = robot.object_pose()
        velocity = robot.object_velocity()
        lift_from_close = float(final_position[2] - before_lift[2])
        lift_from_initial = float(final_position[2] - initial_position[2])
        finite = bool(np.isfinite(final_position).all() and np.isfinite(velocity).all())
        report = {
            "report_version": "r1-small-klt-bimanual-lift-v1",
            "meaning": "physical close/lift/hold only; no attachment, carry, place, or data",
            "source_usd_modified": False,
            "table_moved": False,
            "payload_attachment": False,
            "collision_filters_changed": False,
            "runtime_joint_lock": False,
            "target": setup["target"],
            "initial_object_position_m": initial_position.tolist(),
            "pre_lift_object_position_m": before_lift.tolist(),
            "final_object_position_m": final_position.tolist(),
            "final_object_quaternion_wxyz": final_quaternion.tolist(),
            "final_object_velocity_m_rad_s": velocity.tolist(),
            "lift_from_close_m": lift_from_close,
            "lift_from_initial_m": lift_from_initial,
            "close_contact_steps": contacts,
            "lift_hold_contact_steps": lift_contacts,
            "finite": finite,
            "result": "PASS"
            if finite
            and lift_from_close >= 0.070
            and lift_contacts["left"] >= 15
            and lift_contacts["right"] >= 15
            and lift_contacts["both"] >= 10
            and float(np.linalg.norm(velocity[:3])) <= 0.20
            else "FAIL",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMALL_KLT_BIMANUAL_LIFT=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["result"] == "PASS" else 2
    except Exception as exc:
        traceback.print_exc()
        print(f"SMALL_KLT_BIMANUAL_LIFT=FAIL reason={exc}", flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
