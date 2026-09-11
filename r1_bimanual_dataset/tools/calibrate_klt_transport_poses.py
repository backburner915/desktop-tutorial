"""Paused FK/IK gate for the physical KLT lift and transport waypoints.

This tool is intentionally before the dynamic transport trial.  It starts at
the physically verified bilateral-contact posture, checks a common vertical
lift and a common lateral carry displacement, and reports position plus pinch
axis alignment at every waypoint.  It never closes grippers or moves KLT.
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


def _gap_axis(robot, side: str) -> np.ndarray:
    first, _ = robot.link_pose(f"{side}_gripper_link1")
    second, _ = robot.link_pose(f"{side}_gripper_link2")
    delta = second - first
    return delta / max(float(np.linalg.norm(delta)), 1e-9)


def _waypoint_report(robot, targets: dict[str, np.ndarray]) -> dict[str, object]:
    result: dict[str, object] = {}
    for side in ("left", "right"):
        midpoint, _ = robot.gripper_midpoint_pose(side)
        normal = np.asarray([-1.0, 0.0, 0.0] if side == "left" else [1.0, 0.0, 0.0])
        gap = _gap_axis(robot, side)
        result[side] = {
            "target_midpoint_m": targets[side].tolist(),
            "midpoint_m": midpoint.tolist(),
            "position_error_m": float(np.linalg.norm(midpoint - targets[side])),
            "gap_axis_world": gap.tolist(),
            "side_normal_alignment": float(abs(np.dot(gap, normal))),
            "arm_joint_target": robot.full_positions()[robot.arm_indices[side]].tolist(),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Selected KLT lift/carry paused IK gate")
    parser.add_argument("--lift-m", type=float, default=0.14)
    parser.add_argument("--carry-x-m", type=float, default=0.20)
    parser.add_argument(
        "--grasp-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_grasp_pose_report.json"
    )
    parser.add_argument(
        "--axis-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_gripper_axis_report.json"
    )
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_transport_pose_report.json")
    args = parser.parse_args()
    grasp = json.loads(args.grasp_report.read_text(encoding="utf-8"))
    axis = json.loads(args.axis_report.read_text(encoding="utf-8"))
    if grasp.get("result") != "PASS":
        raise RuntimeError("grasp-pose report is not PASS")

    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        _stage, timeline, robot, _target_path, setup = prepare_fixed_target_scene(app, _selected_cfg())
        timeline.pause()
        q = robot.full_positions()
        grasp_targets: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            q[robot.arm_indices[side]] = np.asarray(grasp["sides"][side]["arm_joint_target"], dtype=np.float64)
            q[int(robot.arm_indices[side][-1])] = float(axis["sides"][side]["wrist_joint6"])
            q[robot.gripper_indices[side]] = 0.05
            grasp_targets[side] = np.asarray(grasp["sides"][side]["target_midpoint_m"], dtype=np.float64)
        robot.robot.set_joint_positions(q.reshape(1, -1))
        robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
        app.update()
        lift_targets = {side: point + np.asarray([0.0, 0.0, args.lift_m]) for side, point in grasp_targets.items()}
        for side in ("left", "right"):
            q = robot.solve_ik(side, lift_targets[side], robot.eef_tip_pose(side)[1], seed=q, max_iterations=70, position_tolerance=0.015)
        robot.robot.set_joint_positions(q.reshape(1, -1))
        app.update()
        lift_report = _waypoint_report(robot, lift_targets)

        carry_targets = {side: point + np.asarray([args.carry_x_m, 0.0, 0.0]) for side, point in lift_targets.items()}
        for side in ("left", "right"):
            q = robot.solve_ik(side, carry_targets[side], robot.eef_tip_pose(side)[1], seed=q, max_iterations=70, position_tolerance=0.015)
        robot.robot.set_joint_positions(q.reshape(1, -1))
        app.update()
        carry_report = _waypoint_report(robot, carry_targets)
        report = {
            "report_version": "r1-small-klt-transport-pose-v1",
            "meaning": "paused lift/carry IK gate only; not contact, transport, or data",
            "source_usd_modified": False,
            "table_moved": False,
            "lift_vector_m": [0.0, 0.0, args.lift_m],
            "carry_vector_m": [args.carry_x_m, 0.0, 0.0],
            "target": setup["target"],
            "lift": lift_report,
            "carry": carry_report,
            "result": "PASS"
            if all(
                float(phase[side]["position_error_m"]) <= 0.025
                and float(phase[side]["side_normal_alignment"]) >= 0.82
                for phase in (lift_report, carry_report)
                for side in ("left", "right")
            )
            else "FAIL",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMALL_KLT_TRANSPORT_POSE=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["result"] == "PASS" else 2
    except Exception as exc:
        traceback.print_exc()
        print(f"SMALL_KLT_TRANSPORT_POSE=FAIL reason={exc}", flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
