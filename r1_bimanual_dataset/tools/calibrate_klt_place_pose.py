"""Paused IK gate for lowering the carried KLT onto its work platform."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Paused KLT carry-to-place IK gate")
    parser.add_argument(
        "--transport-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_transport_pose_report.json"
    )
    parser.add_argument(
        "--axis-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_transport_axis_report.json"
    )
    parser.add_argument("--lower-m", type=float, default=0.14)
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_place_pose_report.json")
    args = parser.parse_args()
    transport = json.loads(args.transport_report.read_text(encoding="utf-8"))
    axes = json.loads(args.axis_report.read_text(encoding="utf-8"))

    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        _stage, timeline, robot, _target_path, setup = prepare_fixed_target_scene(app, _selected_cfg())
        timeline.pause()
        q = robot.full_positions()
        targets: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            q[robot.arm_indices[side]] = np.asarray(transport["carry"][side]["arm_joint_target"], dtype=np.float64)
            q[int(robot.arm_indices[side][-1])] = float(axes["phases"]["carry"][side]["wrist_joint6"])
            q[robot.gripper_indices[side]] = 0.05
            targets[side] = np.asarray(transport["carry"][side]["target_midpoint_m"], dtype=np.float64) - np.asarray([0.0, 0.0, args.lower_m])
        robot.robot.set_joint_positions(q.reshape(1, -1))
        robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
        app.update()
        for side in ("left", "right"):
            q = robot.solve_ik(side, targets[side], robot.eef_tip_pose(side)[1], seed=q, max_iterations=70, position_tolerance=0.015)
        robot.robot.set_joint_positions(q.reshape(1, -1))
        app.update()
        output: dict[str, object] = {}
        for side in ("left", "right"):
            midpoint, _ = robot.gripper_midpoint_pose(side)
            first, _ = robot.link_pose(f"{side}_gripper_link1")
            second, _ = robot.link_pose(f"{side}_gripper_link2")
            gap = second - first
            gap /= max(float(np.linalg.norm(gap)), 1e-9)
            normal = np.asarray([-1.0, 0.0, 0.0] if side == "left" else [1.0, 0.0, 0.0])
            output[side] = {
                "target_midpoint_m": targets[side].tolist(),
                "midpoint_m": midpoint.tolist(),
                "position_error_m": float(np.linalg.norm(midpoint - targets[side])),
                "gap_axis_world": gap.tolist(),
                "side_normal_alignment": float(abs(np.dot(gap, normal))),
                "arm_joint_target": q[robot.arm_indices[side]].tolist(),
            }
        report = {
            "report_version": "r1-small-klt-place-pose-v1",
            "meaning": "paused carry-to-place IK gate only; not contact, place, or data",
            "source_usd_modified": False,
            "table_moved": False,
            "lower_vector_m": [0.0, 0.0, -args.lower_m],
            "target": setup["target"],
            "sides": output,
            "result": "PASS" if all(float(output[s]["position_error_m"]) <= 0.025 for s in ("left", "right")) else "FAIL",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMALL_KLT_PLACE_POSE=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["result"] == "PASS" else 2
    except Exception as exc:
        traceback.print_exc()
        print(f"SMALL_KLT_PLACE_POSE=FAIL reason={exc}", flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
