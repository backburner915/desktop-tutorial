"""Search the redundant wrist joint for a true KLT side-wall pinch direction.

The previously calibrated midpoint positions remain fixed.  Only joint 6 of
each arm is sampled while Kit is paused, so this tool cannot contact, lift,
or move the target.  It prevents a position-only IK pose from being mistaken
for a valid two-finger grasp.
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
from r1_bimanual_dataset.core.robot_interface import quat_rotate
from r1_bimanual_dataset.tools.calibrate_klt_grasp_pose import _selected_cfg
from r1_bimanual_dataset.core.physical_scene import prepare_fixed_target_scene


def _gap_axis(robot, side: str) -> np.ndarray:
    first, _ = robot.link_pose(f"{side}_gripper_link1")
    second, _ = robot.link_pose(f"{side}_gripper_link2")
    delta = second - first
    return delta / max(float(np.linalg.norm(delta)), 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paused wrist-axis search for the selected physical KLT")
    parser.add_argument(
        "--grasp-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_grasp_pose_report.json"
    )
    parser.add_argument("--samples", type=int, default=121)
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_gripper_axis_report.json")
    args = parser.parse_args()
    if args.samples < 21:
        raise ValueError("--samples must be >= 21")
    grasp = json.loads(args.grasp_report.read_text(encoding="utf-8"))
    if grasp.get("result") != "PASS":
        raise RuntimeError("grasp-pose report is not PASS")

    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        _stage, timeline, robot, _target_path, setup = prepare_fixed_target_scene(app, _selected_cfg())
        timeline.pause()
        q = robot.full_positions()
        for side in ("left", "right"):
            q[robot.arm_indices[side]] = np.asarray(grasp["sides"][side]["arm_joint_target"], dtype=np.float64)
            q[robot.gripper_indices[side]] = 0.05
        robot.robot.set_joint_positions(q.reshape(1, -1))
        robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
        app.update()
        lower, upper = robot.limits()
        results: dict[str, object] = {}
        for side in ("left", "right"):
            arm_ids = robot.arm_indices[side]
            wrist_id = int(arm_ids[-1])
            target = np.asarray(grasp["sides"][side]["target_midpoint_m"], dtype=np.float64)
            normal = np.asarray([-1.0, 0.0, 0.0] if side == "left" else [1.0, 0.0, 0.0])
            best: dict[str, object] | None = None
            for wrist in np.linspace(lower[wrist_id], upper[wrist_id], args.samples):
                candidate = q.copy()
                candidate[wrist_id] = wrist
                robot.robot.set_joint_positions(candidate.reshape(1, -1))
                app.update()
                midpoint, orientation = robot.gripper_midpoint_pose(side)
                gap = _gap_axis(robot, side)
                value = {
                    "wrist_joint6": float(wrist),
                    "midpoint_m": midpoint.tolist(),
                    "position_error_m": float(np.linalg.norm(midpoint - target)),
                    "gap_axis_world": gap.tolist(),
                    "side_normal_alignment": float(abs(np.dot(gap, normal))),
                    "eef_quaternion_wxyz": orientation.tolist(),
                }
                if best is None or value["side_normal_alignment"] > best["side_normal_alignment"]:
                    best = value
            assert best is not None
            results[side] = best
        report = {
            "report_version": "r1-small-klt-gripper-axis-v1",
            "meaning": "paused redundant-wrist orientation search only; not contact, grasp, lift, or data",
            "source_usd_modified": False,
            "table_moved": False,
            "target": setup["target"],
            "sides": results,
            "result": "PASS"
            if all(
                float(results[side]["side_normal_alignment"]) >= 0.90
                and float(results[side]["position_error_m"]) <= 0.030
                for side in ("left", "right")
            )
            else "FAIL",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMALL_KLT_GRIPPER_AXIS=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["result"] == "PASS" else 2
    except Exception as exc:
        traceback.print_exc()
        print(f"SMALL_KLT_GRIPPER_AXIS=FAIL reason={exc}", flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
