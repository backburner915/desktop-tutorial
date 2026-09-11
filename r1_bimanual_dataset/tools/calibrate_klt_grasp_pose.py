"""Refine the selected SmallKLT two-arm pose before any contact trial.

This tool stays paused.  It proves that the two saved workspace seeds can be
refined to the selected exterior-side midpoint targets and records the real
finger opening direction.  It does not close a gripper or step physics.
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

from r1_bimanual_dataset.config import load_scene_config
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.core.physical_scene import prepare_fixed_target_scene
from r1_bimanual_dataset.core.scene_overlay import SMALL_KLT_PHYSICS_PATH
from r1_bimanual_dataset.core.robot_interface import IKError


def _selected_cfg() -> dict[str, object]:
    cfg = dict(load_scene_config(ROOT / "scene_config.json"))
    cfg.update(
        {
            "target_asset_url": SMALL_KLT_PHYSICS_PATH,
            "target_uniform_scale": 1.5,
            "target_spawn_xy": [-3.20, 3.725],
            "target_yaw_rad": float(np.pi / 2.0),
            "table_top": "/World/TaskSetup/Fixtures/StorageRack/Top",
            "target_stand_height_m": 0.539,
            # One static work platform covers the verified pickup and the
            # 20 cm placement point.  It rests on the unchanged source table
            # and is never moved during an episode.
            "target_stand_dimensions_m": [0.80, 0.298, 0.539],
            "target_stand_center_xy": [-3.10, 3.725],
            "fixed_base_side": "near",
            "fixed_base_stand_off_m": 0.58,
            "runtime_base_lock": False,
            "runtime_joint_state_lock": False,
            "scripted_grasp_attachment": False,
            "physical_rollout": True,
            "ik_jacobian_source": "finite_difference",
            "ik_line_search": True,
            "ik_step_limit": 0.05,
            "ik_damping": 0.03,
            "ik_accept_position_error": 0.02,
        }
    )
    return cfg


def _home(robot, cfg: dict[str, object]) -> np.ndarray:
    q = robot.full_positions()
    for index, name in enumerate(robot.all_joint_names):
        if name in cfg["home_joint_positions"]:
            q[index] = float(cfg["home_joint_positions"][name])
    for side in ("left", "right"):
        q[robot.gripper_indices[side]] = 0.05
    return q


def _gap_axis(robot, side: str) -> np.ndarray:
    a, _ = robot.link_pose(f"{side}_gripper_link1")
    b, _ = robot.link_pose(f"{side}_gripper_link2")
    direction = b - a
    return direction / max(float(np.linalg.norm(direction)), 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine two R1 gripper poses for the selected physical KLT")
    parser.add_argument(
        "--workspace-report",
        type=Path,
        default=PACKAGE_ROOT / "reports" / "small_klt_bimanual_workspace_final_report.json",
    )
    parser.add_argument(
        "--report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_grasp_pose_report.json"
    )
    args = parser.parse_args()
    workspace = json.loads(args.workspace_report.read_text(encoding="utf-8"))
    if workspace.get("result") != "PASS":
        raise RuntimeError("workspace report is not PASS; refusing to refine an unvalidated layout")

    cfg = _selected_cfg()
    prepare_isaac_environment()
    from isaacsim import SimulationApp
    from r1_bimanual_dataset.tools.physical_reference_probe import _world_bbox

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        stage, timeline, robot, target_path, setup = prepare_fixed_target_scene(app, cfg)
        timeline.pause()
        q = _home(robot, cfg)
        for side in ("left", "right"):
            q[robot.arm_indices[side]] = np.asarray(workspace["sides"][side]["arm_joint_seed"], dtype=np.float64)
        robot.robot.set_joint_positions(q.reshape(1, -1))
        robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
        app.update()

        lower, upper = _world_bbox(stage, target_path)
        center = (lower + upper) / 2.0
        side_offset = float((upper[0] - lower[0]) / 2.0 - 0.020)
        targets = {
            "left": center + np.asarray([-side_offset, 0.0, 0.0]),
            "right": center + np.asarray([side_offset, 0.0, 0.0]),
        }
        solved: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            q = robot.solve_ik(
                side,
                targets[side],
                robot.eef_tip_pose(side)[1],
                seed=q,
                max_iterations=60,
                position_tolerance=0.012,
            )
            solved[side] = q.copy()
        robot.robot.set_joint_positions(q.reshape(1, -1))
        app.update()

        side_normal = {"left": np.asarray([-1.0, 0.0, 0.0]), "right": np.asarray([1.0, 0.0, 0.0])}
        result: dict[str, object] = {}
        for side in ("left", "right"):
            midpoint, orientation = robot.gripper_midpoint_pose(side)
            gap = _gap_axis(robot, side)
            result[side] = {
                "target_midpoint_m": targets[side].tolist(),
                "refined_midpoint_m": midpoint.tolist(),
                "position_error_m": float(np.linalg.norm(midpoint - targets[side])),
                "gap_axis_world": gap.tolist(),
                "gap_axis_side_normal_alignment": float(abs(np.dot(gap, side_normal[side]))),
                "eef_quaternion_wxyz": orientation.tolist(),
                "full_joint_target": solved[side].tolist(),
                "arm_joint_target": solved[side][robot.arm_indices[side]].tolist(),
            }
        report = {
            "report_version": "r1-small-klt-grasp-pose-v1",
            "meaning": "paused IK/gripper-axis calibration only; not contact, grasp, lift, or data",
            "source_usd_modified": False,
            "table_moved": False,
            "target": setup["target"],
            "sides": result,
            "result": "PASS"
            if all(float(result[s]["position_error_m"]) <= 0.020 for s in ("left", "right"))
            else "FAIL",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMALL_KLT_GRASP_POSE=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["result"] == "PASS" else 2
    except IKError as exc:
        print(f"SMALL_KLT_GRASP_POSE=FAIL IK {exc}", flush=True)
        return 2
    except Exception as exc:
        traceback.print_exc()
        print(f"SMALL_KLT_GRASP_POSE=FAIL reason={exc}", flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
