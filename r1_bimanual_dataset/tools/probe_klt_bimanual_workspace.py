"""Find an honest two-arm workspace seed for the physical SmallKLT target.

This is deliberately a paused forward-kinematics search.  It does *not* close
either gripper, apply a payload constraint, or claim a grasp.  Its only job is
to reject an unsuitable fixed-base/object layout before the physical-contact
trial is written.
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


def _home(robot, cfg: dict[str, object]) -> np.ndarray:
    q = robot.full_positions()
    configured = dict(cfg["home_joint_positions"])
    for index, name in enumerate(robot.all_joint_names):
        if name in configured:
            q[index] = float(configured[name])
    for side in ("left", "right"):
        q[robot.gripper_indices[side]] = 0.05
    return q


def _sample_arm_seeds(robot, side: str, home: np.ndarray, count: int, rng: np.random.Generator) -> list[np.ndarray]:
    lower, upper = robot.limits()
    arm_ids = robot.arm_indices[side]
    spread = np.asarray([1.25, 0.85, 1.05, 1.65, 1.30, 1.70], dtype=np.float64)
    seeds = [home.copy()]
    for _ in range(count):
        q = home.copy()
        q[arm_ids] = np.clip(home[arm_ids] + rng.uniform(-spread, spread), lower[arm_ids], upper[arm_ids])
        seeds.append(q)
    return seeds


def _nearest_seed(robot, app, side: str, seeds: list[np.ndarray], target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    best_distance = float("inf")
    best_q = seeds[0].copy()
    best_midpoint = np.zeros(3, dtype=np.float64)
    for q in seeds:
        robot.robot.set_joint_positions(q.reshape(1, -1))
        app.update()
        midpoint, _ = robot.gripper_midpoint_pose(side)
        distance = float(np.linalg.norm(midpoint - target))
        if distance < best_distance:
            best_distance, best_q, best_midpoint = distance, q.copy(), midpoint.copy()
    return best_distance, best_q, best_midpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="SmallKLT two-arm paused-FK workspace gate")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stand-off", type=float, default=0.78, help="base-to-table-centre offset in metres")
    parser.add_argument("--stand-height", type=float, default=0.55, help="static support height above the unchanged table")
    parser.add_argument("--target-y", type=float, default=3.75, help="world Y centre of the KLT support")
    parser.add_argument("--target-scale", type=float, default=1.0, help="uniform physical SmallKLT scale")
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_bimanual_workspace_report.json")
    args = parser.parse_args()
    if args.samples < 100:
        raise ValueError("--samples must be >= 100")

    cfg = dict(load_scene_config(ROOT / "scene_config.json"))
    # Keep the original table and its legs.  The static stand rests on that
    # table and gives the fixed-base R1 a reachable work height.  Rotating KLT
    # puts its 0.297 m long side along world X, i.e. across the two arms.
    cfg.update(
        {
            "target_asset_url": SMALL_KLT_PHYSICS_PATH,
            "target_spawn_xy": [-3.20, args.target_y],
            "target_yaw_rad": float(np.pi / 2.0),
            "target_uniform_scale": args.target_scale,
            "table_top": "/World/TaskSetup/Fixtures/StorageRack/Top",
            "target_stand_height_m": args.stand_height,
            # The selected 1.5x KLT is 0.297 m deep after yaw.  A 0.298 m
            # support gives it a full, non-overhanging footprint while still
            # remaining inside the authored table boundary.
            "target_stand_dimensions_m": [0.52, 0.298, args.stand_height],
            "fixed_base_side": "near",
            "fixed_base_stand_off_m": args.stand_off,
            "runtime_base_lock": False,
            "runtime_joint_state_lock": False,
            "scripted_grasp_attachment": False,
            "physical_rollout": True,
        }
    )
    prepare_isaac_environment()
    from isaacsim import SimulationApp
    from r1_bimanual_dataset.tools.physical_reference_probe import _world_bbox

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        stage, timeline, robot, target_path, setup = prepare_fixed_target_scene(app, cfg)
        # No dynamic contact/rollout is permitted during a workspace gate.
        # ``stop()`` tears down Isaac Sim 5.1's articulation physics view;
        # pause keeps the view alive while allowing FK to refresh after a
        # direct joint-state write.
        timeline.pause()
        q_home = _home(robot, cfg)
        robot.robot.set_joint_positions(q_home.reshape(1, -1))
        robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
        app.update()
        lower, upper = _world_bbox(stage, target_path)
        center = (lower + upper) / 2.0
        dimensions = upper - lower
        # Exterior side-wall targets: this is a geometric *midpoint* goal,
        # not a statement that contact/orientation has already been solved.
        side_offset = float(dimensions[0]) / 2.0 - 0.020
        if side_offset <= 0.05:
            raise RuntimeError("target is too narrow to provide two independent exterior grasp sides")
        targets = {
            "left": center + np.asarray([-side_offset, 0.0, 0.0]),
            "right": center + np.asarray([side_offset, 0.0, 0.0]),
        }
        rng = np.random.default_rng(args.seed)
        result: dict[str, object] = {}
        for side in ("left", "right"):
            distance, q, midpoint = _nearest_seed(
                robot, app, side, _sample_arm_seeds(robot, side, q_home, args.samples, rng), targets[side]
            )
            result[side] = {
                "target_midpoint_m": targets[side].tolist(),
                "best_midpoint_m": midpoint.tolist(),
                "distance_m": distance,
                "full_joint_seed": q.tolist(),
                "arm_joint_seed": q[robot.arm_indices[side]].tolist(),
            }
        report = {
            "report_version": "r1-small-klt-bimanual-workspace-v1",
            "meaning": "paused FK workspace seed search only; not a contact, grasp, lift, or data episode",
            "source_usd_modified": False,
            "table_moved": False,
            "target": {key: value for key, value in setup["target"].items()},
            "base_placement": setup["base_placement"],
            "actual_base_link_pose": {
                "position_m": robot.link_pose("base_link")[0].tolist(),
                "quaternion_wxyz": robot.link_pose("base_link")[1].tolist(),
            },
            "target_bbox_m": {"min": lower.tolist(), "max": upper.tolist(), "dimensions": dimensions.tolist()},
            "samples_per_arm": args.samples,
            "sides": result,
            "result": "PASS" if all(float(result[s]["distance_m"]) <= 0.055 for s in ("left", "right")) else "FAIL",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMALL_KLT_BIMANUAL_WORKSPACE=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["result"] == "PASS" else 2
    except Exception as exc:
        traceback.print_exc()
        print(f"SMALL_KLT_BIMANUAL_WORKSPACE=FAIL reason={exc}", flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
