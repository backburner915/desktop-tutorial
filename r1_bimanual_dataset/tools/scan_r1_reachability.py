"""Map live R1 wrist reachability without running a grasp or moving payloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import load_scene_config
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.tools.physical_reference_probe import (
    _make_existing_r1_fixed,
    _open_stage,
    _place_existing_robot_in_session,
    _source_base_link_world_translation,
    _world_bbox,
    select_long_edge_pose,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paused-FK reachability scan for R1")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.samples < 20:
        raise ValueError("--samples must be at least 20")

    cfg = load_scene_config(ROOT / "scene_config.json")
    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        stage, timeline = _open_stage(app, cfg["stage_path"])
        obj_min, obj_max = _world_bbox(stage, cfg["target_object"])
        base, rotation, _ = select_long_edge_pose(obj_min, obj_max, side="near", stand_off_m=0.72)
        source_offset = _source_base_link_world_translation(stage, cfg["robot_prim"])
        desired_link = base.copy()
        desired_link[2] = source_offset[2]
        _place_existing_robot_in_session(stage, cfg["robot_prim"], desired_link - source_offset, rotation)
        physical_robot_prim = _make_existing_r1_fixed(stage, cfg["robot_prim"], relocate_articulation_root=False)

        from isaacsim.core.simulation_manager import SimulationManager
        from r1_bimanual_dataset.core.robot_interface import RobotInterface

        physical_cfg = dict(cfg)
        physical_cfg.update(
            {
                "robot_prim": physical_robot_prim,
                "base_movement": "disabled",
                "runtime_base_lock": False,
                "runtime_joint_state_lock": False,
                "physical_rollout": True,
                "scripted_grasp_attachment": False,
            }
        )
        timeline.play()
        SimulationManager.initialize_physics()
        for _ in range(8):
            app.update()
        robot = RobotInterface(physical_cfg)
        robot.initialize()
        home = robot.full_positions()
        root_position, root_quaternion = robot.root_pose()
        object_position, _ = robot.object_pose()
        target = object_position.copy()
        target[2] += 0.03
        timeline.stop()

        # Confirm that a paused Kit update refreshes link transforms after a
        # state write.  This makes sampling kinematic only: no contact forces,
        # payload motion, or gravity integration occur during the scan.
        check = home.copy()
        left0 = int(robot.arm_indices["left"][0])
        lower, upper = robot.limits()
        check[left0] = np.clip(check[left0] + 0.08, lower[left0], upper[left0])
        before, _ = robot.eef_tip_pose("left")
        robot.robot.set_joint_positions(check.reshape(1, -1))
        app.update()
        after, _ = robot.eef_tip_pose("left")
        if float(np.linalg.norm(after - before)) < 1e-4:
            raise RuntimeError("paused FK does not refresh link poses; scan aborted before any dynamic sampling")
        robot.robot.set_joint_positions(home.reshape(1, -1))
        app.update()

        rng = np.random.default_rng(args.seed)
        # Keep 5% clear of hard limits. Joint states are never physics-stepped.
        margin = 0.05 * (upper - lower)
        scan_lower, scan_upper = lower + margin, upper - margin
        results: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
        for side in ("left", "right"):
            indices = robot.arm_indices[side]
            best_distance = float("inf")
            best_q = home.copy()
            best_tip = np.zeros(3)
            # Include home as a deterministic candidate, then broad samples.
            for sample_index in range(args.samples + 1):
                q = home.copy()
                if sample_index:
                    q[indices] = rng.uniform(scan_lower[indices], scan_upper[indices])
                robot.robot.set_joint_positions(q.reshape(1, -1))
                app.update()
                tip, _ = robot.eef_tip_pose(side)
                distance = float(np.linalg.norm(tip - target))
                if distance < best_distance:
                    best_distance, best_q, best_tip = distance, q.copy(), tip.copy()
            results[side] = (best_distance, best_q, best_tip)
            print(
                f"Reach scan {side}: closest={best_distance:.4f}m "
                f"tip={np.round(best_tip, 4).tolist()} arm_q={np.round(best_q[indices], 5).tolist()}",
                flush=True,
            )
        robot.robot.set_joint_positions(home.reshape(1, -1))
        app.update()
        print(
            f"Reach scan: target={np.round(target, 4).tolist()} root={np.round(root_position, 4).tolist()} "
            f"samples_per_arm={args.samples}",
            flush=True,
        )
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
