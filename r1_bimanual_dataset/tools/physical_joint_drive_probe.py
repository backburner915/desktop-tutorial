"""Verify that the real R1 drives move a fixed physical robot in Isaac Sim 5.1.

This is intentionally a small gate before attempting grasp planning.  It
creates the same Session-Layer long-edge placement and world fixed joint as the
physical reference probe, then measures whether native joint position targets
move the live articulation.  It never locks the root/joints, disables
collisions, attaches the payload, or writes the source USD.
"""

from __future__ import annotations

import argparse
import sys
import traceback
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


def _physical_config(scene_cfg: dict, robot_prim: str) -> dict:
    cfg = dict(scene_cfg)
    cfg.update(
        {
            "robot_prim": robot_prim,
            "base_movement": "disabled",
            "runtime_base_lock": False,
            "runtime_joint_state_lock": False,
            "physical_rollout": True,
            "scripted_grasp_attachment": False,
            "runtime_gripper_damping": 100.0,
        }
    )
    return cfg


def _run_target(robot, app, target: np.ndarray, ticks: int) -> None:
    for _ in range(ticks):
        robot.apply_full_target(target)
        app.update()


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical R1 native-drive validation")
    parser.add_argument("--gui", action="store_true", help="show the physical drive motion")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--ticks", type=int, default=45, help="physics ticks per target")
    args = parser.parse_args()
    if args.ticks < 30:
        raise ValueError("--ticks must be at least 30")

    scene_cfg = load_scene_config(ROOT / "scene_config.json")
    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(
        simulation_app_config(headless=not args.gui, renderer="RayTracedLighting" if args.gui else "None")
    )
    try:
        stage, timeline = _open_stage(app, scene_cfg["stage_path"])
        obj_min, obj_max = _world_bbox(stage, scene_cfg["target_object"])
        base_position, base_rotation, _ = select_long_edge_pose(obj_min, obj_max, side="near", stand_off_m=0.72)
        source_offset = _source_base_link_world_translation(stage, scene_cfg["robot_prim"])
        desired_base_link = base_position.copy()
        desired_base_link[2] = source_offset[2]
        _place_existing_robot_in_session(
            stage, scene_cfg["robot_prim"], desired_base_link - source_offset, base_rotation
        )
        physical_robot_prim = _make_existing_r1_fixed(
            stage, scene_cfg["robot_prim"], relocate_articulation_root=False
        )

        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.viewports import set_camera_view
        from r1_bimanual_dataset.core.robot_interface import RobotInterface

        set_camera_view(
            base_position + np.asarray([-1.4, -1.3, 1.1]),
            (obj_min + obj_max) / 2.0 + np.asarray([0.0, 0.0, 0.25]),
        )
        timeline.play()
        SimulationManager.initialize_physics()
        for _ in range(10):
            app.update()
        robot = RobotInterface(_physical_config(scene_cfg, physical_robot_prim), update_fn=app.update)
        robot.initialize()
        home = robot.full_positions()
        root_before, _ = robot.root_pose()
        left_tip_before, _ = robot.eef_tip_pose("left")
        joint_index = int(robot.arm_indices["left"][0])
        lower, upper = robot.limits()
        # A modest target is sufficient to prove a physical drive and stays
        # within limits, avoiding an accidental table strike during this gate.
        delta = 0.18 if home[joint_index] + 0.18 <= upper[joint_index] - 0.02 else -0.18
        target = home.copy()
        target[joint_index] = np.clip(home[joint_index] + delta, lower[joint_index] + 0.02, upper[joint_index] - 0.02)
        print(
            f"Drive probe: target {robot.all_joint_names[joint_index]} {home[joint_index]:.4f} -> {target[joint_index]:.4f}",
            flush=True,
        )
        _run_target(robot, app, target, args.ticks)
        left_tip_after, _ = robot.eef_tip_pose("left")
        root_after, _ = robot.root_pose()
        actual = robot.full_positions()[joint_index]
        joint_motion = abs(float(actual - home[joint_index]))
        target_error = abs(float(target[joint_index] - actual))
        tip_motion = float(np.linalg.norm(left_tip_after - left_tip_before))
        root_drift = float(np.linalg.norm(root_after - root_before))
        print(
            "Drive probe: measured "
            f"joint_motion={joint_motion:.5f}rad target_error={target_error:.5f}rad "
            f"left_tip_motion={tip_motion:.5f}m root_drift={root_drift:.8f}m",
            flush=True,
        )
        passed = joint_motion >= 0.06 and target_error <= 0.10 and tip_motion >= 0.004 and root_drift <= 1.0e-4
        if not passed:
            raise RuntimeError("native drive gate failed; do not attempt grasp planning")
        print("Drive probe: PASS — physical native arm target moves the fixed R1.", flush=True)
        _run_target(robot, app, home, args.ticks)
        timeline.stop()
        if args.keep_open and args.gui:
            print("Drive probe: GUI remains open. The robot is back at its physical home pose.", flush=True)
            while app.is_running():
                app.update()
        return 0
    except Exception as exc:
        print(f"Drive probe: FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
