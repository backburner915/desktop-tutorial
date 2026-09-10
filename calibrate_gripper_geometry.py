"""Measure the live R1 gripper-link motion in the real Isaac Sim articulation.

This is a calibration-only script.  It creates a session-layer fixed base,
then samples gripper link origins at their authored joint limits.  It does
not execute a grasp or collect data, and it deliberately reports *link-origin
separation* rather than pretending that it is a measured finger-pad aperture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import load_scene_config
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.core.scene_overlay import add_sugar_box_overlay, settle_target_on_support
from r1_bimanual_dataset.tools.physical_reference_probe import (
    _make_existing_r1_fixed,
    _open_stage,
    _place_existing_robot_in_session,
    _source_base_link_world_translation,
    _world_bbox,
    select_long_edge_pose,
)


def _sample(robot, app, side: str, joint_target: float) -> dict[str, object]:
    q = robot.full_positions()
    indices = robot.gripper_indices[side]
    q[indices] = float(joint_target)
    robot.robot.set_joint_positions(q.reshape(1, -1))
    robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
    robot.apply_full_target(q)
    for _ in range(20):
        app.update()
    link1, _ = robot.link_pose(f"{side}_gripper_link1")
    link2, _ = robot.link_pose(f"{side}_gripper_link2")
    eef, eef_q = robot.eef_link_pose(side)
    midpoint = (link1 + link2) / 2.0
    return {
        "axis_target_m": float(joint_target),
        "link1_origin_m": link1.tolist(),
        "link2_origin_m": link2.tolist(),
        "link_origin_separation_m": float(np.linalg.norm(link1 - link2)),
        "finger_midpoint_m": midpoint.tolist(),
        "link6_position_m": eef.tolist(),
        "link6_quaternion_wxyz": eef_q.tolist(),
        "link6_to_midpoint_m": (midpoint - eef).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate real R1 gripper link motion")
    parser.add_argument("--scene-config", type=Path, default=ROOT / "scene_config.json")
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "reports" / "gripper_geometry_report.json")
    args = parser.parse_args()
    scene_cfg = load_scene_config(args.scene_config)
    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        stage, timeline = _open_stage(app, scene_cfg["stage_path"])
        overlay = add_sugar_box_overlay(stage, scene_cfg)
        for _ in range(120):
            app.update()
        object_report = settle_target_on_support(stage, overlay)

        # The table, not the object bbox, determines the side on which the
        # fixed mobile base is parked.  This guarantees a long-edge placement
        # even though the Sugar Box itself is narrow.
        table_min, table_max = _world_bbox(stage, overlay.support_path)
        base_position, base_rotation, placement = select_long_edge_pose(
            table_min, table_max, side="near", stand_off_m=0.78
        )
        source_offset = _source_base_link_world_translation(stage, scene_cfg["robot_prim"])
        desired_link = base_position.copy()
        desired_link[2] = source_offset[2]
        _place_existing_robot_in_session(
            stage, scene_cfg["robot_prim"], desired_link - source_offset, base_rotation
        )
        fixed_root = _make_existing_r1_fixed(
            stage, scene_cfg["robot_prim"], relocate_articulation_root=False
        )

        from isaacsim.core.simulation_manager import SimulationManager
        from r1_bimanual_dataset.core.robot_interface import RobotInterface

        timeline.play()
        SimulationManager.initialize_physics()
        for _ in range(10):
            app.update()
        cfg = dict(scene_cfg)
        cfg.update(
            {
                "robot_prim": fixed_root,
                "target_object": overlay.target_path,
                "eef_tip_offsets": {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]},
                "runtime_base_lock": False,
                "runtime_joint_state_lock": False,
                "physical_rollout": True,
                "scripted_grasp_attachment": False,
            }
        )
        robot = RobotInterface(cfg, update_fn=app.update)
        robot.initialize()
        lower, upper = robot.limits()
        report: dict[str, object] = {
            "report_version": "r1-gripper-geometry-v1",
            "source_usd_modified": False,
            "target": object_report,
            "base_placement": placement,
            "fixed_root": fixed_root,
            "sides": {},
            "fit_policy": {
                "narrowest_target_dimension_m": float(min(object_report["dimensions_m"])),
                "status": "PENDING_COLLISION_SURFACE_CALIBRATION",
                "reason": "joint/link-origin travel is not the same as inner finger-pad aperture",
            },
        }
        for side in ("left", "right"):
            indices = robot.gripper_indices[side]
            opened = float(min(upper[indices]))
            closed = float(max(lower[indices]))
            open_sample = _sample(robot, app, side, opened)
            close_sample = _sample(robot, app, side, closed)
            report["sides"][side] = {
                "gripper_joint_indices": indices.tolist(),
                "joint_lower_m": lower[indices].tolist(),
                "joint_upper_m": upper[indices].tolist(),
                "opened": open_sample,
                "closed": close_sample,
                "link_origin_travel_m": float(
                    open_sample["link_origin_separation_m"] - close_sample["link_origin_separation_m"]
                ),
            }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("GRIPPER_GEOMETRY=" + json.dumps(report, ensure_ascii=False), flush=True)
        timeline.stop()
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
