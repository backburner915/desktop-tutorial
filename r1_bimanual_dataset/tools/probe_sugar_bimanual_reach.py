"""Planning-only gate for the corrected fixed-base physical reference.

It verifies the *two independent arm pregrasp positions* above the new YCB
Sugar Box.  It neither closes a gripper nor moves the payload, so a PASS is
not reported as a grasp.  This gate exists to reject unreachable base/object
placements before contact-orientation and lift trials.
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
from r1_bimanual_dataset.core.physical_scene import prepare_fixed_sugar_scene


ASSETS = {
    "sugar_box": str(ROOT / "r1_bimanual_dataset" / "assets" / "004_sugar_box_physics.usd"),
    "cracker_box": str(ROOT / "r1_bimanual_dataset" / "assets" / "003_cracker_box_physics.usd"),
}


def _home_target(robot, scene_cfg: dict) -> np.ndarray:
    # The authored articulation exposes three wheel DOFs in addition to the
    # fixed 16D arm/gripper command.  Start those non-command joints at zero;
    # carrying an unwrapped wheel angle into PhysX's revolute drive target
    # API causes a hard [-2*pi,2*pi] rejection.
    q = np.zeros(robot.dof_count, dtype=np.float64)
    home = scene_cfg["home_joint_positions"]
    for index, name in enumerate(robot.all_joint_names):
        if name in home:
            q[index] = float(home[name])
    for side in ("left", "right"):
        q[robot.gripper_indices[side]] = 0.05
    return q


def _set_planning_state(robot, app, q: np.ndarray, steps: int = 4) -> None:
    robot.robot.set_joint_positions(q.reshape(1, -1))
    robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
    robot.apply_full_target(q)
    for _ in range(steps):
        app.update()


def main() -> int:
    parser = argparse.ArgumentParser(description="R1 candidate bimanual pregrasp reachability gate")
    parser.add_argument("--candidate", choices=sorted(ASSETS), default="sugar_box")
    parser.add_argument("--scene-config", type=Path, default=ROOT / "scene_config.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "candidate_reachability_report.json")
    parser.add_argument("--stand-height", type=float, default=0.0, help="temporary Session-Layer support height in metres")
    parser.add_argument("--spawn-xy", nargs=2, type=float, default=[-3.75, 3.85], metavar=("X", "Y"))
    parser.add_argument("--base-stand-off", type=float, default=0.45, help="base-to-target long-edge stand-off in metres")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--init-only", action="store_true", help="stop after physical initialization (diagnostic)")
    parser.add_argument("--ik-iterations", type=int, default=6, help="maximum iterations per arm for this reachability preflight")
    parser.add_argument("--line-search", action="store_true", help="enable expensive physical IK line search")
    parser.add_argument(
        "--jacobian-source",
        choices=("finite_difference", "native"),
        default="finite_difference",
        help="live position Jacobian used by the planning preflight",
    )
    parser.add_argument("--ik-step-limit", type=float, default=0.08, help="per-iteration arm joint step limit")
    parser.add_argument("--ik-damping", type=float, default=0.035, help="damped least-squares IK damping")
    parser.add_argument(
        "--diagnostic-components",
        default="minimal",
        help="comma-separated init components for --init-only: finger,supports,fixed,baseline,contacts",
    )
    args = parser.parse_args()

    scene_cfg = load_scene_config(args.scene_config)
    # Bimanual manipulation uses the centre of the existing long table edge.
    # This is a Session-Layer pose for the replacement object, not a move of
    # the source tabletop or its four supporting legs.
    scene_cfg = dict(scene_cfg)
    scene_cfg["target_asset_url"] = ASSETS[args.candidate]
    scene_cfg["target_spawn_xy"] = list(args.spawn_xy)
    scene_cfg["table_top"] = "/World/TaskSetup/Fixtures/StorageRack/Top"
    scene_cfg["target_stand_height_m"] = float(args.stand_height)
    scene_cfg["target_stand_dimensions_m"] = [0.34, 0.30, 0.42]
    scene_cfg["fixed_base_stand_off_m"] = float(args.base_stand_off)
    # This tool is a mapping gate.  A loose solver acceptance only lets us
    # record its actually reached boundary point; the strict 3 cm PASS below
    # remains unchanged and is the only value later runners may accept.
    scene_cfg["physical_ik_accept_position_error"] = 0.25
    scene_cfg["ik_line_search"] = bool(args.line_search)
    scene_cfg["ik_jacobian_source"] = str(args.jacobian_source)
    scene_cfg["physical_ik_step_limit"] = float(args.ik_step_limit)
    scene_cfg["physical_ik_damping"] = float(args.ik_damping)
    if args.init_only:
        components = {item.strip().lower() for item in args.diagnostic_components.split(",") if item.strip()}
        scene_cfg["prepare_contact_reports_before_physics"] = "contacts" in components
        scene_cfg["runtime_physics_baseline_enabled"] = "baseline" in components
        scene_cfg["diagnostic_skip_finger_repairs"] = "finger" not in components
        scene_cfg["diagnostic_skip_supports"] = "supports" not in components
        scene_cfg["diagnostic_skip_fixed_root"] = "fixed" not in components
    else:
        # Gate B uses contact reporting before PhysX and the complete runtime
        # baseline; these diagnostics are only selectable with --init-only.
        scene_cfg["prepare_contact_reports_before_physics"] = True
        scene_cfg["runtime_physics_baseline_enabled"] = True
    prepare_isaac_environment()
    from isaacsim import SimulationApp
    from r1_bimanual_dataset.tools.physical_reference_probe import _world_bbox

    app = SimulationApp(simulation_app_config(headless=not args.gui, renderer="RayTracedLighting" if args.gui else "None"))
    try:
        stage, timeline, robot, target_path, setup = prepare_fixed_sugar_scene(app, scene_cfg)
        if args.init_only:
            target_bbox_min, target_bbox_max = _world_bbox(stage, target_path)
            runtime_observation = {
                "root_pose": {
                    "position_m": robot.root_pose()[0].tolist(),
                    "quaternion_wxyz": robot.root_pose()[1].tolist(),
                },
                "eef_midpoints_m": {
                    side: robot.gripper_midpoint_pose(side)[0].tolist() for side in ("left", "right")
                },
                "target_bbox_m": {
                    "min": target_bbox_min.tolist(),
                    "max": target_bbox_max.tolist(),
                },
            }
            report = {
                "report_version": "r1-candidate-runtime-init-v1",
                "candidate": args.candidate,
                "result": "PASS",
                "target_spawn_xy": list(args.spawn_xy),
                "target_stand_height_m": float(args.stand_height),
                "setup": {key: value for key, value in setup.items() if key != "contact_tracker"},
                "runtime_observation": runtime_observation,
                "dataset_episodes_written": 0,
                "dataset_generation_enabled": False,
                "meaning": "runtime initialization diagnostic only; no IK or episode execution",
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("CANDIDATE_RUNTIME_INIT=" + json.dumps(report, ensure_ascii=False), flush=True)
            return 0
        q_home = _home_target(robot, scene_cfg)
        _set_planning_state(robot, app, q_home, steps=30)
        lower, upper = robot.limits()
        if np.any(q_home < lower - 1e-6) or np.any(q_home > upper + 1e-6):
            raise RuntimeError("configured Galaxea R1 home pose violates live joint limits")
        bbox_min, bbox_max = _world_bbox(stage, target_path)
        center = (bbox_min + bbox_max) / 2.0
        # Place the two independent pregrasp stations along the measured
        # object's long axis.  The old probe used +/-X offsets regardless of
        # the asset orientation, which made the reachability result test the
        # narrow dimension instead of the bimanual station separation.
        dimensions_xy = bbox_max[:2] - bbox_min[:2]
        long_axis_index = int(np.argmax(dimensions_xy))
        long_axis = np.zeros(3, dtype=np.float64)
        long_axis[long_axis_index] = 1.0
        station_half_spacing = float(dimensions_xy[long_axis_index]) * 0.25
        pregrasp_height = float(scene_cfg.get("pregrasp_height_m", 0.14))
        targets = {
            "left": center + station_half_spacing * long_axis + np.asarray([0.0, 0.0, pregrasp_height]),
            "right": center - station_half_spacing * long_axis + np.asarray([0.0, 0.0, pregrasp_height]),
        }
        q_left = robot.solve_ik("left", targets["left"], robot.eef_tip_pose("left")[1], seed=q_home, max_iterations=args.ik_iterations)
        q_both = robot.solve_ik("right", targets["right"], robot.eef_tip_pose("right")[1], seed=q_left, max_iterations=args.ik_iterations)
        _set_planning_state(robot, app, q_both, steps=12)
        midpoint_errors = {}
        reached_midpoints = {}
        for side in ("left", "right"):
            measured, _ = robot.gripper_midpoint_pose(side)
            reached_midpoints[side] = measured.tolist()
            midpoint_errors[side] = float(np.linalg.norm(measured - targets[side]))
        contacts = setup["contact_tracker"].sample().as_dict()
        report = {
            "report_version": "r1-candidate-bimanual-reach-v1",
            "candidate": args.candidate,
            "target_spawn_xy": list(args.spawn_xy),
            "target_stand_height_m": float(args.stand_height),
            **{key: value for key, value in setup.items() if key != "contact_tracker"},
            "pregrasp_targets_m": {side: value.tolist() for side, value in targets.items()},
            "reached_finger_midpoints_m": reached_midpoints,
            "pregrasp_midpoint_error_m": midpoint_errors,
            "contact_at_pregrasp": contacts,
            "gripper_closed": False,
            "payload_moved": False,
            "result": "PASS" if all(error <= 0.03 for error in midpoint_errors.values()) else "FAIL",
            "meaning": "two-arm pregrasp workspace map only; this is not a grasp or a lift",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("CANDIDATE_BIMANUAL_REACH=" + json.dumps(report, ensure_ascii=False), flush=True)
        if report["result"] != "PASS":
            return 2
        if args.gui and args.keep_open:
            while app.is_running():
                app.update()
        return 0
    except Exception as exc:
        traceback.print_exc()
        runtime_observation = None
        try:
            if "robot" in locals() and robot is not None:
                target_bbox_min, target_bbox_max = _world_bbox(stage, target_path)
                runtime_observation = {
                    "root_pose": {
                        "position_m": robot.root_pose()[0].tolist(),
                        "quaternion_wxyz": robot.root_pose()[1].tolist(),
                    },
                    "eef_midpoints_m": {
                        side: robot.gripper_midpoint_pose(side)[0].tolist() for side in ("left", "right")
                    },
                    "full_joint_positions": robot.full_positions().tolist(),
                    "arm_joint_positions": {
                        side: robot.full_positions()[robot.arm_indices[side]].tolist() for side in ("left", "right")
                    },
                    "target_bbox_m": {
                        "min": target_bbox_min.tolist(),
                        "max": target_bbox_max.tolist(),
                    },
                }
        except Exception:
            runtime_observation = None
        failure_report = {
            "report_version": "r1-candidate-bimanual-reach-v1",
            "candidate": args.candidate,
            "target_spawn_xy": list(args.spawn_xy),
            "target_stand_height_m": float(args.stand_height),
            "result": "INVALID",
            "invalid_reason": "ik_unsolved",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_observation": runtime_observation,
            "dataset_episodes_written": 0,
            "dataset_generation_enabled": False,
            "meaning": "workspace preflight only; IK failure is INVALID and is not failure-oriented data",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(failure_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"CANDIDATE_BIMANUAL_REACH=FAIL reason={exc}", flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
