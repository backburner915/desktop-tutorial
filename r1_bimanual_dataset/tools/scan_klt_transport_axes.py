"""Find valid wrist-6 pinch directions at the paused KLT lift/carry poses."""

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
    a, _ = robot.link_pose(f"{side}_gripper_link1")
    b, _ = robot.link_pose(f"{side}_gripper_link2")
    delta = b - a
    return delta / max(float(np.linalg.norm(delta)), 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paused wrist scan for KLT lift/carry waypoints")
    parser.add_argument(
        "--transport-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_transport_pose_report.json"
    )
    parser.add_argument("--samples", type=int, default=121)
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_transport_axis_report.json")
    args = parser.parse_args()
    transport = json.loads(args.transport_report.read_text(encoding="utf-8"))

    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        _stage, timeline, robot, _target_path, setup = prepare_fixed_target_scene(app, _selected_cfg())
        timeline.pause()
        lower, upper = robot.limits()
        phases: dict[str, object] = {}
        for phase_name in ("lift", "carry"):
            phase = transport[phase_name]
            q = robot.full_positions()
            for side in ("left", "right"):
                q[robot.arm_indices[side]] = np.asarray(phase[side]["arm_joint_target"], dtype=np.float64)
                q[robot.gripper_indices[side]] = 0.05
            robot.robot.set_joint_positions(q.reshape(1, -1))
            robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
            app.update()
            phase_result: dict[str, object] = {}
            for side in ("left", "right"):
                wrist_id = int(robot.arm_indices[side][-1])
                target = np.asarray(phase[side]["target_midpoint_m"], dtype=np.float64)
                normal = np.asarray([-1.0, 0.0, 0.0] if side == "left" else [1.0, 0.0, 0.0])
                best: dict[str, object] | None = None
                for wrist in np.linspace(lower[wrist_id], upper[wrist_id], args.samples):
                    candidate = q.copy()
                    candidate[wrist_id] = wrist
                    robot.robot.set_joint_positions(candidate.reshape(1, -1))
                    app.update()
                    midpoint, _ = robot.gripper_midpoint_pose(side)
                    gap = _gap_axis(robot, side)
                    entry = {
                        "wrist_joint6": float(wrist),
                        "midpoint_m": midpoint.tolist(),
                        "position_error_m": float(np.linalg.norm(midpoint - target)),
                        "gap_axis_world": gap.tolist(),
                        "side_normal_alignment": float(abs(np.dot(gap, normal))),
                    }
                    if best is None or entry["side_normal_alignment"] > best["side_normal_alignment"]:
                        best = entry
                assert best is not None
                phase_result[side] = best
            phases[phase_name] = phase_result
        report = {
            "report_version": "r1-small-klt-transport-axis-v1",
            "meaning": "paused wrist-axis search for lift/carry only; not contact, transport, or data",
            "source_usd_modified": False,
            "table_moved": False,
            "target": setup["target"],
            "phases": phases,
            "result": "PASS"
            if all(
                float(phases[phase][side]["side_normal_alignment"]) >= 0.84
                and float(phases[phase][side]["position_error_m"]) <= 0.030
                for phase in ("lift", "carry")
                for side in ("left", "right")
            )
            else "FAIL",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMALL_KLT_TRANSPORT_AXIS=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["result"] == "PASS" else 2
    except Exception as exc:
        traceback.print_exc()
        print(f"SMALL_KLT_TRANSPORT_AXIS=FAIL reason={exc}", flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
