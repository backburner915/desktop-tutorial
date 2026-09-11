"""Physical bilateral-contact gate for the selected SmallKLT reference.

This is intentionally narrower than a grasp episode: fixed base, open
grippers at calibrated side-wall poses, gradual close, then a short hold.  No
payload attachment, collision filtering, root/joint state lock, lifting, or
dataset recording is used.  PASS requires actual PhysX contact reports from
both grippers.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical no-lift bilateral contact gate for SmallKLT")
    parser.add_argument(
        "--grasp-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_grasp_pose_report.json"
    )
    parser.add_argument(
        "--axis-report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_gripper_axis_report.json"
    )
    parser.add_argument("--close-steps", type=int, default=90)
    parser.add_argument("--hold-steps", type=int, default=120)
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_bimanual_contact_report.json")
    args = parser.parse_args()
    grasp = json.loads(args.grasp_report.read_text(encoding="utf-8"))
    axis = json.loads(args.axis_report.read_text(encoding="utf-8"))
    if grasp.get("result") != "PASS":
        raise RuntimeError("grasp-pose report is not PASS")

    cfg = _selected_cfg()
    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        _stage, timeline, robot, _target_path, setup = prepare_fixed_target_scene(app, cfg)
        tracker = setup["contact_tracker"]
        q = robot.full_positions()
        for side in ("left", "right"):
            q[robot.arm_indices[side]] = np.asarray(grasp["sides"][side]["arm_joint_target"], dtype=np.float64)
            q[int(robot.arm_indices[side][-1])] = float(axis["sides"][side]["wrist_joint6"])
            q[robot.gripper_indices[side]] = 0.05
        # This is reset/setup before the contact phase, not a per-step lock.
        robot.robot.set_joint_positions(q.reshape(1, -1))
        robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count), dtype=np.float64))
        robot.apply_full_target(q)
        for _ in range(10):
            app.update()
        initial_object, _ = robot.object_pose()
        contact_steps = {"left": 0, "right": 0, "both": 0}
        pair_history: dict[str, set[str]] = {"left": set(), "right": set()}
        close_start = q.copy()
        close_end = q.copy()
        for side in ("left", "right"):
            close_end[robot.gripper_indices[side]] = 0.0
        for step in range(args.close_steps + args.hold_steps):
            alpha = min(1.0, (step + 1) / max(args.close_steps, 1))
            command = close_start + alpha * (close_end - close_start)
            robot.apply_full_target(command)
            app.update()
            sample = tracker.sample()
            if sample.left:
                contact_steps["left"] += 1
            if sample.right:
                contact_steps["right"] += 1
            if sample.left and sample.right:
                contact_steps["both"] += 1
            for side in ("left", "right"):
                pair_history[side].update(sample.pairs[side])
        final_object, _ = robot.object_pose()
        velocity = robot.object_velocity()
        finite = bool(np.isfinite(final_object).all() and np.isfinite(velocity).all())
        displacement = float(np.linalg.norm(final_object - initial_object))
        report = {
            "report_version": "r1-small-klt-bimanual-contact-v1",
            "meaning": "physical bilateral close-and-hold only; no lift and no data episode",
            "source_usd_modified": False,
            "table_moved": False,
            "payload_attachment": False,
            "collision_filters_changed": False,
            "target": setup["target"],
            "close_steps": args.close_steps,
            "hold_steps": args.hold_steps,
            "contact_steps": contact_steps,
            "contact_pairs": {side: sorted(values) for side, values in pair_history.items()},
            "object_displacement_m": displacement,
            "object_velocity_m_rad_s": velocity.tolist(),
            "finite": finite,
            "result": "PASS"
            if finite and contact_steps["left"] >= 5 and contact_steps["right"] >= 5 and contact_steps["both"] >= 3
            else "FAIL",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMALL_KLT_BIMANUAL_CONTACT=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["result"] == "PASS" else 2
    except Exception as exc:
        traceback.print_exc()
        print(f"SMALL_KLT_BIMANUAL_CONTACT=FAIL reason={exc}", flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
