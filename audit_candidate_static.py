"""Run the candidate-object portion of Gate A in a Session Layer.

This diagnostic never writes an episode and never saves the source USD.  It
places one explicitly selected YCB candidate on the authored tabletop,
initializes the CPU PhysX scene at 120 Hz, and measures ten seconds of free
settling.  Gripper aperture remains a separate hard gate in
``inspect_scene_geometry.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "r1_bimanual_dataset"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import load_scene_config
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.core.reference_policy import require_candidate
from r1_bimanual_dataset.core.scene_overlay import (
    add_physical_target_overlay,
    settle_target_on_support,
)
from r1_bimanual_dataset.tools.inspect_scene_geometry import audit, bootstrap_usd
from r1_bimanual_dataset.tools.physical_reference_probe import _open_stage, _world_bbox


ASSETS = {
    "sugar_box": PKG / "assets" / "004_sugar_box_physics.usd",
    "cracker_box": PKG / "assets" / "003_cracker_box_physics.usd",
}


def _pose_array(prim):
    from pxr import UsdGeom

    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim).RemoveScaleShear()
    return np.asarray(matrix.ExtractTranslation(), dtype=np.float64)


def _report_visualization(stage, target_path: str, report_path: Path) -> dict[str, object]:
    """Create a deterministic bbox visualization artifact without claiming a render pass."""

    lower, upper = _world_bbox(stage, target_path)
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    prim = stage.GetPrimAtPath(target_path)
    bound = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    visual_lower = np.asarray(bound.GetMin(), dtype=np.float64)
    visual_upper = np.asarray(bound.GetMax(), dtype=np.float64)
    artifact = report_path.with_suffix(".bbox.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "PASS" if np.isfinite(lower).all() and np.isfinite(upper).all() else "FAIL",
                "visual_bbox_m": {"min": visual_lower.tolist(), "max": visual_upper.tolist()},
                "collision_bbox_m": {"min": lower.tolist(), "max": upper.tolist()},
                "method": "world-aligned visual/collision bounds; dynamic independent pad contact is deferred to Gate B",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "status": "PASS" if np.isfinite(lower).all() and np.isfinite(upper).all() else "FAIL",
        "visual_bbox_m": {"min": visual_lower.tolist(), "max": visual_upper.tolist()},
        "collision_bbox_m": {"min": lower.tolist(), "max": upper.tolist()},
        "visualization_artifact": str(artifact),
        "visualization_method": "world-aligned visual/collision bounds; dynamic independent pad contact is deferred to Gate B",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate A candidate static settle audit")
    parser.add_argument("--candidate", choices=sorted(ASSETS))
    parser.add_argument("--settle-seconds", type=float, default=10.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.settle_seconds <= 0:
        raise ValueError("--settle-seconds must be positive")

    asset = ASSETS[args.candidate]
    require_candidate(asset)
    report_path = args.report or ROOT / "reports" / f"candidate_{asset.stem}_static_audit.json"
    source = Path(load_scene_config(ROOT / "scene_config.json")["stage_path"])
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    report: dict[str, object] = {
        "report_version": "r1-candidate-static-gate-a-v1",
        "candidate": args.candidate,
        "asset": str(asset),
        "status": "INVALID",
        "gate": "A",
        "dataset_episodes_written": 0,
        "source_usd_modified": False,
        "settle_seconds_required": args.settle_seconds,
    }
    app = None
    try:
        if not asset.is_file():
            raise FileNotFoundError(str(asset))
        cfg = dict(load_scene_config(ROOT / "scene_config.json"))
        cfg.update(
            {
                "target_asset_url": str(asset),
                "target_spawn_xy": None,
                "target_yaw_rad": 0.0,
                "target_uniform_scale": 1.0,
                "runtime_base_lock": False,
                "runtime_joint_state_lock": False,
                "scripted_grasp_attachment": False,
                "physical_rollout": True,
            }
        )
        prepare_isaac_environment()
        bootstrap_usd()
        from r1_bimanual_dataset.core.qualified_scene import (
            add_table_supports,
            configure_physics,
            fix_root,
            repair_finger_collision_schema,
        )
        from isaacsim import SimulationApp

        app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
        stage, timeline = _open_stage(app, cfg["stage_path"])
        overlay = add_physical_target_overlay(stage, cfg)
        for _ in range(120):
            app.update()
        target_report = settle_target_on_support(stage, overlay, clearance_m=0.002)
        report["target"] = target_report
        from pxr import Usd

        asset_stage = Usd.Stage.Open(str(asset))
        report["geometry"] = audit(asset_stage, asset_stage.GetPseudoRoot())
        robot = stage.GetPrimAtPath(cfg["robot_prim"])
        report["finger_schema_repairs"] = repair_finger_collision_schema(stage, robot)
        table_path = str(cfg.get("table_top", "/World/TaskSetup/Fixtures/StorageRack/Top"))
        report["supports"] = add_table_supports(stage, table_path)
        report["fixed_root"] = fix_root(stage, robot)
        report["fixed_joint"] = "/World/ReferenceRootFixedJoint"
        report["physics"] = configure_physics(stage, robot)
        report["collision_visualization"] = _report_visualization(stage, overlay.target_path, report_path)

        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.prims import SingleRigidPrim

        timeline.play()
        SimulationManager.initialize_physics()
        for _ in range(20):
            app.update()
        target = SingleRigidPrim(overlay.target_path)
        target.initialize()
        initial_position, initial_quaternion = target.get_world_pose()
        initial_position = np.asarray(initial_position[0] if np.asarray(initial_position).ndim > 1 else initial_position, dtype=np.float64)
        initial_quaternion = np.asarray(initial_quaternion[0] if np.asarray(initial_quaternion).ndim > 1 else initial_quaternion, dtype=np.float64)
        steps = int(round(args.settle_seconds * 120.0))
        positions = []
        velocities = []
        quaternions = []
        for _ in range(steps):
            app.update()
            position, quaternion = target.get_world_pose()
            linear_velocity = target.get_linear_velocity()
            angular_velocity = target.get_angular_velocity()
            velocity = np.concatenate(
                [
                    np.asarray(linear_velocity[0] if np.asarray(linear_velocity).ndim > 1 else linear_velocity, dtype=np.float64),
                    np.asarray(angular_velocity[0] if np.asarray(angular_velocity).ndim > 1 else angular_velocity, dtype=np.float64),
                ]
            )
            positions.append(np.asarray(position[0] if np.asarray(position).ndim > 1 else position, dtype=np.float64))
            quaternions.append(np.asarray(quaternion[0] if np.asarray(quaternion).ndim > 1 else quaternion, dtype=np.float64))
            velocities.append(np.asarray(velocity[0] if np.asarray(velocity).ndim > 1 else velocity, dtype=np.float64))
        p = np.asarray(positions)
        v = np.asarray(velocities)
        q = np.asarray(quaternions)
        final_lower, final_upper = _world_bbox(stage, overlay.target_path)
        table_top = float(target_report["table_top_z_m"])
        bottom_clearance = float(final_lower[2] - table_top)
        report["settle"] = {
            "physics_steps": steps,
            "duration_s": steps / 120.0,
            "finite": bool(np.isfinite(p).all() and np.isfinite(v).all() and np.isfinite(q).all()),
            "max_translation_from_initial_m": float(np.max(np.linalg.norm(p - initial_position, axis=1))) if len(p) else 0.0,
            "max_xy_drift_m": float(np.max(np.linalg.norm((p - initial_position)[:, :2], axis=1))) if len(p) else 0.0,
            "max_speed_m_s": float(np.max(np.linalg.norm(v[:, :3], axis=1))) if len(v) else 0.0,
            "bottom_clearance_final_m": bottom_clearance,
            "bottom_clearance_min_m": float(np.min(p[:, 2] - table_top)) if len(p) else 0.0,
            "initial_pose_m": initial_position.tolist(),
            "final_pose_m": p[-1].tolist() if len(p) else initial_position.tolist(),
            "stable_on_support": bool(
                len(p)
                and np.isfinite(p).all()
                and np.isfinite(v).all()
                and np.max(np.linalg.norm((p - initial_position)[:, :2], axis=1)) <= 0.003
                and np.max(np.abs(v[-120:, :3])) <= 0.02
                and float(np.min(p[:, 2] - table_top)) >= -0.003
            ),
        }
        report["status"] = "PASS" if report["settle"]["stable_on_support"] and report["collision_visualization"]["status"] == "PASS" else "FAIL"
        report["gate_a_candidate_pass"] = bool(report["status"] == "PASS")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["source_usd_sha256_before"] = before_hash
        report["source_usd_sha256_after"] = hashlib.sha256(source.read_bytes()).hexdigest()
        report["source_usd_modified"] = report["source_usd_sha256_before"] != report["source_usd_sha256_after"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print("CANDIDATE_STATIC_AUDIT=" + json.dumps(report, ensure_ascii=False), flush=True)
        if app is not None:
            app.close()
    return 0 if report.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
