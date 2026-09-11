"""Audit the actual physics and collision setup of the active SmallKLT reference.

This diagnostic is deliberately read-only with respect to the source USD.  It
opens the same Session-Layer setup as the physical reference and reports the
resolved payload and finger colliders which PhysX will actually use.  It does
not run a grasp trajectory.
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
from r1_bimanual_dataset.tools.physical_reference_probe import _world_bbox


def _collision_record(stage, prim):
    """Return authored/derived collision evidence without making any edits."""

    from pxr import PhysxSchema, UsdPhysics

    collision = prim.HasAPI(UsdPhysics.CollisionAPI)
    rigid_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    mass_api = prim.HasAPI(UsdPhysics.MassAPI)
    material_api = prim.HasAPI(UsdPhysics.MaterialAPI)
    record = {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "rigid_body_api": rigid_body,
        "collision_api": collision,
        "collision_enabled": prim.GetAttribute("physics:collisionEnabled").Get() if collision else None,
        "mass_api": mass_api,
        "mass_kg": prim.GetAttribute("physics:mass").Get() if mass_api else None,
        "density_kg_m3": prim.GetAttribute("physics:density").Get() if mass_api else None,
        "material_api": material_api,
        "static_friction": prim.GetAttribute("physics:staticFriction").Get() if material_api else None,
        "dynamic_friction": prim.GetAttribute("physics:dynamicFriction").Get() if material_api else None,
        "restitution": prim.GetAttribute("physics:restitution").Get() if material_api else None,
        "contact_offset_m": prim.GetAttribute("physxCollision:contactOffset").Get() if collision else None,
        "rest_offset_m": prim.GetAttribute("physxCollision:restOffset").Get() if collision else None,
        "torsional_patch_radius_m": prim.GetAttribute("physxCollision:torsionalPatchRadius").Get() if collision else None,
        "physx_collision_api": prim.HasAPI(PhysxSchema.PhysxCollisionAPI),
    }
    if collision:
        try:
            lower, upper = _world_bbox(stage, str(prim.GetPath()))
            record["bbox_min_m"] = lower.tolist()
            record["bbox_max_m"] = upper.tolist()
        except Exception as exc:  # one malformed child must not hide the rest
            record["bbox_error"] = str(exc)
    return record


def _records_under(stage, root_path: str):
    return [_collision_record(stage, prim) for prim in stage.Traverse() if str(prim.GetPath()).startswith(root_path)]


def main() -> int:
    from r1_bimanual_dataset.core.reference_policy import INVALID_KLT
    print(INVALID_KLT + ": frozen audit retained in reports/small_klt_physics_audit.json")
    return 2


def _retired_main_for_source_regression_only() -> int:
    parser = argparse.ArgumentParser(description="Read-only SmallKLT / R1 gripper PhysX audit")
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "reports" / "small_klt_physics_audit.json")
    args = parser.parse_args()

    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        stage, timeline, _robot, target_path, setup = prepare_fixed_target_scene(app, _selected_cfg())
        timeline.pause()
        app.update()
        target_records = _records_under(stage, target_path)
        fingers = {
            side: _records_under(stage, f"{setup['fixed_root']}/{side}_gripper_link")
            for side in ("left", "right")
        }
        enabled_target_colliders = [item for item in target_records if item["collision_api"] and item["collision_enabled"] is not False]
        enabled_finger_colliders = {
            side: [item for item in records if item["collision_api"] and item["collision_enabled"] is not False]
            for side, records in fingers.items()
        }
        axis_path = PACKAGE_ROOT / "reports" / "small_klt_gripper_axis_report.json"
        axis = json.loads(axis_path.read_text(encoding="utf-8")) if axis_path.exists() else {"result": "MISSING"}
        target_min = np.asarray(setup["target"]["bbox_min_m"], dtype=float)
        target_max = np.asarray(setup["target"]["bbox_max_m"], dtype=float)
        # A collision hull can be a few millimetres outside a visual mesh, but
        # it cannot extend metres beyond it.  This catches unit/transform
        # mismatches that produce visually obvious clipping despite contact
        # events being present.
        envelope_min = target_min - 0.03
        envelope_max = target_max + 0.03
        envelope_violations = []
        for item in enabled_target_colliders:
            if "bbox_min_m" not in item or "bbox_max_m" not in item:
                envelope_violations.append({"path": item["path"], "reason": "no finite collision bbox"})
                continue
            lower = np.asarray(item["bbox_min_m"], dtype=float)
            upper = np.asarray(item["bbox_max_m"], dtype=float)
            if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
                    or np.any(lower < envelope_min) or np.any(upper > envelope_max)):
                envelope_violations.append(
                    {"path": item["path"], "bbox_min_m": item["bbox_min_m"], "bbox_max_m": item["bbox_max_m"]}
                )
        collider_geometry_matches_visual = bool(enabled_target_colliders) and not envelope_violations
        structural_physics_ok = bool(enabled_target_colliders) and all(bool(enabled_finger_colliders[side]) for side in ("left", "right"))
        report = {
            "report_version": "r1-small-klt-physics-audit-v1",
            "meaning": "resolved Session-Layer PhysX audit; no grasp is run and source USD is unchanged",
            "source_usd_modified": False,
            "target": setup["target"],
            "target_prims": target_records,
            "finger_prims": fingers,
            "enabled_target_collider_count": len(enabled_target_colliders),
            "enabled_finger_collider_count": {side: len(items) for side, items in enabled_finger_colliders.items()},
            "collision_visual_envelope_margin_m": 0.03,
            "target_collision_visual_envelope_violations": envelope_violations,
            "target_collision_geometry_matches_visual": collider_geometry_matches_visual,
            "gripper_axis_calibration": axis,
            "structural_physics_ok": structural_physics_ok,
            "safe_for_data_reference": structural_physics_ok
            and collider_geometry_matches_visual
            and axis.get("result") == "PASS",
            "result": "PASS" if structural_physics_ok and collider_geometry_matches_visual else "FAIL",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMALL_KLT_PHYSICS_AUDIT=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["result"] == "PASS" else 2
    except Exception as exc:
        traceback.print_exc()
        print(f"SMALL_KLT_PHYSICS_AUDIT=FAIL reason={exc}", flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
