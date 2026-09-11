"""Read-only runtime gravity and rigid-body status check for spacerobot.usd."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import load_scene_config
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.tools.physical_reference_probe import _open_stage


def _attr_value(prim, name):
    attr = prim.GetAttribute(name)
    return attr.Get() if attr and attr.IsValid() else None


def _vec3(value) -> list[float] | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).reshape(3).round(6).tolist()


def _body_status(prim) -> dict[str, object]:
    from pxr import UsdPhysics

    kinematic = _attr_value(prim, "physics:kinematicEnabled")
    dynamic = bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)) and kinematic is not True
    return {
        "valid": bool(prim and prim.IsValid()),
        "has_rigid_body_api": bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)),
        "has_collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
        "kinematic": bool(kinematic) if kinematic is not None else False,
        "dynamic": dynamic,
        "static": bool(prim.HasAPI(UsdPhysics.CollisionAPI)) and not bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Physics Scene gravity check")
    parser.add_argument("--scene-config", type=Path, default=PACKAGE_ROOT.parent / "scene_config.json")
    parser.add_argument("--target-object", default="/World/TaskSetup/MovablePayloads/T03")
    parser.add_argument("--support", default="/World/TaskSetup/Fixtures/StorageRack/Top")
    args = parser.parse_args()

    cfg = load_scene_config(args.scene_config)
    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        stage, _timeline = _open_stage(app, cfg["stage_path"], warmup_updates=0)
        from pxr import UsdPhysics, UsdGeom

        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        active = [prim for prim in scenes if prim.IsActive()]
        if len(active) == 1:
            active_scene = active[0]
        elif len(scenes) == 1:
            active_scene = scenes[0]
        else:
            primary = stage.GetPrimAtPath("/World/PhysicsScene")
            active_scene = primary if primary and primary.IsValid() else None

        print("R1 RUNTIME GRAVITY CHECK")
        print(f"stage: {cfg['stage_path']}")
        print(f"stage meters per unit: {meters_per_unit:g}")
        if active_scene is None:
            print("active PhysicsScene: MISSING")
            return 2

        direction = _attr_value(active_scene, "physics:gravityDirection")
        authored_magnitude = _attr_value(active_scene, "physics:gravityMagnitude")
        gravity_stage_units = float(authored_magnitude) if authored_magnitude is not None else float("nan")
        gravity_m_s2 = gravity_stage_units * meters_per_unit
        gravity_g = gravity_m_s2 / 9.81
        print(f"active PhysicsScene: {active_scene.GetPath()}")
        print(f"gravity direction: {_vec3(direction)}")
        print(f"gravity magnitude: {gravity_m_s2:.6f} m/s^2 ({gravity_g:.6f} g)")
        print(f"gravity authored magnitude: {gravity_stage_units:.6f} stage-units/s^2")
        print(f"target dynamic/kinematic status: {_body_status(stage.GetPrimAtPath(args.target_object))}")
        print(f"support static/dynamic status: {_body_status(stage.GetPrimAtPath(args.support))}")
        print(f"target path: {args.target_object}")
        print(f"support path: {args.support}")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
