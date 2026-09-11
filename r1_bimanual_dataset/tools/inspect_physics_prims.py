"""Inspect collision/rigid prims in the corrected spacerobot stage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import load_scene_config
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.tools.physical_reference_probe import _open_stage, _world_bbox


def main() -> int:
    scene_cfg = load_scene_config(ROOT / "scene_config.json")
    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        print("INSPECT: app ready", flush=True)
        stage, _ = _open_stage(app, scene_cfg["stage_path"])
        print(f"INSPECT: stage ready {stage.GetRootLayer().realPath}", flush=True)
        from pxr import Gf, UsdGeom, UsdPhysics
        probe_matrix = Gf.Matrix4d(1.0)
        probe_return = probe_matrix.SetScale(Gf.Vec3d(1.65, 0.55, 0.07))
        print(f"MATRIX_PROBE after_set_scale={probe_matrix} return={probe_return}", flush=True)
        probe_matrix.SetTranslate(Gf.Vec3d(-3.75, 3.63, 1.27))
        print(f"MATRIX_PROBE after_set_translate={probe_matrix}", flush=True)

        target_min, target_max = _world_bbox(stage, scene_cfg["target_object"])
        target_center = (target_min + target_max) / 2.0
        print(f"TARGET {scene_cfg['target_object']} bbox={np.round(target_min, 4).tolist()}..{np.round(target_max, 4).tolist()}", flush=True)
        support_path = "/World/TaskSetup/Fixtures/StorageRack/Top"
        print("SUPPORT_TRANSFORMS", flush=True)
        xform_cache = UsdGeom.XformCache()
        for path in [support_path, "/World/TaskSetup/Fixtures/StorageRack"]:
            prim = stage.GetPrimAtPath(path)
            xformable = UsdGeom.Xformable(prim)
            matrix = xform_cache.GetLocalToWorldTransform(prim)
            print(
                f"  {path} type={prim.GetTypeName()} ops={[op.GetOpName() for op in xformable.GetOrderedXformOps()]} "
                f"matrix={matrix} bbox={np.round(np.asarray(_world_bbox(stage, path)[0]), 4).tolist()}..{np.round(np.asarray(_world_bbox(stage, path)[1]), 4).tolist()}",
                flush=True,
            )
        print("TARGET_TREE")
        target = stage.GetPrimAtPath(scene_cfg["target_object"])
        for prim in [target, *list(stage.Traverse())[list(stage.Traverse()).index(target) + 1 :]]:
            if not str(prim.GetPath()).startswith(str(target.GetPath()) + "/"):
                break
            print(
                f"  {prim.GetPath()} type={prim.GetTypeName()} schemas={prim.GetAppliedSchemas()} "
                f"collision={prim.HasAPI(UsdPhysics.CollisionAPI)} rigid={prim.HasAPI(UsdPhysics.RigidBodyAPI)}"
            )
        print("COLLIDERS_NEAR_TARGET")
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            try:
                low, high = _world_bbox(stage, str(prim.GetPath()))
            except Exception:
                continue
            if np.all(high >= target_center - np.asarray([1.0, 1.0, 1.0])) and np.all(
                low <= target_center + np.asarray([1.0, 1.0, 1.0])
            ):
                enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                print(f"  {prim.GetPath()} type={prim.GetTypeName()} enabled={enabled} bbox={np.round(low, 4).tolist()}..{np.round(high, 4).tolist()}", flush=True)
        return 0
    except Exception as exc:
        print(f"INSPECT: FAILED {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
