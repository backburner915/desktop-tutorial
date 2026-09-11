"""Read a USD asset's physical readiness and world-space bounds.

The installed Isaac Sim 5.1 Python distribution exposes ``pxr`` only after a
``SimulationApp`` is created.  The tool therefore starts one no-render Kit
process and can inspect multiple assets in that one process.  It never edits
the target asset or the task stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def inspect(path: str) -> dict[str, object]:
    from pxr import Usd, UsdGeom, UsdPhysics

    path_text = str(path)
    if "://" in path_text:
        import omni.client

        result, _entry = omni.client.stat(path_text)
        if result != omni.client.Result.OK:
            return {"path": path_text, "available": False, "availability_result": str(result)}
    elif not Path(path_text).is_file():
        return {"path": path_text, "available": False}
    stage = Usd.Stage.Open(path_text)
    if stage is None:
        return {"path": path_text, "available": True, "openable": False}
    default_prim = stage.GetDefaultPrim()
    root = default_prim if default_prim and default_prim.IsValid() else stage.GetPseudoRoot()
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    try:
        bound = cache.ComputeWorldBound(root).ComputeAlignedBox()
        lower = np.asarray(bound.GetMin(), dtype=np.float64)
        upper = np.asarray(bound.GetMax(), dtype=np.float64)
    except Exception:
        lower = upper = None
    rigid = []
    colliders = []
    meshes = []
    for prim in stage.Traverse():
        path_text = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid.append(path_text)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(path_text)
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(path_text)
    return {
        "path": path_text,
        "available": True,
        "openable": True,
        "default_prim": str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else None,
        "bbox_min_m": lower.tolist() if lower is not None else None,
        "bbox_max_m": upper.tolist() if upper is not None else None,
        "dimensions_m": (upper - lower).tolist() if lower is not None else None,
        "rigid_body_prims": rigid,
        "collision_prims": colliders,
        "mesh_count": len(meshes),
        "physics_ready": bool(rigid and colliders),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a candidate USD without changing it")
    parser.add_argument("asset", nargs="+")
    args = parser.parse_args()
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config

    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True, renderer="None"))
    try:
        print("ASSET_INSPECT: app ready", flush=True)
        report = [inspect(path) for path in args.asset]
        print("ASSET_INSPECT_RESULT=" + json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
