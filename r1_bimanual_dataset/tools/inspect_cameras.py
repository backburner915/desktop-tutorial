"""Print camera poses, lights, and one rendered sample for the corrected USD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=Path("D:/Galaxea_Lab-galaxea-main/spacerobot.usd"))
    args = parser.parse_args()

    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True))
    try:
        import omni.usd
        from isaacsim.sensors.camera import Camera
        from pxr import Sdf, UsdGeom, UsdLux

        context = omni.usd.get_context()
        if not context.open_stage(str(args.scene.resolve())):
            raise RuntimeError(f"unable to open stage: {args.scene}")
        for _ in range(2400):
            app.update()
            if context.get_stage() is not None:
                break
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("stage did not load")
        for _ in range(240):
            app.update()

        print("CAMERAS", flush=True)
        for label, path in {
            "robot": "/World/garobot2_driveable_final/r1_DVT_colored",
            "torso_link4": "/World/garobot2_driveable_final/r1_DVT_colored/torso_link4",
        }.items():
            prim = stage.GetPrimAtPath(path)
            print(f"{label}: transform={UsdGeom.XformCache().GetLocalToWorldTransform(prim)}", flush=True)
        paths = {
            "front": "/World/garobot2_driveable_final/r1_DVT_colored/torso_link4/front_camera",
            "left_wrist": "/World/garobot2_driveable_final/r1_DVT_colored/left_arm_link6/left_wrist_camera",
            "right_wrist": "/World/garobot2_driveable_final/r1_DVT_colored/right_arm_link6/right_wrist_camera",
        }
        for name, path in paths.items():
            prim = stage.GetPrimAtPath(path)
            xform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            parent_path = Sdf.Path(path).GetParentPath()
            parent_xform = UsdGeom.XformCache().GetLocalToWorldTransform(
                stage.GetPrimAtPath(parent_path)
            )
            print(
                f"{name}: prim={prim.IsValid()} "
                f"transform={xform}",
                flush=True,
            )
            print(f"{name}_parent: path={parent_path} transform={parent_xform}", flush=True)
        probe = Camera(paths["front"], name="r1_camera_pose_probe", resolution=(320, 240))
        probe.initialize(attach_rgb_annotator=True)
        for _ in range(8):
            app.update()
        probe_position, probe_quaternion = probe.get_world_pose(camera_axes="usd")
        print(
            f"front_camera_api_pose: position={probe_position} quaternion={probe_quaternion}",
            flush=True,
        )
        print("LIGHTS", flush=True)
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.Light):
                print(f"{prim.GetPath()} type={prim.GetTypeName()}", flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
