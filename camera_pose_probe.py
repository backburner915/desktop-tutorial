"""Fast camera pose/render probe for the spacerobot USD."""

from __future__ import annotations

import argparse
from pathlib import Path

from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.core.lighting import (
    _look_at_row_rotation,
    _quat_from_row_rotation,
    ensure_fixed_dataset_lighting,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=Path("D:/Galaxea_Lab-galaxea-main/spacerobot.usd"))
    parser.add_argument("--output", type=Path, default=Path("dataset_camera_probe.png"))
    args = parser.parse_args()

    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=True))
    try:
        import omni.usd
        from isaacsim.sensors.camera import Camera

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
        for _ in range(120):
            app.update()
        ensure_fixed_dataset_lighting(stage)
        path = "/World/garobot2_driveable_final/r1_DVT_colored/torso_link4/front_camera"
        camera = Camera(path, name="r1_camera_pose_probe", resolution=(320, 240))
        camera.initialize(attach_rgb_annotator=True)
        position = [0.0, 3.85, 2.2]
        target = [-3.75, 3.85, 0.70]
        rotation = _look_at_row_rotation(position, target)
        camera.set_world_pose(
            position=position,
            orientation=_quat_from_row_rotation(rotation),
            camera_axes="usd",
        )
        for _ in range(30):
            app.update()
        image = camera.get_rgb(device="cpu")
        if image is None:
            raise RuntimeError("camera returned no RGB frame")
        import numpy as np
        from PIL import Image

        array = np.asarray(image)
        if array.ndim == 4:
            array = array[0]
        Image.fromarray(np.asarray(array)[..., :3].astype(np.uint8)).save(args.output)
        print(
            f"camera probe saved={args.output.resolve()} min={array.min()} max={array.max()} "
            f"mean={float(array.mean()):.3f}",
            flush=True,
        )
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
