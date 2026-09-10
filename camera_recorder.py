from __future__ import annotations

from typing import Any

import numpy as np

from .lighting import _look_at_row_rotation, _quat_from_row_rotation, _row_rotation_from_quaternion


class CameraRecorder:
    """Read all three camera annotators after one shared simulation step."""

    def __init__(self, scene_cfg: dict[str, Any], width: int = 320, height: int = 240):
        from isaacsim.sensors.camera import Camera

        self.cameras = {
            name: Camera(path, name=f"r1_dataset_{name}", resolution=(width, height))
            for name, path in scene_cfg["cameras"].items()
        }
        self.paths = dict(scene_cfg["cameras"])
        self._printed_runtime_pose = False

    def initialize(self) -> None:
        for camera in self.cameras.values():
            camera.initialize(attach_rgb_annotator=True)

    def update_runtime_views(
        self,
        parent_poses: dict[str, tuple[np.ndarray, np.ndarray]],
        target_position: np.ndarray,
        local_offsets: dict[str, list[float]],
    ) -> None:
        """Aim all cameras from live parent poses in the USD camera frame.

        The source scene labels two cameras as wrist cameras but their authored
        transforms are not physically near the wrist links.  Camera.set_world_pose
        is used here with ``camera_axes='usd'`` so the existing camera prims and
        RGB annotators remain intact while their runtime pose follows the live
        PhysX link pose.
        """

        target = np.asarray(target_position, dtype=np.float64).reshape(3)
        for name, camera in self.cameras.items():
            if name not in parent_poses:
                continue
            parent_position, parent_rotation = parent_poses[name]
            parent_position = np.asarray(parent_position, dtype=np.float64).reshape(3)
            parent_rotation = np.asarray(parent_rotation, dtype=np.float64).reshape(3, 3)
            offset = np.asarray(local_offsets.get(name, [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            camera_position = offset @ parent_rotation + parent_position
            desired_world = _look_at_row_rotation(camera_position, target)
            camera.set_world_pose(
                position=camera_position,
                orientation=_quat_from_row_rotation(desired_world),
                camera_axes="usd",
            )
            if not self._printed_runtime_pose and name == "front":
                actual_position, actual_quaternion = camera.get_world_pose(camera_axes="usd")
                actual_rotation = _row_rotation_from_quaternion(actual_quaternion)
                print(
                    "R1 runner: camera API pose check "
                    f"expected_position={np.round(camera_position, 3).tolist()} "
                    f"actual_position={np.round(actual_position, 3).tolist()} "
                    f"expected_forward={np.round(-desired_world[2], 3).tolist()} "
                    f"actual_forward={np.round(-actual_rotation[2], 3).tolist()}",
                    flush=True,
                )
                self._printed_runtime_pose = True

    def capture_after_step(self, step_index: int, sim_time_s: float) -> dict[str, Any]:
        frames: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for name, camera in self.cameras.items():
            image = camera.get_rgb(device="cpu")
            if image is None:
                missing.append(name)
                continue
            image = np.asarray(image)
            if image.ndim == 4:
                image = image[0]
            if image.ndim != 3 or image.shape[-1] < 3:
                missing.append(name)
                continue
            image = image[..., :3]
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            frames[name] = image.copy()
        return {
            "step_index": int(step_index),
            "timestamp": float(sim_time_s),
            "frames": frames,
            "missing": missing,
            "aligned": len(missing) == 0,
        }
