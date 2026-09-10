from __future__ import annotations

from typing import Any

import numpy as np


def _matrix_numpy(matrix: Any) -> np.ndarray:
    return np.asarray(
        [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def _look_at_row_rotation(camera_position: np.ndarray, target_position: np.ndarray) -> np.ndarray:
    """Return a USD row-vector camera basis with local -Z looking at target."""

    forward = np.asarray(target_position, dtype=np.float64) - np.asarray(camera_position, dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1.0e-9)
    up_reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_reference))) > 0.96:
        up_reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, up_reference)
    right /= max(float(np.linalg.norm(right)), 1.0e-9)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1.0e-9)
    back = -forward
    return np.vstack([right, up, back])


def _quat_from_row_rotation(row_rotation: np.ndarray) -> np.ndarray:
    """Convert a USD row-basis rotation to Isaac's ``[w,x,y,z]`` quaternion."""

    # Quaternion conversion formulas use the usual column-vector matrix.
    matrix = np.asarray(row_rotation, dtype=np.float64).T
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    return quaternion / max(float(np.linalg.norm(quaternion)), 1.0e-9)


def _row_rotation_from_quaternion(quaternion: Any) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    column_rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return column_rotation.T


def configure_camera_views(
    stage: Any,
    camera_paths: dict[str, str],
    target_position: Any,
    camera_objects: dict[str, Any] | None = None,
    live_parent_poses: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    camera_parent_local_offsets: dict[str, Any] | None = None,
) -> None:
    """Aim the existing cameras at the fixed task workspace in Session Layer.

    Camera prims remain the ones authored by the supplied USD.  Only their
    local rotation is overridden at runtime so the imported camera axes do not
    point into empty space after the robot is repositioned near T01.
    """

    from pxr import Sdf, Usd, UsdGeom

    target = np.asarray(target_position, dtype=np.float64)
    cache = UsdGeom.XformCache()
    session_layer = stage.GetSessionLayer()
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(Usd.EditTarget(session_layer))
    try:
        for name, path in camera_paths.items():
            camera_prim = stage.GetPrimAtPath(path)
            if not camera_prim or not camera_prim.IsValid():
                raise RuntimeError(f"camera prim is invalid: {name}={path}")
            local_xform = UsdGeom.Xformable(camera_prim)
            local_matrix = local_xform.GetLocalTransformation()
            local = _matrix_numpy(local_matrix)
            if live_parent_poses is not None and name in live_parent_poses:
                parent_position, parent_rotation = live_parent_poses[name]
                parent_position = np.asarray(parent_position, dtype=np.float64)
                parent_rotation = np.asarray(parent_rotation, dtype=np.float64)
                # Use the authored camera-to-parent offset, not the local
                # matrix returned by GetLocalTransformation().  Imported
                # camera prims can have inherited/ordered xform ops for which
                # that method is not a plain parent-local matrix.
                parent_path = Sdf.Path(path).GetParentPath()
                parent_world = _matrix_numpy(
                    cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path))
                )
                camera_world = _matrix_numpy(cache.GetLocalToWorldTransform(camera_prim))
                parent_static_rotation = parent_world[:3, :3]
                local_translation = (
                    camera_world[3, :3] - parent_world[3, :3]
                ) @ parent_static_rotation.T
                if camera_parent_local_offsets and name in camera_parent_local_offsets:
                    local_translation = np.asarray(
                        camera_parent_local_offsets[name], dtype=np.float64
                    ).reshape(3)
                camera_position = local_translation @ parent_rotation + parent_position
            else:
                parent_path = Sdf.Path(path).GetParentPath()
                parent_matrix = _matrix_numpy(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path)))
                world_matrix = _matrix_numpy(cache.GetLocalToWorldTransform(camera_prim))
                camera_position = world_matrix[3, :3]
                parent_rotation = parent_matrix[:3, :3]
                local_translation = local[3, :3]
            desired_world = _look_at_row_rotation(camera_position, target)
            desired_local = desired_world @ parent_rotation.T

            # Replace only the camera's local transform in the session layer;
            # preserve the original camera attributes and the authored local
            # translation.  The local translation is recomputed above only
            # for the runtime camera-position calculation.
            local_xform.ClearXformOpOrder()
            transform_op = local_xform.AddTransformOp()
            from pxr import Gf

            new_local = Gf.Matrix4d(1.0)
            for row in range(3):
                for column in range(3):
                    new_local[row][column] = float(desired_local[row, column])
            for column in range(3):
                new_local[3][column] = float(local_translation[column])
            new_local[3][3] = 1.0
            transform_op.Set(new_local)
            cache.Clear()
            final_world = _matrix_numpy(cache.GetLocalToWorldTransform(camera_prim))
            actual_forward = -final_world[2, :3]
            print(
                f"R1 runner: camera view aligned name={name} target="
                f"({target[0]:.3f},{target[1]:.3f},{target[2]:.3f}) "
                f"camera=({camera_position[0]:.3f},{camera_position[1]:.3f},{camera_position[2]:.3f}) "
                f"forward=({actual_forward[0]:.3f},{actual_forward[1]:.3f},{actual_forward[2]:.3f})",
                flush=True,
            )
    finally:
        stage.SetEditTarget(previous_edit_target)


def ensure_fixed_dataset_lighting(stage: Any, target_position: Any | None = None) -> None:
    """Add fixed session-only lighting for camera renders.

    The imported ``spacerobot.usd`` contains no UsdLux light prims.  The
    viewport can still look usable because it has its own preview lighting,
    but camera render products then contain almost-black images.  These
    lights are authored into the current session layer only; the source USD
    is never saved or modified.
    """

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    target = np.asarray(
        target_position if target_position is not None else [-3.75, 3.85, 0.70],
        dtype=np.float64,
    ).reshape(3)

    session_layer = stage.GetSessionLayer()
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(Usd.EditTarget(session_layer))
    try:
        root = Sdf.Path("/World/R1DatasetLighting")
        if not stage.GetPrimAtPath(root):
            stage.DefinePrim(root, "Scope")

        dome_path = root.AppendChild("Dome")
        dome = UsdLux.DomeLight.Define(stage, dome_path)
        dome.CreateIntensityAttr(100.0)
        dome.CreateExposureAttr(0.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        key_path = root.AppendChild("Key")
        key = UsdLux.DistantLight.Define(stage, key_path)
        key.CreateIntensityAttr(500.0)
        key.CreateAngleAttr(0.5)
        key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))
        UsdGeom.XformCommonAPI(key.GetPrim()).SetRotate(Gf.Vec3f(-35.0, -25.0, 25.0))

        # The imported environment has no authored UsdLux lights.  Local
        # session-only sphere lights make the robot and T01 readable even
        # when the station textures are unavailable or unlit in headless RTX.
        light_specs = [
            ("PayloadFill", target + np.asarray([0.0, 0.0, 2.5]), 5000.0, 1.0),
            ("PayloadFront", target + np.asarray([1.5, -1.5, 1.2]), 2500.0, 0.6),
            ("PayloadBack", target + np.asarray([-1.0, 1.0, 1.0]), 1500.0, 0.6),
        ]
        for name, position, intensity, radius in light_specs:
            light_path = root.AppendChild(name)
            sphere = UsdLux.SphereLight.Define(stage, light_path)
            sphere.CreateIntensityAttr(float(intensity))
            sphere.CreateRadiusAttr(float(radius))
            sphere.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(
                Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
            )
    finally:
        stage.SetEditTarget(previous_edit_target)


def set_gui_task_view(eye: Any, target: Any) -> None:
    """Move only the Kit default viewport camera for a lightweight preview.

    This uses Isaac Sim's installed viewport helper and does not create a
    Replicator render product, so GUI inspection does not compete with the
    three dataset cameras for GPU memory.
    """

    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64))


def live_camera_parent_poses(
    stage: Any,
    camera_paths: dict[str, str],
    robot_path: str,
    root_pose: tuple[Any, Any],
    eef_poses: dict[str, tuple[Any, Any]],
    link_poses: dict[str, tuple[Any, Any]] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return runtime parent poses for the existing front/wrist cameras.

    ``link_poses`` contains live PhysX poses for non-EEF camera parents, for
    example ``torso_link4``. The root-plus-static-link reconstruction remains
    a fallback for callers that do not provide it.
    """

    from pxr import Sdf, UsdGeom

    cache = UsdGeom.XformCache()
    root_position = np.asarray(root_pose[0], dtype=np.float64)
    root_rotation = _row_rotation_from_quaternion(root_pose[1])
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, camera_path in camera_paths.items():
        parent_path = str(Sdf.Path(camera_path).GetParentPath())
        if name.startswith("left"):
            pose = eef_poses["left"]
            result[name] = (
                np.asarray(pose[0], dtype=np.float64),
                _row_rotation_from_quaternion(pose[1]),
            )
            continue
        if name.startswith("right"):
            pose = eef_poses["right"]
            result[name] = (
                np.asarray(pose[0], dtype=np.float64),
                _row_rotation_from_quaternion(pose[1]),
            )
            continue

        if link_poses and "front" in link_poses:
            pose = link_poses["front"]
            result[name] = (
                np.asarray(pose[0], dtype=np.float64),
                _row_rotation_from_quaternion(pose[1]),
            )
            continue

        root_static = _matrix_numpy(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(robot_path)))
        parent_static = _matrix_numpy(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent_path)))
        parent_local_rotation = parent_static[:3, :3] @ root_static[:3, :3].T
        parent_local_position = (parent_static[3, :3] - root_static[3, :3]) @ root_static[:3, :3].T
        result[name] = (
            parent_local_position @ root_rotation + root_position,
            parent_local_rotation @ root_rotation,
        )
    return result
