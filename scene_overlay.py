"""Session-layer composition for the physical R1 grasp reference.

Nothing in this module saves or flattens the user's source stage.  In
particular, it never moves the authored table top: the table's four authored
legs remain physically aligned with its collision surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUGAR_BOX_SOURCE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"
)
# Isaac Sim's USD resolver cannot cache remote S3 references on this Windows
# workstation even though the URL itself is reachable.  These are unmodified
# official companion USDs downloaded into the project with their required
# relative layout (``assets/..`` -> ``Axis_Aligned/..``).
SUGAR_BOX_PHYSICS_PATH = str(
    Path(__file__).resolve().parents[1] / "assets" / "004_sugar_box_physics.usd"
)
# This project already vendors a small collection of Isaac Lab task assets.
# Keeping the physical reference local avoids an unrepeatable network/Nucleus
# dependency and, unlike the previous cube, SmallKLT has two long exterior
# sides that can be held cooperatively by R1's two grippers.
SMALL_KLT_PHYSICS_PATH = str(
    Path("D:/Galaxea_Lab-galaxea-main/Galaxea_Lab-galaxea-main")
    / "source/extensions/omni.isaac.lab_assets/data/Props/KLT_Bin/small_KLT.usd"
)
OVERLAY_ROOT = "/World/R1DatasetOverlay"
TARGET_POSE_PATH = f"{OVERLAY_ROOT}/TargetPose"
TARGET_PATH = f"{TARGET_POSE_PATH}/PhysicalTarget"
TARGET_STAND_PATH = f"{OVERLAY_ROOT}/TargetStand"


@dataclass(frozen=True)
class TargetOverlay:
    source_target_path: str
    target_path: str
    pose_path: str
    asset_url: str
    table_path: str
    support_path: str
    source_target_center: np.ndarray
    support_top_z: float


def _set_translate(xformable: Any, value: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    op = next((item for item in xformable.GetOrderedXformOps() if item.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if op is None:
        op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    op.Set(Gf.Vec3d(*[float(item) for item in value]))


def _set_orient(xformable: Any, quaternion_wxyz: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    op = next((item for item in xformable.GetOrderedXformOps() if item.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    if op is None:
        op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    op.Set(
        Gf.Quatf(
            float(quaternion_wxyz[0]),
            Gf.Vec3f(*[float(value) for value in quaternion_wxyz[1:]]),
        )
    )


def _yaw_quaternion(yaw_rad: float) -> np.ndarray:
    return np.asarray(
        [np.cos(float(yaw_rad) / 2.0), 0.0, 0.0, np.sin(float(yaw_rad) / 2.0)],
        dtype=np.float64,
    )


def _set_uniform_scale(xformable: Any, scale: float) -> None:
    from pxr import Gf, UsdGeom

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("target_uniform_scale must be a finite positive number")
    op = next((item for item in xformable.GetOrderedXformOps() if item.GetOpType() == UsdGeom.XformOp.TypeScale), None)
    if op is None:
        op = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat)
    op.Set(Gf.Vec3f(float(scale), float(scale), float(scale)))


def add_physical_target_overlay(stage: Any, scene_cfg: dict[str, Any]) -> TargetOverlay:
    """Replace only the authored source target with a physical task asset.

    The object is initially authored above the unchanged table.  Call
    :func:`settle_target_on_support` after the reference is resolved to align
    its actual collision/visual bounding box with the source table top.
    """

    from pxr import Sdf, Usd, UsdGeom

    source_target_path = str(scene_cfg["target_object"])
    support_path = str(scene_cfg.get("table_top", "/World/TaskSetup/Fixtures/StorageRack/Top"))
    from .reference_policy import require_candidate
    require_candidate(scene_cfg.get("target_asset_url"))
    asset_url = str(scene_cfg["target_asset_url"])
    source_target = stage.GetPrimAtPath(source_target_path)
    support = stage.GetPrimAtPath(support_path)
    if not source_target or not source_target.IsValid():
        raise RuntimeError(f"source target is missing: {source_target_path}")
    if not support or not support.IsValid():
        raise RuntimeError(f"table support is missing: {support_path}")

    from r1_bimanual_dataset.tools.physical_reference_probe import _world_bbox

    source_min, source_max = _world_bbox(stage, source_target_path)
    table_min, table_max = _world_bbox(stage, support_path)
    source_center = (source_min + source_max) / 2.0
    # A bimanual reference must be centred in the two-arm workspace, not at
    # whichever corner happened to contain the source cube.  The optional
    # value changes only this new overlay; the authored T01 and its support
    # remain untouched.  Per-episode ScenarioConfig will later supply the
    # same field through a bounded sampler.
    requested_xy = scene_cfg.get("target_spawn_xy")
    if requested_xy is None:
        spawn_center = source_center.copy()
    else:
        requested_xy = np.asarray(requested_xy, dtype=np.float64).reshape(2)
        spawn_center = source_center.copy()
        spawn_center[:2] = requested_xy
        if not np.all(requested_xy >= table_min[:2]) or not np.all(requested_xy <= table_max[:2]):
            raise ValueError(
                f"target_spawn_xy={requested_xy.tolist()} lies outside table top {support_path}"
            )

    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    # T01 remains in the source layer.  Deactivation is a reversible session
    # opinion, so the original scene is unchanged after closing the app.
    source_target.SetActive(False)

    # R1's supplied stationary scene has a 0.58 m table, while the upstream
    # Galaxea R1 lift examples operate around a 1.0 m task height.  A static
    # tabletop stand is a physically valid environmental fixture: it is a
    # collision-enabled cube with no RigidBodyAPI (therefore static), rests on
    # the *unchanged* original table, and is never moved once physics starts.
    # Setting this height to zero preserves the direct-table preview mode.
    stand_height = float(scene_cfg.get("target_stand_height_m", 0.0))
    if stand_height < 0.0:
        raise ValueError("target_stand_height_m must be non-negative")
    if stand_height > 0.0:
        from pxr import Gf, UsdPhysics

        stand_dims = np.asarray(scene_cfg.get("target_stand_dimensions_m", [0.34, 0.30, stand_height]), dtype=np.float64)
        if stand_dims.shape != (3,) or np.any(stand_dims <= 0.0):
            raise ValueError("target_stand_dimensions_m must contain three positive metres")
        stand_dims[2] = stand_height
        requested_stand_xy = scene_cfg.get("target_stand_center_xy")
        stand_center = spawn_center[:2] if requested_stand_xy is None else np.asarray(requested_stand_xy, dtype=np.float64).reshape(2)
        half = stand_dims[:2] / 2.0
        if np.any(stand_center - half < table_min[:2]) or np.any(stand_center + half > table_max[:2]):
            raise ValueError("target stand would overhang the unchanged table top")
        stand = UsdGeom.Cube.Define(stage, TARGET_STAND_PATH)
        stand_prim = stand.GetPrim()
        stand.CreateSizeAttr(1.0)
        stand_xform = UsdGeom.Xformable(stand_prim)
        _set_translate(
            stand_xform,
            np.asarray([stand_center[0], stand_center[1], table_max[2] + stand_height / 2.0]),
        )
        scale_op = next((item for item in stand_xform.GetOrderedXformOps() if item.GetOpType() == UsdGeom.XformOp.TypeScale), None)
        if scale_op is None:
            scale_op = stand_xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat)
        scale_op.Set(Gf.Vec3f(*[float(value) for value in stand_dims]))
        UsdPhysics.CollisionAPI.Apply(stand_prim)
        support_path = TARGET_STAND_PATH
    support_min, support_max = _world_bbox(stage, support_path)

    UsdGeom.Xform.Define(stage, OVERLAY_ROOT)
    pose = UsdGeom.Xform.Define(stage, TARGET_POSE_PATH)
    pose_xform = UsdGeom.Xformable(pose.GetPrim())
    # Initial Z only needs to be above the table.  The resolved asset bbox is
    # used below to place its true bottom on the support surface.
    _set_translate(pose_xform, spawn_center + np.asarray([0.0, 0.0, 0.25 + stand_height]))
    _set_orient(pose_xform, _yaw_quaternion(float(scene_cfg.get("target_yaw_rad", 0.0))))
    _set_uniform_scale(pose_xform, float(scene_cfg.get("target_uniform_scale", 1.0)))

    target = UsdGeom.Xform.Define(stage, TARGET_PATH)
    target_prim = target.GetPrim()
    target_prim.GetReferences().ClearReferences()
    target_prim.GetReferences().AddReference(asset_url, Sdf.Path.emptyPath)
    return TargetOverlay(
        source_target_path=source_target_path,
        target_path=TARGET_PATH,
        pose_path=TARGET_POSE_PATH,
        asset_url=asset_url,
        table_path=str(scene_cfg.get("table_top", "/World/TaskSetup/Fixtures/StorageRack/Top")),
        support_path=support_path,
        source_target_center=spawn_center,
        support_top_z=float(support_max[2]),
    )


def settle_target_on_support(stage: Any, overlay: TargetOverlay, *, clearance_m: float = 0.002) -> dict[str, object]:
    """Align the referenced target's actual bbox bottom to the unchanged table."""

    from pxr import UsdGeom
    from r1_bimanual_dataset.tools.physical_reference_probe import _world_bbox

    lower, upper = _world_bbox(stage, overlay.target_path)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(upper <= lower):
        raise RuntimeError(
            "physical target reference did not resolve to a finite bounding box; "
            "the target is not safe to use."
        )
    delta_z = float(overlay.support_top_z + clearance_m - lower[2])
    pose = UsdGeom.Xformable(stage.GetPrimAtPath(overlay.pose_path))
    translate_op = next(item for item in pose.GetOrderedXformOps() if item.GetOpType() == UsdGeom.XformOp.TypeTranslate)
    current = np.asarray(translate_op.Get(), dtype=np.float64)
    _set_translate(pose, current + np.asarray([0.0, 0.0, delta_z]))
    final_lower, final_upper = _world_bbox(stage, overlay.target_path)
    return {
        "target_path": overlay.target_path,
        "asset_url": overlay.asset_url,
        "asset_source_url": SUGAR_BOX_SOURCE_URL if overlay.asset_url == SUGAR_BOX_PHYSICS_PATH else None,
        "bbox_min_m": final_lower.tolist(),
        "bbox_max_m": final_upper.tolist(),
        "dimensions_m": (final_upper - final_lower).tolist(),
        "table_top_z_m": overlay.support_top_z,
        "bottom_clearance_m": float(final_lower[2] - overlay.support_top_z),
    }


def add_sugar_box_overlay(stage: Any, scene_cfg: dict[str, Any]) -> TargetOverlay:
    """Backward-compatible explicit YCB Sugar Box overlay for old diagnostics.

    New physical reference tools should use :func:`add_physical_target_overlay`
    with SmallKLT.  This alias preserves old experiments without silently
    changing their chosen target.
    """

    sugar_cfg = dict(scene_cfg)
    sugar_cfg.setdefault("target_asset_url", SUGAR_BOX_PHYSICS_PATH)
    return add_physical_target_overlay(stage, sugar_cfg)
