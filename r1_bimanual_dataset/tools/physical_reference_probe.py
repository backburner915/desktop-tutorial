"""Physical fixed-base R1 probe for the corrected ``spacerobot.usd`` scene.

This script deliberately does *not* use the old dataset runner's root-pose
lock, collision filtering, or payload attachment. It places the corrected
scene's existing R1 through the USD session layer, creates a real PhysX fixed
root before simulation starts, and applies the high-PD values from the local
Galaxea reference configuration. It is the required first gate before
implementing a physical bimanual grasp trajectory.

The source USD is never saved or modified.  Closing Isaac Sim discards all
session-layer edits.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import load_scene_config
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config


def _register_local_galaxea_extensions() -> None:
    """Expose the checked-out Galaxea Isaac Lab extensions to Isaac Sim Python.

    The workstation's ``D:\\isaaclab_env`` Python distribution provides Isaac
    Sim but does not install the development extensions as wheels.  The
    corrected scene and the Galaxea source checkout are both local, so adding
    these package roots is equivalent to the project's normal ``isaaclab.sh``
    launcher and needs no package installation.
    """

    extension_root = Path(r"D:\Galaxea_Lab-galaxea-main\Galaxea_Lab-galaxea-main\source\extensions")
    required = (
        extension_root / "omni.isaac.lab",
        extension_root / "omni.isaac.lab_assets",
    )
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise RuntimeError(f"local Galaxea Isaac Lab extensions are missing: {missing}")
    for path in reversed(required):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _world_bbox(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the target's world-aligned visual bound without stepping physics."""

    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"missing target prim: {prim_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    bound = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    return np.asarray(bound.GetMin(), dtype=np.float64), np.asarray(bound.GetMax(), dtype=np.float64)


def _rotation_wxyz_from_forward(forward_xy: np.ndarray) -> np.ndarray:
    """Return a world-Z yaw rotation mapping R1 local +X to ``forward_xy``."""

    forward = np.asarray(forward_xy, dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    yaw = math.atan2(float(forward[1]), float(forward[0]))
    return np.asarray([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def select_long_edge_pose(
    object_min: np.ndarray, object_max: np.ndarray, *, side: str, stand_off_m: float
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Select an interpretable fixed-base pose beside T01's long table edge.

    R1's nominal manipulation setup has local +X pointing from the base toward
    the table and the two arms split across local Y.  For T01, the wider bbox
    axis is used as the bimanual (left/right) direction, and the base is placed
    on the perpendicular side.  This keeps both wrists facing the object
    rather than placing the robot at an arbitrary world-space offset.
    """

    center = (object_min + object_max) / 2.0
    dimensions = object_max - object_min
    lateral_axis = int(np.argmax(dimensions[:2]))
    forward_axis = 1 - lateral_axis
    sign = -1.0 if side == "near" else 1.0
    forward = np.zeros(2, dtype=np.float64)
    forward[forward_axis] = -sign  # base -> object
    position = center.copy()
    position[:2] -= forward * float(stand_off_m)
    # The R1 asset's root rests on the world ground plane at z=0.
    position[2] = 0.0
    rotation = _rotation_wxyz_from_forward(forward)
    report = {
        "object_center": np.round(center, 4).tolist(),
        "object_dimensions": np.round(dimensions, 4).tolist(),
        "lateral_bbox_axis": "x" if lateral_axis == 0 else "y",
        "base_forward_axis": "x" if forward_axis == 0 else "y",
        "side": side,
        "stand_off_m": float(stand_off_m),
        "base_position": np.round(position, 4).tolist(),
        "base_quaternion_wxyz": np.round(rotation, 6).tolist(),
    }
    return position, rotation, report


def _open_stage(app, stage_path: str, *, warmup_updates: int = 120):
    import omni.timeline
    import omni.usd

    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    context = omni.usd.get_context()
    if not context.open_stage(stage_path):
        raise RuntimeError(f"cannot open stage: {stage_path}")
    stage = None
    for step in range(2400):
        app.update()
        stage = context.get_stage()
        if stage is not None:
            break
        if step and step % 120 == 0:
            print(f"Physical probe: waiting for stage ({step}/2400)", flush=True)
    if stage is None:
        raise RuntimeError("stage did not become available")
    for _ in range(int(warmup_updates)):
        app.update()
    timeline.stop()
    return stage, timeline


def _source_base_link_world_translation(stage, robot_path: str) -> np.ndarray:
    """Return the authored base-link world offset before Session placement."""

    from pxr import UsdGeom

    base_link = stage.GetPrimAtPath(f"{robot_path}/base_link")
    if not base_link or not base_link.IsValid():
        raise RuntimeError(f"missing R1 base_link under {robot_path}")
    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(base_link).ExtractTranslation(), dtype=np.float64)


def _apply_session_layer(stage, source_robot_path: str) -> None:
    """Deactivate only the authored floating R1 in a non-persistent layer."""

    from pxr import Usd

    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    source = stage.GetPrimAtPath(source_robot_path)
    if not source or not source.IsValid():
        raise RuntimeError(f"authored R1 prim is missing: {source_robot_path}")
    source.SetActive(False)
    print(f"Physical probe: deactivated authored floating robot in Session Layer: {source_robot_path}", flush=True)


def _place_existing_robot_in_session(stage, robot_path: str, position: np.ndarray, rotation_wxyz: np.ndarray) -> None:
    """Move the existing R1 Xform before physics is parsed (Session Layer only)."""

    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(robot_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"missing R1 xform: {robot_path}")
    # The corrected USD's R1 parent is under /World, so this local transform
    # is also its world placement.  This happens before the fixed joint is
    # created; it is not an after-the-fact teleport or runtime root lock.
    yaw = math.atan2(2.0 * (rotation_wxyz[0] * rotation_wxyz[3]), 1.0 - 2.0 * rotation_wxyz[3] ** 2)
    xformable = UsdGeom.Xformable(prim)
    # The referenced asset has a non-common transform stack.  Preserve it and
    # append only Session-Layer ops; replacing its stack can invalidate rigid
    # body transforms.  The probe supports explicit calibration poses below
    # so we can solve this composition safely for this particular USD.
    translate = xformable.AddTranslateOp(opSuffix="physical_reference", precision=UsdGeom.XformOp.PrecisionDouble)
    rotate = xformable.AddRotateZOp(opSuffix="physical_reference", precision=UsdGeom.XformOp.PrecisionFloat)
    translate.Set(Gf.Vec3d(*[float(value) for value in position]))
    rotate.Set(float(math.degrees(yaw)))
    print(f"Physical probe: authored-session R1 placement={np.round(position, 4).tolist()} yaw_deg={math.degrees(yaw):.2f}", flush=True)


def _make_existing_r1_fixed(stage, robot_path: str, *, relocate_articulation_root: bool = True) -> str:
    """Create a real world fixed joint for the corrected USD's R1 before PhysX starts."""

    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    articulation_roots = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(robot_path) and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    rigid_links = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(robot_path) and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if not articulation_roots:
        raise RuntimeError("the corrected-scene R1 has no ArticulationRootAPI")
    if not rigid_links:
        raise RuntimeError("the corrected-scene R1 has no RigidBodyAPI link")
    print(
        "Physical probe: topology "
        f"articulation_roots={[str(item.GetPath()) for item in articulation_roots]} "
        f"first_rigid_links={[str(item.GetPath()) for item in rigid_links[:4]]}",
        flush=True,
    )
    articulation_root = articulation_roots[0]
    # The corrected R1 is authored with ArticulationRootAPI on an outer Xform
    # and the actual root rigid body one level below at base_link.  PhysX only
    # parses this as a fixed-base articulation if we move that API *above the
    # existing articulation Xform*, not merely above base_link.
    root_link = articulation_root if articulation_root.HasAPI(UsdPhysics.RigidBodyAPI) else rigid_links[0]
    fixed_root_owner = articulation_root.GetParent()
    if not fixed_root_owner or not fixed_root_owner.IsValid():
        raise RuntimeError(f"cannot create fixed root above {root_link.GetPath()}")

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, root_link.GetPath().AppendChild("PhysicalReferenceFixedJoint"))
    # An empty body0 relationship denotes the world frame; body1 is the
    # actual rigid base link. This is a PhysX constraint, not a kinematic
    # update or an articulation-state lock.
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(root_link.GetPath())])
    # A world joint must be anchored at the root link's current world pose.
    # Leaving these anchors at zero creates a valid joint in USD but makes
    # PhysX solve the base at the world origin instead of its table-side pose.
    world_pose = UsdGeom.XformCache().GetLocalToWorldTransform(root_link).RemoveScaleShear()
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(world_pose.ExtractTranslation()))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(world_pose.ExtractRotationQuat()))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    fixed_joint.CreateBreakForceAttr().Set(3.40282347e38)
    fixed_joint.CreateBreakTorqueAttr().Set(3.40282347e38)

    if not relocate_articulation_root:
        print("Physical probe: keeping original articulation root for Isaac Sim 5.1 Jacobian compatibility", flush=True)
        return str(articulation_root.GetPath())

    UsdPhysics.ArticulationRootAPI.Apply(fixed_root_owner)
    if "PhysxArticulationAPI" not in fixed_root_owner.GetAppliedSchemas():
        fixed_root_owner.AddAppliedSchema("PhysxArticulationAPI")
    # Preserve every authored articulation solver setting while moving the
    # parser marker; the source USD remains untouched because this is Session.
    for attr_name in UsdPhysics.ArticulationRootAPI(articulation_root).GetSchemaAttributeNames():
        source_attr = articulation_root.GetAttribute(attr_name)
        if source_attr and source_attr.Get() is not None:
            target_attr = fixed_root_owner.GetAttribute(attr_name)
            if not target_attr:
                target_attr = fixed_root_owner.CreateAttribute(attr_name, source_attr.GetTypeName())
            target_attr.Set(source_attr.Get())
    for source_attr in articulation_root.GetAttributes():
        attr_name = source_attr.GetName()
        if attr_name.startswith("physxArticulation:") and source_attr.Get() is not None:
            target_attr = fixed_root_owner.GetAttribute(attr_name)
            if not target_attr:
                target_attr = fixed_root_owner.CreateAttribute(attr_name, source_attr.GetTypeName())
            target_attr.Set(source_attr.Get())
    articulation_root.RemoveAppliedSchema("PhysxArticulationAPI")
    articulation_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    print(
        f"Physical probe: relocated articulation root API {articulation_root.GetPath()} -> {fixed_root_owner.GetPath()}",
        flush=True,
    )
    return str(fixed_root_owner.GetPath())


def main() -> int:
    parser = argparse.ArgumentParser(description="Spawn a true fixed-base Galaxea R1 beside T01")
    parser.add_argument("--scene-config", type=Path, default=ROOT / "scene_config.json")
    parser.add_argument("--gui", action="store_true", help="show Isaac Sim for visual inspection")
    parser.add_argument("--keep-open", action="store_true", help="keep the GUI open after the stability gate")
    parser.add_argument("--side", choices=("near", "far"), default="near", help="which long table edge receives the base")
    parser.add_argument("--stand-off", type=float, default=0.72, help="base-to-object center distance in metres")
    parser.add_argument("--base-position", nargs=3, type=float, metavar=("X", "Y", "Z"), help="calibration-only local placement override")
    parser.add_argument("--yaw-deg", type=float, help="calibration-only yaw override (requires --base-position)")
    parser.add_argument("--settle-seconds", type=float, default=3.0, help="physical settling duration")
    parser.add_argument("--keep-articulation-root", action="store_true", help="retain original root API to preserve Isaac Sim 5.1 Jacobians")
    args = parser.parse_args()
    if args.stand_off < 0.45:
        raise ValueError("--stand-off must be at least 0.45 m to keep the fixed base clear of the support")

    scene_cfg = load_scene_config(args.scene_config)
    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=not args.gui))
    try:
        print("Physical probe: opening corrected stage", flush=True)
        stage, timeline = _open_stage(app, scene_cfg["stage_path"])
        target_min, target_max = _world_bbox(stage, scene_cfg["target_object"])
        base_position, base_rotation, placement = select_long_edge_pose(
            target_min, target_max, side=args.side, stand_off_m=args.stand_off
        )
        # ``base_position`` is the desired physical base-link XY location.
        # The referenced robot carries its own non-zero base-link transform;
        # compensate it dynamically before appending the Session placement.
        source_base_offset = _source_base_link_world_translation(stage, scene_cfg["robot_prim"])
        desired_base_link = base_position.copy()
        # ``base_position.z`` denotes the wheel-ground support plane.  The
        # R1's rigid base_link is 0.3689 m above it; preserve that authored
        # height instead of driving the wheel assembly through the ground.
        desired_base_link[2] = source_base_offset[2]
        placement_request = desired_base_link - source_base_offset
        placement["source_base_link_offset"] = np.round(source_base_offset, 4).tolist()
        placement["session_placement_translation"] = np.round(placement_request, 4).tolist()
        if args.base_position is not None:
            if args.yaw_deg is None:
                raise ValueError("--yaw-deg is required with --base-position")
            base_position = np.asarray(args.base_position, dtype=np.float64)
            placement_request = base_position
            yaw_rad = math.radians(args.yaw_deg)
            base_rotation = np.asarray([math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)])
            placement["calibration_override"] = True
            placement["base_position"] = np.round(base_position, 4).tolist()
            placement["base_quaternion_wxyz"] = np.round(base_rotation, 6).tolist()
        print(f"Physical probe: selected long-edge placement: {placement}", flush=True)
        _place_existing_robot_in_session(stage, scene_cfg["robot_prim"], placement_request, base_rotation)
        fixed_root_path = _make_existing_r1_fixed(
            stage, scene_cfg["robot_prim"], relocate_articulation_root=not args.keep_articulation_root
        )
        from pxr import UsdGeom
        prephysics_link = stage.GetPrimAtPath(f"{scene_cfg['robot_prim']}/base_link")
        prephysics_world = UsdGeom.XformCache().GetLocalToWorldTransform(prephysics_link).ExtractTranslation()
        print(f"Physical probe: prephysics base_link world={np.round(np.asarray(prephysics_world), 4).tolist()}", flush=True)

        # Use only APIs supplied by the current Isaac Sim 5.1 installation.
        # The older external Galaxea Lab checkout cannot be imported in this
        # Python environment because its Torch/TensorDict binary ABI differs.
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.viewports import set_camera_view
        from r1_bimanual_dataset.core.robot_interface import RobotInterface

        set_camera_view(
            base_position + np.asarray([-1.8, -1.8, 1.35]),
            (target_min + target_max) / 2.0 + np.asarray([0.0, 0.0, 0.25]),
        )
        timeline.play()
        SimulationManager.initialize_physics()
        for _ in range(3):
            app.update()
        timeline.stop()
        physical_cfg = dict(scene_cfg)
        physical_cfg.update(
            {
                "robot_prim": fixed_root_path,
                "base_movement": "disabled",
                "runtime_base_lock": False,
                "runtime_joint_state_lock": False,
                "physical_rollout": True,
                "scripted_grasp_attachment": False,
            }
        )
        robot = RobotInterface(physical_cfg)
        robot.initialize()
        # Isaac Sim 5.1's Articulation wrapper does not refresh this flag after
        # a Session-Layer topology edit.  The actual gate below is physical
        # root drift while gravity is enabled; do not reject a valid PhysX
        # world joint solely because the Python convenience flag is stale.
        wrapper_fixed = bool(getattr(robot.robot, "is_fixed_base", False))
        if not wrapper_fixed:
            print(
                f"Physical probe: wrapper still reports floating-base at {fixed_root_path}; validating actual root drift.",
                flush=True,
            )

        initial_root, _ = robot.root_pose()
        default_q = robot.full_positions()
        robot.apply_full_target(default_q)
        print(
            "Physical probe: PASS setup "
            f"fixed_base_wrapper={wrapper_fixed} joints={robot.dof_count} root={np.round(initial_root, 4).tolist()}",
            flush=True,
        )

        timeline.play()
        dt = float(scene_cfg.get("physics_dt_fallback", 1.0 / 60.0))
        steps = max(1, int(round(args.settle_seconds / dt)))
        max_root_drift = 0.0
        for _ in range(steps):
            robot.apply_full_target(default_q)
            app.update()
            drift = float(np.linalg.norm(robot.root_pose()[0] - initial_root))
            max_root_drift = max(max_root_drift, drift)
        timeline.stop()
        if max_root_drift > 1e-4:
            raise RuntimeError(f"fixed base drifted {max_root_drift:.6f} m during settle")
        print(
            f"Physical probe: STABILITY PASS after {args.settle_seconds:.2f}s; root drift={max_root_drift:.8f} m. "
            "No collision filtering, root locking, or payload attachment was used.",
            flush=True,
        )
        if args.keep_open and args.gui:
            print("Physical probe: GUI remains open. Inspect R1 at the long table edge, then close Isaac Sim.", flush=True)
            while app.is_running():
                app.update()
        return 0
    except Exception as exc:
        # SimulationApp's fast shutdown otherwise swallows a Python exception
        # after it has closed the USD stage.  Preserve the exact version/path
        # diagnostic so this probe never silently looks like a physics crash.
        print(f"Physical probe: FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
