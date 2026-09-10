"""Real-physics R1 bimanual grasp runner for ``spacerobot.usd``.

This is deliberately separate from the legacy dataset runner.  It creates a
Session-Layer fixed-base topology, uses the live 19-DOF articulation and its
DriveAPI targets, and never attaches or teleports the payload during a grasp.
It is first a reachability gate; use ``--execute`` only after that gate passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from r1_bimanual_dataset.config import ScenarioConfig, load_scene_config
from r1_bimanual_dataset.core.launch import prepare_isaac_environment, simulation_app_config
from r1_bimanual_dataset.tools.physical_reference_probe import (
    _make_existing_r1_fixed,
    _open_stage,
    _place_existing_robot_in_session,
    _source_base_link_world_translation,
    _world_bbox,
    select_long_edge_pose,
)


def _move_existing_support_top(
    stage, support_path: str, object_center: np.ndarray, object_dimensions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move the authored StorageRack top in the Session Layer.

    The corrected USD already contains a working static collision body at
    this path. Reusing it is reliable for this imported stage and avoids
    depending on runtime collider cooking. The source USD is never saved.
    """

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(support_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"missing authored support collider: {support_path}")
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"support prim is not a collision body: {support_path}")
    source_min, source_max = _world_bbox(stage, support_path)
    source_center = (source_min + source_max) / 2.0
    source_height = float(source_max[2] - source_min[2])
    object_half_height = float(object_dimensions[2]) / 2.0
    target_top_z = float(object_center[2] - object_half_height)
    target_center = np.asarray(object_center, dtype=np.float64).copy()
    target_center[2] = target_top_z - source_height / 2.0
    delta_world = target_center - source_center

    # Keep the parent's authored non-uniform scale. StorageRack uses that
    # scale for the authored top dimensions; removing it would turn the
    # replacement into a unit cube.
    parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
    # The source prim is a Cube whose authored dimensions are carried by its
    # world extent rather than by the matrix returned from the XformCache.
    # Rebuild the same axis-aligned rectangular collider explicitly while
    # changing only its world center.
    source_dimensions = source_max - source_min
    scale_matrix = Gf.Matrix4d(1.0)
    scale_matrix.SetScale(Gf.Vec3d(*[float(value) for value in source_dimensions]))
    translate_matrix = Gf.Matrix4d(1.0)
    translate_matrix.SetTranslate(Gf.Vec3d(*[float(value) for value in target_center]))
    # Compose the two matrices instead of calling SetScale/SetTranslate on
    # the same matrix (each setter resets the other component in Gf).  USD's
    # Gf matrices use row-vector composition here, so scale * translate keeps
    # the requested centre in world coordinates; translate * scale would also
    # scale the translation and move the support far from T01.
    target_world = scale_matrix * translate_matrix
    desired_local = parent_world.GetInverse() * target_world
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    xformable = UsdGeom.Xformable(prim)
    # This prim's authored stack contains more than a simple translate.  A
    # session override on the prim's local matrix preserves the authored
    # collider geometry while making its world pose unambiguous.
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp(opSuffix="physical_workspace", precision=UsdGeom.XformOp.PrecisionDouble).Set(
        desired_local
    )
    print(f"Physical grasp: support desired_local_matrix={desired_local}", flush=True)
    moved_min, moved_max = _world_bbox(stage, support_path)
    print(
        "Physical grasp: reused authored support collider in Session Layer "
        f"path={support_path} target_top_z={target_top_z:.4f}m "
        f"bbox={np.round(moved_min, 4).tolist()}..{np.round(moved_max, 4).tolist()} "
        f"collision_api={prim.HasAPI(UsdPhysics.CollisionAPI)}",
        flush=True,
    )
    return moved_min, moved_max


def _ensure_runtime_target_pose_ops(stage, target_path: str) -> None:
    """Repair the target's malformed pose stack in the Session Layer.

    T01 in the supplied USD has an ``xformOp:orient`` property with an empty
    type name.  Isaac Sim 5.1's USD-backed RigidPrim wrapper cannot write a
    world pose through that malformed authored property.  Rebuild the target's
    equivalent current local transform as a standard typed translate/orient/
    scale stack in the non-persistent Session Layer; no mesh, rigid-body,
    collision, or articulation topology is changed.
    """

    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(target_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"missing target prim: {target_path}")
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    cache = UsdGeom.XformCache()
    parent = prim.GetParent()
    parent_world = (
        cache.GetLocalToWorldTransform(parent).RemoveScaleShear()
        if parent and parent.IsValid()
        else Gf.Matrix4d(1.0)
    )
    current_world = cache.GetLocalToWorldTransform(prim)
    local = parent_world.GetInverse() * current_world
    translation = local.ExtractTranslation()
    rotation = local.ExtractRotationQuat()
    scale = Gf.Transform(local).GetScale()

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
    scale_op = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat)
    translate_op.Set(Gf.Vec3d(translation))
    orient_op.Set(
        Gf.Quatf(
            float(rotation.GetReal()),
            Gf.Vec3f(*[float(value) for value in rotation.GetImaginary()]),
        )
    )
    scale_op.Set(Gf.Vec3f(*[float(value) for value in scale]))
    print(
        f"Physical grasp: target pose stack rebuilt at {target_path} "
        f"translation={np.round(np.asarray(translation), 4).tolist()} "
        f"scale={np.round(np.asarray(scale), 4).tolist()}",
        flush=True,
    )


def _add_kinematic_workspace_support(
    stage, object_center: np.ndarray, object_dimensions: np.ndarray, support_dimensions: np.ndarray
):
    """Create a real kinematic rigid support after the stage physics is live."""

    from isaacsim.core.api.objects import DynamicCuboid
    from pxr import Sdf, Usd, UsdPhysics

    object_half_height = float(object_dimensions[2]) / 2.0
    support_height = float(support_dimensions[2])
    support_center = np.asarray(object_center, dtype=np.float64).copy()
    support_center[2] = float(object_center[2]) - object_half_height - support_height / 2.0
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    support = DynamicCuboid(
        prim_path="/World/R1SessionWorkspaceKinematicSupport",
        name="r1_session_workspace_kinematic_support",
        position=support_center,
        orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        # DynamicCuboid's newly-authored cube applies the requested scale to
        # both extent and xform; sqrt preserves the requested live dimensions.
        scale=np.sqrt(np.asarray(support_dimensions, dtype=np.float64)),
        size=1.0,
        mass=1000.0,
        visible=True,
    )
    kinematic_attr = support.prim.GetAttribute("physics:kinematicEnabled")
    if not kinematic_attr.IsValid():
        kinematic_attr = support.prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool)
    kinematic_attr.Set(True)
    support.initialize()
    support.set_collision_enabled(True)
    support.set_collision_approximation("boundingCube")
    print(
        "Physical grasp: created kinematic workspace support "
        f"center={np.round(support_center, 4).tolist()} dimensions={np.round(support_dimensions, 4).tolist()} "
        f"collision={support.prim.HasAPI(UsdPhysics.CollisionAPI)} kinematic={kinematic_attr.Get()}",
        flush=True,
    )
    return support, support_center


def _numeric_ik(robot, app, side: str, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Position-only DLS IK from measured live EEF poses.

    The imported fixed-root R1 has a valid physical articulation but Isaac
    Sim 5.1 exposes an all-zero tensor Jacobian after the parser's required
    root relocation.  This finite-difference Jacobian uses the exact same
    articulation forward kinematics, so it stays physical and version-local.
    """
    q = np.asarray(seed, dtype=np.float64).copy()
    indices = np.asarray(robot.arm_indices[side], dtype=np.int64)
    lower, upper = robot.limits()
    eps, damping = 0.004, 0.025
    for _ in range(70):
        robot.robot.set_joint_positions(q.reshape(1, -1))
        app.update()
        current, _ = robot.eef_tip_pose(side)
        error = np.asarray(target, dtype=np.float64) - current
        if float(np.linalg.norm(error)) <= 0.018:
            return q
        jacobian = np.zeros((3, len(indices)), dtype=np.float64)
        for column, joint_index in enumerate(indices):
            trial = q.copy()
            trial[joint_index] = np.clip(trial[joint_index] + eps, lower[joint_index], upper[joint_index])
            robot.robot.set_joint_positions(trial.reshape(1, -1))
            app.update()
            moved, _ = robot.eef_tip_pose(side)
            jacobian[:, column] = (moved - current) / (trial[joint_index] - q[joint_index] + 1e-9)
        robot.robot.set_joint_positions(q.reshape(1, -1))
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * damping * np.eye(3), error)
        q[indices] = np.clip(q[indices] + np.clip(delta, -0.14, 0.14), lower[indices], upper[indices])
    final, _ = robot.eef_tip_pose(side)
    raise RuntimeError(f"numeric {side} IK did not converge; error={np.linalg.norm(np.asarray(target) - final):.3f} m")


def _build_numeric_trajectory(robot, app, scenario, nominal, cfg):
    from r1_bimanual_dataset.core.nominal_grasp import pregrasp_pose
    from r1_bimanual_dataset.core.trajectory import JointTrajectory, Waypoint
    from r1_bimanual_dataset.core.types import Phase

    q_initial = robot.full_positions()
    q_initial[robot.arm_indices["left"]] = np.asarray(scenario.left_start_pose)
    q_initial[robot.arm_indices["right"]] = np.asarray(scenario.right_start_pose)
    for side in ("left", "right"):
        q_initial[robot.gripper_indices[side]] = np.asarray(cfg["gripper_open"])
    left_pre = pregrasp_pose(nominal.left, nominal.object_center, scenario.pregrasp_distance)
    right_pre = pregrasp_pose(nominal.right, nominal.object_center, scenario.pregrasp_distance)
    q_pre = _numeric_ik(robot, app, "left", left_pre.position, q_initial)
    q_pre = _numeric_ik(robot, app, "right", right_pre.position, q_pre)
    q_grasp = _numeric_ik(robot, app, "left", nominal.left.position, q_pre)
    q_grasp = _numeric_ik(robot, app, "right", nominal.right.position, q_grasp)
    lift = np.asarray(scenario.lift_vector, dtype=np.float64)
    q_lift = _numeric_ik(robot, app, "left", nominal.left.position + lift, q_grasp)
    q_lift = _numeric_ik(robot, app, "right", nominal.right.position + lift, q_lift)

    def with_grippers(q, closed):
        result = q.copy()
        for side in ("left", "right"):
            result[robot.gripper_indices[side]] = np.asarray(scenario.gripper_target[side] if closed else cfg["gripper_open"])
        return result

    # Do not lower through the table: return to the physically reached grasp
    # height, release, then keep the object on its true support surface.
    waypoints = [
        Waypoint(Phase.INITIAL, 1.0, with_grippers(q_initial, False)),
        Waypoint(Phase.PREGRASP, 2.5, with_grippers(q_pre, False)),
        Waypoint(Phase.APPROACH, 1.5, with_grippers(q_grasp, False)),
        Waypoint(Phase.CLOSE, 1.0, with_grippers(q_grasp, True)),
        Waypoint(Phase.HOLD, 0.8, with_grippers(q_grasp, True)),
        Waypoint(Phase.LIFT, 1.5, with_grippers(q_lift, True)),
        Waypoint(Phase.LIFT_HOLD, 1.0, with_grippers(q_lift, True)),
        Waypoint(Phase.PLACE, 1.5, with_grippers(q_grasp, True)),
        Waypoint(Phase.RELEASE, 1.0, with_grippers(q_grasp, False)),
    ]
    robot.robot.set_joint_positions(q_initial.reshape(1, -1))
    robot.robot.set_joint_velocities(np.zeros((1, robot.dof_count)))
    return JointTrajectory(waypoints, {s: np.asarray(cfg["gripper_open"]) for s in ("left", "right")}, {s: np.asarray(scenario.gripper_target[s]) for s in ("left", "right")}, robot.gripper_indices, 5.0, 10.8)


def _physical_scene_config(scene_cfg: dict, fixed_root_path: str) -> dict:
    cfg = dict(scene_cfg)
    cfg.update(
        {
            "robot_prim": fixed_root_path,
            "base_movement": "disabled",
            "runtime_base_lock": False,
            "runtime_joint_state_lock": False,
            "physical_rollout": True,
            "scripted_grasp_attachment": False,
            "nominal_grasp_horizontal": True,
            # Return the EEFs to the same geometric grasp centre before
            # opening.  T01's rigid-body origin is calibrated separately in
            # the physical runner because its authored bbox is offset inside
            # the body prim.
            "place_hover_height_m": 0.0,
            # The imported 5.1 articulation publishes a zero tensor Jacobian
            # despite responding correctly to native targets.  During the
            # planning-only precheck we therefore measure its local FK by
            # finite differences; rollout below remains pure PD position
            # control with contact physics enabled.
            "ik_jacobian_source": "finite_difference",
            "ik_max_iterations": 80,
            # These are the scene-local values that previously converged for
            # the calibrated R1 home pose and this reachable object height.
            "ik_damping": 0.05,
            "ik_step_limit": 0.08,
            "ik_line_search": True,
            # Galaxea's verified R1 examples use left/right_arm_link6 itself
            # as the EEF body. The prior 0.15 m local-Z offset placed the
            # physical fingers well outside T01 even though IK converged.
            "eef_tip_offsets": {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]},
            # The corrected scene's authored R1 arm pose is the calibrated
            # safe home used by the Galaxea reference; retain it as the first
            # IK seed for this scene-specific runner.
            "use_live_initial_state": False,
            # T01 has an authored mesh offset inside its rigid-body prim.
            # Use the live rigid-body pose as the physical center and retain
            # source bbox dimensions captured before runtime relocation.
            "nominal_object_center_from_rigid_pose": True,
            # The old scene config accepted loose poses because its root and
            # joint states were locked.  Real physics needs tight IK gates.
            "ik_accept_position_error": 0.035,
        }
    )
    # These values were produced by the legacy root/joint-lock pipeline.  A
    # physical fixed-base solve must never silently reuse them as reachability
    # evidence.
    cfg.pop("ik_fallback_seed", None)
    cfg["ik_fallback_seeds"] = {}
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="R1 real-physics bimanual grasp (no attachment)")
    parser.add_argument("--scenario", type=Path, default=PACKAGE_ROOT / "reference" / "scenario_reference.json")
    parser.add_argument("--scene-config", type=Path, default=WORKSPACE_ROOT / "scene_config.json")
    parser.add_argument(
        "--target-object",
        type=str,
        default=None,
        help="override the target rigid-body Prim path without editing scene_config.json",
    )
    parser.add_argument("--execute", action="store_true", help="run the full physical approach/close/lift/place trajectory")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument(
        "--steady-place",
        action="store_true",
        help="use slower initial positioning and a conservative place/release profile",
    )
    parser.add_argument(
        "--authored-table-target",
        action="store_true",
        help="leave T01 at the 0.64m authored table height (expected to fail R1 reachability)",
    )
    args = parser.parse_args()

    scenario = ScenarioConfig.from_json(args.scenario)
    scene_cfg = load_scene_config(args.scene_config)
    if args.target_object:
        scene_cfg["target_object"] = args.target_object
    if args.steady_place:
        scene_cfg["motion_profile"] = "steady_place"
    target_label = str(scene_cfg["target_object"]).rstrip("/").rsplit("/", 1)[-1]
    prepare_isaac_environment()
    from isaacsim import SimulationApp

    app = SimulationApp(
        simulation_app_config(headless=not args.gui, renderer="RayTracedLighting" if args.gui else "None")
    )
    try:
        # Do not warm up the loaded scene before the Session-Layer support
        # move; otherwise the existing static collider is already cooked at
        # its authored table position.
        stage, timeline = _open_stage(app, scene_cfg["stage_path"], warmup_updates=0)
        obj_min, obj_max = _world_bbox(stage, scene_cfg["target_object"])
        desired_base, base_rot, placement = select_long_edge_pose(obj_min, obj_max, side="near", stand_off_m=0.72)
        source_offset = _source_base_link_world_translation(stage, scene_cfg["robot_prim"])
        desired_link = desired_base.copy()
        desired_link[2] = source_offset[2]
        _place_existing_robot_in_session(stage, scene_cfg["robot_prim"], desired_link - source_offset, base_rot)
        object_dimensions = obj_max - obj_min
        from pxr import UsdGeom

        target_prim = stage.GetPrimAtPath(scene_cfg["target_object"])
        source_target_origin = np.asarray(
            UsdGeom.XformCache().GetLocalToWorldTransform(target_prim).ExtractTranslation(),
            dtype=np.float64,
        )
        source_bbox_center = (obj_min + obj_max) / 2.0
        # The authored target rigid-body origin may be at the lower face of its mesh;
        # its visual/geometric centre is offset upward by about 0.0601 m.
        target_bbox_offset = source_bbox_center - source_target_origin
        # R1's validated arm workspace is centred near local [0.50, 0, 1.00]
        # relative to its physical base. This keeps the mobile base outside
        # the table while moving only the target/support into a reachable band.
        from r1_bimanual_dataset.core.robot_interface import quat_rotate

        # The source table top is below the fixed R1 arm's reliable physical
        # band. Raise the runtime kinematic support and T01 together by 6 cm
        # and bring them 10 cm toward the fixed base. The physical grasp
        # target is then near the calibrated home EEF positions without
        # placing the robot inside the authored table.
        workspace_center = desired_link + quat_rotate(base_rot, np.asarray([0.40, 0.0, 1.06]))
        runtime_object_position = workspace_center - target_bbox_offset
        print(
            "Physical grasp: target pose calibration "
            f"source_body_origin={np.round(source_target_origin, 4).tolist()} "
            f"bbox_center={np.round(source_bbox_center, 4).tolist()} "
            f"bbox_offset={np.round(target_bbox_offset, 4).tolist()} "
            f"runtime_body_position={np.round(runtime_object_position, 4).tolist()} "
            f"visual_center={np.round(workspace_center, 4).tolist()}",
            flush=True,
        )
        use_workspace_target = not args.authored_table_target
        authored_support_path = "/World/TaskSetup/Fixtures/StorageRack/Top"
        authored_support_min, authored_support_max = _world_bbox(stage, authored_support_path)
        authored_support_dimensions = authored_support_max - authored_support_min
        support_path = authored_support_path
        support_min = support_max = None
        if use_workspace_target:
            # Reuse the authored static collision body.  It must be moved
            # before physics is initialized so PhysX cooks it at the runtime
            # Session-Layer pose.  Creating a DynamicCuboid after physics is
            # live invalidates Isaac Sim 5.1's shared articulation tensor view.
            support_min, support_max = _move_existing_support_top(
                stage, authored_support_path, workspace_center, object_dimensions
            )
        _ensure_runtime_target_pose_ops(stage, scene_cfg["target_object"])
        fixed_root_path = _make_existing_r1_fixed(
            stage, scene_cfg["robot_prim"], relocate_articulation_root=False
        )

        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.viewports import set_camera_view
        from r1_bimanual_dataset.core.bimanual_controller import BimanualController
        from r1_bimanual_dataset.core.nominal_grasp import NominalGraspGenerator
        from r1_bimanual_dataset.core.robot_interface import RobotInterface
        from r1_bimanual_dataset.core.success_evaluator import SuccessEvaluator

        camera_target = workspace_center if use_workspace_target else (obj_min + obj_max) / 2.0
        set_camera_view(
            desired_base + np.asarray([-1.5, -1.5, 1.2]),
            camera_target + np.asarray([0.0, 0.0, 0.2]),
        )
        timeline.play()
        SimulationManager.initialize_physics()
        for _ in range(3):
            app.update()
        timeline.stop()
        cfg = _physical_scene_config(scene_cfg, fixed_root_path)
        cfg["object_bbox_dimensions_m"] = object_dimensions.tolist()
        robot = RobotInterface(cfg, update_fn=app.update)
        robot.initialize()
        if use_workspace_target:
            for _ in range(3):
                app.update()
            robot.object.set_world_poses(
                runtime_object_position.reshape(1, 3),
                np.asarray([[1.0, 0.0, 0.0, 0.0]]),
            )
            robot.object.set_velocities(np.zeros((1, 6), dtype=np.float64))
            immediate_object, _ = robot.object_pose()
            support_min, support_max = _world_bbox(stage, support_path)
            print(
                f"Physical grasp: {target_label} immediately after set={np.round(immediate_object, 4).tolist()} "
                f"requested_body={np.round(runtime_object_position, 4).tolist()} "
                f"visual_center={np.round(workspace_center, 4).tolist()} "
                f"support_bbox={np.round(support_min, 4).tolist()}..{np.round(support_max, 4).tolist()}",
                flush=True,
            )
            for _ in range(15):
                app.update()
            object_after_settle, _ = robot.object_pose()
            print(
                f"Physical grasp: {target_label} after support settle={np.round(object_after_settle, 4).tolist()} "
                f"velocity={np.round(robot.object_velocity(), 4).tolist()}",
                flush=True,
            )
            if float(np.linalg.norm(object_after_settle - runtime_object_position)) > 0.03:
                raise RuntimeError(
                    "workspace target did not settle on the static pedestal; refusing to plan a grasp"
                )
            print(
                f"Physical grasp: {target_label} session-reset to reachable center={np.round(object_after_settle, 4).tolist()}",
                flush=True,
            )
        # PhysX publishes articulation Jacobians only while the timeline is
        # advancing.  Building the trajectory on a stopped timeline returns a
        # zero Jacobian and makes every IK target falsely unreachable.
        timeline.play()
        for _ in range(5):
            app.update()
        measured_target_before_plan, _ = robot.object_pose()
        print(
            f"Physical grasp: {target_label} before IK planning="
            f"{np.round(measured_target_before_plan, 4).tolist()} "
            f"requested={np.round(workspace_center, 4).tolist()}",
            flush=True,
        )
        # Use the pose that this runner explicitly placed and verified on the
        # support as the nominal grasp-frame centre.  This avoids counting the
        # authored T01 mesh offset a second time while keeping the live pose
        # check above visible in the log.
        cfg["nominal_object_center_override"] = workspace_center.tolist()
        nominal = NominalGraspGenerator(stage, cfg, robot).generate()
        print(
            "Physical grasp: nominal frames "
            f"center={np.round(nominal.object_center, 4).tolist()} "
            f"left={np.round(nominal.left.position, 4).tolist()} "
            f"right={np.round(nominal.right.position, 4).tolist()}",
            flush=True,
        )
        trajectory = BimanualController(robot, cfg).build(scenario, nominal)
        # IK planning evaluates live FK by moving the articulation through
        # trial configurations. Those planning poses can physically bump a
        # dynamic T01 even though the final joint state is restored. Reset
        # the payload once more after planning so the rollout starts from the
        # same clean supported state that produced the nominal grasp frames.
        if use_workspace_target:
            robot.object.set_world_poses(
                runtime_object_position.reshape(1, 3),
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
            )
            robot.object.set_velocities(np.zeros((1, 6), dtype=np.float64))
            # Let the real rigid body resolve any small support penetration
            # introduced by the planning poses.  The settled Z is allowed to
            # differ from the requested pose because the source mesh's
            # collision shape is not centred exactly on its USD transform.
            for _ in range(30):
                app.update()
            post_plan_object, _ = robot.object_pose()
            post_plan_velocity = robot.object_velocity()
            horizontal_delta = float(np.linalg.norm((post_plan_object - runtime_object_position)[:2]))
            vertical_settle_delta = float(post_plan_object[2] - runtime_object_position[2])
            linear_speed = float(np.linalg.norm(post_plan_velocity[:3]))
            print(
                f"Physical grasp: {target_label} reset after IK planning "
                f"position={np.round(post_plan_object, 4).tolist()} "
                f"velocity={np.round(post_plan_velocity, 4).tolist()} "
                f"horizontal_delta={horizontal_delta:.4f}m "
                f"vertical_settle_delta={vertical_settle_delta:.4f}m",
                flush=True,
            )
            if (
                not np.all(np.isfinite(post_plan_object))
                or not np.all(np.isfinite(post_plan_velocity))
                or horizontal_delta > 0.03
                or linear_speed > 0.05
            ):
                raise RuntimeError(
                    f"{target_label} did not return to a stable supported reset pose after IK planning"
                )
            print(
                "Physical grasp: supported reset accepted; Z settle is physical contact, "
                "not a planning failure",
                flush=True,
            )
        print(f"Physical grasp: REACHABILITY PASS; {len(trajectory.waypoints)} phases, duration={trajectory.duration_s:.2f}s", flush=True)
        if not args.execute:
            timeline.stop()
            return 0

        initial_object_position, initial_object_quaternion = robot.object_pose()
        initial_object_pose = np.concatenate([initial_object_position, initial_object_quaternion])
        evaluator = SuccessEvaluator(cfg, scenario, nominal, initial_object_pose)
        initial_root, _ = robot.root_pose()
        last_phase = None
        gripper_indices = np.concatenate(
            [robot.gripper_indices["left"], robot.gripper_indices["right"]]
        )
        dt = 1.0 / float(cfg["fps"])
        frame_count = 0
        for frame_index, t in enumerate(np.arange(0.0, trajectory.duration_s + 1e-9, dt)):
            command, phase = trajectory.sample(float(t))
            robot.apply_full_target(command)
            for _ in range(max(1, int(round(dt / float(cfg["physics_dt_fallback"]))))):
                app.update()
            root, _ = robot.root_pose()
            if float(np.linalg.norm(root - initial_root)) > 1e-4:
                raise RuntimeError("fixed base moved during physical rollout")
            object_pos, object_quaternion = robot.object_pose()
            object_velocity = robot.object_velocity()
            left_eef, left_quaternion = robot.eef_tip_pose("left")
            right_eef, right_quaternion = robot.eef_tip_pose("right")
            measured_state, measured_velocity = robot.state16()
            object_bbox_min, object_bbox_max = _world_bbox(stage, scene_cfg["target_object"])
            evaluator.update(
                {
                    "object_pose": np.concatenate([object_pos, object_quaternion]),
                    "object_velocity": object_velocity,
                    "left_eef_pose": np.concatenate([left_eef, left_quaternion]),
                    "right_eef_pose": np.concatenate([right_eef, right_quaternion]),
                    "joint_position_16d": measured_state,
                    "joint_velocity_16d": measured_velocity,
                    "action_16d": robot.action_from_full(command),
                    "gripper_command": robot.action_from_full(command),
                    "phase": phase.value,
                    "object_bbox_min": object_bbox_min,
                    "object_bbox_max": object_bbox_max,
                    "support_bbox_min": support_min if use_workspace_target else None,
                    "support_bbox_max": support_max if use_workspace_target else None,
                }
            )
            frame_count += 1
            if phase != last_phase:
                measured = robot.full_positions()
                print(
                    "Physical grasp: phase transition "
                    f"{phase.value} object={np.round(object_pos, 4).tolist()} "
                    f"left_eef={np.round(left_eef, 4).tolist()} "
                    f"right_eef={np.round(right_eef, 4).tolist()} "
                    f"gripper_actual={np.round(measured[gripper_indices], 4).tolist()} "
                    f"gripper_command={np.round(command[gripper_indices], 4).tolist()}",
                    flush=True,
                )
                last_phase = phase
            if frame_index % 10 == 0:
                object_pos, _ = robot.object_pose()
                print(f"Physical grasp: t={t:.1f}s phase={phase.value} object={np.round(object_pos, 4).tolist()}", flush=True)
        final_command, final_phase = trajectory.sample(trajectory.duration_s)
        final_object, _ = robot.object_pose()
        final_velocity = robot.object_velocity()
        final_measured = robot.full_positions()
        final_bbox_min, final_bbox_max = _world_bbox(stage, scene_cfg["target_object"])
        support_bbox_text = "unavailable"
        if use_workspace_target:
            support_bbox_min, support_bbox_max = _world_bbox(stage, support_path)
            support_bbox_text = (
                f"{np.round(support_bbox_min, 4).tolist()}.."
                f"{np.round(support_bbox_max, 4).tolist()}"
            )
        print(
            "Physical grasp: final state "
            f"phase={final_phase.value} object={np.round(final_object, 4).tolist()} "
            f"linear_velocity={np.round(final_velocity[:3], 4).tolist()} "
            f"gripper_actual={np.round(final_measured[gripper_indices], 4).tolist()} "
            f"gripper_command={np.round(final_command[gripper_indices], 4).tolist()} "
            f"object_bbox={np.round(final_bbox_min, 4).tolist()}.."
            f"{np.round(final_bbox_max, 4).tolist()} "
            f"support_bbox={support_bbox_text}",
            flush=True,
        )
        result = evaluator.finalize(trajectory.duration_s, frame_count)
        timeline.stop()
        print(
            f"R1 runner: episode finished with {result.actual_outcome}\n"
            + json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
            flush=True,
        )
        if args.keep_open and args.gui:
            while app.is_running():
                app.update()
        return 0
    except Exception as exc:
        print(f"Physical grasp: FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
