"""Non-destructive setup shared by the real fixed-base R1 reference tools."""

from __future__ import annotations

from typing import Any

import numpy as np

from .scene_overlay import add_physical_target_overlay, settle_target_on_support


def prepare_fixed_target_scene(app: Any, scene_cfg: dict[str, Any]) -> tuple[Any, Any, Any, str, dict[str, object]]:
    """Open the source stage and build the physical reference in Session Layer.

    The source robot is placed once beside the *unchanged* long table edge and
    fixed with a true world joint before the first PhysX step.  The function
    does not alter the source USD, table, collision filters, or payload state
    during rollout.
    """

    from r1_bimanual_dataset.tools.physical_reference_probe import (
        _open_stage,
        _place_existing_robot_in_session,
        _source_base_link_world_translation,
        _world_bbox,
        select_long_edge_pose,
    )
    from .contact_tracker import GripperContactTracker
    from .robot_interface import RobotInterface

    stage, timeline = _open_stage(app, scene_cfg["stage_path"])
    overlay = add_physical_target_overlay(stage, scene_cfg)
    for _ in range(120):
        app.update()
    target_report = settle_target_on_support(stage, overlay)
    print(f"Physical scene: target settled={target_report}", flush=True)
    from .qualified_scene import add_table_supports, configure_physics, fix_root, repair_finger_collision_schema

    robot_prim = stage.GetPrimAtPath(scene_cfg["robot_prim"])
    if not robot_prim or not robot_prim.IsValid():
        raise RuntimeError(f"R1 robot prim is missing: {scene_cfg['robot_prim']}")
    # Repair imported finger collision schemas before adding Session-Layer
    # placement ops.  This keeps the collision meshes' local transforms
    # stable while PhysX builds the shapes.
    finger_repairs = []
    if not bool(scene_cfg.get("diagnostic_skip_finger_repairs", False)):
        finger_repairs = repair_finger_collision_schema(stage, robot_prim)
    print(f"Physical scene: finger collider repairs={len(finger_repairs)}", flush=True)
    table_path = str(scene_cfg.get("table_top", "/World/TaskSetup/Fixtures/StorageRack/Top"))
    support_paths = []
    if not bool(scene_cfg.get("diagnostic_skip_supports", False)):
        support_paths = add_table_supports(stage, table_path)
    print(f"Physical scene: support fixtures={len(support_paths)}", flush=True)
    # Place the base beside the selected target's long edge.  Using the full
    # tabletop bound here can put the robot metres away from the object on
    # imported scenes whose tabletop includes the whole storage rack.
    target_min, target_max = _world_bbox(stage, overlay.target_path)
    print(f"Physical scene: target bbox after settle min={np.round(target_min,4).tolist()} max={np.round(target_max,4).tolist()}", flush=True)
    base_position, base_rotation, placement = select_long_edge_pose(
        target_min,
        target_max,
        side=str(scene_cfg.get("fixed_base_side", "near")),
        stand_off_m=float(scene_cfg.get("fixed_base_stand_off_m", 0.78)),
    )
    source_offset = _source_base_link_world_translation(stage, scene_cfg["robot_prim"])
    desired_link = base_position.copy()
    desired_link[2] = source_offset[2]
    _place_existing_robot_in_session(
        stage, scene_cfg["robot_prim"], desired_link - source_offset, base_rotation
    )
    # Constrain the one true base rigid body to World with a Session-Layer
    # FixedJoint.  The articulation wrapper path is preserved for live DOFs.
    fixed_root = None
    if not bool(scene_cfg.get("diagnostic_skip_fixed_root", False)):
        fixed_root = fix_root(stage, robot_prim)
    print(f"Physical scene: fixed articulation root={fixed_root}", flush=True)
    if bool(scene_cfg.get("runtime_physics_baseline_enabled", True)):
        physics_report = configure_physics(stage, robot_prim)
        print("Physical scene: physics baseline authored", flush=True)
    else:
        physics_report = {
            "status": "SKIPPED_DIAGNOSTIC",
            "reason": "runtime_physics_baseline_disabled_for_init_isolation",
        }
        print("Physical scene: runtime physics baseline skipped (diagnostic)", flush=True)
    runtime_cfg = dict(scene_cfg)
    runtime_cfg.update(
        {
            # Keep the articulation wrapper on the authored articulation root;
            # ``fixed_root`` is the articulation path returned by the
            # validated constructor above.
            "robot_prim": scene_cfg["robot_prim"],
            "target_object": overlay.target_path,
            # Resolve the TCP from the live finger link midpoint after the
            # articulation is initialized.  No fixed link6-local offset is
            # used for physical contact planning.
            "eef_tip_mode": "gripper_midpoint",
            "eef_tip_offsets": {},
            "runtime_base_lock": False,
            "runtime_joint_state_lock": False,
            "physical_rollout": True,
            "scripted_grasp_attachment": False,
            "base_movement": "disabled",
            "ik_jacobian_source": "finite_difference",
            "ik_orientation_weight": float(scene_cfg.get("ik_orientation_weight", 0.0)),
            "ik_damping": float(scene_cfg.get("physical_ik_damping", 0.035)),
            "ik_step_limit": float(scene_cfg.get("physical_ik_step_limit", 0.08)),
            "ik_line_search": True,
            "ik_accept_position_error": float(scene_cfg.get("physical_ik_accept_position_error", 0.025)),
        }
    )
    tracker = GripperContactTracker(stage, scene_cfg["robot_prim"], overlay.target_path)
    prepare_contacts_before_physics = bool(scene_cfg.get("prepare_contact_reports_before_physics", True))
    if prepare_contacts_before_physics:
        tracker.prepare()
        print("Physical scene: contact reports prepared", flush=True)
    # Keep the same initialization path as the passing fixed-scene audits.
    # SimulationContext can block on this imported stage's legacy context;
    # SimulationManager parses the active Session-Layer PhysicsScene safely.
    from isaacsim.core.simulation_manager import SimulationManager

    print("Physical scene: initializing PhysX", flush=True)
    timeline.play()
    SimulationManager.initialize_physics()
    if not prepare_contacts_before_physics:
        tracker.prepare()
        print("Physical scene: contact reports prepared after init", flush=True)
    print("Physical scene: PhysX initialized", flush=True)
    for _ in range(20):
        app.update()
    robot = RobotInterface(runtime_cfg, update_fn=app.update)
    robot.initialize()
    tracker.initialize()
    report = {
        "source_usd_modified": False,
        "table_moved": False,
        "target": target_report,
        "base_placement": placement,
        "fixed_root": fixed_root,
        "fixed_joint": "/World/ReferenceRootFixedJoint",
        "finger_schema_repairs": finger_repairs,
        "supports": support_paths,
        "physics": physics_report,
        "tcp": {
            "mode": "gripper_midpoint",
            "measured_link6_local_offset_m": {
                side: robot.measured_eef_tip_offset(side).tolist() for side in ("left", "right")
            },
        },
    }
    return stage, timeline, robot, overlay.target_path, {**report, "contact_tracker": tracker}


# Kept solely so old planning-only Sugar Box tools remain reproducible.  New
# code must call ``prepare_fixed_target_scene`` and explicitly select its
# target asset in SceneConfig.
prepare_fixed_sugar_scene = prepare_fixed_target_scene
