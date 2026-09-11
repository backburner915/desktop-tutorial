# R1 corrected-scene PREFLIGHT REPORT

Isaac Sim: 5.1.0.0 (`D:/isaaclab_env`)

Stage: `D:/Galaxea_Lab-galaxea-main/spacerobot.usd`

Robot: `/World/garobot2_driveable_final/r1_DVT_colored`

Target object selected for phase one: `/World/TaskSetup/MovablePayloads/T01`

## Verified mapping

```text
left_arm_joint1
left_arm_joint2
left_arm_joint3
left_arm_joint4
left_arm_joint5
left_arm_joint6
left_gripper_axis1
left_gripper_axis2
right_arm_joint1
right_arm_joint2
right_arm_joint3
right_arm_joint4
right_arm_joint5
right_arm_joint6
right_gripper_axis1
right_gripper_axis2
```

`len(ACTION_JOINT_NAMES) == 16`.  The scene also contains three wheel DOFs;
they remain in `all_joint_positions/all_joint_velocities` but are not policy
actions.

LEFT EEF: `/World/garobot2_driveable_final/r1_DVT_colored/left_arm_link6`

RIGHT EEF: `/World/garobot2_driveable_final/r1_DVT_colored/right_arm_link6`

Control method: live PhysX articulation position targets with a damped
Jacobian IK precheck and joint-space trajectory execution.  This follows the
Jacobian/position-target control family used by the Galaxea source examples;
there is no scene-specific golden script to copy for this corrected USD.

Front camera: `/World/garobot2_driveable_final/r1_DVT_colored/torso_link4/front_camera`

Left wrist camera: `/World/garobot2_driveable_final/r1_DVT_colored/left_arm_link6/left_wrist_camera`

Right wrist camera: `/World/garobot2_driveable_final/r1_DVT_colored/right_arm_link6/right_wrist_camera`

T01 world bounding box from the scan: min `[-3.87, 3.799227, 0.58]`, max
`[-3.63, 3.900773, 0.700238]` m.  Its generated nominal grasp uses the local
bbox and the robot lateral axis; it is not a hard-coded `x ± 0.05` grasp.

## Status

PREFLIGHT structural checks: PASS.

PREFLIGHT execution readiness: BLOCKED.  The target center is approximately
4.2 m from the robot base in the current USD, while `base_movement` is
disabled for phase one.  The runner marks this as `INVALID_CONFIGURATION`
before a full simulation episode.  No 10-repeat Reference Gate or 20-episode
simulation pilot has been run from this workspace.
