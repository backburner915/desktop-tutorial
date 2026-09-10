> Reference frozen (2026-09-06): KLT is INVALID_ASSET_COLLISION_MISMATCH.
> The legacy execution and recording entry points now refuse to run. Commands below
> describe historical artifacts; they do not qualify a physical Reference.
> See [current qualification status](../reports/reference_rebuild_status.md).

# R1 bimanual RAW dataset generator

This project targets the corrected scene:

`D:/Galaxea_Lab-galaxea-main/spacerobot.usd`

It does not modify, flatten, or overwrite that USD.  The corrected scene had
no scene-specific success script, so the first runner in this project is the
new Golden Reference implementation.  The three existing Galaxea examples in
the source checkout are used only as API/control references.

The fixed SmolVLA action ordering is:

```text
left_arm_joint1..6, left_gripper_axis1..2,
right_arm_joint1..6, right_gripper_axis1..2
```

The live articulation is checked against those exact names before execution.
The full articulation state is retained in `all_joint_positions` and
`all_joint_velocities`; the policy-facing `observation.state` and `action`
remain 16D.

## Current scene adapter

The source stage remains read-only. Candidate objects are selected explicitly
in a Session Layer; `scene_config.json` intentionally has no default target
asset while the Reference is unqualified.

```text
robot: /World/garobot2_driveable_final/r1_DVT_colored
target: /World/TaskSetup/MovablePayloads/T01
left EEF link: /World/garobot2_driveable_final/r1_DVT_colored/left_arm_link6
right EEF link: /World/garobot2_driveable_final/r1_DVT_colored/right_arm_link6
front camera: /World/garobot2_driveable_final/r1_DVT_colored/torso_link4/front_camera
left wrist: /World/garobot2_driveable_final/r1_DVT_colored/left_arm_link6/left_wrist_camera
right wrist: /World/garobot2_driveable_final/r1_DVT_colored/right_arm_link6/right_wrist_camera
```

The Session Layer adds real table legs and crossbeams, repairs the imported
finger collision schemas on the actual meshes, and constrains the true
`base_link` to World with a PhysX FixedJoint. Runtime root/joint locks,
collision filters, payload attachment, and source-stage edits are disabled.
The three wheel joints remain in the raw 19-DOF state but are not part of the
policy-facing 16D action.

## Commands

Run the tools with the Isaac Sim Python environment.  The exact environment
used during preflight was `D:/isaaclab_env/python.exe`; add the Galaxea source
extensions to `PYTHONPATH` as shown below in PowerShell:

```powershell
$repo = 'D:/Galaxea_Lab-galaxea-main/Galaxea_Lab-galaxea-main'
$env:PYTHONPATH = "$repo/source/extensions/omni.isaac.lab;$repo/source/extensions/omni.isaac.lab_assets;$repo/source/extensions/omni.isaac.lab_tasks"
# Gate A: unified geometry precheck (no episode output)
D:/isaaclab_env/python.exe tools/inspect_scene_geometry.py --candidate sugar_box
D:/isaaclab_env/python.exe tools/inspect_scene_geometry.py --candidate cracker_box
D:/isaaclab_env/python.exe r1_bimanual_dataset/tools/audit_candidate_static.py --candidate sugar_box
D:/isaaclab_env/python.exe r1_bimanual_dataset/tools/audit_candidate_static.py --candidate cracker_box
D:/isaaclab_env/python.exe r1_bimanual_dataset/tools/build_contact_frames.py --candidate sugar_box
D:/isaaclab_env/python.exe r1_bimanual_dataset/tools/build_contact_frames.py --candidate cracker_box

# Gate status and reports
Get-Content reports/reference_gates.json
Get-Content reports/reference_rebuild_status.md
```

For visual checking, add `--gui --keep-open`. GUI mode is intentionally a
lightweight preview: it executes the grasp and automatically frames the task
area, but does not create the three camera render products or write dataset
files. Use headless mode for actual RAW data generation. The optional
`--record-cameras-in-gui` flag is not recommended on this 16 GB machine.

After Gate A and dynamic Gates B–E pass, the intended execution order is:

```powershell
D:/isaaclab_env/python.exe r1_bimanual_dataset/run_reference.py --repetitions 10
python r1_bimanual_dataset/tools/generate_scenarios.py --num-scenarios 20 --output scenarios
D:/isaaclab_env/python.exe r1_bimanual_dataset/run_batch.py --scenarios scenarios --limit 20
python r1_bimanual_dataset/tools/validate_dataset.py dataset_raw
python r1_bimanual_dataset/tools/dataset_report.py dataset_raw
```

`run_reference.py`, `run_scenario.py`, `run_batch.py`, and the old physical
grasp entry point currently fail closed until a replacement Reference passes
the recorded gates. `run_batch.py` stops after three consecutive `INVALID`
results. Invalid
configuration, IK/controller errors, camera errors, reset errors, NaN and
timeouts are never relabeled as physical failures.

## Episode format

Each episode contains `metadata.json`, `trajectory.npz`, `result.json`, and
three camera directories.  All three camera images are read after the same
simulation step.  The trajectory stores commanded action separately from
measured state and includes phase, EEF pose, object pose/velocity, contact
proxy information, and all live articulation DOFs.

Scenario generation uses seeded, layered ranges.  Success design is exactly
6:4 against failure design for divisible counts, and failures inject one
primary mode at small/medium/large continuous magnitudes.  `intended_outcome`
and `actual_outcome` are recorded independently.

LeRobot/SmolVLA export is intentionally not coupled to simulation execution;
an exporter can be added after RAW validation without rerunning Isaac Sim.
