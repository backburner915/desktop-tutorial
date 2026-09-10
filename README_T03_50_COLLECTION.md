# R1 / T03 50-episode collection

本目录中的 `run_aligned_t03_batch.py` 是本次 50 条数据采集入口。它使用
`D:/Galaxea_Lab-galaxea-main/spacerobot.usd`，在独立 Session Layer 中固定底盘、
复用已验证的 R1 关节驱动和有限差分 IK，不覆盖原始 USD 或原始成功脚本。

## 已对齐的固定项

- target object: `/World/TaskSetup/MovablePayloads/T03`
- support: `/World/TaskSetup/Fixtures/StorageRack/Top`
- robot root: `/World/garobot2_driveable_final/r1_DVT_colored`
- EEF: `left_arm_link6`, `right_arm_link6`
- control/data rate: 30 Hz
- camera output: front / left_wrist / right_wrist, 640x480 RGB
- action: 16D absolute joint position target
- action order: left arm 6 + left gripper 2 + right arm 6 + right gripper 2

## Generate scenarios

```powershell
& 'D:\isaaclab_env\python.exe' `
  'C:\Users\许久轻\Documents\ChatGPT\通信\r1_bimanual_dataset\tools\generate_scenarios.py' `
  --num-scenarios 50 --success-ratio 0.6 --master-seed 20260909 `
  --target-object '/World/TaskSetup/MovablePayloads/T03' `
  --scene-config 'C:\Users\许久轻\Documents\ChatGPT\通信\scene_config.json' `
  --preflight 'C:\Users\许久轻\Documents\ChatGPT\通信\preflight_report.json' `
  --output 'D:\r1_t03_scenarios_50'
```

## Run batch

```powershell
& 'D:\isaaclab_env\python.exe' `
  'C:\Users\许久轻\Documents\ChatGPT\通信\r1_bimanual_dataset\run_aligned_t03_batch.py' `
  --scenarios-dir 'D:\r1_t03_scenarios_50' `
  --output 'D:\r1_t03_smollvla_50' --limit 50
```

## Validate for SmolVLA / LeRobot conversion

```powershell
& 'D:\isaaclab_env\python.exe' `
  'C:\Users\许久轻\Documents\ChatGPT\通信\r1_bimanual_dataset\tools\check_smolvla_compatibility.py' `
  'D:\r1_t03_smollvla_50'
```

RAW 每个 episode 都包含 `metadata.json`、`result.json`、`trajectory.npz` 和三路
PNG。训练核心字段为 `observation.images.front`、`observation.images.left_wrist`、
`observation.images.right_wrist`、`observation.state`、`action`、`task`；额外的
物体姿态、接触、phase、失败诊断和全关节 state 保留在 RAW/metadata 中。

本阶段不修改 SmolVLA 模型，也不把失败 action 当作成功标签；本阶段只验证原始
数据可完整导出和追溯。LeRobot 导出应在 RAW 验收通过后单独执行。
