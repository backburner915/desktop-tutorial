# Dataset Specification V0.2

状态：**已冻结**（原始继承 TRA-01，2026-09-07；V0.2 于 2026-09-10 按用户明确指示做了一处
偏离 TRA-01 原文的breaking change，见下方"与 TRA-01 的偏离"）。批量采集前不得再改；如需
再改必须先更新 TRA-01 并同步这里。

本文档是训练端（TRA-01）与仿真端（本仓库 `episode_gen/`）之间的**唯一数据合同**。`episode_gen/dataset_writer.py` 的实现必须逐字段满足本文档；任何一项不满足，`tests/` 里的校验测试必须失败并阻止批量采集，对应 TRA-01 §11 的验收清单。

## 0. 与 TRA-01 的偏离（V0.1 → V0.2，需要回头找训练/VLA同学同步）

TRA-01 原文任务是"抓取 Crew Lock Bag，放入绿色目标环"。经与 sim 同学核对真实场景
（完整 prim 扫描 + 近100条真实测试的 success 判据），**场景里没有任何叫"目标环"/"ring"
的物体，抓取对象目前也是占位方块（T01/T02/T03），不是 Crew Lock Bag**。用户已明确指示
删除"放入目标环"这个要求。V0.2 改为：抓取 → 抬升 → 稳定保持 → 移动到放置位置附近 → 
放下 → 松开，不要求"环形标记物"，放置位置退化为一个坐标+半径/高度容差的邻近性判定
（见 `episode_gen/scenario.py::ScenarioConfig.place_target_pose`）。

**这是本仓库内部对实现现实的适配，不是训练/VLA同学正式批准的 TRA-01 修订**——如果他们
的训练方案依赖"放入环形目标"这个具体动作语义，需要专门找他们同步这个变化。

## 1. 基础配置（不可变更）

| 项目 | 值 |
|---|---|
| 基础模型 | `lerobot/smolvla_base` |
| 数据格式 | LeRobot Dataset v3 |
| 任务 | R1 双臂抓取物体，抬升、稳定保持后放下（**不要求放入目标环，见上）** |
| 相机 | `front`、`left_wrist`、`right_wrist` |
| 图像 | RGB，640×480，30 FPS，`shape=[480,640,3]` |
| state | `observation.state`: `float32[16]`，实际测得关节位置 |
| action | `action`: `float32[16]`，观测后实际下发的**绝对**关节位置目标 |
| 频率 | 30 Hz（state/action/video 同一采样周期），单帧周期 33.333 ms |
| 任务文本 | 固定，逐字：`Grasp the object with both hands, lift it, hold it steadily, and place it down.`（与 sim 端近100条真实测试实际使用的指令文本一致） |

## 2. 16D 顺序（永久固定，与 galaxea_lab 源码逐字一致）

已核对 `userguide-galaxea/galaxea_lab` 仓库 `omni/isaac/lab_assets/galaxea_robots.py` 中 `GALAXEA_R1_CFG.init_state.joint_pos` 的关节命名，顺序与 TRA-01 完全一致，不需要额外映射层：

```
0  left_arm_joint1     4  left_arm_joint5      8  right_arm_joint1    12 right_arm_joint5
1  left_arm_joint2     5  left_arm_joint6      9  right_arm_joint2    13 right_arm_joint6
2  left_arm_joint3     6  left_gripper_axis1   10 right_arm_joint3    14 right_gripper_axis1
3  left_arm_joint4     7  left_gripper_axis2   11 right_arm_joint4    15 right_gripper_axis2
```

禁止：目标值代替实际 state / 绝对位置与增量混用 / 位置与速度混用 / 位置与力矩混用。夹爪单位、正方向、限位全程一致，以 `galaxea_robots.py` 里的 `ImplicitActuatorCfg` 限位为准。

## 3. 相机（与 galaxea_lab 现有实现对齐，需覆盖分辨率/帧率）

`lift_env_cfg.py` 里的 `R1LiftEnvCfg` 已经定义了这三个相机的 prim path，可以直接复用装配关系，但要显式覆盖两处默认值使其满足 TRA-01：

| 相机 | prim path（已验证） | 需要覆盖的默认值 |
|---|---|---|
| front | `{ENV_REGEX_NS}/Robot/torso_link4/front_camera` | `height=480, width=640`（源码里部分默认是 240×320，必须显式传参覆盖，不能依赖默认）；`update_period=1/30`（源码注释写"30Hz"但 `GALAXEA_CAMERA_CFG` 默认 `update_period=1/60`，必须显式设为 `1/30` 才是真的 30Hz） |
| left_wrist | `{ENV_REGEX_NS}/Robot/left_arm_link6/left_wrist_camera` | 同上 |
| right_wrist | `{ENV_REGEX_NS}/Robot/right_arm_link6/right_wrist_camera` | 同上 |

## 4. 每帧字段（LeRobot Dataset v3 schema）

```
observation.images.front          video, [480,640,3]
observation.images.left_wrist     video, [480,640,3]
observation.images.right_wrist    video, [480,640,3]
observation.state                 float32[16]
action                            float32[16]
timestamp
frame_index
episode_index
index
task_index                        -> meta/tasks.parquet 关联固定任务文本
```

## 5. Policy 输入 与 诊断信息 必须分离

**原则**（对应会议纪要 §5）：仿真里能拿到的"上帝视角"信息（`object_pose`、`contact_force`、`collision_type`、完美 GT bbox 等）在真机推理时通常不存在。这些信息**必须记录**（用于 taxonomy 分类、failure 分析、R/F 数据筛选、以后做 critic/value 训练），但**不得**默认混入上表的 policy 输入字段。

`episode_gen/dataset_writer.py` 因此拆成两个输出：

- `episode.jsonl`（→ 未来接入真实 LeRobotDataset 的 5 个 policy 字段 + timestamp/index 系列）：**只包含上表第4节的字段**
- `diagnostics.jsonl`（同一 `frame_index` 对齐，不作为 policy 输入）：

```
scenario_id        # N01 / R11 / ...
phase               # APPROACH / GRASP / ... （见 episode_gen/fsm.py）
contacts             # 当前碰撞/接触事件列表
object_pose            # 物体 GT pose
object_velocity
left_contact / right_contact
left_grasp_error / right_grasp_error
recovery_count
anomaly_event          # 若本帧发生异常检测，记录触发的 R-ID
config                # 本 episode 的随机参数（用于复现）
seed
```

`success` / `failure_reason` 记录在 episode 级别的 `episode_meta.json`，不是逐帧字段。

## 6. 成功判据（V0.2，见 §0 的偏离说明）

同时满足：双侧有效夹持 / 物体离开支撑面 / 运输过程中未掉落 / 到达 `place_target_pose`
附近（半径+高度容差判定，不要求环形标记物，见 `IsaacLabR1Adapter._inside_target`）/ 完成释放 / 释放后 0.5–1.0s 内保持稳定（未倒落、未明显位移）。禁止用 FixedJoint 伪造成功轨迹——sim 端 `scene_config_source_snapshot.json` 里 `scripted_grasp_attachment: false` 确认了这点。

## 7. 成功 / 恢复 / 纯失败 三类数据如何参与训练（对应会议纪要 §7）

这三类不能无条件混合进同一份监督数据，具体怎么用由训练负责人（TRA-01 owner）决定，本仓库只负责**在采集阶段就把它们分开存**，不做训练侧假设：

| 类型 | 存储位置 | 首轮 SFT 默认策略 |
|---|---|---|
| N/P 成功 episode | `episode_success=true` | 进入首轮 SFT |
| R 类恢复 episode | `episode_success=true`，且 `diagnostics` 里保留完整异常与恢复过程 | 只有"专家纠正及其后的成功动作"作为正向数据，异常发生前到纠正开始这段视具体 R-ID 由训练侧决定是否截断（本仓库默认整段都保留在 diagnostics，是否截断在训练侧的数据加载器处理，而不是采集阶段丢弃原始数据） |
| F 类纯失败 episode | `episode_success=false` | 不进入首轮 SFT，单独保存供 label / critic / 对比分析使用 |

## 8. 校验（对应 TRA-01 §11，V0.1 阶段可在无 Isaac Sim 环境下验证的子集）

`tests/test_r11_recovery_loop.py` 用 `MockSimAdapter` 验证以下与仿真引擎无关的项（已在本仓库跑通，见测试文件）：

- state/action 均为长度 16 的 float 序列，无 NaN/Inf
- 帧间隔 ≈ 33.333 ms（容差可配置）
- `episode_index`/`frame_index`/`index`/`task_index` 单调且边界正确
- policy 字段与 diagnostics 字段严格分离，没有诊断字段泄漏进 `episode.jsonl`
- `success` 与实际 FSM 终止阶段（SUCCESS/FAILURE）一致

以下各项依赖真实 Isaac Sim / 真实 LeRobotDataset 库，标记为「仅在你的 Isaac Sim 环境验证」，本仓库暂不能执行：视频 shape/帧数与 parquet 帧数一致、`SmolVLAPolicy.forward()` 返回有限 loss、`stats.json` 生成、动作 chunk 构造、训练/验证集按 episode 隔离。这些留在 TRA-01 §11 的清单里，采集脚本产出后由你在有 GPU/Isaac Sim 的环境里跑一遍确认。
