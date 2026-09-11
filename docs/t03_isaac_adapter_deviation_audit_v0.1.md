# T03 `IsaacLabR1Adapter` 偏差清单 v0.1

**状态：阶段 0 冻结，待确认；不包含实现改动。**
**基线：** `claude/gifted-carson-67qkgn`，审计时 HEAD 为 `f4338fb`。
**范围：** 仅比对既有 `episode_gen` 契约和 T03 sim 实现；不改变 USD、资产、相机配置、articulation 或任何 Python 代码。

## 1. 结论范围与责任边界

已有契约是本次唯一权威来源：

- [`episode_gen/types.py`](../episode_gen/types.py) 的 `Observation` 和 `AnomalyEvent`；
- [`episode_gen/sim_adapter.py`](../episode_gen/sim_adapter.py) 的 `SimAdapter` 五方法；
- [`dataset_spec_v0.1.md`](dataset_spec_v0.1.md)、[`scenario_taxonomy_v0.1.md`](scenario_taxonomy_v0.1.md) 和 [`sim_adapter_handoff.md`](sim_adapter_handoff.md)。

本清单不重定义上述契约。它区分三种状态：

| 标记 | 含义 |
|---|---|
| 已有实现 | 代码已有相应路径；不等于已经在真实 Isaac/T03 运行验证。 |
| 静态阻断 | 按当前调用与类型约定必然无法走通的接缝。 |
| 待真实验证 | 没有运行记录；阶段 1 必须按本文给出的最小判定方法取证。 |

`--env-factory module:function` 的 bootstrap（创建并初始化场景、`RobotInterface`，并将后者注入 env）属于 **sim 侧交付责任**，不是无主的框架空白。它是 `scripts/run_episode.py` 为真实 Isaac adapter 预留、但未交付的启动接缝。框架侧不拥有或改写 IK、真实物理读取及场景初始化逻辑。

旧 `r1_bimanual_dataset/run_aligned_t03_batch.py` 没有构造 `IsaacLabR1Adapter` 或调用 `EpisodeRunner`，故它采集的既有结果不能证明下表任何“待真实验证”项。

## 2. `SimAdapter` 五方法偏差

| 方法 | 当前实现位置与行为 | 真实 T03 状态 | 阶段 1 最小验收 / 阻断 |
|---|---|---|---|
| `reset(scenario)` | [`sim_adapter.py:606-662`](../episode_gen/sim_adapter.py#L606-L662) 绑定已打开的 R1/T03 scene、初始化 native handles、重置后读取 `Observation`。 | 待真实验证。未提供 env factory，因而无真实调用记录。 | factory 必须返回已经启动 Isaac、已初始化 `RobotInterface` 且 `env.robot_interface` 指向该实例的 env。调用一次 `reset(n01)`，保存完整 Observation；比较 `qpos/qvel`、object pose 与同一 backend 同 tick 的直接读数。 |
| `step(phase, scenario)` | [`sim_adapter.py:664-681`](../episode_gen/sim_adapter.py#L664-L681) 计算 16D action、下发、推进一周期并读取 Observation。 | **静态阻断。** backend 分支在 [`1298-1305`](../episode_gen/sim_adapter.py#L1298-L1305) 把 16D action 直接传入 `RobotInterface.apply_full_target()`；后者要求完整 `dof_count` vector（[`robot_interface.py:357-359`](../r1_bimanual_dataset/core/robot_interface.py#L357-L359)）。 | 阶段 1 的指定修补：backend 分支先调用既有 [`_expand_action_to_live_dofs()`](../episode_gen/sim_adapter.py#L460-L488)，再调 `apply_full_target()`；记录 `action_from_full(full_target)` 并与原始 16D action 对比。不得新写转换规则、不得改 `apply_full_target()` 签名。 |
| `render_cameras()` | [`sim_adapter.py:1404-1438`](../episode_gen/sim_adapter.py#L1404-L1438) 查找 env sensor 或包装既有 camera prim，取 RGB。 | 待真实验证。旧 runner 使用独立 `CameraRecorder`，没有证明本方法可读到三路 RGB。 | `reset` 后和至少一次 `step` 后分别调用；`front`、`left_wrist`、`right_wrist` 三键必须都存在，每帧为 640×480 RGB，且逐 tick 不是同一空/冻结缓冲。记录 shape、dtype、逐帧哈希/像素差及实际 API 来源。 |
| `check_place_success(obs, scenario)` | [`sim_adapter.py:1476-1498`](../episode_gen/sim_adapter.py#L1476-L1498) 只根据 Observation 和 adapter 跟踪状态检查双侧有效抓取、离开支撑、未掉落、目标区、释放与稳定。 | 逻辑已有；其输入依赖 grip force、object pose/velocity 和 release 状态，尚未真实验证。 | 阶段 1 smoke 只记录各前置布尔值，不把返回 `True` 作为成功声明。阶段 2 完整 episode 再验证其结果与 `EpisodeRunner` 终态一致。 |
| `check_timeout(obs, scenario)` | [`sim_adapter.py:1500-1510`](../episode_gen/sim_adapter.py#L1500-L1510) 比较 adapter 维护的 `t` 和 scenario/env timeout。 | 纯数据逻辑可调用；真实 tick 时间尚未比对。 | 记录连续 Observation 的 `t` 差；应为 `1 / control_hz = 1/30 s`，并以配置 timeout 触发一次受控检查（不做批量）。 |

### 2.1 `step()` 的 backend 接缝

以下两条转换已经存在，阶段 1 必须复用：

```text
16D action --_expand_action_to_live_dofs()--> full articulation target
full articulation target --RobotInterface.action_from_full()--> 16D dataset action
```

`_expand_action_to_live_dofs()` 已在 [`sim_adapter.py:460-488`](../episode_gen/sim_adapter.py#L460-L488) 实现；反向投影已有 [`robot_interface.py:376-378`](../r1_bimanual_dataset/core/robot_interface.py#L376-L378) 与其 `action_indices`。本次不授权替代转换实现。

## 3. `Observation` 字段与 detector 前提

坐标约定若未特别说明，adapter 当前读取的是 Isaac 运行时 world frame；位置单位应为 **m**，时间为 **s**。任何“应”为单位的项都要由阶段 1 probe 实测确认，不能以源码注释代替记录。

| 字段 | 当前来源 | detector 预期 / 当前偏差 | 阶段 1 最小判定方法 |
|---|---|---|---|
| `t` | adapter 每次 `_advance_sim(..., update_time=True)` 加 `1/30`。 | 所有事件的时间戳，单位 s。 | 连续记录至少 3 个 `step`；`Δt` 必须为 `0.033333...s`，并与真实 app update 数相符。 |
| `qpos`, `qvel` | 注入 backend 时由 [`state16()`](../r1_bimanual_dataset/core/robot_interface.py#L265-L271) 读取；adapter 在 [`683-723`](../episode_gen/sim_adapter.py#L683-L723) 投入 Observation。 | 必须是 `JOINT_ORDER` 的 16D 数据集顺序。detector 当前不直接使用，但 writer 与 action 对齐依赖它。转动关节的单位应为 rad/rad·s⁻¹；夹爪轴是否为 m/m·s⁻¹需保留原 articulation 语义。 | 同 tick 读取 backend `state16()` 并逐元素比较；记录 joint name/order、长度 16、最大绝对误差和各 DOF 单位来源。 |
| `left_ee_pos`, `right_ee_pos` | backend `eef_tip_pose(side)` 优先，其次 native EEF wrapper（[`746-797`](../episode_gen/sim_adapter.py#L746-L797)）。 | world-frame m。为 `dist_to_grasp_frame` 的输入；detector 不直接读取。 | 与 `RobotInterface.eef_tip_pose()` 的位置分量逐轴比对；固定同一 physics tick，误差阈值须先记录而非隐式接受。 |
| `object_pos` | backend `object_pose()` 优先（[`799-825`](../episode_gen/sim_adapter.py#L799-L825)）。 | R16 以它和 scenario 初始 pose 的欧氏距离（m）判定。若读取失败会静默退回 YAML pose，可能伪造“未位移”。 | 与 `RobotInterface.object_pose()` 的位置逐轴比对；reset 后还要比对 `object_pos` 与 scenario pose，若 adapter 保留 prepared live pose 而二者不一致，标记 R16 语义不成立并停止，不补偿。 |
| `object_vel` | backend `object_velocity()` 的前三个分量优先（[`827-841`](../episode_gen/sim_adapter.py#L827-L841)）。 | R19 的 event detail 和 place stability 使用；应是 world-frame linear velocity（m/s）。 | 同 tick 比较 adapter 值与 backend 6D velocity 的 `[0:3]`；静止期间范数接近零且物理步后能变化。 |
| `contacts` | env/scene `get_contacts`、接触对字段或已有 sensor；再归一化为短 link name（[`844-967`](../episode_gen/sim_adapter.py#L844-L967)）。 | R11 要 `{("left_arm_link6", "table")}` 等；R12 要 `.../object`；R14 要左右 arm 名。当前 adapter 不创建 sensor，也不消费旧 runner 的 `GripperContactTracker`，所以真实可得性未知。 | 在已存在的 contact API/sensor 上读取一次无接触和一次可确认接触事件；记录原始 prim/path、adapter 归一化 pair、时间戳。只有归一化结果能命中 detector 所需 pair 才算通过；若实际接触仅发生于 gripper link 而 taxonomy 要求 `arm_link6`，列为契约偏差，由我方决定，不擅自改 detector 或伪造 link 名。 |
| `dist_to_grasp_frame` | adapter 计算 EEF 到 bbox center + config offset 的欧氏距离（[`1157-1188`](../episode_gen/sim_adapter.py#L1157-L1188)、[`1360-1378`](../episode_gen/sim_adapter.py#L1360-L1378)）。 | R11 要每 arm 的 m 距离并和 `object_distance_at_contact` 比较。数值语义直接依赖下文“抓取几何归属”决策。 | 使用冻结的同一物体 bbox、EEF pose、scenario offset 离线复算，比较两 arm 值；在几何归属未决前，只验证单位和可复现性，不宣称 R11 阈值已物理校准。 |
| `grip_force` | backend `get_grip_force`/`get_gripper_force`，否则 env、映射或 joint effort fallback（[`969-1014`](../episode_gen/sim_adapter.py#L969-L1014)）。 | R13 以 N 与 `grip_force_min` 比较。`RobotInterface` 不暴露上述 force API；joint effort 不自动等于夹持力 N。 | 分别读闭合前/闭合后双侧值，记录调用 API、原始单位、是否由 calibration 转为 N。只有可追溯的 N 值才允许 R13/R15 物理验收；若仅有 torque/effort，报告缺口，不能把它标为 N。 |
| `joint_limit_margin` | 从 live robot data 的 joint limits 减 qpos 得到（[`1016-1038`](../episode_gen/sim_adapter.py#L1016-L1038)）。 | R18 需要每个 `JOINT_ORDER` joint 的剩余 margin；其单位沿每个 joint 的 position unit，不能把线性夹爪与角关节混为统一 m。 | 同 tick 用 articulation limits 和 `state16()` 独立复算；记录 16 个键、最小 margin、对应 joint 和单位。 |
| `extra["gripper_closed"]` | `_is_closed()` 根据 qpos 与 command 阈值填充。 | R13 的前置条件。 | 与实际 4 个 gripper axis state 对照，记录关闭阈值和闭合前后布尔值。 |
| `extra["left/right_grasp_alignment_error"]` | adapter 不计算；只允许 env diagnostics 注入。 | **R12 必需**，当前默认缺失时 detector 会读 `0.0`，不会触发。 | 查询 env `get_observation_extra` / `get_diagnostics` 是否提供这两个 m 值；若无，R12 标为不可验证缺口。 |
| `extra["left/right_grasp_slip_m"]` | adapter 不计算；只允许 env diagnostics 注入。 | **R15 必需**，单位 m。 | 在闭合与抬升期间读取，确认有相对 gripper/object 的累计位移定义与 m 单位；无该诊断即 R15 不可验收。 |
| `extra["left/right_approach_stall_s"]` | adapter 不计算；只允许 env diagnostics 注入。 | **R17 必需**，单位 s。 | 记录连续 approach ticks 的该字段及来源；无累计停滞时间即 R17 不可验收。 |
| `extra["lift_stalled"]` | adapter 不计算；只允许 env diagnostics 注入。 | **R19 必需**；detector 仅读取 bool。 | 在 LIFT 期间读取该 flag 及其 object-height/velocity 判据；无来源即 R19 不可验收。 |
| `extra["grasp_valid"]`, `object_left_support`, `object_dropped`, `object_in_target`, `release_stable_s` | adapter 基于闭合/force、物体运动和 place target 推导（[`1328-1353`](../episode_gen/sim_adapter.py#L1328-L1353)、[`1476-1498`](../episode_gen/sim_adapter.py#L1476-L1498)）。 | `check_place_success()` 的输入，不是 R11–R19 的直接 detector 输入。 | 将每项 source 值与其 underlying pose/velocity/force 记录在同一 smoke trace；没有该 trace 时不得以其推导成功。 |

## 4. 待我方决策：抓取几何与轨迹唯一归属

当前存在两套重叠实现。阶段 1 前必须由我方选择一侧；Codex 不自行选择，也不会在阶段 0 改动任何代码。

### 选项 A：由 `IsaacLabR1Adapter` 自算

使用 `IsaacLabR1Adapter._grasp_targets()` / `_approach_vector()` / `_ik_result()` / `_make_action()`（[`sim_adapter.py:1157-1296`](../episode_gen/sim_adapter.py#L1157-L1296)）。

| 项目 | 内容 |
|---|---|
| 改动范围 | 阶段 1 仅修 bootstrap 注入和既定 full-DOF 转换；不引入 sim 侧 nominal/controller/trajectory。 |
| 优点 | 它与 `SimAdapter.step(Phase, ScenarioConfig)` 一对一，阶段状态机、Observation 和 retry 的拥有者单一；接口表面最小。 |
| 风险 | 没有复用旧 T03 已验证物理 runner 的 `NominalGraspGenerator`、`BimanualController`、`JointTrajectory`。其 phase-wise IK、pregrasp/轨迹时序及 `coordination_mode` 行为需要从零进行真实验证。 |

### 选项 B：在既有 `IsaacLabR1Adapter` 内复用 sim 侧抓取几何与轨迹

adapter 内部包装 `NominalGraspGenerator.generate()`、`BimanualController.build()` 和 `JointTrajectory.sample()`，并仍由 adapter 将 FSM phase 转译为对应轨迹片段；不创建第二个 adapter。

| 项目 | 内容 |
|---|---|
| 改动范围 | 修改仅限 `episode_gen/sim_adapter.py` 与 sim bootstrap；需要定义框架 `ScenarioConfig` 到现有 controller config/scene config 的显式字段映射，以及 RETREAT/REAPPROACH/REGRASP 如何形成轨迹片段。 |
| 优点 | 直接使用旧 T03 runner 已用于真实物理的 bbox nominal grasp、完整 DOF trajectory、夹爪相位控制和 IK 路径，最接近既有有效执行链。 |
| 风险 | 两种 scenario schema、阶段枚举与 retry 动作没有现成一一映射；若不先冻结映射，容易将旧 `failure_mode` 预制扰动重新带回 N/R/F/P 链路。必须保持失败由 detector/recovery 的真实 Observation 驱动。 |

**需要我方的明确选择：A 或 B。** 选择前，阶段 1 不开始实现任何几何/轨迹归属改动。

## 5. 阶段 1 不可越过的验收记录

一次且仅一次真实 T03 smoke 必须产出以下原始记录，才可作为“Observation 字段完整”的证据：

1. bootstrap 的实际 backend 类型、`robot_interface` 注入位置、live DOF 数、16D joint order；
2. reset 与至少 3 个 step 的完整 Observation JSON（含每项 `extra` 键）；
3. 每步 full articulation action 和 `action_from_full(full_action)` 产生的 16D action，以及二者的映射核对；
4. 三路 RGB 的 API 来源、shape、dtype 和相邻帧差异；
5. contacts/grip force 的原始读取和归一化/单位证据；
6. 不可取得或单位不可证明的字段，必须逐项报告为缺口，不能以默认值伪装为真实观测。

本阶段不运行 batch、不生成数据集，且不以旧 `episode_000001` 为任何验证或迁移基准。
