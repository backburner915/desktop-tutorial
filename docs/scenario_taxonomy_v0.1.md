# Scenario Taxonomy V0.1

状态：草案，待确认后冻结为 V1.0。
适用对象：Galaxea R1 双臂（带底盘）机器人，单物体双臂协同抓取任务（Crew Lock Bag → 绿色目标环）。

## 0. 设计原则

这份文档不是"第11号脚本长什么样"，而是"第11类情境的边界条件是什么"。每个 ID 只定义：

1. **trigger**：什么仿真条件触发这个分类（可用代码判定的谓词，不是文字描述）
2. **config 轴**：这个分类下可以自由采样、从而衍生出成百上千个 episode 的参数
3. **recovery**（仅 R 类）：检测到异常后要执行的恢复策略族

所有分类共享同一份 [`episode_gen/fsm.py`](../episode_gen/fsm.py) 状态机与同一份 [Dataset Spec](./dataset_spec_v0.1.md)。新增一个 ID **不允许**新建脚本文件，只允许新增一份 `configs/scenarios/*.yaml` 和（如需要）在 `episode_gen/detectors.py` / `episode_gen/recovery.py` 里加一个 trigger/recovery 函数。

FSM 阶段（完整版，含 TRA-01 要求的放置目标环）：

```
RESET → PREGRASP → APPROACH → CONTACT_CHECK → GRASP → LIFT → HOLD
      → MOVE_TO_TARGET → PLACE → RELEASE → SUCCESS
任意阶段 → (异常检测) → RETREAT → 重新规划目标 → 回到 APPROACH（或更早阶段）
恢复次数超限 → FAILURE
```

---

## 1. Family N：Normal（正常成功轨迹）

条件分布采样，不是十套不同算法。共享的 config 轴：

`object_x/y/z`、`object_roll/pitch/yaw`、`left_start_pose`、`right_start_pose`、`left_grasp_offset`、`right_grasp_offset`、`left_approach_angle`、`right_approach_angle`、`pregrasp_distance`、`lift_direction`、`target_ring_pose`。

| ID | 条件 | 说明 |
|---|---|---|
| N01 | nominal 标准抓取 | 所有参数取默认值，作为基线 episode |
| N02 | object x/y/z 偏移 | 物体初始位置在工作空间内随机偏移 |
| N03 | object roll/pitch/yaw 偏移 | 物体初始姿态小角度旋转 |
| N04 | 左右臂初始姿态变化 | `left_start_pose`/`right_start_pose` 在关节空间内小范围采样 |
| N05 | grasp offset 变化 | 抓取点相对 bounding-box 生成的 nominal grasp frame 有偏移 |
| N06 | approach angle 变化 | 接近方向角度采样 |
| N07 | pregrasp distance 变化 | 预抓取距离采样 |
| N08 | 左右臂到达时间轻微不同步 | 两臂 APPROACH→GRASP 的时间偏差在容忍范围内 |
| N09 | 工作空间边缘抓取 | 物体位置接近可达域边界但仍可达 |
| N10 | 接近路径靠近桌面但不发生碰撞 | 轨迹 clearance 小但保持在安全阈值之上 |

## 2. Family R：Recovery（异常后恢复成功）

这是当前阶段的重点。每个 ID = 一个 `(检测条件, 恢复策略)` 的**闭包**，不是一段固定录像。同一个 ID 通过改配置可以扩展出：碰桌面/碰柜体/碰障碍物/碰物体侧面、左臂/右臂、不同碰撞位置和程度等几十种变体，程序不用重写。

| ID | 异常 | 检测条件（谓词，见 `episode_gen/detectors.py`） | 恢复策略（见 `episode_gen/recovery.py`） | 首批实现 |
|---|---|---|---|---|
| R11 | 提前碰撞（premature_surface_contact） | phase==APPROACH 且 `contact(arm, surface)==True` 且 `dist_to_grasp_frame(arm) > d_safe` | RETREAT 到安全位 → 按 `retry_approach_angle`/`retry_grasp_offset` 重新规划 → 回 APPROACH | ✅ 优先 |
| R12 | 接触物体错误表面（wrong_surface_contact） | phase==APPROACH/CONTACT_CHECK 且 `contact(arm, object)==True` 且 grasp 对齐误差 > 阈值 | 松开 → 后退 → 调整 `grasp_offset` → 重新 APPROACH | ✅ 优先 |
| R13 | 单侧夹爪抓住（single_side_grasp） | phase==GRASP 结束时仅一侧 `grip_force(arm) > f_min` | 保持已抓侧 / 视 config 决定是否释放 → 未成功侧单独重新 APPROACH | ✅ 优先 |
| R14 | 两臂互撞（arm_arm_collision） | 任意阶段 `contact(left_link, right_link)==True` | 双臂同时 RETREAT → 增大 `approach_offset` 后重新规划双侧轨迹 | ✅ 优先 |
| R15 | 夹爪打滑（grip_slip） | phase==LIFT/HOLD 且 object 相对 gripper 的位移持续增长 | 降低/松开 → 回 GRASP 前重新夹紧 | 待排期 |
| R16 | 物体被推走（object_displaced） | phase==APPROACH 且 `object_pose` 偏离初始目标 > 阈值 | 用当前 `object_pose` 重新计算 grasp frame → 重新 APPROACH | 待排期 |
| R17 | 接近但无法到达（stalled_approach） | phase==APPROACH 且 `position_error` 在 `T_stall` 内未下降 | 回 PREGRASP → 更换 `approach_angle` | 待排期 |
| R18 | 关节接近极限（joint_limit_margin） | 任意阶段 `min(joint_limit_margin) < margin_min` | RETREAT → 选择备选姿态/IK 解 | 待排期 |
| R19 | 抬升失败（lift_failed） | phase==LIFT 且 gripper closed 但 object 未随之上升 | 松开 → 重新 GRASP | 待排期 |
| R20 | 恢复超限（recovery_exhausted） | `recovery_count > max_recovery_attempts`（由 `RecoveryManager` 统一判定，不是单独检测器） | 终止 episode，标记为 F 类 failure，`failure_reason="recovery_exhausted"` | ✅ 首批实现（兜底逻辑） |

## 3. Family F：Failure（最终失败）

不是"随便让它撞烂"，而是 R 类恢复升级失败时的落地状态，或直接判定为不可恢复的异常。

| ID | 定义 |
|---|---|
| F01 | 任一 R-family 异常触发 `recovery_count > max_recovery_attempts`（对应 R20） |
| F02 | 恢复过程中触发第二类不同的异常（如 R11 恢复中又发生 R14） |
| F03 | episode 超时（`time_out`）仍未完成 GRASP |

Failure 数据单独落盘（按 TRA-01 §8），默认不进首轮 SFT。`failure_type`/`failure_phase`/`object_pose`/`object_velocity`/`left_contact`/`right_contact`/`left_grasp_error`/`right_grasp_error` 必须记录在 diagnostics 里。

## 4. Family P：Perturbation（边界/扰动，理论上应完成）

在 N 的参数分布基础上取到边界值，但不引入检测器意义上的"异常"——用于测试正常路径在极限条件下的鲁棒性。

| ID | 条件 |
|---|---|
| P21 | `pregrasp_distance` 取分布下限 |
| P22 | `approach_angle` 取分布边界 |
| P23 | 物体姿态旋转取边界值（仍在可抓取范围内） |
| P24 | 双臂时间不同步取容忍上限（N08 的边界版） |

## 5. Scenario ID → Config 映射

一个 ID 对应一份 `configs/scenarios/<id>_<slug>.yaml`。字段结构见 `episode_gen/scenario.py::ScenarioConfig`。`batch_generate.py` 从一份基准 config 采样出同一 ID 下的多个 episode（不同 seed），而不是新建新的 ID。

## 6. 待确认 / 后续排期

- R15-R19、F02、P21-P24：等 R11-R14 的检测器/恢复闭环在真实 Isaac Sim 中跑通验收后再排期实现，避免在没有物理引擎验证的情况下堆参数。
- 障碍物/柜体等碰撞面的 taxonomy 扩展（R11 的"闭包"能力）依赖场景 USD 里实际有哪些可碰撞物体，待场景文件同步后补齐 `surface` 枚举。
