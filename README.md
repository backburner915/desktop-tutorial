# Dual-Arm Grasp Episode Generation Framework

一个可配置的双臂抓取数据生成框架，用于给 Galaxea R1（带底盘、双臂、三路 RGB 相机）批量生成
符合 TRA-01 数据契约的 imitation-learning episode，覆盖正常 / 异常恢复 / 失败 / 边界扰动
四类情境。设计动机与需求对齐记录见对话历史；正式文档见下。

## 唯一工作基线：`claude/gifted-carson-67qkgn`

本分支是双方代码的合并基线，同时包含：

- `episode_gen/` 等（框架侧）— 场景配置、N/R/F/P 分类、检测、recovery/retry 状态机、数据落盘
- `r1_bimanual_dataset/`（sim 侧）— Isaac Sim 接入、articulation 控制、IK、抓取几何、轨迹

其他分支不要用于融合工作：`main` 为空；`claude/dual-arm-grasp-framework-84hhi9` 与
`claude/gifted-carson-67qkgn` 合并前的状态等价，只含框架侧；`sim端同步` 是 sim 代码的早期
上传，所有文件被拍平到仓库根目录、丢失了 `core/`、`tools/` 层级，import 会失败，**不要使用**。

两侧当前尚未在运行时接通：`r1_bimanual_dataset` 不引用 `episode_gen`。接缝按设计应在
`episode_gen/sim_adapter.py::IsaacLabR1Adapter` 与 `r1_bimanual_dataset/core/robot_interface.py`
之间（前者按名字探测后端的 `state16` / `solve_ik`，后者恰好提供这两个方法）。

## 文档

- [`docs/scenario_taxonomy_v0.1.md`](docs/scenario_taxonomy_v0.1.md) — N/R/F/P 分类定义，每类的触发条件与配置轴
- [`docs/dataset_spec_v0.1.md`](docs/dataset_spec_v0.1.md) — 已冻结的 TRA-01 数据契约在本仓库的落地方式

## 架构

```
episode_gen/
  fsm.py            # 共享状态机阶段定义（唯一一份，所有情境复用）
  scenario.py        # ScenarioConfig：一个episode的全部可变参数，从YAML加载
  types.py           # Observation / AnomalyEvent：仿真无关的纯数据类型
  detectors.py        # R11-R14 异常检测谓词（(scenario, phase, obs) -> AnomalyEvent|None）
  recovery.py         # RecoveryManager：检测到异常后retreat+调整参数重试，超限转FAILURE
  sim_adapter.py       # SimAdapter接口；MockSimAdapter(可测试)；IsaacLabR1Adapter(骨架，见下)
  episode_runner.py     # 主循环：把上面几个部件粘起来，逐帧写数据
  dataset_writer.py      # 按dataset_spec写episode.jsonl(policy输入) + diagnostics.jsonl(仿真标注)
  batch_generate.py      # 一份基准config -> 批量采样出N个episode config

configs/scenarios/*.yaml  # 人工编写的基准scenario（N01, R11-R14）
tests/                    # 用MockSimAdapter跑通的单元测试，纯Python，无需Isaac Sim
```

新增一个情境（比如"R11但碰的是柜子而不是桌子"）应该只需要写一个新的 `configs/scenarios/*.yaml`
（或者用 `batch_generate.py` 批量采样），**不需要**新建 Python 文件。只有真正引入新的检测逻辑
（比如 taxonomy 里排期中的 R15-R19）才需要在 `detectors.py`/`recovery.py` 里加函数。

## 运行测试（本仓库内，无需 Isaac Sim）

```bash
python3 -m unittest discover -s tests -v
```

`tests/test_r11_recovery_loop.py` 用 `MockSimAdapter`（纯 Python 运动学占位，不是物理引擎）实际跑通了：

- R11-R14 各自的"异常检测 → RETREAT → 调整参数重试 → 成功"闭环，且每个场景只触发一次恢复
- 恢复次数超限（`max_recovery_attempts`）正确转为 `FAILURE`，标记 `F01_recovery_exhausted`
- N01 基线场景全程无异常、无 recovery
- 逐帧数据满足 `docs/dataset_spec_v0.1.md`：state/action 均为 float[16]、帧间隔 ≈33.333ms（30Hz）、
  policy 输入字段与 diagnostics 严格分离
- `batch_generate.py` 从一份 R11 基准 config 采样出的多组变体（不同碰撞臂、不同碰撞距离/退回距离/
  重接近角度）全部能正确走完恢复闭环

这些验证的是**框架的控制流正确性**，不是物理真实性——`MockSimAdapter` 用线性插值占位关节角，
不做真正的 IK/碰撞仿真。

## 接入真实 Isaac Sim（尚未完成，需要你在有 Isaac Sim 的机器上做）

`episode_gen/sim_adapter.py::IsaacLabR1Adapter` 是骨架，已经把从 `userguide-galaxea/galaxea_lab`
源码核实过的部分预填好（16D 关节顺序与命名、三路相机的 prim path、机器人默认关节角），其余标了
`TODO` 需要对照你的 Isaac Lab 版本和实际场景 USD 补上，重点是：

1. 场景 USD 里 Crew Lock Bag / 桌子 / 绿色目标环的 prim path 与初始 pose（当前未知，见下）
2. bounding-box → nominal grasp frame → IK 的具体实现（可复用 `galaxea_lab` 里已有的
   `DifferentialIKController` 用法）
3. `ContactSensorCfg` 接入，把真实碰撞事件填进 `Observation.contacts`
4. 相机分辨率/帧率覆盖为 640×480@30Hz（`galaxea_lab` 现有配置默认是 240×320@60Hz，必须显式覆盖）

## 待办

- [ ] 场景 USD（`spacerobot.usd`）尚未同步进本仓库，`object_x/y/z` 等默认值目前是占位值，需要替换成
      真实物体/目标环的可达范围
- [ ] R15-R19、F02、P21-P24：按 taxonomy 文档排期，等 R11-R14 在真实 Isaac Sim 里验收通过后再实现
- [ ] `dataset_writer.py` 目前是本地 JSONL 占位实现，接入真实 Isaac Sim 后需要换成真正的
      `lerobot.common.datasets.lerobot_dataset.LeRobotDataset` 写入（接口不变，见该文件顶部注释）
