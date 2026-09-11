# Isaac Sim 对接说明（交给 sim 同学）

状态：**`IsaacLabR1Adapter` 已经由 sim 同学实现并合并进本仓库**（约1400行，真实场景 T03，
详见 `docs/architecture.md`），下面关于接口契约的部分仍然有效、仍然是权威说明；"还卡住的
部分：场景USD"那一节已经过时（信息已经拿到，见 `docs/scene_grounding_t03.md`），保留只是
为了留痕，不代表还需要做。现在真正待收集的信息清单见 `docs/sim_data_request_v1.md`。

## 你要做的事，边界在哪

只实现一个类：`episode_gen/sim_adapter.py::IsaacLabR1Adapter`，让它满足同文件里
`SimAdapter` 这个抽象基类的接口。**不要改** `episode_gen/` 下任何其他文件（`fsm.py`
/ `detectors.py` / `recovery.py` / `episode_runner.py` / `dataset_writer.py`）——
这些已经用 `MockSimAdapter` 跑通单元测试验证过控制流是对的
（`tests/test_r11_recovery_loop.py`），你接的 adapter 只要遵守接口契约，直接免费拿到
状态机、异常检测、恢复重试、数据落盘这一整套，不需要重新实现。

## 接口契约（必须实现的 5 个方法）

```python
class SimAdapter(ABC):
    control_hz: float = 30.0  # 必须是 30，见 dataset_spec_v0.1.md

    def reset(self, scenario: ScenarioConfig) -> Observation: ...

    def step(self, phase: Phase, scenario: ScenarioConfig) -> tuple[Observation, list[float]]:
        """推进一个控制周期，返回 (最新Observation, 本步实际下发的16D action)"""

    def render_cameras(self) -> dict[str, Any]:
        """返回 {'front': rgb, 'left_wrist': rgb, 'right_wrist': rgb}"""

    def check_place_success(self, obs, scenario) -> bool:
        """TRA-01 §7 成功判据"""

    def check_timeout(self, obs, scenario) -> bool: ...
```

`Observation` / `AnomalyEvent` 的字段定义在 `episode_gen/types.py`，字段名不要改——
`detectors.py` 里 R11-R14 的判定逻辑是按这些字段名写的（比如
`obs.contacts` 里要有 `("left_arm_link6", "table")` 这种 pair，`obs.dist_to_grasp_frame["left"]`
要是米制距离）。具体每个字段谁用、怎么用，直接看 `episode_gen/detectors.py`
里 `detect_r11`-`detect_r14` 四个函数，比文字描述准确。

## 已经核实、不需要你再确认的部分

来自 `userguide-galaxea/galaxea_lab` 源码，已经写进 `episode_gen/sim_adapter.py` 常量里：

- `DEFAULT_QPOS` / `JOINT_ORDER`：16D 关节顺序与默认值，对应
  `omni/isaac/lab_assets/galaxea_robots.py :: GALAXEA_R1_CFG`
- `CAMERA_PRIM_PATHS`：三路相机的 prim path，对应
  `galaxea/manager_based/lift/config/joint_pos_env_cfg.py :: R1LiftEnvCfg`
- 相机必须显式覆盖成 **640×480, update_period=1/30**（源码里默认是 240×320 或
  `1/60`，直接用默认值会不满足 dataset_spec_v0.1.md §3）

## 还卡住的部分：场景 USD

Crew Lock Bag / 桌子 / 绿色目标环这三个物体在场景里的 prim path、bbox、初始
pose，我们这边还没拿到（`D:\robotics\spacerobot.usd`，本地文件，队友本机）。
在这个信息同步之前，`reset()` 里 spawn/摆放物体这部分没法写死路径，建议你直接
用你们本地能访问的那份 USD 走，把用到的 prim path 记下来回传给我们更新
`ScenarioConfig` 里 `object_x/y/z` 等字段的默认取值范围。

## 建议的实现顺序（复用 galaxea_lab 里已有的写法，不用发明新的）

1. 场景搭建：参考 `galaxea/manager_based/lift/config/joint_pos_env_cfg.py` 里
   `R1LiftCubeEnvCfg` 的写法，用 `GALAXEA_R1_HIGH_PD_GRIPPER_CFG` 作为 robot，
   加 table/object/target_ring 的 `AssetBaseCfg`/`RigidObjectCfg`
2. 抓取点计算：物体 bounding box（Isaac Sim `compute_obb`）+
   `scenario.left_grasp_offset`/`right_grasp_offset` → nominal grasp frame，
   IK 解 16D 关节目标（`galaxea_lab` 里已经在用 `DifferentialIKController`，
   直接复用）
3. 碰撞检测：`ContactSensorCfg` 挂在 `left_arm_link6`/`right_arm_link6`/桌面/
   物体上，喂进 `Observation.contacts`（R11/R14 直接依赖这个）
4. 抓力：夹爪 actuator 的 effort/力反馈 → `Observation.grip_force`（R13 依赖）
5. 相机渲染：`scene.sensors[...].data.output["rgb"]`

## 怎么验证接对了

不需要等接完整个数据集才能验证。`configs/scenarios/` 下已经有 4 份写好的
R11-R14 场景配置（真实碰桌面/碰错表面/单侧抓取/双臂互撞），在纯 Python 假仿真
（`MockSimAdapter`）上跑出来的预期行为是：**每个场景只触发一次异常检测 → 一次
RETREAT → 重新接近 → 成功**（`tests/test_r11_recovery_loop.py` 已经锁死这个断言）。

你接完 `IsaacLabR1Adapter` 后，建议照着 `scripts/run_episode.py` 的样子自己写一个
`scripts/run_episode_isaaclab.py`（Isaac Lab 的 app 需要在 import 其他东西之前先
`AppLauncher` 启动，没法直接复用现成的 `run_episode.py`），跑同样这 4 份 config，
预期在真实物理下应该也是"异常→退回→重试→成功"这个模式（允许物理细节不同，
但成功与否、recovery_count 应该匹配）。如果某个场景不是这个模式，大概率是
`Observation` 里某个字段没填对（比如 contacts 的 link 命名对不上
detectors.py 里期望的 `f"{arm}_arm_link6"`），照着 detectors.py 对一下命名。

## 明确不要做的事

- 不要在 `IsaacLabR1Adapter` 之外的地方加"if scenario_id == 'R11'"这种特判——
  违反 taxonomy 的设计原则（见 `docs/scenario_taxonomy_v0.1.md` §0）
- 不要改 `dataset_writer.py` 的 policy/diagnostics 字段分离逻辑
- 不要为了让某个测试通过而放宽 `detectors.py` 的判定阈值——阈值是 config
  (`ScenarioConfig.anomaly`) 里的字段，改配置，不改代码
