# 架构说明

这份文档是为了回答一个反复被问到的问题："这一堆文件里，真正被反复调用、不用为每个
episode 重写的那个核心到底是哪个？" 直接给结论，再逐层说明。

## 结论：母脚本是 `episode_gen/episode_runner.py::EpisodeRunner.run()`

它是唯一一个"每个 episode 都跑一遍、内容完全一样"的函数。跑 N01 和跑 R11 调用的是
**同一份** `run()` 代码，区别只在于传进去的 `ScenarioConfig` 不同。它自己不认识
"R11"这个字符串，也不认识"桌子"这个词——它只做四件通用的事：

1. 按 `fsm.py::NOMINAL_NEXT` 表推进阶段（RESET→PREGRASP→...→SUCCESS）
2. 每步调用 `detectors.py::run_detectors()`，问一句"这步有没有异常"
3. 有异常就交给 `recovery.py::RecoveryManager.handle()`，问一句"该怎么办"
4. 每步把结果写进 `dataset_writer.py::EpisodeWriter`

## 各模块的完成度（如实列出，不夸大）

| 层 | 文件 | 做的是什么 | 完成度 |
|---|---|---|---|
| 编排/母脚本 | `episode_runner.py` | 状态机主循环，对所有 episode 通用 | ✅ 完成，单测覆盖 |
| 阶段定义 | `fsm.py` | 12 个 Phase + 顺序表，对所有 episode 通用 | ✅ 完成 |
| 配置 | `scenario.py` | `ScenarioConfig`/`AnomalyConfig` 数据结构 + yaml 读写 | ✅ 完成 |
| 异常检测 | `detectors.py` | R11-R19 九个判定函数 + F02 二次异常检测 | ✅ taxonomy里定义的R类已全部实现，控制流层面单测覆盖 |
| 恢复策略 | `recovery.py` | 通用的重试次数/超限判定 + 9 个 class 专属的参数调整逻辑 | ✅ 同上 |
| 数据落盘 | `dataset_writer.py` | 按 TRA-01 写 policy/diagnostics 两路 | ✅ 结构完成；⚠️ 目前是 JSONL 占位，真实 LeRobotDataset(视频编码+parquet) 未接 |
| 仿真适配 | `sim_adapter.py::MockSimAdapter` | 纯 Python 假仿真，只为测试母脚本逻辑 | ✅ 完成（但这不是最终产物） |
| 仿真适配 | `sim_adapter.py::IsaacLabR1Adapter` | 真正接 Isaac Sim、驱动真机器人 | ❌ 骨架，未实现，见 `sim_adapter_handoff.md` |
| 批量配置 | `batch_generate.py` | 从一份基准 yaml 采样出很多变体 yaml（**不是母脚本，是配置工厂**） | ✅ 完成 |
| CLI 入口 | `scripts/run_episode.py` / `run_batch.py` | 调 `EpisodeRunner`，串起 config→adapter→输出 | ✅ 完成 |

## 数据流一图流

```
configs/scenarios/r11_xxx.yaml  ──┐
（人工写1份，或batch_generate.py  │
 批量采样出几百份）                │
                                  ▼
scripts/run_batch.py 循环读取每份config
                                  │
                                  ▼
        EpisodeRunner.run(scenario, adapter)   ← 这是母脚本，对每份config都调这一份代码
             │         │         │
             ▼         ▼         ▼
        fsm.py    detectors.py  recovery.py     ← 通用调度；R11-R14的差异化逻辑分别封装在这两个文件内部
             │
             ▼
        SimAdapter.step(phase, scenario)        ← 真正"干活"的地方：Mock=假仿真验证逻辑；IsaacLabR1Adapter=真物理(待实现)
             │
             ▼
        EpisodeWriter.append_frame(...)         ← 每步落盘：policy帧 + diagnostics帧分离
             │
             ▼
   out/episodes/<scenario_id>/
     episode.jsonl        ← 未来直接对应 VLA 训练要读的东西
     diagnostics.jsonl     ← 只给我们自己用，不进训练
     episode_meta.json     ← success/failure_reason/recovery_count
```

## "母脚本是否已经完成"该怎么回答

分两层说，不能笼统说"完成"或"没完成"：

- **编排层（母脚本本身）**：完成了。`episode_runner.py` 不需要为任何新场景改代码，
  新增场景永远只是加 yaml（属于同一 R-class）或加一个新的 detector/recovery 函数
  （属于全新 R-class，比如以后做 R15）。这是老师要求的"少数通用控制逻辑 + 大量配置"
  已经落地。
- **执行层（真正让机器人动起来产生数据）**：没完成。`MockSimAdapter` 证明的是
  "如果仿真器按承诺的接口给我数据，母脚本能正确处理"，它本身不产生任何可用于训练
  的真实数据。这一层等 `IsaacLabR1Adapter` 接完才算完成，这是当前最大的缺口。

所以"脚本还没到母脚本的程度，需要完善"这句话，准确的说法是："母脚本（编排逻辑）已经
写完并且验证了内部一致性；缺的是驱动真实物理的执行层"——不是编排逻辑本身需要推倒重做。
截至目前 taxonomy 定义的 R11-R19、F01-F03、P21-P23 在控制流层面（`MockSimAdapter`）
均已实现并有单测覆盖，见 `docs/scenario_taxonomy_v0.1.md` 的实现状态列。

## sim 端分支整合情况（2026-09-10 记录）

sim 同学把本地工作目录打包上传到了 `sim端同步` 分支（`git fetch` 才能看到，默认
clone 不带）。检查后发现：

- **不是另起炉灶**：`episode_runner.py`/`scenario.py` 内容和我们这边基本一致，说明
  是在我们的框架基础上做的，不是平行设计。
- **`IsaacLabR1Adapter` 已经从骨架写成了约1400行的真实实现**（配套 `robot_interface.py`
  726行），控制方式是阻尼最小二乘 Jacobian IK + 关节空间轨迹执行，抓取点从物体
  bbox 在局部坐标系里现算——这条路子和我们最初设计的"bbox→grasp frame"是一致的。
- **上传方式导致目录结构被拉平**：92个文件都在仓库根目录，没有 `episode_gen/`
  文件夹。但 `sim_adapter.py` 内部的 import 语句仍然是 `from episode_gen.xxx import
  yyy`，说明这大概率是上传方式（网页拖拽）把文件夹层级弄丢了，不是故意改的设计。
  **已确认的合并方向：sim 端后续改成我们这边的 package 结构**，双方后续都用
  `git push` 保留分支历史，不再用整体打包上传。
- **上传内容内部不完全一致**：例如 `sim_adapter.py` 引用
  `from episode_gen.types import Observation`，但同一次上传里的 `types.py` 并没有
  `Observation` 这个类，定义的是另一套字段完全不同的类（`Phase`/`Outcome`/`Pose`/
  `NominalGrasp`/`EpisodeResult`）。大概率是把开发过程中不同阶段的文件一起打包
  上传了（那一堆 `calibrate_*`/`probe_*`/`audit_*` 脚本也印证这点，都是调试期间
  的一次性脚本）。**在双方确认清楚哪些文件是当前有效版本之前，不要贸然自动合并**
  ——正在跟 sim 同学对齐这件事，见任务 #13。
- **和当前 taxonomy/dataset_spec 有实质冲突、需要澄清的点**：sim 端 `config.py` 里
  `DATASET_FPS = 10.0`（TRA-01 要求30Hz）；目标物体是占位正方体 T01/T03，不是
  Crew Lock Bag（据用户确认：这批占位物体本身没有设计抓取点，用于先跑通pipeline
  机制，不是最终任务定义）；任务文本也是占位通用句，不是 TRA-01 冻结的那句。这些
  在合并真实 adapter 代码之前需要先谈清楚，不然合并进来的坐标系/契约细节可能是
  错的。
