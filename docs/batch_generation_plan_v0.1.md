# 首次大批量数据生成计划 V0.1

状态：草案。当前处于"框架就绪，等 sim 端 adapter 对齐"阶段。这份文档定义：
① 大批量生成之前必须先完成什么（不能跳），② 对齐之后具体分几步跑、每步的
验收门槛，③ 现在卡在哪一步、需要谁去推进。

## 1. 现状盘点：本仓库范围内能做的已经做完

| 项目 | 状态 |
|---|---|
| Taxonomy 分类定义（N/R/F/P） | ✅ 完整，见 `docs/scenario_taxonomy_v0.1.md` |
| R11-R19 + F01-F03 + P21-P23 | ✅ 控制流层面已实现并有单测（11个测试全过，`MockSimAdapter`） |
| 数据契约 | ✅ TRA-01 已固化为 `docs/dataset_spec_v0.1.md` |
| 编排框架（母脚本） | ✅ `episode_runner.py` + detectors/recovery/dataset_writer，见 `docs/architecture.md` |
| 批量生成机制 | ✅ `batch_generate.py`（配置采样）+ `scripts/run_batch.py`（批量执行+汇总报告） |
| N02-N10 具体配置 | ❌ 只有定义，没有 yaml（原因见下） |
| P24 | ❌ 缺字段，做不了（见 taxonomy 文档 §4） |

**唯一真正阻塞"大批量生成真实数据"的**：`IsaacLabR1Adapter` 还没有一个双方确认过、
内部一致、可实际运行的版本（sim 端已经写了约1400行真实实现，但存在文件版本/
目录结构/物体身份/帧率等需要先对齐的问题，见 `docs/architecture.md` 的记录）。
N02-N10 的采样范围之所以还没写，是因为 `object_x/y/z` 这些数值要基于真实场景的
可达工作空间来定，在场景坐标系确认前写了也是白写（这一点在 `docs/architecture.md`
里也记录了：sim 端 preflight 报告里的真实物体坐标和我们占位值完全不是一个量级）。

## 2. 分阶段计划

### 阶段 0（现在 → adapter 对齐前）：本仓库范围内能做的
- [x] taxonomy 定义、R11-R19/F01-F03/P21-P23 控制流实现+验证
- [ ] 与 sim 同学对齐：哪些文件是当前有效版本、目录结构、物体身份（Crew Lock Bag
      vs 占位正方体）、任务文本、帧率（sim 端目前是10Hz，需要改成TRA-01要求的30Hz）
- [ ] 场景坐标系确定后，补 N02-N10 的 yaml + `batch_generate.py` 采样范围、补 P24

### 阶段 1：Adapter 接通，试跑最小闭环（对应 TRA-01"5条成功episode试采"）
**门槛**：`IsaacLabR1Adapter` 合并完成、`py_compile` 通过、能在真实 Isaac Sim 里
`reset()`+`step()` 跑通不报错。

1. 用 `configs/scenarios/n01_nominal.yaml`（物体坐标先按真实场景更新）+
   `scripts/run_episode.py --adapter isaaclab` 跑1条
2. **人工看**（不能只看程序返回 `success=True`）：`episode.jsonl` 的图像是不是
   真的渲染出来了、分辨率对不对、`qpos` 数值是不是在合理范围内变化（不是像
   `MockSimAdapter` 那样全程不动）
3. 扩到 R11-R14 各跑1-2条，确认真实物理下也能表现出"碰撞→退回→重试→成功"，
   不只是 Mock 里的假信号能触发

### 阶段 2：小批量验证（建议20-50条），对应 TRA-01 首轮训练下限
**门槛**：阶段1通过，`object_x/y/z` 等采样范围已按真实场景 bbox 重新标定（不是
占位值）。

1. `batch_generate.py` 对 N01（nominal 采样）+ R11-R14 各生成10-15条
2. `scripts/run_batch.py --adapter isaaclab` 批量跑，产出 `batch_summary.json`
3. **人工抽查不低于10%**的 episode（实际看图像/轨迹，不只看 summary 里的 success
   flag），必须包含至少一个"异常→恢复→成功"的例子
4. 过一遍 TRA-01 §11 验收清单里"不依赖 Isaac Sim 也能查"的那些项：视频 shape、
   帧数一致、state/action 无 NaN、关节未越限、时间戳间隔 ≈33.333ms——这些
   `dataset_writer.py` 已经在写入时强制校验，阶段2主要是确认真实数据也满足

### 阶段 3：正式大批量（覆盖 taxonomy 已实现的全部类别），对应 TRA-01 建议的100-200条
**门槛**：阶段2抽查通过、没有系统性 bug（比如某个 R 类总是恢复失败、图像总是黑屏
之类）。

1. 对 N01-N10（如果都补齐了）+ R11-R19 + P21-P23 分别用 `batch_generate.py` 按
   比例采样
2. 成功:恢复:失败 的比例需要和训练侧对一下（sim 端 `generate_scenarios.py` 里已经
   有 `--success-ratio` 这个参数的先例，可以直接借鉴，不用重新设计）
3. `dataset_writer.py` 从当前 JSONL 占位切换成真正的 LeRobotDataset 写入（接口不变，
   见该文件顶部注释）
4. 跑一遍 SmolVLA 兼容性校验（sim 端已经有 `check_smolvla_compatibility.py`，可以
   直接复用，不用重写）
5. 交给 VLA 同学做首轮 SFT

## 3. 每个阶段"不要跳过"的检查项

- **阶段1→2**：至少一条真实 episode 必须有人亲眼看过图像/轨迹，程序返回
  `success=True` 不能作为唯一依据——Mock 阶段已经证明过"程序逻辑正确"不等于
  "物理上真的抓起来了"
- **阶段2→3**：抽查比例不低于10%，且必须覆盖至少一个 R 类的"恢复成功"和一个
  F 类的"确认失败"，不能只看 N 类
- **任何阶段**：`DATASET_FPS`/相机分辨率/16D关节顺序，必须和
  `docs/dataset_spec_v0.1.md` 逐项核对一致，不一致就停下来修，不能"先采着再说"
  ——之前发现 sim 端草稿里 `DATASET_FPS=10.0` 就是一个必须在阶段1之前修掉的例子

## 4. 现在卡在哪一步、需要谁做什么

| 谁 | 要做的事 |
|---|---|
| 你 | 和 sim 同学确认"上传的文件哪些是当前有效版本"，推进目录结构对齐到我们的 package 结构 |
| sim 同学 | 确认有效版本；把 `DATASET_FPS` 改成30；确认物体是继续用占位正方体验证 pipeline，还是已经可以切到 Crew Lock Bag |
| 我这边 | 对齐信息到位后：协助/执行 `IsaacLabR1Adapter` 的合并与联调（阶段1），以及 N02-N10 配置补全（阶段0剩余项）——这两项现在就可以在对齐信息陆续到位的过程中并行推进，不需要等全部信息100%到齐才开始 |
