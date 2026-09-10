# 场景真实坐标校准记录（T03，corrected spacerobot.usd）

状态：已用真实数据校准。来源：sim 同学提供的 `preflight_report.json`（场景结构扫描）+
`metadata.json`（一条真实成功 episode 的记录，scenario_id=2, actual_outcome=SUCCESS）+
`r1_bimanual_dataset/reference/scenario_t03.json`（他们的参考配置）。三份交叉核对后采用。

## 数据质量提示

`reference/scenario_t03.json` 里 `object_position: [-3.75, 3.85, 0.64]` 标的是
**T01 的真实坐标**（对得上 preflight 报告里 T01 的 bbox），不是 T03 的——大概率是从
`scenario_reference.json`（T01用）复制过来改 `target_object` 字段时漏改了坐标。
**本文档不采用这个数值**，改用下面交叉验证过的 T03 真实坐标。

## 采用的真实值（T03）

| 字段 | 值 | 来源 |
|---|---|---|
| T03 bbox（世界坐标） | min `[-2.702, 3.798, 0.580]` max `[-2.598, 3.902, 0.694]` | `preflight_report.json` |
| T03 bbox 中心（采用为 nominal object 位置） | `(-2.65, 3.85, 0.637)` | 上面 bbox 算出 |
| 物体尺寸 | `[0.104, 0.104, 0.114]` m（约10cm小方块） | `metadata.json::scenario.metadata.object_bbox_dimensions_m` |
| 一条真实成功 episode 的物体初始位置 | `(-2.6327, 3.8262, 0.6370)`（bbox中心附近的小扰动采样） | `metadata.json`，scenario_id=2，实测 SUCCESS |
| 支撑架 support | `/World/TaskSetup/Fixtures/StorageRack/Top`，bbox min `[-3.458,3.231,1.302]` max `[-1.808,3.781,1.372]` | `metadata.json::runtime` |
| 物体被放置后的位置（真实 place 目标，不是原地） | `(-2.633, 3.506, 1.429)` | `metadata.json::runtime.target_runtime_bbox_center` |
| 机器人底盘 stand-off 摆位 | `base_position=(-2.65, 3.13, 0.0)`，`stand_off_m=0.72` | `metadata.json::runtime.base_placement` |
| pregrasp_distance（参考值） | `0.1` m | `reference/scenario_t03.json` |
| lift 高度（参考值） | `0.12` m（`lift_vector=[0,0,0.12]`） | `reference/scenario_t03.json` |
| 双臂初始关节角 | 和我们的 `DEFAULT_QPOS` 几乎一致（差异在小数点后2-3位） | `reference/scenario_t03.json` 与 `metadata.json` 均一致 |

## ⚠️ 最重要的缺口：没有"绿色目标环"这个东西

近100条通过的真实测试，`success_flags` 只有 `{grasp, lift, hold, place, release}`，`place`
指的是把物体放到 `StorageRack/Top` 支撑架上方（坐标见上表"物体被放置后的位置"），**不是
放进 TRA-01 冻结文本里说的"绿色目标环"**。场景扫描（`preflight_report.json` 的
`rigid_body_candidates`/`object_candidates`）里也没有任何名字带 "ring" 的物体。

这意味着：目前配置里的 `target_ring_pose` 字段**没有一个真实验证过的"目标环"坐标可用**。
下面配置更新时，`target_ring_pose` 临时借用了上表"物体被放置后的位置"这个**真实、可达、
测试过的坐标**，比之前瞎猜的占位值可靠，但请明确：**这不是目标环，只是一个真实可达的
放置位置占位**，等确认场景里到底有没有目标环、或者该用哪个物体/哪个位置代表目标环之后，
再换成正确值。
