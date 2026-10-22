# LSA V0.2a 修复报告：数据结构与计算边界

日期：2026-10-22  
范围：`my-project/scripts/behavior_sequence.py`、UI 行为序列标签页、LSA 测试  
未修改：原始课堂编码数据、旧 `behavior_sequence/` 结果

## 1. 修改文件清单

- `my-project/scripts/behavior_sequence.py`
  - 增加序列边界、时间校验、并发/重复/重叠检测；
  - 增加不可定义统计量处理；
  - 增加稀疏诊断与单元状态；
  - 转换图增加 `P(j|i)`、E<5 标记；
  - 输出目录默认改为 `behavior_sequence_v02a`。
- `my-project/ui/app.py`
  - 行为序列页增加 sequence_id / 单序列确认；
  - 增加观察序列数、不可检验单元、E<5 显著指标；
  - 增加“稀疏诊断”标签页；
  - 默认读取 `behavior_sequence_v02a`；
  - 图注说明低期望频数与边标签。
- `my-project/tests/test_behavior_sequence_audit.py`
  - 从 13 项审计测试扩展为 20 项 V0.2a 回归测试。
- 新结果目录：
  - `results/ui_runs/20261022_073434/03_iflytek_classroom/behavior_sequence_v02a/`
- 旧结果保留：
  - `results/ui_runs/20261022_073434/03_iflytek_classroom/behavior_sequence/`

## 2. 分析单位

当前数据：

- 一行 = 一个固定 5 秒窗口；
- 一个窗口 = 一个主要行为编码；
- 连续相同编码 = 同一状态的持续，不自动合并；
- 不删除自转移 `A→A`；
- 当前没有同一时间多个行为；
- `window_id` 在当前文件中唯一，`start_hms` 单调递增；
- 本阶段仍是“固定时间窗口状态转移分析”，不是“事件级行为序列分析”；
- 不把窗口状态自动转换为事件。

## 3. 序列边界处理

- 支持 `sequence_id` 列，按序列分组后分别计算 Lag k 转移；
- 每个序列最多产生 `max(0, n-k)` 次转移；
- 禁止跨序列边界生成转移；
- 无 `sequence_id` 时必须显式传入 `--single-sequence`；
- UI 中无序列列时，必须勾选“我确认当前文件是一个完整连续观察序列”；
- 元数据记录：
  - `sequence_mode`
  - `sequence_count`
  - `sequence_lengths`
  - `boundary_assumption`
- 当前 flower 结果：
  - `sequence_mode = single_sequence_assumed`
  - `sequence_count = 1`
  - 明确记录“无 sequence_id，用户确认整个文件为一个连续观察序列”。

## 4. 时间与并发处理

- 时间优先级：`start_ms` → `start_hms` → `window_id`；
- 当前数据实际使用 `start_hms`；
- 按序列 + 时间排序，不再按 CSV 行号排序；
- 检查：
  - 缺失时间；
  - 重复 `window_id`；
  - 同一时间戳多条记录；
  - 时间窗口重叠；
  - 输入时间倒序（排序并记录 `input_time_out_of_order`）；
  - 非法 Lag。
- 当前数据：
  - 437 个唯一窗口；
  - 时间单调；
  - 无重复时间戳；
  - 无重叠窗口；
  - `input_time_out_of_order = false`。
- 重复窗口、并发时间戳、重叠窗口当前直接报错，不擅自按行号转成先后关系。

## 5. 不可定义统计量

当以下情况发生时：

- `Eij = 0`
- 行边际为 0
- 列边际为 0
- 调整后残差分母为 0
- 总转移数为 0

当前实现：

- `adjusted_residual = NaN`
- `raw_p_value = NaN`
- `adjusted_residuals_starred = "NA"`
- `transition_probability = NaN`（零行合计）
- `status = "undefined"`
- `warning_reason` 记录原因
- `testable = False`

当前数据中有 10 个不可检验单元，全部是目标列为 N 的单元（N 只出现 1 次且没有转入）：

```text
A→N, B→N, C→N, D→N, E→N, F→N, G→N, I→N, J→N, N→N
```

旧版本把这些单元写成 `Z=0, p=1`；V0.2a 改为 `NaN`/`NA`，避免把“不可定义”解释为“零关联”。

## 6. 稀疏诊断

每个转移单元输出：

- observed
- expected
- transition_probability
- adjusted_residual
- raw_p_value
- expected_lt_5
- expected_lt_1
- testable
- warning_reason

当前结果：

- 可检验单元：90
- 不可检验单元：10
- `E<5` 单元：70（包含不可检验单元）
- 显著转换：27
- 其中 `E<5` 的显著转换：9

低期望频数仍保留描述性统计和显著性标记，但增加警告；本阶段没有引入 exact/Monte Carlo。

## 7. 修复前后对比

当前真实数据：

| 指标 | 修复前 | V0.2a |
|---|---:|---:|
| 窗口数 | 437 | 437 |
| Lag 1 转移数 | 436 | 436 |
| 转移频数矩阵 | 10×10 | 10×10 |
| 转移频数完全一致 | — | 是 |
| 显著转换 | 27 | 27 |
| 显著边集合 | — | 完全一致 |
| 可检验单元 | 未标注 | 90 |
| 不可检验单元 | 未标注（Z=0） | 10（NaN/NA） |
| 有效单元调整后残差 | 保留 | 最大差异 0.0（旧结果保留 3 位小数） |

说明：

- 旧输出将不可定义单元写成 `Z=0`；
- V0.2a 对这些单元输出 `NaN`，因此“有效单元残差完全一致”指的是 `testable=True` 的 90 个单元；
- 27 条显著边集合未变化，但其中 9 条现在明确标记 `E<5`，应在解释时降级为探索性。

## 8. 测试

运行：

```bash
python -m unittest discover -s my-project/tests -p "test_behavior_sequence_audit.py" -v
```

结果：

```text
Ran 20 tests in 0.361s
OK
```

覆盖：

1. 单序列 Lag 1
2. 单序列 Lag 2
3. 多序列边界
4. 单事件/零转移
5. 零边际与零期望
6. 时间倒序
7. 重复窗口
8. 并发编码
9. 连续相同编码
10. 多序列汇总转移矩阵
11. 当前真实课堂数据回归
12. 期望频数与 SciPy 交叉核对
13. 调整后残差公式
14. Pearson 残差与调整后残差区分
15. 转移概率行归一化
16. 低期望频数显著标记
17. 正负显著方向
18. DOT 图正向/负向与 P/Z 标签
19. 排除 N 不跨 N 连接
20. 无 sequence_id 时必须显式确认

## 9. UI

行为序列页现在显示：

- 观察序列数；
- 窗口数、转换数；
- 显著转换数；
- 不可检验单元数；
- E<5 显著数；
- “稀疏诊断”标签页；
- 不可检验单元与 E<5 单元表；
- 转换图：
  - 蓝色实线：正向显著；
  - 红色虚线：负向显著；
  - 橙色点线：E<5；
  - 边标签：`Z` 与 `P(j|i)`。

## 10. 尚未解决的统计推断问题

V0.2a 没有解决：

- 多重比较校正（Bonferroni/FDR）；
- 连续转移之间的依赖；
- 5 秒窗口状态的自相关；
- 低期望频数的 exact/Monte Carlo 推断；
- 多课堂/多小组的聚类稳健标准误或多层模型；
- 事件级行为序列编码；
- 多标签并发行为处理；
- 当前单序列假设的独立外部验证。

因此，当前 27 条显著边仍应视为探索性结果，尤其 9 条 `E<5` 的边。
