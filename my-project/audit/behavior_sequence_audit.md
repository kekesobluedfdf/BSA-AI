# 滞后序列分析（LSA）统计方法审计报告

审计对象：`my-project/scripts/behavior_sequence.py`  
审计数据：`results/ui_runs/20261022_073434/03_iflytek_classroom/lesson_ai_codes.csv`  
审计日期：2026-10-22  
审计方式：代码逐行检查 + 独立单元测试 + SciPy 交叉核对  
代码和现有分析结果未修改。

## 一、结论摘要

| # | 检查项 | 结论 |
|---|---|---|
| 1 | Lag 阶数是否明确 | **PASS**（代码默认 lag=1，元数据和 UI 有 lag；表格本身未写 lag） |
| 2 | 是否只在独立观察序列内部计算转移 | **WARNING（当前单课堂）/ 多序列输入时会 FAIL** |
| 3 | 时间排序、同时间戳、重叠/并发行为 | **PASS（当前数据）/ WARNING（代码未校验）** |
| 4 | 期望频数 Eij = Ri×Cj/N | **PASS** |
| 5 | 调整后残差公式 | **PASS** |
| 6 | 零期望、零边际、小样本 | **WARNING**（无 NaN，但未定义残差被写成 0） |
| 7 | 转移概率 P(j|i)=Oij/Ri | **PASS** |
| 8 | 正向/负向显著转移区分 | **PASS** |
| 9 | 低期望频数与正态近似限制 | **FAIL**（未检查；27 条显著中有 9 条 E<5） |
| 10 | 多重检验、连续转移依赖、跨课堂汇总 | **WARNING**（未校正、未建模依赖、无多序列结构） |
| 11 | 转换图是否正向显著、是否提供概率与 Z | **PASS（默认仅正向，边标 Z）/ WARNING（概率不在边上）** |

## 二、当前实际结果

- 行为窗口：437
- Lag：1
- 行为转换：436
- 原始显著转换：27 条（15 正向，12 负向）
- 调整后残差最大绝对值：8.008
- 10 个观测编码 → 10×10 = 100 个检验单元
- 27 条原始显著中，9 条期望频数小于 5
- 按 Bonferroni 校正（α=0.05/100=0.0005），仍显著 11 条
- 最小期望频数为 0（存在结构性零单元）

低期望频数但仍被标为显著的例子：

| 转移 | 观测 | 期望 | Z | p |
|---|---:|---:|---:|---:|
| E → E | 2 | 0.147 | 4.927 | 8.33e-07 |
| I → I | 13 | 4.241 | 4.719 | 2.37e-06 |
| B → F | 11 | 3.991 | 3.905 | 9.43e-05 |
| I → A | 1 | 0.099 | 3.027 | 2.47e-03 |
| A → I | 1 | 0.099 | 3.027 | 2.47e-03 |
| B → E | 2 | 0.550 | 2.043 | 4.10e-02 |
| E → B | 2 | 0.550 | 2.043 | 4.10e-02 |
| N → G | 1 | 0.197 | 2.020 | 4.34e-02 |
| F → B | 0 | 3.991 | -2.223 | 2.62e-02 |

## 三、逐项审计

### 1. Lag 阶数

代码：

- `behavior_sequence.py:72` 使用 `sort_values("window_id")`；
- `compute_tables(sequence, lag, alpha)` 用 `sequence[:-lag]` 和 `sequence[lag:]`；
- CLI 默认 `--lag 1`；
- 结果元数据包含 `"lag": 1`；
- UI 提供“滞后阶数”输入。

结论：**PASS**。  
限制：输出的频次表、残差表、图标题没有直接写 `lag=1`，建议在表格/图题中标注。

### 2. 独立观察序列

代码把所有行合并成一个 `sequence`，没有 `sequence_id`、`class_id`、`group_id` 或 `observation_id` 字段，也没有在边界处分段。

当前 flower 结果只有一个课堂文件，因此没有实际跨课堂连接错误。  
但如果把两个课堂、两个小组或两个独立观察片段放进同一个 CSV，程序会在前一段最后一个行为和后一段第一个行为之间生成一条伪转换。

结论：**WARNING（当前数据）/ FAIL（多序列输入）**。  
建议：增加 `sequence_id`，按序列分组后分别计算转移矩阵，或至少禁止一个文件包含多个独立序列。

### 3. 时间顺序、同时间戳、重叠、并发行为

代码按 `window_id` 排序，而不是按 `start_ms` 排序；没有检查：

- `window_id` 是否唯一；
- `start_ms` 是否单调；
- 是否存在重叠窗口；
- 是否存在同时间戳多个行为。

当前数据检查：

- 437 行，437 个唯一 `window_id`；
- `window_id` 单调递增；
- `start_hms` 单调递增；
- 无重复窗口。

因此当前结果没有排序错误。  
但代码本身不保证这一点：如果 `window_id` 与时间顺序冲突，转移序列会错；如果同时间戳有多条记录，会被当成连续行为。

结论：**PASS（当前数据）/ WARNING（一般实现）**。  
建议：排序改为 `start_ms`，并断言 `window_id` 唯一、时间不重叠；并发行为需要定义多标签状态或明确拒绝。

### 4. 期望频数

代码 `behavior_sequence.py:121`：

```python
expected_value = (row_total * col_total / n_transitions)
```

与经典公式一致：

```text
Eij = Ri × Cj / N
```

测试用 SciPy 1.18.1 的 `chi2_contingency` 交叉核对：

- 2×2 序列 `A B A B A` 的期望矩阵为 `[[1,1],[1,1]]`；
- 程序结果与 SciPy 完全一致。

结论：**PASS**。

### 5. 调整后残差

代码 `behavior_sequence.py:124-128`：

```python
denominator = expected_value * (1 - row_total / n_transitions) * (1 - col_total / n_transitions)
denominator = max(0.0, denominator)
z = (observed - expected_value) / sqrt(denominator)
```

与目标公式一致：

```text
Zij = (Oij - Eij) / sqrt(Eij × (1 - Ri/N) × (1 - Cj/N))
```

测试中：

- 对 `A B A B A`，调整后残差为 ±2；
- 普通 Pearson 标准化残差为 ±1；
- 测试明确断言两者不相等，避免把 Pearson 残差误认为调整后残差。

结论：**PASS**。  
限制：`max(0.0, ...)` 会把理论上非法/退化的分母压成 0，随后 Z 被写成 0；这在零边际场景下会掩盖“残差未定义”。

### 6. 零期望、零边际、小样本

- `E=0` 或行列边际为 0 时，代码返回 `z=0`、`p=1`，不产生 NaN；
- 不会因此产生显著结果；
- 但 `z=0` 不是“没有关联”的证据，而是“调整后残差未定义”；
- 代码没有标记这些单元为不可解释/不可检验；
- 没有最小期望频数检查；
- 没有 exact test 或 Monte Carlo 替代。

结论：**WARNING**。  
建议：对 `E=0`、`Ri=0`、`Cj=0` 或分母为 0 的单元输出 `NaN` 和 `reliable=False`，不要输出 0 作为统计结论。

### 7. 转移概率

代码 `behavior_sequence.py:135`：

```python
transition_counts.div(row_totals.replace(0, pd.NA), axis=0).fillna(0.0)
```

即：

```text
P(j|i) = Oij / Ri
```

测试验证每行概率和为 1；行合计为 0 时输出 0。

结论：**PASS**。

### 8. 正向/负向显著转移

代码 `behavior_sequence.py:153` 使用 `abs(z) >= critical`，并记录：

- `direction = "positive"` 当 `z > 0`；
- `direction = "negative"` 当 `z < 0`；
- p 值为双侧 `erfc(|z|/sqrt(2))`。

当前结果：15 条正向、12 条负向。  
转换图默认只画正向显著，也支持正负都画。

结论：**PASS**。

### 9. 低期望频数与正态近似

代码没有：

- 检查 `expected < 5` 或 `expected < 1`；
- 对低期望单元使用 Fisher 精确检验、超几何检验或 Monte Carlo；
- 在显著表/图中标记“近似不可靠”。

当前结果有 9/27 条显著转移的 `E < 5`，其中：

- `I → A`、`A → I` 的 E=0.099；
- `N → G` 的 E=0.197；
- `E → E` 的 E=0.147。

这些单元的正态近似 p 值不可靠，尤其 `N → G` 只基于 1 次观测。

结论：**FAIL**。  
这是当前分析最重要的统计效度问题。

建议：

1. 增加 `expected >= 5` 的可靠性标记；
2. 对 `E < 5` 的单元改用 exact/Monte Carlo，或至少不要把它们作为确认性结论；
3. 图中默认不绘制低期望显著边；
4. 对低频编码（如 N）单独做敏感性分析。

### 10. 多重检验、转移依赖、跨课堂汇总

当前实现：

- 对 10×10=100 个单元逐一做 |Z|≥1.96 检验；
- 没有 Bonferroni、Holm 或 FDR 校正；
- 27 条原始显著，Bonferroni 后仍有 11 条；
- 连续转移共享窗口，`t→t+1` 与 `t+1→t+2` 不独立；
- 5 秒窗口内的行为具有自相关；
- 没有按课堂/小组做分层、聚类稳健标准误或多层模型；
- 没有跨课堂随机效应。

结论：**WARNING**。  
当前 p 值和显著边应视为探索性结果，不应直接作为强推断。  
建议：

- 报告原始 p 与 FDR/Bonferroni 校正结果；
- 用按课堂/小组的 block bootstrap 或聚类稳健方法评估不确定性；
- 多课堂研究使用多层模型或按课堂分别分析后再汇总；
- 明确 LSA 的 first-order Markov 假设和窗口依赖性。

### 11. 转换图

代码 `build_dot`：

- 默认 `edge_mode="positive"`，只画正向显著边；
- 支持 `both` 和 `negative`；
- 边标签为 `Z` 和显著性星号；
- 边宽随 |Z| 变化；
- 正向蓝色实线，负向红色虚线；
- 概率 `P(j|i)` 存在 `lesson_significant_transitions.csv` 中，但没有直接写在图的边上。

结论：**PASS（正向显著主路径）/ WARNING（概率未上图）**。  
建议：图例或边标签同时显示 `Z` 与 `P(j|i)`，或在图旁附显著转移表。

## 四、单元测试

测试文件：

```text
my-project/tests/test_behavior_sequence_audit.py
```

运行：

```bash
python -m unittest discover -s my-project/tests -p "test_behavior_sequence_audit.py" -v
```

结果：

```text
Ran 13 tests in 0.104s
OK
```

覆盖：

- 行为频次；
- 转移频次；
- Lag 2；
- 期望频数与 SciPy 交叉核对；
- 调整后残差公式；
- Pearson 残差与调整后残差的区分；
- 转移概率行归一化；
- 零边际/零期望；
- 低期望频数仍被标显著；
- 跨独立序列边界；
- `window_id` 排序而非时间排序；
- 重复 `window_id` 并发行为；
- 正负显著方向；
- DOT 图正向/负向边。

注意：部分测试是“记录当前实现行为”的审计测试，例如跨序列边界和重复窗口测试；测试通过不等于这些行为在统计上正确。

## 五、未修改内容

本次审计未修改：

- `behavior_sequence.py`；
- 现有 `behavior_sequence/` 结果文件；
- 现有 AI 编码结果；
- UI 行为序列逻辑。

以上修复建议需要后续单独实施。
