# LSA V0.2b 报告：统计推断可靠性增强

日期：2026-10-22  
统计口径更新：`Z ≤ -1.96` 的负向显著不进入主研究发现，只保留在诊断表中。以下主结果表为正向显著计数。  
数据：`results/ui_runs/20261022_073434/03_iflytek_classroom/lesson_ai_codes.csv`  
新结果：`results/ui_runs/20261022_073434/03_iflytek_classroom/behavior_sequence_v02b/`  
旧 V0.2a 结果保留：`behavior_sequence_v02a/`  
原始课堂编码 CSV 未修改。

## 1. 修改文件清单

- `my-project/scripts/behavior_sequence.py`
  - 经典 LSA：增加 Holm、BH-FDR；
  - 增加序列内随机排列 Monte Carlo；
  - 增加逐单元推断明细；
  - 增加方法对照与显著边重叠；
  - 增加连续同码合并的描述性敏感性分析；
  - 新输出默认目录 `behavior_sequence_v02b`。
- `my-project/ui/app.py`
  - 行为序列页增加统计推断方式选择，默认 Holm；
  - 增加 Monte Carlo 次数与随机种子；
  - 增加方法对照、稀疏诊断、敏感性标签页；
  - 转换图按当前方法切换；
  - 图边标签显示当前统计指标与 `P(j|i)`。
- `my-project/tests/test_behavior_sequence_audit.py`
  - 从 20 项扩展到 33 项测试；
  - 增加 statsmodels 交叉核对、Monte Carlo、经验 p、敏感性测试。
- `my-project/requirements.txt`
  - 增加 `statsmodels==0.15.0`。

## 2. 分析单位

本阶段继续使用：

```text
analysis_unit = fixed_time_window_state
window_duration_seconds = 5.0
event_level = false
lag = 1
```

- 一行仍代表一个 5 秒窗口；
- 一个窗口一个主编码；
- 连续相同编码代表状态持续，不合并、不删除自转移；
- 当前数据仍是一条连续课堂观察序列。

连续同码合并仅作为独立的描述性敏感性分析，不覆盖原始窗口分析。

## 3. 经典 LSA 多重比较校正

### 检验族

校正族固定为：

```text
all_testable_directed_transitions
```

即所有具有可定义调整后残差的有向转移单元。

当前：

```text
候选单元：100
可检验单元：90
不可检验单元：10
参与校正：90
```

不可定义单元不按 p=1 参与校正。

### 方法

保留：

```text
Zij = (Oij - Eij) / sqrt(Eij * (1 - Ri/N) * (1 - Cj/N))
```

保留双侧正态近似 p：

```text
p = erfc(|Z| / sqrt(2))
```

新增：

- Holm 校正 p；
- Benjamini–Hochberg FDR q；
- 原始、Holm、BH 三种显著性标记。

Holm/BH 实现已与 `statsmodels.stats.multitest.multipletests` 交叉核对。

## 4. Monte Carlo 零假设与经验 p 值

### 零假设

```text
在每条独立观察序列内部，行为编码的出现次数固定，
但排列顺序是随机的。
```

### 实现

- 使用 V0.2a 的序列边界；
- 只在序列内部打乱；
- 每条序列长度与编码频数保持不变；
- 使用相同 Lag；
- 每次模拟重新计算转移频数；
- 默认 10,000 次；
- NumPy `default_rng`，种子可复现。

当前运行：

```text
simulation_count = 10000
random_seed = 20261022
```

### 经验 p 值

```text
upper = (b_upper + 1) / (B + 1)
lower = (b_lower + 1) / (B + 1)
two_sided = min(1, 2 * min(upper, lower))
```

双侧定义为离散分布下的一种约定，不等价于所有可能的精确双侧检验。

模拟分布退化时输出 `simulation_degenerate=True`，均值/标准差可为 0，不产生无穷大统计量。

## 5. 稀疏诊断

新增/保留：

```text
lesson_cell_diagnostics.csv
lesson_monte_carlo_diagnostics.csv
lesson_cell_inference.csv
```

每个单元包含：

```text
observed
expected
transition_probability
adjusted_residual
raw_p_value
p_holm
q_bh
sig_raw
sig_holm
sig_bh
simulated_mean
simulated_sd
empirical_p_upper
empirical_p_lower
empirical_p_two_sided
empirical_p_holm
empirical_q_bh
simulation_significance*
expected_lt_5
expected_lt_1
testable
warning_reason
```

当前：

```text
不可检验单元：10
E<5 单元：70
```

所有低期望频数结果仍保留描述性信息，但标记 `expected_lt_5`，不能仅凭正态近似视为可靠确认。

## 6. 当前结果对比

### 显著转移数量

| 方法 | 正向显著数 | 其中 E<5 |
|---|---:|---:|
| 经典原始 p | 15 | 8 |
| 经典 Holm | 9 | 3 |
| 经典 BH-FDR | 11 | 5 |
| Monte Carlo 原始 | 10 | 3 |
| Monte Carlo Holm | 5 | 1 |
| Monte Carlo BH | 8 | 2 |

这些结果不应被预设为一致；它们对应不同零假设和不同校正规则。

### 共同识别

六种方法共同识别 5 条：

```text
F→F, G→G, I→I, J→B, J→J
```

经典 Holm 与 Monte Carlo Holm 共同识别上述 5 条。

只在经典 Holm 下正向显著：

```text
B→F, D→D, E→E, F→D
```

当前正向结果中，没有只在 Monte Carlo Holm 下显著的转移；负向显著不纳入主结果比较。

完整逐边对照：

```text
lesson_significant_overlap.csv
```

### 原始结果保持不变

```text
窗口数：437
Lag 1 转移数：436
转移频数矩阵：与 V0.2a 完全一致
有效单元调整后残差：与 V0.2a 数值一致（旧文件保留 3 位小数）
不可定义单元：继续为 NaN/NA
```

## 7. 敏感性分析

合并每条序列内部连续相同状态：

```text
A A A B B C C A  →  A B C A
```

当前：

```text
原始窗口：437
合并后状态段：274
合并后转移数：273
分析单位：behavior_state_segment
推断：descriptive_only
```

该结果不覆盖原始窗口分析，也不能解释为每 5 秒的转移概率。

## 8. 测试

运行：

```bash
python -m unittest discover -s my-project/tests -p "test_behavior_sequence_audit.py" -v
```

结果：

```text
Ran 33 tests in 0.697s
OK
```

覆盖：

- V0.2a 全部回归测试；
- Holm/BH 与 statsmodels 交叉核对；
- 不可定义 p 不参与校正；
- Monte Carlo 不跨序列；
- 序列内频数保持；
- 固定种子可复现；
- Lag 1/2 模拟转移数；
- 经验 p 加一修正；
- 上尾/下尾/双侧；
- 零方差模拟；
- 合并状态不跨序列；
- 敏感性不改变原始结果；
- 图方法切换；
- 当前课堂数据回归。

## 9. UI

行为序列页：

- 默认展示“经典 LSA：Holm 校正”；
- 可切换原始 p、Holm、BH、Monte Carlo 原始/Holm/BH；
- 显著表和转换图随方法同步切换；
- 图边标签显示当前统计指标与 `P(j|i)`；
- 新增“方法对照”和“敏感性”；
- 低期望频数边使用橙色点线；
- 显示分析限制。

## 10. 已知统计限制

V0.2b 解决/增加了：

- 多重比较校正；
- Monte Carlo 序列随机排列；
- 经验 p 值；
- 稀疏性诊断；
- 描述性状态合并敏感性分析。

仍未解决：

- 低期望频数的 exact/Monte Carlo 推断仍需谨慎，MC 不是经典精确检验；
- Monte Carlo 零假设不保留原始自相关；
- 相邻转移共享窗口；
- 固定窗口状态自相关；
- 单课堂不能外推；
- 多课堂/多小组聚类或多层模型；
- 事件级行为编码；
- 多标签并发行为；
- 显著不等于因果。

因此当前结果应继续作为探索性 LSA 结果使用。
