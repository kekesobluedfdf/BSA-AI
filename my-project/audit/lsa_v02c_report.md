# LSA V0.2c 交付报告：研究解释、可视化与报告导出

日期：2026-10-22  
数据：`results/ui_runs/20261022_073434/03_iflytek_classroom/lesson_ai_codes.csv`  
新结果：`results/ui_runs/20261022_073434/03_iflytek_classroom/behavior_sequence_v02c/`  
历史结果：V0.2a / V0.2b 目录保留，未覆盖；原始编码 CSV 未修改。  
统计口径更新：`Z ≤ -1.96` 的负向显著不再进入主研究发现，只保留在诊断表中。

## 1. 修改文件

- `my-project/scripts/behavior_sequence.py`
  - 增加自我转移/持续性汇总；
  - 增加状态段时长、状态段明细和时长统计；
  - 元数据增加代码版本、输入 SHA256；
  - 默认输出目录改为 `behavior_sequence_v02c`；
  - 图函数支持编码字典、标签模式、自转移/频数/概率筛选、节点频次。
- `my-project/scripts/coding_dictionary.py`
  - 支持 CSV/JSON 编码字典；
  - 支持 UI 保存/清除当前字典；
  - 未知编码回退到原始编码。
- `my-project/scripts/graph_export.py`
  - 使用 NetworkX + Matplotlib 导出 SVG/PNG；
  - 支持节点频次、边频次/概率、低期望频数样式。
- `my-project/scripts/report_builder.py`
  - 生成 HTML 研究报告；
  - 报告包含方法、显著转移、解释模板、稀疏诊断、双视角、方法对照、限制和可复现参数。
- `my-project/ui/app.py`
  - 行为序列页增加编码字典上传；
  - 增加 A/B 双视角；
  - 增加图形筛选、SVG/PNG 导出；
  - 增加 HTML 报告导出；
  - 方法切换默认 Holm。
- `my-project/tests/test_lsa_v02c.py`
  - 新增 12 项 V0.2c 测试。
- `my-project/requirements.txt`
  - 增加 `statsmodels==0.15.0`、`matplotlib==3.11.2`。

## 2. 编码字典

支持字段：

```text
code,label,definition,category
```

UI 可上传 CSV/JSON，保存到 `my-project/configs/coding/`。  
缺失字典时，图表和报告使用原始 A–P 编码；未知编码不会导致崩溃。

## 3. 双视角

### 视角 A

固定 5 秒窗口状态转移，保留 V0.2b 的统计结果：

- 437 个窗口；
- 436 次 Lag 1 转移；
- 调整后残差、Holm/BH、Monte Carlo、稀疏诊断。

新增描述性统计：

```text
自转移：163
非自转移：273
自转移比例：37.39%
非自转移比例：62.61%
```

### 视角 B

连续同码合并后的状态段，仅描述性：

```text
状态段数量：274
状态段转移：273
持续时间可计算：274
持续时间缺失：0
平均状态段：7.974 秒
中位状态段：5.0 秒
最短：5.0 秒
最长：105.0 秒
```

B 视角不显示、不继承 A 视角的 p 值或显著性星号。

## 4. 图形与导出

保留 DOT；新增：

- 统计方法筛选；
- 正向显著筛选；
- 自转移开关；
- 最低转移频数；
- 最低条件转移概率；
- 编码名称/原始编码/两者显示；
- 节点频次与边频次；
- 低期望频数橙色点线提示；
- SVG/PNG 导出。

图形筛选只影响展示，不改变统计表或检验结果。

示例导出：

```text
results/ui_runs/20261022_073434/03_iflytek_classroom/behavior_sequence_v02c/lesson_report_graph.svg
results/ui_runs/20261022_073434/03_iflytek_classroom/behavior_sequence_v02c/lesson_report_graph.png
```

## 5. HTML 研究报告

示例报告：

```text
results/ui_runs/20261022_073434/03_iflytek_classroom/behavior_sequence_v02c/lesson_report_holm.html
```

报告包含：

- 数据来源、分析单位、窗口与序列；
- 编码体系；
- 行为频次、转移频数、转移概率；
- 统计方法与校正；
- 显著转移；
- 研究解释模板；
- 低期望频数诊断；
- 自转移/非自转移；
- 状态段敏感性；
- 方法对照；
- 统计限制；
- 生成时间、代码版本、输入 SHA256、随机种子。

## 6. V0.2b 与 V0.2c 数值一致性

```text
窗口数：437 = 437
Lag 1 转移数：436 = 436
转移频数矩阵：完全一致
有效单元调整后残差：最大差异 0.0
不可定义单元：继续为 NaN/NA
```

V0.2c 没有改变转移频数和有效单元调整后残差。

统计口径更新后，主结果只报告正向显著转移：

```text
经典原始 p：15
经典 Holm：9
经典 BH：11
Monte Carlo 原始：10
Monte Carlo Holm：5
Monte Carlo BH：8
```

负向显著只保留在诊断表中，不计入上述主结果：

```text
经典原始 p：12
经典 Holm：2
经典 BH：6
Monte Carlo 原始：11
Monte Carlo Holm：3
Monte Carlo BH：6
```

当前负向规则：`Z ≤ -1.96`。

## 7. 测试

运行：

```bash
python -m unittest discover -s my-project/tests -p "test_*.py" -v
```

结果：

```text
Ran 85 tests in 8.919s
OK
```

新增 V0.2c 覆盖：

- 字典映射与未知编码；
- 自转移筛选只影响展示；
- 图形筛选不改变统计；
- 状态段时长与跨序列边界；
- 状态段视角不继承显著性；
- 输入哈希与随机种子；
- SVG/PNG 有效；
- HTML 报告数值与方法说明；
- 低期望频数警告；
- 历史结果未被覆盖。

## 8. UI 截图

```text
results/ui_screenshots/lsa_v02c.png
results/ui_screenshots/lsa_v02c_graph.png
```

截图来自实际 Streamlit 页面，加载任务 `20261022_073434`，进入“行为序列”页。

## 9. 已知限制

- 固定窗口状态转移不是事件级编码；
- 连续同码合并是描述性敏感性分析，不是新的显著性检验；
- 低期望频数仍可能使正态近似不可靠；
- Monte Carlo 零假设不保留原始自相关；
- 相邻窗口共享观测，转移不独立；
- 单课堂结果不能外推；
- 多课堂/多小组聚类或多层模型尚未实现；
- 显著转移不代表因果关系；
- 编码字典内容必须由研究者提供，系统不自动编造定义。
