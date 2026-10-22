# LSA V0.3 交付报告：多课堂联合分析

日期：2026-10-22  
程序版本：`LSA V0.3`  
示例研究项目：`studies/teacher_case_synthetic/study_manifest.json`  
示例结果：`results/studies/teacher_case_synthetic/lsa_v03/20261022_102336/`

## 1. 修改文件清单

- `my-project/scripts/multi_classroom.py`
  - 多课堂 manifest 校验；
  - 课堂内独立序列分析；
  - 联合转移矩阵；
  - 按转移机会加权概率；
  - 课堂等权概率；
  - 课堂比较；
  - 跨课堂稳定性；
  - Leave-One-Class-Out；
  - 多课堂图形和 HTML 报告。
- `my-project/ui/app.py`
  - 侧边栏“单课堂分析 / 多课堂分析”切换；
  - 研究项目创建与导入；
  - 多课堂联合概览；
  - 课堂比较；
  - 稳定性；
  - 结果导出。
- `my-project/tests/test_multi_classroom.py`
  - 新增 20 项多课堂测试。
- `studies/teacher_case_synthetic/`
  - 三课堂合成演示数据，明确标记为 synthetic。
- `results/studies/teacher_case_synthetic/lsa_v03/20261022_102336/`
  - 示例联合分析结果和 HTML 报告。

V0.2a、V0.2b、V0.2c 结果均保留，未覆盖；原始课堂编码 CSV 未修改。

## 2. 多课堂数据结构

研究项目目录：

```text
studies/
└── teacher_case_synthetic/
    ├── study_manifest.json
    ├── classes/
    │   ├── lesson_001/
    │   ├── lesson_002/
    │   └── lesson_003/
    └── ...
```

`study_manifest.json` 记录：

```text
study_id
teacher_id
coding_scheme_id
window_duration_seconds
lag
classes[]
```

每个课堂记录：

```text
class_id
teacher_id
source_video
coding_scheme_id
window_duration_seconds
review_status
file
sha256
sequence_id_column
single_sequence
```

结果目录：

```text
results/studies/<study_id>/lsa_v03/<run_id>/
```

包含：

```text
study_manifest.json
analysis_metadata.json
classroom_summary.csv
pooled_frequency.csv
pooled_transition_counts.csv
pooled_transition_probabilities.csv
equal_weight_transition_probabilities.csv
equal_weight_class_counts.csv
classroom_transition_comparison.csv
transition_stability.csv
leave_one_class_out.csv
pooled_transition_graph.svg
pooled_transition_graph.png
multi_classroom_report.html
```

## 3. 三种分析模式

### 模式 A：联合描述分析

联合转移矩阵：

```text
T_total = T1 + T2 + ... + Tn
```

每节课堂、每个 sequence_id 独立生成转移，不跨课堂、不跨片段。

示例数据：

```text
课堂数：3
独立序列：4
观察窗口：17
有效转移：13
```

联合转移频数：

```text
   A  B  C
A  0  3  2
B  1  1  3
C  2  1  0
```

### 模式 B：课堂比较

每节课堂独立输出：

- 行为频次；
- 转移频次；
- 条件转移概率；
- 自转移/非自转移；
- 行内排名；
- 稀疏性诊断。

不同课堂长度不一致时，比较表同时显示 `row_total` 和条件概率，避免仅按原始频数比较。

### 模式 C：稳定性与 LOCO

输出：

- 有效课堂数；
- 出现课堂数；
- 出现比例；
- 跨课堂概率均值、中位数、标准差、极值、四分位距；
- Leave-One-Class-Out 联合概率变化。

这些均为描述性指标，不是跨课堂显著性检验。

## 4. 两种汇总权重

### 按转移机会加权

```text
P_pooled(j|i) = Σ_c O_cij / Σ_c R_ci
```

其中 `R_ci` 是课堂 c 中以 i 为起始状态的转移总数。

### 课堂等权平均

```text
P_equal(j|i) = mean_{c: R_ci > 0} P_c(j|i)
```

未出现起始行为 i 的课堂不参与该行平均，不按 0 处理。

示例中 A 行只有 2 个有效课堂，B/C 行为 3 个有效课堂。

## 5. 数据质量检查

已实现：

- 重复 `class_id`；
- 重复文件 SHA256；
- 编码体系不一致阻止合并；
- 窗口长度声明不一致阻止合并；
- 实际 `duration_s` 中位数与声明窗口长度核对；
- 时间源记录；
- 时间倒序标记 `time_order_ok=False`；
- 重复窗口、同时间戳、重叠窗口错误；
- 零转移课堂安全处理；
- 缺少 sequence_id 时必须显式确认。

## 6. 单课堂回归

验证结果：

```text
窗口数：437
Lag 1 转移数：436
转移频数矩阵：与 V0.2b/V0.2c 一致
V0.2b/V0.2c 历史结果：未覆盖
原始编码 CSV：未修改
```

## 7. 多课堂测试

新增 20 项测试，加上原有 45 项，共：

```text
Ran 65 tests in 7.811s
OK
```

覆盖：

- 不跨课堂；
- 不跨片段；
- 联合频数等于各课堂矩阵之和；
- 联合概率；
- 等权概率；
- 缺失起始行为的课堂不参与平均；
- 不同长度课堂的加权差异；
- 重复课堂/文件；
- 编码体系与窗口冲突；
- 时间倒序；
- 零转移课堂；
- 稳定性、标准差和 IQR；
- LOCO；
- 多课堂无显著性继承；
- 单课堂回归；
- 元数据可追溯；
- 原始数据不被修改。

## 8. UI 截图

```text
results/ui_screenshots/lsa_v03_multi.png
results/ui_screenshots/lsa_v03_overview.png
```

截图来自实际 Streamlit 页面，模式为“多课堂分析”。

## 9. 示例导出报告

```text
results/studies/teacher_case_synthetic/lsa_v03/20261022_102336/multi_classroom_report.html
```

报告包含课堂汇总、联合频次、联合概率、等权概率、稳定性、LOCO、图形和统计限制。

## 10. 统计限制与后续扩展

- 多课堂联合结果只做描述性汇总；
- 未实现课堂层级聚类稳健标准误、多层模型或课堂层级 bootstrap；
- 不把单课堂显著性数量相加解释为联合显著性；
- 不把多课堂拼接成一条长序列后计算调整残差；
- LOCO 是描述性稳健性检查，不是显著性检验；
- 合成数据仅用于演示；
- 当前尚未实现 SNA、ENA 或复杂多层统计模型。
