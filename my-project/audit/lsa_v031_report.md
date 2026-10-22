# LSA V0.3.1 交付报告：多课堂工作流整合

日期：2026-10-22  
程序版本：`LSA V0.3.1`  
课堂注册表：`studies/classroom_registry.json`  
示例联合分析：`results/studies/teacher_case_synthetic/lsa_v03/20261022_110725/`

## 1. 修改文件清单

- `my-project/scripts/classroom_registry.py`
  - 扫描 `results/ui_runs/`；
  - 建立课堂索引；
  - 识别 AI 编码、人工复核和单课堂 LSA；
  - 稳定 `class_id` 与可编辑显示名称；
  - 原子写入注册表。
- `my-project/scripts/multi_classroom.py`
  - 支持 `code_source = ai / human`；
  - 支持 `human_file`、人工复核完整性校验；
  - manifest 原子写入；
  - 追加课堂、移除课堂、修改显示名称；
  - 记录来源 run、音频、编码哈希、人工复核哈希；
  - 外部 CSV 字段映射规范化。
- `my-project/ui/app.py`
  - 侧边栏单课堂/多课堂模式；
  - “从已完成课堂选择”；
  - 课堂注册表刷新；
  - 外部 CSV 字段映射；
  - 研究项目追加、移除、重命名；
  - 打开来源单课堂；
  - 研究项目摘要。
- `my-project/tests/test_workflow_v031.py`
  - 新增 20 项工作流测试。

## 2. 课堂注册表

注册表路径：

```text
studies/classroom_registry.json
```

每条记录包含：

```text
class_id
run_id
run_dir
display_name
status
source_audio
audio_sha256
source_video
ai_codes_path
human_codes_path
coding_scheme_id
window_duration_seconds
review_status
reviewed_windows
total_windows
behavior_sequence_path
created_at
```

扫描规则：

- 一个 `run_id` 只登记一条课堂记录；
- 同一 run 的多个输出目录只选择优先的 AI 编码文件；
- 不根据目录名猜测编码体系，缺失时为 `unknown`；
- 缺少 AI 编码的 run 标记为 `missing_ai_codes`；
- 缺少音频标记为 `missing_audio`；
- 重复扫描通过 `run_id` 去重；
- 可修改显示名称，但保留稳定 `class_id`；
- 不修改原始 CSV，不移动历史目录。

## 3. 实际历史课堂发现结果

当前扫描 `results/ui_runs/` 得到 4 条 run 记录：

| run_id | 状态 | 窗口数 | AI 编码 | 人工复核 | 单课堂 LSA |
|---|---|---:|---|---|---|
| 20261022_073434 | ready | 437 | 有 | absent | 有 |
| 20261022_035058 | missing_ai_codes | 0 | 无 | absent | 无 |
| 20261021_185157 | missing_ai_codes | 0 | 无 | absent | 无 |
| 20261021_174034 | missing_ai_codes | 0 | 无 | absent | 无 |

说明：

- 只有 `20261022_073434` 目前可直接进入多课堂分析；
- 合成课堂位于 `studies/teacher_case_synthetic`，不会被注册表误认为真实 `ui_runs` 课堂；
- 真实课堂的 `coding_scheme_id` 为 `unknown`，因为现有 AI 元数据没有显式编码体系字段，系统没有从目录名猜测。

## 4. AI 编码与人工复核

注册表区分：

- `lesson_ai_codes.csv`；
- `lesson_human_codes.csv`；
- `review_status = absent / partial / complete / invalid`。

规则：

- 人工复核文件存在不等于全部完成；
- 只有所有 AI 窗口都有合法人工编码，才标记 `complete`；
- 部分复核标记 `partial`；
- UI 默认对完整人工复核推荐 `human`，否则默认 `ai`；
- 选择 `human` 时校验必须完整，否则阻止合并；
- manifest 记录 `code_source`、`file`、`human_file` 及两者哈希；
- 不覆盖 `lesson_ai_codes.csv`。

当前真实课堂 `20261022_073434` 人工复核状态为 `absent`，因此使用 AI 编码。

## 5. 研究项目创建与维护

### 从已有课堂创建

多课堂页面默认显示课堂注册表，研究者可以：

- 刷新注册表；
- 勾选课堂；
- 编辑显示名称；
- 选择编码来源；
- 选择新建项目或追加到已有项目；
- 创建/追加时校验重复、编码体系、窗口长度、序列边界和编码来源。

### 外部 CSV

保留外部导入，并新增字段映射：

```text
行为编码字段
时间字段
时间类型 start_ms / start_hms
窗口编号字段
独立观察片段字段
窗口长度字段
```

导入时生成规范化副本，不修改原始文件；没有片段编号时必须确认是连续完整记录。

### 维护

支持：

- 追加课堂；
- 移除课堂；
- 修改显示名称；
- 重新运行分析；
- 保留旧分析目录；
- 原子更新 `study_manifest.json`。

## 6. 单课堂与多课堂导航

manifest 中的课堂记录保存：

```text
source_run_dir
source_final_dir
source_audio
behavior_sequence_path
```

多课堂页面提供“打开来源课堂的单课堂分析”。  
如果课堂来自外部 CSV，没有来源 run，则不会显示无效跳转。

## 7. 测试

全量测试：

```text
Ran 85 tests in 8.919s
OK
```

V0.3.1 新增 20 项，覆盖：

- 历史课堂发现；
- 重复扫描去重；
- 同 run 多输出只算一节课堂；
- 缺失元数据安全处理；
- 稳定 `class_id`；
- 显示名称修改；
- AI/人工复核状态区分；
- 部分复核阻止 human 来源；
- human 编码实际参与分析；
- manifest 追加/移除；
- 重复 class 拒绝；
- 外部字段映射；
- 外部无来源 run；
- 原子保存；
- 原始 CSV 不修改；
- 历史分析结果不覆盖；
- 项目重开保持来源关联；
- 单课堂统计不变。

## 8. UI 截图

```text
results/ui_screenshots/lsa_v03_multi.png
results/ui_screenshots/lsa_v03_overview.png
```

多课堂页面现在默认展示“从已完成课堂选择”，下方保留外部 CSV 导入。

## 9. 实际操作指南

1. 先完成单课堂转录、人工修订和 AI 编码；
2. 切换到“多课堂分析”；
3. 点击“刷新课堂注册表”；
4. 勾选已完成的课堂；
5. 编辑显示名称和编码来源；
6. 选择新建项目或追加到已有项目；
7. 创建项目；
8. 在“已导入研究项目”中运行联合分析；
9. 在“联合概览/课堂比较/稳定性/结果导出”查看结果；
10. 需要时点击“打开来源课堂的单课堂分析”。

外部研究者数据使用“方式 B：导入外部 CSV”并按列映射。

## 10. 已知限制

- 注册表扫描是按钮触发或首次创建时触发，不是后台实时文件监控；
- 现有 AI 元数据缺少显式 `coding_scheme_id`，历史记录显示 `unknown`；
- `source_video` 多数历史记录为空；
- 部分人工复核不能作为 human 来源直接合并；
- 外部 CSV 字段映射一次处理一个文件；
- 多课堂统计仍为描述性汇总，没有课堂层级显著性推断；
- 不开发视频播放器、SNA、ENA 或批量视频转录；
- 合成数据只用于工作流演示，不能作为正式研究结果。
