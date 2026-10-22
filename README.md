# Behavior sequence analysis (轻量版/学生分发包）

这是一个轻量分发包，包含程序代码、教程、编码示例和 `20261022_073434` 的转录/编码/LSA 结果。

## 先说最简单的用法

如果你只想**阅读和复现已有结果**，不需要安装 Python：

1. 解压压缩包；
2. 双击打开 `docs/user_guide.html`；
3. 查看 `results/ui_runs/20261022_073434/` 下的 CSV、逐字稿和 HTML 报告。

推荐先看：

```text
results/ui_runs/20261022_073434/03_iflytek_classroom/lesson_verbatim.txt
results/ui_runs/20261022_073434/03_iflytek_classroom/lesson_ai_codes.csv
results/ui_runs/20261022_073434/03_iflytek_classroom/behavior_sequence_v02c/lesson_report_holm.html
```

分发包**没有包含原始 `lesson.wav`**，所以播放音频、重新做本地转录等功能需要另外获取音频。

## 如果想运行界面

### 1. 准备 Python

建议安装 Python 3.12。安装后打开 PowerShell，进入本目录下的：

```text
my-project
```

例如：

```powershell
cd D:\ClassroomResearchAI_student\my-project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 准备模型

本地 FunASR 模式需要把模型放在**分发包根目录**的 `models/`，不是在 `my-project/models/`：

```text
student_release/
├── my-project/
├── models/
│   ├── SenseVoiceSmall/
│   ├── fsmn-vad/
│   ├── campplus/
│   ├── ct-punc/
│   └── Fun-ASR-Nano/
└── results/
```

模型文件较大，完整版连接地址（https://pan.baidu.com/s/1oDFwtolloFK9MEBeb0gPrg?pwd=cihe 提取码: cihe 
），不在分发包内。

如果只使用“讯飞云端高精度”转写，可以暂时不需要本地 ASR 模型；但 AI 编码、人工复核和 LSA 仍然可用。

### 3. 配置 API Key

复制：

```text
my-project/.env.example
```

为：

```text
my-project/.env
```

填写：

```env
IFLYTEK_APP_ID=
IFLYTEK_API_KEY=
IFLYTEK_API_SECRET=
deepseek_api_key=
```

- 只有使用讯飞云端转写时才需要讯飞三项；
- 只有使用 AI 编码时才需要 DeepSeek Key；
- 不要把自己的 `.env` 发给别人。

### 4. 启动

双击：

```text
my-project/start_ui.bat
```

或：

```powershell
cd my-project
.venv\Scripts\activate
python -m streamlit run ui/app.py
```

浏览器打开：

```text
http://localhost:8501
```

## 使用现有 073434 结果

启动后：

1. 左侧选择“单课堂分析”；
2. 历史任务选择 `20261022_073434`；
3. 点击“加载选中的历史任务”；
4. 查看逐字稿、片段、行为序列和报告。

如果没有音频文件，部分依赖音频的功能会不可用，但 CSV 和 HTML 结果仍可查看。

## 常见问题

### 启动时报 `streamlit` 找不到

确认已经激活虚拟环境：

```powershell
.venv\Scripts\activate
```

并重新执行：

```powershell
pip install -r requirements.txt
```

### 本地模式提示模型不存在

检查 `models/` 是否在分发包根目录，而不是 `my-project/models/`。

### AI 编码提示没有 Key

检查 `my-project/.env` 中是否有 `deepseek_api_key`，修改后重启 UI。

### 没有 NVIDIA 显卡

可以在界面里选择 `cpu`，但本地模型会明显变慢。教室电脑建议由老师提供已配置好的环境或使用云端模式。

### 分发包里没有音频

这是为了控制 GitHub/发送体积。需要原始音频时，向老师获取单独的音频文件。
#
