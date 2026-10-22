from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

UI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = UI_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
RESULTS_DIR = ROOT_DIR / "results"
RUNS_DIR = RESULTS_DIR / "ui_runs"
CONFIG_DIR = PROJECT_DIR / "configs"
STUDIES_DIR = ROOT_DIR / "studies"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


for env_path in (PROJECT_DIR / ".env", ROOT_DIR / ".env"):
    load_env_file(env_path)


def iflytek_missing_credentials() -> list[str]:
    return [
        key
        for key in ("IFLYTEK_APP_ID", "IFLYTEK_API_KEY", "IFLYTEK_API_SECRET")
        if not os.environ.get(key, "").strip()
    ]


def deepseek_configured() -> bool:
    return any(
        os.environ.get(key, "").strip()
        for key in ("deepseek_api_key", "DEEPSEEK_API_KEY", "DEEPSEEK_KEY")
    )


sys.path.insert(0, str(SCRIPTS_DIR))
from code_agreement import VALID_CODES, build_agreement
from coding_dictionary import active_dictionary_path, label_for, load_dictionary, reset_dictionary, save_dictionary
from behavior_sequence import build_dot
from graph_export import render_transition_graph
import multi_classroom as multi
import classroom_registry as classroom_registry
from coding_config import (
    CONFIG_DIR as CODING_CONFIG_DIR,
    active_paths as active_coding_paths,
    read_document as read_coding_document,
    reset_active as reset_coding_config,
    save_active as save_coding_config,
    save_source as save_coding_source,
)
from transcript_exports import export_all, ms_to_clock

st.set_page_config(page_title="课堂语音研究助手", page_icon="🎙️", layout="wide")

PYTHON = sys.executable
DEFAULT_HOTWORDS = CONFIG_DIR / "guwen_hotwords.txt"
DEFAULT_CORRECTIONS = CONFIG_DIR / "guwen_corrections.json"


def run_command(command: list[str], log_box) -> None:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env.setdefault("MODELSCOPE_CACHE", str(ROOT_DIR / "models" / ".cache"))
    env.setdefault("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn")
    process = subprocess.Popen(
        [str(item) for item in command],
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        if len(lines) > 80:
            lines = lines[-80:]
        log_box.code("\n".join(lines), language="text")
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"命令执行失败，退出码 {code}")


def script(name: str) -> str:
    return str(SCRIPTS_DIR / name)


def detect_final_dir(run_dir: Path) -> Path | None:
    for name in (
        "07_chorus_applied",
        "11_cue_scoped",
        "10_phrase_split",
        "09_chorus_cue",
        "08_chorus",
        "06_intra_applied",
        "04_final",
        "03_iflytek",
        "03_hybrid",
        "02_refined",
        "01_transcribe",
    ):
        candidate = run_dir / name
        if (candidate / "lesson_segments.csv").exists():
            return candidate
    return None


def detect_review_dir(run_dir: Path) -> Path | None:
    for name in ("06_intra_applied", "05_intra_review"):
        candidate = run_dir / name
        if (candidate / "lesson_intra_review.csv").exists():
            return candidate
    return None


def save_upload(upload, target: Path) -> Path:
    target.write_bytes(upload.getbuffer())
    return target


def run_iflytek_pipeline(
    run_dir: Path,
    audio_path: Path,
    settings: dict,
    corrections_file: Path | None,
    status,
    log_box,
) -> tuple[Path, Path | None]:
    step = run_dir / "03_iflytek"
    status.update(label="步骤 1：讯飞录音文件转写大模型", state="running")
    command = [
        PYTHON,
        script("iflytek_asr.py"),
        audio_path,
        "--output-dir",
        step,
        "--teacher-id",
        settings["teacher_id"],
        "--teacher-speaker",
        settings.get("iflytek_teacher_speaker", "auto"),
        "--role-type",
        "1",
        "--role-num",
        str(settings.get("iflytek_role_num", 2)),
        "--pd",
        settings.get("iflytek_pd", "edu"),
        "--window-seconds",
        str(settings["window_seconds"]),
        "--stats-window-min",
        str(settings["stats_window_min"]),
    ]
    run_command(command, log_box)
    final_dir = step

    if settings.get("use_corrections") and corrections_file:
        status.update(label="步骤 2：术语纠正与导出", state="running")
        step4 = run_dir / "04_final"
        run_command(
            [
                PYTHON,
                script("apply_corrections.py"),
                step / "lesson_segments.jsonl",
                corrections_file,
                "--output-dir",
                step4,
                "--teacher-id",
                settings["teacher_id"],
                "--window-seconds",
                str(settings["window_seconds"]),
                "--stats-window-min",
                str(settings["stats_window_min"]),
            ],
            log_box,
        )
        final_dir = step4

    return final_dir, None


def run_pipeline(
    run_dir: Path,
    audio_path: Path,
    settings: dict,
    hotwords_file: Path | None,
    corrections_file: Path | None,
    status,
    log_box,
) -> tuple[Path, Path | None]:
    if settings.get("asr_engine") == "iflytek":
        return run_iflytek_pipeline(run_dir, audio_path, settings, corrections_file, status, log_box)

    step1 = run_dir / "01_transcribe"
    status.update(label="步骤 1：SenseVoice 识别与说话人聚类", state="running")
    command = [
        PYTHON,
        script("transcribe_classroom.py"),
        audio_path,
        "--output-dir",
        step1,
        "--device",
        settings["device"],
        "--teacher-id",
        settings["teacher_id"],
        "--window-seconds",
        str(settings["window_seconds"]),
        "--window-mode",
        "fixed",
        "--stats-window-min",
        str(settings["stats_window_min"]),
        "--log-level",
        "INFO",
    ]
    if settings["preset_spk_num"]:
        command += ["--preset-spk-num", str(settings["preset_spk_num"])]
    if not settings["use_punc"]:
        command += ["--no-punc"]
    run_command(command, log_box)

    source_jsonl = step1 / "lesson_segments.jsonl"
    final_dir = step1

    if settings["use_refine"]:
        status.update(label="步骤 2：短片段说话人重判", state="running")
        step2 = run_dir / "02_refined"
        run_command(
            [
                PYTHON,
                script("refine_speakers.py"),
                source_jsonl,
                audio_path,
                "--output-dir",
                step2,
                "--teacher-id",
                settings["teacher_id"],
                "--short-threshold",
                "1.5",
                "--margin",
                "0.05",
                "--min-student-sim",
                "0.2",
                "--apply",
                "--window-seconds",
                str(settings["window_seconds"]),
                "--stats-window-min",
                str(settings["stats_window_min"]),
            ],
            log_box,
        )
        source_jsonl = step2 / "lesson_segments.jsonl"
        final_dir = step2

    nano_result_path = None
    if settings["use_hybrid"]:
        status.update(label="步骤 3：Fun-ASR-Nano 高精度文本", state="running")
        step3 = run_dir / "03_hybrid"
        command = [
            PYTHON,
            script("hybrid_asr.py"),
            source_jsonl,
            audio_path,
            "--output-dir",
            step3,
            "--teacher-id",
            settings["teacher_id"],
            "--device",
            settings["device"],
            "--chunk-seconds",
            "60",
            "--overlap-seconds",
            "0.5",
            "--window-seconds",
            str(settings["window_seconds"]),
            "--stats-window-min",
            str(settings["stats_window_min"]),
        ]
        if settings.get("use_hotwords") and hotwords_file:
            command += ["--hotwords-file", hotwords_file]
        run_command(command, log_box)
        source_jsonl = step3 / "lesson_segments.jsonl"
        final_dir = step3
        nano_result_path = step3 / "lesson_nano_result.json"

    if settings["use_corrections"] and corrections_file:
        status.update(label="步骤 4：术语纠正与导出", state="running")
        step4 = run_dir / "04_final"
        run_command(
            [
                PYTHON,
                script("apply_corrections.py"),
                source_jsonl,
                corrections_file,
                "--output-dir",
                step4,
                "--teacher-id",
                settings["teacher_id"],
                "--window-seconds",
                str(settings["window_seconds"]),
                "--stats-window-min",
                str(settings["stats_window_min"]),
            ],
            log_box,
        )
        source_jsonl = step4 / "lesson_segments.jsonl"
        final_dir = step4

    if settings.get("auto_chorus", True) and nano_result_path and nano_result_path.exists():
        status.update(label="附加步骤：声纹短语级齐读检测", state="running")
        step10 = run_dir / "10_phrase_split"
        embeddings = run_dir / "02_refined" / "lesson_embeddings.npz"
        command = [
            PYTHON,
            script("split_chorus_phrases.py"),
            source_jsonl,
            audio_path,
            "--nano-result",
            nano_result_path,
            "--output-dir",
            step10,
            "--teacher-id",
            settings["teacher_id"],
            "--output-window-seconds",
            str(settings["window_seconds"]),
            "--stats-window-min",
            str(settings["stats_window_min"]),
        ]
        if embeddings.exists():
            command += ["--embeddings", embeddings]
        run_command(command, log_box)
        source_jsonl = step10 / "lesson_segments.jsonl"
        final_dir = step10

        phrase_review_path = step10 / "lesson_phrase_chorus_review.csv"
        needs_cue_split = True
        if phrase_review_path.exists():
            try:
                needs_cue_split = pd.read_csv(phrase_review_path).empty
            except Exception:
                needs_cue_split = True
        if needs_cue_split:
            status.update(label="附加步骤：指令词范围内齐读切分", state="running")
            step11 = run_dir / "11_cue_scoped"
            run_command(
                [
                    PYTHON,
                    script("apply_cue_scoped_chorus.py"),
                    source_jsonl,
                    audio_path,
                    "--nano-result",
                    nano_result_path,
                    "--output-dir",
                    step11,
                    "--teacher-id",
                    settings["teacher_id"],
                    "--output-window-seconds",
                    str(settings["window_seconds"]),
                    "--stats-window-min",
                    str(settings["stats_window_min"]),
                ],
                log_box,
            )
            source_jsonl = step11 / "lesson_segments.jsonl"
            final_dir = step11

    if settings.get("auto_punctuate", True):
        status.update(label="附加步骤：逐字稿自动标点", state="running")
        run_command(
            [
                PYTHON,
                script("punctuate_verbatim.py"),
                source_jsonl,
                "--output-dir",
                final_dir,
                "--device",
                settings["device"],
            ],
            log_box,
        )

    review_dir = None
    if settings["use_intra_review"]:
        status.update(label="附加步骤：片段内部候选检测", state="running")
        step5 = run_dir / "05_intra_review"
        embeddings = run_dir / "02_refined" / "lesson_embeddings.npz"
        command = [
            PYTHON,
            script("detect_intra_segment_changes.py"),
            source_jsonl,
            audio_path,
            "--output-dir",
            step5,
            "--teacher-id",
            settings["teacher_id"],
            "--window-seconds",
            "2",
            "--hop-seconds",
            "0.5",
            "--min-region-seconds",
            "1.5",
            "--margin",
            "0.15",
            "--min-student-sim",
            "0.35",
            "--consistency",
            "0.6",
            "--output-window-seconds",
            str(settings["window_seconds"]),
            "--stats-window-min",
            str(settings["stats_window_min"]),
        ]
        if embeddings.exists():
            command += ["--embeddings", embeddings]
        run_command(command, log_box)
        review_dir = step5

    return final_dir, review_dir


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def find_duplicate_run(audio_sha256: str, asr_engine: str = "local") -> str | None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for settings_path in RUNS_DIR.glob("*/settings.json"):
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("audio_sha256") == audio_sha256 and data.get("asr_engine", "local") == asr_engine:
            return settings_path.parent.name
    return None


def window_csv_path(final_dir: Path) -> Path | None:
    settings = st.session_state.get("settings", {})
    seconds = settings.get("window_seconds")
    if seconds:
        candidate = final_dir / f"lesson_window_{float(seconds):g}s_fixed.csv"
        if candidate.exists():
            return candidate
    candidates = sorted(final_dir.glob("lesson_window_*s_fixed.csv"))
    return candidates[0] if candidates else None


def stats_csv_path(final_dir: Path) -> Path | None:
    settings = st.session_state.get("settings", {})
    minutes = settings.get("stats_window_min")
    if minutes:
        candidate = final_dir / f"lesson_stats_{float(minutes):g}min.csv"
        if candidate.exists():
            return candidate
    candidates = sorted(final_dir.glob("lesson_stats_*min.csv"))
    return candidates[0] if candidates else None


def parse_time(value: str) -> int:
    text = value.strip().replace("：", ":")
    if not text:
        raise ValueError("时间不能为空")
    parts = text.split(":")
    if len(parts) == 1:
        seconds = float(parts[0])
    elif len(parts) == 2:
        seconds = int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError("时间格式应为 mm:ss、hh:mm:ss 或秒数")
    return int(round(seconds * 1000))


def sidebar_run_selector():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = sorted([path.name for path in RUNS_DIR.iterdir() if path.is_dir()], reverse=True)
    options = ["当前任务"] + runs
    selected = st.sidebar.selectbox("历史任务", options, index=0, key="history_select")
    if st.sidebar.button("加载选中的历史任务") and selected != "当前任务":
        run_dir = RUNS_DIR / selected
        final_dir = detect_final_dir(run_dir)
        if final_dir:
            st.session_state["run_dir"] = str(run_dir)
            st.session_state["final_dir"] = str(final_dir)
            review_dir = detect_review_dir(run_dir)
            st.session_state["review_dir"] = str(review_dir) if review_dir else None
            chorus_review_dir = None
            for candidate in (final_dir, run_dir / "08_chorus", run_dir / "07_chorus_review"):
                if (candidate / "lesson_chorus_review.csv").exists():
                    chorus_review_dir = candidate
                    break
            st.session_state["chorus_review_dir"] = (
                str(chorus_review_dir) if chorus_review_dir else None
            )
            audio_candidates = list(run_dir.glob("lesson.*"))
            if audio_candidates:
                st.session_state["audio_path"] = str(audio_candidates[0])
            settings_path = run_dir / "settings.json"
            if settings_path.exists():
                st.session_state["settings"] = json.loads(settings_path.read_text(encoding="utf-8"))
            st.sidebar.success(f"已加载：{selected}")
    st.sidebar.caption(f"运行目录：{RUNS_DIR}")
    if st.session_state.get("settings", {}).get("asr_engine") == "iflytek":
        st.sidebar.caption("当前引擎：讯飞云端高精度（教师/学生组）")
    if st.session_state.get("final_dir"):
        st.sidebar.success(f"当前结果：{Path(st.session_state['final_dir']).name}")


def render_start_tab():
    st.subheader("新建课堂转录")
    st.info(
        "本地模式：SenseVoice + CAM++ + Fun-ASR-Nano，可离线，默认。"
        "讯飞模式：上传到讯飞录音文件转写大模型，通常文字更准，但需要网络和讯飞额度。"
    )
    engine_label = st.radio(
        "识别引擎",
        ["本地 FunASR（默认）", "讯飞云端高精度"],
        index=0,
        horizontal=True,
        help="本地模式保留说话人细分和齐读后处理；讯飞模式只输出教师/学生组，不跑本地齐读。",
    )
    is_iflytek = engine_label.startswith("讯飞")
    missing_credentials = iflytek_missing_credentials() if is_iflytek else []
    if is_iflytek:
        if missing_credentials:
            st.error("讯飞凭证未配置完整：" + "、".join(missing_credentials))
            st.caption(f"请在 {PROJECT_DIR / '.env'} 中填写三项凭证，格式见 .env.example。")
        else:
            st.success("讯飞凭证已配置。")

    uploaded = st.file_uploader(
        "上传课堂录音",
        type=["wav", "mp3", "m4a", "flac", "aac", "ogg"],
        help="建议使用 16 kHz 单声道 WAV；其他格式会自动交给 ffmpeg 处理。",
    )
    upload_sha256 = None
    if uploaded:
        upload_sha256 = hashlib.sha256(uploaded.getvalue()).hexdigest()
        st.audio(uploaded)
        st.caption(
            f"文件：{uploaded.name}，大小：{uploaded.size / 1024 / 1024:.1f} MB，"
            f"SHA256：{upload_sha256[:16]}…"
        )
        duplicate = find_duplicate_run(upload_sha256, "iflytek" if is_iflytek else "local")
        if duplicate:
            st.warning(
                f"这份音频与历史任务 {duplicate} 完全相同，重新转录不会产生新的识别结果。"
                "请确认是否上传了错误的文件。"
            )

    preset_spk_num = 0
    role_num = 2
    pd_edu = True
    teacher_speaker = "auto"
    use_refine = use_hybrid = use_intra_review = False
    auto_chorus = auto_punctuate = use_punc = False

    with st.expander("处理参数", expanded=True):
        first, second = st.columns(2)
        with first:
            if is_iflytek:
                pd_edu = st.checkbox("启用教育领域参数 pd=edu", value=True)
                role_num = st.number_input(
                    "角色数量（教师 + 学生组）",
                    min_value=2,
                    max_value=10,
                    value=2,
                    step=1,
                    help="讯飞返回多个 speaker 时，除教师外的所有学生统一映射为 S02。",
                )
                teacher_speaker = st.text_input(
                    "教师原始说话人",
                    value="auto",
                    help="auto = 按总发言时长推断；也可填讯飞的 speaker 编号，例如 1。",
                )
            else:
                preset_spk_num = st.number_input(
                    "预设说话人数（0 = 自动估计）", min_value=0, max_value=50, value=9, step=1
                )
            teacher_id = st.text_input("教师说话人编号", value="S01")
            window_seconds = st.number_input("时间片段（秒）", min_value=1.0, max_value=30.0, value=3.0, step=1.0)
            device = st.selectbox("运行设备", ["cuda:0", "cpu"], index=0)
        with second:
            stats_window_min = st.number_input("统计窗口（分钟）", min_value=1.0, max_value=30.0, value=5.0, step=1.0)
            use_corrections = st.checkbox("使用术语纠正表", value=False)
            if is_iflytek:
                st.caption("讯飞模式直接返回标点文本；不运行本地 CAM++、Fun-ASR-Nano、齐读检测和 ct-punc。")
            else:
                use_refine = st.checkbox("短片段说话人重判", value=True)
                use_hybrid = st.checkbox("使用 Fun-ASR-Nano 提升文本", value=True)
                use_intra_review = st.checkbox("同时生成片段内部候选（较慢）", value=False)
                auto_chorus = st.checkbox("自动检测齐读（声纹短语）", value=True)
                auto_punctuate = st.checkbox("自动标点（逐字稿）", value=True)
                use_punc = st.checkbox("使用 ct-punc 句级分段", value=True)

        if is_iflytek:
            st.caption(
                "讯飞通常输出 speaker 1（教师）和 speaker 2（学生组）。"
                "本工具会把教师映射为 S01，其余说话人统一映射为 S02；齐读不单独区分。"
            )

    with st.expander("热词与术语纠正", expanded=False):
        st.markdown(
            "**热词（txt）**：在识别前给模型提供课程专用词，帮助它优先识别这些词。"
            "每行一个词，或用逗号分隔。例如：`一人家贫`、`卖画以度日`、`无人问津`。"
            "热词只影响识别，不会修改已经识别出的文字。"
        )
        st.markdown(
            "**术语纠正表（JSON）**：在识别完成后做固定文本替换，用于修正课程术语的固定误识别。"
            "格式为 `{\"错误写法\": \"正确写法\"}`，例如 "
            "`{\"一人佳品\": \"一人家贫\", \"可授\": \"可售\"}`。"
            "只建议写明确的专有词或固定短语，不要写“的”“是”这类泛化词，避免误改。"
        )
        use_hotwords = st.checkbox(
            "使用热词",
            value=False,
            disabled=is_iflytek,
            help="讯飞录音文件转写大模型接口当前不接入本地热词；需要纠正时请用术语纠正表。",
        )
        if not is_iflytek:
            hotwords_upload = st.file_uploader("自定义热词文件", type=["txt"], key="hotwords_upload")
        else:
            hotwords_upload = None
        corrections_upload = st.file_uploader("自定义纠正表 JSON", type=["json"], key="corrections_upload")
        st.caption(
            f"默认热词/纠正表是为古文课准备的示例文件（{DEFAULT_HOTWORDS.name} / "
            f"{DEFAULT_CORRECTIONS.name}）。其他课程请上传自己的热词和纠正表，或保持关闭。"
        )

    settings = {
        "asr_engine": "iflytek" if is_iflytek else "local",
        "preset_spk_num": int(preset_spk_num) if preset_spk_num else None,
        "teacher_id": teacher_id.strip() or "S01",
        "window_seconds": float(window_seconds),
        "stats_window_min": float(stats_window_min),
        "device": device,
        "use_refine": use_refine,
        "use_hybrid": use_hybrid,
        "use_corrections": use_corrections,
        "use_intra_review": use_intra_review,
        "auto_chorus": auto_chorus,
        "auto_punctuate": auto_punctuate,
        "use_punc": use_punc,
        "use_hotwords": use_hotwords,
        "iflytek_pd": "edu" if is_iflytek and pd_edu else "",
        "iflytek_role_num": int(role_num),
        "iflytek_teacher_speaker": teacher_speaker.strip() or "auto",
    }

    if st.button(
        "开始处理",
        type="primary",
        disabled=uploaded is None or (is_iflytek and bool(missing_credentials)),
    ):
        run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(uploaded.name).suffix or ".wav"
        audio_path = save_upload(uploaded, run_dir / f"lesson{suffix}")

        hotwords_file = None
        if use_hotwords:
            if hotwords_upload:
                hotwords_file = save_upload(hotwords_upload, run_dir / "hotwords.txt")
            elif DEFAULT_HOTWORDS.exists():
                hotwords_file = DEFAULT_HOTWORDS

        corrections_file = None
        if use_corrections:
            if corrections_upload:
                corrections_file = save_upload(corrections_upload, run_dir / "corrections.json")
            elif DEFAULT_CORRECTIONS.exists():
                corrections_file = DEFAULT_CORRECTIONS

        settings["audio_sha256"] = upload_sha256
        settings["audio_filename"] = uploaded.name
        with (run_dir / "settings.json").open("w", encoding="utf-8") as handle:
            json.dump(settings, handle, ensure_ascii=False, indent=2)

        with st.status("正在处理课堂录音…", expanded=True) as status:
            log_box = st.empty()
            try:
                final_dir, review_dir = run_pipeline(
                    run_dir,
                    audio_path,
                    settings,
                    hotwords_file,
                    corrections_file,
                    status,
                    log_box,
                )
            except Exception as exc:
                status.update(label="处理失败", state="error")
                st.exception(exc)
                return

            st.session_state["run_dir"] = str(run_dir)
            st.session_state["final_dir"] = str(final_dir)
            st.session_state["review_dir"] = str(review_dir) if review_dir else None
            st.session_state["audio_path"] = str(audio_path)
            st.session_state["settings"] = settings
            chorus_review_dir = None
            for candidate in (final_dir, run_dir / "08_chorus", run_dir / "07_chorus_review"):
                if (candidate / "lesson_chorus_review.csv").exists():
                    chorus_review_dir = candidate
                    break
            st.session_state["chorus_review_dir"] = (
                str(chorus_review_dir) if chorus_review_dir else None
            )
            status.update(label="处理完成", state="complete")

        segments_path = final_dir / "lesson_segments.csv"
        if segments_path.exists():
            segments = pd.read_csv(segments_path)
            windows_path = window_csv_path(final_dir)
            windows = pd.read_csv(windows_path) if windows_path else pd.DataFrame()
            one, two, three = st.columns(3)
            one.metric("发言片段", len(segments))
            two.metric("说话人数", segments["speaker_id"].nunique() if "speaker_id" in segments else 0)
            three.metric(f"{settings['window_seconds']:g} 秒窗口", len(windows))
            st.success(f"结果目录：{final_dir}")


def render_verbatim_tab():
    final_dir = Path(st.session_state["final_dir"]) if st.session_state.get("final_dir") else None
    if not final_dir:
        st.info("请先在“开始转录”中处理一段录音。")
        return
    path = final_dir / "lesson_verbatim.txt"
    if not path.exists():
        st.warning("尚未生成逐字稿。")
        return
    text = path.read_text(encoding="utf-8-sig")
    st.subheader("逐字稿")
    st.caption("按连续说话人合并；同一说话人连续发言只保留一个时间标记。")
    st.text_area("逐字稿内容", value=text, height=620, disabled=True, label_visibility="collapsed")
    st.download_button(
        "下载逐字稿 TXT",
        data=text.encode("utf-8-sig"),
        file_name="lesson_verbatim.txt",
        mime="text/plain",
        key="download_verbatim",
    )


def render_segments_and_windows_tab():
    final_dir = Path(st.session_state["final_dir"]) if st.session_state.get("final_dir") else None
    if not final_dir:
        st.info("请先在“开始转录”中处理一段录音。")
        return

    segment_tab, window_tab, speaker_tab = st.tabs(["说话人片段", "时间窗口", "说话人汇总"])
    with segment_tab:
        path = final_dir / "lesson_segments.csv"
        if not path.exists():
            st.warning("尚未生成片段 CSV。")
        else:
            df = pd.read_csv(path)
            st.caption("可修改说话人编号、角色和文字；保存后会重新生成逐字稿、时间窗口和统计。")
            edited = st.data_editor(
                df,
                hide_index=True,
                width="stretch",
                height=520,
                disabled=[
                    column
                    for column in [
                        "recording_id",
                        "segment_id",
                        "start_ms",
                        "end_ms",
                        "start_hms",
                        "end_hms",
                        "duration_s",
                        "raw_text",
                        "speaker_original",
                        "speaker_refined",
                        "split_from_segment_id",
                        "split_index",
                    ]
                    if column in df.columns
                ],
                key="segments_editor",
            )
            if st.button("保存修改并重新生成", type="primary"):
                teacher_id = st.session_state.get("settings", {}).get("teacher_id", "S01")
                records = edited.fillna("").to_dict(orient="records")
                for record in records:
                    record["speaker_id"] = str(record["speaker_id"])
                    if record["speaker_id"].upper() == "CHORUS":
                        record["role"] = "chorus"
                    elif record["speaker_id"] == teacher_id:
                        record["role"] = "teacher"
                    else:
                        record["role"] = "student"
                    record["segment_id"] = int(record["segment_id"])
                backup = final_dir / "lesson_segments_before_edit.jsonl"
                source = final_dir / "lesson_segments.jsonl"
                if source.exists() and not backup.exists():
                    shutil.copy2(source, backup)
                with source.open("w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                export_all(
                    source,
                    final_dir,
                    window_seconds=float(st.session_state.get("settings", {}).get("window_seconds", 3)),
                    stats_window_min=float(st.session_state.get("settings", {}).get("stats_window_min", 5)),
                    teacher_id=teacher_id,
                    window_mode="fixed",
                )
                st.success("已保存并重新生成。")
                st.rerun()

    with window_tab:
        path = window_csv_path(final_dir)
        if path:
            windows = pd.read_csv(path)
            st.caption(
                "`窗口长度_s` 是固定窗口长度，应始终等于设置值；"
                "`有效语音_s` 是窗口内真正有人说话的秒数，会因停顿、静音和录音结尾而小于窗口长度。"
            )
            display = windows.rename(
                columns={
                    "duration_s": "窗口长度_s",
                    "speech_s": "有效语音_s",
                    "start_hms": "开始",
                    "end_hms": "结束",
                }
            )
            st.dataframe(display, hide_index=True, width="stretch", height=520)
            st.download_button(
                "下载时间窗口 CSV",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/csv",
                key="download_window",
            )
        else:
            st.warning("尚未生成时间窗口。")

    with speaker_tab:
        path = final_dir / "lesson_speaker_summary.csv"
        if path.exists():
            summary = pd.read_csv(path)
            st.dataframe(summary, hide_index=True, width="stretch")
            st.download_button(
                "下载说话人汇总 CSV",
                data=path.read_bytes(),
                file_name="lesson_speaker_summary.csv",
                mime="text/csv",
                key="download_speaker_summary",
            )
        else:
            st.warning("尚未生成说话人汇总。")


def render_intra_tab():
    if st.session_state.get("settings", {}).get("asr_engine") == "iflytek":
        st.info("讯飞云端模式只输出教师/学生组，不生成片段内部候选。")
        return
    final_dir = Path(st.session_state["final_dir"]) if st.session_state.get("final_dir") else None
    audio_path = Path(st.session_state["audio_path"]) if st.session_state.get("audio_path") else None
    if not final_dir or not audio_path:
        st.info("请先完成一次转录。")
        return

    review_dir = Path(st.session_state["review_dir"]) if st.session_state.get("review_dir") else None
    st.subheader("片段内部说话人变化候选")

    if not review_dir:
        st.caption("扫描教师长片段，检测内部是否存在连续学生声纹区域。该步骤需要额外 2–3 分钟。")
        if st.button("运行候选检测", type="primary"):
            target = Path(st.session_state["run_dir"]) / "05_intra_review"
            embeddings = Path(st.session_state["run_dir"]) / "02_refined" / "lesson_embeddings.npz"
            command = [
                PYTHON,
                script("detect_intra_segment_changes.py"),
                final_dir / "lesson_segments.jsonl",
                audio_path,
                "--output-dir",
                target,
                "--teacher-id",
                st.session_state.get("settings", {}).get("teacher_id", "S01"),
                "--window-seconds",
                "2",
                "--hop-seconds",
                "0.5",
                "--min-region-seconds",
                "1.5",
                "--margin",
                "0.15",
                "--min-student-sim",
                "0.35",
                "--consistency",
                "0.6",
                "--output-window-seconds",
                str(st.session_state.get("settings", {}).get("window_seconds", 3)),
                "--stats-window-min",
                str(st.session_state.get("settings", {}).get("stats_window_min", 5)),
            ]
            if embeddings.exists():
                command += ["--embeddings", embeddings]
            with st.status("正在检测片段内部说话人变化…", expanded=True) as status:
                log_box = st.empty()
                try:
                    run_command(command, log_box)
                except Exception as exc:
                    status.update(label="检测失败", state="error")
                    st.exception(exc)
                    return
                status.update(label="检测完成", state="complete")
            st.session_state["review_dir"] = str(target)
            st.rerun()
        return

    review_path = review_dir / "lesson_intra_review.csv"
    if not review_path.exists():
        st.warning("候选文件不存在。")
        return

    review = pd.read_csv(review_path)
    samples_path = review_dir / "split_samples" / "split_samples.csv"
    samples = pd.read_csv(samples_path) if samples_path.exists() else pd.DataFrame()
    if not samples.empty:
        review = review.merge(
            samples[["candidate_id", "clip"]],
            on="candidate_id",
            how="left",
        )
    st.dataframe(
        review.drop(columns=["clip"], errors="ignore"),
        hide_index=True,
        width="stretch",
        height=340,
    )
    st.download_button(
        "下载候选清单 CSV",
        data=review_path.read_bytes(),
        file_name=review_path.name,
        mime="text/csv",
        key="download_intra_review",
    )
    choices = {
        int(row["candidate_id"]): (
            f"候选 {int(row['candidate_id'])} · {ms_to_clock(int(row['region_start_ms']))} · "
            f"建议 {row['proposed_speaker']} · {str(row.get('region_text') or '')[:28]}"
        )
        for _, row in review.iterrows()
    }
    selected = st.multiselect(
        "选择要应用的候选",
        options=list(choices.keys()),
        format_func=lambda value: choices[value],
        key="intra_selected",
    )

    for _, row in review.iterrows():
        with st.expander(
            f"候选 {int(row['candidate_id'])} · {ms_to_clock(int(row['region_start_ms']))} · "
            f"建议 {row['proposed_speaker']}"
        ):
            st.write(f"文本：{row.get('region_text') or '（无文本）'}")
            st.write(
                f"平均声纹差距：{row.get('average_margin')}；一致性：{row.get('consistency')}"
            )
            clip = row.get("clip")
            if isinstance(clip, str) and Path(clip).exists():
                st.audio(clip)

    if st.button("应用选中的候选", type="primary", disabled=not selected):
        accept_ids = ",".join(str(int(value)) for value in selected)
        target = Path(st.session_state["run_dir"]) / "06_intra_applied"
        embeddings = Path(st.session_state["run_dir"]) / "02_refined" / "lesson_embeddings.npz"
        command = [
            PYTHON,
            script("detect_intra_segment_changes.py"),
            final_dir / "lesson_segments.jsonl",
            audio_path,
            "--output-dir",
            target,
            "--teacher-id",
            st.session_state.get("settings", {}).get("teacher_id", "S01"),
            "--window-seconds",
            "2",
            "--hop-seconds",
            "0.5",
            "--min-region-seconds",
            "1.5",
            "--margin",
            "0.15",
            "--min-student-sim",
            "0.35",
            "--consistency",
            "0.6",
            "--accept-ids",
            accept_ids,
            "--output-window-seconds",
            str(st.session_state.get("settings", {}).get("window_seconds", 3)),
            "--stats-window-min",
            str(st.session_state.get("settings", {}).get("stats_window_min", 5)),
        ]
        if embeddings.exists():
            command += ["--embeddings", embeddings]
        with st.status("正在应用选中的切分…", expanded=True) as status:
            log_box = st.empty()
            try:
                run_command(command, log_box)
            except Exception as exc:
                status.update(label="应用失败", state="error")
                st.exception(exc)
                return
            status.update(label="应用完成", state="complete")
        st.session_state["final_dir"] = str(target)
        st.session_state["review_dir"] = None
        st.success("已应用选中候选，并重新生成逐字稿、时间窗口和统计。")
        st.rerun()


def render_chorus_tab():
    if st.session_state.get("settings", {}).get("asr_engine") == "iflytek":
        st.info("讯飞云端模式只输出教师/学生组，不单独检测齐读。")
        return
    final_dir = Path(st.session_state["final_dir"]) if st.session_state.get("final_dir") else None
    audio_path = Path(st.session_state["audio_path"]) if st.session_state.get("audio_path") else None
    if not final_dir or not audio_path:
        st.info("请先完成一次转录。")
        return

    run_dir = Path(st.session_state["run_dir"])
    chorus_dir = (
        Path(st.session_state["chorus_review_dir"])
        if st.session_state.get("chorus_review_dir")
        else None
    )
    st.subheader("学生齐读检测")
    st.caption(
        "按“语音短语 + 子窗口声纹”检测：只有教师声纹明显下降、同时出现多个学生声纹的短语，"
        "才会自动切出并标记为 CHORUS。"
    )
    phrase_review_path = final_dir / "lesson_phrase_chorus_review.csv"
    if phrase_review_path.exists():
        st.success("本任务已使用声纹短语级齐读检测，以下区域已自动标记为齐读。")
        st.dataframe(
            pd.read_csv(phrase_review_path),
            hide_index=True,
            width="stretch",
            height=260,
        )
        st.download_button(
            "下载齐读切分记录 CSV",
            data=phrase_review_path.read_bytes(),
            file_name=phrase_review_path.name,
            mime="text/csv",
            key="download_phrase_chorus_review",
        )
        return

    cue_scoped_review = final_dir / "lesson_cue_scoped_review.csv"
    if cue_scoped_review.exists():
        st.success("本任务已按“一起读/齐读”指令范围精确切分齐读。")
        st.dataframe(
            pd.read_csv(cue_scoped_review),
            hide_index=True,
            width="stretch",
            height=220,
        )
        st.download_button(
            "下载指令范围齐读记录 CSV",
            data=cue_scoped_review.read_bytes(),
            file_name=cue_scoped_review.name,
            mime="text/csv",
            key="download_cue_scoped_review",
        )
        return

    cue_review_path = final_dir / "lesson_chorus_cue_review.csv"
    if cue_review_path.exists():
        with st.expander("查看“预备齐/一起读”切分记录", expanded=False):
            st.dataframe(
                pd.read_csv(cue_review_path),
                hide_index=True,
                width="stretch",
                height=220,
            )
            st.download_button(
                "下载切分记录 CSV",
                data=cue_review_path.read_bytes(),
                file_name=cue_review_path.name,
                mime="text/csv",
                key="download_chorus_cue_review",
            )
    if not chorus_dir and (final_dir / "lesson_chorus_review.csv").exists():
        chorus_dir = final_dir

    if not chorus_dir:
        if st.button("运行齐读检测", type="primary"):
            target = run_dir / "07_chorus_review"
            embeddings = run_dir / "02_refined" / "lesson_embeddings.npz"
            command = [
                PYTHON,
                script("detect_chorus.py"),
                final_dir / "lesson_segments.jsonl",
                audio_path,
                "--output-dir",
                target,
                "--teacher-id",
                st.session_state.get("settings", {}).get("teacher_id", "S01"),
                "--energy-ratio",
                "1.6",
                "--teacher-sim",
                "0.35",
                "--output-window-seconds",
                str(st.session_state.get("settings", {}).get("window_seconds", 3)),
                "--stats-window-min",
                str(st.session_state.get("settings", {}).get("stats_window_min", 5)),
            ]
            if embeddings.exists():
                command += ["--embeddings", embeddings]
            with st.status("正在检测齐读候选…", expanded=True) as status:
                log_box = st.empty()
                try:
                    run_command(command, log_box)
                except Exception as exc:
                    status.update(label="检测失败", state="error")
                    st.exception(exc)
                    return
                status.update(label="检测完成", state="complete")
            st.session_state["chorus_review_dir"] = str(target)
            st.rerun()
        return

    review_path = chorus_dir / "lesson_chorus_review.csv"
    if not review_path.exists():
        st.warning("候选文件不存在。")
        return

    review = pd.read_csv(review_path)
    samples_path = chorus_dir / "chorus_samples" / "chorus_samples.csv"
    samples = pd.read_csv(samples_path) if samples_path.exists() else pd.DataFrame()
    if not samples.empty:
        review = review.merge(samples[["candidate_id", "clip"]], on="candidate_id", how="left")

    st.dataframe(
        review.drop(columns=["clip"], errors="ignore"),
        hide_index=True,
        width="stretch",
        height=320,
    )
    st.download_button(
        "下载齐读候选 CSV",
        data=review_path.read_bytes(),
        file_name=review_path.name,
        mime="text/csv",
        key="download_chorus_review",
    )
    choices = {
        int(row["candidate_id"]): (
            f"候选 {int(row['candidate_id'])} · {row['start_hms']} · "
            f"能量 {row['energy_ratio']}x · {str(row.get('text') or '')[:28]}"
        )
        for _, row in review.iterrows()
    }
    selected = st.multiselect(
        "选择要标记为齐读的候选",
        options=list(choices.keys()),
        format_func=lambda value: choices[value],
        key="chorus_selected",
    )

    for _, row in review.iterrows():
        with st.expander(
            f"候选 {int(row['candidate_id'])} · {row['start_hms']} · 当前 {row['current_speaker']}"
        ):
            st.write(f"文本：{row.get('text') or '（无文本）'}")
            st.write(
                f"教师声纹相似度：{row.get('teacher_sim')}；"
                f"能量比：{row.get('energy_ratio')}；上下文提示：{row.get('context_cue')}"
            )
            clip = row.get("clip")
            if isinstance(clip, str) and Path(clip).exists():
                st.audio(clip)

    if st.button("标记为齐读", type="primary", disabled=not selected):
        accept_ids = ",".join(str(int(value)) for value in selected)
        target = run_dir / "07_chorus_applied"
        embeddings = run_dir / "02_refined" / "lesson_embeddings.npz"
        command = [
            PYTHON,
            script("detect_chorus.py"),
            final_dir / "lesson_segments.jsonl",
            audio_path,
            "--output-dir",
            target,
            "--teacher-id",
            st.session_state.get("settings", {}).get("teacher_id", "S01"),
            "--energy-ratio",
            "1.6",
            "--teacher-sim",
            "0.35",
            "--accept-ids",
            accept_ids,
            "--output-window-seconds",
            str(st.session_state.get("settings", {}).get("window_seconds", 3)),
            "--stats-window-min",
            str(st.session_state.get("settings", {}).get("stats_window_min", 5)),
        ]
        if embeddings.exists():
            command += ["--embeddings", embeddings]
        with st.status("正在标记齐读并重新生成…", expanded=True) as status:
            log_box = st.empty()
            try:
                run_command(command, log_box)
            except Exception as exc:
                status.update(label="标记失败", state="error")
                st.exception(exc)
                return
            status.update(label="标记完成", state="complete")
        st.session_state["final_dir"] = str(target)
        st.session_state["chorus_review_dir"] = None
        st.success("已标记齐读，并重新生成逐字稿、时间窗口和统计。")
        st.rerun()


def render_manual_chorus_tab():
    if st.session_state.get("settings", {}).get("asr_engine") == "iflytek":
        st.info("讯飞云端模式只输出教师/学生组，不区分齐读。")
        return
    final_dir = Path(st.session_state["final_dir"]) if st.session_state.get("final_dir") else None
    audio_path = Path(st.session_state["audio_path"]) if st.session_state.get("audio_path") else None
    if not final_dir or not audio_path:
        st.info("请先完成一次转录。")
        return

    run_dir = Path(st.session_state["run_dir"])
    st.subheader("手动标记齐读区间")
    st.caption(
        "当自动检测漏掉某段齐读时，输入起止时间并点击标记。系统会按该区间切开原片段，"
        "把区间标为 CHORUS，然后重新生成逐字稿、时间窗口和统计。"
    )
    st.audio(str(audio_path))
    st.caption("时间格式支持：`mm:ss`、`hh:mm:ss`、`mm:ss.mmm`，或直接填秒数。")

    one, two, three = st.columns([1, 1, 2])
    with one:
        start_text = st.text_input("开始时间", value="00:00", key="manual_start")
    with two:
        end_text = st.text_input("结束时间", value="00:05", key="manual_end")
    with three:
        chorus_text = st.text_input("齐读文字（可选）", key="manual_text")

    if st.button("标记为齐读", type="primary"):
        try:
            start_ms = parse_time(start_text)
            end_ms = parse_time(end_text)
            if end_ms <= start_ms:
                raise ValueError("结束时间必须晚于开始时间")
        except Exception as exc:
            st.error(str(exc))
            return

        nano_result = run_dir / "03_hybrid" / "lesson_nano_result.json"
        settings = st.session_state.get("settings", {})
        command = [
            PYTHON,
            script("mark_interval_chorus.py"),
            final_dir / "lesson_segments.jsonl",
            "--start-ms",
            str(start_ms),
            "--end-ms",
            str(end_ms),
            "--output-dir",
            str(final_dir),
            "--teacher-id",
            settings.get("teacher_id", "S01"),
            "--output-window-seconds",
            str(settings.get("window_seconds", 3)),
            "--stats-window-min",
            str(settings.get("stats_window_min", 5)),
        ]
        if chorus_text.strip():
            command += ["--text", chorus_text.strip()]
        if nano_result.exists():
            command += ["--nano-result", str(nano_result)]

        with st.status("正在标记齐读并重新生成…", expanded=True) as status:
            log_box = st.empty()
            try:
                run_command(command, log_box)
                run_command(
                    [
                        PYTHON,
                        script("punctuate_verbatim.py"),
                        final_dir / "lesson_segments.jsonl",
                        "--output-dir",
                        str(final_dir),
                        "--device",
                        settings.get("device", "auto"),
                    ],
                    log_box,
                )
            except Exception as exc:
                status.update(label="标记失败", state="error")
                st.exception(exc)
                return
            status.update(label="标记完成", state="complete")
        st.success("已标记齐读，并重新生成逐字稿、时间窗口和统计。")
        st.rerun()

    review_path = final_dir / "lesson_manual_chorus_review.csv"
    if review_path.exists():
        st.markdown("**已手动标记的区间**")
        st.dataframe(pd.read_csv(review_path), hide_index=True, width="stretch")
        st.download_button(
            "下载手动标记记录 CSV",
            data=review_path.read_bytes(),
            file_name=review_path.name,
            mime="text/csv",
            key="download_manual_chorus_review",
        )


def window_candidates(final_dir: Path, run_dir: Path | None) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    paths = list(final_dir.glob("*_window_*_fixed.csv"))
    if run_dir and run_dir.exists():
        paths += list(run_dir.glob("*/*_window_*_fixed.csv"))
    for path in sorted(paths):
        try:
            label = str(path.relative_to(run_dir)) if run_dir else str(path)
        except ValueError:
            label = str(path)
        candidates[label] = path
    return candidates


def render_coding_setup_tab():
    st.subheader("编码体系与 Prompt")
    st.caption(
        "在这里上传编码体系和 AI 编码 prompt，之后 AI 编码会自动使用当前生效的文件；"
        "结果目录会保存一份副本，便于复现。"
    )
    coding_path, prompt_path = active_coding_paths()

    def info(path: Path):
        if path.exists():
            st.caption(f"当前文件：`{path}`（{path.stat().st_size / 1024:.1f} KB）")
        else:
            st.error(f"文件不存在：{path}")

    left, right = st.columns(2)
    with left:
        st.markdown("**编码体系**")
        info(coding_path)
        coding_upload = st.file_uploader(
            "上传编码体系（.docx / .txt / .md）",
            type=["docx", "txt", "md"],
            key="coding_system_upload",
        )
    with right:
        st.markdown("**编码 Prompt**")
        info(prompt_path)
        prompt_upload = st.file_uploader(
            "上传编码 Prompt（.docx / .txt / .md）",
            type=["docx", "txt", "md"],
            key="prompt_upload",
        )

    save_col, reset_col = st.columns([1, 1])
    with save_col:
        if st.button(
            "保存上传文件并设为当前",
            type="primary",
            disabled=coding_upload is None and prompt_upload is None,
        ):
            try:
                new_coding = coding_path
                new_prompt = prompt_path
                if coding_upload is not None:
                    new_coding = save_coding_source(
                        coding_upload.getvalue(), coding_upload.name, "coding_system"
                    )
                if prompt_upload is not None:
                    new_prompt = save_coding_source(
                        prompt_upload.getvalue(), prompt_upload.name, "prompt"
                    )
                save_coding_config(new_coding, new_prompt)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.success("已保存并设为当前编码文件。")
                st.rerun()
    with reset_col:
        if st.button("恢复默认编码文件"):
            reset_coding_config()
            st.success("已恢复默认文件。")
            st.rerun()

    st.markdown("**当前内容预览**")
    preview_tab, prompt_preview_tab = st.tabs(["编码体系", "Prompt"])
    with preview_tab:
        if coding_path.exists():
            text = read_coding_document(coding_path)
            st.text_area(
                "编码体系内容",
                value=text[:20000],
                height=360,
                label_visibility="collapsed",
            )
    with prompt_preview_tab:
        if prompt_path.exists():
            text = read_coding_document(prompt_path)
            st.text_area(
                "Prompt 内容",
                value=text[:30000],
                height=360,
                label_visibility="collapsed",
            )
    st.caption(f"上传文件保存在：{CODING_CONFIG_DIR}；点击“恢复默认编码文件”会回到 `coding item/` 中的文档。")


def render_ai_code_tab():
    st.subheader("AI 课堂行为编码")
    st.caption(
        "按固定时间窗口编码。每个窗口输出一个主编码 A–P，并记录次要编码、置信度和理由。"
        "当前使用的编码文件和 prompt 可在“编码设置”标签页上传替换。"
    )
    if not deepseek_configured():
        st.error("未配置 DeepSeek API Key。请在 `my-project/.env` 中设置 `deepseek_api_key`。")
        return
    coding_path, prompt_path = active_coding_paths()
    if not coding_path.exists() or not prompt_path.exists():
        st.error("当前编码体系或 Prompt 文件不存在，请到“编码设置”标签页重新上传。")
        return
    st.caption(f"编码体系：`{coding_path.name}` · Prompt：`{prompt_path.name}`")

    final_dir = Path(st.session_state["final_dir"]) if st.session_state.get("final_dir") else None
    if not final_dir:
        st.info("请先完成一次转录。")
        return
    run_dir = Path(st.session_state["run_dir"]) if st.session_state.get("run_dir") else None
    candidates = window_candidates(final_dir, run_dir)
    if not candidates:
        st.warning("当前结果目录没有找到固定窗口 CSV。")
        return

    labels = list(candidates.keys())
    default_index = 0
    for index, label in enumerate(labels):
        if "03_iflytek_classroom" in label:
            default_index = index
            break
    selected_label = st.selectbox("选择时间窗口文件", labels, index=default_index)
    window_path = candidates[selected_label]
    windows = pd.read_csv(window_path)
    if "window_id" not in windows.columns:
        windows["window_id"] = range(1, len(windows) + 1)
    speech_seconds = float(windows["speech_s"].fillna(0).sum()) if "speech_s" in windows.columns else 0.0
    one, two, three = st.columns(3)
    one.metric("窗口数", len(windows))
    two.metric("窗口长度", f"{windows['duration_s'].iloc[0]:g} 秒" if "duration_s" in windows.columns else "-")
    three.metric("有效语音", f"{speech_seconds / 60:.1f} 分钟")

    preview_columns = [
        column
        for column in ["window_id", "start_hms", "end_hms", "duration_s", "speech_s", "dominant_role", "text"]
        if column in windows.columns
    ]
    with st.expander("窗口预览", expanded=False):
        st.dataframe(windows[preview_columns].head(20), hide_index=True, width="stretch")

    with st.expander("编码设置", expanded=True):
        left, right = st.columns(2)
        with left:
            model = st.selectbox("DeepSeek 模型", ["deepseek-chat", "deepseek-reasoner"], index=0)
            batch_size = st.selectbox("每批窗口数", [20, 40, 60, 80], index=2)
        with right:
            context_windows = st.number_input("前后上下文窗口数", min_value=0, max_value=5, value=2, step=1)
            max_windows = st.number_input(
                "调试：只编码前 N 个窗口（0 = 全部）", min_value=0, max_value=len(windows), value=0, step=1
            )

    match = re.match(r"(.+)_window_\d+(?:\.\d+)?s_fixed\.csv$", window_path.name)
    prefix = match.group(1) if match else "lesson"
    output_dir = window_path.parent
    codes_path = output_dir / f"{prefix}_ai_codes.csv"
    summary_path = output_dir / f"{prefix}_ai_code_summary.csv"
    low_path = output_dir / f"{prefix}_ai_code_low_confidence.csv"
    review_path = output_dir / f"{prefix}_ai_code_review.csv"
    sequence_path = output_dir / f"{prefix}_ai_code_sequence.txt"
    metadata_path = output_dir / f"{prefix}_ai_code_metadata.json"

    if st.button("开始 AI 编码", type="primary"):
        command = [
            PYTHON,
            script("ai_code_windows.py"),
            window_path,
            "--output-dir",
            output_dir,
            "--coding-system",
            coding_path,
            "--prompt",
            prompt_path,
            "--model",
            model,
            "--batch-size",
            str(batch_size),
            "--context-windows",
            str(context_windows),
        ]
        if int(max_windows) > 0:
            command += ["--max-windows", str(int(max_windows))]
        with st.status("正在调用 DeepSeek 进行窗口编码…", expanded=True) as status:
            log_box = st.empty()
            try:
                run_command(command, log_box)
            except Exception as exc:
                status.update(label="AI 编码失败", state="error")
                st.exception(exc)
                return
            status.update(label="AI 编码完成", state="complete")
        st.rerun()

    if not codes_path.exists():
        st.info("尚未生成 AI 编码结果。点击上方按钮开始。")
        return

    codes = pd.read_csv(codes_path)
    st.markdown("**编码结果**")
    one, two, three, four, five = st.columns(5)
    one.metric("已编码窗口", len(codes))
    two.metric("高置信度", int((codes["confidence"] == "High").sum()))
    three.metric("中置信度", int((codes["confidence"] == "Medium").sum()))
    four.metric("低置信度", int((codes["confidence"] == "Low").sum()))
    five.metric("角色冲突", int(codes["role_conflict"].sum()) if "role_conflict" in codes.columns else 0)

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.bar_chart(summary.set_index("code_name")["frequency"])
        st.dataframe(summary, hide_index=True, width="stretch")

    display_columns = [
        column
        for column in [
            "window_id",
            "start_hms",
            "end_hms",
            "speech_s",
            "dominant_role",
            "text",
            "code",
            "code_name",
            "confidence",
            "role_conflict",
            "alternative_code",
            "secondary_codes",
            "reason",
        ]
        if column in codes.columns
    ]
    st.dataframe(codes[display_columns], hide_index=True, width="stretch", height=480)

    if review_path.exists():
        review = pd.read_csv(review_path)
        if not review.empty:
            st.markdown("**低置信度或角色冲突窗口（建议人工复核）**")
            st.dataframe(review[display_columns], hide_index=True, width="stretch", height=260)
            st.download_button(
                "下载人工复核 CSV",
                data=review_path.read_bytes(),
                file_name=review_path.name,
                mime="text/csv",
                key="download_ai_review",
            )

    if sequence_path.exists():
        with st.expander("查看行为序列", expanded=False):
            st.text_area(
                "行为序列",
                value=sequence_path.read_text(encoding="utf-8-sig"),
                height=260,
                label_visibility="collapsed",
            )

    st.download_button(
        "下载 AI 编码 CSV",
        data=codes_path.read_bytes(),
        file_name=codes_path.name,
        mime="text/csv",
        key="download_ai_codes",
    )
    if summary_path.exists():
        st.download_button(
            "下载编码统计 CSV",
            data=summary_path.read_bytes(),
            file_name=summary_path.name,
            mime="text/csv",
            key="download_ai_summary",
        )
    if metadata_path.exists():
        st.caption(f"元数据：{metadata_path}")


def ai_code_candidates(final_dir: Path, run_dir: Path | None) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    paths = list(final_dir.glob("*_ai_codes.csv"))
    if run_dir and run_dir.exists():
        paths += list(run_dir.glob("*/*_ai_codes.csv"))
    for path in sorted(paths):
        try:
            label = str(path.relative_to(run_dir)) if run_dir else str(path)
        except ValueError:
            label = str(path)
        candidates[label] = path
    return candidates


def render_human_review_tab():
    st.subheader("人工复核与 AI-人工一致率")
    st.caption(
        "人工编码留空表示尚未复核；只有填写了 A–P 的窗口才计入一致率和 Cohen's κ。"
    )
    final_dir = Path(st.session_state["final_dir"]) if st.session_state.get("final_dir") else None
    if not final_dir:
        st.info("请先完成一次转录和 AI 编码。")
        return
    run_dir = Path(st.session_state["run_dir"]) if st.session_state.get("run_dir") else None
    candidates = ai_code_candidates(final_dir, run_dir)
    if not candidates:
        st.info("还没有 AI 编码结果。请先在“AI编码”标签页运行编码。")
        return
    labels = list(candidates.keys())
    default_index = 0
    for index, label in enumerate(labels):
        if "03_iflytek_classroom" in label:
            default_index = index
            break
    selected_label = st.selectbox("选择 AI 编码文件", labels, index=default_index)
    ai_path = candidates[selected_label]
    prefix = ai_path.name.removesuffix("_ai_codes.csv")
    human_path = ai_path.parent / f"{prefix}_human_codes.csv"
    agreement_json_path = ai_path.parent / f"{prefix}_ai_human_agreement.json"
    agreement_csv_path = ai_path.parent / f"{prefix}_ai_human_confusion.csv"

    ai = pd.read_csv(ai_path)
    human = pd.read_csv(human_path) if human_path.exists() else pd.DataFrame(
        columns=["window_id", "human_code", "human_note"]
    )
    merged = ai.merge(human, on="window_id", how="left", suffixes=("", "_human"))
    merged["human_code"] = merged.get("human_code", "").fillna("").astype(str).str.upper()
    merged["human_note"] = merged.get("human_note", "").fillna("").astype(str)

    top_left, top_right = st.columns([1, 2])
    with top_left:
        if st.button("接受全部 AI 编码为人工初值"):
            accepted = ai[["window_id", "code"]].copy()
            accepted = accepted.rename(columns={"code": "human_code"})
            accepted["human_note"] = "接受 AI 编码"
            accepted["reviewed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            accepted.to_csv(human_path, index=False, encoding="utf-8-sig")
            st.success("已把所有 AI 编码填入人工编码，请继续检查和修改。")
            st.rerun()
    with top_right:
        st.caption(
            "可以先“接受全部”快速建立初值，再逐条修改分歧；也可以保持空白，只复核部分窗口。"
        )

    columns = [
        column
        for column in [
            "window_id",
            "start_hms",
            "speech_s",
            "dominant_role",
            "text",
            "code",
            "confidence",
            "reason",
            "human_code",
            "human_note",
        ]
        if column in merged.columns
    ]
    display = merged[columns].copy()
    edited = st.data_editor(
        display,
        hide_index=True,
        width="stretch",
        height=560,
        num_rows="fixed",
        column_config={
            "human_code": st.column_config.SelectboxColumn(
                "人工编码",
                options=[""] + VALID_CODES,
                required=False,
            ),
            "human_note": st.column_config.TextColumn("复核备注"),
        },
        disabled=[column for column in columns if column not in {"human_code", "human_note"}],
        key=f"human_review_editor_{prefix}_{ai_path.parent.name}",
    )

    if st.button("保存人工编码并计算一致率", type="primary"):
        records = edited[["window_id", "human_code", "human_note"]].copy()
        records["human_code"] = records["human_code"].fillna("").astype(str).str.strip().str.upper()
        records["human_note"] = records["human_note"].fillna("").astype(str)
        invalid = sorted(set(records.loc[~records["human_code"].isin([""] + VALID_CODES), "human_code"]))
        if invalid:
            st.error("存在非法人工编码：" + "、".join(invalid) + "。只能填 A–P 或留空。")
            return
        reviewed = records[records["human_code"].isin(VALID_CODES)].copy()
        if reviewed.empty:
            human_path.unlink(missing_ok=True)
            st.warning("没有填写任何人工编码。")
            st.rerun()
        reviewed["reviewed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        reviewed.to_csv(human_path, index=False, encoding="utf-8-sig")

        report = build_agreement(ai, reviewed)
        stats = report["stats"]
        confusion = report["confusion"]
        agreement_payload = {
            "window_csv": selected_label,
            "reviewed": stats["reviewed"],
            "matches": stats["matches"],
            "observed_agreement": stats["observed_agreement"],
            "expected_agreement": stats["expected_agreement"],
            "cohen_kappa": stats["kappa"],
            "interpretation": stats["interpretation"],
            "high_confidence": report["high_confidence_stats"],
            "confusion": confusion.to_dict(),
        }
        agreement_json_path.write_text(
            json.dumps(agreement_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        confusion.to_csv(agreement_csv_path, encoding="utf-8-sig")
        st.success("人工编码已保存，并已计算一致率。")
        st.rerun()

    if human_path.exists():
        human = pd.read_csv(human_path)
        report = build_agreement(ai, human)
        stats = report["stats"]
        if stats["reviewed"]:
            st.markdown("**一致率结果**")
            one, two, three, four = st.columns(4)
            one.metric("已复核窗口", stats["reviewed"])
            one.caption(f"共 {len(ai)} 个窗口")
            two.metric("一致率", f"{stats['observed_agreement'] * 100:.1f}%")
            three.metric("Cohen's κ", f"{stats['kappa']:.3f}")
            four.metric("一致性评价", stats["interpretation"])
            if report["high_confidence_stats"]:
                high = report["high_confidence_stats"]
                st.caption(
                    f"AI 高置信度子集：{high['reviewed']} 个窗口，"
                    f"一致率 {high['observed_agreement'] * 100:.1f}%，"
                    f"κ = {high['kappa']:.3f}。"
                )

            confusion = report["confusion"]
            used_codes = [code for code in VALID_CODES if confusion.loc[code].sum() or confusion[code].sum()]
            if used_codes:
                st.markdown("**混淆矩阵（AI 行 × 人工列）**")
                st.dataframe(
                    confusion.loc[used_codes, used_codes],
                    width="stretch",
                )

            disagreements = report["disagreements"]
            if not disagreements.empty:
                st.markdown(f"**AI 与人工不一致窗口：{len(disagreements)} 个**")
                disagreement_columns = [
                    column
                    for column in [
                        "window_id",
                        "start_hms",
                        "text",
                        "code",
                        "human_code",
                        "confidence",
                        "reason",
                        "human_note",
                    ]
                    if column in disagreements.columns
                ]
                st.dataframe(
                    disagreements[disagreement_columns],
                    hide_index=True,
                    width="stretch",
                    height=300,
                )

            st.download_button(
                "下载人工编码 CSV",
                data=human_path.read_bytes(),
                file_name=human_path.name,
                mime="text/csv",
                key="download_human_codes",
            )
            if agreement_json_path.exists():
                st.download_button(
                    "下载一致率 JSON",
                    data=agreement_json_path.read_bytes(),
                    file_name=agreement_json_path.name,
                    mime="application/json",
                    key="download_agreement_json",
                )
            if agreement_csv_path.exists():
                st.download_button(
                    "下载混淆矩阵 CSV",
                    data=agreement_csv_path.read_bytes(),
                    file_name=agreement_csv_path.name,
                    mime="text/csv",
                    key="download_agreement_csv",
                )
        else:
            st.info("人工编码文件存在，但还没有有效的 A–P 编码。")
    else:
        st.info("尚未保存人工编码。填写表中的“人工编码”后保存。")


def render_behavior_sequence_tab():
    st.subheader("行为序列与滞后序列分析")
    st.caption(
        "使用调整后残差（adjusted residual, Z-score）分析行为转换。"
        "|Z| ≥ 1.96 为 p < .05，≥ 2.58 为 p < .01，≥ 3.29 为 p < .001。"
    )
    final_dir = Path(st.session_state["final_dir"]) if st.session_state.get("final_dir") else None
    if not final_dir:
        st.info("请先完成一次转录和 AI 编码。")
        return
    run_dir = Path(st.session_state["run_dir"]) if st.session_state.get("run_dir") else None
    candidates = ai_code_candidates(final_dir, run_dir)
    if not candidates:
        st.info("还没有 AI 编码结果。请先在“AI编码”标签页运行编码。")
        return
    labels = list(candidates.keys())
    default_index = 0
    for index, label in enumerate(labels):
        if "03_iflytek_classroom" in label:
            default_index = index
            break
    selected_label = st.selectbox("选择 AI 编码文件", labels, index=default_index, key="seq_ai_file")
    ai_path = candidates[selected_label]
    prefix = ai_path.name.removesuffix("_ai_codes.csv")
    human_path = ai_path.parent / f"{prefix}_human_codes.csv"
    output_dir = ai_path.parent / "behavior_sequence_v02b"
    ai_columns = pd.read_csv(ai_path, nrows=0).columns

    source_options = ["AI 编码"]
    source_map = {"AI 编码": "ai"}
    if human_path.exists():
        source_options += ["人工复核（仅已复核窗口）", "人工复核 + AI 补全"]
        source_map["人工复核（仅已复核窗口）"] = "human"
        source_map["人工复核 + AI 补全"] = "human_or_ai"
    source_label = st.selectbox("序列编码来源", source_options)
    source = source_map[source_label]

    with st.expander("行为编码字典", expanded=False):
        dictionary_path = active_dictionary_path()
        if dictionary_path:
            st.caption(f"当前字典：`{dictionary_path}`")
        else:
            st.caption("未加载字典；图表和报告将显示原始 A–P 编码。")
        dictionary_upload = st.file_uploader(
            "上传编码字典（CSV / JSON）",
            type=["csv", "json"],
            key="coding_dictionary_upload",
        )
        left, right = st.columns(2)
        with left:
            if st.button("保存编码字典", disabled=dictionary_upload is None):
                try:
                    dictionary_path = save_dictionary(
                        dictionary_upload.getvalue(), dictionary_upload.name
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.success("编码字典已保存。")
                    st.rerun()
        with right:
            if st.button("清除编码字典"):
                reset_dictionary()
                st.rerun()
        dictionary = load_dictionary(dictionary_path)
        if dictionary:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "code": code,
                            "label": entry.get("label", ""),
                            "definition": entry.get("definition", ""),
                            "category": entry.get("category", ""),
                        }
                        for code, entry in dictionary.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
    dictionary = load_dictionary(dictionary_path)

    collapsed_frequency_path = output_dir / f"{prefix}_collapsed_frequency.csv"
    collapsed_transition_path = output_dir / f"{prefix}_collapsed_transition_frequency.csv"
    collapsed_probability_path = output_dir / f"{prefix}_collapsed_transition_probability.csv"
    collapsed_segments_path = output_dir / f"{prefix}_collapsed_segments.csv"
    collapsed_stats_path = output_dir / f"{prefix}_collapsed_duration_stats.csv"
    collapsed_metadata_path = output_dir / f"{prefix}_collapsed_metadata.json"

    view = st.radio(
        "分析视角",
        ["A 固定窗口状态", "B 合并状态段（仅描述）"],
        horizontal=True,
        help="B 视角不继承 A 视角的显著性检验结果。",
    )
    if view.startswith("B"):
        st.caption(
            "B 视角将每条序列内部连续相同编码合并为状态段，仅报告描述性频次、概率和持续时间；"
            "不显示 p 值或显著性星号。"
        )
        if not collapsed_transition_path.exists():
            st.info("尚未生成合并状态结果。请先在 A 视角运行一次 V0.2c 分析。")
            return
        collapsed_frequency = pd.read_csv(collapsed_frequency_path) if collapsed_frequency_path.exists() else pd.DataFrame()
        collapsed_transition = pd.read_csv(collapsed_transition_path, index_col=0)
        collapsed_probability = pd.read_csv(collapsed_probability_path, index_col=0) if collapsed_probability_path.exists() else pd.DataFrame()
        collapsed_segments = pd.read_csv(collapsed_segments_path) if collapsed_segments_path.exists() else pd.DataFrame()
        collapsed_stats = pd.read_csv(collapsed_stats_path) if collapsed_stats_path.exists() else pd.DataFrame()
        collapsed_meta = json.loads(collapsed_metadata_path.read_text(encoding="utf-8")) if collapsed_metadata_path.exists() else {}
        one, two, three, four = st.columns(4)
        one.metric("状态段数", collapsed_meta.get("state_count", "-"))
        two.metric("状态段转移", collapsed_meta.get("transition_count", "-"))
        three.metric("状态段序列", collapsed_meta.get("sequence_count", "-"))
        four.metric("分析单位", "状态段")
        if not collapsed_stats.empty:
            st.markdown("**状态段持续时间**")
            st.dataframe(collapsed_stats, hide_index=True, width="stretch")
        if not collapsed_frequency.empty:
            st.markdown("**状态段行为频次**")
            st.dataframe(collapsed_frequency, hide_index=True, width="stretch")
        if not collapsed_transition.empty:
            st.markdown("**状态段转换频次**")
            st.dataframe(collapsed_transition, width="stretch")
        if not collapsed_probability.empty:
            st.markdown("**状态段条件转移概率**")
            st.dataframe(collapsed_probability, width="stretch")
        if not collapsed_segments.empty:
            with st.expander("状态段明细", expanded=False):
                st.dataframe(collapsed_segments, hide_index=True, width="stretch", height=320)
        st.caption("B 视角为描述性敏感性分析，不可解释为每 5 秒窗口的转移概率。")
        return

    st.markdown("**观察序列边界**")
    sequence_column = st.text_input(
        "序列 ID 列名",
        value="sequence_id",
        help="如果 CSV 没有该列，必须确认整个文件是一个连续观察序列。",
    )
    sequence_present = sequence_column in ai_columns
    single_sequence = False
    if sequence_present:
        st.success(f"检测到序列列 `{sequence_column}`，将按该列分组计算，不跨序列连接。")
    else:
        st.warning(
            "当前 CSV 没有序列 ID 列。只有在确认整个文件是一个完整连续观察序列时，才可继续。"
        )
        single_sequence = st.checkbox(
            "我确认当前文件是一个完整连续观察序列（无 sequence_id）",
            value=False,
        )

    left, right = st.columns(2)
    with left:
        lag = st.number_input("滞后阶数", min_value=1, max_value=5, value=1, step=1)
        alpha = st.selectbox("显著性水平", [0.05, 0.01, 0.001], index=0)
    with right:
        edge_mode_label = st.selectbox("转换图边类型", ["仅正向显著", "正向 + 负向显著"], index=0)
        edge_mode = "positive" if edge_mode_label == "仅正向显著" else "both"
        max_edges = st.number_input("最多绘制边数", min_value=5, max_value=100, value=30, step=5)
    exclude_n = st.checkbox("排除 N（停顿/沉默）", value=False)
    sim_left, sim_right = st.columns(2)
    with sim_left:
        simulation_count = st.number_input(
            "Monte Carlo 模拟次数", min_value=100, max_value=50000, value=10000, step=1000
        )
    with sim_right:
        random_seed = st.number_input("随机种子", min_value=0, max_value=2**31 - 1, value=20261022, step=1)

    disabled = not sequence_present and not single_sequence
    if st.button("生成行为序列分析", type="primary", disabled=disabled):
        command = [
            PYTHON,
            script("behavior_sequence.py"),
            ai_path,
            "--output-dir",
            output_dir,
            "--source",
            source,
            "--sequence-id-column",
            sequence_column,
            "--lag",
            str(int(lag)),
            "--alpha",
            str(float(alpha)),
            "--edge-mode",
            edge_mode,
            "--max-edges",
            str(int(max_edges)),
            "--simulation-count",
            str(int(simulation_count)),
            "--random-seed",
            str(int(random_seed)),
        ]
        if human_path.exists():
            command += ["--human-codes", human_path]
        if not sequence_present:
            command += ["--single-sequence"]
        if exclude_n:
            command += ["--exclude-codes", "N"]
        with st.status("正在计算行为频次、调整残差和转换图…", expanded=True) as status:
            log_box = st.empty()
            try:
                run_command(command, log_box)
            except Exception as exc:
                status.update(label="行为序列分析失败", state="error")
                st.exception(exc)
                return
            status.update(label="行为序列分析完成", state="complete")
        st.rerun()

    frequency_path = output_dir / f"{prefix}_behavior_frequency.csv"
    transition_path = output_dir / f"{prefix}_transition_frequency.csv"
    adjusted_path = output_dir / f"{prefix}_adjusted_residuals_starred.csv"
    diagnostics_path = output_dir / f"{prefix}_cell_diagnostics.csv"
    inference_path = output_dir / f"{prefix}_cell_inference.csv"
    overlap_path = output_dir / f"{prefix}_significant_overlap.csv"
    method_summary_path = output_dir / f"{prefix}_method_summary.csv"
    status_path = output_dir / f"{prefix}_cell_status.csv"
    metadata_path = output_dir / f"{prefix}_behavior_sequence_metadata.json"
    collapsed_frequency_path = output_dir / f"{prefix}_collapsed_frequency.csv"
    collapsed_transition_path = output_dir / f"{prefix}_collapsed_transition_frequency.csv"
    collapsed_probability_path = output_dir / f"{prefix}_collapsed_transition_probability.csv"
    collapsed_metadata_path = output_dir / f"{prefix}_collapsed_metadata.json"
    if not frequency_path.exists():
        st.info("尚未生成行为序列结果。点击上方按钮开始。")
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if metadata.get("sequence_mode") == "single_sequence_assumed":
        st.warning(metadata.get("boundary_assumption", "当前按单一连续序列处理。"))
    with st.expander("分析限制", expanded=False):
        st.markdown(
            "- 低期望频数可能使正态近似不可靠。\n"
            "- 相邻转移共享观察窗口，不是独立观测。\n"
            "- 固定时间窗口可能存在行为持续性和自相关。\n"
            "- Monte Carlo 随机排列不保留原始自相关。\n"
            "- 单课堂结果不能直接推广到其他课堂。\n"
            "- 显著转移不代表因果关系。"
        )
        if metadata.get("warnings"):
            st.caption("元数据警告：" + "；".join(metadata["warnings"]))
    method_options = {
        "经典 LSA：原始 p": "raw",
        "经典 LSA：Holm 校正": "holm",
        "经典 LSA：BH-FDR": "bh",
        "Monte Carlo：原始经验 p": "mc_raw",
        "Monte Carlo：Holm 校正": "mc_holm",
        "Monte Carlo：BH-FDR": "mc_bh",
    }
    method_label = st.selectbox(
        "统计推断方式",
        list(method_options),
        index=1,
        help="默认使用经典 LSA 的 Holm 校正。不同推断方式不会混合绘制。",
    )
    method = method_options[method_label]
    st.caption("负向显著（Z ≤ -1.96）只保留在诊断表中，不进入主显著表、主图和论文研究发现。")
    significant_path = output_dir / f"{prefix}_significant_transitions_{method}.csv"
    dot_path = output_dir / f"{prefix}_transition_graph_{method}.dot"

    st.markdown("**分析概要**")
    one, two, three, four, five, six = st.columns(6)
    one.metric("观察序列", metadata.get("sequence_count", "-"))
    two.metric("行为窗口", metadata.get("total_windows", "-"))
    three.metric("行为转换", metadata.get("total_transitions", "-"))
    four.metric("显著转换", len(pd.read_csv(significant_path)) if significant_path.exists() else "-")
    five.metric("不可检验单元", metadata.get("untestable_cells", "-"))
    expected_lt5_by_method = metadata.get("expected_lt5_significant", {})
    six.metric("当前方法 E<5", expected_lt5_by_method.get(method, "-"))

    frequency = pd.read_csv(frequency_path)
    transition = pd.read_csv(transition_path, index_col=0)
    adjusted = pd.read_csv(adjusted_path, index_col=0)
    status = pd.read_csv(status_path, index_col=0) if status_path.exists() else pd.DataFrame()
    significant = pd.read_csv(significant_path) if significant_path.exists() else pd.DataFrame()
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    inference = pd.read_csv(inference_path) if inference_path.exists() else pd.DataFrame()
    overlap = pd.read_csv(overlap_path) if overlap_path.exists() else pd.DataFrame()
    method_summary = pd.read_csv(method_summary_path) if method_summary_path.exists() else pd.DataFrame()
    collapsed_frequency = pd.read_csv(collapsed_frequency_path) if collapsed_frequency_path.exists() else pd.DataFrame()
    collapsed_transition = pd.read_csv(collapsed_transition_path, index_col=0) if collapsed_transition_path.exists() else pd.DataFrame()
    collapsed_probability = pd.read_csv(collapsed_probability_path, index_col=0) if collapsed_probability_path.exists() else pd.DataFrame()
    collapsed_metadata = json.loads(collapsed_metadata_path.read_text(encoding="utf-8")) if collapsed_metadata_path.exists() else {}

    table_tab, residual_tab, significant_tab, sparse_tab, comparison_tab, sensitivity_tab, graph_tab = st.tabs(
        ["行为频次", "转换频次", "调整残差", "稀疏诊断", "方法对照", "敏感性", "转换图"]
    )
    with table_tab:
        st.bar_chart(frequency.set_index("code_name")["frequency"])
        st.dataframe(frequency, hide_index=True, width="stretch")
    with residual_tab:
        st.dataframe(transition, width="stretch")
        st.caption("单元格为给定行为（行）→ 目标行为（列）的转换频次。")
        st.markdown("**调整后残差表（含显著性星号；NA 表示不可定义）**")
        st.dataframe(adjusted, width="stretch")
        if not status.empty:
            st.markdown("**单元状态**")
            st.dataframe(status, width="stretch")
    with significant_tab:
        st.caption(
            f"当前推断方式：**{method_label}**。"
            "不可定义单元不参与校正；E<5 表示正态近似需谨慎解释。"
        )
        if significant.empty:
            st.info("当前参数下没有显著转换。")
        else:
            st.dataframe(significant, hide_index=True, width="stretch", height=360)
    with sparse_tab:
        if diagnostics.empty:
            st.info("尚未生成稀疏诊断。")
        else:
            st.caption(
                "`testable=False` 表示期望频数或边际为 0，调整后残差和 p 值不可定义。"
                "`expected_lt_5=True` 表示正态近似需要谨慎解释，但不等于不显著。"
            )
            untestable = diagnostics[~diagnostics["testable"].astype(bool)]
            if not untestable.empty:
                st.markdown(f"**不可检验单元：{len(untestable)} 个**")
                st.dataframe(untestable, hide_index=True, width="stretch", height=260)
            low_expected = diagnostics[
                diagnostics["testable"].astype(bool) & diagnostics["expected_lt_5"].astype(bool)
            ]
            if not low_expected.empty:
                st.markdown(f"**可检验但 E<5 的单元：{len(low_expected)} 个**")
                st.dataframe(low_expected, hide_index=True, width="stretch", height=320)
    with comparison_tab:
        if not method_summary.empty:
            st.markdown("**各方法显著转移数量**")
            st.dataframe(method_summary, hide_index=True, width="stretch")
        if not overlap.empty:
            st.markdown("**显著边重叠情况**")
            st.dataframe(overlap, hide_index=True, width="stretch", height=360)
        if not inference.empty:
            st.markdown("**逐单元经典与 Monte Carlo 推断对照**")
            inference_columns = [
                column
                for column in [
                    "from_code",
                    "to_code",
                    "observed",
                    "expected",
                    "adjusted_residual",
                    "raw_p_value",
                    "p_holm",
                    "q_bh",
                    "sig_raw",
                    "sig_holm",
                    "sig_bh",
                    "simulated_mean",
                    "empirical_p_two_sided",
                    "empirical_p_holm",
                    "empirical_q_bh",
                    "simulation_significance",
                    "simulation_significance_holm",
                    "simulation_significance_bh",
                    "expected_lt_5",
                    "warning_reason",
                ]
                if column in inference.columns
            ]
            st.dataframe(inference[inference_columns], hide_index=True, width="stretch", height=360)
    with sensitivity_tab:
        st.caption(
            "这是描述性敏感性分析：仅将每条序列内部连续相同编码合并为状态段；"
            "原始 5 秒窗口分析仍保留，二者单位不同，不能直接比较概率。"
        )
        if not collapsed_frequency.empty:
            st.dataframe(collapsed_frequency, hide_index=True, width="stretch")
        if not collapsed_transition.empty:
            st.markdown("**合并状态后的转换频次**")
            st.dataframe(collapsed_transition, width="stretch")
        if not collapsed_probability.empty:
            st.markdown("**合并状态后的条件转移概率**")
            st.dataframe(collapsed_probability, width="stretch")
        if collapsed_metadata:
            st.caption(
                f"状态段数：{collapsed_metadata.get('state_count')}，"
                f"状态转移数：{collapsed_metadata.get('transition_count')}；"
                "不进行显著性推断。"
            )
    with graph_tab:
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            show_self = st.checkbox("显示自转移", value=True, key="seq_show_self")
        with g2:
            min_count = st.number_input("最低转移频数", min_value=0, max_value=1000, value=0, step=1)
        with g3:
            min_probability = st.number_input(
                "最低条件转移概率", min_value=0.0, max_value=1.0, value=0.0, step=0.05
            )
        with g4:
            label_mode = st.selectbox("节点标签", ["code", "label", "both"], index=2)
        if significant.empty:
            st.info("当前方法下没有显著转换，图中只显示状态节点。")
        node_frequency = {
            str(row["code"]): int(row["frequency"]) for _, row in frequency.iterrows()
        }
        dot_text = build_dot(
            {"labels": list(frequency["code"])},
            float(alpha),
            edge_mode,
            int(max_edges),
            method=method,
            significant=significant,
            dictionary=dictionary,
            label_mode=label_mode,
            show_self=show_self,
            min_count=int(min_count),
            min_probability=float(min_probability),
            node_frequency=node_frequency,
        )
        st.graphviz_chart(dot_text, width="stretch")
        st.caption(
            f"当前图：{method_label}。蓝色实线=正向显著；红色虚线=负向显著；"
            "橙色点线=E<5；边标签为当前统计指标与 P(j|i)。"
        )
        if st.button("导出当前图 SVG / PNG", key="export_seq_graph"):
            selected_edges = significant.copy()
            if edge_mode == "positive":
                selected_edges = selected_edges[selected_edges["direction"] == "positive"]
            if not show_self:
                selected_edges = selected_edges[selected_edges["from_code"] != selected_edges["to_code"]]
            if min_count > 0:
                selected_edges = selected_edges[selected_edges["observed"] >= min_count]
            if min_probability > 0:
                selected_edges = selected_edges[
                    selected_edges["transition_probability"].fillna(0) >= min_probability
                ]
            export_nodes = frequency.copy()
            export_nodes["display_label"] = [
                label_for(code, dictionary, label_mode) for code in export_nodes["code"]
            ]
            if method in {"raw", "holm", "bh"}:
                selected_edges["weight"] = selected_edges["adjusted_residual"].abs()
                selected_edges["direction"] = selected_edges["adjusted_residual"].apply(
                    lambda value: "positive" if value >= 0 else "negative"
                )
            else:
                selected_edges["weight"] = selected_edges["empirical_p_two_sided"].apply(
                    lambda value: max(0.5, -math.log10(max(float(value), 1e-12))) if pd.notna(value) else 0.5
                )
                selected_edges["direction"] = selected_edges.get(
                    "mc_direction", selected_edges.get("direction", "positive")
                )
            paths = render_transition_graph(
                export_nodes,
                selected_edges,
                output_dir,
                f"lesson_graph_{method}",
                f"{method_label} 显著转移图",
                label_mode=label_mode,
            )
            st.session_state["seq_graph_paths"] = paths
        graph_paths = st.session_state.get("seq_graph_paths")
        if graph_paths:
            if graph_paths.get("png"):
                st.image(graph_paths["png"], caption="当前筛选后的转换图")
            for kind in ("svg", "png"):
                path = Path(graph_paths[kind])
                if path.exists():
                    st.download_button(
                        f"下载 {kind.upper()}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="image/svg+xml" if kind == "svg" else "image/png",
                        key=f"download_seq_graph_{kind}",
                    )

    report_path = output_dir / f"lesson_report_{method}.html"
    report_col, report_info = st.columns([1, 2])
    with report_col:
        if st.button("生成 HTML 研究报告", type="primary"):
            command = [
                PYTHON,
                script("report_builder.py"),
                output_dir,
                "--method",
                method,
                "--output",
                report_path,
            ]
            if dictionary_path:
                command += ["--dictionary", dictionary_path]
            with st.status("正在生成 HTML 研究报告…", expanded=True) as status:
                log_box = st.empty()
                try:
                    run_command(command, log_box)
                except Exception as exc:
                    status.update(label="报告生成失败", state="error")
                    st.exception(exc)
                    return
                status.update(label="报告生成完成", state="complete")
            st.rerun()
    with report_info:
        if report_path.exists():
            st.caption(f"报告：`{report_path}`")
            st.download_button(
                "下载 HTML 研究报告",
                data=report_path.read_bytes(),
                file_name=report_path.name,
                mime="text/html",
                key="download_lsa_report",
            )
        else:
            st.caption("可生成包含数据单位、方法、显著转移、稀疏诊断、敏感性分析和限制说明的 HTML 报告。")

    st.download_button(
        "下载行为频次 CSV",
        data=frequency_path.read_bytes(),
        file_name=frequency_path.name,
        mime="text/csv",
        key="download_behavior_frequency",
    )
    if transition_path.exists():
        st.download_button(
            "下载转换频次 CSV",
            data=transition_path.read_bytes(),
            file_name=transition_path.name,
            mime="text/csv",
            key="download_transition_frequency",
        )
    if adjusted_path.exists():
        st.download_button(
            "下载调整残差 CSV",
            data=adjusted_path.read_bytes(),
            file_name=adjusted_path.name,
            mime="text/csv",
            key="download_adjusted_residuals",
        )
    if diagnostics_path.exists():
        st.download_button(
            "下载稀疏诊断 CSV",
            data=diagnostics_path.read_bytes(),
            file_name=diagnostics_path.name,
            mime="text/csv",
            key="download_cell_diagnostics",
        )
    if significant_path.exists():
        st.download_button(
            "下载当前方法显著转移 CSV",
            data=significant_path.read_bytes(),
            file_name=significant_path.name,
            mime="text/csv",
            key="download_current_significant",
        )
    if inference_path.exists():
        st.download_button(
            "下载逐单元推断对照 CSV",
            data=inference_path.read_bytes(),
            file_name=inference_path.name,
            mime="text/csv",
            key="download_cell_inference",
        )
    if method_summary_path.exists():
        st.download_button(
            "下载方法对照 CSV",
            data=method_summary_path.read_bytes(),
            file_name=method_summary_path.name,
            mime="text/csv",
            key="download_method_summary",
        )
    if collapsed_transition_path.exists():
        st.download_button(
            "下载合并状态转换频次 CSV",
            data=collapsed_transition_path.read_bytes(),
            file_name=collapsed_transition_path.name,
            mime="text/csv",
            key="download_collapsed_transition",
        )
    if dot_path.exists():
        st.download_button(
            "下载当前方法转换图 DOT",
            data=dot_path.read_bytes(),
            file_name=dot_path.name,
            mime="text/vnd.graphviz",
            key="download_transition_dot",
        )


def render_stats_tab():
    final_dir = Path(st.session_state["final_dir"]) if st.session_state.get("final_dir") else None
    if not final_dir:
        st.info("请先在“开始转录”中处理一段录音。")
        return
    path = stats_csv_path(final_dir)
    if not path:
        st.warning("尚未生成统计。")
        return
    stats = pd.read_csv(path)
    st.subheader("教学行为窗口统计")
    st.caption(
        "统计中的 `teacher_s`、`student_s`、`total_s` 都是有效语音秒数，不是窗口总时长；"
        "例如 5 分钟统计窗口最多 300 秒，但静音和停顿不会计入。"
    )
    st.dataframe(stats, hide_index=True, width="stretch", height=420)
    chart_columns = [
        column
        for column in ["teacher_s", "student_s", "chorus_s"]
        if column in stats.columns
    ]
    if "start_hms" in stats.columns and chart_columns:
        chart = stats.set_index("start_hms")[chart_columns]
        st.bar_chart(chart)
    ratio_columns = [
        column
        for column in ["student_ratio", "chorus_ratio"]
        if column in stats.columns
    ]
    if "start_hms" in stats.columns and ratio_columns:
        st.line_chart(stats.set_index("start_hms")[ratio_columns])
    st.download_button(
        "下载统计 CSV",
        data=path.read_bytes(),
        file_name=path.name,
        mime="text/csv",
        key="download_stats",
    )


def render_export_tab():
    final_dir = Path(st.session_state["final_dir"]) if st.session_state.get("final_dir") else None
    if not final_dir:
        st.info("请先在“开始转录”中处理一段录音。")
        return
    st.subheader("导出结果")
    files = {
        "片段 CSV": final_dir / "lesson_segments.csv",
        "逐字稿 TXT": final_dir / "lesson_verbatim.txt",
        "说话人汇总 CSV": final_dir / "lesson_speaker_summary.csv",
    }
    window_path = window_csv_path(final_dir)
    if window_path:
        files["时间窗口 CSV"] = window_path
        files["时间窗口 TXT"] = window_path.with_suffix(".txt")
    stats_path = stats_csv_path(final_dir)
    if stats_path:
        files["统计 CSV"] = stats_path
    if (final_dir / "lesson_iflytek_meta.json").exists():
        files["讯飞元数据 JSON"] = final_dir / "lesson_iflytek_meta.json"
    for label, path in files.items():
        if path.exists():
            st.download_button(
                f"下载 {label}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="text/csv" if path.suffix == ".csv" else "text/plain",
                key=f"download_export_{label}",
            )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in final_dir.glob("*"):
            if path.is_file():
                archive.write(path, arcname=path.name)
    st.download_button(
        "下载全部结果 ZIP",
        data=buffer.getvalue(),
        file_name="classroom_results.zip",
        mime="application/zip",
        key="download_zip",
    )
    st.caption(f"结果目录：{final_dir}")


def render_multi_classroom_page():
    st.header("多课堂联合分析")
    st.caption(
        "每节课堂作为独立观察序列；联合矩阵只做描述性汇总，不把多课堂拼接成一条长序列，"
        "也不提供课堂层级显著性推断。"
    )
    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    if not classroom_registry.REGISTRY_PATH.exists():
        classroom_registry.scan_and_update()
    project_tab, overview_tab, compare_tab, stability_tab, export_tab = st.tabs(
        ["研究项目", "联合概览", "课堂比较", "稳定性", "结果导出"]
    )

    with project_tab:
        st.subheader("创建或导入研究项目")
        st.markdown("**方式 A：从已完成课堂选择（推荐）**")
        if st.button("刷新课堂注册表", key="refresh_class_registry"):
            classroom_registry.scan_and_update()
            st.rerun()
        registry = classroom_registry.load_registry()
        records = [
            record
            for record in registry.get("classrooms", {}).values()
            if record.get("status") == "ready" and record.get("ai_codes_path")
        ]
        manifest_paths = sorted(STUDIES_DIR.glob("*/study_manifest.json"))
        existing_targets = {str(path.parent.name): path for path in manifest_paths}
        if records:
            record_by_id = {record["class_id"]: record for record in records}
            display_rows = []
            for record in records:
                display_rows.append(
                    {
                        "include": True,
                        "class_id": record["class_id"],
                        "display_name": record.get("display_name", record["class_id"]),
                        "source_audio": Path(record.get("source_audio", "")).name if record.get("source_audio") else "",
                        "windows": record.get("total_windows", 0),
                        "coding_scheme": record.get("coding_scheme_id", "unknown"),
                        "window_seconds": record.get("window_duration_seconds"),
                        "review_status": record.get("review_status", "absent"),
                        "code_source": "human" if record.get("review_status") == "complete" else "ai",
                        "has_lsa": bool(record.get("behavior_sequence_path")),
                    }
                )
            edited_records = st.data_editor(
                pd.DataFrame(display_rows),
                hide_index=True,
                width="stretch",
                disabled=[
                    "class_id",
                    "source_audio",
                    "windows",
                    "coding_scheme",
                    "window_seconds",
                    "review_status",
                    "has_lsa",
                ],
                column_config={
                    "include": st.column_config.CheckboxColumn("选择"),
                    "code_source": st.column_config.SelectboxColumn(
                        "编码来源", options=["ai", "human"], required=True
                    ),
                },
                key="registry_class_selection",
            )
            target = st.selectbox(
                "目标研究项目",
                ["新建研究项目"] + list(existing_targets),
                key="registry_target_study",
            )
            if target == "新建研究项目":
                new_study_id = st.text_input("新研究项目编号", value="teacher_case_new", key="registry_new_study")
                new_teacher_id = st.text_input("教师编号", value="T01", key="registry_new_teacher")
            if st.button("从所选课堂创建 / 追加研究项目", type="primary", key="registry_create_study"):
                selected = edited_records[edited_records["include"].astype(bool)]
                if selected.empty:
                    st.error("请至少选择一节课堂。")
                else:
                    entries = []
                    for _, row in selected.iterrows():
                        record = record_by_id[row["class_id"]]
                        entry = classroom_registry.record_to_manifest_entry(
                            record, code_source=row["code_source"]
                        )
                        entry["display_name"] = str(row["display_name"]).strip() or record["class_id"]
                        entries.append(entry)
                    try:
                        if target == "新建研究项目":
                            study_dir = STUDIES_DIR / new_study_id.strip()
                            if (study_dir / "study_manifest.json").exists():
                                raise ValueError("研究项目已存在，请选择目标研究项目进行追加。")
                            study_dir.mkdir(parents=True, exist_ok=True)
                            manifest = {
                                "study_id": new_study_id.strip(),
                                "teacher_id": new_teacher_id,
                                "coding_scheme_id": entries[0].get("coding_scheme_id", "unknown"),
                                "window_duration_seconds": entries[0].get("window_duration_seconds"),
                                "lag": 1,
                                "classes": entries,
                            }
                            errors = multi.validate_manifest(study_dir, manifest)
                            if errors:
                                raise ValueError("\n".join(errors))
                            multi.save_manifest(study_dir / "study_manifest.json", manifest)
                        else:
                            manifest_path = existing_targets[target]
                            multi.append_classes(manifest_path.parent, manifest_path, entries)
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success("研究项目已更新。")
                        st.rerun()
        else:
            st.info("尚未发现可用的历史课堂。请先完成转录和 AI 编码，或使用下方外部 CSV 导入。")

        with st.expander("新建研究项目（上传外部 CSV）", expanded=False):
            one, two = st.columns(2)
            with one:
                study_id = st.text_input("study_id", value="teacher_case_001")
                teacher_id = st.text_input("teacher_id", value="T01")
            with two:
                coding_scheme_id = st.text_input("coding_scheme_id", value="default_scheme")
                window_seconds = st.number_input(
                    "窗口长度（秒）", min_value=1.0, max_value=60.0, value=5.0, step=1.0
                )
            uploads = st.file_uploader(
                "选择多个已完成的课堂编码 CSV",
                type=["csv"],
                accept_multiple_files=True,
                key="multi_class_uploads",
            )
            no_sequence_confirm = st.checkbox(
                "确认没有 sequence_id 的文件各自是一个完整连续观察序列",
                value=False,
            )
            if uploads:
                rows = []
                for upload in uploads:
                    frame = pd.read_csv(upload, nrows=0)
                    rows.append(
                        {
                            "file": upload.name,
                            "class_id": Path(upload.name).stem,
                            "review_status": "reviewed",
                            "has_sequence_id": "sequence_id" in frame.columns,
                        }
                    )
                edited = st.data_editor(
                    pd.DataFrame(rows),
                    hide_index=True,
                    width="stretch",
                    disabled=["file", "has_sequence_id"],
                    key="multi_class_mapping",
                )
                if st.button("保存研究项目", type="primary"):
                    errors = []
                    class_ids = edited["class_id"].astype(str).str.strip()
                    if class_ids.duplicated().any():
                        errors.append("存在重复 class_id")
                    if (class_ids == "").any():
                        errors.append("存在空 class_id")
                    by_name = {upload.name: upload for upload in uploads}
                    hashes = {}
                    entries = []
                    study_dir = STUDIES_DIR / study_id.strip()
                    for _, row in edited.iterrows():
                        class_id = str(row["class_id"]).strip()
                        upload = by_name.get(row["file"])
                        if upload is None:
                            continue
                        data = upload.getvalue()
                        file_hash = hashlib.sha256(data).hexdigest()
                        if file_hash in hashes:
                            errors.append(f"重复文件：{class_id} 与 {hashes[file_hash]}")
                        hashes[file_hash] = class_id
                        target = study_dir / "classes" / class_id / "lesson_ai_codes.csv"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                        has_sequence = bool(row.get("has_sequence_id", False))
                        if not has_sequence and not no_sequence_confirm:
                            errors.append(f"{class_id} 没有 sequence_id，但尚未确认连续序列")
                        entries.append(
                            {
                                "class_id": class_id,
                                "teacher_id": teacher_id,
                                "source_video": "",
                                "coding_scheme_id": coding_scheme_id,
                                "window_duration_seconds": float(window_seconds),
                                "review_status": str(row.get("review_status", "reviewed")),
                                "file": str(target.relative_to(study_dir)),
                                "sha256": file_hash,
                                "sequence_id_column": "sequence_id",
                                "single_sequence": not has_sequence,
                            }
                        )
                    if errors:
                        for error in errors:
                            st.error(error)
                    else:
                        manifest = {
                            "study_id": study_id.strip(),
                            "teacher_id": teacher_id,
                            "coding_scheme_id": coding_scheme_id,
                            "window_duration_seconds": float(window_seconds),
                            "lag": 1,
                            "classes": entries,
                        }
                        multi.save_manifest(study_dir / "study_manifest.json", manifest)
                        st.success(f"研究项目已保存：{study_dir / 'study_manifest.json'}")
                        st.rerun()

        with st.expander("方式 B：导入外部 CSV（字段映射）", expanded=False):
            external_upload = st.file_uploader(
                "选择一个外部课堂编码 CSV",
                type=["csv"],
                key="external_mapping_upload",
            )
            if external_upload:
                raw = pd.read_csv(external_upload, nrows=20)
                columns = list(raw.columns)
                m1, m2 = st.columns(2)
                with m1:
                    map_code = st.selectbox("行为编码字段", columns, key="map_code")
                    map_time = st.selectbox("时间字段", columns, key="map_time")
                    map_time_kind = st.selectbox(
                        "时间类型", ["start_hms", "start_ms"], key="map_time_kind"
                    )
                with m2:
                    map_window = st.selectbox(
                        "窗口编号字段（可选）", [""] + columns, key="map_window"
                    )
                    map_sequence = st.selectbox(
                        "独立观察片段字段（可选）", [""] + columns, key="map_sequence"
                    )
                    map_duration = st.selectbox(
                        "窗口长度字段（可选）", [""] + columns, key="map_duration"
                    )
                e1, e2 = st.columns(2)
                with e1:
                    external_class_id = st.text_input(
                        "课堂编号", value=Path(external_upload.name).stem, key="external_class_id"
                    )
                    external_scheme = st.text_input(
                        "编码体系版本", value="external_scheme", key="external_scheme"
                    )
                with e2:
                    external_window = st.number_input(
                        "窗口长度（秒）", min_value=1.0, max_value=60.0, value=5.0, step=1.0,
                        key="external_window",
                    )
                    external_target = st.selectbox(
                        "目标研究项目",
                        ["新建研究项目"] + list(existing_targets),
                        key="external_target_study",
                    )
                if not map_sequence:
                    external_confirm = st.checkbox(
                        "确认该文件是一段连续、完整的观察记录", value=False, key="external_confirm"
                    )
                else:
                    external_confirm = True
                if st.button("按映射导入外部 CSV", key="external_mapping_import"):
                    if not external_confirm:
                        st.error("没有片段编号时必须确认文件是连续完整序列。")
                    else:
                        mapping = {
                            "code": map_code,
                            "time": map_time,
                            "time_kind": map_time_kind,
                            "window_id": map_window,
                            "sequence_id": map_sequence,
                            "duration": map_duration,
                        }
                        if external_target == "新建研究项目":
                            study_dir = STUDIES_DIR / "external_study"
                            if (study_dir / "study_manifest.json").exists():
                                raise ValueError("external_study 已存在，请选择目标研究项目进行追加。")
                            study_dir.mkdir(parents=True, exist_ok=True)
                            manifest_path = study_dir / "study_manifest.json"
                            manifest = {
                                "study_id": "external_study",
                                "teacher_id": "T01",
                                "coding_scheme_id": external_scheme,
                                "window_duration_seconds": float(external_window),
                                "lag": 1,
                                "classes": [],
                            }
                            multi.save_manifest(manifest_path, manifest)
                        else:
                            manifest_path = existing_targets[external_target]
                            study_dir = manifest_path.parent
                            manifest = multi.load_manifest(manifest_path)
                        target = study_dir / "classes" / external_class_id / "lesson_external_codes.csv"
                        try:
                            multi.normalize_external_csv(external_upload, mapping, target)
                            entry = {
                                "class_id": external_class_id,
                                "display_name": external_class_id,
                                "teacher_id": manifest.get("teacher_id", ""),
                                "source_video": "",
                                "coding_scheme_id": external_scheme,
                                "window_duration_seconds": float(external_window),
                                "review_status": "external",
                                "file": str(target.relative_to(study_dir)),
                                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                                "human_file": "",
                                "code_source": "ai",
                                "sequence_id_column": "sequence_id",
                                "single_sequence": not bool(map_sequence),
                                "source_run_dir": "",
                                "source_final_dir": "",
                                "source_audio": "",
                                "audio_sha256": "",
                                "behavior_sequence_path": "",
                            }
                            if not manifest.get("classes"):
                                manifest["classes"] = [entry]
                                multi.save_manifest(manifest_path, manifest)
                            else:
                                multi.append_classes(study_dir, manifest_path, [entry])
                        except Exception as exc:
                            st.error(str(exc))
                        else:
                            st.success("外部 CSV 已导入研究项目。")
                            st.rerun()

        st.subheader("已导入研究项目")
        manifests = sorted(STUDIES_DIR.glob("*/study_manifest.json"))
        if not manifests:
            st.info("尚无研究项目。请先导入课堂 CSV。")
            return
        options = {str(path.relative_to(STUDIES_DIR)): path for path in manifests}
        selected_label = st.selectbox("选择研究项目", list(options), key="multi_study_select")
        manifest_path = options[selected_label]
        study_dir = manifest_path.parent
        manifest = multi.load_manifest(manifest_path)
        st.caption(
            f"study_id：`{manifest.get('study_id')}`；teacher_id：`{manifest.get('teacher_id')}`；"
            f"编码体系：`{manifest.get('coding_scheme_id')}`；窗口：{manifest.get('window_duration_seconds')} 秒"
        )
        st.dataframe(
            pd.DataFrame(manifest.get("classes", [])),
            hide_index=True,
            width="stretch",
        )
        class_ids = [str(entry.get("class_id")) for entry in manifest.get("classes", [])]
        with st.expander("维护已有研究项目", expanded=False):
            remove_ids = st.multiselect("要移除的课堂", class_ids, key="remove_study_classes")
            if st.button("从研究项目移除所选课堂", key="remove_study_classes_button"):
                try:
                    multi.remove_classes(study_dir, manifest_path, remove_ids)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.success("已移除课堂。原始数据未删除。")
                    st.rerun()
            display_id = st.selectbox("修改显示名称的课堂", class_ids, key="rename_study_class")
            display_name = st.text_input("课堂显示名称", value=display_id, key="study_class_display_name")
            if st.button("保存课堂显示名称", key="rename_study_class_button"):
                try:
                    multi.update_display_name(manifest_path, display_id, display_name)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.success("显示名称已更新。")
                    st.rerun()
            view_id = st.selectbox("查看单课堂分析", class_ids, key="view_study_class")
            if st.button("打开来源课堂的单课堂分析", key="view_study_class_button"):
                entry = next(
                    (item for item in manifest.get("classes", []) if item.get("class_id") == view_id),
                    None,
                )
                if not entry or not entry.get("source_run_dir"):
                    st.info("该课堂来自外部 CSV，没有可跳转的来源处理 run。")
                else:
                    run_dir = Path(entry["source_run_dir"])
                    final_dir = Path(entry.get("source_final_dir", ""))
                    if run_dir.exists() and final_dir.exists():
                        st.session_state["analysis_mode"] = "单课堂分析"
                        st.session_state["run_dir"] = str(run_dir)
                        st.session_state["final_dir"] = str(final_dir)
                        st.session_state["audio_path"] = entry.get("source_audio", "")
                        settings_path = run_dir / "settings.json"
                        if settings_path.exists():
                            st.session_state["settings"] = json.loads(
                                settings_path.read_text(encoding="utf-8")
                            )
                        st.rerun()
                    else:
                        st.warning("来源 run 或单课堂结果目录不存在。")

        errors = multi.validate_manifest(study_dir, manifest)
        if errors:
            st.error("数据质量检查未通过：")
            for error in errors:
                st.write(f"- {error}")
        else:
            st.success("数据质量检查通过。")
            if st.button("运行多课堂联合分析", type="primary"):
                command = [
                    PYTHON,
                    script("multi_classroom.py"),
                    manifest_path,
                    "--output-root",
                    RESULTS_DIR / "studies",
                    "--lag",
                    str(int(manifest.get("lag", 1))),
                ]
                with st.status("正在计算多课堂联合矩阵…", expanded=True) as status:
                    log_box = st.empty()
                    try:
                        run_command(command, log_box)
                    except Exception as exc:
                        status.update(label="多课堂分析失败", state="error")
                        st.exception(exc)
                        return
                    status.update(label="多课堂分析完成", state="complete")
                run_root = RESULTS_DIR / "studies" / str(manifest["study_id"]) / "lsa_v03"
                runs = sorted([path for path in run_root.iterdir() if path.is_dir()])
                if runs:
                    st.session_state["multi_run_dir"] = str(runs[-1])
                st.rerun()

    run_dirs = []
    selected_manifest = None
    manifests = sorted(STUDIES_DIR.glob("*/study_manifest.json"))
    if manifests:
        options = {str(path.relative_to(STUDIES_DIR)): path for path in manifests}
        selected_label = st.session_state.get("multi_study_select")
        selected_manifest = options.get(selected_label) if selected_label else None
    if selected_manifest:
        run_root = RESULTS_DIR / "studies" / selected_manifest.parent.name / "lsa_v03"
        if run_root.exists():
            run_dirs = sorted([path for path in run_root.iterdir() if path.is_dir()])
    if st.session_state.get("multi_run_dir"):
        current_run = Path(st.session_state["multi_run_dir"])
    elif run_dirs:
        current_run = run_dirs[-1]
    else:
        current_run = None

    if current_run and (current_run / "analysis_metadata.json").exists():
        metadata = json.loads((current_run / "analysis_metadata.json").read_text(encoding="utf-8"))
        classroom_summary = pd.read_csv(current_run / "classroom_summary.csv")
        frequency = pd.read_csv(current_run / "pooled_frequency.csv")
        pooled_counts = pd.read_csv(current_run / "pooled_transition_counts.csv", index_col=0)
        pooled_probability = pd.read_csv(current_run / "pooled_transition_probabilities.csv", index_col=0)
        equal_probability = pd.read_csv(current_run / "equal_weight_transition_probabilities.csv", index_col=0)
        stability = pd.read_csv(current_run / "transition_stability.csv")
        comparison = pd.read_csv(current_run / "classroom_transition_comparison.csv")
        loco = pd.read_csv(current_run / "leave_one_class_out.csv")
        with overview_tab:
            st.caption(f"结果目录：`{current_run}`")
            one, two, three, four = st.columns(4)
            one.metric("课堂数", metadata.get("class_count", "-"))
            two.metric("独立序列", metadata.get("sequence_count", "-"))
            three.metric("观察窗口", metadata.get("total_windows", "-"))
            four.metric("有效转移", metadata.get("total_transitions", "-"))
            st.markdown("**课堂汇总**")
            st.dataframe(classroom_summary, hide_index=True, width="stretch")
            st.markdown("**联合行为频次**")
            st.bar_chart(frequency.set_index("code_name")["frequency"])
            st.dataframe(frequency, hide_index=True, width="stretch")
            graph_png = current_run / "pooled_transition_graph.png"
            if graph_png.exists():
                st.image(graph_png, caption="多课堂联合转移图（描述性）")
            with st.expander("联合转移频数与概率", expanded=False):
                st.dataframe(pooled_counts, width="stretch")
                st.dataframe(pooled_probability.round(4), width="stretch")
                st.dataframe(equal_probability.round(4), width="stretch")
        with compare_tab:
            st.caption("不同课堂长度不同，不能仅凭原始转移频数判断课堂差异。")
            if not comparison.empty:
                f1, f2 = st.columns(2)
                from_code = f1.selectbox("起始编码", sorted(comparison["from_code"].unique()))
                targets = sorted(comparison.loc[comparison["from_code"] == from_code, "to_code"].unique())
                to_code = f2.selectbox("目标编码", targets)
                selected = comparison[
                    (comparison["from_code"] == from_code) & (comparison["to_code"] == to_code)
                ]
                st.dataframe(selected, hide_index=True, width="stretch")
                pivot = selected.pivot_table(
                    index="class_id", values="probability", aggfunc="first"
                )
                st.bar_chart(pivot)
            st.dataframe(stability, hide_index=True, width="stretch", height=360)
        with stability_tab:
            st.caption("描述性稳定性，不是跨课堂显著性检验。")
            st.dataframe(stability, hide_index=True, width="stretch", height=420)
            with st.expander("Leave-One-Class-Out 明细", expanded=False):
                st.dataframe(loco, hide_index=True, width="stretch", height=360)
        with export_tab:
            report_path = current_run / "multi_classroom_report.html"
            for label, path, mime in [
                ("下载多课堂 HTML 报告", report_path, "text/html"),
                ("下载联合转移频数 CSV", current_run / "pooled_transition_counts.csv", "text/csv"),
                ("下载联合转移概率 CSV", current_run / "pooled_transition_probabilities.csv", "text/csv"),
                ("下载等权转移概率 CSV", current_run / "equal_weight_transition_probabilities.csv", "text/csv"),
                ("下载稳定性 CSV", current_run / "transition_stability.csv", "text/csv"),
                ("下载 LOCO CSV", current_run / "leave_one_class_out.csv", "text/csv"),
            ]:
                if path.exists():
                    st.download_button(
                        label,
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime=mime,
                        key=f"multi_download_{path.name}",
                    )
            st.caption(f"元数据：`{current_run / 'analysis_metadata.json'}`")
    else:
        with overview_tab:
            st.info("请选择研究项目并运行联合分析。")
        with compare_tab:
            st.info("尚无联合分析结果。")
        with stability_tab:
            st.info("尚无稳定性结果。")
        with export_tab:
            st.info("尚无导出结果。")


st.title("🎙️ 课堂语音研究助手")
st.caption(
    "本地 FunASR（默认）· 可选讯飞云端高精度 · 教师/学生组 · 可配置时间窗口 · AI 行为编码"
)

analysis_mode = st.sidebar.radio(
    "分析模式", ["单课堂分析", "多课堂分析"], index=0, key="analysis_mode"
)
if analysis_mode == "多课堂分析":
    st.sidebar.caption("多课堂联合结果为描述性汇总，不提供课堂层级显著性推断。")
    render_multi_classroom_page()
    st.stop()

sidebar_run_selector()

(
    start_tab,
    verbatim_tab,
    segments_tab,
    intra_tab,
    chorus_tab,
    manual_chorus_tab,
    ai_code_tab,
    human_review_tab,
    behavior_seq_tab,
    coding_setup_tab,
    stats_tab,
    export_tab,
) = st.tabs(
    [
        "开始转录",
        "逐字稿",
        "片段与窗口",
        "片段内部候选",
        "齐读检测",
        "手动齐读",
        "AI编码",
        "人工复核",
        "行为序列",
        "编码设置",
        "统计",
        "导出",
    ]
)

with start_tab:
    render_start_tab()
with verbatim_tab:
    render_verbatim_tab()
with segments_tab:
    render_segments_and_windows_tab()
with intra_tab:
    render_intra_tab()
with chorus_tab:
    render_chorus_tab()
with manual_chorus_tab:
    render_manual_chorus_tab()
with ai_code_tab:
    render_ai_code_tab()
with human_review_tab:
    render_human_review_tab()
with behavior_seq_tab:
    render_behavior_sequence_tab()
with coding_setup_tab:
    render_coding_setup_tab()
with stats_tab:
    render_stats_tab()
with export_tab:
    render_export_tab()
