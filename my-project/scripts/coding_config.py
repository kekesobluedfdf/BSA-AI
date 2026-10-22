from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from docx_text import extract_docx_text

PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
CONFIG_DIR = PROJECT_DIR / "configs" / "coding"
ACTIVE_PATH = CONFIG_DIR / "active.json"
DEFAULT_CODING_SYSTEM = ROOT_DIR / "coding item" / "小学习作课程编码系统.docx"
DEFAULT_PROMPT = ROOT_DIR / "coding item" / "AI课堂行为编码+prompt.docx"
SUPPORTED_SUFFIXES = {".docx", ".txt", ".md"}


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return extract_docx_text(path)
    return path.read_text(encoding="utf-8-sig")


def active_paths() -> tuple[Path, Path]:
    coding_path = DEFAULT_CODING_SYSTEM
    prompt_path = DEFAULT_PROMPT
    if ACTIVE_PATH.exists():
        try:
            data = json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))
            candidate = Path(data.get("coding_system", "")).expanduser()
            if candidate.exists():
                coding_path = candidate
            candidate = Path(data.get("prompt", "")).expanduser()
            if candidate.exists():
                prompt_path = candidate
        except Exception:
            pass
    return coding_path, prompt_path


def save_source(data: bytes, original_name: str, kind: str) -> Path:
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的编码文件格式：{suffix}")
    if kind not in {"coding_system", "prompt"}:
        raise ValueError(f"未知文件类型：{kind}")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    target = CONFIG_DIR / f"{kind}{suffix}"
    target.write_bytes(data)
    return target


def save_active(coding_system: Path, prompt: Path) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_PATH.write_text(
        json.dumps(
            {
                "coding_system": str(coding_system),
                "prompt": str(prompt),
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def reset_active() -> None:
    ACTIVE_PATH.unlink(missing_ok=True)


def copy_sources(coding_system: Path, prompt: Path, output_dir: Path) -> dict[str, str]:
    target_dir = output_dir / "coding_sources"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for key, source in (("coding_system", coding_system), ("prompt", prompt)):
        suffix = source.suffix.lower()
        target = target_dir / f"{key}{suffix}"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied[key] = str(target)
    return copied
