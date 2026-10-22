from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_DIR / "configs" / "coding"
ACTIVE_PATH = CONFIG_DIR / "dictionary_active.json"


def load_dictionary(path: Path | None) -> dict[str, dict]:
    if not path or not Path(path).exists():
        return {}
    source = Path(path)
    if source.suffix.lower() == ".json":
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        records = data if isinstance(data, list) else []
        if isinstance(data, dict):
            records = []
            for code, value in data.items():
                if isinstance(value, dict):
                    records.append({"code": code, **value})
                else:
                    records.append({"code": code, "label": str(value)})
    elif source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            records = list(csv.DictReader(handle))
    else:
        raise ValueError("编码字典只支持 CSV 或 JSON")

    result = {}
    for record in records:
        code = str(record.get("code", "")).strip().upper()
        if not code:
            continue
        result[code] = {
            "code": code,
            "label": str(record.get("label", "") or "").strip(),
            "definition": str(record.get("definition", "") or "").strip(),
            "category": str(record.get("category", "") or "").strip(),
        }
    return result


def label_for(code: str, dictionary: dict[str, dict] | None, mode: str = "both") -> str:
    code = str(code).upper()
    entry = (dictionary or {}).get(code, {})
    label = str(entry.get("label", "") or "").strip()
    if mode == "code" or not label:
        return code
    if mode == "label":
        return label
    return f"{code} {label}"


def save_dictionary(data: bytes, original_name: str) -> Path:
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".csv", ".json"}:
        raise ValueError("编码字典只支持 CSV 或 JSON")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    target = CONFIG_DIR / f"dictionary{suffix}"
    target.write_bytes(data)
    ACTIVE_PATH.write_text(
        json.dumps({"dictionary": str(target)}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def active_dictionary_path() -> Path | None:
    if not ACTIVE_PATH.exists():
        return None
    try:
        data = json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))
        path = Path(data.get("dictionary", ""))
        return path if path.exists() else None
    except Exception:
        return None


def reset_dictionary() -> None:
    ACTIVE_PATH.unlink(missing_ok=True)
