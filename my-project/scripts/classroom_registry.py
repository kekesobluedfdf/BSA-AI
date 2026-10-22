from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_RUNS_DIR = ROOT_DIR / "results" / "ui_runs"
REGISTRY_PATH = ROOT_DIR / "studies" / "classroom_registry.json"
VALID_CODES = set("ABCDEFGHIJKLMNOP")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    if not path.exists():
        return {"version": 1, "classrooms": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "classrooms": {}}
    if not isinstance(data, dict):
        return {"version": 1, "classrooms": {}}
    data.setdefault("version", 1)
    data.setdefault("classrooms", {})
    return data


def choose_ai_codes(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.rglob("*_ai_codes.csv"))
    if not candidates:
        return None
    preference = ["03_iflytek_classroom", "03_iflytek", "10_phrase_split", "11_cue_scoped", "04_final"]
    for name in preference:
        for candidate in candidates:
            if name in candidate.parts:
                return candidate
    return candidates[0]


def human_status(ai_path: Path, human_path: Path | None) -> tuple[str, int, int]:
    ai = pd.read_csv(ai_path)
    total = len(ai)
    if not human_path or not human_path.exists():
        return "absent", 0, total
    human = pd.read_csv(human_path)
    if "window_id" not in human.columns or "human_code" not in human.columns:
        return "invalid", 0, total
    human = human.copy()
    human["human_code"] = human["human_code"].fillna("").astype(str).str.strip().str.upper()
    valid = human[human["human_code"].isin(VALID_CODES)]
    reviewed = len(valid)
    complete = reviewed >= total and set(ai["window_id"]).issubset(set(valid["window_id"]))
    return ("complete" if complete else "partial"), reviewed, total


def find_behavior_sequence(parent: Path) -> Path | None:
    for name in ("behavior_sequence_v02c", "behavior_sequence_v02b", "behavior_sequence_v02a", "behavior_sequence"):
        candidate = parent / name
        if (candidate / "lesson_behavior_sequence_metadata.json").exists() or (candidate / "flower_10min_behavior_sequence_metadata.json").exists():
            return candidate
    return None


def discover_runs(runs_dir: Path = DEFAULT_RUNS_DIR) -> list[dict]:
    records = []
    for run_dir in sorted([path for path in runs_dir.iterdir() if path.is_dir()], reverse=True):
        ai_path = choose_ai_codes(run_dir)
        settings = {}
        settings_path = run_dir / "settings.json"
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
            except Exception:
                settings = {}
        if ai_path is None:
            records.append(
                {
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "class_id": run_dir.name,
                    "display_name": run_dir.name,
                    "status": "missing_ai_codes",
                    "source_audio": "",
                    "audio_sha256": settings.get("audio_sha256", ""),
                    "source_video": "",
                    "ai_codes_path": "",
                    "human_codes_path": "",
                    "coding_scheme_id": "unknown",
                    "window_duration_seconds": settings.get("window_seconds"),
                    "review_status": "absent",
                    "reviewed_windows": 0,
                    "total_windows": 0,
                    "behavior_sequence_path": "",
                    "created_at": run_dir.name,
                }
            )
            continue
        meta_path = ai_path.parent / ai_path.name.replace("_ai_codes.csv", "_ai_code_metadata.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        human_path = ai_path.parent / ai_path.name.replace("_ai_codes.csv", "_human_codes.csv")
        review_status, reviewed, total = human_status(ai_path, human_path)
        try:
            ai_frame = pd.read_csv(ai_path)
            window_seconds = float(pd.to_numeric(ai_frame.get("duration_s"), errors="coerce").median())
        except Exception:
            window_seconds = settings.get("window_seconds")
        audio_candidates = [
            path for path in run_dir.iterdir() if path.is_file() and path.stem == "lesson"
        ]
        audio_path = str(audio_candidates[0]) if audio_candidates else ""
        behavior = find_behavior_sequence(ai_path.parent)
        records.append(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "class_id": run_dir.name,
                "display_name": settings.get("audio_filename") or run_dir.name,
                "status": "ready" if audio_path else "missing_audio",
                "source_audio": audio_path,
                "audio_sha256": settings.get("audio_sha256", ""),
                "source_video": "",
                "ai_codes_path": str(ai_path),
                "human_codes_path": str(human_path) if human_path.exists() else "",
                "coding_scheme_id": meta.get("coding_scheme_id", "unknown"),
                "window_duration_seconds": window_seconds,
                "review_status": review_status,
                "reviewed_windows": reviewed,
                "total_windows": total,
                "behavior_sequence_path": str(behavior) if behavior else "",
                "created_at": run_dir.name,
            }
        )
    return records


def scan_and_update(runs_dir: Path = DEFAULT_RUNS_DIR, registry_path: Path = REGISTRY_PATH) -> dict:
    registry = load_registry(registry_path)
    existing = registry.get("classrooms", {})
    discovered = discover_runs(runs_dir)
    updated = {}
    for record in discovered:
        previous = existing.get(record["run_id"], {})
        for key in ("class_id", "display_name", "source_video"):
            if previous.get(key):
                record[key] = previous[key]
        record["class_id"] = record.get("class_id") or record["run_id"]
        updated[record["run_id"]] = record
    registry = {
        "version": 1,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "classrooms": updated,
    }
    atomic_write_json(registry_path, registry)
    return registry


def update_classroom(registry_path: Path, run_id: str, updates: dict) -> dict:
    registry = load_registry(registry_path)
    if run_id not in registry["classrooms"]:
        raise KeyError(run_id)
    allowed = {"display_name", "class_id", "source_video"}
    for key, value in updates.items():
        if key in allowed:
            registry["classrooms"][run_id][key] = value
    atomic_write_json(registry_path, registry)
    return registry


def record_to_manifest_entry(record: dict, code_source: str = "ai") -> dict:
    ai_path = record.get("ai_codes_path", "")
    has_sequence = False
    if ai_path:
        try:
            has_sequence = "sequence_id" in pd.read_csv(ai_path, nrows=0).columns
        except Exception:
            has_sequence = False
    human_path = record.get("human_codes_path", "")
    return {
        "class_id": record["class_id"],
        "display_name": record.get("display_name") or record["class_id"],
        "teacher_id": record.get("teacher_id", ""),
        "source_video": record.get("source_video", ""),
        "coding_scheme_id": record.get("coding_scheme_id", "unknown"),
        "window_duration_seconds": record.get("window_duration_seconds"),
        "review_status": record.get("review_status", "absent"),
        "file": ai_path,
        "sha256": sha256_file(Path(ai_path)) if ai_path else "",
        "human_file": human_path,
        "human_sha256": sha256_file(Path(human_path)) if human_path and Path(human_path).exists() else "",
        "code_source": code_source,
        "sequence_id_column": "sequence_id",
        "single_sequence": not has_sequence,
        "source_run_dir": record.get("run_dir", ""),
        "source_final_dir": str(Path(ai_path).parent) if ai_path else "",
        "source_audio": record.get("source_audio", ""),
        "audio_sha256": record.get("audio_sha256", ""),
        "behavior_sequence_path": record.get("behavior_sequence_path", ""),
    }
