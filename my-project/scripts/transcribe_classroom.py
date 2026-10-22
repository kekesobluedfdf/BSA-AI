from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "default.yaml"

os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT_DIR / "models" / ".cache"))
os.environ.setdefault("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import torch
from funasr import AutoModel

from transcript_exports import export_all

MODEL_IDS = {
    "SenseVoiceSmall": "iic/SenseVoiceSmall",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
    "ct-punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma"}
CSV_COLUMNS = [
    "recording_id",
    "segment_id",
    "start_ms",
    "end_ms",
    "start_hms",
    "end_hms",
    "duration_s",
    "speaker_id",
    "role",
    "text",
    "raw_text",
]

TAG_RE = re.compile(r"<\|[^|]*\|>")
SPACE_RE = re.compile(r"[ \t\u3000]+")
LEADING_PUNCT_RE = re.compile(r"^[\s，。、；：！？,.;:!?]+")
HAS_CONTENT_RE = re.compile(r"[\w\u4e00-\u9fff]")

LOGGER = logging.getLogger("classroom_asr")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def resolve_path(value: str | None, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def configure_logging(output_dir: Path, level: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"transcribe_{datetime.now():%Y%m%d_%H%M%S}.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return log_path


def resolve_model(model_root: Path, name: str) -> str:
    model_dir = model_root / name
    if model_dir.is_dir() and (model_dir / "config.yaml").exists():
        return str(model_dir)
    LOGGER.warning("Local model not found at %s; falling back to ModelScope ID %s", model_dir, MODEL_IDS[name])
    return MODEL_IDS[name]


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


def build_model(args: argparse.Namespace, device: str) -> tuple[AutoModel, str, dict[str, str]]:
    effective_mode = "vad_segment" if (args.no_punc or args.spk_mode == "vad_segment") else "punc_segment"

    model_paths = {
        "asr": resolve_model(args.model_root, "SenseVoiceSmall"),
        "vad": resolve_model(args.model_root, "fsmn-vad"),
        "spk": resolve_model(args.model_root, "campplus"),
    }
    punc_model = None
    if effective_mode == "punc_segment":
        punc_model = resolve_model(args.model_root, "ct-punc")
        model_paths["punc"] = punc_model
    else:
        LOGGER.warning(
            "ct-punc disabled: speaker labels use coarse VAD segments and may merge multiple speakers."
        )

    LOGGER.info("Loading models on %s (spk_mode=%s)", device, effective_mode)
    model = AutoModel(
        model=model_paths["asr"],
        vad_model=model_paths["vad"],
        punc_model=punc_model,
        spk_model=model_paths["spk"],
        spk_mode="punc_segment",
        vad_kwargs={"max_single_segment_time": args.max_segment_ms},
        device=device,
        ncpu=args.ncpu,
        disable_update=True,
    )
    return model, effective_mode, model_paths


def collect_audio_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Audio path does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No supported audio files found in: {input_path}")
    return files


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", value, flags=re.UNICODE).strip("_")
    return cleaned or "recording"


def ms_to_hms(value: int) -> str:
    value = max(0, int(round(value)))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def clean_text(raw: str) -> str:
    text = TAG_RE.sub("", raw)
    text = text.replace("\u00a0", " ")
    text = SPACE_RE.sub(" ", text)
    for punct in "，。、；：！？,.!?;:":
        text = re.sub(rf"{re.escape(punct)}{{2,}}", punct, text)
    text = re.sub(r"[，,]\s*[。.]", "。", text)
    text = LEADING_PUNCT_RE.sub("", text)
    return text.strip()


def as_ms(value: Any) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def extract_records(recording_id: str, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    speaker_map: dict[str, str] = {}
    records: list[dict[str, Any]] = []

    for segment in segments:
        raw_text = str(segment.get("text") or segment.get("sentence") or "")
        text = clean_text(raw_text)
        if not text or not HAS_CONTENT_RE.search(text):
            continue

        start_ms = as_ms(segment.get("start"))
        end_ms = as_ms(segment.get("end"))
        if end_ms <= start_ms:
            continue

        raw_speaker = segment.get("spk")
        speaker_key = "UNKNOWN" if raw_speaker is None else str(raw_speaker)
        if speaker_key not in speaker_map:
            speaker_map[speaker_key] = f"S{len(speaker_map) + 1:02d}"

        row = {
            "recording_id": recording_id,
            "segment_id": len(records) + 1,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_hms": ms_to_hms(start_ms),
            "end_hms": ms_to_hms(end_ms),
            "duration_s": round((end_ms - start_ms) / 1000, 3),
            "speaker_id": speaker_map[speaker_key],
            "text": text,
            "raw_text": raw_text,
        }
        records.append(
            {
                "row": row,
                "speaker_raw": raw_speaker,
                "segment_timestamp": segment.get("timestamp"),
            }
        )

    return records


def write_outputs(
    output_dir: Path,
    recording_id: str,
    records: list[dict[str, Any]],
    run_info: dict[str, Any],
) -> tuple[Path, Path, Path]:
    rows = [record["row"] for record in records]

    csv_path = output_dir / f"{recording_id}_segments.csv"
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False, encoding="utf-8-sig")

    jsonl_path = output_dir / f"{recording_id}_segments.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = dict(record["row"])
            payload["speaker_raw"] = record["speaker_raw"]
            payload["segment_timestamp"] = record["segment_timestamp"]
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    run_path = output_dir / f"{recording_id}_run.json"
    with run_path.open("w", encoding="utf-8") as handle:
        json.dump(run_info, handle, ensure_ascii=False, indent=2, default=str)

    return csv_path, jsonl_path, run_path


def audio_duration_s(path: Path) -> float | None:
    try:
        import soundfile as sf

        return round(float(sf.info(str(path)).duration), 3)
    except Exception:
        return None


def process_file(
    model: AutoModel,
    audio_path: Path,
    recording_id: str,
    args: argparse.Namespace,
    device: str,
    spk_mode: str,
    model_paths: dict[str, str],
) -> dict[str, Any]:
    LOGGER.info("Transcribing %s", audio_path)
    started = time.perf_counter()
    generate_kwargs: dict[str, Any] = {
        "input": str(audio_path),
        "cache": {},
        "language": args.language,
        "use_itn": not args.no_itn,
        "batch_size_s": args.batch_size_s,
        "merge_vad": args.merge_vad,
        "merge_length_s": args.merge_length_s,
        "return_spk_res": True,
        "disable_pbar": not args.progress,
    }
    if args.preset_spk_num:
        generate_kwargs["preset_spk_num"] = args.preset_spk_num

    result = model.generate(**generate_kwargs)
    elapsed = time.perf_counter() - started

    segments = result[0].get("sentence_info", []) if result else []
    records = extract_records(recording_id, segments)
    rows = [record["row"] for record in records]
    duration_s = audio_duration_s(audio_path)
    rtf = round(elapsed / duration_s, 4) if duration_s else None

    speaker_totals: dict[str, float] = {}
    for row in rows:
        speaker_totals[row["speaker_id"]] = speaker_totals.get(row["speaker_id"], 0.0) + row["duration_s"]
    teacher_id = args.teacher_id or "auto"
    if teacher_id == "auto":
        teacher_id = max(speaker_totals, key=speaker_totals.get) if speaker_totals else "S01"
    elif teacher_id not in speaker_totals:
        LOGGER.warning("Teacher speaker %s not found; using the speaker with the most speech.", teacher_id)
        teacher_id = max(speaker_totals, key=speaker_totals.get) if speaker_totals else "S01"

    for record in records:
        role = "teacher" if record["row"]["speaker_id"] == teacher_id else "student"
        record["row"]["role"] = role
        record["role"] = role

    import funasr
    import modelscope

    run_info = {
        "recording_id": recording_id,
        "audio_path": str(audio_path.resolve()),
        "audio_size_bytes": audio_path.stat().st_size,
        "audio_duration_s": duration_s,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed, 3),
        "rtf": rtf,
        "device": device,
        "language": args.language,
        "spk_mode": spk_mode,
        "preset_spk_num": args.preset_spk_num,
        "auto_speaker_count": args.preset_spk_num is None,
        "teacher_id": teacher_id,
        "window_seconds": args.window_seconds,
        "window_mode": args.window_mode,
        "stats_window_min": args.stats_window_min,
        "merge_vad": args.merge_vad,
        "merge_length_s": args.merge_length_s,
        "max_segment_ms": args.max_segment_ms,
        "batch_size_s": args.batch_size_s,
        "model_paths": model_paths,
        "versions": {
            "funasr": funasr.__version__,
            "modelscope": modelscope.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "segment_count": len(rows),
        "speakers": sorted({row["speaker_id"] for row in rows}),
    }

    csv_path, jsonl_path, run_path = write_outputs(args.output_dir, recording_id, records, run_info)
    try:
        exports = export_all(
            jsonl_path,
            args.output_dir,
            args.window_seconds,
            args.stats_window_min,
            teacher_id,
            args.window_mode,
        )
    except Exception as exc:
        LOGGER.exception("Transcription succeeded but transcript export failed: %s", exc)
        exports = {}
    LOGGER.info(
        "Finished %s: %d segments, %d speakers, %.2fs",
        recording_id,
        len(rows),
        len(run_info["speakers"]),
        elapsed,
    )
    return {
        "recording_id": recording_id,
        "audio_path": str(audio_path),
        "audio_duration_s": duration_s,
        "segments": len(rows),
        "speakers": run_info["speakers"],
        "teacher_id": teacher_id,
        "elapsed_s": round(elapsed, 2),
        "rtf": rtf,
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
        "run": str(run_path),
        "verbatim": exports.get("verbatim", ""),
        "window_csv": exports.get("window_csv", ""),
        "window_txt": exports.get("window_txt", ""),
        "stats": exports.get("stats", ""),
        "speaker_summary": exports.get("speaker_summary", ""),
        "segments_with_role": exports.get("segments_with_role", ""),
    }


def build_parser() -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(Path(pre_args.config))

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Transcribe classroom audio with SenseVoice, FSMN-VAD, and CAM++ speaker clustering.",
    )
    parser.add_argument("audio", help="Audio file or directory.")
    parser.add_argument("--model-root", default=str(config.get("model_root", "models")))
    parser.add_argument("--output-dir", default=str(config.get("output_dir", "results")))
    parser.add_argument("--device", default=str(config.get("device", "auto")))
    parser.add_argument("--language", default=str(config.get("language", "zh")))
    parser.add_argument(
        "--spk-mode",
        choices=["punc_segment", "vad_segment"],
        default=str(config.get("spk_mode", "punc_segment")),
    )
    parser.add_argument("--preset-spk-num", type=int, default=config.get("preset_spk_num"))
    parser.add_argument("--teacher-id", default=str(config.get("teacher_id", "S01")))
    parser.add_argument("--window-seconds", type=float, default=float(config.get("window_seconds", 3)))
    parser.add_argument(
        "--window-mode",
        choices=["merge", "fixed"],
        default=str(config.get("window_mode", "fixed")),
    )
    parser.add_argument(
        "--stats-window-min", type=float, default=float(config.get("stats_window_min", 5))
    )
    parser.add_argument("--batch-size-s", type=int, default=int(config.get("batch_size_s", 60)))
    parser.add_argument("--max-segment-ms", type=int, default=int(config.get("max_segment_ms", 30000)))
    parser.add_argument(
        "--merge-vad",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("merge_vad", False)),
    )
    parser.add_argument("--merge-length-s", type=int, default=int(config.get("merge_length_s", 15)))
    parser.add_argument("--ncpu", type=int, default=int(config.get("ncpu", 4)))
    parser.add_argument("--no-punc", action="store_true", help="Skip ct-punc and force vad_segment mode.")
    parser.add_argument("--no-itn", action="store_true", help="Disable inverse text normalization.")
    parser.add_argument("--recursive", action="store_true", help="Search audio files recursively.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files.")
    parser.add_argument("--progress", action="store_true", help="Show FunASR progress bars.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.model_root = resolve_path(args.model_root, ROOT_DIR / "models")
    args.output_dir = resolve_path(args.output_dir, ROOT_DIR / "results")

    log_path = configure_logging(args.output_dir, args.log_level)
    LOGGER.info("Log file: %s", log_path)

    device = resolve_device(args.device)
    audio_files = collect_audio_files(Path(args.audio), args.recursive)
    if args.limit:
        audio_files = audio_files[: args.limit]

    LOGGER.info("Found %d audio file(s)", len(audio_files))
    model, spk_mode, model_paths = build_model(args, device)

    used_ids: set[str] = set()
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, audio_path in enumerate(audio_files, start=1):
        base_id = sanitize_id(audio_path.stem)
        recording_id = base_id
        suffix = 2
        while recording_id in used_ids:
            recording_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(recording_id)

        LOGGER.info("[%d/%d] %s", index, len(audio_files), audio_path)
        try:
            summary = process_file(model, audio_path, recording_id, args, device, spk_mode, model_paths)
            summaries.append(summary)
        except Exception as exc:
            LOGGER.exception("Failed to transcribe %s: %s", audio_path, exc)
            failures.append({"audio_path": str(audio_path), "error": str(exc)})

    print("\nSummary")
    for summary in summaries:
        print(
            f"  {summary['recording_id']}: {summary['segments']} segments, "
            f"speakers={','.join(summary['speakers']) or '-'}, "
            f"audio={summary['audio_duration_s']}s, elapsed={summary['elapsed_s']}s, rtf={summary['rtf']}"
        )
        print(f"    CSV: {summary['csv']}")
        print(f"    Verbatim: {summary['verbatim']}")
        print(f"    Window: {summary['window_csv']}")
        print(f"    Stats: {summary['stats']}")
    if failures:
        print("\nFailures")
        for failure in failures:
            print(f"  {failure['audio_path']}: {failure['error']}")

    if summaries:
        combined_path = args.output_dir / "batch_summary.json"
        with combined_path.open("w", encoding="utf-8") as handle:
            json.dump({"results": summaries, "failures": failures}, handle, ensure_ascii=False, indent=2)
        print(f"\nBatch summary: {combined_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
