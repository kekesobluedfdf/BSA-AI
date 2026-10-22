from __future__ import annotations

import argparse
import bisect
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from funasr import AutoModel

from transcript_exports import export_all

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_MODEL_ROOT = ROOT_DIR / "models"

os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT_DIR / "models" / ".cache"))
os.environ.setdefault("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn")

PUNCTUATION = set("，。！？；：,.!?;:")


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assign_token(tokens: list[dict], rows: list[dict]) -> list[list[str]]:
    starts = [int(row["start_ms"]) for row in rows]
    parts: list[list[str]] = [[] for _ in rows]
    last_target = None
    for token in tokens:
        text = str(token.get("token", ""))
        if not text:
            continue
        if last_target is not None and all(character in PUNCTUATION for character in text):
            parts[last_target].append(text)
            continue

        start = float(token.get("start_time", 0)) * 1000
        end = float(token.get("end_time", 0)) * 1000
        middle = (start + end) / 2
        index = bisect.bisect_right(starts, middle) - 1
        candidates = []
        for position in (index - 1, index, index + 1):
            if 0 <= position < len(rows):
                row = rows[position]
                if float(row["start_ms"]) <= middle <= float(row["end_ms"]):
                    candidates.append(position)
        if candidates:
            target = min(
                candidates,
                key=lambda position: abs(
                    middle
                    - (float(rows[position]["start_ms"]) + float(rows[position]["end_ms"])) / 2
                ),
            )
        else:
            target = min(
                range(len(rows)),
                key=lambda position: min(
                    abs(middle - float(rows[position]["start_ms"])),
                    abs(middle - float(rows[position]["end_ms"])),
                ),
            )
        parts[target].append(text)
        last_target = target
    return parts


def run_chunked(
    model: AutoModel,
    audio_path: Path,
    output_dir: Path,
    chunk_seconds: float,
    overlap_seconds: float,
    batch_size_s: int,
    hotwords: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    if chunk_seconds <= 0:
        chunk_samples = len(audio)
    else:
        chunk_samples = int(chunk_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    step = max(1, chunk_samples - overlap_samples)

    tokens = []
    chunk_texts = []
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        temporary_path = Path(temporary)
        start = 0
        index = 0
        while start < len(audio):
            end = min(len(audio), start + chunk_samples)
            if end - start < int(0.5 * sample_rate):
                break
            chunk_path = temporary_path / f"chunk_{index:04d}.wav"
            sf.write(str(chunk_path), audio[start:end], sample_rate)
            generate_kwargs = {
                "input": str(chunk_path),
                "batch_size_s": batch_size_s,
                "disable_pbar": True,
            }
            if hotwords:
                generate_kwargs["hotwords"] = hotwords
            result = model.generate(**generate_kwargs)
            chunk_texts.append(str(result[0].get("text", "")))
            offset = start / sample_rate
            owned_start = offset + (overlap_seconds / 2 if index > 0 else 0)
            owned_end = end / sample_rate - (overlap_seconds / 2 if end < len(audio) else 0)
            for token in result[0].get("timestamps") or []:
                token_start = float(token.get("start_time", 0)) + offset
                token_end = float(token.get("end_time", 0)) + offset
                middle = (token_start + token_end) / 2
                if owned_start <= middle < owned_end or (index == 0 and middle < owned_end):
                    item = dict(token)
                    item["start_time"] = token_start
                    item["end_time"] = token_end
                    tokens.append(item)
            if end >= len(audio):
                break
            start += step
            index += 1

    tokens.sort(key=lambda token: float(token.get("start_time", 0)))
    return tokens, chunk_texts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Path to SenseVoice/refined *_segments.jsonl")
    parser.add_argument("audio", help="Source audio file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size-s", type=int, default=60)
    parser.add_argument("--chunk-seconds", type=float, default=60)
    parser.add_argument("--overlap-seconds", type=float, default=0.5)
    parser.add_argument("--hotwords", default="")
    parser.add_argument("--hotwords-file", default=None)
    parser.add_argument("--teacher-id", default="S01")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--stats-window-min", type=float, default=5.0)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    audio_path = Path(args.audio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = jsonl_path.stem.removesuffix("_segments")

    rows = load_rows(jsonl_path)
    hotwords = [
        item.strip()
        for item in args.hotwords.replace("，", ",").split(",")
        if item.strip()
    ]
    if args.hotwords_file:
        file_text = Path(args.hotwords_file).read_text(encoding="utf-8")
        hotwords.extend(
            item.strip()
            for item in file_text.replace("，", ",").replace("\n", ",").split(",")
            if item.strip()
        )
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModel(
        model=str(Path(args.model_root) / "Fun-ASR-Nano"),
        vad_model=str(Path(args.model_root) / "fsmn-vad"),
        device=device,
        disable_update=True,
    )
    tokens, chunk_texts = run_chunked(
        model,
        audio_path,
        output_dir,
        args.chunk_seconds,
        args.overlap_seconds,
        args.batch_size_s,
        hotwords,
    )
    with (output_dir / f"{prefix}_nano_result.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"tokens": tokens, "chunk_texts": chunk_texts},
            handle,
            ensure_ascii=False,
            default=str,
            indent=2,
        )

    if not tokens:
        raise RuntimeError("Fun-ASR-Nano returned no token timestamps")

    parts = assign_token(tokens, rows)
    assigned = 0
    for row, token_parts in zip(rows, parts):
        row["sensevoice_text"] = str(row.get("text") or "")
        nano_text = "".join(token_parts).strip()
        if nano_text:
            row["text"] = nano_text
            row["text_source"] = "fun-asr-nano"
            assigned += 1
        else:
            row["text_source"] = "sensevoice"
        row["nano_text"] = nano_text

    hybrid_jsonl = output_dir / f"{prefix}_segments.jsonl"
    with hybrid_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    columns = [
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
        "sensevoice_text",
        "nano_text",
        "text_source",
        "raw_text",
    ]
    for row in rows:
        if str(row["speaker_id"]).upper() == "CHORUS":
            row["role"] = "chorus"
        elif str(row["speaker_id"]) == args.teacher_id:
            row["role"] = "teacher"
        else:
            row["role"] = "student"
    pd.DataFrame([{column: row.get(column, "") for column in columns} for row in rows]).to_csv(
        output_dir / f"{prefix}_segments.csv", index=False, encoding="utf-8-sig"
    )

    metadata_path = output_dir / f"{prefix}_hybrid_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_jsonl": str(jsonl_path),
                "audio": str(audio_path),
                "asr_model": "FunAudioLLM/Fun-ASR-Nano-2512",
                "tokens": len(tokens),
                "segments": len(rows),
                "segments_with_nano_text": assigned,
                "segments_fallback_sensevoice": len(rows) - assigned,
                "chunk_seconds": args.chunk_seconds,
                "overlap_seconds": args.overlap_seconds,
                "hotwords": hotwords,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    paths = export_all(
        hybrid_jsonl,
        output_dir,
        window_seconds=args.window_seconds,
        stats_window_min=args.stats_window_min,
        teacher_id=args.teacher_id,
        window_mode="fixed",
    )

    print(f"Tokens: {len(tokens)}")
    print(f"Segments: {len(rows)}")
    print(f"Segments with Nano text: {assigned}")
    print(f"Metadata: {metadata_path}")
    for name, path in paths.items():
        if name != "teacher_id":
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
