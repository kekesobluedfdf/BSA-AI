from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from transcript_exports import export_all, ms_to_hms


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tokens(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens = data["tokens"] if isinstance(data, dict) and "tokens" in data else data[0]["timestamps"]
    return sorted(tokens, key=lambda token: float(token.get("start_time", 0)))


def tokens_in_range(tokens: list[dict], start_ms: int, end_ms: int) -> str:
    parts = []
    for token in tokens:
        middle = (float(token.get("start_time", 0)) + float(token.get("end_time", 0))) * 500
        if start_ms <= middle < end_ms:
            parts.append(str(token.get("token", "")))
    return "".join(parts).strip().strip("，。！？；：,.!?;:")


def proportional_text(text: str, row_start: int, row_end: int, start: int, end: int) -> str:
    if not text or row_end <= row_start:
        return ""
    left = int(round(len(text) * (start - row_start) / (row_end - row_start)))
    right = int(round(len(text) * (end - row_start) / (row_end - row_start)))
    return text[left:right].strip().strip("，。！？；：,.!?;:")


def make_fragment(row: dict, start: int, end: int, tokens: list[dict]) -> dict:
    fragment = dict(row)
    fragment["start_ms"] = start
    fragment["end_ms"] = end
    fragment["start_hms"] = ms_to_hms(start)
    fragment["end_hms"] = ms_to_hms(end)
    fragment["duration_s"] = round((end - start) / 1000, 3)
    text = tokens_in_range(tokens, start, end)
    if not text:
        text = proportional_text(str(row.get("text") or ""), int(row["start_ms"]), int(row["end_ms"]), start, end)
    fragment["text"] = text
    return fragment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--end-ms", type=int, required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--nano-result", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--teacher-id", default="S01")
    parser.add_argument("--output-window-seconds", type=float, default=3.0)
    parser.add_argument("--stats-window-min", type=float, default=5.0)
    args = parser.parse_args()

    if args.end_ms <= args.start_ms:
        raise ValueError("结束时间必须晚于开始时间")

    jsonl_path = Path(args.jsonl)
    output_dir = Path(args.output_dir) if args.output_dir else jsonl_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = jsonl_path.stem.removesuffix("_segments")

    rows = sorted(load_rows(jsonl_path), key=lambda row: (int(row["start_ms"]), int(row["segment_id"])))
    tokens = load_tokens(Path(args.nano_result) if args.nano_result else None)
    start = args.start_ms
    end = args.end_ms

    if any(int(row["start_ms"]) <= start and int(row["end_ms"]) >= end for row in rows if str(row["speaker_id"]).upper() == "CHORUS"):
        print("该区间已经包含在现有齐读片段中，无需重复标记。")
        return 0

    chorus_text = args.text.strip()
    if not chorus_text:
        chorus_text = tokens_in_range(tokens, start, end)
    if not chorus_text:
        pieces = []
        for row in rows:
            if int(row["end_ms"]) > start and int(row["start_ms"]) < end:
                pieces.append(
                    proportional_text(
                        str(row.get("text") or ""),
                        int(row["start_ms"]),
                        int(row["end_ms"]),
                        max(start, int(row["start_ms"])),
                        min(end, int(row["end_ms"])),
                    )
                )
        chorus_text = "".join(pieces).strip()

    output_rows = []
    for row in rows:
        row_start = int(row["start_ms"])
        row_end = int(row["end_ms"])
        if row_end <= start or row_start >= end:
            output_rows.append(row)
            continue
        if row_start < start:
            fragment = make_fragment(row, row_start, start, tokens)
            if fragment["text"]:
                output_rows.append(fragment)
        if row_end > end:
            fragment = make_fragment(row, end, row_end, tokens)
            if fragment["text"]:
                output_rows.append(fragment)

    output_rows.append(
        {
            "recording_id": rows[0].get("recording_id", "") if rows else "",
            "segment_id": 0,
            "start_ms": start,
            "end_ms": end,
            "start_hms": ms_to_hms(start),
            "end_hms": ms_to_hms(end),
            "duration_s": round((end - start) / 1000, 3),
            "speaker_id": "CHORUS",
            "role": "chorus",
            "text": chorus_text,
            "raw_text": chorus_text,
            "speaker_original": args.teacher_id,
            "chorus_refined": True,
            "chorus_source": "manual_interval",
        }
    )

    output_rows.sort(key=lambda row: (int(row["start_ms"]), int(row.get("segment_id", 0))))
    for index, row in enumerate(output_rows, start=1):
        row["segment_id"] = index
        if str(row["speaker_id"]).upper() == "CHORUS":
            row["role"] = "chorus"
        elif str(row["speaker_id"]) == args.teacher_id:
            row["role"] = "teacher"
        else:
            row["role"] = "student"

    backup_path = output_dir / f"{prefix}_segments_before_manual_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    if jsonl_path.exists():
        shutil.copy2(jsonl_path, backup_path)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
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
        "raw_text",
    ]
    pd.DataFrame([{column: row.get(column, "") for column in columns} for row in output_rows]).to_csv(
        output_dir / f"{prefix}_segments.csv", index=False, encoding="utf-8-sig"
    )

    review_path = output_dir / f"{prefix}_manual_chorus_review.csv"
    existing = pd.read_csv(review_path) if review_path.exists() else pd.DataFrame()
    record = pd.DataFrame(
        [
            {
                "start_ms": start,
                "end_ms": end,
                "start_hms": ms_to_hms(start),
                "end_hms": ms_to_hms(end),
                "text": chorus_text,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        ]
    )
    pd.concat([existing, record], ignore_index=True).to_csv(review_path, index=False, encoding="utf-8-sig")

    paths = export_all(
        jsonl_path,
        output_dir,
        window_seconds=args.output_window_seconds,
        stats_window_min=args.stats_window_min,
        teacher_id=args.teacher_id,
        window_mode="fixed",
    )

    print(f"Manual chorus: {ms_to_hms(start)} - {ms_to_hms(end)}")
    print(f"Text: {chorus_text}")
    print(f"Backup: {backup_path}")
    print(f"Review: {review_path}")
    for name, path in paths.items():
        if name != "teacher_id":
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
