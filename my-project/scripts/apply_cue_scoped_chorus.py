from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from transcript_exports import export_all, ms_to_hms

STOP_WORDS = ["不要", "对了", "这样", "不对", "好了", "停了", "行了", "可以了", "对"]


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tokens(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens = data["tokens"] if isinstance(data, dict) and "tokens" in data else data[0]["timestamps"]
    return sorted(tokens, key=lambda token: float(token.get("start_time", 0)))


def cue_length(tokens: list[dict], index: int) -> int:
    text = "".join(str(token.get("token", "")) for token in tokens[index : index + 3])
    if text.startswith("预备齐") or text.startswith("预备起"):
        return 3
    if text.startswith("一起读"):
        return 3
    if text.startswith("齐读"):
        return 2
    return 0


def stop_at(tokens: list[dict], index: int) -> bool:
    text = "".join(str(token.get("token", "")) for token in tokens[index : index + 3])
    return any(text.startswith(word) for word in STOP_WORDS)


def find_intervals(tokens: list[dict]) -> list[dict]:
    intervals = []
    index = 0
    while index < len(tokens):
        length = cue_length(tokens, index)
        if not length:
            index += 1
            continue
        position = index + length
        while position < len(tokens) and str(tokens[position].get("token", "")) in "，。！？；：,.!?;:":
            position += 1
        start_position = position
        content = []
        while position < len(tokens):
            if stop_at(tokens, position):
                break
            token_text = str(tokens[position].get("token", ""))
            if token_text in "。！？；!?" and len("".join(content)) >= 2:
                break
            content.append(token_text)
            position += 1
        text = "".join(content).strip("，。！？；：,.!?;:")
        if len(text) >= 2 and start_position < len(tokens):
            start = float(tokens[start_position].get("start_time", 0)) * 1000
            end_position = position - 1
            while end_position >= start_position and str(tokens[end_position].get("token", "")) in "，。！？；：,.!?;:":
                end_position -= 1
            end = float(tokens[end_position].get("end_time", 0)) * 1000
            if end - start >= 300:
                intervals.append({"start_ms": int(start), "end_ms": int(end), "text": text})
        index = max(index + 1, position)
    intervals.sort(key=lambda item: item["start_ms"])
    merged = []
    for interval in intervals:
        if merged and interval["start_ms"] < merged[-1]["end_ms"]:
            continue
        merged.append(interval)
    return merged


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
    return text[left:right].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("audio", nargs="?")
    parser.add_argument("--nano-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-id", default="S01")
    parser.add_argument("--output-window-seconds", type=float, default=3.0)
    parser.add_argument("--stats-window-min", type=float, default=5.0)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = jsonl_path.stem.removesuffix("_segments")

    rows = sorted(load_rows(jsonl_path), key=lambda row: (int(row["start_ms"]), int(row["segment_id"])))
    tokens = load_tokens(Path(args.nano_result))
    intervals = find_intervals(tokens)

    output_rows = []
    for row in rows:
        start = int(row["start_ms"])
        end = int(row["end_ms"])
        overlapping = [item for item in intervals if item["start_ms"] < end and item["end_ms"] > start]
        if not overlapping or str(row["speaker_id"]).upper() == "CHORUS":
            output_rows.append(row)
            continue

        cursor = start
        for interval in overlapping:
            if interval["start_ms"] > cursor:
                fragment_text = tokens_in_range(tokens, cursor, interval["start_ms"])
                if not fragment_text:
                    fragment_text = proportional_text(str(row.get("text") or ""), start, end, cursor, interval["start_ms"])
                if fragment_text:
                    fragment = dict(row)
                    fragment["start_ms"] = cursor
                    fragment["end_ms"] = interval["start_ms"]
                    fragment["start_hms"] = ms_to_hms(cursor)
                    fragment["end_hms"] = ms_to_hms(interval["start_ms"])
                    fragment["duration_s"] = round((interval["start_ms"] - cursor) / 1000, 3)
                    fragment["text"] = fragment_text
                    output_rows.append(fragment)
            cursor = max(cursor, interval["end_ms"])
        if cursor < end:
            fragment_text = tokens_in_range(tokens, cursor, end)
            if not fragment_text:
                fragment_text = proportional_text(str(row.get("text") or ""), start, end, cursor, end)
            if fragment_text:
                fragment = dict(row)
                fragment["start_ms"] = cursor
                fragment["end_ms"] = end
                fragment["start_hms"] = ms_to_hms(cursor)
                fragment["end_hms"] = ms_to_hms(end)
                fragment["duration_s"] = round((end - cursor) / 1000, 3)
                fragment["text"] = fragment_text
                output_rows.append(fragment)

    for interval in intervals:
        chorus = {
            "recording_id": rows[0].get("recording_id", ""),
            "segment_id": 0,
            "start_ms": interval["start_ms"],
            "end_ms": interval["end_ms"],
            "start_hms": ms_to_hms(interval["start_ms"]),
            "end_hms": ms_to_hms(interval["end_ms"]),
            "duration_s": round((interval["end_ms"] - interval["start_ms"]) / 1000, 3),
            "speaker_id": "CHORUS",
            "role": "chorus",
            "text": interval["text"],
            "raw_text": interval["text"],
            "speaker_original": args.teacher_id,
            "chorus_refined": True,
            "chorus_source": "cue_scoped",
        }
        output_rows.append(chorus)

    output_rows.sort(key=lambda row: (int(row["start_ms"]), int(row.get("segment_id", 0))))
    for index, row in enumerate(output_rows, start=1):
        row["segment_id"] = index
        if str(row["speaker_id"]).upper() == "CHORUS":
            row["role"] = "chorus"
        elif str(row["speaker_id"]) == args.teacher_id:
            row["role"] = "teacher"
        else:
            row["role"] = "student"

    output_jsonl = output_dir / f"{prefix}_segments.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as handle:
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

    review_path = output_dir / f"{prefix}_cue_scoped_review.csv"
    pd.DataFrame(intervals).to_csv(review_path, index=False, encoding="utf-8-sig")

    metadata_path = output_dir / f"{prefix}_cue_scoped_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_jsonl": str(jsonl_path),
                "nano_result": args.nano_result,
                "teacher_id": args.teacher_id,
                "intervals": len(intervals),
                "output_segments": len(output_rows),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    paths = export_all(
        output_jsonl,
        output_dir,
        window_seconds=args.output_window_seconds,
        stats_window_min=args.stats_window_min,
        teacher_id=args.teacher_id,
        window_mode="fixed",
    )

    print(f"Cue-scoped chorus intervals: {len(intervals)}")
    print(f"Output segments: {len(output_rows)}")
    print(f"Review: {review_path}")
    print(f"Metadata: {metadata_path}")
    for name, path in paths.items():
        if name != "teacher_id":
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
