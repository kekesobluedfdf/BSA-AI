from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from transcript_exports import export_all


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_corrections(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [(str(source), str(target)) for source, target in data.items()]
    return [(str(item["from"]), str(item["to"])) for item in data]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Path to *_segments.jsonl")
    parser.add_argument("corrections", help="JSON file with from/to pairs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-id", default="S01")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--stats-window-min", type=float, default=5.0)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = jsonl_path.stem.removesuffix("_segments")

    rows = load_rows(jsonl_path)
    corrections = load_corrections(Path(args.corrections))
    report = []

    for row in rows:
        original = str(row.get("text") or "")
        corrected = original
        applied = []
        for source, target in corrections:
            if source and source in corrected:
                count = corrected.count(source)
                corrected = corrected.replace(source, target)
                applied.append({"from": source, "to": target, "count": count})
        row["text_before_correction"] = original
        row["text"] = corrected
        row["corrections_applied"] = applied
        if applied:
            report.append(
                {
                    "segment_id": int(row["segment_id"]),
                    "start_hms": row["start_hms"],
                    "speaker_id": row["speaker_id"],
                    "before": original,
                    "after": corrected,
                    "applied": json.dumps(applied, ensure_ascii=False),
                }
            )

    corrected_jsonl = output_dir / f"{prefix}_segments.jsonl"
    with corrected_jsonl.open("w", encoding="utf-8") as handle:
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
        "text_before_correction",
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

    report_path = output_dir / f"{prefix}_corrections.csv"
    pd.DataFrame(report).to_csv(report_path, index=False, encoding="utf-8-sig")

    paths = export_all(
        corrected_jsonl,
        output_dir,
        window_seconds=args.window_seconds,
        stats_window_min=args.stats_window_min,
        teacher_id=args.teacher_id,
        window_mode="fixed",
    )

    print(f"Segments corrected: {len(report)}")
    print(f"Correction report: {report_path}")
    for name, path in paths.items():
        if name != "teacher_id":
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
